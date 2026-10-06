from tilelang.carver.roller import Edge

from cleave import passes
from cleave.graph import (
    AccumEdge,
    Graph,
    GraphNode,
    Iterator,
    OutputNode,
    PlaceHolderNode,
    ScalarScan,
)
from cleave.ops import (
    Accumulate,
    IndexGather,
    Add,
    CausalMaskedExp,
    MaskedDiv,
    IndptrTileScan,
    MaskedLog2,
    Matmul,
    PagedGather,
    Reduction,
)
from cleave.ops.gather import BUFFER_BOUND
from cleave.util import gather_all_nodes, lock_graph

DTYPE = "bfloat16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "bfloat16"

# BM stays at 16: the value accumulator is (BM, head_dim_ckv) in fp32, so BM=64 would
# need 256 registers per thread at 128 threads, past the 255 limit.
BM = 16
# 64, not 32: the KV window is walked in BN-row tiles, and cleave's per-tile fixed cost is
# high enough that halving the tile count wins even though it doubles the staged KV tile to
# BN*(ckv+kpe)*2 = 72 KB of shared memory. Measured 1.355x over BN=32 across all 38
# workloads (geomean vs the reference kernel 1.265x -> 1.064x); BN=128 regresses to 1.729x.
BN = 64
# KV tiles per page-table window; see the decode schedule for why the window pays. The
# cache tile is the same BN x 512 here, so the same reasoning applies. Shallower than
# decode's window because a prefill request's causal window is shorter, so a deeper one
# would stage entries the tiles never reach: measured 44.0us at 4 against 45.2us at 8.
IWT = 4
PAGE_SIZE = 1
N_THREADS = 128

# exp2 is a single instruction where exp is a call, so the scale is carried in the log2
# domain: exp(x * s) == exp2(x * s * LOG2E).
LOG2E = 1.4426950408889634
# Floors the softmax denominator. A query row whose causal window is empty sums to zero
# over a zero numerator; the floor turns that 0/0 into the 0 the operator defines,
# instead of a NaN the benchmark's sanity check would reject outright.
DENOM_FLOOR = 1e-30

# Symbolic extents, one per var axis. Each appears bare in some tensor shape (TOTAL_Q in
# q_nope/q_pe, NUM_PAGES in the caches, LEN_INDPTR in qo_indptr, N_IDX in kv_indices) so
# the runtime infers it from the bound tensor; afterwards it may be used in extent
# expressions.
TOTAL_Q = "total_q"
NUM_PAGES = "num_pages"
N_IDX = "n_idx"
LEN_INDPTR = "len_indptr"

# DeepSeek-V3's captured softmax scale: the pre-absorption qk head dim gives
# 192**-0.5, further scaled by the YaRN mscale correction (0.1*ln(40) + 1)**2. It is
# therefore not derivable from the compressed ckv width. The kernel takes it as an input
# rather than baking it in; this value is what every captured workload passes.
SM_SCALE = 0.1352337747812271


def _connect(src, dst, src_id, dst_id):
    edge = Edge(src, dst, src_id, dst_id)
    src._out_edges.append(edge)
    dst.set_inputs(dst_id, edge)


def get_handwritten_scheduled_graph(dims):
    num_qo_heads, head_dim_ckv, head_dim_kpe = dims

    assert num_qo_heads == 16, num_qo_heads
    assert head_dim_ckv == 512, head_dim_ckv
    assert head_dim_kpe == 64, head_dim_kpe
    # Exactly equal, not just divisible: the query tiles below are affine regions over one
    # token's block of heads, and a tile spanning several tokens is not one region.
    assert BM == num_qo_heads, (BM, num_qo_heads)

    # MLA has a single kv head, so a CTA owns ALL query heads of its tokens: the M axis
    # fuses (token, head) and one staged KV tile feeds every head. With BM == num_qo_heads
    # that is a single token per tile, which is why the store below needs no token axis.
    btok = BM // num_qo_heads

    # One shared page-table window feeds IWT tiles, so the entry each gathered row needs is
    # read from shared and staged once per window rather than read from global once per row
    # per tile. The window tile is written by different threads than read it, so it costs a
    # barrier, but one per window rather than one per tile.
    idx_window = IndexGather(
        (BN * IWT,),
        index_extent=N_IDX,
        row_stride=1,
        page_size=PAGE_SIZE,
        clamp_index=BUFFER_BOUND,
        scalar_args=("kv_base", "win_base", "valid_len"),
        name="idx_window",
    )
    gather_ckv = PagedGather(
        (BN, head_dim_ckv),
        source_rows=NUM_PAGES,
        index_extent=BN * IWT,
        page_size=PAGE_SIZE,
        paged_axes=True,
        clamp_index=BUFFER_BOUND,
        clamp_row=BUFFER_BOUND,
        scalar_args=("win_off",),
        dtype=DTYPE,
        out_dtype=DTYPE,
        name="paged_gather_ckv",
    )
    gather_kpe = PagedGather(
        (BN, head_dim_kpe),
        source_rows=NUM_PAGES,
        index_extent=BN * IWT,
        page_size=PAGE_SIZE,
        paged_axes=True,
        clamp_index=BUFFER_BOUND,
        clamp_row=BUFFER_BOUND,
        scalar_args=("win_off",),
        dtype=DTYPE,
        out_dtype=DTYPE,
        name="paged_gather_kpe",
    )
    mma_0 = Matmul(
        (BM, BN, head_dim_ckv),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="cols",
    )
    mma_pe = Matmul(
        (BM, BN, head_dim_kpe),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_pe",
        transpose_B=True,
        warp_policy="cols",
    )
    add_0 = Add(
        (BM, BN),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="add_0",
        n_batch_dims=0,
    )
    exp_0 = CausalMaskedExp(
        (BM, BN),
        scale=None,
        scalar_args=("tile_base", "valid_len", "diag", "scale2"),
        row_group=num_qo_heads,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="exp_0",
        n_batch_dims=0,
    )
    # The denominator is accumulated as a whole (BM, BN) tile and folded to a row vector
    # once, after the loop, rather than folded every iteration; see the decode schedule.
    psum_0 = Accumulate(
        (BM, BN), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="psum_0"
    )
    mma_1 = Matmul(
        (BM, head_dim_ckv, BN),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="cols",
    )

    mma_1.stage_fragment_operands_in_shared = (0,)

    _connect(gather_ckv, mma_0, 0, 1)
    _connect(gather_kpe, mma_pe, 0, 1)
    _connect(mma_0, add_0, 0, 0)
    _connect(mma_pe, add_0, 0, 1)
    _connect(add_0, exp_0, 0, 0)
    _connect(exp_0, psum_0, 0, 0)
    _connect(exp_0, mma_1, 0, 0)
    # The compressed cache is also the value tensor: MLA has no separate V.
    _connect(gather_ckv, mma_1, 0, 1)

    ph_qn = mma_0.inputs[0].src_node
    ph_ckv = gather_ckv.inputs[0].src_node
    _connect(idx_window, gather_ckv, 0, 1)
    _connect(idx_window, gather_kpe, 0, 1)
    ph_idx = idx_window.inputs[0].src_node
    ph_qp = mma_pe.inputs[0].src_node
    ph_kpe = gather_kpe.inputs[0].src_node

    ph_qn.name = "PlaceHolder_Q_NOPE"
    ph_ckv.name = "PlaceHolder_CKV_CACHE"
    ph_idx.name = "PlaceHolder_KV_INDICES"
    ph_qp.name = "PlaceHolder_Q_PE"
    ph_kpe.name = "PlaceHolder_KPE_CACHE"

    inner_graph = Graph(nodes=gather_all_nodes(psum_0, mma_1))
    # A serial loop with a runtime trip count never reads the static extent.
    inner_graph.extent = 1
    inner_graph.loop_kind = "serial"
    inner_graph.loop_trip = "T.ceildiv(valid_len, %d)" % BN
    inner_graph.window_nodes = [idx_window]
    inner_graph.window_guard = "win_off == 0"
    # No prefetch here, unlike the decode schedule. A prefetch needs the wider 256-thread
    # block to keep the BN x 512 tile out of spill, and this operator's tiles are one query
    # block of 16 rows -- too little work to keep 8 warps busy. Measured 0.86x -> 0.76x of
    # FlashInfer with prefetch and 256 threads.
    inner_graph.sargs = [
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("scale2", "float32"),
    ]
    inner_graph.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("scale2", "float32"),
        ("kv_base", "int32"),
        ("win_base", "int32"),
        ("win_off", "int32"),
    ]
    inner_graph.smap = {
        "kv_base": lambda i: "kv_base",
        "valid_len": lambda i: "valid_len",
        "diag": lambda i: "diag",
        "scale2": lambda i: "scale2",
        "tile_base": lambda i: f"{BN} * {i}",
        # The window this tile falls in, and the tile's offset inside it, both in tokens.
        "win_base": lambda i: f"{i} // {IWT} * {BN * IWT}",
        "win_off": lambda i: f"{i} % {IWT} * {BN}",
    }
    lock_graph(inner_graph)

    node_0 = GraphNode(name="GraphNode_0", graph=inner_graph)
    psum_id = node_0.graph.accum_nodes.index(psum_0)
    value_id = node_0.graph.accum_nodes.index(mma_1)

    inner_qn = inner_graph.binding[ph_qn.name]
    inner_qp = inner_graph.binding[ph_qp.name]
    inner_idx = inner_graph.binding[ph_idx.name]

    rowsum_0 = Reduction(
        (BM, BN), axis=-1, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    e_ps = AccumEdge(node_0, rowsum_0, psum_id, 0, accum_node=psum_0)
    node_0._out_edges.append(e_ps)
    rowsum_0.set_inputs(0, e_ps)

    div_0 = MaskedDiv(
        (BM, head_dim_ckv),
        broadcast=0,
        scalar_args=("n_rows",),
        denom_floor=DENOM_FLOOR,
        row_group=num_qo_heads,
        fuse_dst=False,
        dtype=ACCUM_DTYPE,
        out_dtype=OUT_DTYPE,
        name="div_0",
        n_batch_dims=0,
    )
    e_val = AccumEdge(node_0, div_0, value_id, 0, accum_node=mma_1)
    node_0._out_edges.append(e_val)
    div_0.set_inputs(0, e_val)
    _connect(rowsum_0, div_0, 0, 1)

    log2_0 = MaskedLog2(
        (BM,),
        scalar_args=("n_rows",),
        row_group=num_qo_heads,
        fuse_dst=False,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="log2_0",
        n_batch_dims=0,
    )
    _connect(rowsum_0, log2_0, 0, 0)

    out_o = OutputNode(div_0, dtype=OUT_DTYPE)
    out_o.name = "OutputNode_0"
    out_lse = OutputNode(log2_0, dtype=ACCUM_DTYPE)
    out_lse.name = "OutputNode_1"

    # Read for their values, not gathered from, so they stay out of the *_iter call.
    ph_qo = PlaceHolderNode(name="QO_INDPTR")
    ph_qo.set_shape((1,))
    ph_qo.set_dtype(_int32())
    ph_qo.index_only = True
    ph_kv = PlaceHolderNode(name="KV_INDPTR")
    ph_kv.set_shape((1,))
    ph_kv.set_dtype(_int32())
    ph_kv.index_only = True

    # Inputs lead the node list in the operator's declared order, both so the emitted
    # signature matches it and because a symbolic output extent is resolved at call time
    # from an already-bound input.
    graph = Graph(
        nodes=[ph_qn, ph_qp, ph_ckv, ph_kpe, ph_qo, ph_kv, ph_idx]
        + gather_all_nodes(node_0)
    )
    graph.dyn_dims = [TOTAL_Q, NUM_PAGES, LEN_INDPTR, N_IDX]
    graph.scalar_params = [("sm_scale", "float32")]
    graph.n_threads = N_THREADS
    # No L2 swizzle. Rasterising the grid costs a per-CTA index remap and buys locality
    # between neighbouring blocks; here the blocks do not share enough KV to repay it.
    # Measured vs the reference kernel: 1.064x -> 1.037x.
    graph.rasterization = 1
    # One CTA per query tile; every head lives in the tile. sum(ceil(q_len/btok)) is at
    # most ceil(total_q/btok) + n_requests - 1 (ceil is subadditive up to one per term),
    # so this grid always covers the work list; the slack slots scan to request -1 and the
    # body guard skips them.
    graph.extent = (f"T.ceildiv({TOTAL_Q}, {btok}) + {LEN_INDPTR} - 2",)
    graph.prologue = [
        ScalarScan(
            op=IndptrTileScan(rows=btok),
            buffer="_wl",
            size=2,
            inputs=(ph_qo,),
            scalar_args=lambda i: (i, LEN_INDPTR),
        )
    ]
    graph.block_guard = lambda i: "_wl[0] >= 0"
    # A BufferRegion lets tilelang reach for TMA, whose mbarrier protocol assumes a
    # pipelined producer/consumer body; these nested serial macros provide no arrivals for
    # it and the kernel spins on the barrier. Nothing here needs TMA: the region exists to
    # replace a per-element gather with an affine copy, and this tile can never overrun
    # (its leading extent is 1 and the other two are the buffer's own).
    graph.pass_configs = {
        # Bounds-predication is off because every read this kernel makes already clamps
        # itself: the gathers clamp the page-table subscript to the table and the page id
        # to the cache, and the q tiles guard their own valid rows. Left on, the legalizer
        # wraps each load in a bounds test and, because the page-table entry sits INSIDE
        # the address of the cache read, it re-materialises that entry once per test --
        # three global loads per row instead of one, all on the critical path.
        "tl.disable_safe_memory_legalize": True,
        "tl.disable_tma_lower": True,
        "tl.disable_warp_specialized": True,
    }
    graph.sargs = [
        ("n_rows", "int32"),
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("scale2", "float32"),
    ]

    # Every scalar the body needs, as an expression over the scan result (_wl[0] is the
    # request, _wl[1] the tile within it) and the two indptr tensors.
    q_beg = f"{ph_qo.name}[_wl[0]]"
    q_end = f"{ph_qo.name}[_wl[0] + 1]"
    kv_beg = f"{ph_kv.name}[_wl[0]]"
    kv_end = f"{ph_kv.name}[_wl[0] + 1]"
    q_len = f"({q_end} - {q_beg})"
    kv_len = f"({kv_end} - {kv_beg})"
    tile_base = f"(_wl[1] * {btok})"
    n_rows = f"T.min({btok}, {q_len} - {tile_base})"
    # End-aligned causal bound for the tile's first row, then the KV window causality
    # leaves live for the whole tile.
    diag = f"({kv_len} - {q_len} + {tile_base})"
    graph.smap = {
        "n_rows": lambda i: n_rows,
        "kv_base": lambda i: kv_beg,
        "valid_len": lambda i: f"T.max(T.min({kv_len}, {diag} + {n_rows}), 0)",
        "diag": lambda i: diag,
        "scale2": lambda i: f"sm_scale * {LOG2E}",
    }
    lock_graph(graph)

    whole = lambda shape: Iterator(  # noqa: E731
        lambda i: tuple("0" for _ in shape), shape=tuple(shape), tile_size=tuple(shape)
    )
    qn_shape = (TOTAL_Q, num_qo_heads, head_dim_ckv)
    qp_shape = (TOTAL_Q, num_qo_heads, head_dim_kpe)
    ckv_shape = (NUM_PAGES, PAGE_SIZE, head_dim_ckv)
    kpe_shape = (NUM_PAGES, PAGE_SIZE, head_dim_kpe)
    # One token's whole block of heads, which is contiguous in (total_q, head, dim): the
    # tile the M axis wants is an affine slice, so it goes down as a region instead of a
    # gather. The caches cannot follow -- page_size is 1, so consecutive kv positions are
    # unrelated pages and the read is irreducibly a gather.
    q_tile = lambda shape: Iterator(  # noqa: E731
        lambda i: (f"{q_beg} + {tile_base}", "0", "0"),
        shape=shape,
        tile_size=(btok, num_qo_heads, shape[-1]),
    )
    graph.tmap[ph_qn] = q_tile(qn_shape)
    graph.tmap[ph_qp] = q_tile(qp_shape)
    graph.region_args = [ph_qn, ph_qp]
    graph.tmap[ph_ckv] = whole(ckv_shape)
    graph.tmap[ph_kpe] = whole(kpe_shape)
    graph.tmap[ph_idx] = whole((N_IDX,))
    graph.tmap[ph_qo] = whole((LEN_INDPTR,))
    graph.tmap[ph_kv] = whole((LEN_INDPTR,))
    graph.tmap[node_0] = [
        Iterator(lambda i: ("0",), shape=(BM,), tile_size=(BM,)),
        Iterator(
            lambda i: ("0", "0"),
            shape=(BM, head_dim_ckv),
            tile_size=(BM, head_dim_ckv),
        ),
    ]

    q_abs = f"{q_beg} + {tile_base}"
    out_offset = lambda i: (q_abs, "0", "0")  # noqa: E731
    lse_offset = lambda i: (q_abs, "0")  # noqa: E731
    graph.tmap[rowsum_0] = Iterator(
        lambda i: ("0",), shape=(BM,), tile_size=(BM,)
    )
    graph.tmap[div_0] = Iterator(
        out_offset,
        shape=(TOTAL_Q, num_qo_heads, head_dim_ckv),
        tile_size=(btok, num_qo_heads, head_dim_ckv),
    )
    graph.tmap[out_o] = graph.tmap[div_0]
    graph.tmap[log2_0] = Iterator(
        lse_offset, shape=(TOTAL_Q, num_qo_heads), tile_size=(btok, num_qo_heads)
    )
    graph.tmap[out_lse] = graph.tmap[log2_0]

    inner_graph.tmap[inner_qn] = whole((BM, head_dim_ckv))
    inner_graph.tmap[inner_qp] = whole((BM, head_dim_kpe))
    inner_graph.tmap[inner_idx] = whole((N_IDX,))
    inner_graph.tmap[inner_graph.binding[ph_ckv.name]] = whole(ckv_shape)
    inner_graph.tmap[inner_graph.binding[ph_kpe.name]] = whole(kpe_shape)
    inner_graph.tmap[idx_window] = Iterator(
        lambda i: ("0",), shape=(BN * IWT,), tile_size=(BN * IWT,)
    )
    inner_graph.tmap[gather_ckv] = Iterator(
        lambda i: ("0", "0"), shape=(BN, head_dim_ckv), tile_size=(BN, head_dim_ckv)
    )
    inner_graph.tmap[gather_kpe] = Iterator(
        lambda i: ("0", "0"), shape=(BN, head_dim_kpe), tile_size=(BN, head_dim_kpe)
    )
    inner_graph.tmap[mma_0] = Iterator(
        lambda i: ("0", "0"), shape=(BM, BN), tile_size=(BM, BN)
    )
    inner_graph.tmap[mma_pe] = inner_graph.tmap[mma_0]
    inner_graph.tmap[add_0] = inner_graph.tmap[mma_0]
    inner_graph.tmap[exp_0] = inner_graph.tmap[mma_0]
    inner_graph.tmap[psum_0] = inner_graph.tmap[exp_0]
    inner_graph.tmap[mma_1] = Iterator(
        lambda i: ("0", "0"), shape=(BM, head_dim_ckv), tile_size=(BM, head_dim_ckv)
    )

    graph.tinfo_map = {
        ph_qn: ("PlaceHolder_Q_NOPE", "g"),
        ph_qp: ("PlaceHolder_Q_PE", "g"),
        ph_ckv: ("PlaceHolder_CKV_CACHE", "g"),
        ph_kpe: ("PlaceHolder_KPE_CACHE", "g"),
        ph_idx: ("PlaceHolder_KV_INDICES", "g"),
        ph_qo: ("PlaceHolder_QO_INDPTR", "g"),
        ph_kv: ("PlaceHolder_KV_INDPTR", "g"),
        node_0: ("t_GraphNode_0", "r"),
        rowsum_0: ("t_rowsum_0", "r"),
        div_0: ("OutputNode_0", "g"),
        log2_0: ("OutputNode_1", "g"),
        out_o: ("OutputNode_0", "g"),
        out_lse: ("OutputNode_1", "g"),
    }
    inner_graph.tinfo_map = {
        ph_qn: ("PlaceHolder_Q_NOPE", "g"),
        ph_qp: ("PlaceHolder_Q_PE", "g"),
        ph_ckv: ("PlaceHolder_CKV_CACHE", "g"),
        ph_kpe: ("PlaceHolder_KPE_CACHE", "g"),
        ph_idx: ("PlaceHolder_KV_INDICES", "g"),
        inner_qn: (inner_qn.name, "s"),
        inner_qp: (inner_qp.name, "s"),
        inner_idx: (inner_idx.name, "g"),
        inner_graph.binding[ph_ckv.name]: (
            inner_graph.binding[ph_ckv.name].name,
            "g",
        ),
        inner_graph.binding[ph_kpe.name]: (
            inner_graph.binding[ph_kpe.name].name,
            "g",
        ),
        idx_window: ("t_idx_window", "s"),
        gather_ckv: ("t_paged_gather_ckv", "s"),
        gather_kpe: ("t_paged_gather_kpe", "s"),
        mma_0: ("t_mma_0", "r"),
        mma_pe: ("t_mma_pe", "r"),
        add_0: ("t_add_0", "r"),
        exp_0: ("t_exp_0", "r"),
        psum_0: ("t_psum_0", "r"),
        mma_1: ("t_mma_1", "r"),
    }

    graph = passes.annotate_args(graph, is_main=True)
    graph = passes.annotate_tc_layout(graph)
    graph = passes.annotate_fragment_layout_conflicts(graph)
    inner_graph.tlayout_map[inner_idx] = "linear"
    return graph


def _int32():
    import tvm

    return tvm.DataType("int32")
