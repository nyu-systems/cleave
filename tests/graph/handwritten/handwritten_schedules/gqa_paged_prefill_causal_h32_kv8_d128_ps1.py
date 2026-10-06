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
    CausalMaskedExp,
    ConstFill,
    IndexGather,
    IndptrTileScan,
    MaskedDiv,
    MaskedLog2,
    Matmul,
    PagedGather,
    RangeGather,
    Reduction,
)
from cleave.ops.gather import BUFFER_BOUND
from cleave.util import gather_all_nodes, lock_graph

DTYPE = "bfloat16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "bfloat16"

BM = 64
# Narrow, and one warpgroup rather than two. The captured workloads are dominated by
# requests whose causal window is a handful of KV tokens, so a wide tile stages rows the
# mask throws away and a second warpgroup has nothing to do; measured 1.13x (kv8) and 1.20x
# (kv4) against BN=32 at 256 threads.
BN = 16
PAGE_SIZE = 1
N_THREADS = 128

# exp2 is a single instruction where exp is a call, so the scale is carried in the log2
# domain: exp(x * s) == exp2(x * s * LOG2E).
LOG2E = 1.4426950408889634
# Floors the softmax denominator. A query row whose causal window is empty sums to zero
# over a zero numerator; the floor turns that 0/0 into the 0 the operator defines,
# instead of a NaN the benchmark's sanity check would reject outright.
DENOM_FLOOR = 1e-30

# Symbolic extents, one per var axis. Each appears bare in some tensor shape (TOTAL_Q in q,
# NUM_PAGES in the caches, LEN_INDPTR in qo_indptr, N_IDX in kv_indices) so the runtime
# infers it from the bound tensor; afterwards it may be used in extent expressions.
TOTAL_Q = "total_q"
NUM_PAGES = "num_pages"
N_IDX = "n_idx"
LEN_INDPTR = "len_indptr"


def _connect(src, dst, src_id, dst_id):
    edge = Edge(src, dst, src_id, dst_id)
    src._out_edges.append(edge)
    dst.set_inputs(dst_id, edge)


def get_handwritten_scheduled_graph(dims):
    num_qo_heads, num_kv_heads, head_dim = dims
    group = num_qo_heads // num_kv_heads

    assert num_qo_heads == 32, num_qo_heads
    assert num_kv_heads == 8, num_kv_heads
    assert head_dim == 128, head_dim
    assert group == 4, group
    assert BM % group == 0, (BM, group)

    # One CTA owns a whole query head GROUP, so its M axis fuses (token, head slot):
    # M = BTOK * group. That is the point of the shape -- the KV tile is gathered once per
    # M tile and feeds all `group` query heads, instead of once per head.
    btok = BM // group

    # The kernel takes the operator's declared inputs and nothing else: the work list a
    # host planner used to precompute is derived on the device by scanning qo_indptr
    # (``graph.prologue``), and every per-block scalar below is an expression over that
    # scan's result plus the two indptr tensors.
    # The gathers read the page table in place off a token-unit base rather than through a
    # tile an IndexGather resolved for them. A resolved tile has to live in shared memory,
    # because the thread that gathers row g0 of K is not the thread that computed entry g0
    # of the tile, and that costs a barrier before the tile can be read plus another before
    # the next iteration may overwrite it. Reading kv_indices where it lies keeps the whole
    # address computation thread-local: the chain per tile drops from index -> K -> mma to
    # K -> mma. The entry is 4 bytes per row and every row of a tile hits the same cache
    # line, so the extra global read is served by L1.
    #
    # The valid_len clamp the resolved tile applied goes away with it; a row past the
    # request's window now reads whatever page the neighbouring request owns. That is a
    # live page, so the read is in bounds, and exp_0 masks the column to zero regardless of
    # what was fetched, so the output is unchanged.
    gather_k = PagedGather(
        (BN, head_dim),
        source_rows=NUM_PAGES,
        index_extent=N_IDX,
        page_size=PAGE_SIZE,
        paged_axes=True,
        source_heads=num_kv_heads,
        head_scalar=True,
        clamp_index=BUFFER_BOUND,
        clamp_row=BUFFER_BOUND,
        scalar_args=("kv_pos0", "kv_head"),
        dtype=DTYPE,
        out_dtype=DTYPE,
        name="paged_gather_k",
    )
    mma_0 = Matmul(
        (BM, BN, head_dim),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="cols",
    )
    exp_0 = CausalMaskedExp(
        (BM, BN),
        scale=None,
        scalar_args=("tile_base", "valid_len", "diag", "scale2"),
        row_group=group,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="exp_0",
        n_batch_dims=0,
    )
    rowsum_0 = Reduction(
        (BM, BN), axis=-1, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    gather_v = PagedGather(
        (BN, head_dim),
        source_rows=NUM_PAGES,
        index_extent=N_IDX,
        page_size=PAGE_SIZE,
        paged_axes=True,
        source_heads=num_kv_heads,
        head_scalar=True,
        clamp_index=BUFFER_BOUND,
        clamp_row=BUFFER_BOUND,
        scalar_args=("kv_pos0", "kv_head"),
        dtype=DTYPE,
        out_dtype=DTYPE,
        name="paged_gather_v",
    )
    mma_1 = Matmul(
        (BM, head_dim, BN),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="cols",
    )

    mma_1.stage_fragment_operands_in_shared = (0,)

    _connect(gather_k, mma_0, 0, 1)
    _connect(mma_0, exp_0, 0, 0)
    _connect(exp_0, rowsum_0, 0, 0)
    _connect(exp_0, mma_1, 0, 0)
    _connect(gather_v, mma_1, 0, 1)

    ph_kf = gather_k.inputs[0].src_node
    ph_vf = gather_v.inputs[0].src_node
    ph_q = mma_0.inputs[0].src_node
    # One page table feeds both gathers, so K's index placeholder is rewired to V as well
    # rather than each gather declaring its own copy of the same kernel input.
    ph_idx = gather_k.inputs[1].src_node
    _connect(ph_idx, gather_v, 0, 1)

    ph_q.name = "PlaceHolder_Q"
    ph_kf.name = "PlaceHolder_K_CACHE"
    ph_vf.name = "PlaceHolder_V_CACHE"
    ph_idx.name = "PlaceHolder_KV_INDICES"

    inner_graph = Graph(nodes=gather_all_nodes(rowsum_0, mma_1))
    # A serial loop with a runtime trip count never reads the static extent.
    inner_graph.extent = 1
    inner_graph.loop_kind = "serial"
    inner_graph.loop_trip = "T.ceildiv(valid_len, %d)" % BN
    # Both KV tiles are read one iteration ahead, so the two gathers' global latency is
    # covered by the tile's own two matmuls and its softmax rather than standing in front
    # of them. This is the loop's only source of load/compute overlap: the reads are
    # gathers, which tilelang's pipeliner cannot double-buffer.
    inner_graph.prefetch_nodes = [gather_k, gather_v]
    inner_graph.prefetch_guard = f"tile_base + {BN} < valid_len"
    inner_graph.sargs = [
        ("q_row0", "int32"),
        ("q_tile_base", "int32"),
        ("q_len", "int32"),
        ("q_head0", "int32"),
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("kv_head", "int32"),
        ("scale2", "float32"),
    ]
    inner_graph.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("scale2", "float32"),
        ("kv_pos0", "int32"),
        ("kv_head", "int32"),
    ]
    inner_graph.smap = {
        "q_row0": lambda i: "q_row0",
        "q_tile_base": lambda i: "q_tile_base",
        "q_len": lambda i: "q_len",
        "q_head0": lambda i: "q_head0",
        "kv_base": lambda i: "kv_base",
        "valid_len": lambda i: "valid_len",
        "diag": lambda i: "diag",
        "kv_head": lambda i: "kv_head",
        "scale2": lambda i: "scale2",
        "tile_base": lambda i: f"{BN} * {i}",
        # The tile's first kv position, in tokens, which is where the gathers start
        # reading the page table.
        "kv_pos0": lambda i: f"kv_base + {BN} * {i}",
    }
    lock_graph(inner_graph)

    node_0 = GraphNode(name="GraphNode_0", graph=inner_graph)
    rowsum_id = node_0.graph.accum_nodes.index(rowsum_0)
    value_id = node_0.graph.accum_nodes.index(mma_1)

    inner_q = inner_graph.binding[ph_q.name]
    inner_idx = inner_graph.binding[ph_idx.name]

    q_tile = RangeGather(
        (BM, head_dim),
        source_rows=TOTAL_Q,
        src_dims=3,
        source_heads=num_qo_heads,
        row_group=group,
        clamp_row=BUFFER_BOUND,
        scalar_args=("q_row0", "q_tile_base", "q_len", "q_head0"),
        dtype=DTYPE,
        out_dtype=DTYPE,
        name="q_tile",
    )
    q_tile.source_args = ("PlaceHolder_Q",)
    inner_q.producer_op = q_tile

    div_0 = MaskedDiv(
        (BM, head_dim),
        broadcast=0,
        scalar_args=("n_rows",),
        denom_floor=DENOM_FLOOR,
        row_group=group,
        dtype=ACCUM_DTYPE,
        out_dtype=OUT_DTYPE,
        name="div_0",
        n_batch_dims=0,
    )
    e_val = AccumEdge(node_0, div_0, value_id, 0, accum_node=mma_1)
    node_0._out_edges.append(e_val)
    div_0.set_inputs(0, e_val)
    e_rs = AccumEdge(node_0, div_0, rowsum_id, 1, accum_node=rowsum_0)
    node_0._out_edges.append(e_rs)
    div_0.set_inputs(1, e_rs)

    log2_0 = MaskedLog2(
        (BM,),
        scalar_args=("n_rows",),
        row_group=group,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="log2_0",
        n_batch_dims=0,
    )
    e_lse = AccumEdge(node_0, log2_0, rowsum_id, 0, accum_node=rowsum_0)
    node_0._out_edges.append(e_lse)
    log2_0.set_inputs(0, e_lse)

    out_o = OutputNode(div_0, dtype=OUT_DTYPE)
    out_o.name = "OutputNode_0"
    out_lse = OutputNode(log2_0, dtype=ACCUM_DTYPE)
    out_lse.name = "OutputNode_1"

    # The alternative producers for a tile with no causal window. They write the same
    # tiles as div_0/log2_0, at the same per-kv-head shape -- one CTA per kv head, so the
    # store keeps the parallelism the attention path has. (Filling all `num_qo_heads`
    # from a single CTA instead measured 1.58x SLOWER than not guarding at all.)
    fill_o = ConstFill(
        (BM, head_dim),
        value=0.0,
        out_dtype=OUT_DTYPE,
        row_group=group,
        scalar_args=("n_rows",),
        name="fill_o",
    )
    fill_lse = ConstFill(
        (BM,),
        value=float("-inf"),
        out_dtype=ACCUM_DTYPE,
        row_group=group,
        scalar_args=("n_rows",),
        name="fill_lse",
    )

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
        nodes=[ph_q, ph_kf, ph_vf, ph_qo, ph_kv, ph_idx] + gather_all_nodes(node_0)
    )
    graph.dyn_dims = [TOTAL_Q, NUM_PAGES, LEN_INDPTR, N_IDX]
    graph.scalar_params = [("sm_scale", "float32")]
    # Bounds-predication is off because every read this kernel makes already clamps
    # itself: the gathers clamp the page-table subscript to the table and the page id to
    # the cache, and the q tile guards its own valid rows. Left on, the legalizer wraps
    # each load in a bounds test and, because the page-table entry sits INSIDE the address
    # of the cache read, it re-materialises that entry once per test -- three global loads
    # per row instead of one, all on the critical path of the address.
    graph.pass_configs = {"tl.disable_safe_memory_legalize": True}
    graph.n_threads = N_THREADS
    # No L2 swizzle. Rasterising the grid costs a per-CTA index remap and buys locality
    # between neighbouring blocks; here the blocks do not share enough KV to repay it.
    # Measured vs the reference kernel: 1.207x -> 1.153x.
    graph.rasterization = 1
    # One CTA per (query tile, kv head). sum(ceil(q_len/BTOK)) is at most
    # ceil(total_q/BTOK) + n_requests - 1 (ceil is subadditive up to one per term), so this
    # grid always covers the work list; the slack slots scan to request -1 and the body
    # guard skips them.
    graph.extent = (
        f"T.ceildiv({TOTAL_Q}, {btok}) + {LEN_INDPTR} - 2",
        f"{num_kv_heads}",
    )
    graph.prologue = [
        ScalarScan(
            op=IndptrTileScan(rows=btok),
            buffer="_wl",
            size=2,
            inputs=(ph_qo,),
            scalar_args=lambda i, h: (i, LEN_INDPTR),
        )
    ]
    graph.block_guard = lambda i, h: "_wl[0] >= 0"
    # With an empty causal window the KV loop runs zero trips, so the accumulators stay
    # zero and the epilogue's floored divide yields exactly the 0 and -inf the fills write
    # -- the two branches agree by construction. Taking the fill instead skips clearing a
    # 64x128 accumulator, gathering Q, and the divide, which on these workloads is 92-100%
    # of the tiles.
    graph.work_guard = lambda i, h: f"{live} > 0"
    graph.else_nodes = [fill_o, fill_lse]
    graph.sargs = [
        ("n_rows", "int32"),
        ("q_row0", "int32"),
        ("q_tile_base", "int32"),
        ("q_len", "int32"),
        ("q_head0", "int32"),
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("kv_head", "int32"),
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
    # Live KV length for this tile before flooring: <= 0 means every row's causal window
    # is empty. `valid_len` is its floored form and the work guard is its sign, so the two
    # cannot drift apart.
    live = f"T.min({kv_len}, {diag} + {n_rows})"
    graph.smap = {
        "n_rows": lambda i, h: n_rows,
        "q_row0": lambda i, h: q_beg,
        "q_tile_base": lambda i, h: tile_base,
        "q_len": lambda i, h: q_len,
        "q_head0": lambda i, h: f"{h} * {group}",
        "kv_base": lambda i, h: kv_beg,
        "valid_len": lambda i, h: f"T.max({live}, 0)",
        "diag": lambda i, h: diag,
        "kv_head": lambda i, h: f"{h}",
        "scale2": lambda i, h: f"sm_scale * {LOG2E}",
    }
    lock_graph(graph)

    cache_shape = (NUM_PAGES, PAGE_SIZE, num_kv_heads, head_dim)
    whole = lambda shape: Iterator(  # noqa: E731
        lambda i, h: tuple("0" for _ in shape),
        shape=tuple(shape),
        tile_size=tuple(shape),
    )
    inner_whole = lambda shape: Iterator(  # noqa: E731
        lambda i: tuple("0" for _ in shape), shape=tuple(shape), tile_size=tuple(shape)
    )
    graph.tmap[ph_q] = whole((TOTAL_Q, num_qo_heads, head_dim))
    graph.tmap[ph_kf] = whole(cache_shape)
    graph.tmap[ph_vf] = whole(cache_shape)
    graph.tmap[ph_idx] = whole((N_IDX,))
    graph.tmap[ph_qo] = whole((LEN_INDPTR,))
    graph.tmap[ph_kv] = whole((LEN_INDPTR,))
    graph.tmap[node_0] = [
        Iterator(lambda i, h: ("0",), shape=(BM,), tile_size=(BM,)),
        Iterator(
            lambda i, h: ("0", "0"), shape=(BM, head_dim), tile_size=(BM, head_dim)
        ),
    ]

    q_abs = f"{q_beg} + {tile_base}"
    out_offset = lambda i, h: (q_abs, f"{h} * {group}", "0")  # noqa: E731
    lse_offset = lambda i, h: (q_abs, f"{h} * {group}")  # noqa: E731
    graph.tmap[div_0] = Iterator(
        out_offset,
        shape=(TOTAL_Q, num_qo_heads, head_dim),
        tile_size=(btok, group, head_dim),
    )
    graph.tmap[out_o] = graph.tmap[div_0]
    graph.tmap[log2_0] = Iterator(
        lse_offset, shape=(TOTAL_Q, num_qo_heads), tile_size=(btok, group)
    )
    graph.tmap[out_lse] = graph.tmap[log2_0]
    graph.tmap[fill_o] = graph.tmap[div_0]
    graph.tmap[fill_lse] = graph.tmap[log2_0]

    inner_graph.extent = 1
    inner_graph.tmap[ph_q] = inner_whole((TOTAL_Q, num_qo_heads, head_dim))
    inner_graph.tmap[inner_q] = inner_whole((BM, head_dim))
    inner_graph.tmap[inner_idx] = inner_whole((N_IDX,))
    inner_graph.tmap[inner_graph.binding[ph_kf.name]] = inner_whole(cache_shape)
    inner_graph.tmap[inner_graph.binding[ph_vf.name]] = inner_whole(cache_shape)
    inner_graph.tmap[gather_k] = Iterator(
        lambda i: ("0", "0"), shape=(BN, head_dim), tile_size=(BN, head_dim)
    )
    inner_graph.tmap[gather_v] = Iterator(
        lambda i: ("0", "0"), shape=(BN, head_dim), tile_size=(BN, head_dim)
    )
    inner_graph.tmap[mma_0] = Iterator(
        lambda i: ("0", "0"), shape=(BM, BN), tile_size=(BM, BN)
    )
    inner_graph.tmap[exp_0] = inner_graph.tmap[mma_0]
    inner_graph.tmap[rowsum_0] = Iterator(
        lambda i: ("0",), shape=(BM,), tile_size=(BM,)
    )
    inner_graph.tmap[mma_1] = Iterator(
        lambda i: ("0", "0"), shape=(BM, head_dim), tile_size=(BM, head_dim)
    )

    graph.tinfo_map = {
        ph_q: ("PlaceHolder_Q", "g"),
        ph_kf: ("PlaceHolder_K_CACHE", "g"),
        ph_vf: ("PlaceHolder_V_CACHE", "g"),
        ph_idx: ("PlaceHolder_KV_INDICES", "g"),
        ph_qo: ("PlaceHolder_QO_INDPTR", "g"),
        ph_kv: ("PlaceHolder_KV_INDPTR", "g"),
        node_0: ("t_GraphNode_0", "r"),
        div_0: ("OutputNode_0", "g"),
        log2_0: ("OutputNode_1", "g"),
        fill_o: ("OutputNode_0", "g"),
        fill_lse: ("OutputNode_1", "g"),
        out_o: ("OutputNode_0", "g"),
        out_lse: ("OutputNode_1", "g"),
    }
    inner_graph.tinfo_map = {
        ph_q: ("PlaceHolder_Q", "g"),
        ph_kf: ("PlaceHolder_K_CACHE", "g"),
        ph_vf: ("PlaceHolder_V_CACHE", "g"),
        ph_idx: ("PlaceHolder_KV_INDICES", "g"),
        inner_q: (inner_q.name, "s"),
        inner_idx: (inner_idx.name, "g"),
        inner_graph.binding[ph_kf.name]: (inner_graph.binding[ph_kf.name].name, "g"),
        inner_graph.binding[ph_vf.name]: (inner_graph.binding[ph_vf.name].name, "g"),
        gather_k: ("t_paged_gather_k", "s"),
        gather_v: ("t_paged_gather_v", "s"),
        mma_0: ("t_mma_0", "r"),
        exp_0: ("t_exp_0", "r"),
        rowsum_0: ("t_rowsum_0", "r"),
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
