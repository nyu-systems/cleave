from tilelang.carver.roller import Edge

from cleave import passes
from cleave.graph import (
    AccumEdge,
    Graph,
    GraphNode,
    Iterator,
    OutputNode,
    PlaceHolderNode,
)
from cleave.ops import (
    Accumulate,
    IndexGather,
    Add,
    Div,
    Log2,
    MaskedExp,
    Matmul,
    PagedGather,
    Reduction,
    RowGather,
)
from cleave.ops.gather import BUFFER_BOUND
from cleave.util import gather_all_nodes, lock_graph

DTYPE = "bfloat16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "bfloat16"

MQ = 16
BN = 64
# KV tiles per page-table window. The gathers need one page-table entry per row of every
# tile they stage; read from global that entry is a dependent load in front of the cache
# read, repeated by every thread of every tile. Staged in shared once per IWT tiles it is
# read from shared instead, and the staging itself is amortised over IWT iterations. This
# is only worth it because the cache tile here is BN x 512 -- wide enough that it streams
# through L1 and evicts the page table between tiles.
IWT = 8
N_THREADS = 256
PAGE_SIZE = 1

# DeepSeek-V3's captured softmax scale: the pre-absorption qk head dim gives
# 192**-0.5, further scaled by the YaRN mscale correction (0.1*ln(40) + 1)**2. It is
# therefore not derivable from the compressed ckv width, so the kernel takes it as the
# declared ``sm_scale`` input; this constant is only what the tests bind to it.
SM_SCALE = 0.1352337747812271

# Symbolic extents, one per var axis of the operator. Each appears bare in some tensor
# shape (BATCH in q_nope/q_pe, NUM_PAGES in the caches, LEN_INDPTR in kv_indptr, N_IDX in
# kv_indices) so the runtime infers it from the bound tensor; afterwards it may be used
# inside extent and grid expressions.
BATCH = "batch"
NUM_PAGES = "num_pages"
N_IDX = "n_idx"
LEN_INDPTR = "len_indptr"

# exp2 is a single instruction where exp is a call, so the scale is carried in the log2
# domain: exp(x * s) == exp2(x * s * LOG2E).
LOG2E = 1.4426950408889634
# Floors the softmax denominator. A request whose KV window is empty runs zero KV tiles,
# leaving a zero numerator over a zero denominator; the floor turns that 0/0 into the 0 the
# operator defines, instead of a NaN the benchmark's sanity check would reject outright.
DENOM_FLOOR = 1e-30


def _connect(src, dst, src_id, dst_id):
    edge = Edge(src, dst, src_id, dst_id)
    src._out_edges.append(edge)
    dst.set_inputs(dst_id, edge)


def get_handwritten_scheduled_graph(dims):
    num_qo_heads, head_dim_ckv, head_dim_kpe = dims

    assert num_qo_heads == 16, num_qo_heads
    assert head_dim_ckv == 512, head_dim_ckv
    assert head_dim_kpe == 64, head_dim_kpe
    assert num_qo_heads == MQ, num_qo_heads

    # Every bound the gathers need is read off the buffer they receive, so no extent has
    # to be pinned at compile time: one compiled kernel serves every workload shape.
    # One shared page-table window feeds IWT tiles. The alternative this replaced -- each
    # gather reading kv_indices in place -- keeps the address computation thread-local but
    # pays a dependent global load per row per tile; measured geomean 68.0us against 62.5us
    # for the window. A window tile is written by different threads than read it, so it
    # costs a barrier, but one per window rather than one per tile.
    idx_window = IndexGather(
        (BN * IWT,),
        index_extent=N_IDX,
        row_stride=1,
        clamp_index=BUFFER_BOUND,
        scalar_args=("kv_base", "win_base", "valid_len"),
        name="idx_window",
    )
    # MLA has a single kv head, so the cache carries only (page, slot, dim) and the gather
    # picks no head; page_size 1 leaves no slot axis for the macro either.
    gather_ckv = PagedGather(
        (BN, head_dim_ckv),
        source_rows=NUM_PAGES,
        index_extent=BN * IWT,
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
        paged_axes=True,
        clamp_index=BUFFER_BOUND,
        clamp_row=BUFFER_BOUND,
        scalar_args=("win_off",),
        dtype=DTYPE,
        out_dtype=DTYPE,
        name="paged_gather_kpe",
    )
    mma_0 = Matmul(
        (MQ, BN, head_dim_ckv),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="cols",
    )
    mma_pe = Matmul(
        (MQ, BN, head_dim_kpe),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_pe",
        transpose_B=True,
        warp_policy="cols",
    )
    add_0 = Add(
        (MQ, BN),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="add_0",
        n_batch_dims=0,
    )
    exp_0 = MaskedExp(
        (MQ, BN),
        scale=None,
        scalar_args=("tile_base", "valid_len", "scale2"),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="exp_0",
        n_batch_dims=0,
    )
    # The denominator is accumulated as a whole (MQ, BN) tile and folded to a row vector
    # once, after the loop, rather than folded every iteration. A row fold is cross-lane
    # work; MLA runs the most iterations of any of these operators, so paying it once per
    # block instead of once per tile is worth the register the tile costs. Measured on the
    # reference kernel, folding per tile costs 62.0 -> 67.8 us geomean.
    psum_0 = Accumulate(
        (MQ, BN), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="psum_0"
    )
    mma_1 = Matmul(
        (MQ, head_dim_ckv, BN),
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

    ph_qn.name = "PlaceHolder_QN"
    ph_ckv.name = "PlaceHolder_CKV"
    ph_idx.name = "PlaceHolder_KVI"
    ph_qp.name = "PlaceHolder_QP"
    ph_kpe.name = "PlaceHolder_KPE"

    inner_graph = Graph(nodes=gather_all_nodes(psum_0, mma_1))
    # A serial loop with a runtime trip count never reads the static extent; the real
    # bound is loop_trip, which is per-request.
    inner_graph.extent = 1
    inner_graph.loop_kind = "serial"
    inner_graph.loop_trip = "T.ceildiv(valid_len, %d)" % BN
    inner_graph.window_nodes = [idx_window]
    inner_graph.window_guard = "win_off == 0"
    # Both cache tiles are also read one iteration ahead, out of the same shared window.
    # The look-ahead stops at the window's edge, so the first tile of every window is not
    # covered by it; codegen re-seeds the fragments from the window on the iteration that
    # refills it, which is exactly where that tile's indices become available.
    #
    # This is what makes 256 threads the right block size: the compressed-kv tile is
    # BN x 512, which is 128 elements a thread at 256 and 256 at 128, and the latter spills.
    # Neither the prefetch nor the wider block pays on its own -- measured 0.24x at 128 with
    # prefetch, 0.29x at 256 without it -- so they are one choice, not two.
    inner_graph.prefetch_nodes = [gather_ckv, gather_kpe]
    inner_graph.prefetch_guard = (
        f"(win_off + {BN} < {BN * IWT}) & (tile_base + {BN} < valid_len)"
    )
    inner_graph.sargs = [
        ("q_row0", "int32"),
        ("q_head0", "int32"),
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("scale2", "float32"),
    ]
    inner_graph.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("scale2", "float32"),
        ("kv_base", "int32"),
        ("win_base", "int32"),
        ("win_off", "int32"),
    ]
    inner_graph.smap = {
        "q_row0": lambda i: "q_row0",
        "q_head0": lambda i: "q_head0",
        "kv_base": lambda i: "kv_base",
        "valid_len": lambda i: "valid_len",
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

    # A decode request is one query token, so the tile's rows walk that token's head axis
    # from head 0 and the whole head count is valid.
    qn_tile = RowGather(
        (MQ, head_dim_ckv),
        source_rows=BATCH,
        n_valid=MQ,
        src_dims=3,
        source_heads=num_qo_heads,
        scalar_args=("q_row0", "q_head0"),
        dtype=DTYPE,
        out_dtype=DTYPE,
        name="qn_tile",
    )
    qn_tile.source_args = ("PlaceHolder_QN",)
    inner_qn.producer_op = qn_tile

    qp_tile = RowGather(
        (MQ, head_dim_kpe),
        source_rows=BATCH,
        n_valid=MQ,
        src_dims=3,
        source_heads=num_qo_heads,
        scalar_args=("q_row0", "q_head0"),
        dtype=DTYPE,
        out_dtype=DTYPE,
        name="qp_tile",
    )
    qp_tile.source_args = ("PlaceHolder_QP",)
    inner_qp.producer_op = qp_tile

    rowsum_0 = Reduction(
        (MQ, BN), axis=-1, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    e_ps = AccumEdge(node_0, rowsum_0, psum_id, 0, accum_node=psum_0)
    node_0._out_edges.append(e_ps)
    rowsum_0.set_inputs(0, e_ps)

    div_0 = Div(
        (MQ, head_dim_ckv),
        broadcast=0,
        denom_floor=DENOM_FLOOR,
        dtype=ACCUM_DTYPE,
        out_dtype=OUT_DTYPE,
        name="div_0",
        n_batch_dims=0,
    )
    e_val = AccumEdge(node_0, div_0, value_id, 0, accum_node=mma_1)
    node_0._out_edges.append(e_val)
    div_0.set_inputs(0, e_val)
    _connect(rowsum_0, div_0, 0, 1)

    log2_0 = Log2(
        (MQ,), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="log2_0", n_batch_dims=0
    )
    _connect(rowsum_0, log2_0, 0, 0)

    out_o = OutputNode(div_0, dtype=OUT_DTYPE)
    out_o.name = "OutputNode_0"
    out_lse = OutputNode(log2_0, dtype=ACCUM_DTYPE)
    out_lse.name = "OutputNode_1"

    # Read for its values, not gathered from, so it stays out of the *_iter call.
    ph_indptr = PlaceHolderNode(name="KV_INDPTR")
    ph_indptr.set_shape((1,))
    ph_indptr.set_dtype(_int32())
    ph_indptr.index_only = True

    # Inputs lead the node list in the operator's declared order, both so the emitted
    # signature matches it and because a symbolic output extent is resolved at call time
    # from an already-bound input.
    graph = Graph(
        nodes=[ph_qn, ph_qp, ph_ckv, ph_kpe, ph_indptr, ph_idx]
        + gather_all_nodes(node_0)
    )
    graph.dyn_dims = [BATCH, NUM_PAGES, LEN_INDPTR, N_IDX]
    graph.scalar_params = [("sm_scale", "float32")]
    graph.pass_configs = {
        # Bounds-predication is off because every read this kernel makes already clamps
        # itself: the gathers clamp the page-table subscript to the table and the page id
        # to the cache, and the q tiles guard their own valid rows. Left on, the legalizer
        # wraps each load in a bounds test and, because the page-table entry sits INSIDE
        # the address of the cache read, it re-materialises that entry once per test --
        # three global loads per row instead of one, all on the critical path.
        "tl.disable_safe_memory_legalize": True,
    }
    graph.n_threads = N_THREADS
    # One CTA per request: MLA has a single kv head, so there is no head axis to split the
    # grid on and all 16 query heads are the tile's rows.
    graph.extent = (BATCH,)
    graph.sargs = [
        ("q_row0", "int32"),
        ("q_head0", "int32"),
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("scale2", "float32"),
    ]
    graph.smap = {
        "q_row0": lambda i: f"{i}",
        "q_head0": lambda i: "0",
        "kv_base": lambda i: f"{ph_indptr.name}[{i}]",
        "valid_len": lambda i: (f"{ph_indptr.name}[{i} + 1] - {ph_indptr.name}[{i}]"),
        "scale2": lambda i: f"sm_scale * {LOG2E}",
    }
    lock_graph(graph)

    qn_shape = (BATCH, num_qo_heads, head_dim_ckv)
    qp_shape = (BATCH, num_qo_heads, head_dim_kpe)
    ckv_shape = (NUM_PAGES, PAGE_SIZE, head_dim_ckv)
    kpe_shape = (NUM_PAGES, PAGE_SIZE, head_dim_kpe)
    whole = lambda shape: Iterator(  # noqa: E731
        lambda i: tuple("0" for _ in shape), shape=tuple(shape), tile_size=tuple(shape)
    )
    graph.tmap[ph_qn] = whole(qn_shape)
    graph.tmap[ph_qp] = whole(qp_shape)
    graph.tmap[ph_ckv] = whole(ckv_shape)
    graph.tmap[ph_kpe] = whole(kpe_shape)
    graph.tmap[ph_idx] = whole((N_IDX,))
    graph.tmap[ph_indptr] = whole((LEN_INDPTR,))
    graph.tmap[node_0] = [
        Iterator(lambda i: ("0",), shape=(MQ,), tile_size=(MQ,)),
        Iterator(
            lambda i: ("0", "0"),
            shape=(MQ, head_dim_ckv),
            tile_size=(MQ, head_dim_ckv),
        ),
    ]

    out_offset = lambda i: (f"{i}", "0", "0")  # noqa: E731
    lse_offset = lambda i: (f"{i}", "0")  # noqa: E731
    graph.tmap[rowsum_0] = Iterator(
        lambda i: ("0",), shape=(MQ,), tile_size=(MQ,)
    )
    graph.tmap[div_0] = Iterator(
        out_offset,
        shape=(BATCH, num_qo_heads, head_dim_ckv),
        tile_size=(1, num_qo_heads, head_dim_ckv),
    )
    graph.tmap[out_o] = graph.tmap[div_0]
    graph.tmap[log2_0] = Iterator(
        lse_offset, shape=(BATCH, num_qo_heads), tile_size=(1, num_qo_heads)
    )
    graph.tmap[out_lse] = graph.tmap[log2_0]

    inner_graph.tmap[ph_qn] = whole(qn_shape)
    inner_graph.tmap[ph_qp] = whole(qp_shape)
    inner_graph.tmap[inner_qn] = whole((MQ, head_dim_ckv))
    inner_graph.tmap[inner_qp] = whole((MQ, head_dim_kpe))
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
        lambda i: ("0", "0"), shape=(MQ, BN), tile_size=(MQ, BN)
    )
    inner_graph.tmap[mma_pe] = inner_graph.tmap[mma_0]
    inner_graph.tmap[add_0] = inner_graph.tmap[mma_0]
    inner_graph.tmap[exp_0] = inner_graph.tmap[mma_0]
    inner_graph.tmap[psum_0] = inner_graph.tmap[mma_0]
    inner_graph.tmap[mma_1] = Iterator(
        lambda i: ("0", "0"), shape=(MQ, head_dim_ckv), tile_size=(MQ, head_dim_ckv)
    )

    graph.tinfo_map = {
        ph_qn: ("PlaceHolder_QN", "g"),
        ph_qp: ("PlaceHolder_QP", "g"),
        ph_ckv: ("PlaceHolder_CKV", "g"),
        ph_kpe: ("PlaceHolder_KPE", "g"),
        ph_idx: ("PlaceHolder_KVI", "g"),
        ph_indptr: ("PlaceHolder_KV_INDPTR", "g"),
        node_0: ("t_GraphNode_0", "r"),
        rowsum_0: ("t_rowsum_0", "r"),
        div_0: ("OutputNode_0", "g"),
        log2_0: ("OutputNode_1", "g"),
        out_o: ("OutputNode_0", "g"),
        out_lse: ("OutputNode_1", "g"),
    }
    inner_graph.tinfo_map = {
        ph_qn: ("PlaceHolder_QN", "g"),
        ph_qp: ("PlaceHolder_QP", "g"),
        ph_ckv: ("PlaceHolder_CKV", "g"),
        ph_kpe: ("PlaceHolder_KPE", "g"),
        ph_idx: ("PlaceHolder_KVI", "g"),
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
