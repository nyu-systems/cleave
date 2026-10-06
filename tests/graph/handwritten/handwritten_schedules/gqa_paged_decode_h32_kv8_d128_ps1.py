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
PAGE_SIZE = 1

# Symbolic extents, one per var axis of the operator. Each appears bare in some tensor
# shape (BATCH in q, NUM_PAGES in k_cache/v_cache, LEN_INDPTR in kv_indptr, N_IDX in
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
    num_qo_heads, num_kv_heads, head_dim = dims
    group = num_qo_heads // num_kv_heads

    assert num_qo_heads == 32, num_qo_heads
    assert num_kv_heads == 8, num_kv_heads
    assert head_dim == 128, head_dim
    assert group == 4, group

    # Every bound the gathers need is read off the buffer they receive, so no extent has
    # to be pinned at compile time: one compiled kernel serves every workload shape.
    #
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
        paged_axes=True,
        source_heads=num_kv_heads,
        head_scalar=True,
        clamp_index=BUFFER_BOUND,
        clamp_row=BUFFER_BOUND,
        scalar_args=("kv_pos0", "head"),
        dtype=DTYPE,
        out_dtype=DTYPE,
        name="paged_gather_k",
    )
    mma_0 = Matmul(
        (MQ, BN, head_dim),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="cols",
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
    rowsum_0 = Reduction(
        (MQ, BN), axis=-1, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    gather_v = PagedGather(
        (BN, head_dim),
        source_rows=NUM_PAGES,
        index_extent=N_IDX,
        paged_axes=True,
        source_heads=num_kv_heads,
        head_scalar=True,
        clamp_index=BUFFER_BOUND,
        clamp_row=BUFFER_BOUND,
        scalar_args=("kv_pos0", "head"),
        dtype=DTYPE,
        out_dtype=DTYPE,
        name="paged_gather_v",
    )
    mma_1 = Matmul(
        (MQ, head_dim, BN),
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
    # A serial loop with a runtime trip count never reads the static extent; the real
    # bound is loop_trip, which is per-request.
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
        ("q_head0", "int32"),
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("head", "int32"),
        ("scale2", "float32"),
    ]
    inner_graph.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("scale2", "float32"),
        ("kv_pos0", "int32"),
        ("head", "int32"),
    ]
    inner_graph.smap = {
        "q_row0": lambda i: "q_row0",
        "q_head0": lambda i: "q_head0",
        "kv_base": lambda i: "kv_base",
        "head": lambda i: "head",
        "valid_len": lambda i: "valid_len",
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

    q_tile = RowGather(
        (MQ, head_dim),
        source_rows=BATCH,
        n_valid=group,
        src_dims=3,
        source_heads=num_qo_heads,
        scalar_args=("q_row0", "q_head0"),
        dtype=DTYPE,
        out_dtype=DTYPE,
        name="q_tile",
    )
    q_tile.source_args = ("PlaceHolder_Q",)
    inner_q.producer_op = q_tile

    div_0 = Div(
        (group, head_dim),
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
    e_rs = AccumEdge(node_0, div_0, rowsum_id, 1, accum_node=rowsum_0)
    node_0._out_edges.append(e_rs)
    div_0.set_inputs(1, e_rs)

    log2_0 = Log2(
        (group,),
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

    # Read for their values, not gathered from, so they stay out of the *_iter call.
    ph_indptr = PlaceHolderNode(name="KV_INDPTR")
    ph_indptr.set_shape((1,))
    ph_indptr.set_dtype(_int32())
    ph_indptr.index_only = True

    # Inputs lead the node list in the operator's declared order, both so the emitted
    # signature matches it and because a symbolic output extent is resolved at call time
    # from an already-bound input.
    graph = Graph(
        nodes=[ph_q, ph_kf, ph_vf, ph_indptr, ph_idx] + gather_all_nodes(node_0)
    )
    graph.dyn_dims = [BATCH, NUM_PAGES, LEN_INDPTR, N_IDX]
    # Bounds-predication is off because every read this kernel makes already clamps
    # itself: the gathers clamp the page-table subscript to the table and the page id to
    # the cache, and the q tile guards its own valid rows. Left on, the legalizer wraps
    # each load in a bounds test and, because the page-table entry sits INSIDE the address
    # of the cache read, it re-materialises that entry once per test -- three global loads
    # per row instead of one, all on the critical path of the address. Measured over these
    # workloads: 106us -> 52us per workload geomean.
    graph.pass_configs = {"tl.disable_safe_memory_legalize": True}
    graph.scalar_params = [("sm_scale", "float32")]
    # One CTA per (request, kv head): a decode request is a single query token, so its
    # whole head group fits one tile and there is no work list to scan.
    # No L2 swizzle. Rasterising the grid costs a per-CTA index remap and buys locality
    # between neighbouring blocks; here the blocks do not share enough KV to repay it.
    # Measured vs the reference kernel: 1.159x -> 1.109x.
    graph.rasterization = 1
    graph.extent = (f"{BATCH}", f"{num_kv_heads}")
    graph.sargs = [
        ("q_row0", "int32"),
        ("q_head0", "int32"),
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("head", "int32"),
        ("scale2", "float32"),
    ]
    graph.smap = {
        "q_row0": lambda b, h: f"{b}",
        "q_head0": lambda b, h: f"{h} * {group}",
        "kv_base": lambda b, h: f"{ph_indptr.name}[{b}]",
        "valid_len": lambda b, h: (
            f"{ph_indptr.name}[{b} + 1] - {ph_indptr.name}[{b}]"
        ),
        "head": lambda b, h: f"{h}",
        "scale2": lambda b, h: f"sm_scale * {LOG2E}",
    }
    lock_graph(graph)

    q_shape = (BATCH, num_qo_heads, head_dim)
    cache_shape = (NUM_PAGES, PAGE_SIZE, num_kv_heads, head_dim)
    whole = lambda shape: Iterator(  # noqa: E731
        lambda b, h: tuple("0" for _ in shape),
        shape=tuple(shape),
        tile_size=tuple(shape),
    )
    inner_whole = lambda shape: Iterator(  # noqa: E731
        lambda i: tuple("0" for _ in shape), shape=tuple(shape), tile_size=tuple(shape)
    )
    graph.tmap[ph_q] = whole(q_shape)
    graph.tmap[ph_kf] = whole(cache_shape)
    graph.tmap[ph_vf] = whole(cache_shape)
    graph.tmap[ph_idx] = whole((N_IDX,))
    graph.tmap[ph_indptr] = whole((LEN_INDPTR,))
    graph.tmap[node_0] = [
        Iterator(lambda b, h: ("0",), shape=(MQ,), tile_size=(MQ,)),
        Iterator(
            lambda b, h: ("0", "0"), shape=(MQ, head_dim), tile_size=(MQ, head_dim)
        ),
    ]

    out_offset = lambda b, h: (f"{b}", f"{group} * {h}", "0")  # noqa: E731
    lse_offset = lambda b, h: (f"{b}", f"{group} * {h}")  # noqa: E731
    graph.tmap[div_0] = Iterator(
        out_offset,
        shape=(BATCH, num_qo_heads, head_dim),
        tile_size=(1, group, head_dim),
    )
    graph.tmap[out_o] = graph.tmap[div_0]
    graph.tmap[log2_0] = Iterator(
        lse_offset, shape=(BATCH, num_qo_heads), tile_size=(1, group)
    )
    graph.tmap[out_lse] = graph.tmap[log2_0]

    inner_graph.extent = 1
    inner_graph.tmap[ph_q] = inner_whole(q_shape)
    inner_graph.tmap[inner_q] = inner_whole((MQ, head_dim))
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
        lambda i: ("0", "0"), shape=(MQ, BN), tile_size=(MQ, BN)
    )
    inner_graph.tmap[exp_0] = inner_graph.tmap[mma_0]
    inner_graph.tmap[rowsum_0] = Iterator(
        lambda i: ("0",), shape=(MQ,), tile_size=(MQ,)
    )
    inner_graph.tmap[mma_1] = Iterator(
        lambda i: ("0", "0"), shape=(MQ, head_dim), tile_size=(MQ, head_dim)
    )

    graph.tinfo_map = {
        ph_q: ("PlaceHolder_Q", "g"),
        ph_kf: ("PlaceHolder_K_CACHE", "g"),
        ph_vf: ("PlaceHolder_V_CACHE", "g"),
        ph_idx: ("PlaceHolder_KV_INDICES", "g"),
        ph_indptr: ("PlaceHolder_KV_INDPTR", "g"),
        node_0: ("t_GraphNode_0", "r"),
        div_0: ("OutputNode_0", "g"),
        log2_0: ("OutputNode_1", "g"),
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
