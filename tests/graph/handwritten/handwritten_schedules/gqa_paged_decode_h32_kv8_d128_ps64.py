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
PAGE_SIZE = 64

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
N_THREADS = 128
# Floors the softmax denominator. A request whose KV window is empty runs zero KV tiles,
# leaving a zero numerator over a zero denominator; the floor turns that 0/0 into the 0 the
# operator defines, instead of a NaN the benchmark's sanity check would reject outright.
DENOM_FLOOR = 1e-30


def _connect(src, dst, src_id, dst_id):
    edge = Edge(src, dst, src_id, dst_id)
    src._out_edges.append(edge)
    dst.set_inputs(dst_id, edge)


def get_handwritten_scheduled_graph(dims):
    # The page size is part of the operator's identity rather than a schedule knob, so a
    # workload tuple may carry it but may not disagree with it.
    num_qo_heads, num_kv_heads, head_dim = dims[:3]
    group = num_qo_heads // num_kv_heads

    assert num_qo_heads == 32, num_qo_heads
    assert num_kv_heads == 8, num_kv_heads
    assert head_dim == 128, head_dim
    assert group == 4, group
    assert dims[3:] in ((), (PAGE_SIZE,)), dims

    # A tile of BN keys lies inside one page, so the whole page table lookup one tile
    # needs is a single entry: the gathers read kv_indices in place off a token-unit
    # base, which is what makes ``page_size`` tokens per entry a pure address split.
    assert PAGE_SIZE % BN == 0, (PAGE_SIZE, BN)

    # Every bound the gathers need is read off the buffer they receive, so no extent has
    # to be pinned at compile time: one compiled kernel serves every workload shape.
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
    mma_1 = Matmul(
        (MQ, head_dim, BN),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="cols",
    )

    mma_1.stage_fragment_operands_in_shared = (0,)

    _connect(mma_0, exp_0, 0, 0)
    _connect(exp_0, rowsum_0, 0, 0)
    _connect(exp_0, mma_1, 0, 0)

    ph_q = mma_0.inputs[0].src_node
    ph_kf = mma_0.inputs[1].src_node
    ph_vf = mma_1.inputs[1].src_node
    # The page table is read for the addresses in the cache slices below, so it feeds no
    # node; it is listed among the graph's nodes to stay a declared kernel input.
    ph_idx = PlaceHolderNode(name="KV_INDICES")
    ph_idx.set_shape((1,))
    ph_idx.set_dtype(_int32())
    ph_idx.is_batch_dim = False

    ph_q.name = "PlaceHolder_Q"
    ph_kf.name = "PlaceHolder_K_CACHE"
    ph_vf.name = "PlaceHolder_V_CACHE"
    ph_idx.name = "PlaceHolder_KV_INDICES"

    # The page table is an input of the tier that loops over pages: the slab offsets
    # below read it, so it has to be in scope inside that tier, not just the kernel.
    inner_graph = Graph(nodes=[ph_idx] + gather_all_nodes(rowsum_0, mma_1))
    # A serial loop with a runtime trip count never reads the static extent; the real
    # bound is loop_trip, which is per-request.
    inner_graph.extent = 1
    # Pipelined, and it has to be: from tilelang 0.1.13 a copy is only lowered to TMA when
    # it sits inside a pipelined producer/consumer loop, because that is what supplies the
    # mbarrier arrive/wait protocol (`ProducerConsumerWarpSpecialized` returns early on a
    # loop with no `num_stages` annotation). A serial loop here therefore gets no TMA, and
    # the KV slabs -- passed as BufferRegions for TMA's sake alone -- become pure cost:
    # measured 8.5x slower than the reference kernel, against 1.03x when pipelined.
    #
    # This loop was serial until 0.1.13 because pipelining it was WRONG on 0.1.5: at depth 2
    # it corrupted long KV windows and at depth 3 any window shorter than the depth. Both
    # were tilelang defects and both are fixed -- re-verified at depths 2, 3 and 4 over trip
    # counts 0, 1, 2, 3, 4, 64 and 256 tiles plus heterogeneous batches mixing one-tile and
    # 128-tile requests, all bit-exact against the reference.
    inner_graph.loop_kind = "pipelined"
    inner_graph.loop_stages = 2
    inner_graph.loop_trip = "T.ceildiv(valid_len, %d)" % BN
    inner_graph.sargs = [
        ("q_row0", "int32"),
        ("q_head0", "int32"),
        ("kv_page_base", "int32"),
        ("valid_len", "int32"),
        ("head", "int32"),
        ("scale2", "float32"),
    ]
    inner_graph.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("scale2", "float32"),
    ]
    inner_graph.smap = {
        "q_row0": lambda i: "q_row0",
        "q_head0": lambda i: "q_head0",
        "kv_page_base": lambda i: "kv_page_base",
        "head": lambda i: "head",
        "valid_len": lambda i: "valid_len",
        "scale2": lambda i: "scale2",
        "tile_base": lambda i: f"{BN} * {i}",
    }
    lock_graph(inner_graph)

    node_0 = GraphNode(name="GraphNode_0", graph=inner_graph)
    rowsum_id = node_0.graph.accum_nodes.index(rowsum_0)
    value_id = node_0.graph.accum_nodes.index(mma_1)

    inner_q = inner_graph.binding[ph_q.name]
    inner_idx = inner_graph.binding[ph_idx.name]
    inner_k = inner_graph.binding[ph_kf.name]
    inner_v = inner_graph.binding[ph_vf.name]

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
    ph_last = PlaceHolderNode(name="KV_LAST_PAGE_LEN")
    ph_last.set_shape((1,))
    ph_last.set_dtype(_int32())
    ph_last.index_only = True

    # Inputs lead the node list in the operator's declared order, both so the emitted
    # signature matches it and because a symbolic output extent is resolved at call time
    # from an already-bound input.
    graph = Graph(
        nodes=[ph_q, ph_kf, ph_vf, ph_indptr, ph_idx, ph_last]
        + gather_all_nodes(node_0)
    )
    graph.dyn_dims = [BATCH, NUM_PAGES, LEN_INDPTR, N_IDX]
    graph.scalar_params = [("sm_scale", "float32")]
    graph.n_threads = N_THREADS
    # No L2 swizzle. Rasterising the grid costs a per-CTA index remap and buys locality
    # between neighbouring blocks; here the blocks do not share enough KV to repay it.
    # Measured vs the reference kernel: 1.284x -> 1.192x.
    graph.rasterization = 1
    # One CTA per (kv head, request): a decode request is a single query token, so its
    # whole head group fits one tile and there is no work list to scan. The kv head leads
    # the grid so that adjacent CTAs share a request's page-table entries and, at page
    # sizes above one, the very cache pages those entries point at.
    graph.extent = (f"{num_kv_heads}", f"{BATCH}")
    graph.sargs = [
        ("q_row0", "int32"),
        ("q_head0", "int32"),
        ("kv_page_base", "int32"),
        ("valid_len", "int32"),
        ("head", "int32"),
        ("scale2", "float32"),
    ]

    # Only kv_last_page_len says how full the request's last page is, so the page count
    # alone would round the length up to a whole page.
    n_pages = lambda b: (  # noqa: E731
        f"({ph_indptr.name}[{b} + 1] - {ph_indptr.name}[{b}])"
    )
    graph.smap = {
        "q_row0": lambda h, b: f"{b}",
        "q_head0": lambda h, b: f"{h} * {group}",
        "kv_page_base": lambda h, b: f"{ph_indptr.name}[{b}]",
        # Clamped as a whole to the pages the request owns: that bounds a corrupt last-page
        # length to this request instead of letting it attend to the next one's pages, and
        # it is also what makes an empty request (zero pages) come out as zero.
        "valid_len": lambda h, b: (
            f"T.min(T.max({n_pages(b)} - 1, 0) * {PAGE_SIZE}"
            f" + T.max({ph_last.name}[{b}], 0), {n_pages(b)} * {PAGE_SIZE})"
        ),
        "head": lambda h, b: f"{h}",
        "scale2": lambda h, b: f"sm_scale * {LOG2E}",
    }
    lock_graph(graph)

    q_shape = (BATCH, num_qo_heads, head_dim)
    cache_shape = (NUM_PAGES, PAGE_SIZE, num_kv_heads, head_dim)
    whole = lambda shape: Iterator(  # noqa: E731
        lambda h, b: tuple("0" for _ in shape),
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
    graph.tmap[ph_last] = whole((BATCH,))
    graph.tmap[node_0] = [
        Iterator(lambda h, b: ("0",), shape=(MQ,), tile_size=(MQ,)),
        Iterator(
            lambda h, b: ("0", "0"), shape=(MQ, head_dim), tile_size=(MQ, head_dim)
        ),
    ]

    out_offset = lambda h, b: (f"{b}", f"{group} * {h}", "0")  # noqa: E731
    lse_offset = lambda h, b: (f"{b}", f"{group} * {h}")  # noqa: E731
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
    page_of = lambda i: (  # noqa: E731
        f"T.min({inner_idx.name}[T.min(kv_page_base + ({BN} * {i}) // {PAGE_SIZE}, "
        f"{N_IDX} - 1)], {NUM_PAGES} - 1)"
    )
    cache_slab = lambda: Iterator(  # noqa: E731
        lambda i: (page_of(i), f"({BN} * {i}) % {PAGE_SIZE}", "head", "0"),
        shape=cache_shape,
        tile_size=(1, BN, 1, head_dim),
    )
    inner_graph.tmap[inner_idx] = inner_whole((N_IDX,))
    # A BufferRegion is only worth it here BECAUSE it unlocks TMA -- the earlier
    # measurement that rejected regions for this operator (170us -> 952us, 5.6x) was taken
    # with TMA lowering disabled, which is the right setting for the ragged/MLA-prefill
    # graphs but the wrong one here. Without TMA a region buys no wider access (a slab is
    # contiguous, so the `_S` view already copies as uint4 vectors) and only adds
    # tilelang's bounds predicate, which cannot be discharged because the leading index is
    # a page-table value -- so every vector load is guarded on a re-read of kv_indices.
    inner_graph.tmap[inner_k] = cache_slab()
    inner_graph.tmap[inner_v] = cache_slab()
    # Hand the slabs down as BufferRegions and leave TMA lowering ON. A slab lies inside
    # one page addressed by a scalar page id, which tilelang can turn into a TMA
    # descriptor: one bulk-copy instruction per tile instead of a per-element gather
    # (measured 28x fewer global load instructions for identical DRAM bytes, 1.61x faster).
    # Do NOT disable TMA here -- without it the region form is 5.5x SLOWER than `_S`.
    inner_graph.region_args = [inner_k, inner_v]
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
        ph_last: ("PlaceHolder_KV_LAST_PAGE_LEN", "g"),
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
        inner_graph.binding[ph_kf.name]: (inner_graph.binding[ph_kf.name].name, "g"),
        inner_graph.binding[ph_vf.name]: (inner_graph.binding[ph_vf.name].name, "g"),
        inner_idx: (inner_idx.name, "g"),
        inner_k: (inner_k.name, "s"),
        inner_v: (inner_v.name, "s"),
        mma_0: ("t_mma_0", "r"),
        exp_0: ("t_exp_0", "r"),
        rowsum_0: ("t_rowsum_0", "r"),
        mma_1: ("t_mma_1", "r"),
    }

    graph = passes.annotate_args(graph, is_main=True)
    graph = passes.annotate_tc_layout(graph)
    graph = passes.annotate_fragment_layout_conflicts(graph)
    return graph


def _int32():
    import tvm

    return tvm.DataType("int32")
