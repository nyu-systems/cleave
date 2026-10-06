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
    IndptrTileScan,
    MaskedDiv,
    MaskedLog2,
    Matmul,
    Reduction,
)
from cleave.util import gather_all_nodes, lock_graph

DTYPE = "bfloat16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "bfloat16"

BM = 64
# The reference kernel runs BN=16, but it pipelines its KV loop and so wants the smallest
# tile that keeps the matmuls fed; this loop is serial, so a wider tile is strictly better
# -- it amortises the barrier and the loop's own scalar work over more columns. Measured
# geomean over these workloads: BN=16 46.1us, BN=32 41.7us, BN=64 33.3us.
BN = 64

# exp2 is a single instruction where exp is a call, so the scale is carried in the log2
# domain: exp(x * s) == exp2(x * s * LOG2E).
LOG2E = 1.4426950408889634
# Floors the softmax denominator. A query row whose causal window is empty sums to zero
# over a zero numerator; the floor turns that 0/0 into the 0 the operator defines,
# instead of a NaN the benchmark's sanity check would reject outright.
DENOM_FLOOR = 1e-30

# Symbolic extents, one per var axis. Each appears bare in some tensor shape (TOTAL_Q in
# q, TOTAL_KV in k/v, LEN_INDPTR in qo_indptr) so the runtime infers it from the bound
# tensor; afterwards it may be used in extent expressions.
TOTAL_Q = "total_q"
TOTAL_KV = "total_kv"
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

    # The kernel takes the operator's declared inputs and nothing else: the work list a
    # host planner used to precompute is derived on the device by scanning qo_indptr
    # (``graph.prologue``), and every per-block scalar below is an expression over that
    # scan's result plus the two indptr tensors.
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
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="exp_0",
        n_batch_dims=0,
    )
    rowsum_0 = Reduction(
        (BM, BN), axis=-1, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    mma_1 = Matmul(
        (BM, head_dim, BN),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="cols",
    )

    # P (the softmax result) is left in its fragment rather than staged through shared:
    # tilelang then feeds it to the second matmul in place (wgmma_rs) instead of storing it
    # with ptx_stmatrix and reading it back (wgmma_ss). Worth 1.04x here. The paged
    # operators keep the shared hop -- their P tile is the one their layout pass needs.

    _connect(mma_0, exp_0, 0, 0)
    _connect(exp_0, rowsum_0, 0, 0)
    _connect(exp_0, mma_1, 0, 0)

    ph_q = mma_0.inputs[0].src_node
    ph_kf = mma_0.inputs[1].src_node
    ph_vf = mma_1.inputs[1].src_node

    ph_q.name = "PlaceHolder_Q"
    ph_kf.name = "PlaceHolder_K"
    ph_vf.name = "PlaceHolder_V"

    inner_graph = Graph(nodes=gather_all_nodes(rowsum_0, mma_1))
    # A serial loop with a runtime trip count never reads the static extent.
    inner_graph.extent = 1
    # Pipelined, and warp-specialised with it. The KV tiles are affine slices handed down
    # as BufferRegions, which is exactly what TMA and the producer/consumer mbarrier
    # protocol want; a serial body gave that protocol no arrivals, which is why both were
    # switched off here. Pipelining supplies them, so both come back on and the loads
    # overlap the matmuls. Measured over these workloads: total 4353us -> 3329us (kv8) and
    # 2327us -> 1887us (kv4), and the kernel compiles 6x faster.
    #
    # It has to be the warp-specialised path, not tilelang's cp.async pipeline: that one
    # peels its epilogue over `trip + stages - trip` without simplifying the extent
    # (inject_pipeline.cc, `EmitImpl`), so a data-dependent trip count -- which is what
    # ragged means -- leaves a symbolic loop around `ptx_wait_group` and fails codegen.
    inner_graph.loop_kind = "pipelined"
    inner_graph.loop_stages = 2
    inner_graph.loop_trip = "T.ceildiv(valid_len, %d)" % BN
    inner_graph.sargs = [
        ("q_row0", "int32"),
        ("q_tile_base", "int32"),
        ("q_len", "int32"),
        ("q_head", "int32"),
        ("kv_row0", "int32"),
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
        ("kv_row0", "int32"),
        ("kv_head", "int32"),
    ]
    inner_graph.smap = {
        "q_row0": lambda i: "q_row0",
        "q_tile_base": lambda i: "q_tile_base",
        "q_len": lambda i: "q_len",
        "q_head": lambda i: "q_head",
        "kv_row0": lambda i: "kv_row0",
        "valid_len": lambda i: "valid_len",
        "diag": lambda i: "diag",
        "kv_head": lambda i: "kv_head",
        "scale2": lambda i: "scale2",
        "tile_base": lambda i: f"{BN} * {i}",
    }
    lock_graph(inner_graph)

    node_0 = GraphNode(name="GraphNode_0", graph=inner_graph)
    rowsum_id = node_0.graph.accum_nodes.index(rowsum_0)
    value_id = node_0.graph.accum_nodes.index(mma_1)

    inner_q = inner_graph.binding[ph_q.name]
    inner_k = inner_graph.binding[ph_kf.name]
    inner_v = inner_graph.binding[ph_vf.name]

    div_0 = MaskedDiv(
        (BM, head_dim),
        broadcast=0,
        scalar_args=("n_rows",),
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

    log2_0 = MaskedLog2(
        (BM,),
        scalar_args=("n_rows",),
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
        nodes=[ph_q, ph_kf, ph_vf, ph_qo, ph_kv] + gather_all_nodes(node_0)
    )
    graph.dyn_dims = [TOTAL_Q, TOTAL_KV, LEN_INDPTR]
    graph.scalar_params = [("sm_scale", "float32")]
    # One CTA per (query tile, qo head). sum(ceil(q_len/BM)) is at most
    # ceil(total_q/BM) + n_requests - 1 (ceil is subadditive up to one per term), so this
    # grid always covers the work list; the slack slots scan to request -1 and the body
    # guard skips them.
    graph.extent = (
        f"T.ceildiv({TOTAL_Q}, {BM}) + {LEN_INDPTR} - 2",
        f"{num_qo_heads}",
    )
    graph.prologue = [
        ScalarScan(
            op=IndptrTileScan(rows=BM),
            buffer="_wl",
            size=2,
            inputs=(ph_qo,),
            scalar_args=lambda i, h: (i, LEN_INDPTR),
        )
    ]
    graph.block_guard = lambda i, h: "_wl[0] >= 0"
    graph.region_args = [ph_q]
    # A BufferRegion lets tilelang reach for TMA, whose mbarrier protocol assumes a
    # pipelined producer/consumer body; these nested serial macros provide no arrivals for
    # it and the kernel spins on the barrier. Bounds-predication, which is the reason for
    # the regions, survives with TMA lowering off.
    # Fast math was OFF here, and had to be while the KV loop was serial: it cost
    # 2,532us -> 5,409us over these workloads. That is no longer true of the pipelined body
    # -- with warp specialisation the flag is worth 1.54x (14.61us -> 9.47us geomean) -- so
    # the seeded default stands, and only the bounds-predication opt-out is declared.
    #
    # Bounds-predication is off because every read is already in bounds by construction:
    # the KV slices are clamped to the request's window and the q tile guards its own valid
    # rows. Left on, the legalizer wraps each device-computed indptr read in a bounds test,
    # 72 of 84 predicated blocks in the loop body. Verified memcheck-clean.
    graph.pass_configs = {"tl.disable_safe_memory_legalize": True}
    graph.sargs = [
        ("n_rows", "int32"),
        ("q_row0", "int32"),
        ("q_tile_base", "int32"),
        ("q_len", "int32"),
        ("q_head", "int32"),
        ("kv_row0", "int32"),
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
    tile_base = f"(_wl[1] * {BM})"
    n_rows = f"T.min({BM}, {q_len} - {tile_base})"
    # End-aligned causal bound for the tile's first row, then the KV window causality
    # leaves live for the whole tile.
    diag = f"({kv_len} - {q_len} + {tile_base})"
    graph.smap = {
        "n_rows": lambda i, h: n_rows,
        "q_row0": lambda i, h: q_beg,
        "q_tile_base": lambda i, h: tile_base,
        "q_len": lambda i, h: q_len,
        "q_head": lambda i, h: f"{h}",
        "kv_row0": lambda i, h: kv_beg,
        "valid_len": lambda i, h: f"T.max(T.min({kv_len}, {diag} + {n_rows}), 0)",
        "diag": lambda i, h: diag,
        "kv_head": lambda i, h: f"{h} // {group}",
        "scale2": lambda i, h: f"sm_scale * {LOG2E}",
    }
    lock_graph(graph)

    whole = lambda shape: Iterator(  # noqa: E731
        lambda i, h: tuple("0" for _ in shape), shape=tuple(shape), tile_size=tuple(shape)
    )
    graph.tmap[ph_q] = Iterator(
        lambda i, h: (f"{q_beg} + {tile_base}", f"{h}", "0"),
        shape=(TOTAL_Q, num_qo_heads, head_dim),
        tile_size=(BM, 1, head_dim),
    )
    graph.tmap[ph_kf] = whole((TOTAL_KV, num_kv_heads, head_dim))
    graph.tmap[ph_vf] = whole((TOTAL_KV, num_kv_heads, head_dim))
    graph.tmap[ph_qo] = whole((LEN_INDPTR,))
    graph.tmap[ph_kv] = whole((LEN_INDPTR,))
    graph.tmap[node_0] = [
        Iterator(lambda i, h: ("0",), shape=(BM,), tile_size=(BM,)),
        Iterator(
            lambda i, h: ("0", "0"), shape=(BM, head_dim), tile_size=(BM, head_dim)
        ),
    ]

    q_abs = f"{q_beg} + {tile_base}"
    out_offset = lambda i, h: (q_abs, f"{h}", "0")  # noqa: E731
    lse_offset = lambda i, h: (q_abs, f"{h}")  # noqa: E731
    graph.tmap[div_0] = Iterator(
        out_offset,
        shape=(TOTAL_Q, num_qo_heads, head_dim),
        tile_size=(BM, 1, head_dim),
    )
    graph.tmap[out_o] = graph.tmap[div_0]
    graph.tmap[log2_0] = Iterator(
        lse_offset, shape=(TOTAL_Q, num_qo_heads), tile_size=(BM, 1)
    )
    graph.tmap[out_lse] = graph.tmap[log2_0]

    inner_whole = lambda shape: Iterator(  # noqa: E731
        lambda i: tuple("0" for _ in shape), shape=tuple(shape), tile_size=tuple(shape)
    )
    kv_tile = lambda: Iterator(  # noqa: E731
        lambda i: (f"kv_row0 + {BN} * {i}", "kv_head", "0"),
        shape=(TOTAL_KV, num_kv_heads, head_dim),
        tile_size=(BN, 1, head_dim),
    )
    inner_graph.tmap[inner_q] = inner_whole((BM, head_dim))
    inner_graph.tmap[inner_k] = kv_tile()
    inner_graph.tmap[inner_v] = kv_tile()
    inner_graph.region_args = [inner_k, inner_v]
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
        ph_kf: ("PlaceHolder_K", "g"),
        ph_vf: ("PlaceHolder_V", "g"),
        ph_qo: ("PlaceHolder_QO_INDPTR", "g"),
        ph_kv: ("PlaceHolder_KV_INDPTR", "g"),
        node_0: ("t_GraphNode_0", "r"),
        div_0: ("OutputNode_0", "g"),
        log2_0: ("OutputNode_1", "g"),
        out_o: ("OutputNode_0", "g"),
        out_lse: ("OutputNode_1", "g"),
    }
    inner_graph.tinfo_map = {
        ph_q: ("PlaceHolder_Q", "g"),
        ph_kf: ("PlaceHolder_K", "g"),
        ph_vf: ("PlaceHolder_V", "g"),
        inner_q: (inner_q.name, "s"),
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
