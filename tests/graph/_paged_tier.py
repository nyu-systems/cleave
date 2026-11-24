"""A hand-built paged-shaped kernel with the exact (running-max) masked softmax.

The shape of a paged attention block, built from IR constructors only -- none of the
``cleave.dynamic.paging`` helpers -- so it probes the ops and the codegen the paged builders
rely on, apart from the builders themselves:

* a ``(ROWS, BN)`` score tile out of a column-split gemm at 128 threads (a decode block:
  ``ROWS`` staged query rows of which ``GROUP`` are real), or with ``causal=True`` the
  ``(tokens * GROUP, BN)`` tile of a prefill block whose row axis fuses the head group;
* a serial KV loop with a *runtime* trip count ``ceildiv(valid_len, BN)``, which may be 0;
* the softmax scale as a runtime scalar in the log2 domain, and a mask from loop scalars.

    per KV tile:
        s      = mask ? (q . k) * scale2 : -FLT_MAX        MaskedScore
        m_prev = m                                         lag node
        m      = max(m, rowmax(s))                         carried, starts at -FLT_MAX
        P      = mask ? exp2(s - m) : 0                    MaskedExp(shift=True)
        alpha  = exp2(m_prev - m)                          Exp2
        l      = l * alpha + rowsum(P)                     ReduceSum
        O      = O * alpha[row] + P @ V                    Matmul
    after the loop:
        out = O / max(l, 1e-30)        lse = log2(l) + m   Div, Lse2 (row-masked if causal)

The KV length is a declared scalar input here (``kv_len``) rather than read off an indptr,
and K / V are plain slices rather than page-table gathers: neither changes the loop body.
"""

from __future__ import annotations

from types import SimpleNamespace

from tilelang.carver.roller import Edge

from cleave.codegen import TilelangCodeGen, passes
from cleave.ir.graph import AccumEdge, Graph, GraphNode, Iterator, OutputNode
from cleave.ir.graph_utils import gather_all_nodes, lock_graph
from cleave.ir.ops import (
    NEG_FLT_MAX,
    CausalMaskedExp,
    CausalMaskedScore,
    Div,
    Exp2,
    Lse2,
    MaskedDiv,
    MaskedExp,
    MaskedLse2,
    MaskedScore,
    Matmul,
    ReduceMax,
    ReduceSum,
)
from cleave.ir.rolling import rolling_nodes

DTYPE, ACCUM_DTYPE = "bfloat16", "float32"
BATCH, KV, D, GROUP = 2, 256, 128, 4
LOG2E = 1.4426950408889634
DENOM_FLOOR = 1e-30
F32 = dict(dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE)


def _connect(src, dst, src_id, dst_id):
    edge = Edge(src, dst, src_id, dst_id)
    src._out_edges.append(edge)
    dst.set_inputs(dst_id, edge)


def _accum_edge(loop, dst, src_id, dst_id, accum):
    edge = AccumEdge(loop, dst, src_id, dst_id, accum_node=accum)
    loop._out_edges.append(edge)
    dst.set_inputs(dst_id, edge)


def build_paged_tier(causal: bool = False) -> SimpleNamespace:
    """The scheduled kernel graph and its nodes. ``causal``: the prefill shape."""
    # decode: 16 staged rows, 64 keys per tile; prefill: 16 tokens x GROUP rows, 16 keys
    rows, bn = (64, 16) if causal else (16, 64)
    tokens = rows // GROUP if causal else 1
    mask = ("tile_base", "valid_len", "diag") if causal else ("tile_base", "valid_len")
    masked = dict(row_group=GROUP) if causal else {}

    mma_0 = Matmul((rows, bn, D), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_0",
                   transpose_B=True, warp_policy="cols")
    score_cls, exp_cls = (CausalMaskedScore, CausalMaskedExp) if causal else (
        MaskedScore, MaskedExp)
    score = score_cls((rows, bn), scalar_args=(*mask, "scale2"), name="score_0",
                      n_batch_dims=0, **masked, **F32)
    row_max = ReduceMax((rows, bn), axis=-1, name="exp_0_max", accum_init=NEG_FLT_MAX, **F32)
    exp = exp_cls((rows, bn), scale=None, scalar_args=mask, shift=True, name="exp_0",
                  n_batch_dims=0, **masked, **F32)
    max_prev, max_step, alpha = rolling_nodes(
        row_max, stem="exp_0", rows=(rows,), n_batch_dims=0, exp_cls=Exp2)
    rowsum = ReduceSum((rows, bn), axis=-1, name="rowsum_0", **F32)
    mma_1 = Matmul((rows, D, bn), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_1",
                   warp_policy="cols")
    mma_1.stage_fragment_operands_in_shared = (0,)
    _connect(mma_0, score, 0, 0)
    _connect(score, row_max, 0, 0)
    _connect(score, exp, 0, 0)
    _connect(row_max, exp, 0, 1)
    _connect(exp, rowsum, 0, 0)
    _connect(exp, mma_1, 0, 0)
    ph_q, ph_k = (e.src_node for e in mma_0.inputs)
    ph_v = mma_1.inputs[1].src_node
    ph_q.name, ph_k.name, ph_v.name = "PlaceHolder_Q", "PlaceHolder_K", "PlaceHolder_V"

    inner = Graph(nodes=gather_all_nodes(rowsum, mma_1))
    # Declared before the GraphNode exists: alpha has no dataflow consumer and must not be
    # taken for an accumulator.
    inner.lag_nodes = [max_prev]
    inner.accum_scale = {rowsum: alpha, mma_1: alpha}
    inner.extent = 1  # a serial loop with a runtime trip count never reads it
    inner.loop_kind = "serial"
    inner.loop_trip = f"T.ceildiv(valid_len, {bn})"
    scalars = [("valid_len", "int32"), ("scale2", "float32")]
    scalars += [("diag", "int32")] if causal else []
    inner.sargs = list(scalars)
    inner.iter_sargs = [("tile_base", "int32")] + scalars
    inner.smap = {name: (lambda v: lambda i: v)(name) for name, _ in scalars}
    inner.smap["tile_base"] = lambda i: f"{bn} * {i}"
    lock_graph(inner)

    loop = GraphNode(name="GraphNode_0", graph=inner)
    rowsum_id, value_id = inner.accum_nodes.index(rowsum), inner.accum_nodes.index(mma_1)
    # The max is read inside the loop, so it is carried by declaration; appended last.
    max_id = loop.carry(row_max)
    inner_q, inner_k, inner_v = (inner.binding[p.name] for p in (ph_q, ph_k, ph_v))

    out_rows = rows if causal else GROUP
    if causal:
        div = MaskedDiv((rows, D), broadcast=0, denom_floor=DENOM_FLOOR, row_group=GROUP,
                        scalar_args=("n_rows",), dtype=ACCUM_DTYPE, out_dtype=DTYPE,
                        name="div_0", n_batch_dims=0)
        lse = MaskedLse2((rows,), row_group=GROUP, scalar_args=("n_rows",), name="lse_0",
                         n_batch_dims=0, **F32)
    else:
        div = Div((GROUP, D), broadcast=0, denom_floor=DENOM_FLOOR, dtype=ACCUM_DTYPE,
                  out_dtype=DTYPE, name="div_0", n_batch_dims=0)
        lse = Lse2((GROUP,), name="lse_0", n_batch_dims=0, **F32)
    _accum_edge(loop, div, value_id, 0, mma_1)
    _accum_edge(loop, div, rowsum_id, 1, rowsum)
    _accum_edge(loop, lse, rowsum_id, 0, rowsum)
    _accum_edge(loop, lse, max_id, 1, row_max)
    out_o, out_lse = OutputNode(div, dtype=DTYPE), OutputNode(lse, dtype=ACCUM_DTYPE)
    out_o.name, out_lse.name = "OutputNode_0", "OutputNode_1"

    main = Graph(nodes=[ph_q, ph_k, ph_v] + gather_all_nodes(loop))
    main.n_threads = 128
    main.scalar_params = [("kv_len", "int32"), ("sm_scale", "float32")]
    main.rasterization = 1
    main.extent = (BATCH,)
    main.sargs = list(scalars) + ([("n_rows", "int32")] if causal else [])
    main.smap = {
        "valid_len": lambda b: "kv_len",
        "scale2": lambda b: f"sm_scale * {LOG2E}",
        # the last query token of the block may attend to every key
        "diag": lambda b: f"kv_len - {tokens}",
        "n_rows": lambda b: f"{tokens}",
    }
    lock_graph(main)

    def block(shape, tile):
        return Iterator(lambda b: (f"{b}",) + ("0",) * (len(shape) - 1), shape=shape,
                        tile_size=tile)

    def whole(shape):
        return Iterator(lambda i: ("0",) * len(shape), shape=shape, tile_size=shape)

    main.tmap[ph_q] = block((BATCH, rows, D), (1, rows, D))
    main.tmap[ph_k] = block((BATCH, KV, D), (1, KV, D))
    main.tmap[ph_v] = block((BATCH, KV, D), (1, KV, D))
    main.tmap[loop] = [whole((rows,)), whole((rows, D)), whole((rows,))]
    if causal:  # the destination keeps (token, head) apart; the source rows fuse them
        out_tile = block((BATCH, tokens, GROUP, D), (1, tokens, GROUP, D))
        lse_tile = block((BATCH, tokens, GROUP), (1, tokens, GROUP))
    else:
        out_tile = block((BATCH, out_rows, D), (1, out_rows, D))
        lse_tile = block((BATCH, out_rows), (1, out_rows))
    main.tmap[div] = main.tmap[out_o] = out_tile
    main.tmap[lse] = main.tmap[out_lse] = lse_tile

    inner.tmap[inner_q] = whole((rows, D))
    for node in (inner_k, inner_v):
        inner.tmap[node] = Iterator(
            lambda i: (f"{bn} * {i}", "0"), shape=(KV, D), tile_size=(bn, D))
    for node in (mma_0, score, exp):
        inner.tmap[node] = whole((rows, bn))
    for node in (row_max, max_prev, max_step, alpha, rowsum):
        inner.tmap[node] = whole((rows,))
    inner.tmap[mma_1] = whole((rows, D))

    main.tinfo_map = {
        ph_q: (ph_q.name, "g"), ph_k: (ph_k.name, "g"), ph_v: (ph_v.name, "g"),
        loop: ("t_GraphNode_0", "r"),
        div: ("OutputNode_0", "g"), out_o: ("OutputNode_0", "g"),
        lse: ("OutputNode_1", "g"), out_lse: ("OutputNode_1", "g"),
    }
    inner.tinfo_map = {
        ph_q: (ph_q.name, "g"), ph_k: (ph_k.name, "g"), ph_v: (ph_v.name, "g"),
        inner_q: (inner_q.name, "s"), inner_k: (inner_k.name, "s"), inner_v: (inner_v.name, "s"),
        mma_0: ("t_mma_0", "r"),
        score: ("t_score_0", "r"),
        row_max: ("t_exp_0_max", "r"),
        max_prev: ("t_exp_0_max_prev", "r"),
        max_step: ("t_exp_0_max_step", "r"),
        alpha: ("t_exp_0_alpha", "r"),
        exp: ("t_exp_0", "r"),
        rowsum: ("t_rowsum_0", "r"),
        mma_1: ("t_mma_1", "r"),
    }
    main = passes.annotate_args(main, is_main=True)
    main = passes.annotate_tc_layout(main)
    main = passes.annotate_fragment_layout_conflicts(main)
    return SimpleNamespace(
        main=main, inner=inner, loop=loop, score=score, row_max=row_max, exp=exp,
        max_prev=max_prev, max_step=max_step, alpha=alpha, rowsum=rowsum, mma_0=mma_0,
        mma_1=mma_1, div=div, lse=lse, rows=rows, bn=bn,
    )


def paged_tier_code(causal: bool = False) -> tuple:
    """``(tier, generated source)`` of the hand-built paged-shaped kernel."""
    t = build_paged_tier(causal)
    return t, TilelangCodeGen().gen(t.main)
