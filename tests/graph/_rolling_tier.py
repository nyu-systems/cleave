"""A hand-built loop tier that declares a rolling softmax through the ``Graph`` fields only.

Built the way the paged builders work -- ops at tile dims, ``Graph(...)`` first,
``GraphNode(...)`` after -- with no ``apply_tiling`` and no rewrite pass: the loop carries a
running max, lags it, and rescales a ``ReduceSum`` accumulator (GQA style) and an
``Accumulate`` accumulator (MLA style) by the same factor.

    per iteration (a ``[ROWS, STEP]`` slice of the ``[ROWS, KV]`` scores):
        m_prev = m                      lag node
        m      = max(m, rowmax(s))      carried, starts at NEG_FLT_MAX
        P      = exp(s - m)
        alpha  = exp(m_prev - m)
        l      = l * alpha + rowsum(P)  ReduceSum accumulator
        A      = A * alpha[row] + P     Accumulate accumulator
    after the loop:  out = A / l[row]
"""

from __future__ import annotations

from types import SimpleNamespace

from cleave.codegen import TilelangCodeGen, passes
from cleave.ir.graph import Graph, GraphNode, Iterator, OutputNode
from cleave.ir.graph_utils import find_topo_sort, lock_graph
from cleave.ir.ops import NEG_FLT_MAX, Accumulate, Div, Exp, ReduceMax, ReduceSum, Sub, wire_input
from cleave.ir.rolling import rolling_nodes

ROWS, KV, STEP, TOTAL = 16, 256, 64, 64
F32 = dict(dtype="float32", out_dtype="float32")


def build_rolling_tier(declare: bool = True, carry: bool = True, max_axis: int = -1) -> SimpleNamespace:
    """The loop and its reader, wired but not scheduled.

    ``declare=False`` leaves ``lag_nodes`` / ``accum_scale`` empty (what the tier looks like
    to code that does not know the fields); ``carry=False`` stops before ``loop.carry(m)``.
    ``max_axis`` is how the max names its axis: ``-1`` (as lowering does) or ``1``, the
    same axis from the front.
    """
    m = ReduceMax((ROWS, KV), axis=max_axis, name="exp_0_max", accum_init=NEG_FLT_MAX, **F32)
    scores = m.inputs[0].src_node  # the tier's input tile
    m_prev, step, alpha = rolling_nodes(m, stem="exp_0", rows=(ROWS,), n_batch_dims=0, exp_cls=Exp)
    shifted = Sub((ROWS, KV), broadcast=0, name="exp_0_shifted", n_batch_dims=0, **F32)
    wire_input(scores, shifted, 0)
    wire_input(m, shifted, 1)
    p = Exp((ROWS, KV), name="exp_0", n_batch_dims=0, **F32)
    wire_input(shifted, p, 0)
    rowsum = ReduceSum((ROWS, KV), axis=-1, name="rowsum_0", **F32)
    wire_input(p, rowsum, 0)
    psum = Accumulate((ROWS, STEP), name="psum_0", **F32)
    wire_input(p, psum, 0)
    # the outer tier reads both accumulators
    div = Div((TOTAL, STEP), broadcast=0, name="div_0", n_batch_dims=0, **F32)
    wire_input(psum, div, 0)
    wire_input(rowsum, div, 1)
    out = OutputNode(div, dtype="float32")

    inner = Graph(nodes=[scores, m, m_prev, step, alpha, shifted, p, rowsum, psum])
    if declare:
        # Declared BEFORE the GraphNode exists, so that alpha (no dataflow consumer) is
        # not mistaken for an accumulator.
        inner.lag_nodes = [m_prev]
        inner.accum_scale = {rowsum: alpha, psum: alpha}
    loop = GraphNode(name="GraphNode_0", graph=inner)
    if carry:
        loop.carry(m)
    return SimpleNamespace(
        loop=loop, inner=inner, outer_in=loop.inputs[0].src_node, m=m, m_prev=m_prev, step=step,
        alpha=alpha, shifted=shifted, p=p, rowsum=rowsum, psum=psum, div=div, out=out,
    )


def schedule_rolling_tier(t: SimpleNamespace) -> Graph:
    """Lock the tier of :func:`build_rolling_tier`, give every node its tile, annotate."""
    inner = t.inner
    lock_graph(inner)
    main = Graph(nodes=find_topo_sort([t.out]))
    main.n_threads = 128
    lock_graph(main)
    main.extent = (TOTAL // ROWS,)
    main.tmap[t.outer_in] = Iterator(
        lambda i: (f"{ROWS} * {i}", "0"), shape=(TOTAL, KV), tile_size=(ROWS, KV))
    for node in (t.div, t.out):
        main.tmap[node] = Iterator(
            lambda i: (f"{ROWS} * {i}", "0"), shape=(TOTAL, STEP), tile_size=(ROWS, STEP))
    inner.extent = KV // STEP
    inner_in = inner.binding[t.outer_in.name]
    inner.tmap[inner_in] = Iterator(
        lambda i: ("0", f"{STEP} * {i}"), shape=(ROWS, KV), tile_size=(ROWS, STEP))
    for node in (t.shifted, t.p, t.psum):
        inner.tmap[node] = Iterator(
            lambda i: ("0", "0"), shape=(ROWS, STEP), tile_size=(ROWS, STEP))
    for node in (t.m, t.m_prev, t.step, t.alpha, t.rowsum):
        inner.tmap[node] = Iterator(lambda i: ("0",), shape=(ROWS,), tile_size=(ROWS,))
    main = passes.annotate_args(main, is_main=True)
    main = passes.annotate_tinfo(main, is_main=True)
    t.main = main
    return main


def rolling_tier_code() -> tuple:
    """``(tier, generated source)`` of the scheduled hand-built tier."""
    t = build_rolling_tier()
    return t, TilelangCodeGen().gen(schedule_rolling_tier(t))
