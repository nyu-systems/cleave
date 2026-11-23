"""Softmax programs written with their max, and the streaming records the assembly tests use (CPU).

The searched path keeps the stable softmax in the flat graph as it is written,
``exp(x - max(x)) / sum(...)``: the max is an ordinary reduction. The programs here write
it out (``cl.max``, ``cl.exp``, ``cl.sum``), so they trace and lower to the same graph
whatever ``CLEAVE_SEARCHED_UPDATES`` says, and they are lowered as identity graphs with the
switch OFF: nothing is derived for them.

:func:`streamed_ir` then classifies every streaming obstacle of the graph
(``schedule.streaming.obstacles``, what ``pipeline.plans`` asks), CHECKS it against what
the test expects, and does what lowering does after the search: a classified obstacle gets
a ``Streaming`` record keyed by the graph itself (the rule is the one the engine proves for
every program here, ``acc * exp(m - m')``), a refused one its reason in
``ir.streaming_refusals``. The assembly is then held to the record; whether the record is
right is the engine's business (``tests/update``).
"""

from __future__ import annotations

import cleave.lang as cl
from cleave.frontend import trace
from cleave.ir.ops import Exp, ReduceMax
from cleave.ir.update_rule import Streaming, loop_key, use_searched_updates
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet

FACTOR = ("exp", ("sub", "m", "m'"))  # acc *= exp(m - m'), then the accumulator's own macro


def stable_exp(x, dim: int = -1):
    """``exp(x - max(x, dim))``: what ``cl.softmax`` records where update rules are searched."""
    return cl.exp(x - cl.max(x, dim, keepdim=True))


# ------------------------------------------------------------------------------- programs


def flash_attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                    v: cl.Tensor["H", "SKV", "D"]):
    """Flash form: the row sum and the value matmul are siblings, the division comes last."""
    e = stable_exp((q @ k.mT) / q.shape[-1] ** 0.5)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def traced_attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                     v: cl.Tensor["H", "SKV", "D"]):
    """Traced order: normalize, then the value matmul, which reads the finished row sum."""
    e = stable_exp((q @ k.mT) / q.shape[-1] ** 0.5)
    return (e / cl.sum(e, -1, keepdim=True)) @ v


def unscaled_attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                       v: cl.Tensor["H", "SKV", "D"]):
    """Traced order without the 1/sqrt(d): the max and the exp read the QK^T matmul itself."""
    e = stable_exp(q @ k.mT)
    return (e / cl.sum(e, -1, keepdim=True)) @ v


def flash_and_its_max(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                      v: cl.Tensor["H", "SKV", "D"]):
    """The flash form that also returns its row max (what a split partial hands on)."""
    s = (q @ k.mT) / q.shape[-1] ** 0.5
    m = cl.max(s, -1, keepdim=True)
    e = cl.exp(s - m)
    return (e @ v) / cl.sum(e, -1, keepdim=True), m


def softmax_rows(x: cl.Tensor["R", "C"]):
    """A softmax that is returned itself."""
    e = stable_exp(x)
    return e / cl.sum(e, -1, keepdim=True)


def softmax_mid(x: cl.Tensor["H", "C", "R"]):
    """Softmax over the middle axis of a batched input."""
    e = stable_exp(x, 1)
    return e / cl.sum(e, 1, keepdim=True)


def flash_rows(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """Flash form whose logits are an input: the max and the subtraction read storage."""
    e = stable_exp(x)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def softmax_over_queries(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                         v: cl.Tensor["H", "SKV", "D"]):
    """Attention normalized over the queries (``dim=-2``): every key's column sums to 1."""
    e = stable_exp((q @ k.mT) / q.shape[-1] ** 0.5, -2)
    return (e / cl.sum(e, -2, keepdim=True)) @ v


def column_average(x: cl.Tensor["H", "K", "N"], y: cl.Tensor["H", "K", "N"]):
    """``sum_k(softmax_k(x) * y)``: the softmax axis is the FIRST non-batch axis; both sums
    reduce it, so their accumulators lead with the remaining axis ``N``."""
    e = stable_exp(x, 1)
    return cl.sum(e * y, 1, keepdim=True) / cl.sum(e, 1, keepdim=True)


def column_average_2d(x: cl.Tensor["K", "N"], y: cl.Tensor["K", "N"]):
    """:func:`column_average` without the batch dim."""
    e = stable_exp(x, 0)
    return cl.sum(e * y, 0, keepdim=True) / cl.sum(e, 0, keepdim=True)


def two_softmaxes(a: cl.Tensor["R", "C"], b: cl.Tensor["R", "C"]):
    """``sum(pa * pb) / (sum(pa) * sum(pb))``: one sum is computed from both maxes."""
    pa, pb = stable_exp(a), stable_exp(b)
    s = cl.sum(pa * pb, -1, keepdim=True)
    return s / (cl.sum(pa, -1, keepdim=True) * cl.sum(pb, -1, keepdim=True))


def max_normalized(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """``(e @ v) / max(e)``: a max of the shifted exp loops under the max of the logits."""
    e = stable_exp(x)
    return (e @ v) / cl.max(e, -1, keepdim=True)


def column_weighted(v: cl.Tensor["M", "K"], x: cl.Tensor["K", "N"]):
    """``(v @ e) / sum(e, 0)``: the max is indexed by the accumulator's TRAILING axis."""
    e = stable_exp(x, 0)
    return (v @ e) / cl.sum(e, 0, keepdim=True)


def self_weighted(x: cl.Tensor["K", "N"]):
    """``(x^T @ e) / sum(e, 0)``: :func:`column_weighted` with ``v = x^T``. One input dim,
    the columns of ``x``, is both the accumulator's rows and the max's: the factor still
    belongs to the accumulator's TRAILING axis."""
    e = stable_exp(x, 0)
    return (x.mT @ e) / cl.sum(e, 0, keepdim=True)


def sum_next_to_exp(x: cl.Tensor["R", "C"]):
    """``sum(e + l) / l``: the exp is added to its own finished row sum."""
    e = stable_exp(x)
    l = cl.sum(e, -1, keepdim=True)
    return cl.sum(e + l, -1, keepdim=True) / l


# ----------------------------------------------------------------------------- lowering


def lowered(fn, shapes, dtype=cl.f16):
    """The identity (un-searched) graph of ``fn``, lowered to IR with the switch off."""
    with use_searched_updates(False):
        (graph,) = list(GraphSet.identity(trace(fn, shapes, dtype)))
        (ir,) = lower(graph)
    return ir


def exps(ir) -> list:
    return [n for n in ir.nodes if type(n) is Exp]


def maxes(ir) -> list:
    return [n for n in ir.nodes if type(n) is ReduceMax]


def obstacles_of(ir) -> list:
    """``schedule.streaming.obstacles`` of ``ir``, as ``pipeline.plans`` asks them."""
    from cleave.schedule.streaming import obstacles
    from cleave.schedule.tiling import reduction_ancestors, scheduled_reductions

    nodes, red = scheduled_reductions(ir)
    return obstacles(nodes, red, reduction_ancestors(nodes))


def record(ir, obstacle, factor=FACTOR):
    """Put a record for ``obstacle`` on its max: ``factor`` for every accumulator, keyed by
    ``ir`` itself. Returns ``(the max, the record)``."""
    full = next(n for n in ir.nodes if n.name == obstacle.full)
    full.streaming = Streaming(obstacle.loops, tuple((a, factor) for a in obstacle.loops),
                               loop_key(ir, obstacle.full, obstacle.loops))
    return full, full.streaming


def streamed_ir(fn, shapes, *refused, dtype=cl.f16, factor=FACTOR):
    """Lower ``fn`` and check its obstacles against ``refused``: one entry per obstacle in
    graph order, ``None`` for one that is classified as streaming (it gets a record) or a
    phrase of the reason it is not (it goes to ``ir.streaming_refusals``, as lowering
    writes it). No entry at all: one obstacle, classified."""
    ir = lowered(fn, shapes, dtype)
    found = obstacles_of(ir)
    refused = refused or (None,) * len(found)
    assert len(refused) == len(found), f"{len(found)} obstacles, {len(refused)} expectations"
    reasons = []
    for obstacle, want in zip(found, refused):
        if want is None:
            assert not obstacle.refused, (obstacle.full, obstacle.refused)
            record(ir, obstacle, factor)
        else:
            assert want in obstacle.refused, (obstacle.full, obstacle.refused, want)
            reasons.append(f"{obstacle.full}: {obstacle.refused}")
    if found:
        ir.streaming_refusals = reasons
    return ir
