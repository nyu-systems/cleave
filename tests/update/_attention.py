"""The attention program of the searched-update tests, and how they pick its plans.

``cl.softmax`` records the explicit max, ``exp(x - max(x)) / sum``.
"""

from __future__ import annotations

import cleave.lang as cl
from cleave import pipeline as P
from cleave.frontend import trace

MAX_OPS = 11  # demo/mha_prefill.py
FACTOR = ("exp", ("sub", "m", "m'"))  # acc *= exp(m - m'), then the accumulator's own macro


def attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
              v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


def mha(sq: int, skv: int):
    """``(program, searched graphs)`` of MHA attention, H=32, D=64, float16, at
    ``max_ops=11``."""
    program = trace(attention, dict(H=32, SQ=sq, SKV=skv, D=64), cl.f16)
    return program, P.superoptimize(program, P.SuperoptConfig(max_ops=MAX_OPS, cache=False))


def flash(graphs):
    """The graph whose row sum and value matmul are siblings (the divide comes last)."""
    (graph,) = [g for g in graphs if g.summary().endswith("sum>matmul>div")]
    return graph


def traced(graphs):
    """The graph in traced order: the value matmul reads the normalized exp."""
    (graph,) = [g for g in graphs if g.summary().endswith("sum>div>matmul")]
    return graph


def steps(tile) -> list:
    """The reduction steps of a tiling, in the order the policy lists them."""
    return [int(x) for v in tile.rstep_map.values() for x in v.values()]


def pick(plans, *, tile, threads: int, d_step: int, kv_step: int):
    """The one plan with that output tile and thread count whose score matmul steps
    ``d_step`` over ``D`` and whose every other reduction steps ``kv_step`` over the keys."""
    (plan,) = [p for p in plans if list(p.tile.output_tile) == list(tile)
               and p.n_threads == threads and steps(p.tile)[0] == d_step
               and set(steps(p.tile)[1:]) == {kv_step}]
    return plan


def flash_by_hand(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                  v: cl.Tensor["H", "SKV", "D"]):
    """The flash form with its max written out: row sum and value matmul are siblings."""
    s = (q @ k.mT) / q.shape[-1] ** 0.5
    e = cl.exp(s - cl.max(s, -1, keepdim=True))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def lowered(fn, shapes, dtype=cl.f16):
    """The identity (un-searched) graph of ``fn``, lowered to IR without the derivation
    ``plans.lower`` adds (no record, no refusal list): the tests put a record by hand
    (``record_by_hand``) where they want one."""
    from cleave.pipeline.plans import _lower_structure
    from cleave.pipeline.superopt import GraphSet

    (graph,) = list(GraphSet.identity(trace(fn, shapes, dtype)))
    (ir,) = _lower_structure(graph, graph.structure)
    return ir


def node_of(ir, cls, nth: int = 0):
    """The ``nth`` node of ``ir`` whose class is exactly ``cls``."""
    return [n for n in ir.nodes if type(n) is cls][nth]


def record_by_hand(ir, factor=FACTOR):
    """A ``Streaming`` record for the (one) streaming obstacle of ``ir``, written by hand
    with the same ``factor`` for every accumulator and the key of ``ir`` itself: what
    ``pipeline.plans.lower`` writes after the search, for the tests of the hooks that only
    read it. Returns ``(full-extent node, record)``; the record is on the node."""
    from cleave.ir.update_rule import Streaming, loop_key
    from cleave.schedule.streaming import obstacles
    from cleave.schedule.tiling import reduction_ancestors, scheduled_reductions

    nodes, red = scheduled_reductions(ir)
    (obstacle,) = obstacles(nodes, red, reduction_ancestors(nodes))
    assert not obstacle.refused, obstacle.refused
    full = next(n for n in ir.nodes if n.name == obstacle.full)
    full.streaming = Streaming(obstacle.loops, tuple((a, factor) for a in obstacle.loops),
                               loop_key(ir, obstacle.full, obstacle.loops))
    return full, full.streaming
