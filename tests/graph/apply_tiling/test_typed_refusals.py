"""``apply_tiling`` refuses a tiling it cannot build with ``ScheduleRejected`` (CPU).

A value computed inside a loop and also read outside it: the loop's inner placeholders
leaked into the main graph and the candidate died with ``KeyError: 'PlaceHolder_...'`` at
the main ``tmap`` fill. It is the traced-order softmax, ``exp / sum(exp)`` followed by the
value matmul, whenever the row sum is looped: ``exp`` goes into the loop with the sum and
is still read by the ``div`` outside it.

It is a per-candidate refusal now: callers that drop a candidate on ``ScheduleRejected``
(the pipeline's schedule stage, the softmax accept filter) can tell it from a bug.
"""

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.lang as cl
from cleave.core.errors import ScheduleRejected
from cleave.frontend import trace
from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops import Div, Exp, ReduceSum
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet
from cleave.schedule import apply_tiling, search_tiling


def traced_order_attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                           v: cl.Tensor["H", "SKV", "D"]):
    """softmax(QK^T) @ V with the softmax written out: the exp feeds the sum AND the div."""
    e = cl.exp((q @ k.mT) / q.shape[-1] ** 0.5)
    return (e / cl.sum(e, -1, keepdim=True)) @ v


def _identity_graph(fn, shapes):
    (graph,) = list(GraphSet.identity(trace(fn, shapes, cl.f16)))
    (ir,) = lower(graph)
    return ir


def _one(ir, cls, where=lambda n: True):
    (node,) = [n for n in ir.nodes if type(n) is cls and where(n)]
    return node


def _step(cand, node):
    """The reduction step ``cand`` gives the node named like ``node``."""
    (rstep,) = [v for n, v in cand.rstep_map.items() if n.name == node.name]
    (step,) = rstep.values()
    return int(step)


def _outcomes(cands):
    built, refused = [], []
    for cand in cands:
        try:
            built.append((cand, apply_tiling(cand)))
        except ScheduleRejected as e:
            refused.append((cand, str(e)))
    return built, refused


def test_value_read_inside_and_outside_a_loop_is_a_typed_refusal():
    ir = _identity_graph(traced_order_attention, dict(H=32, SQ=64, SKV=256, D=64))
    exp, rowsum = _one(ir, Exp), _one(ir, ReduceSum)
    div = _one(ir, Div, lambda n: n.const is None)
    cands = search_tiling(ir, topk=1000)

    # Anything but ScheduleRejected propagates out of here: KeyError was the defect.
    built, refused = _outcomes(cands)

    looped = [c for c in cands if _step(c, rowsum) < 256]
    assert looped and len(looped) < len(cands), "the case needs both kinds of tiling"
    # Exactly the tilings that loop the row sum are refused, and for this reason.
    assert [c for c, _ in refused] == looped
    assert {text for _, text in refused} == {
        f"{exp.name} is computed per iteration of GraphNode_{rowsum.name} but also read "
        f"by {div.name} outside it"
    }
    # The others hold the whole row in one tile: no loop contains the exp.
    assert len(built) == len(cands) - len(looped)
    for _, tiled in built:
        for tier, owner in graph_tiers(tiled):
            assert owner is None or exp.name not in {n.name for n in tier.nodes}
