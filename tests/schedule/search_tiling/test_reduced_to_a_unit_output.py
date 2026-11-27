"""An output that reduces every dim of its operand has no tile: a refusal with its reason (CPU).

``sum(x, 0)`` of a vector, or ``sum(sum(x, -1), -2)`` of a matrix, writes one element. Its
tile has no dim, and the policy's memory-traffic estimate used to die on it: every such
graph was refused with ``IndexError: list index out of range`` (``coalesced_factor`` of an
empty tile), a refusal that names no node. Unit dims that are not batch dims have no tile
anywhere in the scheduler (``refuse_unit_extents``); this is the same limit, said where the
policy meets it. A row reduced to a unit dim next to a dim that is kept (``sum(x, -1)`` of
a matrix with one row) still has a tile, and tilings.
"""

import pytest

pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import ScheduleRejected
from cleave.frontend import trace
from cleave.pipeline.superopt import GraphSet

NO_DIM = ("every dim of the output is reduced to one element, so its tile has no dim; a unit "
          "dim that is not a batch dim has no tile")


def vector_sum(x: cl.Tensor["C"]):
    return cl.sum(x, 0, keepdim=True)


def total(x: cl.Tensor["R", "C"]):
    return cl.sum(cl.sum(x, -1, keepdim=True), -2, keepdim=True)


def ratio_of_totals(x: cl.Tensor["R", "C"], w: cl.Tensor["R", "C"]):
    e = cl.exp(x - cl.max(x, -1, keepdim=True))
    return (cl.sum(cl.sum(e * w, -1, keepdim=True), -2, keepdim=True)
            / cl.sum(cl.sum(e, -1, keepdim=True), -2, keepdim=True))


def row_sum(x: cl.Tensor["R", "C"]):
    return cl.sum(x, -1, keepdim=True)


@pytest.mark.parametrize("fn, sizes, node", [
    (vector_sum, dict(C=256), "reduction_0_"),
    (total, dict(R=64, C=256), "reduction_0_"),
    (ratio_of_totals, dict(R=64, C=256), "div_"),
], ids=["vector_sum", "total", "ratio_of_totals"])
def test_an_output_without_a_dim_is_refused_with_the_reason(fn, sizes, node):
    plans = P.schedule(GraphSet.identity(trace(fn, sizes, cl.f16)), P.ScheduleConfig(topk=4))
    assert not plans and plans.refused
    for refusal in plans.refused:
        assert isinstance(refusal.error, ScheduleRejected)
        text = str(refusal.error)
        assert text.startswith(node) and text.endswith(f": {NO_DIM}"), text


def test_a_row_reduced_next_to_a_kept_dim_still_has_tilings():
    plans = P.schedule(GraphSet.identity(trace(row_sum, dict(R=1, C=256), cl.f16)),
                       P.ScheduleConfig(topk=4))
    assert len(plans) >= 1, plans.summary()
