"""A reduction whose extent the tensor-core step does not divide (CPU).

The tensor-core policy starts a matmul's reduction at the tensor-core step whatever its
extent is (``TCPolicy._assign_reduce_step``, "for the case of padding"), and the walk over
the iterative tilings only knows the steps that divide the extent. Nothing pads the last tile
of a reduction, so such a graph has no tiling. The policy used to find that out with
``ValueError: 32 is not in list`` from a list lookup; it is a refusal with its reason now.

The attention at S_KV = 1000 is the case that showed it: the value matmul contracts over the
keys. It is the same for the exact softmax, the fixed shift and a plain ``exp``, so the
exact softmax still has no plan at such a length. That needs a padded last tile, which is
not built here.
"""

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import ScheduleRejected
from cleave.frontend import trace
from cleave.pipeline.superopt import GraphSet
from cleave.schedule import search_tiling

from tests import _softmax as S

NOT_PADDED = "the last tile of a reduction is not padded"


def flash_fixed_shift(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                      v: cl.Tensor["H", "SKV", "D"]):
    """The flash form under a fixed shift: no request, no record."""
    e = cl.exp((q @ k.mT) / q.shape[-1] ** 0.5 + -6.25)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def _shapes(skv):
    return dict(H=32, SQ=64, SKV=skv, D=64)


@pytest.mark.parametrize("fn", [S.flash_attention, flash_fixed_shift],
                         ids=["exact", "fixed_shift"])
def test_an_extent_the_tensor_core_step_does_not_divide_is_refused_with_its_reason(fn):
    plans = P.schedule(GraphSet.identity(trace(fn, _shapes(1000), cl.f16)),
                       P.ScheduleConfig(topk=16))
    assert not plans and plans.refused
    reasons = {str(r.error) for r in plans.refused}
    assert len(reasons) == 1, reasons
    (reason,) = reasons
    assert "reduction over 1000 elements" in reason and "tensor-core step 32" in reason
    assert reason.endswith(NOT_PADDED)
    assert "ValueError" not in reason and "not in list" not in reason
    assert NOT_PADDED in plans.summary()


def test_the_policy_itself_raises_the_refusal():
    ir = S.lowered(S.flash_attention, _shapes(1000))
    for force in (False, True):
        with pytest.raises(ScheduleRejected, match=NOT_PADDED):
            list(search_tiling(ir, topk=16, force_fusion=force))


@pytest.mark.parametrize("skv", [96, 1024, 1536])
def test_extents_the_step_divides_schedule_as_before(skv):
    plans = P.schedule(GraphSet.identity(trace(S.flash_attention, _shapes(skv), cl.f16)),
                       P.ScheduleConfig(topk=16))
    assert len(plans) >= 1
    assert not any(NOT_PADDED in str(r.error) for r in plans.refused)
