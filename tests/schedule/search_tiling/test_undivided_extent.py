"""A reduction whose extent the tensor-core step does not divide (CPU).

The tensor-core policy used to start a matmul's reduction at 16 or at the tensor-core step
whatever its extent was (``TCPolicy._assign_reduce_step``, "for the case of padding"), and
the walk over the iterative tilings only knows the steps that divide the extent. Nothing
pads the last tile of a reduction, so such a graph has no tiling. The policy used to find
that out with ``ValueError: 32 is not in list`` from a list lookup.

It is a refusal with its reason now, in two places:

* ``TCPolicy._assign_reduce_step`` no longer starts outside the axis's own steps. A matmul
  of 16-bit operands over an extent that no multiple of 16 divides is refused there
  (TileLang's gemm takes such operands in K blocks of a multiple of 16; see
  ``tests/graph/apply_tiling/test_typed_refusals.py``), and a wider operand starts at a
  step it has;
* the walk itself (``expand_iterative_tiles``) refuses a start that is none of the steps it
  knows, instead of dying on the lookup, whoever hands it one.

The attention at S_KV = 1000 is the case that showed it: the value matmul contracts over the
keys, in float16, and 1000 = 8 x 125. It is the same for the exact softmax, the fixed shift
and a plain ``exp``, so the exact softmax still has no plan at such a length. That needs a
padded last tile, which is not built here.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import ScheduleRejected
from cleave.frontend import trace
from cleave.pipeline.superopt import GraphSet
from cleave.schedule import search_tiling
from cleave.schedule.policy.default import TCPolicy

from tests import _softmax as S

NO_MULTIPLE_OF_16 = (
    r"mma_\d+ contracts 1000 float16 elements: a tensor-core matmul of 16-bit operands "
    r"steps its reduction axis in multiples of 16, and none divides that extent"
)
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
    # the value matmul, over the keys: the refusal names the node, the extent and the cause
    assert re.fullmatch(NO_MULTIPLE_OF_16, reason), reason
    assert "ValueError" not in reason and "not in list" not in reason
    assert reason in plans.summary()


def test_the_policy_itself_raises_the_refusal():
    ir = S.lowered(S.flash_attention, _shapes(1000))
    for force in (False, True):
        with pytest.raises(ScheduleRejected, match=NO_MULTIPLE_OF_16):
            list(search_tiling(ir, topk=16, force_fusion=force))


@pytest.mark.parametrize("skv", [96, 1024, 1536])
def test_extents_the_step_divides_schedule_as_before(skv):
    plans = P.schedule(GraphSet.identity(trace(S.flash_attention, _shapes(skv), cl.f16)),
                       P.ScheduleConfig(topk=16))
    assert len(plans) >= 1
    assert not any(NOT_PADDED in str(r.error) or "none divides that extent" in str(r.error)
                   for r in plans.refused)


def test_the_walk_refuses_a_start_that_is_none_of_its_steps(monkeypatch):
    """``expand_iterative_tiles`` walks the steps that divide a reduction's extent, from the
    step ``_assign_reduce_step`` starts at. The tensor-core policy starts at one of them
    today, so a program no longer reaches this; a start from anywhere else is still a
    refusal that names the reduction, its extent and the step, not a ``ValueError`` from the
    lookup. Here a matmul over 1024 elements is started at 24."""
    from tests._graphs import matmul_graph

    real = TCPolicy._assign_reduce_step
    started = []

    def start_outside_the_steps(self, node):
        steps = real(self, node)
        if node.get_tag("tensorcore_config"):
            started.append(node.name)
            return {name: 24 for name in steps}
        return steps

    monkeypatch.setattr(TCPolicy, "_assign_reduce_step", start_outside_the_steps)
    with pytest.raises(ScheduleRejected) as refusal:
        list(search_tiling(matmul_graph(128, 128, 1024), topk=16))
    assert started == ["mma_0"]
    assert str(refusal.value) == (
        "mma_0: its reduction over 1024 elements starts at step 24, which does not "
        f"divide it; {NOT_PADDED}")
