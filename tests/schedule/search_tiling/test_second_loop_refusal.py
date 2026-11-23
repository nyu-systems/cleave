"""A tiling that needs a second loop is refused by the schedule, not at codegen (CPU).

The 18-op merged attention-backward graph (per-output search: dQ from D = sum_D(dO * o),
dK from the naive delta = sum_SKV(P * dP)) needs delta complete before dK's dS; in either
accumulation context the loop runs over SKV or holds SKV in steps, so delta would need a
loop of its own. ``apply_tiling`` refused every candidate ("shared with an absorbed
ancestor reduction"), so all 40 plans failed in the build's codegen stage.
"""

import pytest

import cleave.lang as cl
from cleave.core.errors import ScheduleRejected
from cleave.frontend import Program
from cleave.pipeline import GraphSet, ScheduleConfig, schedule
from cleave.schedule import apply_tiling

C = 64**-0.5


def mixed_bwd(H=2, SQ=512, SKV=512, D=64):
    p = Program("mixed_bwd", dtype=cl.f16)
    p.dims(H=H, SQ=SQ, SKV=SKV, D=D)
    q, k = p.input("q", ("H", "SQ", "D")), p.input("k", ("H", "SKV", "D"))
    v, dO = p.input("v", ("H", "SKV", "D")), p.input("dO", ("H", "SQ", "D"))
    L = p.input("L", ("H", "SQ", 1), cl.f32)
    o = p.input("o", ("H", "SQ", "D"))
    P = cl.exp(q @ k.mT * C - L)
    dP = dO @ v.mT
    dS_q = P * (dP - cl.sum(dO * o, -1, keepdim=True))
    dS_k = P * (dP - cl.sum(P * dP, -1, keepdim=True))
    p.output((dS_q @ k) * C, "dQ")
    p.output((dS_k.mT @ q) * C, "dK")
    p.output(P.mT @ dO, "dV")
    return p


def test_a_tiling_needing_a_second_loop_is_refused():
    plans = schedule(GraphSet.identity(mixed_bwd()), ScheduleConfig(topk=1000))
    assert len(plans) == 0
    assert plans.refused
    for refusal in plans.refused:
        assert "every tiling needs a second loop" in refusal.reason, refusal.reason
        assert "absorbed ancestor reduction" in refusal.reason


def test_an_inter_output_reading_a_finished_loop_is_refused():
    # With SQ in one tile the inter dK/dV steps are whole (not looped) and delta has a
    # loop of its own, merged with its sibling dQ over SKV: the policy keeps these
    # tilings, and dK reads delta's finished sum. ``_join_inter_outputs`` crashed
    # ("'GraphNode' object has no attribute 'raxis'"); it now refuses them.
    plans = schedule(GraphSet.identity(mixed_bwd(SQ=256, SKV=256)), ScheduleConfig(topk=1000))
    assert len(plans)
    for plan in plans:
        with pytest.raises(ScheduleRejected, match="would need a second loop"):
            apply_tiling(plan.tile)
