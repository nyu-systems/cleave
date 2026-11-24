"""The softmax accept filter: a tiling is kept iff ``apply_tiling`` builds it (CPU).

Once the max is gone from the searched graph, the policy ranks tilings the stabilization
rewrite has to refuse above the ones it serves. ``stabilize_accept`` drops them before the
top-k cut, by running the build itself, and ``pipeline.plans`` installs it for every lowered
graph that carries a ``Stabilize`` record.
"""

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.schedule
import cleave.schedule.stabilize as stab
from cleave import pipeline as P
from cleave.core.errors import ScheduleRejected
from cleave.frontend import trace
from cleave.pipeline import plans
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling

import cleave.lang as cl
from tests import _softmax as S
from tests._graphs import mha_bhsd_graph

SMALL = dict(H=32, SQ=64, SKV=256, D=64)
NEXT_TO_SUM = "div reads the exp next to the finished reduction"


@pytest.fixture(scope="module")
def traced():
    """Traced-order attention at S_KV=256: only the tilings that hold the row whole build."""
    return S.softmax_ir(S.traced_attention, SMALL, dict(rolling=NEXT_TO_SUM))


def _cases(tiles):
    return [{site.case for site in stab.sites(apply_tiling(t))} for t in tiles]


def _scheduled(monkeypatch, fn, shapes, ir, **config):
    """``pipeline.schedule`` of the identity graph of ``fn``, lowered to ``ir``."""
    graphs = GraphSet.identity(trace(fn, shapes, cl.f16))
    monkeypatch.setattr(plans, "lower", lambda graph, split_factor=None: [ir])
    return P.schedule(graphs, P.ScheduleConfig(**config))


def test_no_filter_without_a_record(monkeypatch):
    unmarked = S.lowered(S.flash_attention, SMALL)
    for exp in S.exps(unmarked):
        exp.stabilize = None
    assert stab.stabilize_accept(unmarked) is None
    assert stab.stabilize_accept(mha_bhsd_graph(1, 32, 4096, 4096, 64)) is None

    seen = []

    def spy(graph, **kw):
        seen.append(kw)
        return search_tiling(graph, **kw)

    monkeypatch.setattr(cleave.schedule, "search_tiling", spy)
    assert len(plans._search(unmarked, 8, False, None)) == 8
    assert seen == [dict(topk=8, force_fusion=False)]  # no accept: the search is as it was


def test_the_filter_is_the_build(traced):
    accept = stab.stabilize_accept(traced)
    raw = search_tiling(traced, topk=1000)
    kept = [cand for cand in raw if accept(cand)]
    # (64 query rows have no 256-thread tiling: TCPolicy._register_operand_layout_conflict)
    assert (len(raw), len(kept)) == (16, 4)
    for cand in raw:
        try:
            build_candidate_code(cand)  # anything but a refusal propagates
            built = True
        except ScheduleRejected:
            built = False
        assert built == any(cand is k for k in kept)
    exp, rowsum = S.exps(traced)[0].name, "reduction_2_6"
    assert dict(accept.reasons) == {
        f"{exp} is computed per iteration of GraphNode_{rowsum} but also read by div_7 "
        "outside it": 12
    }
    again = search_tiling(traced, topk=1000, accept=stab.stabilize_accept(traced))
    assert [plans.describe_tile(t) for t in again] == [plans.describe_tile(t) for t in kept]


def test_topk_is_cut_after_the_filter(traced):
    # The policy's own first four are all tilings that loop the row sum.
    for cand in search_tiling(traced, topk=4):
        with pytest.raises(ScheduleRejected):
            apply_tiling(cand)
    tiles = plans._search(traced, 4, False, None)
    assert len(tiles) == 4 and _cases(tiles) == [{"in-tile"}] * 4


def test_a_user_accept_hint_still_applies(traced):
    asked = []

    def user(cand):
        asked.append(cand)
        return cand.output_tile[-1] == 64

    refused = plans.Counter()
    tiles = plans._search(traced, 1000, False, {"accept": user}, refused)
    assert len(tiles) == 2 and {t.output_tile[-1] for t in tiles} == {64}
    assert _cases(tiles) == [{"in-tile"}] * 2
    assert len(asked) == 16
    # only what the user let through was tiled: 8 candidates, 6 of them refused
    assert sum(refused.values()) == 6


def test_schedule_keeps_the_served_tilings(traced, monkeypatch):
    planset = _scheduled(monkeypatch, S.traced_attention, SMALL, traced, topk=4)
    assert len(planset) == 4 and not planset.refused
    assert _cases([p.tile for p in planset]) == [{"in-tile"}] * 4
    for plan in planset:
        assert "ukernel_reduce_max_axis2_rr_clear(" in build_candidate_code(plan.tile)


def test_a_bug_inside_the_filter_is_a_refused_graph_that_names_it(traced, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(stab, "_rewrite", broken)
    with pytest.raises(ScheduleRejected, match="^RuntimeError: boom$"):
        plans._search(traced, 4, False, None)
    planset = _scheduled(monkeypatch, S.traced_attention, SMALL, traced, topk=4, fusion="off")
    assert len(planset) == 0
    assert [r.reason for r in planset.refused] == ["ScheduleRejected: RuntimeError: boom"]
    assert "1x ScheduleRejected: RuntimeError: boom" in planset.summary()


def test_no_candidate_left_says_why(monkeypatch):
    shapes = dict(R=64, C=16384)  # a row longer than any block, and nothing to roll it into
    ir = S.softmax_ir(S.softmax_rows, shapes, dict(rolling=NEXT_TO_SUM))
    planset = _scheduled(monkeypatch, S.softmax_rows, shapes, ir, topk=16, fusion="off")
    assert len(planset) == 0
    (refusal,) = planset.refused
    assert isinstance(refusal.error, ScheduleRejected)
    assert str(refusal.error) == (
        "no tiling candidates (19 refused: "
        "18x exp_1 is computed per iteration of GraphNode_reduction_1_2 but also read by "
        "div_3 outside it; "
        "1x exp_1: its tile (N, N) holds a single row)"
    )
    assert planset.summary().splitlines()[1].startswith(
        "  refused: 1x ScheduleRejected: no tiling candidates (19 refused: 18x exp_1 is computed"
    )


def test_reason_key_folds_bare_numbers_only():
    key = stab._reason_key
    assert key(ScheduleRejected(
        "exp_5: the softmax axis is spread over grid blocks (128 of 4096 per block); a max"
    )) == "exp_5: the softmax axis is spread over grid blocks (N of N per block); a max"
    assert key(ScheduleRejected(
        "GraphNode_reduction_0_4 runs 1024 iterations but dim 0 of exp_2 is cut into 32 "
        "tiles (128 of 4096)"
    )) == ("GraphNode_reduction_0_4 runs N iterations but dim N of exp_2 is cut into N "
           "tiles (N of N)")
    assert key(ScheduleRejected("exp_2: softmax axis -2 is not the last axis\nmore")) == (
        "exp_2: softmax axis -2 is not the last axis")
