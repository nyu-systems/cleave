"""The streaming accept filter: a tiling is kept iff the build schedules it (CPU).

For a graph with a streaming obstacle, served or not, the policy proposes tilings the build
has to refuse: where the obstacle is served it drops the pin and steps the max with its
sums (a tiling the assembly cannot serve is refused); where it is not, it still steps the
max on its own while the sums are pinned, and the scores would be needed per iteration and
whole at once. ``streaming_accept`` drops those before the top-k cut, by running the build
itself, and ``pipeline.plans`` installs it for every lowered graph that has a record or a
list of reasons (``streaming_refusals``).
"""

from collections import Counter

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.lang as cl
import cleave.schedule
import cleave.schedule.streaming as stream
from cleave import pipeline as P
from cleave.core.errors import ScheduleRejected
from cleave.frontend import trace
from cleave.ir.update_rule import use_searched_updates
from cleave.pipeline import plans
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import search_tiling
from cleave.schedule.tiling import schedule_kernel

from tests import _explicit_softmax as X
from tests._graphs import mha_bhsd_graph

SMALL = dict(H=32, SQ=64, SKV=256, D=64)
TWO_PASSES = "two passes"


@pytest.fixture(scope="module")
def traced():
    """Traced-order attention at S_KV=256: the obstacle is refused (T3); only the tilings
    that hold the row whole build."""
    return X.streamed_ir(X.traced_attention, SMALL, TWO_PASSES)


@pytest.fixture(scope="module")
def flash():
    """The flash form at S_KV=256 with its record."""
    return X.streamed_ir(X.flash_attention, SMALL)


def _kinds(tiles):
    return Counter("looped" if stream.sites(schedule_kernel(t)) else "one tile" for t in tiles)


def _scheduled(monkeypatch, fn, shapes, ir, **config):
    """``pipeline.schedule`` of the identity graph of ``fn``, lowered to ``ir``."""
    graphs = GraphSet.identity(trace(fn, shapes, cl.f16))
    monkeypatch.setattr(plans, "lower", lambda graph, split_factor=None: [ir])
    return P.schedule(graphs, P.ScheduleConfig(**config))


def test_no_filter_without_an_obstacle(monkeypatch):
    """No record and no list of reasons: no filter, the search is as it was."""
    plain = X.lowered(X.flash_attention, SMALL)
    assert stream.streaming_accept(plain) is None
    assert stream.streaming_accept(mha_bhsd_graph(1, 32, 4096, 4096, 64)) is None
    # an obstacle that is served, or refused, installs it; an empty list of reasons alone
    # (an obstacle-free graph's) does not
    assert stream.streaming_accept(X.streamed_ir(X.flash_attention, SMALL)) is not None
    plain.streaming_refusals = []
    assert stream.streaming_accept(plain) is None

    seen = []

    def spy(graph, **kw):
        seen.append(kw)
        return search_tiling(graph, **kw)

    monkeypatch.setattr(cleave.schedule, "search_tiling", spy)
    del plain.streaming_refusals
    assert len(plans._search(plain, 8, False, None)) == 8
    assert seen == [dict(topk=8, force_fusion=False)]  # no accept: the search is as it was


def test_the_filter_is_the_build(traced):
    accept = stream.streaming_accept(traced)
    raw = search_tiling(traced, topk=1000)
    kept = [cand for cand in raw if accept(cand)]
    assert (len(raw), len(kept)) == (32, 8)
    for cand in raw:
        try:
            build_candidate_code(cand)  # anything but a refusal propagates
            built = True
        except ScheduleRejected:
            built = False
        assert built == any(cand is k for k in kept)
    assert dict(accept.reasons) == {
        "div_4 is computed per iteration of GraphNode_reduce_max_2_5 but also read by sub_6 "
        "outside it": 24
    }
    again = search_tiling(traced, topk=1000, accept=stream.streaming_accept(traced))
    assert [plans.describe_tile(t) for t in again] == [plans.describe_tile(t) for t in kept]


def test_a_served_obstacle_keeps_every_tiling_that_builds(flash):
    accept = stream.streaming_accept(flash)
    raw = search_tiling(flash, topk=1000)
    kept = [cand for cand in raw if accept(cand)]
    assert len(raw) == len(kept) == 32 and not accept.reasons
    assert _kinds(kept) == {"looped": 24, "one tile": 8}


def test_topk_is_cut_after_the_filter(traced):
    # The policy's own first four are all tilings that step the max on its own.
    for cand in search_tiling(traced, topk=4):
        with pytest.raises(ScheduleRejected):
            schedule_kernel(cand)
    tiles = plans._search(traced, 4, False, None)
    assert len(tiles) == 4 and _kinds(tiles) == {"one tile": 4}


def test_a_user_accept_hint_still_applies(traced):
    asked = []

    def user(cand):
        asked.append(cand)
        return cand.n_threads == 256

    refused = Counter()
    tiles = plans._search(traced, 1000, False, {"accept": user}, refused)
    assert len(tiles) == 4 and {t.n_threads for t in tiles} == {256}
    assert _kinds(tiles) == {"one tile": 4}
    assert len(asked) == 32
    # only what the user let through was tiled: 16 candidates, 12 of them refused
    assert sum(refused.values()) == 12


def test_schedule_keeps_the_served_tilings(flash, monkeypatch):
    planset = _scheduled(monkeypatch, X.flash_attention, SMALL, flash, topk=4)
    assert len(planset) == 4 and not planset.refused
    for plan in planset:
        code = build_candidate_code(plan.tile)
        assert "ukernel_reduce_max" in code


def test_a_bug_inside_the_filter_is_a_refused_graph_that_names_it(flash, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(stream, "_carry", broken)
    with pytest.raises(ScheduleRejected, match="^RuntimeError: boom$"):
        plans._search(flash, 4, False, None)
    planset = _scheduled(monkeypatch, X.flash_attention, SMALL, flash, topk=4, fusion="off")
    assert len(planset) == 0
    assert [r.reason for r in planset.refused] == ["ScheduleRejected: RuntimeError: boom"]
    assert "1x ScheduleRejected: RuntimeError: boom" in planset.summary()


@pytest.mark.parametrize("searched", [True, False], ids=["searched", "switch off"])
def test_no_candidate_left_says_why(monkeypatch, searched):
    """A returned softmax whose row is longer than any block: its obstacle is refused (T4),
    and the report names that reason after the row limit, then the tilings the filter
    dropped. The text does not depend on the switch: the list of reasons is the graph's."""
    shapes = dict(R=64, C=16384)
    ir = X.streamed_ir(X.softmax_rows, shapes, "is also read by")
    (why,) = ir.streaming_refusals
    with use_searched_updates(searched):
        planset = _scheduled(monkeypatch, X.softmax_rows, shapes, ir, topk=16, fusion="off")
    assert len(planset) == 0
    (refusal,) = planset.refused
    assert isinstance(refusal.error, ScheduleRejected)
    text = str(refusal.error)
    head = ("a max needs its whole row in one block, and this row is longer; it is not "
            f"carried across a loop: {why} (")
    assert text.startswith(head) and text.endswith(")"), text
    counted = text[len(head):-1]
    assert counted.split(" refused: ")[0].isdigit() and "x " in counted, counted
    assert planset.summary().splitlines()[1].startswith(
        "  refused: 1x ScheduleRejected: a max needs its whole row in one block")


def test_reason_key_folds_bare_numbers_only():
    key = stream._reason_key
    assert key(ScheduleRejected(
        "reduce_max_2_5: the dim it reduces is spread over grid blocks (128 of 4096 per "
        "block); a value")) == ("reduce_max_2_5: the dim it reduces is spread over grid "
                                "blocks (N of N per block); a value")
    assert key(ScheduleRejected(
        "GraphNode_reduction_0_4 runs 1024 iterations but dim 0 of exp_2 is cut into 32 "
        "tiles (128 of 4096)"
    )) == ("GraphNode_reduction_0_4 runs N iterations but dim N of exp_2 is cut into N "
           "tiles (N of N)")
    assert key(ScheduleRejected("reduce_max_2_5: its tile (1,) holds a single row\nmore")) == (
        "reduce_max_2_5: its tile (N,) holds a single row")
    # the hand-derived pass reads the same function
    import cleave.schedule.stabilize as stab

    assert stab._reason_key is key and stab._weighted_slices is stream._weighted_slices
