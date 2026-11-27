"""The streaming accept filter: a tiling is kept iff the build schedules it (CPU).

For a graph with a streaming obstacle, served or not, the policy proposes tilings the build
has to refuse: where the obstacle is served it drops the pin and steps the max with its
sums (a tiling the assembly cannot serve is refused), next to the tilings it would make
without the record; where it is not, it still steps the max on its own while the sums are
pinned, and the scores would be needed per iteration and whole at once. ``streaming_accept`` drops those before the top-k cut, by running the build
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
from cleave.pipeline import plans
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.compile import build_candidate_code
from cleave.schedule.policy.default import REGISTER_OPERAND_CONFLICT
from cleave.schedule import search_tiling
from cleave.schedule.tiling import schedule_kernel

from tests import _explicit_softmax as X
from tests._graphs import mha_bhsd_graph

SMALL = dict(H=32, SQ=64, SKV=256, D=64)


@pytest.fixture(scope="module")
def traced():
    """Traced-order attention at S_KV=256: both conflicts are refused (no update of the
    accumulator form reproduces the flat graph); only the tilings that hold the row whole
    build."""
    return X.streamed_ir(X.traced_attention, SMALL, X.NO_RULE, X.NO_RULE)


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
    # (64 query rows have no 256-thread tiling: the value matmul would read 8 rows a warp
    # of the exp's registers, TCPolicy._register_operand_layout_conflict)
    assert (len(raw), len(kept)) == (16, 4)
    for cand in raw:
        try:
            build_candidate_code(cand)  # anything but a refusal propagates
            built = True
        except ScheduleRejected:
            built = False
        assert built == any(cand is k for k in kept)
    assert dict(accept.reasons) == {
        "div_4 is computed per iteration of GraphNode_reduce_max_2_5 but also read by sub_6 "
        "outside it": 12
    }
    again = search_tiling(traced, topk=1000, accept=stream.streaming_accept(traced))
    assert [plans.describe_tile(t) for t in again] == [plans.describe_tile(t) for t in kept]


def test_a_served_obstacle_keeps_every_tiling_that_builds(flash):
    accept = stream.streaming_accept(flash)
    raw = search_tiling(flash, topk=1000)
    kept = [cand for cand in raw if accept(cand)]
    assert len(raw) == len(kept) == 16 and not accept.reasons
    assert _kinds(kept) == {"looped": 12, "one tile": 4}


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
        return cand.output_tile[-1] == 64

    refused = Counter()
    tiles = plans._search(traced, 1000, False, {"accept": user}, refused)
    assert len(tiles) == 2 and {t.output_tile[-1] for t in tiles} == {64}
    assert _kinds(tiles) == {"one tile": 2}
    assert len(asked) == 16
    # only what the user let through was tiled: 8 candidates, 6 of them refused
    assert sum(refused.values()) == 6


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


UNIT_EXTENT = "a unit extent that is not a batch dim"
UNIT_DIM = "a unit dim that is not a batch dim has no tile"


def _no_tiling(planset, head):
    """The one refusal of a graph without tilings starts with ``head`` and counts the
    tilings the filter dropped; returns the count part."""
    assert len(planset) == 0
    (refusal,) = planset.refused
    assert isinstance(refusal.error, ScheduleRejected)
    text = str(refusal.error)
    assert text.startswith(head) and text.endswith(")"), text
    counted = text[len(head):-1]
    assert counted.split(" refused: ")[0].isdigit() and "x " in counted, counted
    assert planset.summary().splitlines()[1].startswith(
        f"  refused: 1x ScheduleRejected: {head[:40]}")
    return counted


def test_no_candidate_left_says_why(monkeypatch):
    """A returned softmax at 16384 columns: its max has a record, but every tiling that
    loops the row is refused (the exp is read by the division, outside the loop), so the row
    has to be in one block, and the only tilings that hold it there have one row, which no
    kernel has (a unit extent). That is the reason, not the row limit: the report names the
    tilings the filter dropped, and why."""
    shapes = dict(R=64, C=16384)
    ir = X.streamed_ir(X.softmax_rows, shapes)
    assert not ir.streaming_refusals
    planset = _scheduled(monkeypatch, X.softmax_rows, shapes, ir, topk=16, fusion="off")
    counted = _no_tiling(planset, f"{plans.NO_TILING} (")
    assert UNIT_EXTENT in counted and plans.ROW_LIMIT not in str(planset.refused[0].error)
    assert "is computed per iteration of GraphNode_" in counted, counted


def two_outputs(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                v: cl.Tensor["H", "SKV", "D"]):
    """Attention in traced order next to the plain ``exp(s) @ v`` of its scores."""
    s = (q @ k.mT) / q.shape[-1] ** 0.5
    e = X.stable_exp(s)
    return (e / cl.sum(e, -1, keepdim=True)) @ v, cl.exp(s) @ v


def test_the_row_limit_where_every_tiling_the_filter_dropped_steps_the_max(monkeypatch):
    """The traced order next to ``exp(s) @ v`` at S_KV=256: the policy steps the max with
    the second matmul in every tiling it makes (32, 64, 128 of 256) and none holds the row
    in one block, so the row limit is the reason, then why the max is not carried (no update
    reproduces the flat graph), then what the filter dropped."""
    ir = X.streamed_ir(two_outputs, dict(H=2, SQ=64, SKV=256, D=64), X.NO_RULE, X.NO_RULE)
    why = "; ".join(ir.streaming_refusals)
    planset = _scheduled(monkeypatch, two_outputs, dict(H=2, SQ=64, SKV=256, D=64), ir,
                         topk=16, fusion="off")
    counted = _no_tiling(planset, f"{plans.ROW_LIMIT}; it is not carried across a loop: {why} (")
    assert "is computed per iteration of GraphNode_mma_" in counted, counted


@pytest.mark.parametrize("fn, shapes, refused, reason", [
    # served: the max is carried, it needs no row in one block; every tiling would have a
    # unit dim, and before that a value matmul reading one row of the exp's registers
    (X.flash_attention, dict(H=2, SQ=1, SKV=256, D=64), (), REGISTER_OPERAND_CONFLICT),
    # served, and the row of 64 fits; the one tiling that holds it has one row, and every
    # one that loops it is refused (its exp is read outside the loop)
    (X.softmax_rows, dict(R=1, C=64), (), UNIT_DIM),
], ids=["carried_max_one_query", "returned_softmax_one_row"])
def test_a_row_that_fits_is_not_called_longer(monkeypatch, fn, shapes, refused, reason):
    """Every tiling was refused for another reason than the row: the report gives those
    reasons, after why a max is not carried where one is not, and never the row limit."""
    ir = X.streamed_ir(fn, shapes, *refused)
    why = "; ".join(ir.streaming_refusals)
    head = f"{plans.NO_TILING}; a reduction is not carried across a loop: {why} (" if why else \
        f"{plans.NO_TILING} ("
    planset = _scheduled(monkeypatch, fn, shapes, ir, topk=16, fusion="off")
    counted = _no_tiling(planset, head)
    assert reason in counted and plans.ROW_LIMIT not in str(planset.refused[0].error)


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
