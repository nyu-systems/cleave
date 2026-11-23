"""Which overlaps of a max and the reductions under it are streaming obstacles, and every
refusal text of the classification and of the step maps (CPU).

``schedule.streaming.obstacles`` classifies a flat graph once, from what the policy already
has: for a running max over a dim, the reductions that loop under it over that dim (the
state proposal) and, where it cannot be streamed, why (T1-T5). A classified obstacle is
served only with a record derived for the very graph (R1). Given a record, a step map in
which the carried max does not step with its loop is a typed refusal (C1, C2), both in
``apply_tiling`` and in the twin the policy rates its step maps with; without a record the
same maps get what the tiler always said. A graph left without tilings says why its max is
not carried.
"""

import copy
import dataclasses
from types import SimpleNamespace

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import ScheduleRejected
from cleave.frontend import trace
from cleave.ir.ops import Div, Matmul, ReduceMax, ReduceSum
from cleave.ir.update_rule import Obstacle, use_searched_updates
from cleave.pipeline import plans as plans_module
from cleave.pipeline.superopt import GraphSet
from cleave.schedule import search_tiling
from cleave.schedule import streaming as stream
from cleave.schedule.tiling import (
    absorbed_axis_conflict,
    apply_tiling,
    carried_reductions,
    reduction_ancestors,
    schedule_kernel,
    scheduled_reductions,
)

from tests import _explicit_softmax as X

SMALL = dict(H=32, SQ=64, SKV=256, D=64)


def _only(ir) -> Obstacle:
    (obstacle,) = X.obstacles_of(ir)
    return obstacle


def _names(ir, cls, nth=0):
    return [n.name for n in ir.nodes if type(n) is cls][nth]


# ---------------------------------------------------------------- the classification, T1-T5


def test_the_flash_form_is_a_streaming_obstacle():
    """The state proposal: the max, and the row sum and the value matmul that loop under
    it; the division behind them is not state."""
    ir = X.lowered(X.flash_attention, SMALL)
    assert _only(ir) == Obstacle(_names(ir, ReduceMax), (_names(ir, Matmul, 1),
                                                           _names(ir, ReduceSum)), "")
    # an obstacle that is classified is not served without a record
    nodes, red = scheduled_reductions(ir)
    assert stream.streamed(nodes, red, reduction_ancestors(nodes), ir) == {}


class _Stub:
    """A reduction as the classifier sees it (hashed by identity, as graph nodes are)."""

    def __init__(self, name, raxes, **kw):
        self.name, self.outputs = name, []
        self.raxis = tuple(SimpleNamespace(dom=SimpleNamespace(extent=8)) for _ in range(raxes))
        self.__dict__.update(kw)


_stub = _Stub


def test_t1_a_max_over_two_axes():
    """No reduction lowering emits reduces two axes (``ReduceSum`` takes one); the rule is
    held on the classifier's own inputs."""
    full = _stub("m2", 2, running_max=True)
    d = _stub("s", 1)
    red = {full: [{(1, 0, 1)}, {(1, 0, 2)}], d: [{(1, 0, 1)}]}
    found = stream.obstacles([full, d], red, {d: frozenset({full}), full: frozenset()})
    assert found == [Obstacle("m2", ("s",), "m2 reduces 2 axes; one is carried")]


def max_of_a_sum(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]):
    """The max of ``x + y`` over a row shifts the exp of ``x`` alone: the sum runs over the
    dim of ``x`` only, the max over the dims of both."""
    m = cl.max(x + y, -1, keepdim=True)
    return cl.sum(cl.exp(x - m), -1, keepdim=True)


def test_t2_a_loop_that_does_not_run_over_the_whole_dim_of_the_max():
    ir = X.lowered(max_of_a_sum, dict(R=64, C=4096))
    m, s = _names(ir, ReduceMax), _names(ir, ReduceSum)
    assert _only(ir).refused == (
        f"{s} does not reduce the whole dim {m} reduces along one axis of the same extent; "
        "one loop counter cannot step both")


def test_t3_the_traced_order():
    ir = X.lowered(X.traced_attention, SMALL)
    m, total, reader = _names(ir, ReduceMax), _names(ir, ReduceSum), _names(ir, Matmul, 1)
    assert _only(ir).refused == (
        f"{reader} reads {total}, a finished reduction over the dim {m} reduces: that needs "
        "two passes over the row")


def test_t4_a_returned_softmax():
    ir = X.lowered(X.softmax_rows, dict(R=64, C=1024))
    m = _names(ir, ReduceMax)
    exp, div = X.exps(ir)[0].name, _names(ir, Div)
    assert _only(ir).refused == (
        f"{exp}, computed from {m} for every element of the looped dim, is also read by "
        f"{div}, which is not summed over that dim: that needs two passes over the row")


def test_t4_allows_the_max_itself_to_be_returned():
    ir = X.lowered(X.flash_and_its_max, SMALL)
    assert not _only(ir).refused


def test_t5_a_max_of_the_shifted_exp():
    """``(e @ v) / max(e)``: the second max loops under the first, and starts at ``-inf``;
    no repair of a sum describes it."""
    ir = X.lowered(X.max_normalized, dict(R=256, C=4096, D=64))
    outer, inner = (n.name for n in X.maxes(ir))
    obstacle = _only(ir)
    assert obstacle.full == outer and inner in obstacle.loops
    assert obstacle.refused == (
        f"{inner} is a ReduceMax: the reductions that loop under {outer} must be sums or "
        "matmuls that start at zero")


def test_t5_a_sum_that_does_not_start_at_zero():
    ir = X.lowered(X.flash_attention, SMALL)
    rowsum = next(n for n in ir.nodes if type(n) is ReduceSum)
    rowsum.accum_init = 1.0
    assert _only(ir).refused == (
        f"{rowsum.name} is a ReduceSum: the reductions that loop under {_names(ir, ReduceMax)} "
        "must be sums or matmuls that start at zero")


def test_t5_comes_last():
    """The rules are asked in order: a graph that fails an earlier one is refused for it."""
    ir = X.lowered(X.traced_attention, SMALL)
    next(n for n in ir.nodes if type(n) is ReduceSum).accum_init = 1.0
    assert "two passes over the row" in _only(ir).refused


def test_a_refused_obstacle_keeps_its_pin_whatever_record_it_has():
    """A record on a max whose obstacle is refused is not served: T5 is decided before
    the record is looked at, so the pin of the descendants stays."""
    ir = X.lowered(X.max_normalized, dict(R=256, C=4096, D=64))
    X.record(ir, _only(ir))
    nodes, red = scheduled_reductions(ir)
    assert stream.streamed(nodes, red, reduction_ancestors(nodes), ir) == {}
    assert carried_reductions(ir) == {}


# ------------------------------------------------------------------------------------- R1


def test_r1_a_record_of_another_graph_keeps_the_pin():
    ir = X.streamed_ir(X.flash_attention, SMALL)
    (m,) = X.maxes(ir)
    m.streaming = dataclasses.replace(m.streaming, key="0" * 64)
    ir.streaming_refusals = []
    nodes, red = scheduled_reductions(ir)
    assert stream.streamed(nodes, red, reduction_ancestors(nodes), ir) == {}
    assert ir.streaming_refusals == [f"{m.name}: its update rule was derived for another graph"]


# ------------------------------------------------------------- out-of-step step maps, C1/C2


def _with_steps(cand, **steps):
    """``cand`` with the reduction steps of the named nodes replaced: a step map the
    policy does not emit, built by hand."""
    stepped = copy.copy(cand)
    stepped.rstep_map = {node: {axis: steps.get(node.name, step) for axis, step in own.items()}
                         for node, own in cand.rstep_map.items()}
    return stepped


def _verdicts(ir, cand):
    """``(what apply_tiling says, what the policy's twin of it says)``: the refusal text,
    or None where it is built / has no conflict."""
    nodes, red = scheduled_reductions(ir)
    twin = absorbed_axis_conflict(nodes, cand.rstep_map, red, reduction_ancestors(nodes),
                                  carried_reductions(ir))
    try:
        apply_tiling(cand)
    except ScheduleRejected as refusal:
        return str(refusal), twin
    return None, twin


@pytest.fixture(scope="module")
def flash():
    """The flash graph with and without its record, and one looped candidate whose step map
    the cases rewrite (its nodes are those of the graph it is run on)."""
    recorded = X.streamed_ir(X.flash_attention, SMALL)
    plain = X.lowered(X.flash_attention, SMALL)
    base = next(c for c in search_tiling(recorded, topk=1000)
                if stream.sites(schedule_kernel(c)))
    names = SimpleNamespace(m=_names(recorded, ReduceMax), l=_names(recorded, ReduceSum),
                            o=_names(recorded, Matmul, 1))

    def on(ir):
        cand = copy.copy(base)
        by_name = {n.name: n for n in ir.nodes}
        cand.rstep_map = {by_name[n.name]: dict(v) for n, v in base.rstep_map.items()}
        cand.flat_graph = ir
        return cand

    return SimpleNamespace(recorded=recorded, plain=plain, on=on, **vars(names))


def _carried_by(m, loop, step, own):
    return (f"{m} is carried by the loop of {loop}, which steps {step}, but is itself stepped "
            f"{own}: one loop counter advances both")


def _needed_whole(m, reader, own):
    return (f"{m} is stepped {own} but {reader}, which reads its finished value, is not: the "
            "value would be needed whole and per iteration at once")


ABSORBED = ("Reduction axis of {} is shared with an absorbed ancestor reduction; one loop "
            "cannot both advance it and hold it whole.")
OUTSIDE = "{} is computed per iteration of GraphNode_{} but also read by {} outside it"

# label -> (steps of (max, row sum, value matmul),
#           (with the record: apply_tiling, twin), (without: apply_tiling, twin))
MAPS = {
    "in step": ((32, 32, 32), lambda f: (None, None),
                lambda f: (ABSORBED.format(f.o),) * 2),
    "the max whole, the sums stepped": (
        (256, 32, 32), lambda f: (_carried_by(f.m, f.o, 32, 256),) * 2,
        lambda f: (None, None)),
    "the max stepped 64, the sums 32": (
        (64, 32, 32), lambda f: (_carried_by(f.m, f.o, 32, 64),) * 2,
        lambda f: (ABSORBED.format(f.o),) * 2),
    "only the row sum stepped": (
        (256, 32, 256), lambda f: (_carried_by(f.m, f.l, 32, 256),) * 2,
        lambda f: (OUTSIDE.format("exp_7", f.l, f.o), None)),
    "only the value matmul stepped": (
        (256, 256, 32), lambda f: (_carried_by(f.m, f.o, 32, 256),) * 2,
        lambda f: (OUTSIDE.format("exp_7", f.o, f.l), None)),
    "the max stepped alone": (
        (32, 256, 256), lambda f: (_needed_whole(f.m, f.o, 32),) * 2,
        lambda f: (OUTSIDE.format("div_4", f.m, "sub_6"), None)),
}


@pytest.mark.parametrize("label", list(MAPS))
def test_out_of_step_maps(flash, label):
    """The four hand-built step maps of the design (C1 three times, C2 once; the row-sum
    case in both orders), and the map that is served, on the flash graph: with the record,
    the exact refusal in both places; without it, what the tiler said before there were
    records (the max whole under stepped sums is then a two-pass kernel, and builds)."""
    (m, l, o), with_record, without = MAPS[label]
    steps = {flash.m: m, flash.l: l, flash.o: o}
    assert _verdicts(flash.recorded, _with_steps(flash.on(flash.recorded), **steps)) \
        == with_record(flash), label
    assert _verdicts(flash.plain, _with_steps(flash.on(flash.plain), **steps)) \
        == without(flash), label
    # the flat graphs were not touched by any of the refused candidates
    assert not hasattr(flash.plain, "streaming_refusals")
    assert flash.recorded.streaming_refusals == []


# -------------------------------------------------------------------- the no-tiling report


def _schedule_ir(monkeypatch, fn, shapes, ir, **config):
    """``pipeline.schedule`` of the identity graph of ``fn``, lowered to ``ir``."""
    graphs = GraphSet.identity(trace(fn, shapes, cl.f16))
    monkeypatch.setattr(plans_module, "lower", lambda graph, split_factor=None: [ir])
    return P.schedule(graphs, P.ScheduleConfig(**config))


def test_a_graph_without_tilings_says_why_its_max_is_not_carried(monkeypatch):
    """The traced order at a row longer than a block: no tiling, and the report names the
    obstacle's reason, after the row limit."""
    shapes = dict(H=32, SQ=64, SKV=4096, D=64)
    ir = X.streamed_ir(X.traced_attention, shapes, "two passes")
    (why,) = ir.streaming_refusals
    plans = _schedule_ir(monkeypatch, X.traced_attention, shapes, ir, topk=8)
    assert not plans and {str(r.error).split(" (")[0] for r in plans.refused} == {
        "a max needs its whole row in one block, and this row is longer; it is not carried "
        f"across a loop: {why}"}
    assert why.startswith(f"{X.maxes(ir)[0].name}: mma_") and "a finished reduction" in why


@pytest.mark.parametrize("searched", [True, False], ids=["searched", "switch off"])
def test_a_max_without_an_obstacle_entry_and_without_tilings(monkeypatch, searched):
    """No list of reasons on the graph (no obstacle was looked at): the row limit alone.
    With the switch off it is the text of the hand-derived path, whose advice is about
    ``cl.softmax``; where update rules are searched a max is a max, whoever wrote it, and
    the advice goes. (Which graph has no tiling is the policy's business; here it has
    none.)"""
    ir = X.lowered(X.flash_attention, SMALL)
    assert not hasattr(ir, "streaming_refusals")
    monkeypatch.setattr(plans_module, "_search", lambda *args, **kwargs: [])
    with use_searched_updates(searched):
        plans = _schedule_ir(monkeypatch, X.flash_attention, SMALL, ir, topk=8)
    assert not plans and plans.refused
    texts = {str(r.error) for r in plans.refused}
    if searched:
        assert texts == {"a max needs its whole row in one block, and this row is longer"}
        assert plans_module.ROW_LIMIT in texts
    else:
        assert texts == {plans_module.MAX_ROW_LIMIT}
