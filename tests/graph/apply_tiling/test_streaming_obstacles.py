"""Which reductions are conflicts, and every refusal text of the step maps (CPU).

``schedule.streaming.obstacles`` lists the conflicts of a flat graph once, from what the
policy already has: for a reduction, the reductions under it that reduce a dim it reduces
(the state proposal), whatever the ops. A conflict is served only with a record derived for
the very graph. Given a record, a step map in which the carried value does not step with
its loop is a typed refusal (C1, C2), both in ``apply_tiling`` and in the twin the policy
rates its step maps with; without a record the same maps get what the tiler always said. A
graph left without tilings says why its conflict is not carried.
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
from cleave.ir.ops import Matmul, ReduceMax, ReduceSum
from cleave.ir.update_rule import Obstacle
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


# -------------------------------------------------------------------- the conflicts


def test_the_flash_form_is_one_conflict():
    """The state proposal: the max, and the row sum and the value matmul that loop under
    it; the division behind them is not state."""
    ir = X.lowered(X.flash_attention, SMALL)
    assert _only(ir) == Obstacle(_names(ir, ReduceMax), (_names(ir, Matmul, 1),
                                                           _names(ir, ReduceSum)))
    # a conflict is not served without a record
    nodes, red = scheduled_reductions(ir)
    assert stream.streamed(nodes, red, reduction_ancestors(nodes), ir) == {}


class _Stub:
    """A reduction as ``obstacles`` sees it (hashed by identity, as graph nodes are)."""

    def __init__(self, name, raxes, **kw):
        self.name, self.outputs = name, []
        self.raxis = tuple(SimpleNamespace(dom=SimpleNamespace(extent=8)) for _ in range(raxes))
        self.__dict__.update(kw)


def test_any_reduction_a_descendant_over_its_dim_depends_on_is_a_conflict():
    """Whatever the op (a stub of no class at all), and over either axis of an ancestor
    that reduces two; a descendant over another dim is none."""
    full, d, other = _Stub("c", 2), _Stub("s", 1), _Stub("t", 1)
    for shared in ((1, 0, 1), (1, 0, 2)):
        red = {full: [{(1, 0, 1)}, {(1, 0, 2)}], d: [{shared}], other: [{(2, 0, 0)}]}
        ancestors = {d: frozenset({full}), other: frozenset({full}), full: frozenset()}
        assert stream.obstacles([full, d, other], red, ancestors) == [Obstacle("c", ("s",))]


def test_the_traced_order_is_two_conflicts():
    """The max, with the row sum and the value matmul under it, and the row sum, with the
    value matmul under it (it reads the finished row sum)."""
    ir = X.lowered(X.traced_attention, SMALL)
    m, total, reader = _names(ir, ReduceMax), _names(ir, ReduceSum), _names(ir, Matmul, 1)
    assert X.obstacles_of(ir) == [Obstacle(m, (total, reader)), Obstacle(total, (reader,))]


def test_a_returned_softmax_and_a_returned_max_are_conflicts_like_any_other():
    """What a loop could not carry (an exp read outside it) is the tiler's to refuse, per
    tiling; the conflict itself is listed."""
    for fn, shapes in ((X.softmax_rows, dict(R=64, C=1024)), (X.flash_and_its_max, SMALL)):
        ir = X.lowered(fn, shapes)
        assert _only(ir).full == _names(ir, ReduceMax)


# ------------------------------------------- a record derived for another graph is not served


def test_a_record_of_another_graph_keeps_the_pin():
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
    "the max and the row sum stepped, the value matmul whole": (
        (32, 32, 256), lambda f: (_needed_whole(f.m, f.o, 32),) * 2,
        lambda f: (ABSORBED.format(f.l),) * 2),
    # no loop of the record is stepped: the record is not in use (``records_in_use``)
    "the max stepped alone": (
        (32, 256, 256), lambda f: (OUTSIDE.format("div_4", f.m, "sub_6"), None),
        lambda f: (OUTSIDE.format("div_4", f.m, "sub_6"), None)),
}


@pytest.mark.parametrize("label", list(MAPS))
def test_out_of_step_maps(flash, label):
    """The hand-built step maps of the design (C1 three times, C2 once; the row-sum case in
    both orders), the map that is served, and the max stepped alone, on the flash graph:
    with the record, the exact refusal in both places; without it, what the tiler said
    before there were records (the max whole under stepped sums is then a two-pass kernel,
    and builds). A map that steps no loop of the record does not use it, and gets what it
    gets without it."""
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
    conflicts' reasons, after the row limit."""
    shapes = dict(H=32, SQ=64, SKV=4096, D=64)
    ir = X.streamed_ir(X.traced_attention, shapes, "no update of", "no update of")
    why, why_sum = ir.streaming_refusals
    plans = _schedule_ir(monkeypatch, X.traced_attention, shapes, ir, topk=8)
    assert not plans and {str(r.error).split(" (")[0] for r in plans.refused} == {
        f"{plans_module.ROW_LIMIT}; it is not carried across a loop: {why}; {why_sum}"}
    assert why.startswith(f"{X.maxes(ir)[0].name}: no update of mma_")


def test_a_conflict_without_an_obstacle_entry_and_without_tilings(monkeypatch):
    """No list of reasons on the graph (no obstacle was looked at): the row limit alone,
    for the ancestor of a conflict, whatever its op: the text gives no advice about
    ``cl.softmax``. (Which graph has no tiling is the policy's business; here it has
    none.)"""
    ir = X.lowered(X.flash_attention, SMALL)
    assert not hasattr(ir, "streaming_refusals")
    monkeypatch.setattr(plans_module, "_search", lambda *args, **kwargs: [])
    plans = _schedule_ir(monkeypatch, X.flash_attention, SMALL, ir, topk=8)
    assert not plans and plans.refused
    texts = {str(r.error) for r in plans.refused}
    assert texts == {"a reduction that another one over its dim depends on needs that whole "
                     "dim in one block, and this dim is longer"}
    assert texts == {plans_module.ROW_LIMIT}
