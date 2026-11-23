"""The searched update rules, end to end, on MHA attention.

``test_search_*`` run the superoptimizer, the update search and the tiling search on the
CPU; ``test_gpu_*`` validate a kernel (root conftest marks). The other tests need neither:
they put a record on an identity graph by hand and test the hooks that read it.

``cl.softmax`` records ``exp(x - max(x)) / sum``. The flat search returns
the flash form (the row sum and the value matmul are siblings) and the traced order. The
flash form's loop over the keys is a streaming obstacle: lowering derives an update rule
for each accumulator, the policy lets the loop step, and ``apply_tiling`` assembles it
with the max carried. The traced order keeps its pin, with the reason.
"""

import copy
import dataclasses
import functools
import re
from collections import Counter

import pytest

torch = pytest.importorskip("torch")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import ScheduleRejected
from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops import NEG_FLT_MAX, Div, Matmul, ReduceMax, ReduceSum
from cleave.ir.update_rule import loop_key, show
from cleave.pipeline import plans as plans_module
from cleave.runtime.compile import build_candidate
from cleave.schedule import streaming
from cleave.schedule.streaming import FOREIGN_RECORD, StreamingRefused, assemble, sites, streamed
from cleave.schedule.tiling import (
    absorbed_axis_conflict,
    absorbed_reductions,
    apply_tiling,
    carried_reductions,
    reduction_ancestors,
    schedule_kernel,
    scheduled_reductions,
)
from tests._tolerances import tolerance_for

from ._attention import FACTOR, flash, flash_by_hand, lowered, mha, node_of, pick, record_by_hand, steps
from ._attention import traced as traced_order

TOPK = 100000  # every tiling
# The tiling of the differential test and of the GPU test (design 13.1 d, e).
KERNEL = dict(tile=[1, 128, 64], threads=128, d_step=64, kv_step=32)


@functools.lru_cache(maxsize=None)
def scheduled(sq: int, skv: int):
    """``(graphs, plans)`` of MHA attention at ``max_ops=11``, every tiling."""
    _, graphs = mha(sq, skv)
    return graphs, P.schedule(graphs, P.ScheduleConfig(topk=TOPK))


def recorded_node(flat_graph):
    (node,) = [n for n in flat_graph.nodes if getattr(n, "streaming", None) is not None]
    return node


def loop_tier(tiled):
    (tier,) = [tier for tier, owner in graph_tiers(tiled) if owner is not None]
    return tier


# ------------------------------------------------------------------------------ the pipeline


def test_search_mha_prefill_is_streamed_in_the_flash_form():
    graphs, plans = scheduled(4096, 4096)
    assert sorted(g.summary() for g in graphs) == [
        "solo matmul>div>max>sub>exp>sum>div>matmul", "solo matmul>div>max>sub>exp>sum>matmul>div"]
    flash_graph, traced_graph = flash(graphs), traced_order(graphs)

    # ---- the flash form: a record on its max, derived for this very graph
    assert len(plans) == 68 and {p.graph_index for p in plans} == {flash_graph.index}
    (flat,) = {id(p.tile.flat_graph): p.tile.flat_graph for p in plans}.values()
    full = recorded_node(flat)
    record = full.streaming
    assert type(full) is ReduceMax
    row_sum, value_matmul = node_of(flat, ReduceSum), node_of(flat, Matmul, 1)
    assert set(record.loops) == {row_sum.name, value_matmul.name}
    assert record.rules == tuple((acc, FACTOR) for acc in record.loops)
    assert record.key == loop_key(flat, full.name, record.loops)
    assert flat.streaming_refusals == []
    (derivation,) = flat.derivations
    assert not derivation.refused and derivation.record() == record
    assert {acc: show(update) for acc, update in derivation.chosen.items()} == {
        acc: "(r + (a * exp((m - m'))))" for acc in record.loops}

    # ---- every one of the 68 candidates is built, with the max carried by the loop
    for plan in plans:
        tiled = schedule_kernel(plan.tile)
        (site,) = sites(tiled)
        assert site.full == full.name and site.extent == site.block == 4096
        assert 1 < site.step < 4096 and set(site.repaired) == set(record.loops)
        tier = loop_tier(tiled)
        assert site.tier == next(o.name for t, o in graph_tiers(tiled) if t is tier)
        carried = next(n for n in tier.nodes if n.name == full.name)
        assert any(carried is n for n in tier.accum_nodes) and carried.accum_init == NEG_FLT_MAX
        assert carried.streaming is None and carried.streamed == site  # consumed on the copy
        assert [n.name for n in tier.lag_nodes] == [f"{full.name}_prev"]
        assert {acc.name: factor.name for acc, factor in tier.accum_scale.items()} == {
            acc: f"{full.name}_upd1" for acc in record.loops}
    assert full.streaming is record and full.accum_init == -float("inf")  # the flat graph keeps it

    # ---- the traced order: the value matmul reads the finished row sum; the pin stays
    (refusal,) = plans.refused
    assert (refusal.graph_index, refusal.variant, refusal.role) == (traced_graph.index, "solo", "main")
    (ir,) = plans_module.lower(traced_graph)
    names = dict(full=node_of(ir, ReduceMax).name, total=node_of(ir, ReduceSum).name,
                 reader=node_of(ir, Matmul, 1).name)
    why = ("{full}: {reader} reads {total}, a finished reduction over the dim {full} reduces: "
           "that needs two passes over the row").format(**names)
    assert ir.streaming_refusals == [why] and not hasattr(ir, "derivations")
    assert not [n for n in ir.nodes if getattr(n, "streaming", None) is not None]
    text = str(refusal.error)
    assert text.startswith("a max needs its whole row in one block, and this row is longer; "
                           "it is not carried across a loop: " + why)


def test_search_a_short_row_has_one_tile_plans_as_well():
    """S_KV = 256 fits one block: next to the looped tilings of the flash form there are
    the ones that hold the row whole, where the flat graph's own max is right as it stands
    and nothing is assembled; the traced order is served by those alone."""
    graphs, plans = scheduled(64, 256)
    flash_graph, traced_graph = flash(graphs), traced_order(graphs)
    assert len(plans) == 40 and not plans.refused
    kinds = Counter()
    for plan in plans:
        tiled = schedule_kernel(plan.tile)
        looped = bool(sites(tiled))
        kinds[plan.graph_index, "looped" if looped else "one tile"] += 1
        if not looped and plan.graph_index == flash_graph.index:
            # the record is still on the max, and it is the main tier's: nothing to carry
            (full,) = [n for n in tiled.nodes if getattr(n, "streaming", None) is not None]
            assert not tiled.lag_nodes and not tiled.accum_scale
            assert full.accum_init == -float("inf")
    assert kinds == {(flash_graph.index, "looped"): 24, (flash_graph.index, "one tile"): 8,
                     (traced_graph.index, "one tile"): 8}
    # a one-tile kernel of either graph is generated (the validator has nothing to object to)
    for graph in (flash_graph, traced_graph):
        plan = next(p for p in plans if p.graph_index == graph.index
                    and not sites(schedule_kernel(p.tile)))
        code, _ = build_candidate(plan.tile)
        assert "ukernel_reduce_max" in code and "ukernel_rescale" not in code
        assert "_prev" not in code


def test_search_the_searched_kernel_carries_the_max_in_the_rolling_order():
    """For one tiling of MHA prefill the kernel assembled from the searched rule has the
    loop body of a rolling kernel: the lag copy before the max, the factor of the rule, the
    repair in front of each accumulator's own macro, and the max started at -FLT_MAX."""
    _, new_plans = scheduled(4096, 4096)
    new = pick(new_plans, **KERNEL)
    assert steps(new.tile) == [64, 32, 32, 32]
    new_code, _ = build_candidate(new.tile)
    body = new_code[new_code.index("def GraphNode_reduction_2_8_iter"):]
    order = ["ukernel_copy_1dims_rr(t_reduce_max_2_5, t_reduce_max_2_5_prev)",
             "ukernel_reduce_max_axis2_rr(t_div_4, t_reduce_max_2_5)",
             "(t_reduce_max_2_5_prev, t_reduce_max_2_5, t_reduce_max_2_5_upd0)",
             "ukernel_exp_1dims_rr(t_reduce_max_2_5_upd0, t_reduce_max_2_5_upd1)",
             "ukernel_rescale_1dims_rr(t_reduce_max_2_5_upd1, t_reduction_2_8)",
             "ukernel_reduction_axis2_rr(t_exp_7, t_reduction_2_8)",
             "ukernel_rescale_lead1_2dims_rr(t_reduce_max_2_5_upd1, t_mma_9)",
             "ukernel_mma_rsr(t_exp_7_cast,"]
    flat = re.sub(r"\s+", " ", body)
    places = [flat.index(statement) for statement in order]
    assert places == sorted(places)
    assert "T.fill(t_reduce_max_2_5, -3.4028234663852886e38)" in new_code
    assert "T.clear(t_reduction_2_8)" in new_code and "T.clear(t_mma_9)" in new_code


def test_gpu_the_searched_kernel_validates_on_every_adversarial_case(tmp_path, monkeypatch,
                                                                    request):
    """That kernel on a GPU, at the bound the exact attention declares (rtol 5e-3, the
    bound of ``demo/mha_prefill.py`` as well): normal inputs and the six adversarial
    softmax cases, which a graph whose exp reads ``x - max(x)`` is held to by structure."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    import json
    import os
    from pathlib import Path

    from cleave.runtime.inputs import ADVERSARIAL

    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path / "cache"))
    log = os.environ.get("CLEAVE_VALIDATION_LOG")  # the session's own, when it keeps one
    if not log:
        log = str(tmp_path / "validation.jsonl")
        monkeypatch.setenv("CLEAVE_VALIDATION_LOG", log)
    tolerance = tolerance_for("exact_attention")
    assert tolerance.rtol == 5e-3 and tolerance.atol == 0
    _, plans = scheduled(4096, 4096)
    plan = pick(plans, **KERNEL)
    built = P.build(plans.filter(lambda p: p is plan),
                    config=P.EvalConfig(warmup=2, iters=20, max_parallel=1, tolerance=tolerance))
    (result,) = built.results
    assert result.ok, (result.stage, str(result.error)[:2000])
    assert result.cases == ("normal",) + ADVERSARIAL and len(result.cases) == 7
    assert not result.cases_note
    records = [json.loads(line) for line in Path(log).read_text().splitlines()]
    mine = [r for r in records if request.node.name in r["context"]]
    passed = Counter(r["what"] for r in mine if r["kind"] is None)
    assert passed == {"": 1, **{f"{case} inputs": 1 for case in ADVERSARIAL}}, mine
    assert all(o["rel"] <= tolerance.rtol for r in mine for o in r["outputs"])


# ------------------------------------------------- a record is honoured on its own graph only

SHORT = dict(H=32, SQ=64, SKV=256, D=64)


def served(ir):
    nodes, red = scheduled_reductions(ir)
    return {full.name: [n.name for n in loops]
            for full, loops in streamed(nodes, red, reduction_ancestors(nodes), ir).items()}


def tilings(ir):
    """``(tilings kept, reasons of the others)`` as the pipeline searches ``ir``."""
    reasons = Counter()
    return plans_module._search(ir, TOPK, False, None, reasons), reasons


def test_a_graph_without_a_record_is_scheduled_as_before(monkeypatch):
    """No record: nothing is served, nothing is classified, no filter is installed."""
    ir = lowered(flash_by_hand, SHORT)

    def classify(*args, **kwargs):
        raise AssertionError("a graph without a record was classified")

    monkeypatch.setattr(streaming, "obstacles", classify)
    assert served(ir) == {} and carried_reductions(ir) == {}
    assert not hasattr(ir, "streaming_refusals")
    assert streaming.streaming_accept(ir) is None
    kept, reasons = tilings(ir)
    assert kept and not reasons
    # the row sum and the value matmul are pinned to the whole row, as they always were
    row = {node_of(ir, ReduceSum).name, node_of(ir, Matmul, 1).name}
    for cand in kept:
        assert {int(x) for n, v in cand.rstep_map.items() if n.name in row
                for x in v.values()} == {256}


def test_an_obstacle_that_is_not_served_keeps_only_the_tilings_that_build():
    """A graph whose obstacle has a reason instead of a record (``streaming_refusals``)
    still gets the filter: the policy steps such a max on its own while the sums are
    pinned, and none of those tilings can be built. They are dropped before the top-k cut,
    counted by reason."""
    ir = lowered(flash_by_hand, SHORT)
    unfiltered, _ = tilings(ir)
    built = []
    for cand in unfiltered:
        try:
            schedule_kernel(cand)
        except ScheduleRejected:
            continue
        built.append(cand)
    assert (len(unfiltered), len(built)) == (32, 8)
    ir.streaming_refusals = [f"{node_of(ir, ReduceMax).name}: it has no update rule"]
    assert streaming.streaming_accept(ir) is not None
    kept, reasons = tilings(ir)
    assert len(kept) == 8
    assert {tuple(steps(c)) for c in kept} == {tuple(steps(c)) for c in built}
    assert sum(reasons.values()) == 24 and len(reasons) == 1
    (reason,) = reasons
    assert "is computed per iteration of" in reason and "outside it" in reason


def test_a_record_lets_the_loop_step_under_its_max():
    ir = lowered(flash_by_hand, SHORT)
    full, record = record_by_hand(ir)
    assert served(ir) == {full.name: list(record.loops)}
    assert {n.name for n in carried_reductions(ir)} == {full.name}
    assert not hasattr(ir, "streaming_refusals")
    kept, reasons = tilings(ir)
    kinds = Counter("looped" if sites(schedule_kernel(cand)) else "one tile" for cand in kept)
    assert kinds == {"looped": 24, "one tile": 8} and not reasons
    # the max steps with the sums it is carried for, one loop counter for the three
    for cand in kept:
        own = {n.name: [int(x) for x in v.values()] for n, v in cand.rstep_map.items()}
        assert own[full.name] == own[record.loops[0]] == own[record.loops[1]]


def test_a_policy_without_its_flat_graph_serves_nothing():
    ir = lowered(flash_by_hand, SHORT)
    full, _ = record_by_hand(ir)
    nodes, red = scheduled_reductions(ir)
    assert streamed(nodes, red, reduction_ancestors(nodes)) == {}
    assert streamed(nodes, red, reduction_ancestors(nodes), None) == {}
    assert not hasattr(ir, "streaming_refusals")


@pytest.mark.parametrize("foreign", ["key", "loops", "edited graph"])
def test_a_record_of_another_graph_is_not_served(foreign):
    """R1: the record says which graph its rules were proven for. On any other graph --
    another key, another loop membership, the graph itself after an edit -- the pin stays,
    and the graph says why."""
    ir = lowered(flash_by_hand, SHORT)
    full, record = record_by_hand(ir)
    if foreign == "key":
        full.streaming = dataclasses.replace(record, key="0" * 64)
    elif foreign == "loops":
        full.streaming = dataclasses.replace(record, loops=record.loops[::-1])
    else:
        node_of(ir, Div).const = 2.0  # the scaling of the scores: another function
    why = f"{full.name}: its update rule was derived for another graph"
    assert FOREIGN_RECORD == "its update rule was derived for another graph"
    assert served(ir) == {} and carried_reductions(ir) == {}
    assert ir.streaming_refusals == [why]
    assert served(ir) == {} and ir.streaming_refusals == [why]  # said once
    # the pipeline: only the tilings that hold the row whole are left, and every tiling in
    # which the policy stepped the max on its own is dropped before the top-k cut
    kept, reasons = tilings(ir)
    assert len(kept) == 8 and all(not sites(schedule_kernel(cand)) for cand in kept)
    assert ir.streaming_refusals == [why]
    row = {node_of(ir, ReduceSum).name, node_of(ir, Matmul, 1).name}
    for cand in kept:
        assert {int(x) for n, v in cand.rstep_map.items() if n.name in row
                for x in v.values()} == {256}
    if foreign == "edited graph":  # the record itself is right again once the edit is undone
        node_of(ir, Div).const = 8.0
        assert served(ir) == {full.name: list(record.loops)}


def test_an_edit_after_the_record_was_served_is_seen_by_apply_tiling():
    """``apply_tiling`` reads what is carried from a memo per flat graph. An edit in place of
    an attribute the record's key covers (the scaling of the scores), made after the memo
    was filled with the record served, must reach it too: otherwise the policy pins the
    sums (it asks the key afresh) while ``apply_tiling`` still carries the max, and the
    tilings that step the max alone are refused for a carried max that is not carried."""
    plain = lowered(flash_by_hand, SHORT)  # no record: what is kept, and why the rest is not
    plain.streaming_refusals = [f"{node_of(plain, ReduceMax).name}: it has no update rule"]
    plain_kept, plain_reasons = tilings(plain)
    ir = lowered(flash_by_hand, SHORT)
    full, record = record_by_hand(ir)
    assert {n.name for n in carried_reductions(ir)} == {full.name}  # the memo, served
    kept, reasons = tilings(ir)
    assert len(kept) == 32 and not reasons
    looped = next(cand for cand in kept if sites(schedule_kernel(cand)))
    node_of(ir, Div).const = 2.0
    assert served(ir) == {} and carried_reductions(ir) == {}
    assert ir.streaming_refusals == [f"{full.name}: {FOREIGN_RECORD}"]
    # a looped candidate made before the edit is refused by the tiler, as it would be on a
    # graph without a record: nothing is carried, so the max is held whole under the sums
    with pytest.raises(ScheduleRejected, match="is shared with an absorbed ancestor") as no:
        schedule_kernel(looped)
    assert not isinstance(no.value, StreamingRefused)
    kept, reasons = tilings(ir)
    assert len(kept) == len(plain_kept) == 8
    assert all(not sites(schedule_kernel(cand)) for cand in kept)
    # the same tilings are kept, and the others dropped for the same reason, as on the
    # graph without a record
    assert {tuple(steps(c)) for c in kept} == {tuple(steps(c)) for c in plain_kept}
    assert reasons == plain_reasons and sum(reasons.values()) == 24
    (reason,) = reasons
    assert "is computed per iteration of" in reason and "outside it" in reason
    node_of(ir, Div).const = 8.0  # undone: the record is served again, to both
    assert {n.name for n in carried_reductions(ir)} == {full.name}
    assert served(ir) == {full.name: list(record.loops)}


def test_a_long_row_with_a_record_of_another_graph_has_no_tiling_and_says_why(monkeypatch):
    """The report of a graph without tilings names why its max is not carried instead."""
    import cleave.lang as cl
    from cleave.frontend import trace
    from cleave.pipeline.superopt import GraphSet

    shapes = dict(H=32, SQ=64, SKV=4096, D=64)
    ir = lowered(flash_by_hand, shapes)
    full, record = record_by_hand(ir)
    assert len(tilings(ir)[0]) > 0
    full.streaming = dataclasses.replace(record, key="f" * 64)
    assert tilings(ir)[0] == []
    assert ir.streaming_refusals == [f"{full.name}: {FOREIGN_RECORD}"]
    monkeypatch.setattr(plans_module, "lower", lambda graph, split_factor=None: [ir])
    plans = P.schedule(GraphSet.identity(trace(flash_by_hand, shapes, cl.f16)),
                       P.ScheduleConfig(topk=8))
    assert not plans and plans.refused
    assert {str(r.error) for r in plans.refused} == {
        "a max needs its whole row in one block, and this row is longer; it is not carried "
        f"across a loop: {full.name}: its update rule was derived for another graph"}
    # without the list (no obstacle was looked at) the text is the one it always was
    del ir.streaming_refusals
    full.streaming = None
    plans = P.schedule(GraphSet.identity(trace(flash_by_hand, shapes, cl.f16)),
                       P.ScheduleConfig(topk=8))
    assert {str(r.error) for r in plans.refused} == {plans_module.MAX_ROW_LIMIT}


def unassembled(cand, monkeypatch):
    """``apply_tiling(cand)`` up to its tail: the loop is built and the max registered as
    carried, but the lag copy and the repairs are not there yet. Returns the tiled graph."""
    with monkeypatch.context() as patch:
        patch.setattr(streaming, "assemble", lambda tiled, flat_graph=None: tiled)
        return apply_tiling(cand)


def looped_candidate(ir):
    kept, _ = tilings(ir)
    return next(cand for cand in kept if sites(schedule_kernel(cand)))


def test_assemble_refuses_a_record_of_another_graph_before_it_edits(monkeypatch):
    ir = lowered(flash_by_hand, SHORT)
    full, record = record_by_hand(ir)
    cand = looped_candidate(ir)
    tiled = unassembled(cand, monkeypatch)
    assert not sites(tiled) and recorded_node(loop_tier(tiled)).streaming == record
    before = [n.name for n in loop_tier(tiled).nodes]
    other = ir.clone()
    node_of(other, Div).const = 2.0
    for flat_graph in (other, None):
        with pytest.raises(StreamingRefused) as refusal:
            assemble(tiled, flat_graph)
        assert str(refusal.value) == f"{full.name}: its update rule was derived for another graph"
        # nothing was edited: the same nodes, no carried state added, the record still there
        tier = loop_tier(tiled)
        assert [n.name for n in tier.nodes] == before
        assert not tier.lag_nodes and not tier.accum_scale and tiled.locked
        assert recorded_node(tier).streaming == record
    # ... so the same tiled graph is assembled once it is asked with its own flat graph
    assert assemble(tiled, cand.flat_graph) is tiled
    (site,) = sites(tiled)
    assert set(site.repaired) == set(record.loops) and loop_tier(tiled).lag_nodes
    assert [n.name for n in loop_tier(tiled).nodes] == before + [
        f"{full.name}_prev", f"{full.name}_upd0", f"{full.name}_upd1"]
    assert assemble(tiled, cand.flat_graph) is tiled and len(sites(tiled)) == 1  # consumed


def test_assemble_refuses_a_factor_without_a_tile_op_before_it_edits(monkeypatch):
    """L14, decided with everything else a tiling can be refused for: before a node is
    touched."""
    ir = lowered(flash_by_hand, SHORT)
    full, record = record_by_hand(ir)
    cand = looped_candidate(ir)
    full.streaming = dataclasses.replace(
        record, rules=tuple((acc, ("max", "m", "m'")) for acc in record.loops))
    # (the key does not hold the rules: they are what it is the key of)
    tiled = unassembled(cand, monkeypatch)
    before = [n.name for n in loop_tier(tiled).nodes]
    with pytest.raises(StreamingRefused) as refusal:
        assemble(tiled, cand.flat_graph)
    assert str(refusal.value) == f"{full.name}: the repair uses 'max', which has no tile op"
    assert [n.name for n in loop_tier(tiled).nodes] == before and not loop_tier(tiled).lag_nodes
    assert not [e for e in recorded_node(loop_tier(tiled)).outputs
                if e.dst_node.name.endswith("_prev")]


# ------------------------------------------- a carried max steps with its loop, or not at all


def with_steps(cand, **steps):
    """``cand`` with the reduction steps of the named nodes replaced: a step map the policy
    does not emit, built by hand."""
    stepped = copy.copy(cand)
    stepped.rstep_map = {node: {axis: steps.get(node.name, step) for axis, step in own.items()}
                         for node, own in cand.rstep_map.items()}
    return stepped


def verdicts(ir, cand):
    """``(what apply_tiling says, what the policy's twin of it says)`` of a candidate: the
    refusal text, or None where it is built / has no conflict."""
    nodes, red = scheduled_reductions(ir)
    twin = absorbed_axis_conflict(nodes, cand.rstep_map, red, reduction_ancestors(nodes),
                                  carried_reductions(ir))
    try:
        apply_tiling(cand)
    except ScheduleRejected as refusal:
        return str(refusal), twin
    return None, twin


def test_a_carried_max_that_is_out_of_step_with_its_loop_is_refused():
    """C1, C2: the loop of the sums has one counter. A step map in which the carried max
    is stepped differently from a sum that iterates, or is stepped while a sum that reads
    its finished value is not, is refused with a typed reason -- by ``apply_tiling`` and,
    in the same words, by the check the policy rates its step maps with."""
    ir = lowered(flash_by_hand, SHORT)
    full, record = record_by_hand(ir)
    base = looped_candidate(ir)
    row_sum, value_matmul = node_of(ir, ReduceSum).name, node_of(ir, Matmul, 1).name
    assert set(record.loops) == {row_sum, value_matmul}
    in_step = {full.name: 32, row_sum: 32, value_matmul: 32}
    assert verdicts(ir, with_steps(base, **in_step)) == (None, None)

    def carried_by(loop, step, own):
        return (f"{full.name} is carried by the loop of {loop}, which steps {step}, but is "
                f"itself stepped {own}: one loop counter advances both")

    def needed_whole(reader, own):
        return (f"{full.name} is stepped {own} but {reader}, which reads its finished value, "
                "is not: the value would be needed whole and per iteration at once")

    first = next(n.name for n in ir.nodes if n.name in record.loops)  # the one named first
    out_of_step = {
        "the max whole, the sums stepped": (
            {**in_step, full.name: 256}, carried_by(first, 32, 256)),
        "the max stepped 64, the sums 32": (
            {**in_step, full.name: 64}, carried_by(first, 32, 64)),
        "only the row sum stepped": (
            {full.name: 256, row_sum: 32, value_matmul: 256}, carried_by(row_sum, 32, 256)),
        "only the value matmul stepped": (
            {full.name: 256, row_sum: 256, value_matmul: 32}, carried_by(value_matmul, 32, 256)),
        "the max stepped alone": (
            {full.name: 32, row_sum: 256, value_matmul: 256}, needed_whole(first, 32)),
    }
    for label, (steps, text) in out_of_step.items():
        assert verdicts(ir, with_steps(base, **steps)) == (text, text), label
    # the flat graph was not touched by any of the refused candidates
    assert full.streaming is record and not hasattr(ir, "streaming_refusals")


def test_without_a_record_the_out_of_step_refusals_do_not_exist():
    """The same step maps on the same graph without a record: whatever the tiler said
    before is what it says (nothing is carried, so nothing can be out of step)."""
    with_record = lowered(flash_by_hand, SHORT)
    full, _ = record_by_hand(with_record)
    base = looped_candidate(with_record)
    ir = lowered(flash_by_hand, SHORT)
    assert carried_reductions(ir) == {}
    row_sum, value_matmul = node_of(ir, ReduceSum).name, node_of(ir, Matmul, 1).name
    absorbed = ("Reduction axis of {} is shared with an absorbed ancestor reduction; one loop "
                "cannot both advance it and hold it whole.")
    plain = with_steps(base)
    plain.flat_graph = ir
    for steps, want in [
        # the step map that is served with a record is the absorbed-axis conflict without
        ({full.name: 32, row_sum: 32, value_matmul: 32}, absorbed),
        ({full.name: 64, row_sum: 32, value_matmul: 32}, absorbed),
        # the max whole in every iteration of the sums' loop: built, as it always was
        ({full.name: 256, row_sum: 32, value_matmul: 32}, None),
    ]:
        said, twin = verdicts(ir, with_steps(plain, **steps))
        if want is None:
            assert said is None and twin is None
            continue
        assert said == twin and said in {want.format(row_sum), want.format(value_matmul)}
        assert "carried" not in said and "finished value" not in said


def test_a_carried_max_is_not_absorbed_by_the_loop_that_carries_it():
    """``absorbed_reductions(carried=)``: an ancestor that iterates under an iterating
    descendant is absorbed into its loop and held whole -- except the value that loop
    carries, with respect to the descendants it is carried for."""
    ir = lowered(flash_by_hand, SHORT)
    full, record = record_by_hand(ir)
    base = looped_candidate(ir)
    nodes, _ = scheduled_reductions(ir)
    ancestors, carried = reduction_ancestors(nodes), carried_reductions(ir)
    scores = node_of(ir, Matmul).name
    step_map = with_steps(base, **{scores: 32}).rstep_map  # every reduction iterates
    names = lambda found: sorted(n.name for n in found)  # noqa: E731
    assert names(absorbed_reductions(nodes, step_map, ancestors)) == sorted([scores, full.name])
    assert names(absorbed_reductions(nodes, step_map, ancestors, {})) == sorted([scores, full.name])
    assert names(absorbed_reductions(nodes, step_map, ancestors, carried)) == [scores]
    # carried for one of its descendants only: still absorbed by the loop of the other
    (key,) = carried
    one = {key: carried[key][:1]}
    assert names(absorbed_reductions(nodes, step_map, ancestors, one)) == sorted([scores, full.name])


def flash_and_its_max(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                      v: cl.Tensor["H", "SKV", "D"]):
    """The flash form that also returns its row max (what a split partial hands on)."""
    s = (q @ k.mT) / q.shape[-1] ** 0.5
    m = cl.max(s, -1, keepdim=True)
    e = cl.exp(s - m)
    return (e @ v) / cl.sum(e, -1, keepdim=True), m


def test_a_returned_max_leaves_its_loop_as_carried_state(monkeypatch):
    """``carry_early``: a carried value that is read outside its loop is registered as an
    accumulator of the loop while ``apply_tiling`` is still building. Otherwise the max
    would be one more per-iteration value that is read outside, which is a refusal."""
    ir = lowered(flash_and_its_max, SHORT)
    full, record = record_by_hand(ir)
    assert sorted(type(e.dst_node).__name__ for e in full.outputs) == ["OutputNode", "Sub"]
    kept, reasons = tilings(ir)
    kinds = Counter("looped" if sites(schedule_kernel(cand)) else "one tile" for cand in kept)
    assert kinds == {"looped": 24, "one tile": 8} and not reasons
    cand = looped_candidate(ir)
    tier = loop_tier(schedule_kernel(cand))
    assert [n.name for n in tier.accum_nodes][-1] == full.name  # after the loop's own
    assert {n.name for n in tier.accum_nodes} == {full.name, *record.loops}
    code, _ = build_candidate(cand)
    assert f"T.fill(t_{full.name}, -3.4028234663852886e38)" in code
    assert f"ukernel_copy_1dims_rr(t_{full.name}, t_{full.name}_prev)" in re.sub(r"\s+", " ", code)
    with monkeypatch.context() as patch:
        patch.setattr(streaming, "carry_early", lambda main, rstep_map: None)
        with pytest.raises(ScheduleRejected) as refusal:
            apply_tiling(cand)
    assert str(refusal.value).startswith(
        f"{full.name} is computed per iteration of {tier_owner(schedule_kernel(cand))} but "
        "also read by ")


def tier_owner(tiled) -> str:
    (owner,) = [owner.name for _, owner in graph_tiers(tiled) if owner is not None]
    return owner


# --------------------------------------------------------------------------- the validator


def test_codegen_refuses_an_update_rule_that_was_not_assembled(monkeypatch):
    """V1': a max that its loop carries (or steps) with its record still on it has no lag
    copy and no repairs. Emitting it would rescale nothing: the sums would be relative to
    the max of one slice."""
    from cleave.codegen import TilelangCodeGen

    ir = lowered(flash_by_hand, SHORT)
    full, _ = record_by_hand(ir)
    cand = looped_candidate(ir)
    tiled = unassembled(cand, monkeypatch)
    with pytest.raises(AssertionError) as error:
        TilelangCodeGen().gen(tiled)
    assert (f"{full.name!r} has an update rule that was not assembled; emitting it as a "
            "plain max would take the max of one slice") in str(error.value)
    # the same candidate, assembled, is generated ...
    code, _ = build_candidate(cand)
    assert f"t_{full.name}_prev" in code and "ukernel_rescale" in code
    # ... and so is a tiling that holds the row whole, whose record is not consumed
    kept, _ = tilings(ir)
    whole = next(c for c in kept if not sites(schedule_kernel(c)))
    assert recorded_node(schedule_kernel(whole)).streaming is not None
    code, _ = build_candidate(whole)
    assert "ukernel_rescale" not in code


def test_codegen_refuses_a_recorded_max_that_only_steps(monkeypatch):
    """V1', second half: not registered as carried either, but its operand tile is a slice
    of its block along the reduced dim."""
    from cleave.codegen import TilelangCodeGen

    ir = lowered(flash_by_hand, SHORT)
    full, _ = record_by_hand(ir)
    cand = looped_candidate(ir)
    with monkeypatch.context() as patch:
        patch.setattr(streaming, "carry_early", lambda main, rstep_map: None)
        tiled = unassembled(cand, patch)
    tier = loop_tier(tiled)
    assert not any(n.name == full.name for n in tier.accum_nodes)
    with pytest.raises(AssertionError, match="has an update rule that was not assembled"):
        TilelangCodeGen().gen(tiled)
