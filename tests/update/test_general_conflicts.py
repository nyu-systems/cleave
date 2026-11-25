"""Reduction conflicts whose ancestor is not a max, through the same path as a softmax's (CPU).

A conflict is a reduction that a descendant reduction over the same dim depends on
(``schedule.streaming.obstacles``); which op the ancestor is plays no part in finding it,
in classifying it, in deriving its update rule or in assembling the loop. Here the ancestor
is a sum:

* ``exp(y - sum(y))``: the row sum and the value matmul loop under the sum of the row. The
  same search finds the same repair as for a max, ``a * exp(m - m')``, the same proof
  proves it, and the loop carries the sum (from zero, its ``carry_init``) next to them;
* its traced order: two conflicts, the first one two passes over the row, the second one
  refused by the engine (the exp it reads changes with the first);
* ``(x / sum(x)) @ v``: every update would divide by the running sum, which may be zero, so
  the conflict is refused before anything is searched (E16) and keeps its pin: the dim is
  held whole, as it always was.

``test_search_*`` run the superoptimizer first.
"""

import time

import pytest

pytest.importorskip("torch")
pytest.importorskip("z3")

from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops import ReduceSum
from cleave.ir.update_rule import loop_key, show
from cleave.pipeline import plans
from cleave.pipeline.update import texts
from cleave.schedule import search_tiling
from cleave.schedule.streaming import sites
from cleave.schedule.tiling import schedule_kernel

from ._attention import FACTOR
from ._programs import searched, sum_normalized, sum_shifted

ROWS = tuple(dict(R=64, C=256, D=64).items())


def _graph(graphs, tail):
    (graph,) = [g for g in graphs if g.summary().endswith(tail)]
    return graph


def _key_step(cand):
    (step,) = {int(x) for v in cand.rstep_map.values() for x in (v or {}).values()}
    return step


def test_search_a_sum_in_front_of_its_sums_is_carried_by_their_loop():
    _, graphs = searched(sum_shifted, ROWS)
    (ir,) = plans.lower(_graph(graphs, "sum>matmul>div"))
    (full,) = [n for n in ir.nodes if getattr(n, "streaming", None) is not None]
    record = full.streaming
    assert type(full) is ReduceSum and ir.streaming_refusals == []
    assert record.rules == tuple((acc, FACTOR) for acc in record.loops) and len(record.loops) == 2
    assert record.key == loop_key(ir, full.name, record.loops)
    (d,) = ir.derivations
    assert {acc: show(u) for acc, u in d.chosen.items()} == {
        acc: "(r + (a * exp((m - m'))))" for acc in record.loops}

    # every tiling is built: the looped ones carry the sum from zero, the others hold the
    # row whole (they stay candidates next to the looped ones)
    steps = {}
    for cand in search_tiling(ir, topk=1000):
        tiled = schedule_kernel(cand)
        found = sites(tiled)
        step = _key_step(cand)
        steps[step] = bool(found)
        if not found:
            continue
        (site,) = found
        assert site.full == full.name and set(site.repaired) == set(record.loops)
        tier = next(t for t, o in graph_tiers(tiled) if o is not None and o.name == site.tier)
        carried = next(n for n in tier.nodes if n.name == full.name)
        assert any(carried is n for n in tier.accum_nodes) and carried.accum_init is None
        assert [n.name for n in tier.lag_nodes] == [f"{full.name}_prev"]
    assert steps == {32: True, 64: True, 128: True, 256: False}


def test_search_the_traced_order_of_a_sum_shift_says_why_twice():
    _, graphs = searched(sum_shifted, ROWS)
    (ir,) = plans.lower(_graph(graphs, "sum>div>matmul"))
    first, second = ir.streaming_refusals
    assert first.endswith("a finished reduction over the dim reduction_1_3 reduces: that "
                          "needs two passes over the row")
    assert second.startswith("reduction_1_6: ") and (
        "is computed from reduction_1_3, another reduction over the dim reduction_1_6 "
        "reduces" in second)
    assert not [n for n in ir.nodes if getattr(n, "streaming", None) is not None]


def test_search_a_division_by_a_running_sum_is_refused_before_anything_is_searched():
    _, graphs = searched(sum_normalized, ROWS)
    t0 = time.perf_counter()
    (ir,) = plans.lower(_graph(graphs, "sum>div>matmul"))
    assert time.perf_counter() - t0 < 5.0  # the search alone took 20 s at budgets 4 and 5
    (d,) = ir.derivations
    assert d.refused == texts.E16.format(node="div_3", full="reduction_1_2", acc="mma_4")
    assert not d.holes and d.recipe_key == ""  # read off the flat graph: the loop structure's
    assert ir.streaming_refusals == [f"reduction_1_2: {d.refused}"]
    # the pin stays: every tiling holds the row of the matmul whole (the policy may still
    # step the sum on its own; apply_tiling refuses those tilings)
    assert {int(s) for c in search_tiling(ir, topk=1000) for n, steps in c.rstep_map.items()
            if n.name == "mma_4" for s in steps.values()} == {256}
    # and the other graph, which divides after the matmul, has no conflict at all
    (other,) = plans.lower(_graph(graphs, "sum>matmul>div"))
    assert not hasattr(other, "streaming_refusals") and not hasattr(other, "derivations")
