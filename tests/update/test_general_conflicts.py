"""Reduction conflicts whose ancestor is not a max, through the same path as a softmax's (CPU).

A conflict is a reduction that a descendant reduction over the same dim depends on
(``schedule.streaming.obstacles``); which op the ancestor is plays no part in finding it,
in deriving its update rule or in assembling the loop. Here the ancestor
is a sum:

* ``exp(y - sum(y))``: the row sum and the value matmul loop under the sum of the row. The
  same search finds the same repair as for a max, ``a * exp(m - m')``, the same tests
  confirm it, and the loop carries the sum (from zero, its ``carry_init``) next to them;
* its traced order: two conflicts, and no update reproduces either (the value matmul reads
  the finished row sum; the exp it reads changes with the first sum);
* ``(x / sum(x)) @ v``: the update divides by the running sum, ``a * (m / m') + r``. The test
  is random, so it cannot tell that a running sum may be zero where the finished one is not
  (the kernel would then divide by zero where the program does not): that is the price of
  testing instead of proving, the one the first pass pays for the program's own divisions;
* ``(x / |x|) @ v``: a norm in front of the contraction over its own dim, repaired by
  ``a * sqrt(m / m') + r``. Its test data divides by zero in the field now and then (a
  running sum of squares that is zero in a residue): those elements are skipped in that
  residue, and the rule is found whatever the draw;
* ``exp(y - mean(y))``: no rule. On a prefix of ``n`` elements the mean is the sum over
  ``n``, which no factor over the value before and after the step gives.

``test_search_*`` run the superoptimizer first.
"""

import time

import pytest

pytest.importorskip("torch")

from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops import ReduceSum
from cleave.ir.update_rule import loop_key, show
from cleave.pipeline import plans
from cleave.schedule import search_tiling
from cleave.schedule.streaming import sites
from cleave.schedule.tiling import schedule_kernel

from ._attention import FACTOR
from ._programs import (derivation, identity, l2_normalized, lowered, mean_shifted, searched,
                        sum_normalized, sum_shifted)

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
    assert first.startswith("reduction_1_3: no update of mma_8 of the form ")
    assert second.startswith("reduction_1_6: no update of mma_8 of the form ")
    assert not [n for n in ir.nodes if getattr(n, "streaming", None) is not None]


def test_search_a_division_by_a_running_sum_divides_by_the_running_sum():
    _, graphs = searched(sum_normalized, ROWS)
    t0 = time.perf_counter()
    (ir,) = plans.lower(_graph(graphs, "sum>div>matmul"))
    assert time.perf_counter() - t0 < 10.0
    (d,) = ir.derivations
    assert not d.refused and d.rules == (("mma_4", ("div", "m", "m'")),)
    assert ir.streaming_refusals == []
    # the other graph, which divides after the matmul, has no conflict at all
    (other,) = plans.lower(_graph(graphs, "sum>matmul>div"))
    assert not hasattr(other, "streaming_refusals") and not hasattr(other, "derivations")


def test_a_norm_in_front_of_its_contraction_is_repaired_by_the_ratio_of_its_norms():
    """The value matmul loops under the sum of squares it divides the row by (through its
    square root). Some of its test data divides by zero in a residue; that is skipped there,
    and the rule is the ratio of the norms before and after the step."""
    from cleave.pipeline.update import engine
    from cleave.pipeline.update.reference import reference_set

    d = derivation(identity(l2_normalized, dict(R=64, C=256, D=64)))
    assert not d.refused, d.refused
    (acc,) = d.loops
    assert show(d.chosen[acc]) == "(r + (a * sqrt((m / m'))))"
    *_, excluded = engine.hole_blocks(reference_set(d.flat), d.full, acc)
    assert excluded.any() and not (excluded == 3).all()


def test_a_mean_shift_has_no_rule():
    """The conflict is the row sum the mean divides by the dim's extent; on each prefix that
    extent is the prefix's, so no factor over the sums before and after the step repairs
    the accumulators."""
    graph = identity(mean_shifted, dict(R=64, C=256, D=64))
    assert sum(type(n).__name__ == "ReduceSum" for n in lowered(graph).nodes) == 2
    d = derivation(graph)
    assert d.refused.startswith("no update of ") and "reproduces the flat graph" in d.refused


def test_search_a_conflict_without_a_rule_keeps_the_dim_whole():
    """The traced order of the sum shift: no rule, so the pin stays and every tiling holds
    the row of the value matmul whole (the policy may still step a sum on its own;
    apply_tiling refuses those tilings)."""
    _, graphs = searched(sum_shifted, ROWS)
    (ir,) = plans.lower(_graph(graphs, "sum>div>matmul"))
    assert {int(s) for c in search_tiling(ir, topk=1000) for n, steps in c.rstep_map.items()
            if n.name == "mma_8" for s in steps.values()} == {256}
