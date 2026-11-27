"""The engine on MHA prefill and qknorm attention. CPU only.

``test_search_*`` run the superoptimizer first (marked ``search``). The derivations here
are made by ``engine.run`` itself (no in-process memo, no store), so every list below is
what the search, the tests in the field and the ranking produce now.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from cleave.ir.ops import ReduceMax
from cleave.ir.update_rule import show
from cleave.pipeline import plans
from cleave.pipeline.update import engine

from ._attention import FACTOR
from ._programs import MHA, attention, derivation, mha_derivation, qknorm, searched

ROOT = Path(__file__).resolve().parents[2]
QKNORM = tuple(dict(HQ=64, HKV=8, SQ=4096, SKV=4096, D=128).items())
CANONICAL = "(r + (a * exp((m - m'))))"


def flash(graphs):
    (graph,) = [g for g in graphs if g.summary().endswith("sum>matmul>div")]
    return graph


def traced(graphs):
    (graph,) = [g for g in graphs if g.summary().endswith("sum>div>matmul")]
    return graph


def mha(budgets=(4,)):
    return mha_derivation(budgets)


def listed(d):
    """Per hole: ``[(update text, emittable)]`` of every match of its last search."""
    return {h.acc: [(show(e), e in h.emittable) for e in h.matches] for h in d.holes}


def test_search_mha_prefill_has_one_right_update_per_hole_at_budget_4():
    d = mha()
    assert not d.refused, d.refused
    assert d.budget == 4 and [h.search.budget for h in d.holes] == [4, 4]
    assert [h.acc for h in d.holes] == ["reduction_2_8", "mma_9"]  # narrowest first
    assert [h.wide for h in d.holes] == [False, True]
    assert listed(d) == {acc: [(CANONICAL, True)] for acc in d.loops}
    assert len(d.ranked) == 1 and not d.rejected
    (best,) = d.ranked
    assert (best.grade, best.cost, best.size) == (0, 2, 8)
    assert {a: show(u) for a, u in d.chosen.items()} == {a: CANONICAL for a in d.loops}
    assert d.rules == tuple((acc, FACTOR) for acc in d.loops)
    assert d.record().rules == d.rules and d.record().key == d.key
    # the test data: every step of every draw, at the first pass's own test sizes
    assert d.cases == [f"draw {s} prefix {p}" for s in (1, 2) for p in (0, 4, 8, 12)]
    (row, wide) = d.holes
    assert row.search.elements[1] == 8 * 20 * 28 and wide.search.elements[1] == 8 * 20 * 28 * 44
    assert list(best.errors) == ["normal@32", "normal@4", "large_logits@32", "increasing@32",
                                 "decreasing@32", "last_spike@32", "constant@32", "offset@32"]
    assert best.errors["normal@4"] == 0.0


def test_search_mha_prefill_lists_at_budget_5_and_the_ranking_rejects_three():
    """At budget 5 a second factor reproduces the flat graph exactly, ``exp(m) / exp(m')``:
    the same field element. Of the four combinations the three that use it are non-finite
    on ``large_logits @ 32`` (``exp(m')`` overflows at the first tile), and the winner is the
    canonical pair."""
    d = mha((5,))
    assert not d.refused, d.refused
    want = [(CANONICAL, True), ("(r + (a * (exp(m) / exp(m'))))", True)]
    assert listed(d) == {acc: want for acc in d.loops}
    assert len(d.rejected) == 3 and len(d.ranked) == 1
    for text, why in d.rejected:
        assert "(exp(m) / exp(m'))" in text and why.startswith("non-finite") \
            and why.endswith("on large_logits@32")
    (best,) = d.ranked
    assert best.text == "; ".join(f"{a}' = {CANONICAL}" for a in sorted(d.loops))
    assert d.rules == tuple((acc, FACTOR) for acc in d.loops)


def test_search_no_repair_is_never_a_rule():
    """``a + r`` (the factor 1, ``m / m``) reproduces a row only where its max does not
    move; random field values always move it."""
    from cleave.pipeline.update.search import evaluate

    from cleave.pipeline.update.reference import reference_set

    ir, ob, recipe, key = _mha_setting()
    d = engine.run(ir, ob, recipe, key=key, budgets=(4,))
    cases = reference_set(d.flat)
    for acc in d.loops:
        leaves, target, _, _ = engine.hole_blocks(cases, d.full, acc)
        no_repair = evaluate(("add", "r", ("mul", "a", ("div", "m", "m"))), leaves)
        assert (no_repair != target).any()


def _mha_setting():
    from cleave.ir.update_rule import loop_key
    from cleave.pipeline.update.instantiate import Recipe

    from ._programs import lowered, the_obstacle

    _, graphs = searched(attention, MHA, 11)
    graph = flash(graphs)
    ir = lowered(graph)
    ob = the_obstacle(ir)
    return ir, ob, Recipe.of(graph), loop_key(ir, ob.full, ob.loops)


def test_search_two_derivations_in_one_process_are_equal():
    _, graphs = searched(attention, MHA, 11)
    one, two = mha(), derivation(flash(graphs))
    assert one is not two
    assert one.rules == two.rules and one.chosen == two.chosen
    assert engine.report(one) == engine.report(two)
    assert "timings" not in engine.report(one)


_CHILD = r"""
import json, sys
from cleave.ir.update_rule import loop_key
from cleave.pipeline.update import engine
from cleave.pipeline.update.instantiate import Recipe
from tests.update._programs import attention, lowered, searched, the_obstacle
from tests.update.test_engine import MHA, flash

_, graphs = searched(attention, MHA, 11)
graph = flash(graphs)
ir = lowered(graph)
ob = the_obstacle(ir)
d = engine.run(ir, ob, Recipe.of(graph), key=loop_key(ir, ob.full, ob.loops))
print("RESULT " + json.dumps({"rules": d.rules, "report": engine.report(d)}))
"""


def _derive_in_a_process(cache_root) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CLEAVE_CACHE_ROOT=str(cache_root),
               PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "python"),
                                           os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, "-c", _CHILD], cwd=cache_root, env=env,
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    (line,) = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
    return json.loads(line[len("RESULT "):])


def test_search_two_derivations_in_two_processes_are_equal(tmp_path):
    """Each process searches and derives from scratch (its own cache root)."""
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    a, b = _derive_in_a_process(first), _derive_in_a_process(second)
    assert a == b
    here = mha()
    assert a["report"] == engine.report(here)
    assert json.loads(json.dumps(here.rules)) == a["rules"]


def test_search_qknorm_flash_form_gets_a_record_and_the_traced_form_its_reason():
    """The flash form gets the same rule as MHA. The sums of squares of the two norms are
    conflicts as well (the score matmul contracts the normed dim under them): no update of
    the form ``a * F + r`` reproduces what is contracted under an ``rsqrt`` of a running
    sum, so they keep their pins; in the traced order no conflict gets a rule."""
    _, graphs = searched(qknorm, QKNORM, None)
    (ir,) = plans.lower(flash(graphs))
    (other,) = plans.lower(traced(graphs))
    (full,) = [n for n in ir.nodes if getattr(n, "streaming", None) is not None]
    record = full.streaming
    assert type(full) is ReduceMax
    assert record.rules == tuple((acc, FACTOR) for acc in record.loops) and len(record.loops) == 2
    assert len(ir.streaming_refusals) == 2
    assert all(" no update of " in why for why in ir.streaming_refusals)
    (d,) = [d for d in ir.derivations if not d.refused]
    assert d.full == full.name and d.rules == record.rules
    assert not [n for n in other.nodes if getattr(n, "streaming", None) is not None]
    assert len(other.streaming_refusals) == 4
    assert all(" no update of " in why for why in other.streaming_refusals)


def test_search_qknorm_tests_at_the_first_pass_sizes():
    """The rows of the test data are the first pass's own: its query heads are a multiple of
    its kv heads, and a row of every case is a test."""
    _, graphs = searched(qknorm, QKNORM, None)
    d = derivation(flash(graphs), cls=ReduceMax)  # the norms' sums are conflicts too
    assert not d.refused, d.refused
    assert d.rules == tuple((acc, FACTOR) for acc in d.loops)
    assert all(d.flat.sizes[s] == s for s in d.flat.sizes)
    assert len(d.cases) == 8
