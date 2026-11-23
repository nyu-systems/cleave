"""The engine on the programs the design measures it on (DESIGN 6, 8; 13.2 b, d). CPU only.

``test_search_*`` run the superoptimizer first (marked ``search``). The derivations here
are made by ``engine.run`` itself (no in-process memo, no store), so every list below is
what the search, the proof and the ranking produce now.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("z3")

from cleave.ir.update_rule import repair_of, show
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
    """Per hole: ``[(update text, proved, emittable)]`` of every match of its last search."""
    return {h.acc: [(show(e), v.proved, repair_of(e) is not None) for e, _, v in h.verdicts]
            for h in d.holes}


def test_search_mha_prefill_has_one_right_update_per_hole_at_budget_4():
    d = mha()
    assert not d.refused, d.refused
    assert d.budget == 4 and [h.search.budget for h in d.holes] == [4, 4]
    assert [h.acc for h in d.holes] == ["reduction_2_8", "mma_9"]  # narrowest first
    assert [h.wide for h in d.holes] == [False, True]
    assert listed(d) == {acc: [(CANONICAL, True, True)] for acc in d.loops}
    assert len(d.ranked) == 1 and not d.rejected
    (best,) = d.ranked
    assert (best.grade, best.stable, best.cost, best.size) == (0, True, 2, 8)
    assert {a: show(u) for a, u in d.chosen.items()} == {a: CANONICAL for a in d.loops}
    assert d.rules == tuple((acc, FACTOR) for acc in d.loops)
    assert d.record().rules == d.rules and d.record().key == d.key
    # every reference row the design counts, and the cases it names
    assert d.adequacy == {"mma_9": 23, "reduction_2_8": 20}
    assert d.shaped == "shaped seed=0 ramp x16"
    labels = [label for label, _ in d.cases]
    assert labels[:4] == ["first seed=1", "normal seed=1 prefix=4", "normal seed=1 prefix=8",
                          "normal seed=1 prefix=12"]
    assert len(labels) == 11 and labels[-1] == "shaped seed=0 ramp x16 prefix=12"
    assert list(best.errors) == ["normal@32", "normal@4", "large_logits@32", "increasing@32",
                                 "decreasing@32", "last_spike@32", "constant@32", "offset@32"]
    assert best.errors["normal@4"] == 0.0


def test_search_mha_prefill_lists_at_budget_5_and_the_ranking_rejects_three():
    """The four matches per hole of DESIGN 6, all proved, two of them emittable; of the four
    combinations of emittable rules the three with ``exp(m) / exp(m')`` are non-finite on
    ``offset @ 32`` (``0 / 0``), and the winner is the canonical pair."""
    d = mha((5,))
    assert not d.refused, d.refused
    want = [(CANONICAL, True, True),
            ("(r + ((a * exp(m)) / exp(m')))", True, False),
            ("(r + (a * (exp(m) / exp(m'))))", True, True),
            ("(r + (exp(m) * (a / exp(m'))))", True, False)]
    assert listed(d) == {acc: want for acc in d.loops}
    for h in d.holes:
        assert [show(e) for e in h.emittable] == [want[0][0], want[2][0]]
    assert len(d.rejected) == 3 and len(d.ranked) == 1
    for text, why in d.rejected:
        assert "(exp(m) / exp(m'))" in text and why.startswith("non-finite") \
            and why.endswith("on offset@32")
    (best,) = d.ranked
    assert best.text == "; ".join(f"{a}' = {CANONICAL}" for a in sorted(d.loops))
    assert d.rules == tuple((acc, FACTOR) for acc in d.loops)
    # a rule the assembly cannot emit is listed, proved, and never ranked
    ranked_rules = [show(u) for r in d.ranked for u in r.rules.values()]
    assert "(r + ((a * exp(m)) / exp(m')))" not in ranked_rules


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
    """RMS-normalized keys move the max little: the reference set's shaped sample is chosen
    for the rows it discriminates, and the flash form gets the same rule as MHA."""
    _, graphs = searched(qknorm, QKNORM, None)
    (ir,) = plans.lower(flash(graphs))
    (other,) = plans.lower(traced(graphs))
    (full,) = [n for n in ir.nodes if getattr(n, "streaming", None) is not None]
    record = full.streaming
    assert record.rules == tuple((acc, FACTOR) for acc in record.loops) and len(record.loops) == 2
    assert ir.streaming_refusals == []
    (d,) = ir.derivations
    assert not d.refused and d.rules == record.rules
    # the traced order: the value matmul reads the finished row sum (T3)
    assert not [n for n in other.nodes if getattr(n, "streaming", None) is not None]
    (why,) = other.streaming_refusals
    assert why.endswith(", a finished reduction over the dim reduce_max_2_19 reduces: that "
                        "needs two passes over the row") and " reads reduction_2_22" in why
    assert not hasattr(other, "derivations")


def test_search_qknorm_reference_set_is_adequate():
    _, graphs = searched(qknorm, QKNORM, None)
    d = derivation(flash(graphs))
    assert not d.refused, d.refused
    assert d.shaped == "shaped seed=9 none x1"
    assert d.adequacy == {"mma_23": 8, "reduction_2_22": 9}
    assert d.references[0].rows == 4
    assert listed(d) == {acc: [(CANONICAL, True, True)] for acc in d.loops}
