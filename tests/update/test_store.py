"""The derivation store: one entry per loop structure on disk, every
stored rule proved again on load, a stored refusal trusted, the key moving with the engine.
CPU only; every test has a cache root of its own.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("z3")

import cleave
import cleave.lang as cl
from cleave.ir.update_rule import loop_key
from cleave.pipeline import cache, plans
from cleave.pipeline import update as update_package
from cleave.pipeline.update import engine, proof, store, texts
from cleave.pipeline.update.instantiate import Recipe

from ._attention import FACTOR
from ._programs import _max, flash_rows, identity, lowered, the_obstacle
from .test_refusals import column_weighted

ROOT = Path(__file__).resolve().parents[2]
SHAPES = dict(R=16, C=64, D=8)


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """An empty store and an empty in-process memo."""
    monkeypatch.setattr(update_package, "_DERIVED", {})
    with cleave.Session(cache_root=tmp_path):
        yield tmp_path


def setting(fn=flash_rows, shapes=SHAPES):
    graph = identity(fn, shapes)
    ir = lowered(graph)
    return ir, the_obstacle(ir), Recipe.of(graph)


def derive_again(ir, ob, recipe, monkeypatch):
    monkeypatch.setattr(update_package, "_DERIVED", {})
    return update_package.derive(ir, ob, recipe)


def no_search(monkeypatch):
    def run(*args, **kwargs):
        raise AssertionError("derived again")

    monkeypatch.setattr(engine, "run", run)


def proofs(monkeypatch):
    calls = []
    real = proof.prove

    def counted(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(proof, "prove", counted)
    return calls


def test_a_derivation_is_stored_and_read_back_with_every_rule_proved_again(fresh, monkeypatch):
    ir, ob, recipe = setting()
    first = update_package.derive(ir, ob, recipe)
    assert not first.refused and not first.from_store
    path = store.path_of(first.key)
    assert path.is_relative_to(cache.root() / "update") and path.parent.name == path.stem[:2]
    entry = json.loads(path.read_text())
    assert set(entry) == {"key", "recipe", "full", "loops", "rules", "refused", "report"}
    assert entry["key"] == first.key == loop_key(ir, ob.full, ob.loops)
    assert entry["recipe"] == "" == first.recipe_key  # a rule holds for the loop structure
    assert entry["rules"] == [[acc, json.loads(json.dumps(FACTOR))] for acc in ob.loops]
    assert entry["report"] == engine.report(first)
    assert update_package.derive(ir, ob, recipe) is first  # the in-process memo

    no_search(monkeypatch)
    calls = proofs(monkeypatch)
    again = derive_again(ir, ob, recipe, monkeypatch)
    assert again.from_store and again.rules == first.rules and again.record() == first.record()
    assert calls == [("add", "r", ("mul", "a", FACTOR))] * len(ob.loops)
    assert engine.report(again) == entry["report"]


_CHILD = r"""
import json
import cleave
from cleave.pipeline import update
from cleave.pipeline.update import engine, proof
from cleave.pipeline.update.instantiate import Recipe
from tests.update._programs import flash_rows, identity, lowered, the_obstacle
from tests.update.test_store import SHAPES

def run(*args, **kwargs):
    raise AssertionError("derived again")

engine.run = run
calls = []
real = proof.prove
def counted(*args, **kwargs):
    calls.append(args[0])
    return real(*args, **kwargs)
proof.prove = counted
graph = identity(flash_rows, SHAPES)
ir = lowered(graph)
ob = the_obstacle(ir)
d = update.derive(ir, ob, Recipe.of(graph))
print("RESULT " + json.dumps({"from_store": d.from_store, "rules": d.rules, "refused": d.refused,
                              "proofs": len(calls)}))
"""


def test_a_second_process_derives_nothing(fresh):
    ir, ob, recipe = setting()
    first = update_package.derive(ir, ob, recipe)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CLEAVE_CACHE_ROOT=str(fresh),
               PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "python"),
                                           os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, "-c", _CHILD], cwd=fresh, env=env, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    (line,) = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
    got = json.loads(line[len("RESULT "):])
    assert got == {"from_store": True, "rules": json.loads(json.dumps(first.rules)),
                   "refused": "", "proofs": len(ob.loops)}


def test_a_tampered_rule_is_proved_wrong_and_derived_again(fresh, monkeypatch):
    ir, ob, recipe = setting()
    first = update_package.derive(ir, ob, recipe)
    path = store.path_of(first.key)
    entry = json.loads(path.read_text())
    entry["rules"][0][1] = ["exp", ["sub", "m'", "m"]]
    path.write_text(json.dumps(entry))
    again = derive_again(ir, ob, recipe, monkeypatch)
    assert not again.from_store and again.rules == first.rules
    assert json.loads(path.read_text())["rules"] == json.loads(json.dumps(first.rules))  # overwritten


E = ["exp", ["sub", "m", "m'"]]


@pytest.mark.parametrize("factor", [
    ["sub", ["add", "a", E], "a"],  # proved over the reals, but reads the accumulator
    ["div", ["exp", "m"], ["exp", "m'"]],  # proved, emittable, but overflows: never ranked
    5,  # malformed: no expression at all
    "x",  # malformed: a leaf the factor cannot read
    ["sqrt", "m"],  # an op the search never forms
    ["exp", ["sub", "m", "m'"], "m"],  # the wrong arity
], ids=["reads_a", "unstable", "int", "unknown_leaf", "unknown_op", "arity"])
def test_a_stored_factor_the_engine_would_not_choose_is_derived_again(fresh, monkeypatch,
                                                                      factor):
    """Re-proving is not enough: a factor that reads ``a``, that the ranking would have
    rejected, or that is not an expression at all is a miss -- derived again and
    overwritten -- never the record, and never an E13 that pins the loop structure."""
    ir, ob, recipe = setting()
    first = update_package.derive(ir, ob, recipe)
    path = store.path_of(first.key)
    entry = json.loads(path.read_text())
    entry["rules"][0][1] = factor
    path.write_text(json.dumps(entry))
    assert store.load(first.key, ir, ob) is None
    runs = []
    real = engine.run

    def counted(*args, **kwargs):
        runs.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "run", counted)
    again = derive_again(ir, ob, recipe, monkeypatch)
    assert not again.bug and not again.refused and not again.from_store and runs == [1]
    assert again.rules == first.rules
    assert json.loads(path.read_text())["rules"] == json.loads(json.dumps(first.rules))
    assert derive_again(ir, ob, recipe, monkeypatch).from_store  # the overwrite is read back


def test_the_stored_factor_checks_accept_what_the_engine_chooses(fresh):
    ir, ob, recipe = setting()
    first = update_package.derive(ir, ob, recipe)
    assert all(store.well_formed(f) for _, f in first.rules)
    assert not store.well_formed(("exp", ("sub", "m", "a")))
    assert not store.well_formed(("const", 1.0))
    assert not store.well_formed(())


@pytest.mark.parametrize("damage", ["key", "unreadable", "loops"])
def test_an_entry_that_is_not_this_graphs_is_a_miss(fresh, monkeypatch, damage):
    ir, ob, recipe = setting()
    first = update_package.derive(ir, ob, recipe)
    path = store.path_of(first.key)
    entry = json.loads(path.read_text())
    if damage == "unreadable":
        path.write_text(path.read_text()[:-7])
    else:
        entry[damage] = "another" if damage == "key" else list(reversed(entry["loops"]))
        path.write_text(json.dumps(entry))
    again = derive_again(ir, ob, recipe, monkeypatch)
    assert not again.from_store and again.rules == first.rules
    assert json.loads(path.read_text())["key"] == first.key


def test_a_stored_refusal_is_returned_without_a_search(fresh, monkeypatch):
    ir, ob, recipe = setting(column_weighted, dict(M=8, K=64, N=16))
    first = update_package.derive(ir, ob, recipe)
    assert first.refused.endswith("do not lead its accumulators")
    # E6 is read off the references: it is kept for this recipe, not for the loop structure
    assert first.recipe_key == recipe.key and not store.path_of(first.key).exists()
    entry = json.loads(store.path_of(first.key, recipe.key).read_text())
    assert entry["refused"] == first.refused and entry["recipe"] == recipe.key
    no_search(monkeypatch)
    calls = proofs(monkeypatch)
    again = derive_again(ir, ob, recipe, monkeypatch)
    assert again.from_store and again.refused == first.refused and not again.rules
    assert calls == []


def literal_rows(x: cl.Tensor["R", 64], v: cl.Tensor[64, "D"]):
    """``flash_rows`` with its looped dim written as a literal: the loop structure of
    ``flash_rows`` at C=64 (the loop key has no extents), but nothing to resize."""
    e = cl.exp(x - _max(x))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def rsqrt_literal(x: cl.Tensor["R", 64], v: cl.Tensor[64, "D"]):
    p = cl.rsqrt(cl.exp(x - _max(x)))
    return (p @ v) / cl.sum(p, -1, keepdim=True)


def test_a_refusal_of_one_recipe_is_not_another_programs(fresh, monkeypatch):
    """E1 is a fact about ``literal_rows``' recipe, not about the loop structure it shares
    with ``flash_rows``: neither the memo nor the store hands it to ``flash_rows``, which
    derives its own rule. A rule is the loop structure's: ``literal_rows`` then gets it,
    proved again on its own graph."""
    lit_ir, lit_ob, lit = setting(literal_rows, dict(R=16, D=8))
    ir, ob, recipe = setting()
    key = loop_key(ir, ob.full, ob.loops)
    assert loop_key(lit_ir, lit_ob.full, lit_ob.loops) == key and lit.key != recipe.key
    e1 = texts.E1.format(full=ob.full)
    refusal = update_package.derive(lit_ir, lit_ob, lit)
    assert refusal.refused == e1 and refusal.recipe_key == lit.key
    assert store.path_of(key, lit.key).exists() and not store.path_of(key).exists()

    d = update_package.derive(ir, ob, recipe)  # one process
    assert not d.refused and not d.from_store and d.rules == tuple((a, FACTOR) for a in ob.loops)

    store.path_of(key).unlink()  # a second process, whose store holds the refusal alone
    monkeypatch.setattr(update_package, "_DERIVED", {})
    again = update_package.derive(lit_ir, lit_ob, lit)
    assert again.from_store and again.refused == e1
    mine = update_package.derive(ir, ob, recipe)
    assert not mine.from_store and mine.rules == d.rules

    monkeypatch.setattr(update_package, "_DERIVED", {})  # a third: the rule is looked up first
    no_search(monkeypatch)
    calls = proofs(monkeypatch)
    shared = update_package.derive(lit_ir, lit_ob, lit)
    assert shared.from_store and not shared.refused and shared.rules == d.rules
    assert len(calls) == len(ob.loops)  # proved on literal_rows' graph


def test_a_refusal_of_the_flat_graph_alone_holds_for_the_loop_structure(fresh, monkeypatch):
    """E7 is read off the flat graph before the recipe is used: it is the loop structure's,
    and another program of that structure gets it without a search."""
    from .test_refusals import rsqrt_on_the_path

    ir, ob, recipe = setting(rsqrt_on_the_path)
    first = update_package.derive(ir, ob, recipe)
    assert first.refused.endswith("only elementwise add, sub, mul, div and exp are read there")
    assert first.recipe_key == ""
    assert json.loads(store.path_of(first.key).read_text())["recipe"] == ""
    lit_ir, lit_ob, lit = setting(rsqrt_literal, dict(R=16, D=8))
    assert loop_key(lit_ir, lit_ob.full, lit_ob.loops) == first.key and lit.key != recipe.key
    no_search(monkeypatch)
    assert update_package.derive(lit_ir, lit_ob, lit) is first  # the memo
    again = derive_again(lit_ir, lit_ob, lit, monkeypatch)
    assert again.from_store and again.refused == first.refused


def test_each_loop_structure_has_an_entry_of_its_own(fresh, monkeypatch):
    from ._programs import column_average

    settings = [setting(), setting(column_average, dict(H=2, K=64, N=8))]
    firsts = [update_package.derive(*s) for s in settings]
    assert all(not d.refused for d in firsts) and firsts[0].key != firsts[1].key
    paths = [store.path_of(d.key) for d in firsts]
    assert paths[0] != paths[1] and all(p.exists() for p in paths)
    monkeypatch.setattr(update_package, "_DERIVED", {})
    no_search(monkeypatch)
    for s, d in zip(settings, firsts):
        again = update_package.derive(*s)
        assert again.from_store and again.rules == d.rules


def test_the_key_moves_with_the_engine_sources_the_settings_and_the_build(tmp_path, monkeypatch):
    key = "0" * 64
    here = store.path_of(key)
    copies = []
    for p in store.engine_sources():
        copy = tmp_path / p.parent.name / p.name
        copy.parent.mkdir(exist_ok=True)
        shutil.copy(p, copy)
        copies.append(copy)
    assert store.engine_digest(copies) == store.ENGINE_DIGEST
    assert {p.name for p in copies} >= {"update_rule.py", "engine.py", "search.py", "proof.py",
                                        "rank.py", "reference.py", "instantiate.py", "store.py",
                                        "texts.py", "__init__.py"}
    copies[3].write_text(copies[3].read_text() + "\n# one more line\n")
    edited = store.engine_digest(copies)
    assert edited != store.ENGINE_DIGEST
    monkeypatch.setattr(store, "ENGINE_DIGEST", edited)
    assert store.path_of(key) != here
    monkeypatch.undo()

    assert store.path_of(key) == here
    monkeypatch.setattr(cache, "build_id", lambda: "another build")
    assert store.path_of(key) != here
    monkeypatch.undo()

    from cleave.pipeline.update import search

    monkeypatch.setattr(search, "LOOSE_RTOL", 3e-2)
    assert store.path_of(key) != here


def test_the_process_keeps_a_light_derivation(fresh):
    """What ``derive`` keeps per loop structure has no small-shape instances, references or
    blocks; the report reads none of them."""
    ir, ob, recipe = setting()
    d = update_package.derive(ir, ob, recipe)
    assert d.flat is None and d.kernel is None and d.references == []
    assert d.holes and all(h.leaves == {} and h.target is None for h in d.holes)
    assert all(h.search is not None and h.verdicts for h in d.holes)
    full = engine.run(ir, ob, recipe, key=d.key)
    assert full.flat is not None and engine.report(full) == engine.report(d)


def test_lowering_reads_the_store(fresh, monkeypatch):
    """``plans.lower``: the first lowering derives, a later one in a
    fresh process (here: memo) reads the record back."""
    graph = identity(flash_rows, SHAPES)
    (ir,) = plans.lower(graph)
    (d,) = ir.derivations
    assert not d.from_store and ir.streaming_refusals == []
    no_search(monkeypatch)
    monkeypatch.setattr(update_package, "_DERIVED", {})
    (again,) = plans.lower(graph)
    (d2,) = again.derivations
    assert d2.from_store
    (full,) = [n for n in again.nodes if getattr(n, "streaming", None) is not None]
    assert full.streaming == d.record()
