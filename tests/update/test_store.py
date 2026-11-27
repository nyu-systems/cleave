"""The derivation store: one entry per loop structure on disk, every stored rule tested again
on load (exactly, on the graph's own test data, and in the float replay), a stored refusal
trusted, the key moving with the engine. CPU only; every test has a cache root of its own.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import cleave
import cleave.lang as cl
from cleave.ir.ops import ReduceMax
from cleave.ir.update_rule import loop_key
from cleave.pipeline import cache, plans
from cleave.pipeline import update as update_package
from cleave.pipeline.update import engine, reference, store, texts
from cleave.pipeline.update.instantiate import Recipe

from ._attention import FACTOR
from ._programs import _max, flash_rows, identity, lowered, the_obstacle

ROOT = Path(__file__).resolve().parents[2]
SHAPES = dict(R=16, C=64, D=8)


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """An empty store and an empty in-process memo."""
    monkeypatch.setattr(update_package, "_DERIVED", {})
    with cleave.Session(cache_root=tmp_path):
        yield tmp_path


def setting(fn=flash_rows, shapes=SHAPES, cls=None):
    graph = identity(fn, shapes)
    ir = lowered(graph)
    return ir, the_obstacle(ir, cls), Recipe.of(graph)


def derive_again(ir, ob, recipe, monkeypatch):
    monkeypatch.setattr(update_package, "_DERIVED", {})
    return update_package.derive(ir, ob, recipe)


def no_search(monkeypatch):
    def run(*args, **kwargs):
        raise AssertionError("derived again")

    monkeypatch.setattr(engine, "run", run)


def retests(monkeypatch):
    calls = []
    real = store.retest

    def counted(flat_graph, obstacle, recipe, rules):
        calls.append(rules)
        return real(flat_graph, obstacle, recipe, rules)

    monkeypatch.setattr(store, "retest", counted)
    return calls


def test_a_derivation_is_stored_and_read_back_with_every_rule_tested_again(fresh, monkeypatch):
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
    calls = retests(monkeypatch)
    again = derive_again(ir, ob, recipe, monkeypatch)
    assert again.from_store and again.rules == first.rules and again.record() == first.record()
    assert calls == [first.rules]
    assert engine.report(again) == entry["report"]


_CHILD = r"""
import json
import cleave
from cleave.pipeline import update
from cleave.pipeline.update import engine, store
from cleave.pipeline.update.instantiate import Recipe
from tests.update._programs import flash_rows, identity, lowered, the_obstacle
from tests.update.test_store import SHAPES

def run(*args, **kwargs):
    raise AssertionError("derived again")

engine.run = run
calls = []
real = store.retest
def counted(*args):
    calls.append(1)
    return real(*args)
store.retest = counted
graph = identity(flash_rows, SHAPES)
ir = lowered(graph)
ob = the_obstacle(ir)
d = update.derive(ir, ob, Recipe.of(graph))
print("RESULT " + json.dumps({"from_store": d.from_store, "rules": d.rules, "refused": d.refused,
                              "retests": len(calls)}))
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
                   "refused": "", "retests": 1}


E = ["exp", ["sub", "m", "m'"]]


@pytest.mark.parametrize("factor", [
    ["exp", ["sub", "m'", "m"]],  # well formed and emittable, but reproduces nothing
    ["sub", ["add", "a", E], "a"],  # reads the accumulator: no factor
    ["div", ["exp", "m"], ["exp", "m'"]],  # reproduces the graph, but overflows: never ranked
    5,  # malformed: no expression at all
    "x",  # malformed: a leaf the factor cannot read
    ["sqrt", "m"],  # an op of the search, but no rule of this graph
    ["log", "m"],  # an op the search never forms
    ["exp", ["sub", "m", "m'"], "m"],  # the wrong arity
], ids=["wrong", "reads_a", "unstable", "int", "unknown_leaf", "other_op", "unknown_op",
        "arity"])
def test_a_stored_factor_the_engine_would_not_choose_is_derived_again(fresh, monkeypatch,
                                                                      factor):
    """A factor that reproduces nothing, that reads ``a``, that the replay rejects or that
    is not an expression at all is a miss -- derived again and overwritten -- never the
    record, and never a BUG that pins the loop structure."""
    ir, ob, recipe = setting()
    first = update_package.derive(ir, ob, recipe)
    path = store.path_of(first.key)
    entry = json.loads(path.read_text())
    entry["rules"][0][1] = factor
    path.write_text(json.dumps(entry))
    assert store.load(first.key, ir, ob, recipe) is None
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


def traced_rows(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """The traced order: the value matmul reads the normalized exp."""
    e = cl.exp(x - _max(x))
    return (e / cl.sum(e, -1, keepdim=True)) @ v


def test_a_stored_refusal_is_returned_without_a_search(fresh, monkeypatch):
    ir, ob, recipe = setting(traced_rows, cls=ReduceMax)
    first = update_package.derive(ir, ob, recipe)
    assert first.refused.startswith("no update of ")
    # a refusal is kept for this recipe, not for the loop structure
    assert first.recipe_key == recipe.key and not store.path_of(first.key).exists()
    entry = json.loads(store.path_of(first.key, recipe.key).read_text())
    assert entry["refused"] == first.refused and entry["recipe"] == recipe.key
    no_search(monkeypatch)
    calls = retests(monkeypatch)
    again = derive_again(ir, ob, recipe, monkeypatch)
    assert again.from_store and again.refused == first.refused and not again.rules
    assert calls == []


def literal_rows(x: cl.Tensor["R", 64], v: cl.Tensor[64, "D"]):
    """``flash_rows`` with its looped dim written as a literal: the loop structure of
    ``flash_rows`` at C=64 (the loop key has no extents), but nothing to resize."""
    e = cl.exp(x - _max(x))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def test_a_refusal_of_one_recipe_is_not_another_programs(fresh, monkeypatch):
    """"No test" is a fact about ``literal_rows``' recipe, not about the loop structure it
    shares with ``flash_rows``: neither the memo nor the store hands it to ``flash_rows``,
    which derives its own rule. That rule is the loop structure's, but it cannot be tested
    again on ``literal_rows``' graph (there is no prefix of a literal dim): ``literal_rows``
    keeps its refusal."""
    lit_ir, lit_ob, lit = setting(literal_rows, dict(R=16, D=8))
    ir, ob, recipe = setting()
    key = loop_key(ir, ob.full, ob.loops)
    assert loop_key(lit_ir, lit_ob.full, lit_ob.loops) == key and lit.key != recipe.key
    refusal = update_package.derive(lit_ir, lit_ob, lit)
    assert refusal.refused == texts.NO_TEST.format(
        full=ob.full, why="the dim it reduces has no symbol: the graph cannot be built "
        "again on a prefix of it")
    assert refusal.recipe_key == lit.key
    assert store.path_of(key, lit.key).exists() and not store.path_of(key).exists()

    d = update_package.derive(ir, ob, recipe)  # one process
    assert not d.refused and not d.from_store and d.rules == tuple((a, FACTOR) for a in ob.loops)

    monkeypatch.setattr(update_package, "_DERIVED", {})  # a second: the rule is looked up first
    again = update_package.derive(lit_ir, lit_ob, lit)
    assert again.from_store and again.refused == refusal.refused  # ... and cannot be tested


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
    assert {p.name for p in copies} == {"update_rule.py", "engine.py", "field.py", "search.py",
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

    monkeypatch.setattr(reference, "DRAWS", (1, 2, 3))
    assert store.path_of(key) != here


def test_the_process_keeps_a_light_derivation(fresh):
    """What ``derive`` keeps per loop structure has no instances or test data; the report
    reads none of them."""
    ir, ob, recipe = setting()
    d = update_package.derive(ir, ob, recipe)
    assert d.flat is None and d.kernel is None and d.references == []
    assert d.holes and all(h.search is not None and h.tried for h in d.holes)
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
