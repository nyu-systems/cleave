"""Every reason a conflict gets no rule, with its text: there is no test (the flat graph
cannot be evaluated on prefixes of the looped dim), no update reproduces the program's own
states, or every one that does fails the float replay; and a bug, which names itself. Each
only keeps the pin. CPU only.
"""

import pytest

pytest.importorskip("torch")

import cleave
import cleave.lang as cl
from cleave.ir.ops import ReduceMax
from cleave.ir.update_rule import loop_key
from cleave.pipeline import plans
from cleave.pipeline import update as update_package
from cleave.pipeline.update import engine, field, store, texts
from cleave.pipeline.update.instantiate import Recipe

from ._attention import FACTOR
from ._programs import _max, divided_rows, identity, lowered, the_obstacle

SHAPES = dict(R=16, C=64, D=8)


def refused(fn, shapes=SHAPES, cls=None, **kw):
    """``(flat graph, conflict, derivation)`` of ``fn``'s identity graph."""
    graph = identity(fn, shapes)
    ir = lowered(graph)
    ob = the_obstacle(ir, cls)
    d = engine.run(ir, ob, Recipe.of(graph), key=loop_key(ir, ob.full, ob.loops), **kw)
    return ir, ob, d


# ------------------------------------------------------------------------------- programs


def flash_rows_explicit(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    e = cl.exp(x - _max(x))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def literal_rows(x: cl.Tensor["R", 64], v: cl.Tensor[64, "D"]):
    """The looped dim written as a literal: nothing to build again on a prefix."""
    e = cl.exp(x - _max(x))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def literal_five(x: cl.Tensor["R", 5], v: cl.Tensor[5, "D"]):
    """A literal looped dim whose extent is the probe's for the first symbol (D: 5)."""
    e = cl.exp(x - _max(x))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def square_rows(x: cl.Tensor["S", "S"], v: cl.Tensor["S", "D"]):
    """An input that carries the looped dim on two axes: no prefix of the loop is one of it."""
    e = cl.exp(x - _max(x))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def self_attention(q: cl.Tensor["H", "S", "D"], k: cl.Tensor["H", "S", "D"],
                   v: cl.Tensor["H", "S", "D"]):
    """Queries and keys written with one dim name: the looped symbol also sizes the rows."""
    s = (q @ k.mT) / q.shape[-1] ** 0.5
    e = cl.exp(s - _max(s))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def traced_rows(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """The traced order: the value matmul reads the finished row sum."""
    e = cl.exp(x - _max(x))
    return (e / cl.sum(e, -1, keepdim=True)) @ v


def centered_max(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """``exp(x - max(x - mean(x)))``: the max is taken of a value that depends on the
    prefix through the mean, so it is no running max of the looped dim."""
    e = cl.exp(x - _max(x - cl.mean(x, -1, keepdim=True)))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def max_of_a_softmax(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """``g = exp(x - max(exp(x - max(x))))``: ``max(exp(x - max(x)))`` is 1 on every prefix."""
    e = cl.exp(x - _max(x))
    g = cl.exp(x - _max(e))
    return (g @ v) / cl.sum(g, -1, keepdim=True)


def nested_softmax(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """A softmax of a softmax: what the second one sums changes with the first max."""
    e = cl.exp(x - _max(x))
    f = cl.exp(e - _max(e))
    return (f @ v) / cl.sum(f, -1, keepdim=True)


def column_weighted(v: cl.Tensor["M", "K"], x: cl.Tensor["K", "N"]):
    """``(v @ e) / sum(e, 0)``: the factor of a row of the max lives on the accumulator's
    TRAILING axis."""
    e = cl.exp(x - _max(x, 0))
    return (v @ e) / cl.sum(e, 0, keepdim=True)


def rsqrt_on_the_path(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    p = cl.rsqrt(cl.exp(x - _max(x)))
    return (p @ v) / cl.sum(p, -1, keepdim=True)


def divided_by_an_input(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    e = cl.exp((x - _max(x)) / y)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def times_its_max(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    m = _max(x)
    p = cl.exp(x - m) * m
    return (p @ v) / cl.sum(p, -1, keepdim=True)


def max_normalized(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """``(e @ v) / max(e)``: a max of the scaled value; it cannot be rescaled from zero."""
    e = cl.exp(x - _max(x))
    return (e @ v) / _max(e)


def overflowing(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """Attention whose output does not fit float16: every rule is non-finite there."""
    e = cl.exp(x - _max(x))
    return (e @ v) / cl.sum(e, -1, keepdim=True) * 60000.0 * 60000.0


# ------------------------------------------------------------------------------- no test


def test_no_test_where_the_looped_dim_has_no_symbol():
    ir, ob, d = refused(literal_rows, dict(R=16, D=8))
    assert d.refused == texts.NO_TEST.format(
        full=ob.full, why="the dim it reduces has no symbol: the graph cannot be built again "
        "on a prefix of it")
    assert not d.references and not d.bug and d.recipe_key == Recipe.of(identity(
        literal_rows, dict(R=16, D=8))).key  # the recipe's, not the loop structure's


def test_no_test_where_a_literal_dim_has_a_probes_extent():
    """The probe gives D the extent 5, and the literal looped dim is 5 too: one more
    lowering with another extent for D shows that the dim does not follow it."""
    from cleave.pipeline.update.reference import probe_looped

    graph = identity(literal_five, dict(R=16, D=8))
    ir, ob, d = refused(literal_five, dict(R=16, D=8))
    recipe = Recipe.of(graph)
    assert recipe.looped(ir, ob.full) is None and probe_looped(recipe, ob.full) is None
    assert d.refused.startswith(f"{ob.full} has no test on prefixes") and not d.bug


def test_no_test_where_an_input_carries_the_looped_dim_twice():
    ir, ob, d = refused(square_rows, dict(S=64, D=8))
    assert d.refused.startswith(f"{ob.full} has no test on prefixes of the dim it reduces: ")
    assert "grows along 2 axes with that dim" in d.refused and not d.bug


def test_no_test_where_the_looped_symbol_also_sizes_the_state():
    """``q, k, v: [H, S, D]``: a prefix of S cuts the queries too, so the state of the loop
    changes shape from one prefix to the next."""
    ir, ob, d = refused(self_attention, dict(H=4, S=256, D=16))
    assert d.refused.startswith(f"{ob.full} has no test on prefixes of the dim it reduces: ")
    assert "changes shape with the looped dim" in d.refused and not d.bug


def test_no_test_for_an_op_the_field_gives_no_meaning(monkeypatch):
    from cleave.ir.ops import Exp

    field.semantics(next(n for n in lowered(identity(flash_rows_explicit, SHAPES)).nodes))
    monkeypatch.delitem(field._SEMANTICS, Exp)
    ir, ob, d = refused(flash_rows_explicit)
    assert d.refused.startswith(f"{ob.full} has no test on prefixes of the dim it reduces: ")
    assert "(Exp) has no field semantics" in d.refused and not d.bug


# ------------------------------------------------------------------------------- no rule


@pytest.mark.parametrize("fn, shapes, cls", [
    (traced_rows, SHAPES, ReduceMax),  # the matmul needs the finished row sum
    (column_weighted, dict(M=8, K=64, N=16), None),  # the factor is on another axis
    (divided_by_an_input, SHAPES, None),  # the factor would depend on the input
    (times_its_max, SHAPES, None),  # ... on the max itself, times the exp
    (rsqrt_on_the_path, SHAPES, None),  # ... on half the exponent
    (max_normalized, SHAPES, None),  # a max accumulates under the max
], ids=["traced", "column_weighted", "divided_by_an_input", "times_its_max", "rsqrt", "max_of"])
def test_no_update_reproduces_the_programs_states(fn, shapes, cls):
    ir, ob, d = refused(fn, shapes, cls)
    assert d.refused.startswith("no update of ") and not d.bug
    acc = d.holes[-1].acc
    assert d.refused == texts.NO_RULE.format(acc=acc, budget=5)
    assert d.refused == (f"no update of {acc} of the form 'accumulator * factor + tile' within "
                         "5 ops reproduces the flat graph on its prefixes")


def test_a_rule_is_found_where_one_exists():
    """``exp(x - max(x)) / w`` (the factor does not read ``w``) and a max that is 1 on every
    prefix (any factor of a max that never moves is 1, and ``exp(m - m')`` is)."""
    for fn, shapes, cls in ((divided_rows, SHAPES, None), (max_of_a_softmax, SHAPES, ReduceMax)):
        ir, ob, d = refused(fn, shapes) if cls is None else _second(fn, shapes)
        assert not d.refused and d.rules == tuple((acc, FACTOR) for acc in ob.loops)


def _second(fn, shapes):
    """The second conflict of ``fn`` (the max under the first)."""
    from tests._explicit_softmax import obstacles_of

    graph = identity(fn, shapes)
    ir = lowered(graph)
    ob = obstacles_of(ir)[-1]
    return ir, ob, engine.run(ir, ob, Recipe.of(graph), key=loop_key(ir, ob.full, ob.loops))


# ----------------------------------------------------------------------------- no replay


def test_every_rule_fails_the_replay_where_the_output_does_not_fit():
    ir, ob, d = refused(overflowing)
    assert d.refused == texts.NO_REPLAY.format(loops=", ".join(ob.loops),
                                               case="non-finite output on normal@32")


def test_the_replay_catches_a_max_that_is_no_running_max():
    """The field data reproduce ``a * exp(m - m')`` (it takes ``m'`` from the program); the
    kernel would carry ``max(m, max of the tile)``, which is not the program's here."""
    ir, ob, d = _second(centered_max, SHAPES)
    assert d.refused == texts.NO_REPLAY.format(
        loops=", ".join(ob.loops), case=f"the carried {ob.full} is not the flat graph's on "
        "normal@32")


def test_the_replay_catches_what_the_field_does_not_see():
    """An exp of an exp reads only the inner exp's residue mod Q, which is 0: in the field
    the second softmax does not depend on the first max, and ``a * exp(m - m')`` reproduces
    it. In floating point it is another value."""
    ir, ob, d = _second(nested_softmax, SHAPES)
    assert d.refused.startswith(f"every update of {', '.join(ob.loops)} that reproduces the "
                                "flat graph fails the float replay (an output ")
    assert "off the flat graph's, 64x its resolution on normal@32)" in d.refused


# ----------------------------------------------------------------------------------- bug


def test_a_bug_in_the_search_keeps_the_pin_and_names_itself(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("boom\nsecond line")

    monkeypatch.setattr(engine.search_module, "enumerate_factors", boom)
    ir, ob, d = refused(flash_rows_explicit)
    assert d.refused == texts.BUG.format(type="RuntimeError", message="boom")
    assert d.refused == "RuntimeError: boom (a bug in the update search)" and d.bug

    # ... and it never leaves ``plans.lower``: the graph keeps its pin and says why
    monkeypatch.setattr(update_package, "_DERIVED", {})
    graph = identity(flash_rows_explicit, SHAPES)
    with cleave.Session(cache_root=tmp_path):
        (ir,) = plans.lower(graph)
        assert ir.streaming_refusals == [f"{ob.full}: RuntimeError: boom (a bug in the update search)"]
        assert not [n for n in ir.nodes if getattr(n, "streaming", None) is not None]
        assert not store.path_of(loop_key(ir, ob.full, ob.loops)).exists()  # never stored
    assert update_package._DERIVED == {}  # nor kept


def test_a_graph_made_without_a_program_keeps_its_pin(tmp_path, monkeypatch):
    """A searched graph built by hand, with no program and no symbol map: there is nothing
    to build it again from, and lowering says so instead of failing."""
    from cleave.pipeline.superopt import SearchedGraph

    graph = identity(flash_rows_explicit, SHAPES)
    bare = SearchedGraph(0, graph.structure, None, False, None)
    monkeypatch.setattr(update_package, "_DERIVED", {})
    with cleave.Session(cache_root=tmp_path):
        (ir,) = plans.lower(bare)
        ob = the_obstacle(ir)
        (why,) = ir.streaming_refusals
        assert why.startswith(f"{ob.full}: {ob.full} has no test on prefixes")
        assert not [n for n in ir.nodes if getattr(n, "streaming", None) is not None]
        # kept for this recipe only: the graph with a program (same loop structure) derives
        key, recipe = loop_key(ir, ob.full, ob.loops), Recipe.of(bare)
        assert store.path_of(key, recipe.key).exists() and not store.path_of(key).exists()
        (served,) = plans.lower(graph)
    assert served.streaming_refusals == [] and loop_key(served, ob.full, ob.loops) == key
