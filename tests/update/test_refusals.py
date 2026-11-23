"""Every reason the engine gives an obstacle no rule, with its text (DESIGN 4.3; 13.2 f).

Each program records its max explicitly and has one obstacle that passes the structural
rules of ``schedule.streaming.obstacles``; the engine refuses it, and the refusal is the
reason the pin stays. CPU only.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("z3")

import cleave
import cleave.lang as cl
from cleave.ir.ops import Div, Matmul, ReduceSum, Sub
from cleave.ir.update_rule import loop_key, use_searched_updates
from cleave.pipeline import plans
from cleave.pipeline import update as update_package
from cleave.pipeline.update import engine, store, texts
from cleave.pipeline.update.instantiate import Recipe

from ._programs import _max, identity, lowered, scaled_rows, the_obstacle, two_softmaxes

SHAPES = dict(R=16, C=64, D=8)


def refused(fn, shapes=SHAPES, edit=None, **kw):
    """``(flat graph, obstacle, derivation)`` of ``fn``'s identity graph."""
    graph = identity(fn, shapes)
    ir = lowered(graph)
    if edit is not None:
        edit(ir)
    ob = the_obstacle(ir)
    with use_searched_updates():
        d = engine.run(ir, ob, Recipe.of(graph), key=loop_key(ir, ob.full, ob.loops), **kw)
    return ir, ob, d


def name_of(ir, cls, nth=0):
    return [n.name for n in ir.nodes if type(n) is cls][nth]


# ------------------------------------------------------------------------------- programs


def centered_max(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """``exp(x - max(x - mean(x)))``: the max is taken of a value that depends on the
    prefix through the mean, so it is no running max of the looped dim."""
    e = cl.exp(x - _max(x - cl.mean(x, -1, keepdim=True)))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def max_of_a_softmax(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """``g = exp(x - max(exp(x - max(x))))``: ``max(exp(x - max(x)))`` is 1 on every prefix
    (the inputs hold float16 values, read at one precision), so it never moves."""
    e = cl.exp(x - _max(x))
    g = cl.exp(x - _max(e))
    return (g @ v) / cl.sum(g, -1, keepdim=True)


def nested_softmax(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """A softmax of a softmax: what the second one sums is computed from the first max."""
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


# -------------------------------------------------------------------------------- refusals


def test_e5_a_max_of_a_value_that_depends_on_the_prefix():
    ir, ob, d = refused(centered_max)
    assert d.refused == texts.E5.format(full=ob.full, label="normal seed=1 prefix=4")
    assert d.refused == (f"the value of {ob.full} over a prefix and over the next tile do not "
                         "combine to its value over both (normal seed=1 prefix=4)")
    assert not d.holes


def test_e10_a_max_that_never_moves():
    """No reference row shows the move of a max that is 1 on every prefix: E10, before any
    expression is searched (and every one would match)."""
    ir, ob, d = refused(max_of_a_softmax)
    assert len(d.cases) == 11
    assert all(moved == (1.0 if label.startswith("first") else 0.0) for label, moved in d.cases)
    assert d.adequacy == {acc: 0 for acc in ob.loops}
    rows = d.references[0].rows
    assert d.refused == texts.E10.format(full=ob.full, acc=ob.loops[0], k=0, n=rows)
    assert d.refused == (f"the references do not show the move of {ob.full} and the old value "
                         f"of {ob.loops[0]} in the same rows (0 rows, {rows} needed)")
    assert not d.holes


def test_e6_an_accumulator_that_does_not_lead_with_the_rows():
    ir, ob, d = refused(column_weighted, dict(M=8, K=64, N=16))
    acc = name_of(ir, Matmul)
    assert d.refused == texts.E6.format(acc=acc, full=ob.full)
    assert d.refused == (f"{acc} of one row changes with {ob.full} of another row: the rows of "
                         f"{ob.full} do not lead its accumulators")
    assert not d.references  # decided before the reference set is built


def test_e7_an_op_the_proof_does_not_read_costs_nothing():
    ir, ob, d = refused(rsqrt_on_the_path)
    (node,) = [n for n in ir.nodes if type(n).__name__ in ("RSqrt", "Sqrt")]
    acc = ob.loops[0]
    assert d.refused == texts.E7.format(node=node.name, cls=type(node).__name__, full=ob.full,
                                        acc=acc)
    assert d.refused.endswith("only elementwise add, sub, mul, div and exp are read there")
    assert d.flat is None and not d.references and not d.timings
    assert d.seconds < 0.5


def test_e8_a_denominator_floor_is_not_modelled():
    def floor(ir):
        next(n for n in ir.nodes if type(n) is Div and n.const is None
             and any(type(e.src_node) is Sub for e in n.inputs)).denom_floor = 1e-6

    ir, ob, d = refused(divided_by_an_input, edit=floor)
    div = next(n for n in ir.nodes if getattr(n, "denom_floor", None) is not None)
    assert d.refused == texts.E8.format(node=div.name, attr="denom_floor", value=1e-06)
    assert d.refused == f"{div.name} has denom_floor=1e-06, which the proof does not model"


def test_e8_an_attribute_no_list_classifies_is_not_modelled(monkeypatch):
    monkeypatch.setattr(Sub, "_macro_attrs", Sub._macro_attrs + ("mystery",), raising=False)
    monkeypatch.setattr(Sub, "mystery", 3, raising=False)
    ir, ob, d = refused(times_its_max)
    sub = name_of(ir, Sub)
    assert d.refused == f"{sub} has mystery=3, which the proof does not model"


def test_e9_two_values_of_the_max_that_are_not_indexed_alike():
    ir, ob, d = refused(times_its_max)
    from cleave.ir.ops import Mul

    mul = next(n for n in ir.nodes if type(n) is Mul and n.broadcast != -1)
    assert d.refused == texts.E9.format(node=mul.name, full=ob.full)
    assert d.refused == (f"{mul.name} combines two values computed from {ob.full} that are "
                         "not indexed alike")


def test_e11_no_rule_within_the_budgets_and_how_far_the_search_got():
    """``exp((x - max(x)) / 8)``: the right factor ``exp((m - m') / 8)`` needs a constant,
    which the search never forms. The narrowest hole is searched at 4, then at 5 ops."""
    ir, ob, d = refused(scaled_rows)
    (hole,) = d.holes
    assert hole.acc == name_of(ir, ReduceSum) and [s.budget for s, _ in hole.tried] == [4, 5]
    assert d.budget == 5 and not hole.emittable
    assert d.refused == texts.E11.format(acc=hole.acc, budget=5, matched=len(hole.search.matches),
                                         proved=len(hole.proven))
    assert d.refused.startswith(f"no proven update of {hole.acc} of the form 'accumulator * "
                                "factor + tile' within 5 ops (")
    assert not d.ranked


def test_e12_every_proven_rule_is_non_finite_on_a_case():
    ir, ob, d = refused(overflowing)
    assert [h.emittable != [] for h in d.holes] == [True, True]
    assert d.rejected and not d.ranked
    assert d.refused == texts.E12.format(loops=", ".join(ob.loops), case="normal@32")
    assert all(why == "non-finite output on normal@32" for _, why in d.rejected)


def test_e13_a_bug_in_the_search_keeps_the_pin_and_names_itself(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("boom\nsecond line")

    monkeypatch.setattr(engine.search_module, "enumerate_updates", boom)
    ir, ob, d = refused(flash_rows_explicit)
    assert d.refused == texts.E13.format(type="RuntimeError", message="boom")
    assert d.refused == "RuntimeError: boom (a bug in the update search)" and d.bug

    # ... and it never leaves ``plans.lower``: the graph keeps its pin and says why
    monkeypatch.setattr(update_package, "_DERIVED", {})
    graph = identity(flash_rows_explicit, SHAPES)
    with use_searched_updates(), cleave.Session(cache_root=tmp_path):
        (ir,) = plans.lower(graph)
        assert ir.streaming_refusals == [f"{ob.full}: RuntimeError: boom (a bug in the update search)"]
        assert not [n for n in ir.nodes if getattr(n, "streaming", None) is not None]
        assert not store.path_of(loop_key(ir, ob.full, ob.loops)).exists()  # never stored
    assert update_package._DERIVED == {}  # nor kept


def flash_rows_explicit(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    e = cl.exp(x - _max(x))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def test_e14_a_value_of_another_max_over_the_same_dim():
    """What the second softmax sums is the first softmax, computed from the first max over
    the same row: a different value on every prefix, not one free real per element."""
    ir, ob, d = refused(nested_softmax)
    first, second = [n.name for n in ir.nodes if type(n).__name__ == "ReduceMax"]
    exp = next(n.name for n in ir.nodes if type(n).__name__ == "Exp")
    assert ob.full == second
    assert d.refused == texts.E14.format(node=exp, acc=ob.loops[0], other=first, full=second)
    assert d.refused == (f"{exp}, which {ob.loops[0]} reads, is computed from {first}, another "
                         f"reduction over the dim {second} reduces: it changes from one prefix "
                         "of the loop to the next")
    assert d.flat is None


def test_e14_one_sum_scaled_by_two_softmaxes():
    graph = identity(two_softmaxes, dict(R=16, C=64))
    ir = lowered(graph)
    from cleave.schedule.streaming import obstacles
    from cleave.schedule.tiling import reduction_ancestors, scheduled_reductions

    nodes, red = scheduled_reductions(ir)
    found = [ob for ob in obstacles(nodes, red, reduction_ancestors(nodes)) if not ob.refused]
    assert len(found) == 2
    for ob in found:
        with use_searched_updates():
            d = engine.run(ir, ob, Recipe.of(graph), key=loop_key(ir, ob.full, ob.loops))
        other = next(o.full for o in found if o is not ob)
        assert f"is computed from {other}, another reduction over the dim {ob.full}" in d.refused
        assert d.seconds < 0.5


def test_t5_a_max_under_the_max_is_not_a_sum():
    ir, ob, d = refused(max_normalized)
    inner = [n.name for n in ir.nodes if type(n).__name__ == "ReduceMax" and n.name != ob.full]
    assert set(inner) <= set(ob.loops)
    assert d.refused == texts.T5.format(d=inner[0], cls="ReduceMax", full=ob.full)
    assert d.refused == (f"{inner[0]} is a ReduceMax: the reductions that loop under {ob.full} "
                         "must be sums or matmuls that start at zero")
