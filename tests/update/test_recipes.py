"""Recipes: each program's own flat graph built again at small shapes (DESIGN 5.2; 13.2 e).

For every program of the design's list the recipe re-instantiates the flat graph with a
signature equal to the program's (without dtypes in float32, with them in the program's
dtype) and the obstacle derives a rule; a graph that cannot be built again says why (E1-E4).
CPU only; ``test_search_*`` run the superoptimizer first.
"""

import functools

import pytest

pytest.importorskip("torch")
pytest.importorskip("z3")

import cleave.lang as cl
from cleave.ir.update_rule import loop_key, signature, use_searched_updates
from cleave.lower.state_split import combine_graph, lower_partial, op_indices
from cleave.pipeline.update import engine, texts
from cleave.pipeline.update.instantiate import Recipe
from cleave.pipeline.update.reference import N

from ._attention import FACTOR
from ._programs import (
    _max,
    attention,
    column_average,
    column_average_2d,
    derivation,
    flash_attention,
    flash_rows,
    gqa,
    identity,
    left_constant,
    lowered,
    mla_decode,
    searched,
    softmax_rows,
    the_obstacle,
    traced_attention,
    two_softmaxes,
)
from .test_refusals import column_weighted, max_normalized, self_attention


def flash(graphs):
    (graph,) = [g for g in graphs if g.summary().endswith("sum>matmul>div")]
    return graph


def items(**shapes):
    return tuple(shapes.items())


def derived(ir, ob, recipe):
    with use_searched_updates():
        return engine.run(ir, ob, recipe, key=loop_key(ir, ob.full, ob.loops))


def assert_derives(d, ir):
    """A record with the canonical factor, from instances equal to the program's graph."""
    assert not d.refused, d.refused
    assert d.rules == tuple((acc, FACTOR) for acc in d.loops)
    assert signature(d.flat.graph(N), dtypes=False) == signature(ir, dtypes=False)
    assert signature(d.flat.graph(d.flat.tile), dtypes=False) == signature(ir, dtypes=False)
    assert signature(d.kernel.graph(32), dtypes=True) == signature(ir, dtypes=True)


def symbol_of(graph, input_index, axis):
    inputs = [op for op in graph.structure if op["op_type"] == "kn_input_op"]
    return inputs[input_index]["output_tensors"][0]["dim"][axis]


# ------------------------------------------------------------------------ searched graphs


@functools.lru_cache(maxsize=None)
def solo(fn, shapes, max_ops):
    _, graphs = searched(fn, shapes, max_ops)
    graph = flash(graphs)
    return graph, lowered(graph)


def test_search_gqa_grouped_heads_lower_at_equal_extents():
    """32 query heads over 8 kv heads: the alternating 2 / 3 sizing does not lower (3 is no
    multiple of 2); the next sizing, all 2, does."""
    graph, ir = solo(gqa, items(HQ=32, HKV=8, SQ=4096, SKV=4096, D=64), 11)
    d = derivation(graph, ir)
    assert_derives(d, ir)
    hq, hkv = symbol_of(graph, 0, 0), symbol_of(graph, 1, 0)
    assert d.flat.small[hq] == d.flat.small[hkv]
    assert d.flat.looped == symbol_of(graph, 1, 1)


def test_search_mqa_keeps_its_one_kv_head():
    graph, ir = solo(gqa, items(HQ=32, HKV=1, SQ=4096, SKV=4096, D=64), 11)
    d = derivation(graph, ir)
    assert_derives(d, ir)
    hkv = symbol_of(graph, 1, 0)
    assert graph.symbol_map[hkv] == 1 and d.flat.small[hkv] == 1
    assert d.kernel.small[hkv] == 1


def test_search_mla_decode_keeps_its_unit_batch():
    """demo/mla_decode.py at ``max_ops=14``: the latent kv is K and V at once."""
    graph, ir = solo(mla_decode, items(B=1, H=128, SKV=1024, D=512, PE=64), 14)
    d = derivation(graph, ir)
    assert_derives(d, ir)
    b = symbol_of(graph, 0, 0)
    assert graph.symbol_map[b] == 1 and d.flat.small[b] == 1


def test_search_the_split_partial_and_combine_of_mha_derive_their_own_rules():
    """MHA decode split 16 ways: the partial loops the keys of its own split (the inputs
    carry 16 runs of them), the combine loops the split axis; each is built again by its
    recipe and gets its own record."""
    graph, ir = solo(attention, items(H=32, SQ=64, SKV=4096, D=64), 11)
    ob = the_obstacle(ir)
    full, accs = ob.full, tuple(ob.loops)
    rules = tuple((acc, FACTOR) for acc in accs)
    partial = lower_partial(graph, *op_indices(ir, full, accs), 16)
    d = derived(partial, the_obstacle(partial), Recipe.partial(graph, full, accs, solo=ir,
                                                               split_factor=16))
    assert_derives(d, partial)
    assert [groups for _, groups in d.flat.looped_layout() if groups > 1] == [
        d.flat.small[next(s for s, v in d.flat.recipe.symbols.items() if v == 16)]] * 2

    combine = combine_graph(ir, full, rules, 16, split_pos=1, out_dtype=graph.dtype)
    ob = the_obstacle(combine)
    assert ob.full == "state_max" and set(ob.loops) == {f"{a}_combined" for a in accs}
    d = derived(combine, ob, Recipe.combine(graph, full, rules, 1, graph.dtype, solo=ir,
                                            split_factor=16))
    assert_derives(d, combine)
    # the combine's inputs are the states of the splits: each carries the split axis
    assert sorted(ax for ax, _ in d.flat.looped_layout()) == [1, 1, 1]


# ------------------------------------------------------------------------ identity graphs


def test_a_logits_input_derives():
    ir = lowered(graph := identity(flash_rows, dict(R=64, C=4096, D=64)))
    d = derivation(graph, ir)
    assert_derives(d, ir)
    assert d.flat.looped == "C" and d.flat.small == {"D": 2, "R": 3}


@pytest.mark.parametrize("fn, looped", [(column_average, "K"), (column_average_2d, "K")])
def test_a_softmax_over_the_first_axis_derives(fn, looped):
    shapes = dict(H=4, K=4096, N=64) if fn is column_average else dict(K=4096, N=64)
    ir = lowered(graph := identity(fn, shapes))
    d = derivation(graph, ir)
    assert_derives(d, ir)
    assert d.flat.looped == looped


def test_an_identity_graph_of_attention_derives():
    ir = lowered(graph := identity(flash_attention, dict(H=32, SQ=64, SKV=4096, D=64)))
    d = derivation(graph, ir)
    assert_derives(d, ir)
    assert d.flat.small == {"D": 2, "H": 3, "SQ": 2}


def test_the_identity_recipe_builds_the_program_at_its_own_extents_unchanged():
    program_graph = identity(flash_attention, dict(H=4, SQ=8, SKV=32, D=16))
    recipe = Recipe.of(program_graph)
    assert recipe.kind == "identity" and recipe.symbols == dict(H=4, SQ=8, SKV=32, D=16)
    again = recipe.at(recipe.symbols, program_graph.dtype)
    assert signature(again, dtypes=True) == signature(lowered(program_graph), dtypes=True)
    small = recipe.at(dict(H=2, SQ=3, SKV=4, D=2), "float32")
    assert signature(small, dtypes=False) == signature(again, dtypes=False)
    assert sorted(tuple(int(s) for s in n.get_shape(0)) for n in small.nodes
                  if n.is_placeholder()) == [(2, 3, 2), (2, 4, 2), (2, 4, 2)]


def test_the_structure_of_a_program_takes_other_extents():
    from cleave.frontend import trace

    program = trace(flash_rows, dict(R=64, C=4096, D=64), cl.f16)
    default = program.to_graph_structure()
    assert program.to_graph_structure(None) == default == program.to_graph_structure({})
    small = program.to_graph_structure({"C": 4})
    dims = [t["dim"] for op in small for t in op["input_tensors"] + op["output_tensors"]]
    assert [64, 4] in dims and [4, 64] in dims and [64, 1] in dims
    assert not any(4096 in d for d in dims)
    assert len(small) == len(default) and [op["op_type"] for op in small] == [
        op["op_type"] for op in default]


# ----------------------------------------------------------------------- what refuses


def literal_rows(x: cl.Tensor["R", 32], v: cl.Tensor[32, "D"]):
    e = cl.exp(x - _max(x))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def square_rows(x: cl.Tensor["S", "S"], v: cl.Tensor["S", "D"]):
    e = cl.exp(x - _max(x))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def refusal(graph, ir=None, recipe=None):
    ir = lowered(graph) if ir is None else ir
    ob = the_obstacle(ir)
    d = derived(ir, ob, recipe or Recipe.of(graph))
    return ob, d


def test_e1_a_literal_looped_dim_has_no_symbol():
    ob, d = refusal(identity(literal_rows, dict(R=16, D=8)))
    assert d.refused == texts.E1.format(full=ob.full)
    assert d.refused == (f"the looped dim of {ob.full} has no symbol: the graph cannot be "
                         "built again on a prefix of it")
    assert d.flat is None and not d.holes


def test_e2_a_graph_that_does_not_lower_at_small_shapes():
    graph = identity(flash_rows, dict(R=16, C=64, D=8))
    honest = Recipe.of(graph)

    def build(sizes, dtype):
        if sizes != honest.symbols:
            raise ValueError("extents the lowering refuses")
        return honest.at(sizes, dtype)

    recipe = Recipe("identity", dict(honest.symbols), honest.dtype, build, honest.find_looped)
    _, d = refusal(graph, recipe=recipe)
    assert d.refused == texts.E2.format(error="ValueError: extents the lowering refuses")
    assert d.refused.startswith("the flat graph does not lower at small shapes: ")


def test_e3_a_graph_that_is_another_graph_at_small_shapes():
    """The guard compares the instance with the program's own graph, node for node."""
    from ._programs import scaled_rows

    graph = identity(flash_rows, dict(R=16, C=64, D=8))
    other = Recipe.of(identity(scaled_rows, dict(R=16, C=64, D=8)))
    recipe = Recipe("identity", dict(other.symbols), other.dtype, other.build,
                    Recipe.of(graph).find_looped)
    _, d = refusal(graph, recipe=recipe)
    prefix = "the flat graph is not the same graph at small shapes: node "
    assert d.refused.startswith(prefix) and " instead of " in d.refused


def test_e4_an_input_that_carries_the_looped_dim_twice():
    ir = lowered(graph := identity(square_rows, dict(S=64, D=8)))
    ob, d = refusal(graph, ir)
    (x,) = [n for n in ir.nodes if n.is_placeholder() and len(set(n.get_shape(0))) == 1]
    assert d.refused == texts.E4.format(input=x.name, k=2)
    assert d.refused == (f"{x.name} carries the looped dim on 2 axes: a prefix of the loop is "
                         "not a prefix of it")


def split_keys(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SK", "D"],
               v: cl.Tensor["H", "SV", "D"]):
    """Keys and values declared with dims of their own names: the value matmul ties them."""
    s = (q @ k.mT) / q.shape[-1] ** 0.5
    e = cl.exp(s - _max(s))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def test_symbols_tied_by_a_contraction_are_looped_together():
    ir = lowered(graph := identity(split_keys, dict(H=4, SQ=8, SK=64, SV=64, D=16)))
    d = derivation(graph, ir)
    assert d.flat.looped == frozenset({"SK", "SV"})
    assert_derives(d, ir)
    assert d.flat.small == {"D": 2, "H": 3, "SQ": 2}


# ------------------------------------------- the softmax programs of the rewrite tests


def unscaled_attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                       v: cl.Tensor["H", "SKV", "D"]):
    s = q @ k.mT
    e = cl.exp(s - _max(s))
    return (e / cl.sum(e, -1, keepdim=True)) @ v


def traced_rows(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    return cl.softmax(x) @ v


def sum_next_to_exp(x: cl.Tensor["R", "C"]):
    e = cl.exp(x - _max(x))
    total = cl.sum(e, -1, keepdim=True)
    return cl.sum(e + total, -1, keepdim=True) / total


_ATT = dict(H=4, SQ=8, SKV=64, D=16)
_ROWS = dict(R=16, C=64, D=8)
CENSUS = [  # program, shapes, what lowering under the switch says for each obstacle
    (flash_attention, _ATT, ["record"]),
    (traced_attention, _ATT, ["T3"]),
    (unscaled_attention, _ATT, ["T3"]),
    (softmax_rows, dict(R=16, C=64), ["T4"]),
    (flash_rows, _ROWS, ["record"]),
    (traced_rows, _ROWS, ["T3"]),
    (column_average, dict(H=4, K=64, N=8), ["record"]),
    (column_average_2d, dict(K=64, N=8), ["record"]),
    (two_softmaxes, dict(R=16, C=64), ["E14", "E14"]),
    (max_normalized, _ROWS, ["T5"]),
    (column_weighted, dict(M=8, K=64, N=16), ["E6"]),
    (sum_next_to_exp, dict(R=16, C=64), ["T3"]),
    (left_constant, _ROWS, ["record"]),
    (self_attention, dict(H=4, S=256, D=16), ["E15"]),
]
WHY = {"T3": " reads ", "T4": ", computed from ", "T5": " is a ReduceMax: ", "E14": ", which ",
       "E6": " of one row changes with ", "E15": " also sizes a dim the loop does not step"}


@pytest.mark.parametrize("fn, shapes, want", CENSUS, ids=[c[0].__name__ for c in CENSUS])
def test_every_softmax_program_is_served_or_says_why(fn, shapes, want, tmp_path, monkeypatch):
    """Lowering under the switch: a record where the engine derives one, the reason
    otherwise -- the structural ones of the scheduler (T3: two passes, T4: a value of the
    max read outside the loop, T5: a reduction under the max that is no sum) without asking
    the engine, the engine's own after."""
    import cleave
    from cleave.frontend import trace
    from cleave.pipeline import plans
    from cleave.pipeline import update as update_package
    from cleave.pipeline.superopt import GraphSet

    monkeypatch.setattr(update_package, "_DERIVED", {})
    with use_searched_updates(), cleave.Session(cache_root=tmp_path):
        (graph,) = list(GraphSet.identity(trace(fn, shapes, cl.f16)))
        (ir,) = plans.lower(graph)
    records = [n.streaming for n in ir.nodes if getattr(n, "streaming", None) is not None]
    refusals = ir.streaming_refusals
    assert len(records) == want.count("record") and len(refusals) == len(want) - len(records)
    for record in records:
        assert record.rules == tuple((a, FACTOR) for a in record.loops)
    for kind, why in zip([w for w in want if w != "record"], refusals):
        assert WHY[kind] in why, why
    asked = [w for w in want if w not in ("T3", "T4", "T5")]
    assert len(getattr(ir, "derivations", [])) == len(asked)
