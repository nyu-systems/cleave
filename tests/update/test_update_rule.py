"""``cleave.ir.update_rule`` and the small pieces the searched path stands on (CPU only):
expressions, the identity of a loop structure, ``build_expression``, ``values(given=)``,
``op_index``, the full-extent reducer's two class attributes.
"""

import pytest

torch = pytest.importorskip("torch")

import cleave.lang as cl
from cleave.ir import ops
from cleave.ir.ops import NEG_FLT_MAX, Div, Exp, Exp2, Matmul, ReduceMax, ReduceSum, Sub
from cleave.ir.rolling import STAT_DTYPE, build_expression, check_expression
from cleave.ir.update_rule import (
    EXTENTS,
    SEMANTIC,
    Obstacle,
    Streaming,
    accumulates,
    loop_key,
    repair_of,
    show,
    signature,
    size,
    subterms,
    update_of,
    uses,
)
from cleave.runtime.reference import execute, make_inputs, placeholder_order, values

from ._attention import FACTOR, flash_by_hand, lowered, node_of

SHAPES = dict(H=4, SQ=8, SKV=32, D=16)
UPDATE = ("add", "r", ("mul", "a", FACTOR))


# ---------------------------------------------------------------------------- expressions


def test_show_is_fully_parenthesized():
    assert show(UPDATE) == "(r + (a * exp((m - m'))))"
    assert show("m'") == "m'"
    assert show(("div", ("exp", "m"), ("exp", "m'"))) == "(exp(m) / exp(m'))"


def test_size_counts_the_op_nodes_of_the_tree():
    assert size("a") == 0 and size(FACTOR) == 2 and size(UPDATE) == 4
    shared = ("mul", FACTOR, FACTOR)  # a tree: the shared subterm is written twice
    assert size(shared) == 5


def test_subterms_are_distinct_and_in_the_order_they_are_built():
    assert subterms(FACTOR) == [("sub", "m", "m'"), FACTOR]
    assert subterms(UPDATE) == [("sub", "m", "m'"), FACTOR, ("mul", "a", FACTOR), UPDATE]
    # two holes with one factor pay for it once
    assert subterms(FACTOR, FACTOR) == subterms(FACTOR)
    other = ("div", ("exp", "m"), ("exp", "m'"))
    assert subterms(FACTOR, other) == [("sub", "m", "m'"), FACTOR, ("exp", "m"),
                                      ("exp", "m'"), other]
    assert subterms("m") == []


def test_uses():
    assert uses(UPDATE, "a") and uses(UPDATE, "m'") and not uses(FACTOR, "a")
    assert uses("r", "r") and not uses("r", "a")


@pytest.mark.parametrize("update", [
    ("add", "r", ("mul", "a", FACTOR)),
    ("add", "r", ("mul", FACTOR, "a")),
    ("add", ("mul", "a", FACTOR), "r"),
    ("add", ("mul", FACTOR, "a"), "r"),
])
def test_repair_of_reads_the_factor_whatever_the_operand_order(update):
    assert repair_of(update) == FACTOR
    assert repair_of(update_of(repair_of(update))) == FACTOR


@pytest.mark.parametrize("update", [
    "r",
    ("add", "r", "a"),  # no repair at all
    ("mul", ("add", "r", "a"), FACTOR),  # not "repair, then accumulate"
    ("add", "r", ("div", "a", ("exp", ("sub", "m'", "m")))),  # a division: no tile op for it
    ("add", "r", ("mul", "a", ("mul", "a", FACTOR))),  # the factor reads the accumulator
    ("add", "r", ("mul", "a", ("mul", "r", FACTOR))),  # ... or the tile
    ("add", "r", ("mul", ("exp", "m"), ("div", "a", ("exp", "m'")))),  # a is not an operand
    ("sub", "r", ("mul", "a", FACTOR)),
    ("add", "a", ("mul", "r", FACTOR)),  # the tile is repaired, not the accumulator
])
def test_repair_of_refuses_what_the_assembly_cannot_emit(update):
    assert repair_of(update) is None


def test_update_of_is_the_canonical_update():
    assert update_of(FACTOR) == UPDATE
    assert show(update_of(FACTOR)) == "(r + (a * exp((m - m'))))"


def test_the_records_are_plain_hashable_data():
    record = Streaming(("l", "acc"), (("l", FACTOR), ("acc", FACTOR)), "k")
    assert record == Streaming(("l", "acc"), (("l", FACTOR), ("acc", FACTOR)), "k")
    assert hash(record) == hash(Streaming(("l", "acc"), (("l", FACTOR), ("acc", FACTOR)), "k"))
    assert record != Streaming(("l", "acc"), (("l", FACTOR), ("acc", FACTOR)), "other")
    with pytest.raises(AttributeError):
        record.key = "x"
    with pytest.raises(TypeError):
        Streaming(("l",), (("l", FACTOR),))  # a record always says which graph it is for
    obstacle = Obstacle("m", ("l", "acc"))
    assert obstacle.refused == "" and obstacle == Obstacle("m", ("l", "acc"), "")


# ------------------------------------------------------------------ the full-extent reducer


def test_every_reducer_says_how_it_is_carried():
    """How a reduction's value over a prefix and over the next tile combine, and where the
    carried value starts: a property of the reducer, read by the conflict detection, the
    references, the ranking and the assembly alike. A contraction declares none (it is
    never carried, only accumulated under a carried value)."""
    assert ReduceMax.combine == "max" and ReduceMax.carry_init == NEG_FLT_MAX
    assert ReduceSum.combine == "add" and ReduceSum.carry_init is None  # zero: T.clear
    assert getattr(Matmul, "combine", None) is None
    # the class default of an accumulated max stays the identity of max
    assert ReduceMax.accum_init == -float("inf")
    node = ReduceMax((4, 8), axis=-1, name="m")
    assert node.accum_init == -float("inf") and type(node).carry_init == NEG_FLT_MAX
    for cls in (ReduceSum, ReduceMax):
        assert "combine" not in cls._macro_attrs and "carry_init" not in cls._macro_attrs


def test_what_accumulates_under_a_carried_value():
    """A repair rescales what an accumulator holds: a sum or a contraction from zero, by the
    reducer's combine, not by the class."""
    assert accumulates(ReduceSum((4, 8), axis=-1, name="s"))
    assert accumulates(Matmul((4, 2, 8), name="mm"))  # (M, N, K)
    assert not accumulates(ReduceMax((4, 8), axis=-1, name="m"))
    started = ReduceSum((4, 8), axis=-1, name="t")
    started.accum_init = 1.0
    assert not accumulates(started)


# ----------------------------------------------------------------- identity of a structure


def test_every_macro_attribute_lowering_can_emit_is_classified():
    """The key of a loop structure holds the SEMANTIC attributes and leaves out the EXTENTS
    (extents, storage and layout choices). A macro attribute in neither list would be left
    out of the key without anybody having decided that it may be."""
    from cleave.lower.convert import _RULES

    assert not set(SEMANTIC) & set(EXTENTS)
    assert len(set(SEMANTIC)) == len(SEMANTIC) and len(set(EXTENTS)) == len(EXTENTS)
    # the classes the rules of lowering build (a new rule: add its class here)
    assert {kind for kind, _ in _RULES} == {
        "input", "output", "split", "matmul", "reduce_sum", "reduce_max", "exp", "silu", "sqrt",
        "rsqrt", "add", "sub", "mul", "div"}
    emitted = (ops.SplitView, ops.Matmul, ops.ReduceSum, ops.ReduceMax, ops.Exp, ops.Silu,
               ops.Sqrt, ops.RSqrt, ops.Add, ops.Sub, ops.Mul, ops.Div)
    unclassified = {(cls.__name__, attr) for cls in emitted for attr in cls._macro_attrs
                    if attr not in SEMANTIC and attr not in EXTENTS}
    assert unclassified == set()
    # ... and the lists do say something about each of them
    assert all(cls._macro_attrs for cls in emitted if cls is not ops.SplitView)


def test_signature_is_plain_data_without_extents():
    small = lowered(flash_by_hand, SHAPES)
    large = lowered(flash_by_hand, dict(H=8, SQ=64, SKV=4096, D=16))
    for dtypes in (False, True):
        assert signature(small, dtypes) == signature(large, dtypes)
    rows = signature(small, dtypes=True)
    assert [row[:2] for row in rows] == [[type(n).__name__, n.name] for n in small.nodes]
    by_name = {row[1]: row for row in rows}
    full = node_of(small, ReduceMax)
    (scores,) = [e.src_node for e in full.inputs]
    assert by_name[full.name][2] == [[scores.name, 0, 0]]
    assert ["flat_axis", repr(full.flat_axis)] in by_name[full.name][3]
    assert ["accum_init", "-inf"] in by_name[full.name][3]
    assert by_name[full.name][4] == ["float32", "float32"]
    assert all(row[4] == [] for row in signature(small, dtypes=False))
    import json

    json.dumps(rows)  # plain data: what a cache key is made of
    # no attribute that holds an extent is in it
    assert {key for row in rows for key, _ in row[3]} <= set(SEMANTIC)
    assert "4096" not in json.dumps(signature(large, dtypes=True))


def test_signature_tells_dtypes_apart_only_when_asked():
    f16 = lowered(flash_by_hand, SHAPES, cl.f16)
    f32 = lowered(flash_by_hand, SHAPES, cl.f32)
    assert signature(f16, dtypes=False) == signature(f32, dtypes=False)
    assert signature(f16, dtypes=True) != signature(f32, dtypes=True)


def test_loop_key_is_equal_across_clone_and_moves_with_the_graph():
    ir = lowered(flash_by_hand, SHAPES)
    full = node_of(ir, ReduceMax).name
    loops = (node_of(ir, Matmul, 1).name, node_of(ir, ReduceSum).name)
    key = loop_key(ir, full, loops)
    assert len(key) == 64 and key == loop_key(ir, full, loops)
    assert loop_key(ir.clone(), full, loops) == key
    assert loop_key(lowered(flash_by_hand, SHAPES), full, loops) == key  # lowered again
    # no extents: the same structure at another shape is the same loop
    assert loop_key(lowered(flash_by_hand, dict(H=8, SQ=64, SKV=4096, D=16)), full, loops) == key
    # the loop membership and the full-extent node are part of it
    assert loop_key(ir, full, loops[::-1]) != key and loop_key(ir, full, loops[:1]) != key
    assert loop_key(ir, loops[0], loops) != key
    # a semantic attribute: the constant of the scaling
    edited = ir.clone()
    node_of(edited, Div).const = 2.0
    assert loop_key(edited, full, loops) != key
    # ... the start value of an accumulated max
    edited = ir.clone()
    node_of(edited, ReduceMax).accum_init = NEG_FLT_MAX
    assert loop_key(edited, full, loops) != key
    # ... a name, the wiring (the subtraction reads its operands the other way round)
    edited = ir.clone()
    node_of(edited, Sub).name = "sub_renamed"
    assert loop_key(edited, full, loops) != key
    edited = ir.clone()
    for edge in node_of(edited, Sub).inputs:
        edge.dst_id = 1 - edge.dst_id
    assert loop_key(edited, full, loops) != key
    # ... the dtype a value is computed in
    assert loop_key(lowered(flash_by_hand, SHAPES, cl.f32), full, loops) != key
    assert loop_key(ir, full, loops) == key  # nothing above touched the graph itself


# ------------------------------------------------------------------------ build_expression


def _state(rows=(4, 8), nb=1):
    lag = ops.Copy(rows, name="m_prev", dtype=STAT_DTYPE, out_dtype=STAT_DTYPE, n_batch_dims=nb)
    full = ReduceMax(rows + (16,), axis=-1, name="m", dtype=STAT_DTYPE, out_dtype=STAT_DTYPE)
    return {"m": lag, "m'": full}


def test_build_expression_makes_one_named_float32_node_per_subterm():
    env, made = _state(), {}
    factor = build_expression(FACTOR, env, made, stem="m", rows=(4, 8), n_batch_dims=1,
                              exp_cls=Exp)
    step = made[("sub", "m", "m'")]
    assert list(made) == subterms(FACTOR) and made[FACTOR] is factor
    assert (type(step), step.name, type(factor), factor.name) == (Sub, "m_upd0", Exp, "m_upd1")
    for node in made.values():
        assert tuple(node.get_shape()) == (4, 8) and node.n_batch_dims == 1
        assert node.dtype == node.out_dtype == STAT_DTYPE == "float32"
    assert [e.src_node for e in sorted(step.inputs, key=lambda e: e.dst_id)] == [env["m"],
                                                                                env["m'"]]
    assert step.broadcast == -1 and step.const is None
    assert [e.src_node for e in factor.inputs] == [step]
    # a leaf is the node it names, and makes nothing
    assert build_expression("m", env, made, stem="m", rows=(4, 8), n_batch_dims=1,
                            exp_cls=Exp) is env["m"]
    assert len(made) == 2


def test_build_expression_shares_subterms_between_the_accumulators_of_a_loop():
    env, made = _state(), {}
    kw = dict(stem="m", rows=(4, 8), n_batch_dims=1, exp_cls=Exp)
    first = build_expression(FACTOR, env, made, **kw)
    assert build_expression(FACTOR, env, made, **kw) is first and len(made) == 2
    other = build_expression(("div", ("exp", "m"), ("exp", "m'")), env, made, **kw)
    assert [n.name for n in made.values()] == ["m_upd0", "m_upd1", "m_upd2", "m_upd3", "m_upd4"]
    assert type(other) is Div and other.name == "m_upd4"
    # a term that needs the shared step again reuses its node
    again = build_expression(("mul", ("sub", "m", "m'"), ("sub", "m", "m'")), env, made, **kw)
    assert {e.src_node.name for e in again.inputs} == {"m_upd0"} and len(made) == 6


def test_build_expression_uses_the_exp_of_the_values_it_repairs():
    made = {}
    factor = build_expression(FACTOR, _state(), made, stem="m", rows=(4, 8), n_batch_dims=1,
                              exp_cls=Exp2)
    assert type(factor) is Exp2


def test_build_expression_broadcasts_a_narrow_leaf_on_the_right():
    """The merged max of a split combine has no split axis: it is broadcast across it."""
    rows = (4, 16, 8)  # batch, splits, rows
    per_split = ops.Copy(rows, name="m_i", dtype=STAT_DTYPE, out_dtype=STAT_DTYPE, n_batch_dims=1)
    merged = ReduceMax(rows, axis=1, name="M", dtype=STAT_DTYPE, out_dtype=STAT_DTYPE)
    made = {}
    factor = build_expression(FACTOR, {"m": per_split, "m'": merged}, made, stem="state",
                              rows=rows, n_batch_dims=1, exp_cls=Exp, narrow=("m'",))
    step = made[("sub", "m", "m'")]
    assert step.broadcast == 1 and step.name == "state_upd0" and factor.name == "state_upd1"
    assert tuple(step.get_shape()) == rows == tuple(factor.get_shape())


@pytest.mark.parametrize("factor, narrow, text", [
    (("max", "m", "m'"), (), "m: the repair uses 'max', which has no tile op"),
    (("exp", ("log", "m")), (), "m: the repair uses 'log', which has no tile op"),
    (("sub", "m'", "m"), ("m'",), "m: the reduced value \"m'\" on the left of 'sub' has no tile op"),
    (("div", ("exp", "m"), ("exp", "m'")), ("m'",),
     "m: the reduced value \"m'\" of 'exp' has no tile op"),
])
def test_build_expression_refuses_before_it_builds(factor, narrow, text):
    env, made = _state(), {}
    kw = dict(stem="m", rows=(4, 8), n_batch_dims=1, exp_cls=Exp, narrow=narrow)
    with pytest.raises(ValueError) as error:
        build_expression(factor, env, made, **kw)
    assert str(error.value) == text
    assert made == {} and not env["m"].outputs and not env["m'"].outputs
    with pytest.raises(ValueError) as error:
        check_expression(factor, stem="m", narrow=narrow)
    assert str(error.value) == text
    check_expression(FACTOR, stem="m", narrow=("m'",))  # ... and says nothing of a right one


# ---------------------------------------------------------------------------- values(given=)


def _inputs(ir, seed=0):
    return list(make_inputs(ir, "float16", seed=seed, device="cpu"))


def test_values_returns_every_node_by_name_and_the_outputs_of_execute():
    ir = lowered(flash_by_hand, SHAPES)
    inputs = _inputs(ir)
    env, outputs = values(ir, inputs)
    want = execute(ir, inputs)
    assert len(outputs) == len(want) == 1 and torch.equal(outputs[0], want[0])
    names = {n.name for n in ir.nodes if not n.is_output()}
    assert set(env) == names
    for placeholder, tensor in zip(placeholder_order(ir), inputs):
        assert env[placeholder.name] is tensor
    scores, full = env[node_of(ir, Div).name], node_of(ir, ReduceMax)
    assert torch.equal(env[full.name], scores.float().amax(-1))
    assert torch.equal(env[node_of(ir, Sub).name], scores.float() - env[full.name][..., None])
    with pytest.raises(ValueError, match="Expected 3 inputs, got 2"):
        values(ir, inputs[:2])
    with pytest.raises(ValueError, match="Expected 3 inputs, got 2"):
        execute(ir, inputs[:2])


def test_values_reads_a_given_node_instead_of_evaluating_it():
    """The tile computation re-wired to read the max of a longer prefix: everything behind
    the given node is computed from the given value."""
    ir = lowered(flash_by_hand, SHAPES)
    inputs = _inputs(ir)
    env, (plain,) = values(ir, inputs)
    full, sub, exp = (node_of(ir, cls).name for cls in (ReduceMax, Sub, Exp))
    higher = env[full] + 2.0
    shifted, (out,) = values(ir, inputs, given={full: higher})
    assert shifted[full] is higher
    assert torch.equal(shifted[sub], env[node_of(ir, Div).name].float() - higher[..., None])
    assert torch.equal(shifted[exp], shifted[sub].exp())
    assert not torch.equal(shifted[exp], env[exp])
    # the softmax does not care which shift it was computed with: the row sum (float32)
    # moves by exp(-2), the output by rounding only
    total = node_of(ir, ReduceSum).name
    torch.testing.assert_close(shifted[total], env[total] * torch.exp(torch.tensor(-2.0)),
                               rtol=1e-5, atol=0)
    torch.testing.assert_close(out.float(), plain.float(), rtol=0, atol=2e-3)
    # what is given need not be a node of the graph: it is then not read
    again, (same,) = values(ir, inputs, given={"no_such_node": higher})
    assert torch.equal(same, plain) and set(again) == set(env)
    # a given accumulator replaces the loop state behind it: the epilogue on a state
    doubled, (halved,) = values(ir, inputs, given={total: env[total] * 2})
    assert doubled[total] is not env[total]
    torch.testing.assert_close(halved.float(), plain.float() / 2, rtol=0, atol=1e-3)


# --------------------------------------------------------------------------------- op_index


def rms_matmul(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    return cl.rms_norm(x, dim=-1) @ w


def _structure_of(fn, shapes):
    from cleave.frontend import trace

    return trace(fn, shapes, cl.f16).to_graph_structure()


KIND = {"PlaceHolderNode": "kn_input_op", "OutputNode": "kn_output_op", "Matmul": "kn_matmul_op",
        "Exp": "kn_exp_op", "Sub": "kn_sub_op", "Div": "kn_div_op", "Mul": "kn_mul_op",
        "Add": "kn_add_op", "RSqrt": "kn_div_op"}


@pytest.mark.parametrize("fn, shapes", [(flash_by_hand, SHAPES),
                                        (rms_matmul, dict(M=16, K=64, N=32))],
                         ids=["attention", "rmsnorm_matmul"])
def test_every_lowered_node_knows_the_op_it_was_lowered_from(fn, shapes):
    ir, structure = lowered(fn, shapes), _structure_of(fn, shapes)
    assert all(isinstance(getattr(n, "op_index", None), int) for n in ir.nodes)
    seen = set()
    for node in ir.nodes:
        op_type = structure[node.op_index]["op_type"]
        name = type(node).__name__
        if name in ("ReduceSum", "ReduceMax"):
            kind = "reduce_sum" if name == "ReduceSum" else "reduce_max"
            assert op_type.startswith(f"kn_{kind}_"), (node.name, op_type)
        else:
            assert op_type == KIND[name], (node.name, op_type)
        # lowering names a compute node after that index, and no two nodes share one
        if not node.is_placeholder() and not node.is_output():
            assert node.name.endswith(f"_{node.op_index}")
        assert node.op_index not in seen
        seen.add(node.op_index)
    # a fused 1 / sqrt(x) is the div's: the sqrt has no node of its own
    if fn is rms_matmul:
        (rsqrt,) = [n for n in ir.nodes if type(n).__name__ == "RSqrt"]
        assert structure[rsqrt.op_index - 1]["op_type"] == "kn_sqrt_op"
        assert not [n for n in ir.nodes if type(n).__name__ == "Sqrt"]
    # it survives a clone, like every other attribute of a node
    assert [n.op_index for n in ir.clone().nodes] == [n.op_index for n in ir.nodes]
