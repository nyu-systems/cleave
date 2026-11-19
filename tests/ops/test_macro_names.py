import ast
import math
import struct

import pytest

pytest.importorskip("tilelang")

from cleave.codegen import TilelangCodeGen
from cleave.ir.ops import (
    NEG_FLT_MAX,
    Accumulate,
    Add,
    Div,
    Matmul,
    Mul,
    ReduceMax,
    ReduceSum,
    Rescale,
    Sub,
)


def _emit(cg, node, mem_scope, clear=False):
    return cg.emit_op_node(node, mem_scope=mem_scope, clear=clear)


def _defined_names(code):
    return [n.name for n in ast.walk(ast.parse(code)) if isinstance(n, ast.FunctionDef)]


def test_matmuls_with_different_warp_policy_get_distinct_names():
    cg = TilelangCodeGen()
    rows = Matmul((64, 64, 32), warp_policy="rows", dtype="float16")
    cols = Matmul((64, 64, 32), warp_policy="cols", dtype="float16")
    a = _emit(cg, rows, "ssr")
    b = _emit(cg, cols, "ssr")
    assert a == "ukernel_mma_ssr"
    assert b != a
    assert _defined_names(cg.prim_code) == [a, b]
    assert "FullRow" in cg.generated_macros[a] and "FullCol" in cg.generated_macros[b]


def test_matmul_dtype_on_global_operand_gets_distinct_name():
    cg = TilelangCodeGen()
    a = _emit(cg, Matmul((64, 64, 32), dtype="float16"), "ggr", clear=True)
    b = _emit(cg, Matmul((64, 64, 32), dtype="bfloat16"), "ggr", clear=True)
    assert a != b


def test_identical_bodies_share_one_macro():
    cg = TilelangCodeGen()
    a = _emit(cg, Matmul((64, 64, 32), warp_policy="rows"), "ssr")
    b = _emit(cg, Matmul((64, 64, 32), warp_policy="rows", name="other"), "ssr")
    assert a == b
    assert len(cg.generated_macros) == 1


@pytest.mark.parametrize(
    "first, second",
    [
        (dict(axis=-1), dict(axis=1)),
        (dict(axis=-1, dtype="float32"), dict(axis=-1, dtype="float16")),
    ],
)
def test_reductions_with_different_bodies_get_distinct_names(first, second):
    cg = TilelangCodeGen()
    a = _emit(cg, ReduceSum((16, 32), **first), "sr")
    b = _emit(cg, ReduceSum((16, 32), **second), "sr")
    assert a == "ukernel_reduction_axis1_sr"
    assert b != a
    assert _defined_names(cg.prim_code) == [a, b]


@pytest.mark.parametrize("op", [Add, Sub, Mul, Div])
@pytest.mark.parametrize("const", [1e20, -math.inf, math.inf, -1e-20])
def test_const_names_are_identifiers_and_bodies_parse(op, const):
    cg = TilelangCodeGen()
    name = _emit(cg, op((16, 32), const=const), "rr")
    assert name.isidentifier()
    tree = ast.parse(cg.prim_code)
    bare = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "inf" not in bare


def test_existing_const_names_unchanged():
    assert Mul((16, 32), const=0.5).get_macro_name(mem_scope="rr").startswith(
        "ukernel_mul_const_0_5"
    )
    assert Add((16, 32), const=-2.0).get_macro_name(mem_scope="rr").startswith(
        "ukernel_add_const_minus_2_0"
    )


def test_reduce_max_macro_name():
    cg = TilelangCodeGen()
    s = _emit(cg, ReduceSum((16, 32)), "rr", clear=True)
    m = _emit(cg, ReduceMax((16, 32)), "rr", clear=True)
    assert m == "ukernel_reduce_max_axis1_rr_clear" and m != s
    assert "T.reduce_max(" in cg.generated_macros[m] and "T.reduce_sum(" in cg.generated_macros[s]
    assert ReduceMax.accum_init == -math.inf and ReduceSum.accum_init is None


def test_carried_max_start_is_per_instance():
    """A carried softmax max starts at the most negative finite float32; `-inf` stays the
    class default, and the start value is no part of the macro (it is the fill of the
    accumulator's buffer, emitted where that is allocated)."""
    assert NEG_FLT_MAX == -3.4028234663852886e38 and math.isfinite(NEG_FLT_MAX)
    assert struct.unpack("f", struct.pack("f", NEG_FLT_MAX))[0] == NEG_FLT_MAX  # a float32
    with pytest.raises(OverflowError):  # ... and no float16: such a buffer would hold -inf
        struct.pack("e", NEG_FLT_MAX)

    plain, carried = ReduceMax((16, 32)), ReduceMax((16, 32), accum_init=NEG_FLT_MAX)
    assert carried.accum_init == NEG_FLT_MAX
    assert plain.accum_init == -math.inf and "accum_init" not in vars(plain)
    assert ReduceMax.accum_init == -math.inf and ReduceSum.accum_init is None
    assert carried.macro_key() == plain.macro_key()
    cg = TilelangCodeGen()
    assert _emit(cg, carried, "rr") == _emit(cg, plain, "rr") == "ukernel_reduce_max_axis1_rr"
    assert len(cg.generated_macros) == 1


def test_only_a_max_is_a_running_accumulator():
    assert ReduceMax.running_max is True and ReduceMax((16, 32)).running_max is True
    for cls in (ReduceSum, Matmul, Accumulate, Mul):
        assert cls.running_max is False, cls.__name__


def test_rescale_macro_names_and_bodies():
    cg = TilelangCodeGen()
    full = _emit(cg, Rescale(rank=1, lead=1), "rr")
    lead = _emit(cg, Rescale(rank=2, lead=1), "rr")
    assert full == "ukernel_rescale_1dims_rr"
    assert lead == "ukernel_rescale_lead1_2dims_rr"
    assert cg.generated_macros[full] == (
        "@T.macro\n"
        "def ukernel_rescale_1dims_rr(S, ACC):\n"
        "    for i0 in T.Parallel(*ACC.shape):\n"
        "        ACC[i0] *= S[i0]\n"
    )
    assert cg.generated_macros[lead] == (
        "@T.macro\n"
        "def ukernel_rescale_lead1_2dims_rr(S, ACC):\n"
        "    for i0, i1 in T.Parallel(*ACC.shape):\n"
        "        ACC[i0, i1] *= S[i0]\n"
    )
    assert _defined_names(cg.prim_code) == [full, lead]
    # Equal ranks, equal body: one macro however many accumulators use it.
    assert _emit(cg, Rescale(rank=2, lead=1), "rr") == lead
    assert len(cg.generated_macros) == 2

    assert _emit(cg, Rescale(rank=3, lead=2), "rr") == "ukernel_rescale_lead2_3dims_rr"
    assert "ACC[i0, i1, i2] *= S[i0, i1]" in cg.generated_macros["ukernel_rescale_lead2_3dims_rr"]


@pytest.mark.parametrize("rank, lead", [(1, 2), (2, 0), (2, 3)])
def test_rescale_factor_is_a_leading_slice(rank, lead):
    with pytest.raises(AssertionError, match="Rescale"):
        Rescale(rank=rank, lead=lead)


@pytest.mark.parametrize("scope", ["sr", "rs", "gr"])
def test_rescale_is_register_only(scope):
    with pytest.raises(AssertionError, match="register"):
        Rescale(rank=1, lead=1).gen_macro(mem_scope=scope, clear=False)
