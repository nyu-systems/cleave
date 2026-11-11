import ast
import math

import pytest

pytest.importorskip("tilelang")

from cleave.codegen import TilelangCodeGen
from cleave.ir.ops import Add, Div, Matmul, ReduceMax, Mul, ReduceSum, Sub


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
