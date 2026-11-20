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
    CausalMaskedExp,
    CausalMaskedScore,
    Div,
    Exp,
    Exp2,
    Log2,
    Lse2,
    MaskedExp,
    MaskedLog2,
    MaskedLse2,
    MaskedScore,
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


# --------------------------------------------------------------- the exact paged softmax
def _text(cg, name):
    """One macro's source with its layout normalised (the module is black-formatted later)."""
    return " ".join(cg.generated_macros[name].split())


def test_masked_score_macro_names_and_bodies():
    """The exp operand of the exact paged softmax: scaled into the log2 domain, masked
    entries at the most negative finite float32 (never -inf), the scale taken at runtime."""
    cg = TilelangCodeGen()
    plain = _emit(cg, MaskedScore((16, 64), n_batch_dims=0), "rr", clear=True)
    causal = _emit(cg, CausalMaskedScore((64, 16), row_group=4, n_batch_dims=0), "rr", clear=True)
    assert plain == "ukernel_maskedscore_scale_rt2_2dims_rr"
    assert causal == "ukernel_causal_maskedscore_scale_rt2_mgrp4_2dims_rr"
    assert _text(cg, plain) == (
        "@T.macro def ukernel_maskedscore_scale_rt2_2dims_rr(A, tile_base, valid_len, scale2, B): "
        "for m0, m1 in T.Parallel(*B.shape): "
        "B[m0, m1] = T.if_then_else( tile_base + m1 < valid_len, A[m0, m1] * scale2, "
        "-3.4028234663852886e+38, )"
    )
    assert _text(cg, causal) == (
        "@T.macro def ukernel_causal_maskedscore_scale_rt2_mgrp4_2dims_rr"
        "(A, tile_base, valid_len, diag, scale2, B): "
        "for m0, m1 in T.Parallel(*B.shape): "
        "B[m0, m1] = T.if_then_else( "
        "(tile_base + m1 < valid_len) & (tile_base + m1 <= diag + m0 // 4), "
        "A[m0, m1] * scale2, -3.4028234663852886e+38, )"
    )
    assert float("-3.4028234663852886e+38") == NEG_FLT_MAX
    assert "infinity" not in cg.prim_code
    # the scale is the last scalar, whatever it is called
    renamed = MaskedScore((16, 64), scalar_args=("base", "live", "s2"))
    assert "A[m0, m1] * s2" in renamed.gen_macro(mem_scope="rr", clear=True)
    with pytest.raises(AssertionError, match="plus the runtime scale"):
        MaskedScore((16, 64), scalar_args=("tile_base", "valid_len"))
    with pytest.raises(AssertionError, match="float32 value"):
        MaskedScore((16, 64), out_dtype="float16")
    with pytest.raises(AssertionError, match="does not read the row index"):
        MaskedScore((16, 64), row_group=4)


def test_shifted_masked_exp_macro_names_and_bodies():
    """``exp2(score - max[row])`` under the mask of the plain form; a masked entry stays the
    literal 0.0 through the predicate. The flag is in the name: the plain macro is unchanged."""
    cg = TilelangCodeGen()
    plain = MaskedExp((16, 64), None, scalar_args=("tile_base", "valid_len", "scale2"),
                      n_batch_dims=0)
    shifted = MaskedExp((16, 64), None, scalar_args=("tile_base", "valid_len"), shift=True,
                        n_batch_dims=0)
    causal = CausalMaskedExp((64, 16), None, scalar_args=("tile_base", "valid_len", "diag"),
                             row_group=4, shift=True, n_batch_dims=0)
    assert [len(op.inputs) for op in (plain, shifted, causal)] == [1, 2, 2]
    assert tuple(shifted.inputs[1].src_node.get_shape(0)) == (16,)  # one max per row
    a = _emit(cg, plain, "rr", clear=True)
    b = _emit(cg, shifted, "rrr", clear=True)
    c = _emit(cg, causal, "rrr", clear=True)
    assert a == "ukernel_maskedexp_scale_rt2_2dims_rr"
    assert b == "ukernel_maskedexp_shift2_2dims_rrr"
    assert c == "ukernel_causal_maskedexp_shift2_mgrp4_2dims_rrr"
    assert _text(cg, a) == (
        "@T.macro def ukernel_maskedexp_scale_rt2_2dims_rr(A, tile_base, valid_len, scale2, B): "
        "for m0, m1 in T.Parallel(*B.shape): "
        "B[m0, m1] = T.if_then_else( tile_base + m1 < valid_len, T.exp2(A[m0, m1] * scale2), 0.0 )"
    )
    assert _text(cg, b) == (
        "@T.macro def ukernel_maskedexp_shift2_2dims_rrr(A, M, tile_base, valid_len, B): "
        "for m0, m1 in T.Parallel(*B.shape): "
        "B[m0, m1] = T.if_then_else( tile_base + m1 < valid_len, T.exp2(A[m0, m1] - M[m0]), 0.0 )"
    )
    assert _text(cg, c) == (
        "@T.macro def ukernel_causal_maskedexp_shift2_mgrp4_2dims_rrr"
        "(A, M, tile_base, valid_len, diag, B): "
        "for m0, m1 in T.Parallel(*B.shape): "
        "B[m0, m1] = T.if_then_else( "
        "(tile_base + m1 < valid_len) & (tile_base + m1 <= diag + m0 // 4), "
        "T.exp2(A[m0, m1] - M[m0]), 0.0, )"
    )
    # still the classes the paged recognisers and tests look for
    assert isinstance(shifted, MaskedExp) and isinstance(causal, MaskedExp)
    assert shifted.shift is True and plain.shift is False
    # the shifted form takes the mask scalars only: its operand is already scaled
    with pytest.raises(AssertionError, match="2 mask scalars, got"):
        MaskedExp((16, 64), None, scalar_args=("tile_base", "valid_len", "scale2"), shift=True)
    with pytest.raises(AssertionError, match="applied twice"):
        MaskedExp((16, 64), 0.5, shift=True)


def test_the_plain_masked_exp_names_are_unchanged():
    assert MaskedExp((16, 32), 0.5).get_macro_name(mem_scope="rr") == (
        "ukernel_maskedexp_scale_0_5_2dims_rr")
    assert CausalMaskedExp((16, 32), 0.5, row_group=2).get_macro_name(mem_scope="rr") == (
        "ukernel_causal_maskedexp_scale_0_5_mgrp2_2dims_rr")
    assert CausalMaskedExp(
        (16, 32), None, scalar_args=("a", "b", "c", "d")
    ).get_macro_name(mem_scope="rr") == "ukernel_causal_maskedexp_scale_rt2_2dims_rr"


def test_exp2_macro_name_and_body():
    cg = TilelangCodeGen()
    name = _emit(cg, Exp2((16,), n_batch_dims=0), "rr", clear=True)
    assert name == "ukernel_exp2_1dims_rr" != _emit(cg, Exp((16,), n_batch_dims=0), "rr", True)
    assert cg.generated_macros[name] == (
        "@T.macro\n"
        "def ukernel_exp2_1dims_rr(A, B):\n"
        "    for i0 in T.Parallel(*B.shape):\n"
        "        B[i0] = T.exp2(A[i0])\n"
    )


def test_lse2_macro_names_and_bodies():
    """``log2(l) + m``: two operands, so its own macro names; ``l`` is not floored."""
    cg = TilelangCodeGen()
    lse = Lse2((4,), n_batch_dims=0)
    assert len(lse.inputs) == 2
    name = _emit(cg, lse, "rrg", clear=True)
    assert name == "ukernel_lse2_1dims_rrg"
    assert name != _emit(cg, Log2((4,), n_batch_dims=0), "rg", clear=True)
    assert cg.generated_macros[name] == (
        "@T.macro\n"
        "def ukernel_lse2_1dims_rrg(A, M, B):\n"
        "    for i0 in T.Parallel(*B.shape):\n"
        "        B[i0] = T.log2(A[i0]) + M[i0]\n"
    )
    assert "T.max" not in cg.generated_macros[name]

    plain = _emit(cg, MaskedLse2((64,), n_batch_dims=0), "rrg", clear=True)
    fused = _emit(cg, MaskedLse2((64,), row_group=4, n_batch_dims=0), "rrg", clear=True)
    slot = _emit(cg, MaskedLse2((64,), row_group=4, fuse_dst=False, n_batch_dims=0), "rrg", True)
    assert (plain, fused, slot) == (
        "ukernel_masked_lse2_1dims_rrg",
        "ukernel_masked_lse2_mgrp4_1dims_rrg",
        "ukernel_masked_lse2_mgrp4_dst2_1dims_rrg",
    )
    assert cg.generated_macros[plain] == (
        "@T.macro\n"
        "def ukernel_masked_lse2_1dims_rrg(A, M, n_rows, B):\n"
        "    for i0 in T.Parallel(*B.shape):\n"
        "        if i0 < n_rows:\n"
        "            B[i0] = T.log2(A[i0]) + M[i0]\n"
    )
    assert cg.generated_macros[fused] == (
        "@T.macro\n"
        "def ukernel_masked_lse2_mgrp4_1dims_rrg(A, M, n_rows, B):\n"
        "    for i0 in T.Parallel(*A.shape):\n"
        "        if i0 // 4 < n_rows:\n"
        "            B[i0 // 4, i0 % 4] = T.log2(A[i0]) + M[i0]\n"
    )
    assert "B[i0 % 4] = T.log2(A[i0]) + M[i0]" in cg.generated_macros[slot]
    # the destination indexing is MaskedLog2's, token for token
    for kw in (dict(), dict(row_group=4), dict(row_group=4, fuse_dst=False)):
        log2 = MaskedLog2((64,), n_batch_dims=0, **kw).gen_macro(mem_scope="rg", clear=True)
        lse2 = MaskedLse2((64,), n_batch_dims=0, **kw).gen_macro(mem_scope="rrg", clear=True)
        dst = lambda text: text.split("\n")[-2].split(" = ")[0]  # noqa: E731
        assert dst(log2) == dst(lse2) and log2.split("\n")[2:4] == lse2.split("\n")[2:4]
