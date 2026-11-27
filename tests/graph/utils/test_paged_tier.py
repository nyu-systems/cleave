"""The exact masked softmax in a hand-built paged-shaped tier (CPU: codegen and lowering).

``tests/graph/_paged_tier.py`` builds the loop tier of a paged attention block from IR
constructors alone: a masked score in the log2 domain, its carried row max, the shifted
masked exp, and the two rescaled accumulators, with ``log2(l) + m`` after the loop. These
tests pin what that tier emits and that TileLang lowers it at the tile shapes the paged
operators use, before any builder in ``cleave.dynamic.paging`` produces it.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.codegen.kernel import emission_order
from cleave.ir.ops import (
    NEG_FLT_MAX,
    CausalMaskedExp,
    CausalMaskedScore,
    Exp,
    Exp2,
    Lse2,
    MaskedExp,
    MaskedLse2,
    MaskedScore,
    Mul,
    ReduceMax,
    Sub,
)

from tests._lowering import lower_to_cuda
from tests.graph._paged_tier import build_paged_tier, paged_tier_code


def _calls(code, macro):
    """The statements of one macro body, one per line, allocations folded onto one line."""
    start = code.index(f"def {macro}(")
    body = code[code.index("):\n", start) + 3:]
    body = body[: body.index("\n\n\n")] if "\n\n\n" in body else body
    return [re.sub(r"\s+", " ", s).strip() for s in re.split(r"\n(?=    \S)", body) if s.strip()]


def _flat(code, macro):
    """One macro body without layout: no whitespace, no trailing commas (long calls wrap)."""
    start = code.index(f"def {macro}(")
    body = code[code.index("):\n", start) + 3:]
    body = body[: body.index("\n\n\n")] if "\n\n\n" in body else body
    return re.sub(r"\s+", "", body).replace(",)", ")")


def _signature(code, macro):
    text = code[code.index(f"def {macro}("):]
    return re.findall(r"(\w+):", text[: text.index("):")])


@pytest.mark.parametrize("causal", [False, True], ids=["decode", "causal"])
def test_tier_structure(causal):
    t = build_paged_tier(causal)
    inner = t.inner
    # the automatic accumulators as before, the carried max appended behind them
    assert [n.name for n in inner.accum_nodes] == ["rowsum_0", "mma_1", "exp_0_max"]
    assert [n.name for n in inner.lag_nodes] == ["exp_0_max_prev"]
    assert {acc.name: f.name for acc, f in inner.accum_scale.items()} == {
        "rowsum_0": "exp_0_alpha", "mma_1": "exp_0_alpha"}
    # every statistic is float32: the start of the max is no float16 / bfloat16 value
    for node in (t.score, t.row_max, t.max_prev, t.max_step, t.alpha):
        assert node.out_dtype == "float32", node.name
    assert t.row_max.accum_init == NEG_FLT_MAX and isinstance(t.row_max, ReduceMax)
    assert type(t.alpha) is Exp2, "alpha uses the base of P = exp2(s - m)"
    # the masked exp is the shifted form of the class the paged tests look for; the max is
    # taken over the scaled score, and the exp reads both
    exp = next(n for n in inner.nodes if isinstance(n, MaskedExp))
    assert exp is t.exp and exp.shift and exp.scale is None
    assert type(exp) is (CausalMaskedExp if causal else MaskedExp)
    assert type(t.score) is (CausalMaskedScore if causal else MaskedScore)
    assert [e.src_node.name for e in exp.inputs] == ["score_0", "exp_0_max"]
    assert t.row_max.inputs[0].src_node is t.score
    assert t.score.scalar_args[-1] == "scale2" and "scale2" not in exp.scalar_args
    assert t.score.scalar_args[:-1] == exp.scalar_args, "one mask for the score and the exp"
    # no plain exp and no multiply in the tier; the one subtraction is the step of the max
    assert not [n.name for n in inner.nodes if type(n) in (Exp, Mul)]
    assert [n.name for n in inner.nodes if type(n) is Sub] == ["exp_0_max_step"]
    # the lse reads the row sum and the carried max, both through the loop's accumulators
    assert type(t.lse) is (MaskedLse2 if causal else Lse2)
    assert [(e.accum_node.name, e.src_id) for e in t.lse.inputs] == [
        ("rowsum_0", 0), ("exp_0_max", 2)]


def test_emission_order():
    t = build_paged_tier()
    order = [n.name for n in emission_order(t.inner)]
    at = {name: i for i, name in enumerate(order)}
    assert at["score_0"] < at["exp_0_max_prev"] < at["exp_0_max"] < at["exp_0"]
    assert at["exp_0_max"] < at["exp_0_max_step"] < at["exp_0_alpha"] < at["rowsum_0"]
    assert at["exp_0_alpha"] < at["mma_1"]


def test_decode_tier_text():
    t, code = paged_tier_code()
    for macro in ("GraphNode_0_iter", "GraphNode_0"):
        assert _signature(code, macro)[-3:] == ["t_rowsum_0", "t_mma_1", "t_exp_0_max"]

    body = _calls(code, "GraphNode_0_iter")
    calls = [s for s in body if s.startswith(("ukernel_", "T.copy"))]
    assert calls == [
        "T.copy(PlaceHolder_K, PlaceHolder_KK)",
        "ukernel_mma_ssr_transB_clear(PlaceHolder_QQ, PlaceHolder_KK, t_mma_0)",
        # s = mask ? score * scale2 : -FLT_MAX
        "ukernel_maskedscore_scale_rt2_2dims_rr(t_mma_0, tile_base, valid_len, scale2, t_score_0)",
        "ukernel_copy_1dims_rr(t_exp_0_max, t_exp_0_max_prev)",  # m_prev = m (lag node)
        "ukernel_reduce_max_axis1_rr(t_score_0, t_exp_0_max)",  # m = max(m, rowmax(s))
        # P = mask ? exp2(s - m) : 0
        "ukernel_maskedexp_shift2_2dims_rrr(t_score_0, t_exp_0_max, tile_base, valid_len, t_exp_0)",
        "ukernel_sub_1dims_rrr(t_exp_0_max_prev, t_exp_0_max, t_exp_0_max_step)",
        "ukernel_exp2_1dims_rr(t_exp_0_max_step, t_exp_0_alpha)",  # alpha = exp2(m_prev - m)
        "ukernel_rescale_1dims_rr(t_exp_0_alpha, t_rowsum_0)",  # l *= alpha
        "ukernel_reduction_axis1_rr(t_exp_0, t_rowsum_0)",  # l += rowsum(P)
        "T.copy(PlaceHolder_V, PlaceHolder_VV)",
        "T.copy(t_exp_0, t_exp_0_cast)",
        "ukernel_rescale_lead1_2dims_rr(t_exp_0_alpha, t_mma_1)",  # O *= alpha[row]
        "ukernel_mma_ssr(t_exp_0_cast, PlaceHolder_VV, t_mma_1)",  # O += P @ V
    ]
    # carried buffers are parameters: never allocated inside the loop body
    assert not any(s.startswith(("t_rowsum_0 =", "t_mma_1 =", "t_exp_0_max =")) for s in body)

    # the loop runs a runtime number of tiles; with zero of them the max keeps its start
    loop = _calls(code, "GraphNode_0")
    assert any(s.startswith("for i0 in T.serial(T.ceildiv(valid_len, 64)):") for s in loop)

    parent = _calls(code, "main_kernel_iter")
    assert [s for s in parent if s.startswith(("T.clear", "T.fill"))] == [
        "T.clear(t_rowsum_0)", "T.clear(t_mma_1)", "T.fill(t_exp_0_max, -3.4028234663852886e38)",
    ]
    epilogue = [s for s in parent if s.startswith("ukernel_")]
    assert epilogue == [
        "ukernel_div_broadcast_0_dfloor_1eminus_30_2dims_rrg(t_mma_1, t_rowsum_0, OutputNode_0)",
        "ukernel_lse2_1dims_rrg(t_rowsum_0, t_exp_0_max, OutputNode_1)",  # log2(l) + m
    ]
    # a masked score is the finite start value of the max, never -inf; a masked P is 0.0
    assert "infinity" not in code
    assert code.count("-3.4028234663852886e38") == 2
    assert "T.exp2(A[m0, m1] - M[m0]), 0.0)" in code
    # the row sum is read unfloored by the lse; the floor guards the divide only
    assert "B[i0] = T.log2(A[i0]) + M[i0]" in code
    assert code.count("T.max(") == 1 and "T.max(B[i0], 1e-30)" in code


def test_causal_tier_text():
    _, code = paged_tier_code(causal=True)
    body = _flat(code, "GraphNode_0_iter")
    statements = [
        "ukernel_causal_maskedscore_scale_rt2_mgrp4_2dims_rr"
        "(t_mma_0,tile_base,valid_len,diag,scale2,t_score_0)",
        "ukernel_copy_1dims_rr(t_exp_0_max,t_exp_0_max_prev)",
        "ukernel_reduce_max_axis1_rr(t_score_0,t_exp_0_max)",
        "ukernel_causal_maskedexp_shift2_mgrp4_2dims_rrr"
        "(t_score_0,t_exp_0_max,tile_base,valid_len,diag,t_exp_0)",
        "ukernel_sub_1dims_rrr(t_exp_0_max_prev,t_exp_0_max,t_exp_0_max_step)",
        "ukernel_exp2_1dims_rr(t_exp_0_max_step,t_exp_0_alpha)",
        "ukernel_rescale_1dims_rr(t_exp_0_alpha,t_rowsum_0)",
        "ukernel_reduction_axis1_rr(t_exp_0,t_rowsum_0)",
        "ukernel_rescale_lead1_2dims_rr(t_exp_0_alpha,t_mma_1)",
        "ukernel_mma_ssr(t_exp_0_cast,PlaceHolder_VV,t_mma_1)",
    ]
    at = [body.index(s) for s in statements]  # each is there ...
    assert at == sorted(at)  # ... in this order
    # the score and the exp are masked by one predicate
    mask = "(tile_base + m1 < valid_len) & (tile_base + m1 <= diag + m0 // 4)"
    assert code.count(mask) == 2
    assert _flat(code, "main_kernel_iter").endswith(
        "ukernel_masked_lse2_mgrp4_1dims_rrg(t_rowsum_0,t_exp_0_max,n_rows,OutputNode_1)")
    assert "B[i0 // 4, i0 % 4] = T.log2(A[i0]) + M[i0]" in code


@pytest.mark.parametrize("causal", [False, True], ids=["decode", "causal"])
def test_paged_tier_lowers_without_a_gpu(causal):
    cuda = lower_to_cuda(paged_tier_code(causal)[1])
    # the carried max starts at the most negative finite float32, and a masked score is it
    assert cuda.count("-0x1.fffffep+127f") == 2
    # the snapshot is taken before the max of this tile is folded in ...
    snapshot = cuda.index("t_exp_0_max_prev[", cuda.index("t_exp_0_max_prev[") + 1)
    fold = cuda.index("tl::MaxOp")
    assert snapshot < fold
    # ... P and alpha are base-2 exponentials of differences against the folded max ...
    shifted = cuda.index("exp2f(", fold)
    alpha = cuda.index("t_exp_0_alpha[", cuda.index("exp2f(t_exp_0_max_step["))
    assert fold < shifted < alpha
    assert "expf(" not in cuda.replace("exp2f(", "")
    # ... and both accumulators are multiplied by alpha before this tile is added
    assert alpha < cuda.index("tl::SumOp")
    assert len(re.findall(r"\* t_exp_0_alpha\[", cuda)) == 2
    # the lse adds the carried max to the log of the unfloored row sum
    assert re.search(r"= \(log2f\(\w+\[\w+\]\) \+ \w+\[\w+\]\);", cuda)
