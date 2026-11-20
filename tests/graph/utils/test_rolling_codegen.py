"""Emission of a loop tier that declares a rolling softmax (CPU: codegen and lowering).

``Graph.lag_nodes`` and ``Graph.accum_scale`` change two things in the emitted loop body:
the lag node is emitted before the accumulator it snapshots, and each rescaled accumulator
is multiplied in place right before its own macro. With both fields empty the emission
order is ``find_topo_sort`` itself, so every other graph generates the text it did.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.codegen import TilelangCodeGen
from cleave.codegen.kernel import emission_order
from cleave.dynamic import OPERATORS
from cleave.ir.graph_utils import find_topo_sort, graph_tiers

from tests._lowering import lower_to_cuda
from tests.graph._rolling_tier import build_rolling_tier, rolling_tier_code, schedule_rolling_tier
from tests.graph.handwritten._cases import CASES
from tests.graph.handwritten._test_utils import gen_handwritten_code


def _calls(code, macro):
    """The statements of one macro body, one per line, allocations folded onto one line."""
    start = code.index(f"def {macro}(")
    body = code[code.index("):\n", start) + 3:]
    body = body[: body.index("\n\n\n")] if "\n\n\n" in body else body
    return [re.sub(r"\s+", " ", s).strip() for s in re.split(r"\n(?=    \S)", body) if s.strip()]


# The hand-written cases whose loop declares a rolling softmax: the static one, and every
# paged operator -- the softmax of every paged layer is exact.
ROLLING = {"mha_bhsd_rolling", *OPERATORS}
assert len(ROLLING) == 10 and ROLLING <= set(CASES)


# ------------------------------------------------------------------------- emission order
@pytest.mark.parametrize("name", sorted(set(CASES) - ROLLING))
def test_without_the_fields_the_order_is_the_topological_sort(name):
    graphs = CASES[name].graph()
    for graph in graphs if isinstance(graphs, list) else [graphs]:
        for tier, _ in graph_tiers(graph):
            assert not tier.lag_nodes and not tier.accum_scale
            order, topo = emission_order(tier), find_topo_sort(tier.nodes)
            assert len(order) == len(topo) and all(a is b for a, b in zip(order, topo))


@pytest.mark.parametrize("name", sorted(ROLLING))
def test_the_rolling_cases_declare_both_fields_on_one_loop_tier(name):
    """``ROLLING`` is exactly the cases the test above leaves out, and for a reason."""
    tiers = [tier for tier, _ in graph_tiers(CASES[name].graph())]
    declared = [tier for tier in tiers if tier.lag_nodes or tier.accum_scale]
    assert len(declared) == 1 and declared[0].lag_nodes and declared[0].accum_scale
    order = emission_order(declared[0])
    assert {id(n) for n in order} == {id(n) for n in declared[0].nodes}
    at = {id(n): i for i, n in enumerate(order)}
    for lag in declared[0].lag_nodes:  # the snapshot precedes the accumulator it reads
        assert at[id(lag)] < at[id(lag.inputs[0].src_node)]
    for acc, factor in declared[0].accum_scale.items():
        assert at[id(factor)] < at[id(acc)]


def test_rolling_order_puts_the_snapshot_and_the_factor_first():
    t = build_rolling_tier()
    schedule_rolling_tier(t)
    order = emission_order(t.inner)
    at = {id(n): i for i, n in enumerate(order)}

    assert len(order) == len(t.inner.nodes) == len(at)
    assert {id(n) for n in order} == {id(n) for n in t.inner.nodes}
    # the snapshot is taken before the max is updated, although it reads the max
    assert at[id(t.m_prev)] < at[id(t.m)]
    # the factor is ready before each accumulator it multiplies
    assert at[id(t.alpha)] < at[id(t.rowsum)] and at[id(t.alpha)] < at[id(t.psum)]
    # every other edge still orders a reader after its producer
    for node in t.inner.nodes:
        if node is t.m_prev:
            continue
        for edge in node.inputs:
            assert at[id(edge.src_node)] < at[id(node)], (edge.src_node.name, node.name)
    assert [n.name for n in order] == [
        "PlaceHolder_AA", "exp_0_max_prev", "exp_0_max", "exp_0_max_step", "exp_0_alpha",
        "exp_0_shifted", "exp_0", "rowsum_0", "psum_0",
    ]


def test_a_factor_that_needs_its_own_accumulator_is_a_cycle():
    t = build_rolling_tier()
    t.inner.accum_scale[t.m] = t.alpha  # alpha = exp(m_prev - m) cannot precede m
    with pytest.raises(AssertionError, match="emitted before itself"):
        emission_order(t.inner)


# --------------------------------------------------------------------------- emitted text
def test_hand_built_tier_text():
    t, code = rolling_tier_code()
    assert [n.name for n in t.inner.accum_nodes] == ["rowsum_0", "psum_0", "exp_0_max"]

    # accumulator order is the signature order; the carried max is appended last
    for macro in ("GraphNode_0_iter", "GraphNode_0"):
        signature = code[code.index(f"def {macro}("):]
        signature = signature[: signature.index("):")]
        assert re.findall(r"(t_\w+):", signature) == ["t_rowsum_0", "t_psum_0", "t_exp_0_max"]

    body = _calls(code, "GraphNode_0_iter")
    calls = [s for s in body if s.startswith(("ukernel_", "T.copy"))]
    assert calls == [
        "T.copy(PlaceHolder_A, PlaceHolder_AA)",
        "ukernel_copy_1dims_rr(t_exp_0_max, t_exp_0_max_prev)",  # m_prev = m (lag node)
        "ukernel_reduce_max_axis1_rr(PlaceHolder_AA, t_exp_0_max)",  # m = max(m, rowmax)
        "ukernel_sub_1dims_rrr(t_exp_0_max_prev, t_exp_0_max, t_exp_0_max_step)",
        "ukernel_exp_1dims_rr(t_exp_0_max_step, t_exp_0_alpha)",  # alpha = exp(m_prev - m)
        "ukernel_sub_broadcast_0_2dims_rrr(PlaceHolder_AA, t_exp_0_max, t_exp_0_shifted)",
        "ukernel_exp_2dims_rr(t_exp_0_shifted, t_exp_0)",  # P = exp(s - m)
        "ukernel_rescale_1dims_rr(t_exp_0_alpha, t_rowsum_0)",  # l *= alpha
        "ukernel_reduction_axis1_rr(t_exp_0, t_rowsum_0)",  # l += rowsum(P)
        "ukernel_rescale_lead1_2dims_rr(t_exp_0_alpha, t_psum_0)",  # A *= alpha[row]
        "ukernel_accumulate_2dims_rr(t_exp_0, t_psum_0)",  # A += P
    ]
    # carried buffers are parameters: never allocated inside the loop body
    assert not any(s.startswith(("t_rowsum_0 =", "t_psum_0 =", "t_exp_0_max =")) for s in body)

    # the parent tier owns the carried buffers: sums start at zero, the max at -FLT_MAX
    parent = _calls(code, "main_kernel_iter")
    assert [s for s in parent if s.startswith(("T.clear", "T.fill"))] == [
        "T.clear(t_rowsum_0)", "T.clear(t_psum_0)", "T.fill(t_exp_0_max, -3.4028234663852886e38)",
    ]
    assert "GraphNode_0(PlaceHolder_A, t_rowsum_0, t_psum_0, t_exp_0_max)" in parent


def test_one_rescale_op_serves_a_reduce_sum_and_an_accumulate():
    """No accumulating op class has a rescaled variant: the multiply is its own macro, one
    per (accumulator rank, factor rank), emitted once however many accumulators use it."""
    _, code = rolling_tier_code()
    assert re.findall(r"def (ukernel_rescale\w*)\(", code) == [
        "ukernel_rescale_1dims_rr", "ukernel_rescale_lead1_2dims_rr"]
    # the accumulators' own macros are today's, in their accumulating (non-clear) form
    for macro in ("ukernel_reduction_axis1_rr", "ukernel_accumulate_2dims_rr",
                  "ukernel_reduce_max_axis1_rr"):
        assert code.count(f"def {macro}(") == 1, macro
    assert "T.reduce_sum(A_ptr, B_ptr, dim=-1, clear=False)" in code
    assert "T.reduce_max(A_ptr, B_ptr, dim=-1, clear=False)" in code
    assert "B[i0, i1] += A[i0, i1]" in code


def test_the_declaration_is_what_adds_the_rescale():
    """Same nodes, fields left empty: no rescale, and the snapshot is taken too late."""
    t = build_rolling_tier(declare=False)
    code = TilelangCodeGen().gen(schedule_rolling_tier(t))
    assert "rescale" not in code
    calls = [s for s in _calls(code, "GraphNode_0_iter") if s.startswith("ukernel_")]
    assert calls.index("ukernel_reduce_max_axis1_rr(PlaceHolder_AA, t_exp_0_max)") < calls.index(
        "ukernel_copy_1dims_rr(t_exp_0_max, t_exp_0_max_prev)")


def test_clone_of_a_rolling_tier_generates_the_same_text():
    t = build_rolling_tier()
    main = schedule_rolling_tier(t)
    clone = main.clone()
    # Codegen mutates its graph, so the clone goes first: the original must not notice.
    assert TilelangCodeGen().gen(clone) == TilelangCodeGen().gen(main)


def test_rolling_handwritten_kernel_differs_from_the_fixed_shift_one_only_by_the_update():
    rolling = gen_handwritten_code(CASES["mha_bhsd_rolling"].graph())
    fixed = gen_handwritten_code(CASES["mha_bhsd"].graph())
    body = [s for s in _calls(rolling, "GraphNode_0_iter") if s.startswith(("ukernel_", "T.copy"))]
    base = [s for s in _calls(fixed, "GraphNode_0_iter") if s.startswith(("ukernel_", "T.copy"))]
    assert base == [
        "T.copy(PlaceHolder_B, PlaceHolder_BB)",
        "ukernel_mma_ssr_transB_clear(PlaceHolder_AA, PlaceHolder_BB, t_mma_0)",
        "ukernel_mul_const_0_125_2dims_rr(t_mma_0, t_mul_0)",
        "ukernel_sub_const_6_25_2dims_rr(t_mul_0, t_sub_0)",
        "ukernel_exp_2dims_rr(t_sub_0, t_exp_0)",
        "T.copy(PlaceHolder_C, PlaceHolder_CC)",
        "T.copy(t_exp_0, t_exp_0_cast)",
        "ukernel_mma_rsr(t_exp_0_cast, PlaceHolder_CC, t_mma_1)",
        "ukernel_reduction_axis2_rr(t_exp_0, t_rowsum_0)",
    ]
    assert body == [
        "T.copy(PlaceHolder_B, PlaceHolder_BB)",
        "ukernel_mma_ssr_transB_clear(PlaceHolder_AA, PlaceHolder_BB, t_mma_0)",
        "ukernel_mul_const_0_125_2dims_rr(t_mma_0, t_mul_0)",
        "ukernel_copy_1dims_rr(t_exp_0_max, t_exp_0_max_prev)",
        "ukernel_reduce_max_axis2_rr(t_mul_0, t_exp_0_max)",
        "ukernel_sub_broadcast_0_2dims_rrr(t_mul_0, t_exp_0_max, t_exp_0_shifted)",
        "ukernel_exp_2dims_rr(t_exp_0_shifted, t_exp_0)",
        "ukernel_sub_1dims_rrr(t_exp_0_max_prev, t_exp_0_max, t_exp_0_max_step)",
        "ukernel_exp_1dims_rr(t_exp_0_max_step, t_exp_0_alpha)",
        "T.copy(PlaceHolder_C, PlaceHolder_CC)",
        "T.copy(t_exp_0, t_exp_0_cast)",
        "ukernel_rescale_lead1_2dims_rr(t_exp_0_alpha, t_mma_1)",  # after the operand cast,
        "ukernel_mma_rsr(t_exp_0_cast, PlaceHolder_CC, t_mma_1)",  # right before the gemm
        "ukernel_rescale_1dims_rr(t_exp_0_alpha, t_rowsum_0)",
        "ukernel_reduction_axis2_rr(t_exp_0, t_rowsum_0)",
    ]
    assert "T.fill(t_exp_0_max, -3.4028234663852886e38)" in rolling and "T.fill" not in fixed


# ------------------------------------------------------------------------------- lowering
def _hand_built():
    return rolling_tier_code()[1]


def _hand_written():
    return gen_handwritten_code(CASES["mha_bhsd_rolling"].graph())


@pytest.mark.parametrize(
    "source", [_hand_built, _hand_written], ids=["hand_built", "mha_bhsd_rolling"])
def test_rolling_kernel_lowers_without_a_gpu(source):
    cuda = lower_to_cuda(source())
    # the carried max starts at the most negative finite float32, not at -inf
    assert "-0x1.fffffep+127f" in cuda
    # the snapshot is taken before the max of this tile is folded in ...
    snapshot = cuda.index("t_exp_0_max_prev[")
    snapshot = cuda.index("t_exp_0_max_prev[", snapshot + 1)  # past the declaration
    fold = cuda.index("tl::MaxOp")
    assert snapshot < fold
    # ... and both accumulators are multiplied by alpha before the sum of this tile is added
    alpha = cuda.index("t_exp_0_alpha[", cuda.index("expf(t_exp_0_max_step["))
    assert fold < alpha < cuda.index("tl::SumOp")
    assert len(re.findall(r"\* t_exp_0_alpha\[", cuda)) == 2
