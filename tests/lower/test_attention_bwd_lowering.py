"""CPU-only: lowering, scheduling and codegen of the single-group attention-backward kernels."""

import re

import pytest

from cleave.codegen.tensor_codegen import gen_tensor_allocation_str
from cleave.lower.convert import _infer_broadcast_index
from cleave.pipeline import GraphSet, ScheduleConfig, schedule
from cleave.runtime.compile import build_candidate_code
from tests.lower import _attention_bwd as A


def _plans(name, topk=8):
    return list(schedule(GraphSet.identity(A.program(name)), ScheduleConfig(topk=topk)))


@pytest.mark.parametrize("other, expect", [
    ((2, 256), 0), ((2, 256, 1), 0),  # a row vector, with or without its keepdim axis
    ((2, 128), 1), ((2, 1, 128), 1),  # a column vector
    ((2, 256, 128), -1), ((2, 128, 1), -1),
])
def test_keepdim_operand_is_a_broadcast(other, expect):
    assert _infer_broadcast_index((2, 256, 128), other) == expect


def test_log_sum_exp_input_lowers_as_row_broadcast():
    code = build_candidate_code(_plans("dv", 1)[0].tile)
    assert "ukernel_sub_broadcast_0_2dims" in code
    assert "ukernel_sub_2dims" not in code


def test_rank1_shared_tile_is_not_swizzled():
    assert "make_swizzled_layout" not in gen_tensor_allocation_str("x", (32,), "float32", "s", True)
    assert "make_swizzled_layout" in gen_tensor_allocation_str("x", (32, 64), "float16", "s", True)


@pytest.mark.parametrize("name", ["dq", "dk"])
def test_input_read_twice_keeps_one_tile(name):
    # k (dq) / q (dk) feeds the score over the whole head dim and the gradient matmul over the
    # output's head-dim tile: only a whole-head-dim output tile gives both the same tile.
    plans = _plans(name, topk=40)
    assert plans
    assert all(p.output_tile[-1] == 64 for p in plans)


def test_transposed_register_operand_is_staged_in_shared():
    code = build_candidate_code(_plans("dv", 1)[0].tile)
    # P^T @ dO: P is computed in registers and read transposed
    assert "ukernel_mma_ssr_transA" in code
    assert "ukernel_mma_rsr_transA" not in code


def test_in_loop_row_sum_broadcast_into_gemm_is_staged_in_shared():
    # dP - sum_D(dO * o): the row sum of plain loads broadcast into a gemm accumulator
    code = build_candidate_code(_plans("dkdv_o", 1)[0].tile)
    assert re.search(r"(t_reduction_\w+)_intermediate = T\.alloc_shared\(\(32,\).*\n"
                     r"\s*T\.copy\(\1, \1_intermediate\)\n"
                     r"\s*ukernel_sub_broadcast_0_\w+\(t_mma_\d+, \1_intermediate,", code)


def test_given_row_operand_is_not_staged():
    code = build_candidate_code(_plans("dkdv", 1)[0].tile)
    assert "_intermediate = T.alloc_shared((32,)" not in code


def _shared_input_pair():
    """Y1 = a @ u and Y2 = a^T @ x2: the input a has two consumers, one reads it transposed."""
    import cleave.lang as cl
    from cleave.frontend import Program

    p = Program("pair_in", dtype=cl.f16)
    p.dims(M=256, N=256, K=64)
    a, u, x2 = p.input("a", ("M", "N")), p.input("u", ("N", "K")), p.input("x2", ("M", "K"))
    p.output(a @ u)
    p.output(a.mT @ x2)
    return p


def test_multi_consumer_input_read_transposed_stays_in_shared():
    # a multi-consumer input is otherwise loaded into registers; the transposed gemm read of
    # that fragment is a TileLang layout conflict ("Get different layout for A_ptr")
    plans = list(schedule(GraphSet.identity(_shared_input_pair()), ScheduleConfig(topk=4)))
    assert plans
    for plan in plans:
        code = build_candidate_code(plan.tile)
        assert "ukernel_mma_ssr_transA" in code
        assert "ukernel_mma_rsr_transA" not in code
