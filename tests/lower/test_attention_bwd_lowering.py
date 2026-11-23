"""CPU-only: lowering, scheduling and codegen of the single-group attention-backward kernels."""

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
