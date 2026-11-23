"""sm_90, 256 threads: a gemm reading its A operand from another gemm's accumulator needs M = 128.

TileLang 0.1.13 cannot lay out such a register A operand at two warpgroups unless the block's M
is 128 (``Layout infer conflict``); the policy no longer emits those candidates. CPU only.
"""
import copy

from cleave.core.target import current_target
from cleave.ir.graph import ComputeOpNode, OutputNode
from cleave.ir.ops import Exp, Matmul
from cleave.schedule import search_tiling
from cleave.schedule.policy.default import TCPolicy

from tests._graphs import (
    ACCUM_DTYPE, DTYPE, OUT_DTYPE, _connect, _finalize, mha_bhsd_graph, mla_decode_bhsd_graph,
)


def _register_a_matmuls(graph):
    return [n for n in graph.nodes if isinstance(n, Matmul) and isinstance(n.inputs[0].src_node, ComputeOpNode)]


def _exp_matmul_graph(M, N, K):
    """exp(A) @ B: the A operand is computed from a loaded tile only."""
    exp = Exp((M, K), dtype=DTYPE, out_dtype=DTYPE, name="exp_0")
    mma = Matmul((M, N, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_0", warp_policy="rows")
    _connect(exp, mma, 0, 0)
    OutputNode(mma, dtype=OUT_DTYPE)
    return _finalize(exp, mma)


def _policy(graph):
    outputs = [n for n in graph.nodes if n.is_output()]
    return TCPolicy(outputs, current_target().arch, flat_graph=graph)


def test_two_warpgroups_read_a_gemm_accumulator_only_at_block_m_128():
    assert current_target().arch.sm_version == 90
    graph = mha_bhsd_graph(1, 2, 1024, 1024, 64)
    (pv,) = _register_a_matmuls(graph)  # exp(q k^T) @ v
    cands = search_tiling(graph, topk=10000)
    m_256 = {int(c.get_tile(pv)[-2]) for c in cands if c.n_threads == 256}
    m_128 = {int(c.get_tile(pv)[-2]) for c in cands if c.n_threads == 128}
    assert m_256 == {128}
    assert len(m_128) > 1  # one warpgroup: every M stays


def test_a_operand_computed_from_loads_keeps_every_block_m():
    graph = _exp_matmul_graph(1024, 1024, 1024)
    (mma,) = _register_a_matmuls(graph)
    cands = search_tiling(graph, topk=10000)
    assert {int(c.get_tile(mma)[-2]) for c in cands if c.n_threads == 256} - {128}


def test_mla_decode_value_gemm_is_not_refused():
    # mla_decode_pass stages its softmax operand through shared: such candidates compile and
    # pass the end2end test (10 of them would have been refused), so they stay.
    graph = mla_decode_bhsd_graph(1, 128, 1024, 512, 64)
    (pv,) = [n for n in _register_a_matmuls(graph) if not n.transpose_B]
    cands = search_tiling(graph, topk=10000)
    assert {int(c.get_tile(pv)[-2]) for c in cands if c.n_threads == 256} - {128}


def test_refusal_is_specific_to_sm_90_and_two_warpgroups():
    graph = mha_bhsd_graph(1, 2, 1024, 1024, 64)
    policy = _policy(graph)
    (pv,) = _register_a_matmuls(graph)
    cand = next(c for c in search_tiling(graph, topk=10000)
                if c.n_threads == 128 and int(c.get_tile(pv)[-2]) != 128)
    cand = copy.copy(cand)
    assert policy._register_operand_layout_conflict(cand, 256)
    assert not policy._register_operand_layout_conflict(cand, 128)
    policy.arch = copy.copy(policy.arch)
    policy.arch.sm_version = 80
    assert not policy._register_operand_layout_conflict(cand, 256)
