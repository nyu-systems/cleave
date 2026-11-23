"""sm_90: a gemm reading its A operand from another gemm's accumulator needs >= 16 rows per warp.

TileLang 0.1.13 cannot lay out such a register A operand with fewer than 16 rows of M per warp,
nor at two warpgroups unless M is exactly 128 (16 per warp) (``Layout infer conflict``); the
policy no longer emits those candidates. CPU only.
"""
import copy

from cleave.core.target import current_target
from cleave.ir.graph import ComputeOpNode, OutputNode
from cleave.ir.ops import Exp, Matmul
from cleave.schedule import search_tiling
from cleave.schedule.policy.default import TCPolicy

from tests._graphs import (
    ACCUM_DTYPE, DTYPE, OUT_DTYPE, _connect, _finalize, lora_graph, mha_bhsd_graph,
    mla_decode_bhsd_graph,
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
    assert m_128 == {m for m in m_128 if m >= 64} and len(m_128) > 1  # one warpgroup: M >= 64


def test_lora_up_projection_needs_16_rows_per_warp():
    # (X@A)@B with M = 16: only one warp (32 threads) gives it 16 rows.
    graph = lora_graph(16, 4096, 4096, 16)
    assert {c.n_threads for c in search_tiling(graph, topk=10000)} == {32}


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
                if c.n_threads == 128 and int(c.get_tile(pv)[-2]) == 64)
    cand = copy.copy(cand)
    assert policy._register_operand_layout_conflict(cand, 256)  # 8 rows per warp
    assert not policy._register_operand_layout_conflict(cand, 128)  # 16 rows per warp
    assert policy._register_operand_layout_conflict(cand, 64 * 32)  # 1 row per warp
    policy.arch = copy.copy(policy.arch)
    policy.arch.sm_version = 80
    assert not policy._register_operand_layout_conflict(cand, 256)


def test_a_operand_staged_through_shared_is_not_refused():
    # The paged layers stage P through shared for the value matmul (all but ragged prefill), so
    # their decode tilings at 16 query rows compile (tests/frontend_dynamic); `stage_as_layer`
    # marks the dense body that way before the search. Unmarked, the same body keeps none.
    from cleave.dynamic import OPERATORS
    from cleave.dynamic.dense import KV_PROXY, flash_attention_sd_graph
    from cleave.dynamic.paging import stage_as_layer

    def threads(spec):
        graph = flash_attention_sd_graph(16, KV_PROXY, 128, dtype=DTYPE, out_dtype=DTYPE)
        if spec is not None:
            graph = stage_as_layer(graph, spec)
        return {c.n_threads for c in search_tiling(graph, topk=10000)}

    decode = OPERATORS["gqa_paged_decode_h32_kv8_d128_ps1"].spec
    ragged = OPERATORS["gqa_ragged_prefill_causal_h32_kv8_d128"].spec
    assert threads(None) == set()
    assert threads(decode) == {128, 256}
    assert threads(ragged) == set()  # ragged prefill reads P from registers
