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
    # M = 256: two instructions a warp group, so codegen stages the A operand through shared
    # (codegen.passes.stage_misplaced_register_operands) and no register operand is read
    assert m_256 == {128, 256}
    assert m_128 == {m for m in m_128 if m >= 64} and len(m_128) > 1  # one warpgroup: M >= 64


def test_an_inter_output_gemm_is_judged_by_the_rows_of_one_iteration():
    """``dQ = dS @ k`` of the attention backward reads ``dS`` from registers. Summed across
    thread blocks (``acc=[inter,intra,intra]``), its block spans the whole of S_Q (4096 rows)
    while the gemm codegen emits multiplies one step of the S_Q loop. Judged by the block,
    every such tiling passed at 128 threads (1024 rows a warp: the steps of 32 rows, 8 a
    warp, failed to lower with ``Layout infer conflict``) and was let through at 256 threads
    as staged (codegen stages by the rows of the step, so none was, and all failed the same
    way): 16 of the 36 tilings of demo/mha_backward. By the iteration's rows only steps of 64
    rows at 128 threads remain, and every tiling that is proposed lowers."""
    from tests._lowering import lower_to_cuda
    from cleave.runtime.compile import build_candidate_code
    from tests.schedule.search_tiling import test_accumulation_contexts as AC

    assert current_target().arch.sm_version == 90
    graph = AC._lower(AC.attention_bwd(32, 4096, 4096))
    policy = AC._policy(graph)
    (dq,) = [n for n in _register_a_matmuls(graph) if not n.transpose_A
             and policy._reads_gemm_accumulator(n.inputs[0].src_node)]
    cands = search_tiling(graph, topk=10000)
    rows = {
        (",".join(AC._contexts(c).values()), c.n_threads, int(c.get_tile(dq)[-2]),
         int(c.get_itile(dq)[-2]))
        for c in cands
    }
    assert {r for r in rows if r[0].startswith("inter")} == {("inter,intra,intra", 128, 4096, 64)}
    assert {(t, m) for acc, t, m, _ in rows if acc.startswith("intra")} == {
        (128, 64), (128, 128), (128, 256), (256, 128), (256, 256)}
    for acc in ("inter,intra,intra", "intra,inter,inter"):
        for threads in (128, 256):
            some = [c for c in cands if c.n_threads == threads
                    and ",".join(AC._contexts(c).values()) == acc]
            if some:
                lower_to_cuda(build_candidate_code(some[0]))


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
