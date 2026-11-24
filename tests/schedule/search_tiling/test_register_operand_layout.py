"""sm_90: a gemm reading its A operand from another gemm's accumulator needs >= 16 rows per warp.

TileLang 0.1.13 cannot lay out such a register A operand with fewer than 16 rows of M per warp,
nor at two warpgroups unless M is exactly 128 (16 per warp) (``Layout infer conflict``); the
policy no longer emits those candidates. CPU only.
"""
import copy

import pytest

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
    way): 16 of the 36 tilings of demo/mha_backward. By the iteration's rows, steps of 64
    and 128 rows at 128 threads remain, and of 128 rows at 256 threads (16 a warp); every
    tiling that is proposed lowers."""
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
    assert {r for r in rows if r[0].startswith("inter")} == {
        ("inter,intra,intra", 128, 4096, 64), ("inter,intra,intra", 128, 4096, 128),
        ("inter,intra,intra", 256, 4096, 128)}
    assert {(t, m) for acc, t, m, _ in rows if acc.startswith("intra")} == {
        (128, 64), (128, 128), (128, 256), (256, 128), (256, 256)}
    for acc in ("inter,intra,intra", "intra,inter,inter"):
        for threads in (128, 256):
            some = [c for c in cands if c.n_threads == threads
                    and ",".join(AC._contexts(c).values()) == acc]
            if some:
                lower_to_cuda(build_candidate_code(some[0]))


def _column_sum(x, w):
    """sum_N(exp(w x^T / 8)): a sum over the rows of a gemm's accumulator, as the per-output
    search writes ``rowsum(exp(x w))`` (tests/pipeline_api/test_per_output_gpu.py)."""
    import cleave.lang as cl

    return cl.sum(cl.exp(w @ x.mT * 0.125), 0, keepdim=True)


def test_a_reduction_over_gemm_rows_holds_at_most_16_of_them_a_warp():
    """TileLang's ReduceOp cannot lower a sum (or max) over dim -2 of a gemm accumulator
    fragment whose warps hold more than 16 of the reduced rows ("source index depends on a
    thread-owned reduce segment"). Such tilings are no longer proposed: steps of 128 and 256
    rows at 128 threads, and of 256 at 256 threads (22 of the 144 tilings of
    per_output_pair, none of which lowered, 4 of them in the top 8 of one of its graphs)."""
    import cleave.lang as cl
    from cleave.codegen.passes import wgmma_row_warps
    from cleave.frontend import trace
    from cleave.ir.ops import ReduceSum
    from cleave.pipeline import GraphSet
    from cleave.pipeline.plans import lower
    from cleave.runtime.compile import build_candidate_code
    from tests._lowering import lower_to_cuda

    def column_sum(x: cl.Tensor["M", "K"], w: cl.Tensor["N", "K"]):
        return _column_sum(x, w)

    (graph,) = lower(GraphSet.identity(trace(column_sum, dict(M=128, K=64, N=1024), cl.f16))[0])
    policy = _policy(graph)
    (rowsum,) = [n for n in graph.nodes if isinstance(n, ReduceSum)]
    (gemm,) = [n for n in graph.nodes if isinstance(n, Matmul)]
    cands = search_tiling(graph, topk=10000)

    def rows_a_warp(c):
        (rows,) = [int(s) for s in c.get_rstep(rowsum).values()]
        warps = wgmma_row_warps(gemm.warp_policy, rows, int(c.get_itile(gemm)[-1]), c.n_threads)
        return rows, min(rows // w for w in warps or (c.n_threads // 32,))

    seen = {(c.n_threads, rows_a_warp(c)) for c in cands}
    assert all(rpw <= 16 for _, (_, rpw) in seen)
    assert {(t, rows) for t, (rows, _) in seen if rows >= 64} == {(128, 64), (256, 64), (256, 128)}
    # the limit itself lowers: 16 rows a warp at 128 and at 256 threads
    for threads, rows in ((128, 64), (256, 128)):
        c = next(c for c in cands if c.n_threads == threads and rows_a_warp(c)[0] == rows)
        lower_to_cuda(build_candidate_code(c))
    # ... and one step more does not: the same tiling with 128 rows a step at 128 threads,
    # built by hand, is refused, and fails to lower with the error the rule stands for
    c = next(c for c in cands if c.n_threads == 128 and rows_a_warp(c)[0] == 64)
    assert not policy._row_reduction_layout_conflict(c, 128)
    rstep_map = {n: dict(s) for n, s in c.rstep_map.items()}
    (name,) = rstep_map[rowsum]
    rstep_map[rowsum][name] = 128
    over = policy.compute_tile_dict(list(c.output_tile), rstep_map)
    over.n_threads = 128
    assert policy._row_reduction_layout_conflict(over, 128)
    with pytest.raises(Exception, match="thread-owned reduce segment"):
        lower_to_cuda(build_candidate_code(over))


def test_a_softmax_over_256_queries_has_no_tiling_and_says_why():
    """The max and the sum of a softmax over the queries (``dim=-2``) reduce the rows of the
    scores' accumulator. At 64 queries a block holds 16 of them a warp; at 256 every tiling
    holds 32 or 64 (none of the 144 of the searched graph lowered), so there is no plan,
    and the refusal names the limit."""
    import cleave.lang as cl
    from cleave.frontend import trace
    from cleave.pipeline import GraphSet, ScheduleConfig, schedule
    from cleave.schedule.policy.default import ROW_REDUCTION_CONFLICT

    def over_queries(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                     v: cl.Tensor["H", "SKV", "D"]):
        return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-2) @ v

    def plans(sq):
        return schedule(GraphSet.identity(trace(over_queries, dict(H=4, SQ=sq, SKV=256, D=64),
                                                cl.f16)), ScheduleConfig(topk=100000))

    assert len(plans(64)) > 0
    refused = plans(256)
    assert len(refused) == 0
    ((reason),) = {str(r.error) for r in refused.refused}
    assert f"x {ROW_REDUCTION_CONFLICT}" in reason


def test_two_staged_accumulators_of_64_rows_keep_two_warp_groups():
    """``dV += P^T dO`` and ``dK += dS^T q`` in one loop over a block of 64 keys on 256
    threads: proposed, like every other assignment. (Ten of them once validated wrong on an
    H100; the cause was a shared operand placed off the 1024-byte boundary its swizzle
    needs, not the thread count: tests/codegen/test_operand_names.py.)"""
    from cleave.pipeline import GraphSet, ScheduleConfig, schedule
    from tests.lower import _attention_bwd as A

    ref = dict(H=32, SQ=4096, SKV=4096, D=64)

    def plans(name):
        return list(schedule(GraphSet.identity(A.program(name, ref)), ScheduleConfig(topk=1000)))

    def wide(p):
        return p.n_threads == 256 and list(p.output_tile) == [1, 64, 64]

    for name, count, n_wide in (("dkdv", 24, 6), ("dkdv_o", 24, 6), ("fused_o", 26, 2)):
        found = plans(name)
        assert (len(found), sum(wide(p) for p in found)) == (count, n_wide), name
    for name, count in (("dk", 16), ("dv", 8)):
        assert sum(wide(p) for p in plans(name)) == count, name


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
