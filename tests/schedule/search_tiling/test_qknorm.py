from collections import Counter

from cleave.core.target import current_target
from cleave.ir.graph import OutputNode
from cleave.schedule import search_tiling
from cleave.schedule.policy.default import TCPolicy

from tests._graphs import qknorm_graph


def test_qknorm():
    # QKNorm (RMSNorm on Q and K, then grouped-query flash attention) fuses two
    # RMSNorm reductions over the head dim with the QK^T / softmax / P@V attention
    # block. Its QK^T matmul takes two computed (RMS-normalized) operands, and the
    # RMSNorm reductions leave rank-2 (batch, seq) intermediates -- the schedule
    # only enumerates candidates for this fused shape under forced fusion. This
    # guards that search_tiling enumerates candidates without crashing.
    B, H_Q, H_KV, S_Q, S_KV, HD = 1, 64, 8, 4096, 4096, 128
    main_graph = qknorm_graph(B, H_Q, H_KV, S_Q, S_KV, HD)

    candidates = search_tiling(main_graph, topk=32, force_fusion=True)

    assert (
        len(candidates) > 0
    ), "Expected to find at least one tiling candidate for QKNorm, but found none."


def test_qknorm_transposed_normalized_key_is_not_charged_a_second_staging_buffer():
    """``q_n @ k_n^T`` reads two computed fragments and no placeholder: codegen stages both
    operands through shared memory (the two-fragment staging, ``t_q_norm_cast`` and
    ``t_k_norm_cast``), whose bytes the gemm's own footprint already charges. The policy's
    mirror of the transposed-operand staging used to charge ``k_n`` once more (32768 bytes
    at 128 keys of D=128 in fp16), which put every 64-query-row tiling at 128 keys and 128
    threads over the shared-memory cap: 27 tilings, 18 of which validated, were lost."""
    graph = qknorm_graph(1, 64, 8, 4096, 4096, 128)
    outputs = [n for n in graph.nodes if isinstance(n, OutputNode)]
    arch = current_target().arch
    policy = TCPolicy(outputs, arch, force_fusion=True, flat_graph=graph)
    template = policy.emit_config(1)[0]
    rstep_map = {k: {ax: 128 for ax in v} for k, v in template.rstep_map.items()}
    td = policy.compute_tile_dict([1, 64, 128], rstep_map)
    own = policy._own_loop_reductions(td)
    (qk,) = [n for n in policy.ordered_nodes if n.name == "mma_0"]
    assert policy._staged_operand_indices(qk, own) == ()
    assert (td.valid, td.smem_cost) == (True, 163840) and td.smem_cost <= arch.smem_cap

    candidates = search_tiling(graph, topk=100000, force_fusion=True)
    by_tile = Counter((tuple(c.output_tile), c.n_threads) for c in candidates)
    assert len(candidates) == 189
    assert {t: by_tile[(t, 128)] for t in ((1, 64, 32), (1, 64, 64), (1, 64, 128))} == {
        (1, 64, 32): 27, (1, 64, 64): 27, (1, 64, 128): 27,
    }
