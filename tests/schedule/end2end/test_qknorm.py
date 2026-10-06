from cleave.schedule import search_tiling

from tests._graphs import qknorm_graph
from tests.schedule.end2end._test_utils import run_end2end_candidates


DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"


def test_qknorm():
    # QKNorm: RMSNorm(Q) and RMSNorm(K) then grouped-query flash attention
    # softmax((Qn @ Kn^T) / sqrt(HD) - 50/sqrt(HD)) @ V. Dims match demo/qknorm.py
    # (GQA H_Q=64 > H_KV=8, HD=128): the fused RMSNorm-into-attention kernel only
    # yields WGMMA-schedulable, numerically-correct candidates at the full
    # sequence length -- small S/tile shapes do not produce a valid tiling. The
    # fused QK^T matmul takes two RMS-normalized (fragment) operands, so it
    # schedules only under forced fusion, and only a handful of the tiling
    # candidates survive the two-fragment staging + multi-reduction fusion, so
    # min_success is 1, as for other tightly constrained fused kernels. topk is
    # capped so the test compiles a bounded number of candidates (the surviving
    # ones rank near the top) rather than the full 1000.
    B, H_Q, H_KV, S_Q, S_KV, HD = 1, 64, 8, 4096, 4096, 128
    main_graph = qknorm_graph(B, H_Q, H_KV, S_Q, S_KV, HD)

    candidates = search_tiling(main_graph, topk=32, force_fusion=True)
    run_end2end_candidates(__file__, candidates, DTYPE, min_success=1)
