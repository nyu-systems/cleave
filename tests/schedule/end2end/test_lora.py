from cleave.schedule import search_tiling

from tests._graphs import lora_graph
from tests.schedule.end2end._test_utils import run_end2end_candidates


DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"


def test_lora():
    # Mirage LoRA update O = X@W + (X@A)@B (rank R, no scaling); dims follow
    # demo/lora.py. M and R stay 16 so the down-projection X@A keeps its
    # (M, R) = (16, 16) output tile -- the narrow tile that only schedules at a
    # low warp count. Few candidates survive (the (16,16) gemm forces 32 threads),
    # so min_success is 1, as for other tightly constrained fused kernels.
    M, N, K, R = 16, 4096, 4096, 16
    main_graph = lora_graph(M, N, K, R)

    candidates = search_tiling(main_graph, topk=1000)
    run_end2end_candidates(__file__, candidates, DTYPE, min_success=1)
