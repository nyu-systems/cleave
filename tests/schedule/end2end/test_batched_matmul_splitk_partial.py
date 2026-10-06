from cleave.schedule import search_tiling

from tests._graphs import batched_matmul_splitk_partial_graph
from tests.schedule.end2end._test_utils import run_end2end_candidates


DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"


def test_batched_matmul_splitk_partial():
    B, SPLIT_K, M, N, K_PARTIAL = 2, 4, 128, 128, 1024
    main_graph = batched_matmul_splitk_partial_graph(B, SPLIT_K, M, N, K_PARTIAL)

    candidates = search_tiling(main_graph, topk=1000)
    run_end2end_candidates(__file__, candidates, DTYPE, min_success=1)
