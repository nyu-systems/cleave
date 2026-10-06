from cleave.schedule import search_tiling

from tests._graphs import matmul_splitk_combine_graph
from tests.schedule.end2end._test_utils import run_end2end_candidates


DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"


def test_matmul_splitk_combine():
    SPLIT_K, M, N = 4, 128, 128
    main_graph = matmul_splitk_combine_graph(SPLIT_K, M, N)

    candidates = search_tiling(main_graph, topk=1000)
    run_end2end_candidates(__file__, candidates, DTYPE, min_success=89)
