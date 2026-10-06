from cleave.schedule import search_tiling

from tests._graphs import matmul_matmul_bhsd_graph
from tests.schedule.end2end._test_utils import run_end2end_candidates


DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"


def test_matmul_matmul_bhsd():
    B, H, S_Q, S_KV, D = 1, 32, 4096, 4096, 64
    main_graph = matmul_matmul_bhsd_graph(B, H, S_Q, S_KV, D)

    candidates = search_tiling(main_graph, topk=1000)
    run_end2end_candidates(__file__, candidates, DTYPE, min_success=19)
