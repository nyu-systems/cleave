from cleave.schedule import search_tiling

from tests._graphs import mha_split_partial_graph
from tests.schedule.end2end._test_utils import run_end2end_candidates


DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"


def test_mha_split_partial():
    B, H, SPLIT_K, S_Q, S_KV_PARTIAL, D = 1, 32, 4, 4096, 1024, 64
    main_graph = mha_split_partial_graph(B, H, SPLIT_K, S_Q, S_KV_PARTIAL, D)

    candidates = search_tiling(main_graph, topk=1000)
    run_end2end_candidates(__file__, candidates, DTYPE, min_success=1)
