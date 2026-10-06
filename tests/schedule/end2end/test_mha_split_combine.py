from cleave.schedule import search_tiling

from tests._graphs import mha_split_combine_graph
from tests.schedule.end2end._test_utils import run_end2end_candidates


DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"


def test_mha_split_combine():
    H_Q, SPLIT_K, S_Q, D = 32, 4, 4096, 64
    main_graph = mha_split_combine_graph(H_Q, SPLIT_K, S_Q, D)

    candidates = search_tiling(main_graph, topk=1000)
    run_end2end_candidates(__file__, candidates, DTYPE, min_success=1)
