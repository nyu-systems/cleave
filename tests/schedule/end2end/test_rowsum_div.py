from cleave.schedule import search_tiling

from tests._graphs import rowsum_div_graph
from tests.schedule.end2end._test_utils import run_end2end_candidates


DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"


def test_rowsum_div():
    S_Q, S_KV = 4096, 4096
    main_graph = rowsum_div_graph(S_Q, S_KV)

    candidates = search_tiling(main_graph, topk=1000)
    run_end2end_candidates(__file__, candidates, DTYPE, min_success=1)
