from cleave.schedule import search_tiling

from tests._graphs import mla_prefill_bhsd_graph
from tests.schedule.end2end._test_utils import run_end2end_candidates


DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"


def test_mla_prefill_bhsd():
    B, H, S_Q, S_KV, HD_IN, HD_OUT = 1, 128, 4096, 4096, 192, 128
    main_graph = mla_prefill_bhsd_graph(B, H, S_Q, S_KV, HD_IN, HD_OUT)

    candidates = search_tiling(main_graph, topk=1000)
    run_end2end_candidates(__file__, candidates, DTYPE, min_success=13)
