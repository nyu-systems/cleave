from cleave.schedule import search_tiling
from cleave.util import CompilerPassOptions

from tests._graphs import mla_decode_bhsd_graph
from tests.schedule.end2end._test_utils import run_end2end_candidates


DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"


def test_mla_decode_bhsd():
    B, H, S_KV, HD, PE = 1, 128, 4096, 512, 64
    main_graph = mla_decode_bhsd_graph(B, H, S_KV, HD, PE)

    candidates = search_tiling(main_graph, topk=1000)
    run_end2end_candidates(
        __file__,
        candidates,
        DTYPE,
        min_success=5,
        compiler_pass_options=CompilerPassOptions(mla_decode_pass=True),
    )
