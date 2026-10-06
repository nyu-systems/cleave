from tests._graphs import mha_bhsd_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import mha_bhsd

DTYPE = "float16"


def test_mha_bhsd():
    dims = (1, 32, 4096, 4096, 64)
    B, H, S_Q, S_KV, D = dims
    graph = mha_bhsd.get_handwritten_scheduled_graph(dims=dims)
    run_handwritten_test(
        graph,
        source_graph=mha_bhsd_graph(B, H, S_Q, S_KV, D),
        dtype=DTYPE,
        kernel_name="mha_bhsd",
    )
