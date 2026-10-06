from tests._graphs import mha_bshd_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import mha_bshd

DTYPE = "float16"


def test_mha_bshd():
    dims = (1, 4096, 4096, 32, 64)
    B, S_Q, S_KV, H, D = dims
    graph = mha_bshd.get_handwritten_scheduled_graph(dims=dims)
    run_handwritten_test(
        graph,
        source_graph=mha_bshd_graph(B, S_Q, S_KV, H, D),
        dtype=DTYPE,
        kernel_name="mha_bshd",
    )
