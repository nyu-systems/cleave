from tests._graphs import swiglu_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import swiglu

DTYPE = "float16"


def test_swiglu():
    dims = (256, 6144, 4096)
    M, N, K = dims
    graph = swiglu.get_handwritten_scheduled_graph(dims=dims)
    run_handwritten_test(
        graph,
        source_graph=swiglu_graph(M, N, K),
        dtype=DTYPE,
        kernel_name="swiglu",
    )
