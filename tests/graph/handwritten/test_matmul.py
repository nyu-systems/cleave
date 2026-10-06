from tests._graphs import matmul_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import matmul

DTYPE = "float16"


def test_matmul():
    M, N, K = 128, 128, 4096
    dims = (M, N, K)
    graph = matmul.get_handwritten_scheduled_graph(dims=dims)
    run_handwritten_test(
        graph,
        source_graph=matmul_graph(M, N, K),
        dtype=DTYPE,
        kernel_name="matmul",
    )
