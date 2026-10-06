from tests._graphs import matmul_matmul_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import matmul_matmul_small_n

DTYPE = "float16"


def test_matmul_matmul_small_n():
    dims = (1024, 64, 64)
    M, N, K = dims
    graph = matmul_matmul_small_n.get_handwritten_scheduled_graph(dims=dims)
    run_handwritten_test(
        graph,
        source_graph=matmul_matmul_graph(M, N, K),
        dtype=DTYPE,
        kernel_name="matmul_matmul_small_n",
    )
