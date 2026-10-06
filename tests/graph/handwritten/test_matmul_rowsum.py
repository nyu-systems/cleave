from tests._graphs import matmul_rowsum_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import matmul_rowsum

DTYPE = "float16"


def test_matmul_rowsum():
    dims = (1024, 1024, 1024)
    M, N, K = dims
    graph = matmul_rowsum.get_handwritten_scheduled_graph(dims=dims)
    run_handwritten_test(
        graph,
        source_graph=matmul_rowsum_graph(M, N, K),
        dtype=DTYPE,
        kernel_name="matmul_rowsum",
    )
