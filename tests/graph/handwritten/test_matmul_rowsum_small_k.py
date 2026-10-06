from tests._graphs import matmul_rowsum_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import matmul_rowsum_small_k

DTYPE = "float16"


def test_matmul_rowsum_small_k():
    dims = (4096, 4096, 32)
    M, N, K = dims
    graph = matmul_rowsum_small_k.get_handwritten_scheduled_graph(dims=dims)
    run_handwritten_test(
        graph,
        source_graph=matmul_rowsum_graph(M, N, K),
        dtype=DTYPE,
        kernel_name="matmul_rowsum_small_k",
    )
