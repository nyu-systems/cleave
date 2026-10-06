from tests._graphs import matmul_splitk_partial_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import matmul_splitk_partial

DTYPE = "float16"


def test_matmul_splitk_partial():
    dims = (4, 128, 128, 1024)
    SPLIT_K, M, N, K_PARTIAL = dims
    graph = matmul_splitk_partial.get_handwritten_scheduled_graph(dims=dims)
    run_handwritten_test(
        graph,
        source_graph=matmul_splitk_partial_graph(SPLIT_K, M, N, K_PARTIAL),
        dtype=DTYPE,
        kernel_name="matmul_splitk_partial",
    )
