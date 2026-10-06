from tests._graphs import batched_matmul_splitk_partial_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import batched_matmul_splitk_partial

DTYPE = "float16"


def test_batched_matmul_splitk_partial():
    dims = (2, 4, 128, 128, 1024)
    B, SPLIT_K, M, N, K_PARTIAL = dims
    graph = batched_matmul_splitk_partial.get_handwritten_scheduled_graph(dims=dims)
    run_handwritten_test(
        graph,
        source_graph=batched_matmul_splitk_partial_graph(B, SPLIT_K, M, N, K_PARTIAL),
        dtype=DTYPE,
        kernel_name="batched_matmul_splitk_partial",
    )
