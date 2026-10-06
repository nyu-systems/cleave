from tests._graphs import batched_matmul_splitk_combine_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import batched_matmul_splitk_combine

DTYPE = "float16"


def test_batched_matmul_splitk_combine():
    dims = (2, 4, 128, 128)
    B, SPLIT_K, M, N = dims
    graph = batched_matmul_splitk_combine.get_handwritten_scheduled_graph(dims=dims)
    run_handwritten_test(
        graph,
        source_graph=batched_matmul_splitk_combine_graph(B, SPLIT_K, M, N),
        dtype=DTYPE,
        kernel_name="batched_matmul_splitk_combine",
    )
