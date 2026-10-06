from tests._graphs import matmul_splitk_combine_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import matmul_splitk_combine

DTYPE = "float16"


def test_matmul_splitk_combine():
    dims = (4, 128, 128)
    SPLIT_K, M, N = dims
    graph = matmul_splitk_combine.get_handwritten_scheduled_graph(dims=dims)
    run_handwritten_test(
        graph,
        source_graph=matmul_splitk_combine_graph(SPLIT_K, M, N),
        dtype=DTYPE,
        kernel_name="matmul_splitk_combine",
    )
