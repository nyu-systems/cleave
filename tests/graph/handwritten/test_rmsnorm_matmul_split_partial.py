from tests._graphs import rmsnorm_matmul_split_partial_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import rmsnorm_matmul_split_partial

DTYPE = "float16"


def test_rmsnorm_matmul_split_partial():
    dims = (4, 1024, 6144, 1024)
    split_k, M, N, k_partial = dims

    graph = rmsnorm_matmul_split_partial.get_handwritten_scheduled_graph(dims=dims)

    run_handwritten_test(
        graph,
        source_graph=rmsnorm_matmul_split_partial_graph(split_k, M, N, k_partial),
        dtype=DTYPE,
        kernel_name="rmsnorm_matmul_split_partial",
    )
