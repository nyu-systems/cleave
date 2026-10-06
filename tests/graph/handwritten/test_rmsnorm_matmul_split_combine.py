from tests._graphs import rmsnorm_matmul_split_combine_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import rmsnorm_matmul_split_combine

DTYPE = "float16"


def test_rmsnorm_matmul_split_combine():
    dims = (4, 1024, 6144, 4096)
    split_k, M, N, K = dims

    graph = rmsnorm_matmul_split_combine.get_handwritten_scheduled_graph(dims=dims)

    run_handwritten_test(
        graph,
        source_graph=rmsnorm_matmul_split_combine_graph(split_k, M, N, K),
        dtype=DTYPE,
        kernel_name="rmsnorm_matmul_split_combine",
    )
