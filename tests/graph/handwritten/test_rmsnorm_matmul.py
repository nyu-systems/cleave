from tests._graphs import rmsnorm_matmul_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import rmsnorm_matmul

DTYPE = "float16"


def test_rmsnorm_matmul():
    dims = (1024, 6144, 4096)
    M, N, K = dims
    graph = rmsnorm_matmul.get_handwritten_scheduled_graph(dims=dims)
    run_handwritten_test(
        graph,
        source_graph=rmsnorm_matmul_graph(M, N, K),
        dtype=DTYPE,
        kernel_name="rmsnorm_matmul",
    )
