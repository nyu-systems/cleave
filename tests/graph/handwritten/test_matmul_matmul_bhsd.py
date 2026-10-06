from tests._graphs import matmul_matmul_bhsd_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import matmul_matmul_bhsd

DTYPE = "float16"


def test_matmul_matmul_bhsd():
    dims = (1, 32, 4096, 4096, 64)
    B, H, S_Q, S_KV, D = dims
    graph = matmul_matmul_bhsd.get_handwritten_scheduled_graph(dims=dims)
    run_handwritten_test(
        graph,
        source_graph=matmul_matmul_bhsd_graph(B, H, S_Q, S_KV, D),
        dtype=DTYPE,
        kernel_name="matmul_matmul_bhsd",
    )
