from tests._graphs import matmul_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import matmul_splitk_partial
from tests.graph.handwritten.handwritten_schedules import matmul_splitk_combine

DTYPE = "float16"


def test_matmul_splitk():
    split_dims = (4, 128, 128, 1024)
    combine_dims = (4, 128, 128)
    split_k, M, N, K_partial = split_dims
    K = split_k * K_partial
    split_graph = matmul_splitk_partial.get_handwritten_scheduled_graph(dims=split_dims)
    combine_graph = matmul_splitk_combine.get_handwritten_scheduled_graph(
        dims=combine_dims
    )

    run_handwritten_test(
        [split_graph, combine_graph],
        dtype=DTYPE,
        kernel_name="matmul_splitk",
        scratch_source_graph=matmul_graph(M, N, K),
    )
