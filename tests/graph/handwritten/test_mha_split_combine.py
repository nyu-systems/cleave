from tests._graphs import mha_split_combine_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import mha_split_combine

DTYPE = "float16"


def test_mha_split_combine():
    dims = (32, 4, 4096, 64)
    H_Q, split_k, S_Q, D = dims

    graph = mha_split_combine.get_handwritten_scheduled_graph(dims=dims)

    run_handwritten_test(
        graph,
        source_graph=mha_split_combine_graph(H_Q, split_k, S_Q, D),
        dtype=DTYPE,
        kernel_name="mha_split_combine",
    )
