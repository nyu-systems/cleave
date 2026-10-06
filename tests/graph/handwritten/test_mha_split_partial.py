from tests._graphs import mha_split_partial_graph
from tests.graph.handwritten._test_utils import run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import mha_split_partial

DTYPE = "float16"


def test_mha_split_partial():
    dims = (1, 32, 4, 4096, 1024, 64)
    B, H, split_k, S_Q, S_KV_PARTIAL, D = dims

    graph = mha_split_partial.get_handwritten_scheduled_graph(dims=dims)

    run_handwritten_test(
        graph,
        source_graph=mha_split_partial_graph(B, H, split_k, S_Q, S_KV_PARTIAL, D),
        dtype=DTYPE,
        kernel_name="mha_split_partial",
    )
