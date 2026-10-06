from tests.graph.handwritten import _paged_workloads as paged_workloads
from tests.graph.handwritten._test_utils import GATHER_TOLERANCE, run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import (
    gqa_paged_decode_h32_kv8_d128_ps64,
)

DTYPE = "bfloat16"
OPERATOR = "gqa_paged_decode_h32_kv8_d128_ps64"


def test_gqa_paged_decode_h32_kv8_d128_ps64():
    dims = (32, 8, 128, 64)  # num_qo_heads, num_kv_heads, head_dim, page_size
    graph = gqa_paged_decode_h32_kv8_d128_ps64.get_handwritten_scheduled_graph(
        dims=dims
    )
    run_handwritten_test(
        graph,
        dtype=DTYPE,
        kernel_name=OPERATOR,
        workloads_fn=lambda: paged_workloads.cases(OPERATOR, dims, page_size=64),
        out_idx=[7, 8],
        tolerance=GATHER_TOLERANCE,
    )
