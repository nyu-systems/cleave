from tests.graph.handwritten import _paged_workloads as paged_workloads
from tests.graph.handwritten._test_utils import GATHER_TOLERANCE, run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import (
    gqa_ragged_prefill_causal_h32_kv8_d128,
)

DTYPE = "bfloat16"
OPERATOR = "gqa_ragged_prefill_causal_h32_kv8_d128"


def test_gqa_ragged_prefill_causal_h32_kv8_d128():
    dims = (32, 8, 128)  # num_qo_heads, num_kv_heads, head_dim
    graph = gqa_ragged_prefill_causal_h32_kv8_d128.get_handwritten_scheduled_graph(
        dims=dims
    )
    run_handwritten_test(
        graph,
        dtype=DTYPE,
        kernel_name=OPERATOR,
        workloads_fn=lambda: paged_workloads.cases(OPERATOR, dims),
        out_idx=[6, 7],
        tolerance=GATHER_TOLERANCE,
    )
