from tests.graph.handwritten import _paged_workloads as paged_workloads
from tests.graph.handwritten._test_utils import GATHER_TOLERANCE, run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import (
    gqa_paged_prefill_causal_h32_kv4_d128_ps1,
)

DTYPE = "bfloat16"
OPERATOR = "gqa_paged_prefill_causal_h32_kv4_d128_ps1"


def test_gqa_paged_prefill_causal_h32_kv4_d128_ps1():
    dims = (32, 4, 128)  # num_qo_heads, num_kv_heads, head_dim
    graph = gqa_paged_prefill_causal_h32_kv4_d128_ps1.get_handwritten_scheduled_graph(
        dims=dims
    )
    run_handwritten_test(
        graph,
        dtype=DTYPE,
        kernel_name=OPERATOR,
        workloads_fn=lambda: paged_workloads.cases(OPERATOR, dims),
        out_idx=[7, 8],
        tolerance=GATHER_TOLERANCE,
    )
