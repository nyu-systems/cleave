from tests.graph.handwritten import _paged_workloads as paged_workloads
from tests.graph.handwritten._test_utils import GATHER_TOLERANCE, run_handwritten_test
from tests.graph.handwritten.handwritten_schedules import (
    mla_paged_decode_h16_ckv512_kpe64_ps1,
)

DTYPE = "bfloat16"
OPERATOR = "mla_paged_decode_h16_ckv512_kpe64_ps1"
SM_SCALE = mla_paged_decode_h16_ckv512_kpe64_ps1.SM_SCALE


def test_mla_paged_decode_h16_ckv512_kpe64_ps1():
    dims = (16, 512, 64)  # num_qo_heads, head_dim_ckv, head_dim_kpe
    graph = mla_paged_decode_h16_ckv512_kpe64_ps1.get_handwritten_scheduled_graph(
        dims=dims
    )
    run_handwritten_test(
        graph,
        dtype=DTYPE,
        kernel_name=OPERATOR,
        workloads_fn=lambda: paged_workloads.cases(OPERATOR, dims, scale=SM_SCALE),
        out_idx=[7, 8],
        tolerance=GATHER_TOLERANCE,
    )
