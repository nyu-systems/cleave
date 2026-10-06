"""The whole pipeline, end to end: dense search -> tiling -> paging -> a running kernel.

The structural half of this is asserted in ``tests/graph/apply_paging/``; what this adds
is that the kernel the pipeline emits is numerically right over the operator's captured
workloads, held to the same references and the same 1e-2 tolerance as the hand-written
schedule for the same operator. The operator name is deliberately shared with
``tests/graph/handwritten/test_gqa_paged_decode_h32_kv8_d128_ps1.py`` so both are checked
against identical inputs.
"""

from __future__ import annotations

from tests.graph.apply_paging._test_utils import build_paged_graph
from tests.graph.handwritten import _paged_workloads as paged_workloads
from tests.graph.handwritten._test_utils import GATHER_TOLERANCE
from cleave.codegen import TilelangCodeGen
from cleave.paging import PagingSpec
from cleave.util.runtime import _load_main_kernel_from_code, compile_and_run_prim_func

DTYPE = "bfloat16"
OPERATOR = "gqa_paged_decode_h32_kv8_d128_ps1"


def test_gqa_paged_decode_h32_kv8_d128_ps1_pipeline():
    dims = (32, 8, 128)  # num_qo_heads, num_kv_heads, head_dim
    num_qo_heads, num_kv_heads, _ = dims

    graph = build_paged_graph(
        PagingSpec(num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads)
    )
    code = TilelangCodeGen().gen(graph)

    compile_and_run_prim_func(
        _load_main_kernel_from_code(code),
        dtype=DTYPE,
        workloads_fn=lambda: paged_workloads.cases(OPERATOR, dims),
        # Five declared inputs and the scale, then the two outputs.
        out_idx=[6, 7],
        tolerance=GATHER_TOLERANCE,
    )
