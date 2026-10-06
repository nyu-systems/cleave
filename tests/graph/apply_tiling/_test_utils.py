"""Shared helpers for the apply_tiling tests.

Every apply_tiling test builds one candidate from a logged (output_tile, rstep)
choice, runs ``apply_tiling`` on it, and asserts the scheduled graph's structure /
offsets. The candidate construction and the compile-and-run assertion are identical
across ops (modulo which policy applies), so they live here.
"""

from __future__ import annotations

import pytest

from cleave.arch import get_arch
from cleave.graph import OutputNode
from cleave.ops import Matmul
from cleave.policy.default import DefaultPolicy, TCPolicy
from cleave.util import CompilerPassOptions, compile_and_run_candidates_parallel


DTYPE = "float16"


def build_tiled_candidate(main_graph, output_tile, rstep_by_name):
    """Reconstruct the TileDict for a logged (output_tile, rstep) candidate.

    Matmul nodes are tagged for tensor-core scheduling and select ``TCPolicy``;
    graphs without a Matmul (e.g. split-k combine) use ``DefaultPolicy``.
    """
    output_nodes = []
    has_matmul = False
    for node in main_graph.nodes:
        if isinstance(node, Matmul):
            node.set_tag("tensorcore_config", True)
            has_matmul = True
        elif isinstance(node, OutputNode):
            output_nodes.append(node)

    policy_cls = TCPolicy if has_matmul else DefaultPolicy
    policy = policy_cls(output_nodes, get_arch("cuda"), flat_graph=main_graph)

    op_nodes_by_name = {
        node.name: node
        for node in main_graph.nodes
        if not node.is_placeholder() and not node.is_output()
    }
    rstep_map = {op_nodes_by_name[name]: rstep for name, rstep in rstep_by_name.items()}

    cand = policy.compute_tile_dict(output_tile, rstep_map)
    assert cand.valid, "Constructed TileDict is invalid for the logged candidate."
    return cand


def assert_candidate_compiles_and_runs(cand, *, compiler_pass_options=None):
    """Compile+run the single candidate through the graph-executor correctness path.

    ``compiler_pass_options`` forwards op-specific compiler passes (e.g. MLA decode
    needs ``CompilerPassOptions(mla_decode_pass=True)``); the default suits every
    other op.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to compile and run candidates.")

    success_info: list = []
    error_graphs: list = []
    errors: list = []
    compile_and_run_candidates_parallel(
        [cand],
        DTYPE,
        1,
        success_info,
        error_graphs,
        errors,
        compiler_pass_options=compiler_pass_options,
    )

    assert not errors, f"Candidate failed to compile/run: {errors[0]}"
    assert len(success_info) == 1
