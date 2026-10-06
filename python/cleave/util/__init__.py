from __future__ import annotations

from typing import TYPE_CHECKING

from .env import _create_parallel_env, _gpu_exec_guard
from .graph_utils import (
    clone_graph,
    find_topo_sort,
    gather_all_nodes,
    get_op_ancestors,
    lock_graph,
    unlock_graph,
)
from .graph_executor import (
    build_inputs_and_reference,
    execute,
    graph_out_indices,
    make_inputs,
)
from .autotune import (
    TIE_BAND,
    WorkloadCandidate,
    benchmark_over_workloads,
    compile_and_validate_over_workloads,
    format_ranking,
    geomean,
    pick_best,
    rank,
    tie_band_members,
    sampled,
    time_kernel,
    workload_key,
)
from .runtime import (
    CompilerPassOptions,
    DEFAULT_TOLERANCE,
    Tolerance,
    benchmark_compiled_kernel,
    benchmark_graphs,
    compile_and_run,
    compile_and_run_candidates_parallel,
    compile_and_run_prim_func,
    publish_reference,
)
from .tensor_codegen import (
    gen_copy,
    gen_tensor_allocation_str,
    gen_tensor_type,
    get_offset_lambda,
)

if TYPE_CHECKING:
    from .splitk import (
        SplitKCompileSummary,
        SplitKCompileResult,
        SplitKStageBenchmarkResult,
        SplitKStageSpec,
        make_mnk_splitk_stage_specs,
        run_splitk_compile,
    )


_SPLITK_EXPORTS = {
    "SplitKStageSpec",
    "SplitKStageBenchmarkResult",
    "SplitKCompileResult",
    "SplitKCompileSummary",
    "make_mnk_splitk_stage_specs",
    "run_splitk_compile",
}

__all__ = [
    "TIE_BAND",
    "WorkloadCandidate",
    "benchmark_over_workloads",
    "compile_and_validate_over_workloads",
    "format_ranking",
    "geomean",
    "pick_best",
    "rank",
    "tie_band_members",
    "sampled",
    "time_kernel",
    "workload_key",
    "_create_parallel_env",
    "_gpu_exec_guard",
    "find_topo_sort",
    "gen_tensor_type",
    "gen_copy",
    "gen_tensor_allocation_str",
    "gather_all_nodes",
    "get_op_ancestors",
    "get_offset_lambda",
    "CompilerPassOptions",
    "DEFAULT_TOLERANCE",
    "Tolerance",
    "clone_graph",
    "lock_graph",
    "unlock_graph",
    "build_inputs_and_reference",
    "execute",
    "graph_out_indices",
    "make_inputs",
    "publish_reference",
    "compile_and_run",
    "compile_and_run_candidates_parallel",
    "compile_and_run_prim_func",
    "benchmark_compiled_kernel",
    "benchmark_graphs",
    "SplitKStageSpec",
    "SplitKStageBenchmarkResult",
    "SplitKCompileResult",
    "SplitKCompileSummary",
    "make_mnk_splitk_stage_specs",
    "run_splitk_compile",
]


def __getattr__(name: str):
    if name in _SPLITK_EXPORTS:
        from . import splitk as _splitk

        value = getattr(_splitk, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SPLITK_EXPORTS)
