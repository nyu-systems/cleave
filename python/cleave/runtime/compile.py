"""Compile generated or hand-written kernels and check them against the graph reference."""

from __future__ import annotations

from dataclasses import dataclass
import tempfile

import tilelang as tl

from .gpu import _create_parallel_env, _gpu_exec_guard
from .reference import graph_out_indices
from .validate import (
    DEFAULT_TOLERANCE,
    BenchOptions,
    Tolerance,
    _as_list,
    _check_kernel,
    _run_and_validate_kernel,
    _stage,
    _validate_kernel_outputs,
)


@dataclass(frozen=True)
class CompilerPassOptions:
    tc_awareness_pass: bool = True
    mla_decode_pass: bool = False


def _compile(prim_func, out_idx):
    """tl.compile under whatever pass configs the generated kernel declared."""
    _create_parallel_env()
    configs = getattr(prim_func, "_cleave_pass_configs", None)
    return tl.compile(prim_func, out_idx=out_idx, pass_configs=configs or None)


def _load_main_kernel_from_code(code_str):
    # write to a tmp file and parse
    import importlib.util
    import uuid

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w") as f:
        f.write(code_str)
        f.flush()

        # import the tmp file as a module
        # use a unique module name to avoid cross-thread collisions in parallel runs.
        module_name = f"tmp_module_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, f.name)
        tmp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tmp_module)  # type: ignore
        main_kernel = tmp_module.main_kernel
        main_kernel._cache_source_hint = code_str
        # A kernel that is only correct under certain tilelang passes carries them with
        # it, so every caller compiles it the same way (see Graph.pass_configs).
        main_kernel._cleave_pass_configs = getattr(tmp_module, "PASS_CONFIGS", {})

    return main_kernel


def compile_and_run_prim_func(
    prim_func,
    *,
    graph=None,
    dtype=None,
    out_idx=None,
    reference_fn=None,
    make_inputs_fn=None,
    workloads_fn=None,
    reference_cache_dir=None,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
):
    """Compile a hand-written prim func and validate it against a reference.

    Reference kernels are hand-written prim funcs rather than generated code. The
    reference is normally built from a cleave ``graph`` via the graph executor;
    layouts the cleave op vocabulary cannot express (e.g. interior-head ``bshd``)
    pass explicit ``make_inputs_fn`` / ``reference_fn`` closures instead.

    A kernel whose tensor extents are symbolic serves many shapes from one compile.
    ``workloads_fn`` validates such a kernel over a sequence of them: it yields
    ``(inputs, reference)`` pairs, one built and freed at a time, because a single
    workload's tensors can run to several GiB.
    """
    _create_parallel_env()

    if workloads_fn is not None:
        if reference_fn is not None or make_inputs_fn is not None:
            raise ValueError(
                "workloads_fn supplies its own inputs; pass it instead of "
                "reference_fn/make_inputs_fn, not alongside them."
            )
        if out_idx is None:
            raise ValueError("workloads_fn requires an explicit out_idx.")
        kernel = _compile(prim_func, out_idx)
        n = 0
        for inputs, reference in workloads_fn():
            # Per workload, not around the loop: the guard is a cross-process lock, so
            # holding it for the whole sequence would stall every other test worker.
            with _gpu_exec_guard():
                _validate_kernel_outputs(
                    kernel, tuple(inputs), _as_list(reference), tolerance=tolerance
                )
            del inputs, reference
            n += 1
        assert n, "workloads_fn yielded no workloads"
        return kernel.get_kernel_source()

    if reference_fn is not None:
        if make_inputs_fn is None:
            raise ValueError("reference_fn requires make_inputs_fn.")
        if out_idx is None:
            raise ValueError("reference_fn requires an explicit out_idx.")
        kernel = _compile(prim_func, out_idx)
        with _gpu_exec_guard():
            inputs = tuple(make_inputs_fn())
            reference = _as_list(reference_fn(*inputs))
            _validate_kernel_outputs(kernel, inputs, reference, tolerance=tolerance)
        return kernel.get_kernel_source()

    if out_idx is None:
        if graph is None:
            raise ValueError("compile_and_run_prim_func requires graph or reference_fn.")
        out_idx = graph_out_indices(graph)

    kernel = _compile(prim_func, out_idx)
    return _run_and_validate_kernel(
        kernel, graph=graph, dtype=dtype, reference_cache_dir=reference_cache_dir,
        tolerance=tolerance,
    )


def build_candidate_code(cand, options: CompilerPassOptions | None = None):
    """Tilelang source for one tiling candidate: the default ``build_fn``."""
    from ..codegen import TilelangCodeGen
    from ..schedule import apply_tiling
    from ..codegen import passes

    options = options or CompilerPassOptions()
    sch_graph = apply_tiling(cand)
    sch_graph = passes.annotate_args(sch_graph, is_main=True)
    sch_graph = passes.annotate_tinfo(sch_graph, is_main=True)
    if options.tc_awareness_pass:
        sch_graph = passes.annotate_tc_layout(sch_graph)
    sch_graph = passes.annotate_fragment_layout_conflicts(sch_graph)
    if options.mla_decode_pass:
        sch_graph = passes.annotate_mla_accumulation_layout_conflict(sch_graph)
    return TilelangCodeGen().gen(sch_graph)


def check_code(
    code_str, *, graph=None, dtype=None, out_idx=None, reference_cache_dir=None,
    tolerance: Tolerance | None = None, benchmark: BenchOptions | None = None,
):
    """``(cuda_src, latency_ms | None)``: compile, validate and optionally time one kernel.

    Workers (the default ``check_fn``) pass ``out_idx`` + ``reference_cache_dir`` (the
    parent built the reference); standalone callers pass ``graph`` + ``dtype``.
    """
    _create_parallel_env()
    if out_idx is None:
        if graph is None:
            raise ValueError("check_code requires either out_idx or graph.")
        out_idx = graph_out_indices(graph)

    with _stage("compile"):
        kernel = _compile(_load_main_kernel_from_code(code_str), out_idx)

    latency_ms = _check_kernel(
        kernel, graph=graph, dtype=dtype, reference_cache_dir=reference_cache_dir,
        tolerance=tolerance or DEFAULT_TOLERANCE, benchmark=benchmark,
    )
    return kernel.get_kernel_source(), latency_ms
