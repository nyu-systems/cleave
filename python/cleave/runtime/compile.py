"""Compile generated or hand-written kernels and check them against the graph reference."""

from __future__ import annotations

from dataclasses import dataclass
import tempfile

import tilelang as tl

from .gpu import _create_parallel_env, _gpu_exec_guard
from .inter_layout import BLOCKED_LAYOUT, unblock_kernel
from .reference import graph_out_indices
from .validate import (
    BenchOptions,
    _as_list,
    _check_kernel,
    _run_and_validate_kernel,
    _stage,
    _validate_kernel_outputs,
    require_tolerance,
)


@dataclass(frozen=True)
class CompilerPassOptions:
    tc_awareness_pass: bool = True
    mla_decode_pass: bool = False


def _compile(prim_func, out_idx):
    """tl.compile under whatever pass configs the generated kernel declared.

    A kernel with outputs summed across CTAs (``INTER_OUTPUTS``) comes back wrapped in
    an :class:`InterOutputKernel`, which owns their fp32 buffers.
    """
    _create_parallel_env()
    configs = getattr(prim_func, "_cleave_pass_configs", None)
    inter = getattr(prim_func, "_cleave_inter_outputs", None)
    if not inter:
        return tl.compile(prim_func, out_idx=out_idx, pass_configs=configs or None)
    if out_idx is None:
        raise NotImplementedError(
            "A kernel with inter (atomically accumulated) outputs allocates them itself; "
            "it cannot be compiled with caller-owned outputs (out_idx=None)."
        )
    positions = _inter_positions(prim_func, inter, out_idx)
    kernel = tl.compile(
        prim_func,
        out_idx=[i for i in _normalized_out_idx(prim_func, out_idx) if i not in positions],
        pass_configs=configs or None,
    )
    names = {prim_func.buffer_map[prim_func.params[i]].name: i for i in positions}
    layouts = {
        names[name]: layout
        for name, layout in getattr(prim_func, "_cleave_inter_layouts", {}).items()
    }
    return InterOutputKernel(kernel, prim_func, out_idx, positions, layouts)


def _normalized_out_idx(prim_func, out_idx) -> list[int]:
    n = len(prim_func.params)
    out_idx = [out_idx] if isinstance(out_idx, int) else list(out_idx)
    return sorted(i % n for i in out_idx)


def _inter_positions(prim_func, inter, out_idx) -> dict[int, tuple]:
    """``{param index: (shape, accumulation torch dtype, output torch dtype)}``."""
    from .reference import torch_dtype

    out_idx = _normalized_out_idx(prim_func, out_idx)
    positions = {}
    for idx, param in enumerate(prim_func.params):
        buffer = prim_func.buffer_map.get(param)
        if buffer is None or buffer.name not in inter:
            continue
        if idx not in out_idx:
            raise ValueError(f"inter output {buffer.name!r} is not among out_idx {out_idx}")
        shape = tuple(int(d) for d in buffer.shape)  # static: codegen refuses symbolic
        positions[idx] = (shape, torch_dtype(buffer.dtype), torch_dtype(inter[buffer.name]))
    missing = set(inter) - {
        prim_func.buffer_map[prim_func.params[i]].name for i in positions
    }
    if missing:
        raise ValueError(f"INTER_OUTPUTS names {sorted(missing)}, which are not parameters")
    return positions


class InterOutputKernel:
    """A compiled kernel whose inter outputs are summed across CTAs with atomic adds.

    Each call zeroes a fresh fp32 buffer per inter output, launches the kernel (which
    takes those buffers as parameters), then casts them to the output dtype. All three
    run on the current stream, so a CUDA graph captured around a call replays the zeroing
    and the cast too: timing covers them, and no replay sees a previous launch's sums.
    Callers see TileLang's interface: inputs in, outputs (declared dtypes) out, and an
    ``adapter`` whose ``result_idx`` counts the inter outputs as results.

    ``layouts`` (``{param index: layout}``, from ``INTER_LAYOUTS``) names the buffers the
    kernel stores in another layout than row-major; their cast also restores the order
    (see ``unblock_kernel``). It is compiled here, never inside a CUDA-graph capture.
    """

    def __init__(self, kernel, prim_func, out_idx, positions, layouts=None):
        self.kernel = kernel
        self.positions = positions
        self.out_idx = _normalized_out_idx(prim_func, out_idx)
        self.adapter = _InterAdapter(kernel.adapter, prim_func, self.out_idx)
        self.casts = {}
        for idx, layout in (layouts or {}).items():
            if layout != BLOCKED_LAYOUT:
                raise ValueError(f"unknown inter output layout {layout!r}")
            shape, acc, dtype = positions[idx]
            self.casts[idx] = unblock_kernel(shape, acc, dtype)

    def __call__(self, *inputs):
        import torch

        device = next(x for x in inputs if isinstance(x, torch.Tensor)).device
        sums = {
            idx: torch.zeros(shape, dtype=acc, device=device)
            for idx, (shape, acc, _) in sorted(self.positions.items())
        }
        res = self.kernel(*inputs, *sums.values())
        if len(self.out_idx) == len(self.positions):
            res = []
        own = iter(res if isinstance(res, (list, tuple)) else [res])
        outs = [
            self._cast(idx, sums[idx]) if idx in sums else next(own)
            for idx in self.out_idx
        ]
        return outs[0] if len(outs) == 1 else outs

    def _cast(self, idx, buffer):
        cast = self.casts.get(idx)
        if cast is None:
            return buffer.to(self.positions[idx][2])
        shape = buffer.shape
        return cast(buffer.view(-1, 8, 8)).view(shape)

    def __getattr__(self, name):
        return getattr(self.kernel, name)


class _InterAdapter:
    """The wrapped kernel's adapter, with the inter outputs counted as results."""

    def __init__(self, adapter, prim_func, out_idx):
        self._adapter = adapter
        self.prim_func = prim_func
        self.result_idx = list(out_idx)

    def __getattr__(self, name):
        return getattr(self._adapter, name)


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
        # Outputs summed across CTAs, ``{param name: output dtype}`` (see _compile).
        main_kernel._cleave_inter_outputs = getattr(tmp_module, "INTER_OUTPUTS", {})
        # Their buffers stored in another layout than row-major, ``{param name: layout}``.
        main_kernel._cleave_inter_layouts = getattr(tmp_module, "INTER_LAYOUTS", {})

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
    tolerance=None,
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

    ``tolerance`` is the operator's declared bound (a ``Tolerance``, or one per output);
    there is no default, so leaving it out is an error.
    """
    require_tolerance(tolerance, "this prim func")
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
    tolerance=None, benchmark: BenchOptions | None = None, derive=None,
):
    """``(cuda_src, latency_ms | None, measured)``: compile, validate and optionally time
    one kernel; ``measured`` is the error per output the comparison found.

    Workers (the default ``check_fn``) pass ``out_idx`` + ``reference_cache_dir`` (the
    parent built the reference); standalone callers pass ``graph`` + ``dtype`` (and the
    ``derive`` of ``cleave.runtime.reference.declared_inputs_hook`` for declared inputs).
    ``tolerance`` is the operator's declared bound and has no default: a kernel nobody
    declared a tolerance for fails here, before it is compiled.
    """
    with _stage("validate"):
        require_tolerance(tolerance)
    _create_parallel_env()
    if out_idx is None:
        if graph is None:
            raise ValueError("check_code requires either out_idx or graph.")
        out_idx = graph_out_indices(graph)

    with _stage("compile"):
        kernel = _compile(_load_main_kernel_from_code(code_str), out_idx)

    measured, latency_ms = _check_kernel(
        kernel, graph=graph, dtype=dtype, reference_cache_dir=reference_cache_dir,
        tolerance=tolerance, benchmark=benchmark, derive=derive,
    )
    return kernel.get_kernel_source(), latency_ms, measured
