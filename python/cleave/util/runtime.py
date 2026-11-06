from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import signal
import shutil
import tempfile
import time
import warnings

from .._cache_env import (
    configure_cache_env,
    get_cleave_runtime_root,
)

configure_cache_env()

import tilelang as tl

from .env import _create_parallel_env, _gpu_exec_guard
from .graph_executor import (
    build_inputs_and_reference,
    graph_out_indices,
    make_inputs,
    order_inputs_for_kernel,
    reconcile_output_layout,
)


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..paging import PagingSpec


@dataclass(frozen=True)
class CompilerPassOptions:
    tc_awareness_pass: bool = True
    mla_decode_pass: bool = False
    # When set, the scheduled dense attention graph is wrapped in a paged/ragged access
    # layer before codegen (see cleave.paging). The paging pass introduces placeholders
    # and an lse output, so it runs INSTEAD of the annotate_* chain below, which it
    # applies itself in the order those additions require.
    paging: "PagingSpec | None" = None


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float
    equal_nan: bool = False


# The historical bound every caller used before tolerance became per-operator. Kept as
# the default so existing kernels keep comparing exactly as loosely as they used to.
DEFAULT_TOLERANCE = Tolerance(rtol=0.5, atol=0.5)


def _live_bytes(tensors):
    return sum(t.element_size() * t.nelement() for t in tensors)


def _effective_compile_and_run_parallelism(max_parallel, io_bytes):
    """Cap parallelism so concurrent workers fit in GPU memory.

    ``io_bytes`` is the per-candidate input+output footprint measured from the
    tensors the parent already built. Fused kernels keep large intermediates
    on-chip, so I/O bytes model a worker's persistent GPU footprint.
    """
    try:
        import torch

        if not torch.cuda.is_available() or not io_bytes:
            return max_parallel

        free_bytes, _ = torch.cuda.mem_get_info()
        usable_bytes = int(free_bytes * 0.8)
        suggested = max(1, usable_bytes // io_bytes)
        return max(1, min(max_parallel, suggested))
    except Exception:
        return max_parallel


def _compile_and_run_mp_context():
    return mp.get_context("spawn")


def _read_optional_positive_float_env(name, default):
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        value = float(raw_value)
    except ValueError:
        return default

    if value <= 0:
        return None
    return value


def _read_bool_env(name, default=False):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    return default


def _compilation_progress_log_enabled():
    return _read_bool_env("COMPILATION_PROGRESS_LOG", default=False)


def _compile_and_run_timeout_secs():
    return _read_optional_positive_float_env(
        "CLEAVE_COMPILE_AND_RUN_TIMEOUT_SECS", 300.0
    )


def _compile_progress_heartbeat_secs():
    return _read_optional_positive_float_env(
        "CLEAVE_COMPILE_PROGRESS_HEARTBEAT_SECS", 15.0
    )


def _candidate_progress_label(candidate_idx, total_candidates, cand):
    return (
        f"candidate {candidate_idx}/{total_candidates} "
        f"(rstep={cand.rstep_map}, output_tile={cand.output_tile})"
    )


def _format_active_compile_jobs(active_jobs, now, limit=3):
    slowest_jobs = sorted(
        active_jobs.values(),
        key=lambda job: now - job["started_at"],
        reverse=True,
    )
    return "; ".join(
        f"{job['label']} running {now - job['started_at']:.1f}s"
        for job in slowest_jobs[:limit]
    )


def _close_process(process):
    if process is None:
        return

    try:
        process.join(timeout=0.1)
    except AssertionError:
        return

    close = getattr(process, "close", None)
    if close is None:
        return
    try:
        close()
    except ValueError:
        pass


def _terminate_process_tree(process):
    if process is None or process.pid is None:
        return

    killed = False
    if hasattr(os, "getpgid") and hasattr(os, "killpg"):
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
                killed = True
        except (PermissionError, ProcessLookupError):
            killed = True

    if not killed and process.is_alive():
        process.kill()

    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)


def _compile_and_run_worker(
    result_queue, candidate_idx, code_str, out_idx, reference_cache_dir,
):
    import traceback

    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass

    try:
        cuda_src = compile_and_run(
            code_str, out_idx=out_idx, reference_cache_dir=reference_cache_dir
        )
        result_queue.put(
            {
                "candidate_idx": candidate_idx,
                "ok": True,
                "cuda_src": cuda_src,
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "candidate_idx": candidate_idx,
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _restore_compile_error(result, label):
    error_type = result.get("error_type", "RuntimeError")
    error_message = result.get("error_message", "")
    traceback_str = result.get("traceback", "").strip()

    parts = [f"{label} failed with {error_type}"]
    if error_message:
        parts.append(error_message)
    if traceback_str:
        parts.append(traceback_str)
    return RuntimeError("\n".join(parts))


# How many launches one captured CUDA graph holds. Replaying the graph in batches
# amortises the replay's own launch cost over many kernel launches.
_BENCHMARK_GRAPH_BATCH = 100


def _capture_launch_graph(kernel, inputs, batch):
    """A CUDA graph of ``batch`` back-to-back ``kernel(*inputs)`` launches, or None.

    Capturing ordinary Python calls works with any TileLang execution backend, and
    replaying the graph keeps Python dispatch out of the timed region. Returns None
    when the kernel cannot be captured, so the caller can fall back.
    """
    import torch

    try:
        graph = torch.cuda.CUDAGraph()
        # Capture needs its own stream, and the kernel must have run once already so
        # any lazy initialisation or allocation happens outside the capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            kernel(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            for _ in range(batch):
                kernel(*inputs)
    except Exception:  # noqa: BLE001 - capture is an optimisation, not a requirement
        return None
    return graph


def benchmark_compiled_kernel(
    kernel,
    input_tensors,
    *,
    warmup_iters=0,
    profile_iters=1,
):
    """Mean milliseconds per launch of ``kernel(*input_tensors)``.

    Launches are captured into a CUDA graph and timed by replaying it, so the reading
    is GPU time without per-call Python dispatch, whatever TileLang execution backend
    compiled the kernel. If the kernel cannot be captured, this falls back to timing a
    plain launch loop and warns, because that reading includes dispatch overhead and
    is not comparable with graph-replay readings.
    """
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be >= 0, got {warmup_iters}")
    if profile_iters <= 0:
        raise ValueError(f"profile_iters must be > 0, got {profile_iters}")

    import torch

    inputs = list(input_tensors)
    for _ in range(warmup_iters):
        kernel(*inputs)
    torch.cuda.synchronize()

    batch = min(profile_iters, _BENCHMARK_GRAPH_BATCH)
    graph = _capture_launch_graph(kernel, inputs, batch)
    if graph is not None:
        replays = -(-profile_iters // batch)  # ceil: time at least profile_iters launches
        launches = replays * batch
        graph.replay()  # untimed: the first replay also uploads the graph
        torch.cuda.synchronize()

        def run():
            for _ in range(replays):
                graph.replay()

    else:
        warnings.warn(
            "CUDA graph capture failed for this kernel; timing a plain launch loop, "
            "which includes per-call Python dispatch overhead",
            RuntimeWarning,
            stacklevel=2,
        )
        launches = profile_iters

        def run():
            for _ in range(launches):
                kernel(*inputs)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    run()
    ender.record()
    torch.cuda.synchronize()

    return starter.elapsed_time(ender) / launches


def _compile(prim_func, out_idx):
    """tl.compile under whatever pass configs the generated kernel declared."""
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


_REFERENCE_FILE_NAME = "reference.pt"


def _reference_cache_file(reference_cache_dir):
    return Path(reference_cache_dir) / _REFERENCE_FILE_NAME


def publish_reference(reference_cache_dir, graph, dtype):
    """Build the (inputs, reference) once in the parent and persist them to disk.

    The cleave ``graph`` wraps TVM objects that need not cross the ``spawn``
    boundary; every candidate worker instead loads the plain tensors written here.
    GPU work is serialized by ``_gpu_exec_guard``. Returns ``(out_idx, io_bytes)``
    where ``io_bytes`` is the input+output footprint used to size parallelism.
    """
    import torch

    with _gpu_exec_guard():
        inputs_by_name, reference, out_idx = build_inputs_and_reference(graph, dtype)
        path = _reference_cache_file(reference_cache_dir)
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        os.makedirs(reference_cache_dir, exist_ok=True)
        torch.save(
            {"inputs_by_name": inputs_by_name, "reference": list(reference)}, tmp_path
        )
        os.replace(tmp_path, path)
        io_bytes = _live_bytes(inputs_by_name.values()) + _live_bytes(reference)
        del inputs_by_name, reference
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return out_idx, io_bytes


def _load_reference(reference_cache_dir):
    import torch

    payload = torch.load(
        _reference_cache_file(reference_cache_dir),
        map_location="cuda",
        weights_only=True,
    )
    return dict(payload["inputs_by_name"]), list(payload["reference"])


def _as_list(reference):
    return list(reference) if isinstance(reference, (tuple, list)) else [reference]


def _validate_kernel_outputs(
    kernel, inputs, reference, *, tolerance: Tolerance = DEFAULT_TOLERANCE
):
    """Run ``kernel(*inputs)`` and assert it matches ``reference`` within tolerance.

    Shared comparison core: reconciles each reference to the kernel's physical
    output shape and compares at ``tolerance``, defaulting to the historical loose
    0.5/0.5 bound. Must run inside a ``_gpu_exec_guard``.
    """
    import torch

    actual = kernel(*inputs)
    actual = list(actual) if isinstance(actual, (tuple, list)) else [actual]
    assert len(actual) == len(reference), (
        f"Kernel returned {len(actual)} outputs but the reference has "
        f"{len(reference)}."
    )
    reference = [
        reconcile_output_layout(ref, act.shape)
        for ref, act in zip(reference, actual)
    ]
    torch.testing.assert_close(
        actual,
        reference,
        rtol=tolerance.rtol,
        atol=tolerance.atol,
        equal_nan=tolerance.equal_nan,
    )
    torch.cuda.synchronize()
    del actual, reference
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _run_and_validate_kernel(
    kernel, *, graph=None, dtype=None, reference_cache_dir=None,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
):
    """Run a compiled Tilelang ``kernel`` against the graph-executor reference.

    Either ``reference_cache_dir`` (worker mode: load the parent-built tensors) or
    ``graph`` + ``dtype`` (standalone mode: build the reference in-process) must be
    provided. Shared execution/validation core for both :func:`compile_and_run` and
    :func:`compile_and_run_prim_func`.
    """
    with _gpu_exec_guard():
        if reference_cache_dir is not None:
            inputs_by_name, reference = _load_reference(reference_cache_dir)
        else:
            inputs_by_name, reference, _ = build_inputs_and_reference(graph, dtype)

        inputs = order_inputs_for_kernel(inputs_by_name, kernel)
        _validate_kernel_outputs(kernel, inputs, reference, tolerance=tolerance)

    return kernel.get_kernel_source()


def compile_and_run(
    code_str, *, graph=None, dtype=None, out_idx=None, reference_cache_dir=None,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
):
    """Compile a generated kernel and validate it against the graph reference.

    Worker mode passes ``out_idx`` + ``reference_cache_dir`` (the parent already
    built the reference). Standalone callers pass ``graph`` + ``dtype`` and the
    reference is built in-process.
    """
    _create_parallel_env()

    if out_idx is None:
        if graph is None:
            raise ValueError("compile_and_run requires either out_idx or graph.")
        out_idx = graph_out_indices(graph)

    main_kernel = _load_main_kernel_from_code(code_str)
    kernel = _compile(main_kernel, out_idx)

    return _run_and_validate_kernel(
        kernel, graph=graph, dtype=dtype, reference_cache_dir=reference_cache_dir,
        tolerance=tolerance,
    )


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


def compile_and_run_candidates_parallel(
    candidates,
    dtype,
    max_parallel,
    success_info,
    error_graphs,
    errors,
    compiler_pass_options: CompilerPassOptions | None = None,
):
    if compiler_pass_options is None:
        compiler_pass_options = CompilerPassOptions()

    if not candidates:
        return

    mp_context = _compile_and_run_mp_context()
    timeout_secs = _compile_and_run_timeout_secs()
    progress_log_enabled = _compilation_progress_log_enabled()
    heartbeat_secs = (
        _compile_progress_heartbeat_secs() if progress_log_enabled else None
    )
    poll_secs = 1.0

    def format_setting(value):
        if value is None:
            return "disabled"
        return f"{value:g}s"

    from tqdm import tqdm

    from ..codegen import TilelangCodeGen
    from ..schedule import apply_tiling
    from .. import passes

    if compiler_pass_options.paging is not None:
        # The reference this driver validates against is built from `cand.flat_graph`,
        # the DENSE graph: three tensors and one output. A paged kernel takes five
        # tensors, a scale and two outputs, and its gathers and masked softmax have no
        # graph-executor semantics at all (graph_executor raises for PagedGather and
        # MaskedExp), so no reference can be derived here. Validate a paged kernel over
        # the operator's captured workloads instead -- see
        # tests/schedule/end2end/test_gqa_paged_decode_h32_kv8_d128_ps1.py.
        raise NotImplementedError(
            "compile_and_run_candidates_parallel cannot validate a paged kernel: its "
            "reference comes from the dense graph, which has a different signature. "
            "Use cleave.paging.apply_paging with a workload sweep "
            "(cleave.util.runtime.compile_and_run_prim_func(..., workloads_fn=...))."
        )

    def build_code(cand):
        sch_graph = apply_tiling(cand)
        sch_graph = passes.annotate_args(sch_graph, is_main=True)
        sch_graph = passes.annotate_tinfo(sch_graph, is_main=True)
        if compiler_pass_options.tc_awareness_pass:
            sch_graph = passes.annotate_tc_layout(sch_graph)
        sch_graph = passes.annotate_fragment_layout_conflicts(sch_graph)
        if compiler_pass_options.mla_decode_pass:
            sch_graph = passes.annotate_mla_accumulation_layout_conflict(sch_graph)
        return TilelangCodeGen().gen(sch_graph)

    total_candidates = len(candidates)
    if progress_log_enabled:
        print(
            "Compiling and running %d scheduled graphs "
            "(max_parallel=%d, timeout=%s, heartbeat=%s):"
            % (
                total_candidates,
                max_parallel,
                format_setting(timeout_secs),
                format_setting(heartbeat_secs),
            ),
            flush=True,
        )
    else:
        print(
            f"Compiling and running {total_candidates} scheduled graphs:",
            flush=True,
        )

    result_queue = mp_context.Queue()
    candidate_iter = iter(enumerate(candidates, start=1))
    active_jobs = {}

    runtime_root = get_cleave_runtime_root()
    os.makedirs(runtime_root, exist_ok=True)
    reference_root = tempfile.mkdtemp(prefix="ref_", dir=str(runtime_root))
    graph_refs = {}  # id(flat_graph) -> (reference_cache_dir, out_idx, io_bytes)

    def reference_for(cand):
        graph = cand.flat_graph
        key = id(graph)
        if key not in graph_refs:
            cache_dir = os.path.join(reference_root, f"g{len(graph_refs)}")
            os.makedirs(cache_dir, exist_ok=True)
            g_out_idx, g_io_bytes = publish_reference(cache_dir, graph, dtype)
            graph_refs[key] = (cache_dir, g_out_idx, g_io_bytes)
        return graph_refs[key]

    def launch_until_full(pbar):
        while len(active_jobs) < max_parallel:
            try:
                candidate_idx, cand = next(candidate_iter)
            except StopIteration:
                return

            try:
                cand_ref_dir, cand_out_idx, _ = reference_for(cand)
                code_str = build_code(cand)
            except Exception as exc:
                error_graphs.append((cand, cand.rstep_map, cand.output_tile))
                errors.append(exc)
                pbar.update(1)
                continue

            process = mp_context.Process(
                target=_compile_and_run_worker,
                args=(
                    result_queue,
                    candidate_idx,
                    code_str,
                    cand_out_idx,
                    cand_ref_dir,
                ),
            )

            try:
                process.start()
            except Exception as exc:
                error_graphs.append((cand, cand.rstep_map, cand.output_tile))
                errors.append(exc)
                _close_process(process)
                pbar.update(1)
                continue

            active_jobs[candidate_idx] = {
                "candidate": cand,
                "code_str": code_str,
                "label": _candidate_progress_label(
                    candidate_idx, total_candidates, cand
                ),
                "process": process,
                "started_at": time.monotonic(),
            }

    try:
        _, _, io_bytes = reference_for(candidates[0])
        max_parallel = _effective_compile_and_run_parallelism(max_parallel, io_bytes)

        with tqdm(total=total_candidates) as pbar:
            launch_until_full(pbar)
            last_heartbeat_at = time.monotonic()

            while active_jobs:
                wait_timeout = poll_secs
                if heartbeat_secs is not None:
                    elapsed_since_heartbeat = time.monotonic() - last_heartbeat_at
                    wait_timeout = min(
                        wait_timeout,
                        max(0.0, heartbeat_secs - elapsed_since_heartbeat),
                    )

                try:
                    result = result_queue.get(timeout=wait_timeout)
                    pending_results = [result]
                except Empty:
                    pending_results = []

                while True:
                    try:
                        pending_results.append(result_queue.get_nowait())
                    except Empty:
                        break

                for result in pending_results:
                    candidate_idx = result["candidate_idx"]
                    job = active_jobs.pop(candidate_idx, None)
                    if job is None:
                        continue

                    cand = job["candidate"]
                    code_str = job["code_str"]
                    process = job["process"]
                    _close_process(process)

                    if result["ok"]:
                        success_info.append(
                            (
                                cand,
                                cand.rstep_map,
                                cand.output_tile,
                                code_str,
                                result["cuda_src"],
                            )
                        )
                    else:
                        error_graphs.append((cand, cand.rstep_map, cand.output_tile))
                        errors.append(_restore_compile_error(result, job["label"]))
                    pbar.update(1)

                dead_candidates = [
                    candidate_idx
                    for candidate_idx, job in active_jobs.items()
                    if not job["process"].is_alive()
                ]
                if dead_candidates:
                    # A worker can enqueue its result and exit within this scan's
                    # window; drain any just-published results before declaring a
                    # dead worker crashed, so a valid result is never dropped.
                    while True:
                        try:
                            drained = result_queue.get_nowait()
                        except Empty:
                            break
                        drained_idx = drained["candidate_idx"]
                        job = active_jobs.pop(drained_idx, None)
                        if job is None:
                            continue
                        _close_process(job["process"])
                        if drained["ok"]:
                            success_info.append(
                                (
                                    job["candidate"],
                                    job["candidate"].rstep_map,
                                    job["candidate"].output_tile,
                                    job["code_str"],
                                    drained["cuda_src"],
                                )
                            )
                        else:
                            cand = job["candidate"]
                            error_graphs.append(
                                (cand, cand.rstep_map, cand.output_tile)
                            )
                            errors.append(_restore_compile_error(drained, job["label"]))
                        pbar.update(1)

                for candidate_idx in dead_candidates:
                    job = active_jobs.pop(candidate_idx, None)
                    if job is None:
                        continue  # its result was drained above

                    cand = job["candidate"]
                    process = job["process"]
                    error_graphs.append((cand, cand.rstep_map, cand.output_tile))
                    errors.append(
                        RuntimeError(
                            f"{job['label']} exited with code {process.exitcode} "
                            "before returning a result."
                        )
                    )
                    _close_process(process)
                    pbar.update(1)

                now = time.monotonic()
                if timeout_secs is not None:
                    for candidate_idx, job in list(active_jobs.items()):
                        elapsed = now - job["started_at"]
                        if elapsed < timeout_secs:
                            continue

                        _terminate_process_tree(job["process"])
                        cand = job["candidate"]
                        active_jobs.pop(candidate_idx)
                        error_graphs.append((cand, cand.rstep_map, cand.output_tile))
                        errors.append(
                            TimeoutError(
                                f"{job['label']} exceeded {timeout_secs:g}s "
                                f"after {elapsed:.1f}s; terminated worker."
                            )
                        )
                        _close_process(job["process"])
                        pbar.update(1)
                        if progress_log_enabled:
                            print(
                                f"[cleave] timed out {job['label']} after {elapsed:.1f}s",
                                flush=True,
                            )

                if progress_log_enabled and heartbeat_secs is not None and active_jobs:
                    now = time.monotonic()
                    if now - last_heartbeat_at >= heartbeat_secs:
                        print(
                            f"[cleave] {pbar.n}/{total_candidates} done, "
                            f"{len(active_jobs)} running. "
                            f"Slowest: {_format_active_compile_jobs(active_jobs, now)}",
                            flush=True,
                        )
                        last_heartbeat_at = now

                launch_until_full(pbar)

    finally:
        result_queue.close()
        result_queue.join_thread()
        shutil.rmtree(reference_root, ignore_errors=True)


def benchmark_graphs(
    success_info,
    dtype,
    warmup_iters=20000,
    profile_iters=20000,
):
    if len(success_info) == 0:
        raise ValueError("No successful graphs to benchmark.")

    import torch
    from tqdm import tqdm

    print(
        "Benchmarking %d successful graphs (warmup=%d, profiling=%d)"
        % (len(success_info), warmup_iters, profile_iters)
    )


    from .graph_executor import placeholder_order

    graph_inputs = {}  # id(flat_graph) -> (inputs_by_name, out_idx)

    def inputs_for(graph):
        key = id(graph)
        if key not in graph_inputs:
            placeholders = placeholder_order(graph)
            inputs_by_name = {
                ph.name: t for ph, t in zip(placeholders, make_inputs(graph, dtype))
            }
            graph_inputs[key] = (inputs_by_name, graph_out_indices(graph))
        return graph_inputs[key]

    best_idx = -1
    best_latency_ms = float("inf")
    best_info = None

    with tqdm(total=len(success_info)) as pbar:
        for idx, info in enumerate(success_info):

            if len(info) < 4:
                raise ValueError(
                    "Invalid success_info entry at index %d: expected code string."
                    % idx
                )

            inputs_by_name, out_idx = inputs_for(info[0].flat_graph)
            code_str = info[3]
            main_kernel = _load_main_kernel_from_code(code_str)
            kernel = _compile(main_kernel, out_idx)

            latency_ms = benchmark_compiled_kernel(
                kernel,
                order_inputs_for_kernel(inputs_by_name, kernel),
                warmup_iters=warmup_iters,
                profile_iters=profile_iters,
            )

            if latency_ms < best_latency_ms:
                best_latency_ms = latency_ms
                best_info = info
                best_idx = idx

            del kernel
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            pbar.update(1)

    graph_inputs.clear()
    from tilelang.cache.kernel_cache import KernelCache

    KernelCache()._memory_cache.clear()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return best_info, best_latency_ms, best_idx
