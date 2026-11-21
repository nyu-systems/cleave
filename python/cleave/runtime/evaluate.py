"""Run many tiling candidates through the workers and time the survivors."""

from __future__ import annotations

from dataclasses import dataclass, replace
import functools
from multiprocessing.connection import wait as _wait_ready
import os
import shutil
import tempfile
import time
import traceback

from .._cache_env import get_cleave_runtime_root
from ..core.settings import Settings
from ..core.errors import (
    BenchmarkError,
    CandidateTimeout,
    CleaveError,
    CodegenError,
    CompileError,
    LaunchCrash,
    NonFiniteOutput,
    OutputMismatch,
    SmemExceeded,
    SpecError,
    ValidationError,
    WorkerCrash,
)

from .compile import (
    CompilerPassOptions,
    _compile,
    _load_main_kernel_from_code,
    build_candidate,
    check_code,
)
from .gpu import _create_parallel_env, _gpu_exec_guard, device_lock_state
from .reference import (
    graph_out_indices,
    make_inputs,
    order_inputs_for_kernel,
    output_order,
    declared_output_places,
    output_roles,
    with_hand_off,
)
from .validate import BenchOptions, benchmark_compiled_kernel, publish_reference, tolerances_of
from .workers import WorkerPool


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


@dataclass
class CandidateResult:
    cand: object
    status: str  # "ok" | "failed"
    stage: str | None = None  # codegen|compile|validate|bench|timeout|crash on failure
    error: CleaveError | None = None
    code: str | None = None
    cuda_src: str | None = None
    # Mean time of one launch. For a kernel that stores an output transposed (a split
    # partial) it includes the copy that hands the outputs over in the logical layout
    # (cleave.runtime.reference.with_hand_off): no caller gets them without it.
    latency_ms: float | None = None
    label: str = ""
    # cleave.runtime.validate.MeasuredError per output, when the comparison ran to the end:
    # for every ok candidate, and for one that failed as an OutputMismatch.
    measured: tuple | None = None
    # How the kernel stores each output: ``(axes, pad)`` per output, in the order it returns
    # them (cleave.runtime.reference.to_stored / to_logical), when the build recorded it.
    stored_layouts: tuple | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


_SMEM_MARKER = "Failed to set the allowed dynamic shared memory size"
# How a kernel failed validation, by the name of what the worker raised; anything else at
# the validate stage (a reference or layout problem) stays a plain ValidationError.
_VALIDATION_KINDS = {c.__name__: c for c in (OutputMismatch, NonFiniteOutput, LaunchCrash)}
_TRACEBACK_LINES = 20


def _short_traceback(text):
    lines = text.strip().splitlines()
    if len(lines) > _TRACEBACK_LINES:
        lines = ["..."] + lines[-_TRACEBACK_LINES:]
    return "\n".join(lines)


def _restore_compile_error(result, label):
    """``(stage, typed error)`` for a worker's failure report."""
    error_type = result.get("error_type", "RuntimeError")
    error_message = result.get("error_message", "")
    traceback_str = _short_traceback(result.get("traceback", ""))

    parts = [f"{label} failed with {error_type}"]
    if error_message:
        parts.append(error_message)
    if traceback_str:
        parts.append(traceback_str)
    text = "\n".join(parts)

    stage = result.get("stage") or "compile"
    if _SMEM_MARKER in error_message or _SMEM_MARKER in traceback_str:
        return "compile", SmemExceeded(text)
    if stage == "validate":
        error_cls = _VALIDATION_KINDS.get(error_type, ValidationError)
    else:
        error_cls = BenchmarkError if stage == "bench" else CompileError
    return stage, error_cls(text)


def _parent_error(error_cls, label, what, exc):
    tb = _short_traceback("".join(traceback.format_exception(exc)))
    error = error_cls(f"{label} {what} failed with {type(exc).__name__}\n{exc}\n{tb}")
    error.__cause__ = exc
    return error


def declared_bounds(tolerance, graph):
    """``(declaration, roles)``: the declaration ``tolerance`` as it applies to the outputs
    of a candidate of ``graph``, and their boundary roles when one of them has a role. (The
    roles are only known for a candidate that comes with its lowered graph.)

    A sequence is one bound per output of the program, in the program's order. A kernel
    returns its outputs in the order its graph lists them, which need not be that order
    (``reference.declared_output_places``), so the sequence is handed on in the kernel's.
    """
    known = tolerance is not None and getattr(graph, "nodes", None) is not None
    roles = output_roles(graph) if known else None
    declared = tolerances_of(tolerance, roles)
    if known and isinstance(tolerance, (list, tuple)):
        places = declared_output_places(graph)
        if places is not None and len(places) == len(tolerance):
            declared = tuple(tolerance[k] for k in places)
    return declared, roles if roles and any(roles) else None


def run_candidates(
    candidates,
    dtype,
    *,
    max_parallel: int = 8,
    tolerance=None,
    timeout_s: float | None = None,
    compiler_pass_options: CompilerPassOptions | None = None,
    benchmark: BenchOptions | None = None,
    build_fn=None,
    check_fn=None,
    reference_fn=None,
    pool: WorkerPool | None = None,
) -> list[CandidateResult]:
    """Codegen, compile and validate every candidate; one result each, in input order.

    Compile/validate (and ``benchmark`` timing) run on up to ``max_parallel`` persistent
    workers; codegen stays in the parent (TileDicts can't be pickled). Injectable stages:
    ``build_fn`` (parent), ``check_fn`` (worker, picklable), ``reference_fn``, ``pool``.
    ``build_fn`` returns the source, or ``(source, stored layouts)`` like the default
    ``build_candidate``; the layouts go to ``check_fn`` and onto the result.

    ``tolerance`` is the operator's declared bound (a ``Tolerance``, one per output, or
    one per boundary role: ``validate.tolerances_of``). ``check_fn`` is handed what it comes
    to for the candidate's own outputs; the default ``check_code`` fails a candidate
    without a bound, and so does a declaration that leaves one of its outputs without.
    """
    if not candidates:
        return []
    if build_fn is None:
        options = compiler_pass_options or CompilerPassOptions()
        build_fn = functools.partial(build_candidate, options=options)
    check_fn = check_fn or check_code
    reference_fn = reference_fn or publish_reference

    _create_parallel_env()
    settings = Settings.from_env()
    if timeout_s is None:
        timeout_secs = settings.compile_timeout_secs
    else:
        timeout_secs = timeout_s if timeout_s > 0 else None
    progress_log_enabled = settings.progress_log
    heartbeat_secs = settings.heartbeat_secs if progress_log_enabled else None
    poll_secs = 1.0

    def format_setting(value):
        if value is None:
            return "disabled"
        return f"{value:g}s"

    from tqdm import tqdm

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

    results: list[CandidateResult | None] = [None] * total_candidates
    candidate_iter = iter(enumerate(candidates, start=1))
    active_jobs = {}  # worker -> job
    runtime_root = get_cleave_runtime_root()
    os.makedirs(runtime_root, exist_ok=True)
    reference_root = tempfile.mkdtemp(prefix="ref_", dir=str(runtime_root))
    owns_pool = pool is None
    if owns_pool:
        pool = WorkerPool()
    # id(flat_graph) -> (reference_cache_dir, out_idx, io_bytes), or what building it raised
    graph_refs = {}

    def publish_references():
        """Build the reference of every flat graph, before the first worker starts.

        A reference is built on the GPU, under the lock the workers validate under
        (``_gpu_exec_guard``). Built at the first candidate of its graph instead, the
        parent would wait for that lock in the middle of the run -- and a worker whose
        kernel never returns holds it for good. The parent is the one that times a
        candidate out, so it would wait with it forever.

        A graph whose outputs the declared tolerance does not cover has no reference built:
        none of its candidates reaches a worker (each fails for the missing bound, in
        ``launch_until_full``). This is for a caller that hands candidates over itself.
        ``pipeline.build`` never gets here with such a graph: it refuses the declaration
        before it starts (``_check_tolerance_fits``), so no program compiled through the
        pipeline had a reference built for nothing.
        """
        for cand in candidates:
            key = id(cand.flat_graph)
            if key in graph_refs:
                continue
            try:
                declared_for(cand)
            except SpecError:
                graph_refs[key] = (None, None, 0)  # nothing of it is used: no worker gets it
                continue
            cache_dir = os.path.join(reference_root, f"g{len(graph_refs)}")
            os.makedirs(cache_dir, exist_ok=True)
            try:
                g_out_idx, g_io_bytes = reference_fn(cache_dir, cand.flat_graph, dtype)
            except Exception as exc:  # noqa: BLE001 - reported by each candidate of the graph
                graph_refs[key] = exc
            else:
                graph_refs[key] = (cache_dir, g_out_idx, g_io_bytes)

    def reference_for(cand):
        reference = graph_refs[id(cand.flat_graph)]
        if isinstance(reference, Exception):
            raise reference
        return reference

    def widest_reference():
        """The largest input + output footprint among the graphs that have a reference:
        what the run's parallelism is sized for.

        A worker loads the inputs and the reference of its candidate's graph onto the
        device, and the graphs of one run differ (a split partial returns ``splits`` times
        the program's output, and its combine reads that), so the cap has to hold for the
        largest. A graph without a reference sends nothing to a worker and has no say:
        one whose reference could not be built (its candidates report why), one no bound
        covers. The first candidate's graph used to speak for all of them: with no bound
        its footprint was 0 and the cap was off for the whole run, and with a reference
        that failed the run itself ended on that error, every candidate unreported.
        """
        return max((ref[2] for ref in graph_refs.values() if not isinstance(ref, Exception)),
                   default=0)

    def declared_for(cand):
        return declared_bounds(tolerance, cand.flat_graph)

    def finish(job, result, pbar):
        results[job["idx"] - 1] = result
        pbar.update(1)

    def fail(job, stage, error, pbar, code=None, measured=None):
        finish(
            job,
            CandidateResult(
                job["candidate"], "failed", stage, error, code=code, label=job["label"],
                measured=measured, stored_layouts=job.get("stored_layouts"),
            ),
            pbar,
        )

    def launch_until_full(pbar):
        while len(active_jobs) < max_parallel:
            try:
                candidate_idx, cand = next(candidate_iter)
            except StopIteration:
                return

            label = _candidate_progress_label(candidate_idx, total_candidates, cand)
            job = {"idx": candidate_idx, "candidate": cand, "label": label}
            try:
                cand_ref_dir, cand_out_idx, _ = reference_for(cand)
            except Exception as exc:
                fail(job, "codegen", _parent_error(CleaveError, label, "reference", exc), pbar)
                continue
            try:
                built = build_fn(cand)
            except Exception as exc:
                fail(job, "codegen", _parent_error(CodegenError, label, "codegen", exc), pbar)
                continue
            code_str, stored_layouts = built if isinstance(built, tuple) else (built, None)
            job["stored_layouts"] = stored_layouts
            try:
                declared, roles = declared_for(cand)
            except SpecError as exc:
                error = _parent_error(ValidationError, label, "tolerance", exc)
                fail(job, "validate", error, pbar, code_str)
                continue

            task = {
                "check_fn": check_fn, "code_str": code_str, "out_idx": cand_out_idx,
                "reference_cache_dir": cand_ref_dir, "tolerance": declared,
                "benchmark": benchmark, "stored_layouts": stored_layouts, "roles": roles,
            }
            worker = None
            try:
                worker = pool.acquire()
                worker.conn.send(task)
            except Exception as exc:
                if worker is not None:
                    pool.discard(worker)
                error = _parent_error(WorkerCrash, label, "worker start", exc)
                fail(job, "crash", error, pbar, code_str)
                continue

            job.update(code_str=code_str, started_at=time.monotonic())
            active_jobs[worker] = job

    def handle(job, result, pbar):
        if result["ok"]:
            finish(
                job,
                CandidateResult(
                    job["candidate"], "ok", None, None, job["code_str"],
                    result["cuda_src"], result.get("latency_ms"), job["label"],
                    result.get("measured"), job.get("stored_layouts"),
                ),
                pbar,
            )
        else:
            stage, error = _restore_compile_error(result, job["label"])
            fail(job, stage, error, pbar, job["code_str"], result.get("measured"))

    waits_charged_at = [time.monotonic()]

    def uncharge_waits(now):
        """Take the time a worker waited for the device, behind another worker of this run,
        off its candidate's clock.

        Workers validate and time one at a time, under the device lock. A kernel that never
        returns holds that lock until its worker is timed out, and every candidate launched
        next to it waits behind it for exactly as long: their time was up in the same pass,
        though none of them had had the device. One spinning kernel so failed every candidate
        in flight, and all of them in a run of no more than ``max_parallel`` (the fast preset
        of demo/swiglu.py: "8 kernels: 0 compiled, 8 failed [8x CandidateTimeout]", after
        its first kernel had validated). The timeout therefore counts what a candidate does,
        not what it waits for: the holder is always a candidate this run times out, so the
        wait ends. A holder that is no worker of this run is another process on the device;
        nothing here ends it, and waiting behind it is charged as before.
        """
        since, waits_charged_at[0] = waits_charged_at[0], now
        holder, waiters = device_lock_state()
        pids = {worker.process.pid: job for worker, job in active_jobs.items()}
        if holder not in pids:
            return
        for pid in waiters & set(pids):
            job = pids[pid]
            job["started_at"] = min(now, job["started_at"] + (now - since))

    def collect(pbar):
        # A worker's result is read before its death is noticed, so a worker that
        # reported and then exited (a retired one) is never counted as crashed.
        for worker, job in list(active_jobs.items()):
            result, dead = worker.poll()
            if result is None and not dead:
                continue
            del active_jobs[worker]
            if result is not None:
                if result.get("retire"):
                    pool.discard(worker)
                else:
                    pool.release(worker)
                handle(job, result, pbar)
                continue
            worker.process.join(timeout=1.0)
            error = WorkerCrash(
                f"{job['label']} exited with code {worker.process.exitcode} "
                "before returning a result."
            )
            pool.discard(worker)
            fail(job, "crash", error, pbar, job["code_str"])

    try:
        publish_references()
        max_parallel = _effective_compile_and_run_parallelism(max_parallel, widest_reference())

        with tqdm(total=total_candidates) as pbar:
            launch_until_full(pbar)
            last_heartbeat_at = waits_charged_at[0] = time.monotonic()

            while active_jobs:
                now = time.monotonic()
                wait_timeout = poll_secs
                if heartbeat_secs is not None:
                    wait_timeout = min(
                        wait_timeout, max(0.0, heartbeat_secs - (now - last_heartbeat_at))
                    )
                if timeout_secs is not None:
                    first_start = min(job["started_at"] for job in active_jobs.values())
                    wait_timeout = min(
                        wait_timeout, max(0.0, first_start + timeout_secs - now)
                    )
                _wait_ready(
                    [w.conn for w in active_jobs] + [w.process.sentinel for w in active_jobs],
                    timeout=wait_timeout,
                )
                collect(pbar)

                now = time.monotonic()
                if timeout_secs is not None:
                    uncharge_waits(now)
                    for worker, job in list(active_jobs.items()):
                        elapsed = now - job["started_at"]
                        if elapsed < timeout_secs:
                            continue

                        del active_jobs[worker]
                        pool.discard(worker)
                        error = CandidateTimeout(
                            f"{job['label']} exceeded {timeout_secs:g}s "
                            f"after {elapsed:.1f}s; terminated worker."
                        )
                        fail(job, "timeout", error, pbar, job["code_str"])
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
        for worker in active_jobs:
            pool.discard(worker)
        if owns_pool:
            pool.close()
        shutil.rmtree(reference_root, ignore_errors=True)

    return results


class _SerialBench:
    """Recompile and time kernels one at a time in this process (after the workers)."""

    def __init__(self, dtype, warmup, iters):
        self.dtype, self.warmup, self.iters = dtype, warmup, iters
        self.graph_inputs = {}  # id(flat_graph) -> (inputs_by_name, out_idx)

    def _inputs_for(self, graph):
        from .reference import placeholder_order

        key = id(graph)
        if key not in self.graph_inputs:
            placeholders = placeholder_order(graph)
            inputs_by_name = {
                ph.name: t
                for ph, t in zip(placeholders, make_inputs(graph, self.dtype))
            }
            self.graph_inputs[key] = (inputs_by_name, graph_out_indices(graph))
        return self.graph_inputs[key]

    def latency_ms(self, graph, code_str, stored_layouts=None):
        import torch

        inputs_by_name, out_idx = self._inputs_for(graph)
        kernel = _compile(_load_main_kernel_from_code(code_str), out_idx)
        logical_shapes = [node.get_shape(0) for node in output_order(graph)]
        try:
            return benchmark_compiled_kernel(
                with_hand_off(kernel, stored_layouts, logical_shapes),  # as the workers time it
                order_inputs_for_kernel(inputs_by_name, kernel),
                warmup_iters=self.warmup,
                profile_iters=self.iters,
            )
        finally:
            del kernel
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def close(self):
        import torch
        from tilelang.cache.kernel_cache import KernelCache

        self.graph_inputs.clear()
        KernelCache()._memory_cache.clear()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def benchmark_results(results, dtype, warmup=16, iters=1000) -> list[CandidateResult]:
    """Copies of ``results`` with ``latency_ms`` filled on the ok ones (serial re-time).

    Timing takes the GPU lock per kernel; a kernel that fails to time comes back failed
    at stage ``"bench"``.
    """
    bench = _SerialBench(dtype, warmup, iters)
    timed = []
    try:
        for result in results:
            if not result.ok:
                timed.append(result)
                continue
            try:
                with _gpu_exec_guard():
                    latency = bench.latency_ms(result.cand.flat_graph, result.code,
                                               result.stored_layouts)
            except Exception as exc:  # noqa: BLE001 - reported per candidate
                error = _parent_error(BenchmarkError, result.label, "benchmark", exc)
                timed.append(replace(result, status="failed", stage="bench", error=error))
            else:
                timed.append(replace(result, latency_ms=latency))
    finally:
        bench.close()
    return timed
