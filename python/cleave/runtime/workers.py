"""Persistent spawned worker processes that compile and check candidates."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import signal
import sys
import traceback
import weakref


def _compile_and_run_mp_context():
    return mp.get_context("spawn")


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


def _run_task(task):
    """One candidate in a worker: ``check_fn``'s result, or the failure, as a plain dict."""
    try:
        cuda_src, latency_ms = task["check_fn"](
            task["code_str"], out_idx=task["out_idx"],
            reference_cache_dir=task["reference_cache_dir"],
            tolerance=task["tolerance"], benchmark=task["benchmark"],
        )
        return {"ok": True, "cuda_src": cuda_src, "latency_ms": latency_ms}
    except BaseException as exc:
        stage = getattr(exc, "_cleave_stage", None)
        if stage is None:
            stage = "validate" if isinstance(exc, AssertionError) else "compile"
        return {
            "ok": False,
            "stage": stage,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }


def _cuda_context_healthy():
    """False once a kernel fault has left this process's CUDA context unusable."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return True
    try:
        torch.cuda.synchronize()
        torch.zeros(1, device="cuda").item()
    except Exception:  # noqa: BLE001 - any failure means the context is gone
        return False
    return True


def _release_task_memory():
    cache = sys.modules.get("tilelang.cache.kernel_cache")
    if cache is not None:
        with contextlib.suppress(Exception):
            cache.KernelCache()._memory_cache.clear()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _pool_worker_main(conn):
    """Run tasks from ``conn`` until it closes; exit after a failure that broke CUDA."""
    if hasattr(os, "setsid"):
        with contextlib.suppress(OSError):
            os.setsid()
    while True:
        try:
            task = conn.recv()
        except (EOFError, OSError):
            return
        if task is None:
            return
        result = _run_task(task)
        result["retire"] = not (result["ok"] or _cuda_context_healthy())
        if not result["retire"]:
            _release_task_memory()
        conn.send(result)
        if result["retire"]:
            return


class _Worker:
    def __init__(self, mp_context):
        self.conn, child_conn = mp_context.Pipe()
        self.process = mp_context.Process(target=_pool_worker_main, args=(child_conn,))
        try:
            self.process.start()
        finally:
            child_conn.close()

    def poll(self):
        """``(result | None, dead)`` without blocking."""
        try:
            if self.conn.poll():
                return self.conn.recv(), False
        except (EOFError, OSError):
            return None, True
        return None, not self.process.is_alive()

    def kill(self):
        _terminate_process_tree(self.process)
        self.conn.close()
        _close_process(self.process)


class WorkerPool:
    """Persistent worker processes for :func:`run_candidates`, spawned on demand.

    Pass one as ``pool=`` to reuse its workers across calls (close it, or use it as a
    context manager); without it each call uses a private pool. A worker that crashes,
    times out or breaks its CUDA context is dropped and a fresh one replaces it.
    """

    def __init__(self, mp_context=None):
        self.mp_context = mp_context or _compile_and_run_mp_context()
        self._idle: list[_Worker] = []
        self._busy: set[_Worker] = set()
        self._finalizer = weakref.finalize(self, _shutdown_workers, self._idle, self._busy)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def pids(self):
        return [w.process.pid for w in [*self._idle, *self._busy]]

    def acquire(self):
        """An idle worker, or a newly spawned one."""
        if not self._finalizer.alive:
            raise RuntimeError("WorkerPool is closed")
        while self._idle:
            worker = self._idle.pop()
            if worker.process.is_alive():
                self._busy.add(worker)
                return worker
            worker.kill()
        worker = _Worker(self.mp_context)
        self._busy.add(worker)
        return worker

    def release(self, worker):
        self._busy.discard(worker)
        self._idle.append(worker)

    def discard(self, worker):
        self._busy.discard(worker)
        worker.kill()

    def close(self):
        self._finalizer()


def _shutdown_workers(idle, busy):
    for worker in idle:
        with contextlib.suppress(OSError):
            worker.conn.send(None)
    for worker in idle:
        worker.process.join(timeout=5.0)
    for worker in [*idle, *busy]:
        worker.kill()
    idle.clear()
    busy.clear()
