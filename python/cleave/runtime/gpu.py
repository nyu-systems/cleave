from __future__ import annotations

import contextlib
import ctypes
import fcntl
import os
import subprocess
from typing import Optional

from .._cache_env import configure_cache_env, get_cleave_runtime_root

_PARALLEL_ENV_CONFIGURED = False
_GPU_LOCK_PATH: Optional[str] = None


def _gpu_lock_key() -> str:
    """Physical identity (UUID) of the device this process runs on (visible device 0).

    Falls back to the raw ``CUDA_VISIBLE_DEVICES`` string when the UUID is unavailable.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    fallback = (visible or "all").replace(",", "_")
    if visible == "":
        return fallback
    first = visible.split(",")[0].strip() if visible else "0"
    if first.startswith(("GPU-", "MIG-")):
        return first
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout
    except Exception:
        return fallback
    for line in out.splitlines():
        index, _, uuid = (s.strip() for s in line.partition(","))
        if index == first and uuid:
            return uuid
    return fallback


def _create_parallel_env():
    global _PARALLEL_ENV_CONFIGURED, _GPU_LOCK_PATH
    if _PARALLEL_ENV_CONFIGURED:
        return

    configure_cache_env()
    _GPU_LOCK_PATH = str(get_cleave_runtime_root() / f"gpu_{_gpu_lock_key()}.lock")
    _PARALLEL_ENV_CONFIGURED = True


_CUDA_CONTEXT_READY = False

# TODO(david): this is hacky, but enables, e.g., usage of `cudaFuncSetAttribute`
# since it needs a live context to act on. Otherwise this usage of
# `cudaFuncSetAttribute` will not be effective, and kernels needing more SMEM
# may crash. Find systematic of solving this in the future.
def _ensure_cuda_context():
    global _CUDA_CONTEXT_READY
    if _CUDA_CONTEXT_READY:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.init()
            torch.zeros(1, device="cuda")
    except Exception:
        return
    _CUDA_CONTEXT_READY = True


_CU_LIMIT_STACK_SIZE = 0  # CUlimit: bytes of stack per device thread
_DRIVER = None  # the CUDA driver library, loaded on first use; False when there is none


def _driver():
    global _DRIVER
    if _DRIVER is None:
        try:
            _DRIVER = ctypes.CDLL("libcuda.so.1")
        except OSError:
            _DRIVER = False
    return _DRIVER or None


def _stack_limit():
    """Bytes of stack this process's CUDA context keeps for every device thread, or None
    where that cannot be asked (no context yet, no driver library)."""
    driver = _driver() if _CUDA_CONTEXT_READY else None
    if driver is None:
        return None
    value = ctypes.c_size_t()
    if driver.cuCtxGetLimit(ctypes.byref(value), _CU_LIMIT_STACK_SIZE) != 0:
        return None
    return value.value


def _release_thread_stacks(limit):
    """Set the context's stack limit back to ``limit`` if a launch has raised it.

    A kernel whose threads keep large local arrays needs more stack than the context
    has (1 KiB per thread). The driver then allocates the larger stack for every thread
    the device can run, about 270,000 on an H100, and keeps it after the kernel is done:
    4.3 GiB for a frame of 15 KiB. Nothing the process frees gives that back, so every
    worker that had launched such a kernel went on holding it, and the workers that came
    after it failed with out-of-memory errors: 7 of the 28 kernels of the row-sum quotient,
    each of them valid, while they kept a row per thread. Lowering the limit frees the
    stacks at once.
    """
    if limit is None or _stack_limit() in (None, limit):
        return
    driver = _driver()
    driver.cuCtxSynchronize()  # no kernel is still running on the stacks to be freed
    driver.cuCtxSetLimit(_CU_LIMIT_STACK_SIZE, ctypes.c_size_t(limit))


@contextlib.contextmanager
def _gpu_exec_guard():
    """One process at a time on the device. Whoever holds the lock leaves its CUDA context
    with the thread stacks it came with (``_release_thread_stacks``): the stacks of the
    kernels it launched are device memory the next holder needs."""
    _create_parallel_env()
    _ensure_cuda_context()
    os.makedirs(os.path.dirname(_GPU_LOCK_PATH), exist_ok=True)
    with open(_GPU_LOCK_PATH, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        stack_limit = _stack_limit()
        try:
            yield
        finally:
            try:
                _release_thread_stacks(stack_limit)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
