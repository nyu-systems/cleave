from __future__ import annotations

import contextlib
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


@contextlib.contextmanager
def _gpu_exec_guard():
    _create_parallel_env()
    _ensure_cuda_context()
    os.makedirs(os.path.dirname(_GPU_LOCK_PATH), exist_ok=True)
    with open(_GPU_LOCK_PATH, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
