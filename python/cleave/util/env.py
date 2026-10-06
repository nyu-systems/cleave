from __future__ import annotations

import contextlib
import fcntl
import os
from typing import Optional

from .._cache_env import configure_cache_env, get_cleave_runtime_root

_PARALLEL_ENV_CONFIGURED = False
_GPU_LOCK_PATH: Optional[str] = None


def _create_parallel_env():
    global _PARALLEL_ENV_CONFIGURED, _GPU_LOCK_PATH
    if _PARALLEL_ENV_CONFIGURED:
        return

    configure_cache_env()

    base_dir = get_cleave_runtime_root()
    pid_dir = base_dir / f"pid_{os.getpid()}"
    os.makedirs(pid_dir, exist_ok=True)

    for key in [
        "CUDA_CACHE_PATH",
        "TORCH_EXTENSIONS_DIR",
        "XDG_CACHE_HOME",
        "TMPDIR",
    ]:
        path = os.environ.get(key)
        if path:
            os.makedirs(path, exist_ok=True)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all") or "all"
    lock_key = visible.replace(",", "_")
    _GPU_LOCK_PATH = str(base_dir / f"gpu_{lock_key}.lock")
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
