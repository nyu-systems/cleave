import os

import pytest


def _assign_gpu_for_xdist_worker():
    """Pin each pytest-xdist worker to one physical GPU (round-robin).

    Runs before any test imports torch, so setting ``CUDA_VISIBLE_DEVICES`` here
    (and inheriting it into the spawned candidate workers) pins all of a worker's
    GPU work to a single device. Tests on distinct xdist workers then run on
    distinct GPUs concurrently; within a device, Cleave's per-GPU lock and
    memory-aware candidate throttle keep concurrent work inside the memory budget.

    No-op when running single-process (no xdist) or when the caller already pinned
    ``CUDA_VISIBLE_DEVICES`` explicitly.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker is None:
        return  # single-process run; respect whatever CUDA_VISIBLE_DEVICES is set
    if os.environ.get("CLEAVE_XDIST_PINNED") == worker:
        return  # already pinned this worker

    gpus = _gpu_pool()
    if not gpus:
        return

    index = int(worker[2:]) if worker.startswith("gw") and worker[2:].isdigit() else 0
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[index % len(gpus)]
    os.environ["CLEAVE_XDIST_PINNED"] = worker


def _gpu_pool():
    """GPU device ids the xdist workers round-robin over.

    ``CLEAVE_GPU_POOL`` (comma-separated) pins the pool explicitly; otherwise all
    devices reported by ``nvidia-smi`` are used.
    """
    explicit = os.environ.get("CLEAVE_GPU_POOL")
    if explicit:
        return [g.strip() for g in explicit.split(",") if g.strip()]

    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def pytest_xdist_auto_num_workers(config):
    """Resolve ``-n auto`` to the number of GPUs (one worker per GPU).

    Cleave tests are GPU-bound, so the useful parallelism is the number of GPUs,
    not CPU cores: each worker is pinned to a distinct device. ``CLEAVE_WORKERS_PER_GPU``
    can pack more than one worker per GPU (the per-GPU exec lock still keeps
    concurrent kernels within the memory budget). Returns 0 (xdist falls back to its
    CPU-based default) when no GPU is detected.
    """
    gpus = _gpu_pool()
    if not gpus:
        return 0
    try:
        per_gpu = max(1, int(os.environ.get("CLEAVE_WORKERS_PER_GPU", "1")))
    except ValueError:
        per_gpu = 1
    return len(gpus) * per_gpu


def pytest_configure(config):
    _assign_gpu_for_xdist_worker()


@pytest.fixture(autouse=True)
def clear_cleave_lru_caches():
    # prevent cross-test leakage from cached shape/tensorcore analysis.
    from cleave.graph import PrimFuncNode

    PrimFuncNode.get_space_dim.cache_clear()
    PrimFuncNode.infer_tensorcore_axis.cache_clear()
    yield
    PrimFuncNode.get_space_dim.cache_clear()
    PrimFuncNode.infer_tensorcore_axis.cache_clear()
