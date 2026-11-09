import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_ROOT_IGNORED = {"__pycache__", ".pytest_cache"}
_ROOT_KEY = pytest.StashKey[set]()

# Node-id prefixes (repo-relative path, optionally "::test_name") that get each marker.
# "gpu" means a CUDA device must be visible (the test compiles or runs kernels). Graph
# building, tiling search and codegen run without one (cleave.core.target offline arch).
_MARKERS = {
    "gpu": (
        "tests/graph/handwritten/test_",
        "tests/schedule/end2end/",
        "tests/reference/test_",
        "tests/perf/",
        "tests/test_runtime_tolerance.py",
        "tests/frontend_dynamic/test_paged_pipeline.py::test_compile_end_to_end",
        "tests/pipeline_api/test_e2e.py",
        "tests/runtime/test_run_candidates.py",
        "tests/frontend/test_exact_softmax.py::test_gpu_",
    ),
    "expected": (
        "tests/graph/handwritten/",
        "tests/graph/apply_paging/test_expected_kernels.py",
        "tests/graph/apply_paging/test_split_decode.py",
        "tests/pipeline/",
    ),
    "search": (
        "tests/schedule/",
        "tests/pipeline/",
        "tests/superoptimizer/python/search/",
        "tests/graph/apply_paging/test_searched_tiling.py",
        "tests/frontend/test_exact_softmax.py::test_search_",
        "tests/frontend/test_exact_softmax.py::test_gpu_",
    ),
    "slow": (
        "tests/schedule/end2end/",
        "tests/perf/",
        "tests/superoptimizer/python/search/",
    ),
}
# CPU-only files matched by a gpu prefix above.
_CPU_ONLY = (
    "tests/graph/handwritten/test_expected_codegen.py",
    "tests/reference/test_graph_executor_refs.py",
)


def pytest_addoption(parser):
    parser.addoption("--update-snapshots", action="store_true",
                     help="rewrite tests/pipeline/snapshots from the current code")
    parser.addoption("--update-expected", action="store_true",
                     help="rewrite tests/graph/handwritten/expected_kernels from the current code")


def _gpu_pool():
    """GPUs the xdist workers round-robin over: none if CUDA_VISIBLE_DEVICES is empty,
    else CLEAVE_GPU_POOL (comma-separated), else every device nvidia-smi reports."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not visible.strip():
        return []
    explicit = os.environ.get("CLEAVE_GPU_POOL")
    if explicit:
        return [g.strip() for g in explicit.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _assign_gpu_for_xdist_worker():
    """Pin each xdist worker (and the candidate processes it spawns) to one GPU, round-robin."""
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker is None or os.environ.get("CLEAVE_XDIST_PINNED") == worker:
        return
    gpus = _gpu_pool()
    if not gpus:
        return
    index = int(worker[2:]) if worker.startswith("gw") and worker[2:].isdigit() else 0
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[index % len(gpus)]
    os.environ["CLEAVE_XDIST_PINNED"] = worker


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config):
    """``-n auto`` = one worker per GPU (CLEAVE_WORKERS_PER_GPU packs more); 0 without GPUs."""
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


def pytest_collection_modifyitems(config, items):
    root = config.rootpath
    for item in items:
        try:
            rel = item.path.relative_to(root).as_posix()
        except ValueError:
            continue
        _, sep, tail = item.nodeid.partition("::")
        key = f"{rel}{sep}{tail}"
        for name, prefixes in _MARKERS.items():
            if name == "gpu" and rel in _CPU_ONLY:
                continue
            if key.startswith(prefixes):
                item.add_marker(getattr(pytest.mark, name))


def _root_entries():
    return {p.name for p in ROOT.iterdir()} - _ROOT_IGNORED


def pytest_sessionstart(session):
    session.config.stash[_ROOT_KEY] = _root_entries()


def pytest_sessionfinish(session, exitstatus):
    """Tests and library defaults must not write into the repo root (the usual CWD)."""
    if hasattr(session.config, "workerinput"):  # the xdist controller checks once, after all workers
        return
    new = sorted(_root_entries() - session.config.stash.get(_ROOT_KEY, set()))
    if new:
        tw = session.config.get_terminal_writer()
        tw.line()
        tw.line(f"ERROR: the test session wrote into the repo root: {new}", red=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
