import fcntl
import multiprocessing as mp
import os
import threading
import time
from types import SimpleNamespace

import pytest

from cleave.core.errors import CandidateTimeout, ValidationError, WorkerCrash
from cleave.runtime.workers import WorkerPool
from cleave.runtime.evaluate import run_candidates

pytestmark = pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="requires fork multiprocessing support",
)


def _build(cand):
    return cand.code


def _no_reference(cache_dir, graph, dtype):
    return [-1], 0


def _check(code_str, *, out_idx, reference_cache_dir, tolerance=None, benchmark=None):
    """Worker-side fake: the code string names what the candidate does."""
    if code_str == "slow":
        time.sleep(1.0)
    elif code_str == "crash":
        os._exit(3)
    elif code_str == "wrong":
        raise AssertionError("mismatch")
    else:
        time.sleep(0.15 if code_str.startswith("fast") else 0.05)
    return f"cuda_{code_str}", os.getpid()


def _candidates(*codes):
    return [
        SimpleNamespace(code=c, rstep_map={"id": i}, output_tile=(i,), flat_graph=None)
        for i, c in enumerate(codes, start=1)
    ]


@pytest.fixture
def pool():
    with WorkerPool(mp.get_context("fork")) as p:
        yield p


def _run(cands, pool, **kw):
    return run_candidates(
        cands, "float16", build_fn=_build, check_fn=_check, reference_fn=_no_reference,
        pool=pool, **kw,
    )


def test_run_candidates_times_out_and_reports_progress(monkeypatch, capsys, pool):
    monkeypatch.setenv("COMPILATION_PROGRESS_LOG", "true")
    monkeypatch.setenv("CLEAVE_COMPILE_AND_RUN_TIMEOUT_SECS", "0.3")
    monkeypatch.setenv("CLEAVE_COMPILE_PROGRESS_HEARTBEAT_SECS", "0.05")

    results = _run(_candidates("fast_a", "slow", "fast_b"), pool, max_parallel=2)

    assert [r.code for r in results if r.ok] == ["fast_a", "fast_b"]
    failed = [r for r in results if not r.ok]
    assert [r.cand.code for r in failed] == ["slow"]
    assert isinstance(failed[0].error, TimeoutError)
    assert "candidate 2/3" in str(failed[0].error)

    output = capsys.readouterr().out
    assert "[cleave]" in output
    assert "2 running. Slowest: candidate 2/3" in output
    assert "[cleave] timed out candidate 2/3" in output


def test_run_candidates_progress_log_disabled_by_default(monkeypatch, capsys, pool):
    monkeypatch.delenv("COMPILATION_PROGRESS_LOG", raising=False)
    monkeypatch.setenv("CLEAVE_COMPILE_PROGRESS_HEARTBEAT_SECS", "0.01")

    results = _run(_candidates("fast_a", "fast_b"), pool, max_parallel=2)

    assert [r.code for r in results if r.ok] == ["fast_a", "fast_b"]
    assert all(r.ok for r in results)

    output = capsys.readouterr().out
    assert output == "Compiling and running 2 scheduled graphs:\n"


def test_workers_are_reused_within_and_across_calls(pool):
    cands = _candidates(*(f"c{i}" for i in range(6)))
    first = _run(cands, pool, max_parallel=2)
    assert [r.cuda_src for r in first] == [f"cuda_c{i}" for i in range(6)]
    pids = {r.latency_ms for r in first}
    assert len(pids) <= 2 and os.getpid() not in pids

    second = _run(cands[:2], pool, max_parallel=2)
    assert {r.latency_ms for r in second} <= pids


def test_crash_and_timeout_replace_only_their_worker(pool):
    cands = _candidates("a", "crash", "b", "slow", "wrong", "c")
    results = _run(cands, pool, max_parallel=1, timeout_s=0.5)

    assert [r.cand for r in results] == cands
    assert [r.stage for r in results] == [None, "crash", None, "timeout", "validate", None]
    assert isinstance(results[1].error, WorkerCrash)
    assert "exited with code 3" in str(results[1].error)
    assert isinstance(results[3].error, CandidateTimeout)
    assert isinstance(results[4].error, ValidationError)
    a, b, c = results[0].latency_ms, results[2].latency_ms, results[5].latency_ms
    assert a != b != c  # a fresh worker after the crash and after the timeout
    assert results[4].ok is False and len(pool.pids) == 1


def _reference_under_lock(cache_dir, graph, dtype):
    """Parent-side fake of ``publish_reference``: built under the lock the workers check under."""
    with open(os.environ["FAKE_GPU_LOCK"], "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
    return [-1], 0


def _check_hang_holds_the_lock(code_str, *, out_idx, reference_cache_dir, tolerance=None,
                               benchmark=None):
    """Worker-side fake: ``hang`` takes the lock and never returns (a kernel that spins)."""
    if code_str == "hang":
        with open(os.environ["FAKE_GPU_LOCK"], "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            time.sleep(600)
    time.sleep(0.3)
    return f"cuda_{code_str}", os.getpid()


def test_a_kernel_that_never_returns_cannot_stall_the_parent(tmp_path, monkeypatch):
    """A worker whose kernel never returns holds the GPU lock until it is timed out, and
    only the parent times it out. The parent must therefore never wait for that lock while
    workers run: every graph's reference is built before the first worker starts.

    (demo/gqa_decode.py stalled for good this way: at candidate 756 of 1723 a split kernel
    spun on the GPU, the next candidate was the first of another flat graph, and the parent
    sat in flock() for its reference instead of timing the worker out.)
    """
    monkeypatch.setenv("FAKE_GPU_LOCK", str(tmp_path / "gpu.lock"))
    first, second = object(), object()  # two flat graphs: each needs its own reference
    cands = _candidates("hang", "a", "b")
    cands[0].flat_graph = cands[1].flat_graph = first
    cands[2].flat_graph = second
    done = {}

    def run(pool):
        try:
            done["results"] = run_candidates(
                cands, "float16", build_fn=_build, check_fn=_check_hang_holds_the_lock,
                reference_fn=_reference_under_lock, pool=pool, max_parallel=2, timeout_s=1.5,
            )
        except BaseException as exc:  # noqa: BLE001 - only reached once the pool is torn down
            done["error"] = exc

    with WorkerPool(mp.get_context("fork")) as pool:
        thread = threading.Thread(target=run, args=(pool,), daemon=True)
        thread.start()
        thread.join(timeout=30)
        stalled = thread.is_alive()
    # leaving the pool kills its workers, which frees the lock: a stalled parent can end
    thread.join(timeout=30)
    assert not stalled, "the parent waited for a lock that a hung worker holds"
    results = done["results"]
    assert [r.stage for r in results] == ["timeout", None, None]
    assert isinstance(results[0].error, CandidateTimeout)
    assert [r.cuda_src for r in results[1:]] == ["cuda_a", "cuda_b"]


def test_a_reference_that_cannot_be_built_fails_the_candidates_of_its_graph(pool):
    good, bad = object(), object()

    def reference(cache_dir, graph, dtype):
        if graph is bad:
            raise ValueError("no reference for this graph")
        return [-1], 0

    cands = _candidates("a", "b", "c")
    cands[0].flat_graph = cands[2].flat_graph = good
    cands[1].flat_graph = bad
    results = run_candidates(cands, "float16", build_fn=_build, check_fn=_check,
                             reference_fn=reference, pool=pool, max_parallel=2)
    assert [r.ok for r in results] == [True, False, True]
    assert results[1].stage == "codegen" and "no reference for this graph" in str(results[1].error)

