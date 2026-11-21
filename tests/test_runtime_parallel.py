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


def _check_under_the_lock(code_str, *, out_idx, reference_cache_dir, tolerance=None,
                          benchmark=None):
    """Worker-side fake of ``check_code``: compile, then validate and time under the device
    lock. ``hang`` is compiled at once and never gives the device back (a kernel that
    spins); the others take a moment to compile, so they find the device taken."""
    if code_str != "hang":
        time.sleep(0.4)
    with open(os.environ["FAKE_GPU_LOCK"], "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if code_str == "hang":
            time.sleep(600)
        time.sleep(0.05)
    return f"cuda_{code_str}", os.getpid()


def _device_lock_is(monkeypatch, path):
    """Point the parent's view of the device lock at the fake one the workers take."""
    from cleave.runtime import evaluate, gpu

    monkeypatch.setenv("FAKE_GPU_LOCK", str(path))
    monkeypatch.setattr(evaluate, "device_lock_state", lambda: gpu.lock_state(str(path)))


def test_a_kernel_that_never_returns_costs_only_its_own_candidate(tmp_path, monkeypatch):
    """Every worker validates under the device lock, so the candidates launched next to a
    kernel that never returns wait behind it until it is timed out. Their own time used to
    be up in that same pass, although none of them had had the device: one spinning kernel
    failed every candidate in flight. demo/swiglu.py at the fast preset lost all eight that
    way ("8 kernels: 0 compiled, 8 failed [8x CandidateTimeout]") after its first kernel
    had validated and then hung while it was timed.

    The time a candidate waits for the device behind another candidate of the run is not
    counted: the one that hangs is timed out, the others then get the device."""
    _device_lock_is(monkeypatch, tmp_path / "gpu.lock")
    cands = _candidates("hang", "a", "b", "c")
    started = time.monotonic()
    with WorkerPool(mp.get_context("fork")) as pool:
        results = run_candidates(
            cands, "float16", build_fn=_build, check_fn=_check_under_the_lock,
            reference_fn=_no_reference, pool=pool, max_parallel=4, timeout_s=2.0,
        )
    assert [r.stage for r in results] == ["timeout", None, None, None]
    assert isinstance(results[0].error, CandidateTimeout)
    assert "exceeded 2s" in str(results[0].error)
    assert [r.cuda_src for r in results[1:]] == ["cuda_a", "cuda_b", "cuda_c"]
    assert time.monotonic() - started < 20  # one timeout, not one per candidate


def test_a_candidate_that_holds_the_device_too_long_is_still_timed_out(tmp_path, monkeypatch):
    """Only waiting is taken off the clock. Two kernels that never return, one after the
    other: each is charged from the moment it has the device, and the third candidate, which
    waited behind both, still runs."""
    _device_lock_is(monkeypatch, tmp_path / "gpu.lock")
    cands = _candidates("hang", "hang", "a")
    with WorkerPool(mp.get_context("fork")) as pool:
        results = run_candidates(
            cands, "float16", build_fn=_build, check_fn=_check_under_the_lock,
            reference_fn=_no_reference, pool=pool, max_parallel=3, timeout_s=1.5,
        )
    assert [r.stage for r in results] == ["timeout", "timeout", None]
    assert results[2].cuda_src == "cuda_a"


def test_waiting_behind_another_process_on_the_device_is_charged(tmp_path, monkeypatch):
    """A lock held by a process that is no worker of this run (another session on the same
    GPU) is not ended by this run's timeouts, so waiting behind it counts as before: the
    candidates time out instead of waiting for good."""
    _device_lock_is(monkeypatch, tmp_path / "gpu.lock")
    cands = _candidates("a", "b")
    with open(tmp_path / "gpu.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # this process: not one of the pool's workers
        with WorkerPool(mp.get_context("fork")) as pool:
            results = run_candidates(
                cands, "float16", build_fn=_build, check_fn=_check_under_the_lock,
                reference_fn=_no_reference, pool=pool, max_parallel=2, timeout_s=1.5,
            )
    assert [r.stage for r in results] == ["timeout", "timeout"]


def _take_and_release(path):
    with open(path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)


def test_lock_state_names_the_holder_and_the_waiters(tmp_path):
    from cleave.runtime.gpu import lock_state

    path = tmp_path / "device.lock"
    assert lock_state(str(path)) == (None, frozenset())  # no such file
    assert lock_state(None) == (None, frozenset())
    path.touch()
    assert lock_state(str(path)) == (None, frozenset())  # nobody holds it
    with open(path, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert lock_state(str(path)) == (os.getpid(), frozenset())
        child = mp.get_context("fork").Process(target=_take_and_release, args=(str(path),))
        child.start()
        deadline = time.monotonic() + 10
        while lock_state(str(path))[1] != {child.pid} and time.monotonic() < deadline:
            time.sleep(0.02)
        assert lock_state(str(path)) == (os.getpid(), frozenset({child.pid}))
        # a lock on another file is another lock
        other = tmp_path / "other.lock"
        other.touch()
        assert lock_state(str(other)) == (None, frozenset())
        fcntl.flock(held, fcntl.LOCK_UN)
    child.join(timeout=10)
    assert child.exitcode == 0 and lock_state(str(path)) == (None, frozenset())


def test_the_processes_of_a_run_agree_on_the_device_lock(monkeypatch, tmp_path):
    """The device lock is named after the device's UUID, which ``nvidia-smi`` is asked for,
    and after ``CUDA_VISIBLE_DEVICES`` where that query fails or takes too long. A lock
    only serializes the processes that agree on its name, so the process that asks first
    hands its answer to the processes it starts. (Each worker used to ask again: one whose
    query timed out -- the driver answers it behind everything else that enters it -- named
    its lock differently and ran on the device next to its siblings.)"""
    import json
    import subprocess
    import sys

    from cleave.runtime import gpu

    asked = []

    def smi(cmd, **kw):
        asked.append(cmd[0])
        if len(asked) > 1:  # every later query: the driver is busy
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        return SimpleNamespace(stdout="0, GPU-aaaa\n3, GPU-dddd\n")

    monkeypatch.setattr(gpu.subprocess, "run", smi)
    monkeypatch.delenv(gpu._LOCK_KEY_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert gpu._gpu_lock_key() == "GPU-dddd" and asked == ["nvidia-smi"]
    assert json.loads(os.environ[gpu._LOCK_KEY_ENV]) == ["3", "GPU-dddd"]
    # a process started from here asks nobody: it has the answer (and so does this one)
    assert gpu._gpu_lock_key() == "GPU-dddd" and len(asked) == 1
    stub = tmp_path / "nvidia-smi"  # what the child would be told if it asked: nothing
    stub.write_text("#!/bin/sh\nexit 1\n")
    stub.chmod(0o755)
    code = "from cleave.runtime import gpu; print('key', gpu._gpu_lock_key())"
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                             env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"})
    assert child.communicate(timeout=300)[0].splitlines()[-1] == "key GPU-dddd"
    # the answer holds for the devices it was found for: a process that sees another one
    # asks for itself (here the query times out, and it falls back to what it sees)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert gpu._gpu_lock_key() == "0" and len(asked) == 2
    assert json.loads(os.environ[gpu._LOCK_KEY_ENV]) == ["0", "0"]
    # ... and something else in the variable is not an answer
    for junk in ("", "GPU-dddd", '["0"]', '["0", ""]', '["0", 7]', "[1, 2, 3]"):
        monkeypatch.setenv(gpu._LOCK_KEY_ENV, junk)
        assert gpu._inherited_lock_key("0") is None, junk
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv(gpu._LOCK_KEY_ENV, json.dumps([None, "GPU-aaaa"]))
    assert gpu._gpu_lock_key() == "GPU-aaaa" and len(asked) == 2


@pytest.mark.parametrize("failing", [1, 0, 2])
def test_a_reference_that_cannot_be_built_fails_the_candidates_of_its_graph(pool, failing):
    """... and only those, wherever the graph comes in the run. (As the first candidate's
    it used to end the run: the run was sized from that graph's reference, so the error
    was raised to the caller and no candidate was reported, of this graph or another.)"""
    good, bad = object(), object()

    def reference(cache_dir, graph, dtype):
        if graph is bad:
            raise ValueError("no reference for this graph")
        return [-1], 0

    cands = _candidates("a", "b", "c")
    for i, cand in enumerate(cands):
        cand.flat_graph = bad if i == failing else good
    results = run_candidates(cands, "float16", build_fn=_build, check_fn=_check,
                             reference_fn=reference, pool=pool, max_parallel=2)
    assert [r.ok for r in results] == [i != failing for i in range(3)]
    failed = results[failing]
    assert failed.stage == "codegen" and "no reference for this graph" in str(failed.error)


GIB = 2**30


@pytest.mark.parametrize("order, sized_for", [
    ("small large mid", 3 * GIB),
    ("large small mid", 3 * GIB),
    ("small mid large", 3 * GIB),
    ("broken small mid", 2 * GIB),  # a graph without a reference has no say ...
    ("small broken", 1 * GIB),
    ("broken", 0),                   # ... and a run without any is not capped
])
def test_a_run_is_sized_for_its_largest_reference(pool, monkeypatch, order, sized_for):
    """Workers load the inputs and the reference of their candidate's graph onto the
    device, so the number of workers is capped by what the largest of them takes -- not by
    the first candidate's, whose graph need not be the largest (a split partial returns
    ``splits`` times the program's output) and need not have a reference at all."""
    from cleave.runtime import evaluate

    footprint = {"small": 1 * GIB, "mid": 2 * GIB, "large": 3 * GIB}
    graphs = {name: SimpleNamespace(name=name) for name in (*footprint, "broken")}

    def reference(cache_dir, graph, dtype):
        if graph.name == "broken":
            raise ValueError("no reference for this graph")
        return [-1], footprint[graph.name]

    sized = []

    def cap(max_parallel, io_bytes):
        sized.append((max_parallel, io_bytes))
        return 2

    monkeypatch.setattr(evaluate, "_effective_compile_and_run_parallelism", cap)
    names = order.split()
    cands = _candidates(*names)
    for cand, name in zip(cands, names):
        cand.flat_graph = graphs[name]
    results = run_candidates(cands, "float16", build_fn=_build, check_fn=_check,
                             reference_fn=reference, pool=pool, max_parallel=5)
    assert sized == [(5, sized_for)]
    assert [r.ok for r in results] == [name != "broken" for name in names]



def _check_values(code_str, *, out_idx, reference_cache_dir, tolerance=None, benchmark=None):
    """Worker-side fake that runs the real comparator: the code string names what the
    candidate's kernel returns."""
    import torch

    from cleave.runtime.validate import compare_outputs

    if code_str.startswith("uncompilable"):
        raise RuntimeError("TileLang could not lower it")
    reference = torch.arange(1.0, 33.0).reshape(4, 8)
    out = reference.clone()
    if code_str.startswith("wrong"):
        out *= 1.5
    elif code_str.startswith("nan"):
        out[0, 0] = float("nan")
    compare_outputs([out], [reference], tolerance)
    return f"cuda_{code_str}", 1.0


def test_end2end_helper_fails_on_a_kernel_that_validates_wrong(monkeypatch, pool):
    """``tests/schedule/end2end`` asserted a minimum number of validated candidates and
    nothing else: a tiling that built, ran and returned wrong or non-finite values was
    written to ``errors.err`` next to the refusals and the compile errors, and the case
    passed while enough others validated."""
    import functools

    from cleave.core.errors import NonFiniteOutput, OutputMismatch
    from cleave.runtime.validate import Tolerance
    from tests.schedule.end2end import _test_utils

    monkeypatch.setattr(_test_utils, "run_candidates", functools.partial(
        run_candidates, build_fn=_build, check_fn=_check_values, reference_fn=_no_reference,
        pool=pool))
    bound = Tolerance(rtol=5e-3)

    def run(*codes, min_success):
        return _test_utils.run_end2end_candidates("fake", _candidates(*codes), "float16",
                                                  min_success=min_success, tolerance=bound,
                                                  log=False)

    # candidates without a kernel leave the count to decide, as before
    results = run("ok_a", "uncompilable_a", "ok_b", "uncompilable_b", min_success=2)
    assert [r.ok for r in results] == [True, False, True, False]
    assert {r.stage for r in results if not r.ok} == {"compile"}
    with pytest.raises(AssertionError, match="2 scheduled graphs compiled and ran successfully, "
                                             "expected >= 3"):
        run("ok_a", "uncompilable_a", "ok_b", min_success=3)
    # one wrong kernel fails the case although the count is met
    with pytest.raises(AssertionError, match="1 scheduled graphs built and ran but validated "
                                             "wrong: .*OutputMismatch"):
        run("ok_a", "ok_b", "wrong", "uncompilable_a", min_success=2)
    with pytest.raises(AssertionError, match="2 scheduled graphs built and ran but validated "
                                             "wrong: .*(OutputMismatch|NonFiniteOutput)"):
        run("ok_a", "nan", "ok_b", "wrong", min_success=2)
    # ... and they are what the comparator calls them
    results = run_candidates(_candidates("wrong", "nan"), "float16", build_fn=_build,
                             check_fn=_check_values, reference_fn=_no_reference, pool=pool,
                             tolerance=bound)
    assert [type(r.error) for r in results] == [OutputMismatch, NonFiniteOutput]
    assert [r.stage for r in results] == ["validate", "validate"]
