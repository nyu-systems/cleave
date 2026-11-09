from types import SimpleNamespace

import pytest

from cleave.core.errors import (
    BenchmarkError,
    CandidateTimeout,
    CleaveError,
    CodegenError,
    CompileError,
    SmemExceeded,
    ValidationError,
)
from cleave.schedule import search_tiling
from cleave.runtime import evaluate as runtime
from cleave.runtime.validate import BenchOptions, Tolerance
from cleave.runtime.evaluate import CandidateResult, benchmark_results, run_candidates

from tests._graphs import matmul_graph

DTYPE = "float16"
STAGES = {"codegen", "compile", "validate", "bench", "timeout", "crash"}


@pytest.fixture(scope="module")
def candidates():
    cands = search_tiling(matmul_graph(128, 128, 512), topk=4)
    assert len(cands) >= 2
    return cands


def _check_shape(results, cands):
    assert len(results) == len(cands)
    for result, cand in zip(results, cands):
        assert isinstance(result, CandidateResult)
        assert result.cand is cand
        assert result.label.startswith("candidate ")
        if result.ok:
            assert result.stage is None and result.error is None
            assert result.code and result.cuda_src
        else:
            assert result.status == "failed" and result.stage in STAGES
            assert isinstance(result.error, CleaveError)


def test_results_are_typed_and_in_input_order(candidates):
    # A candidate that cannot be tiled fails codegen in the parent, mid-list.
    broken = SimpleNamespace(
        flat_graph=candidates[0].flat_graph, rstep_map={}, output_tile=()
    )
    cands = [candidates[0], broken, *candidates[1:]]
    results = run_candidates(cands, DTYPE, max_parallel=4)
    _check_shape(results, cands)

    assert results[1].stage == "codegen"
    assert isinstance(results[1].error, CodegenError)
    assert results[0].ok and results[0].latency_ms is None


def test_tolerance_is_threaded_into_workers(candidates):
    # A negative atol rejects every element, so each kernel fails validation.
    results = run_candidates(
        candidates[:2], DTYPE, tolerance=Tolerance(rtol=0.0, atol=-1.0)
    )
    _check_shape(results, candidates[:2])
    for result in results:
        assert result.stage == "validate"
        assert isinstance(result.error, ValidationError)
        assert isinstance(result.error, AssertionError)


def test_timeout_is_typed(candidates):
    results = run_candidates(candidates[:2], DTYPE, timeout_s=0.01)
    _check_shape(results, candidates[:2])
    for result in results:
        assert result.stage == "timeout"
        assert isinstance(result.error, CandidateTimeout)
        assert isinstance(result.error, TimeoutError)


def test_benchmark_in_worker_and_serial_retime(candidates):
    results = run_candidates(
        candidates, DTYPE, max_parallel=4, benchmark=BenchOptions(warmup=4, iters=200)
    )
    _check_shape(results, candidates)
    ok = [r for r in results if r.ok]
    assert ok
    assert all(r.latency_ms is not None and r.latency_ms > 0 for r in ok)

    retimed = benchmark_results(results, DTYPE, warmup=4, iters=200)
    _check_shape(retimed, candidates)
    for before, after in zip(results, retimed):
        assert after.ok == before.ok
        if after.ok:
            assert after.latency_ms > 0 and after.code == before.code


def test_worker_failures_are_classified():
    def classify(**result):
        return runtime._restore_compile_error(result, "candidate 1/1")

    stage, error = classify(
        stage="compile",
        error_type="RuntimeError",
        error_message="Failed to set the allowed dynamic shared memory size to 1 with error",
        traceback="Traceback\n" + "frame\n" * 100,
    )
    assert stage == "compile" and isinstance(error, SmemExceeded)
    assert isinstance(error, CompileError)
    assert "candidate 1/1 failed with RuntimeError" in str(error)
    assert len(str(error).splitlines()) < 30  # traceback kept short

    assert isinstance(classify(stage="validate", error_type="AssertionError")[1], ValidationError)
    assert isinstance(classify(stage="bench", error_type="RuntimeError")[1], BenchmarkError)
    assert isinstance(classify(error_type="TVMError")[1], CompileError)
