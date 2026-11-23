from types import SimpleNamespace

import pytest

from cleave.core.errors import (
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
)
from cleave.schedule import search_tiling
from cleave.runtime import evaluate as runtime
from cleave.runtime.validate import BenchOptions, Tolerance
from cleave.runtime.evaluate import CandidateResult, benchmark_results, run_candidates

from tests._graphs import matmul_graph
from tests._tolerances import tolerance_for

DTYPE = "float16"
TOLERANCE = tolerance_for(matmul_graph)
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
            # what the comparison measured travels back with every validated kernel
            (measured,) = result.measured
            assert 0 <= measured.rel <= TOLERANCE.rtol and measured.abs >= 0
        else:
            assert result.status == "failed" and result.stage in STAGES
            assert isinstance(result.error, CleaveError)


def test_results_are_typed_and_in_input_order(candidates):
    # A candidate that cannot be tiled fails codegen in the parent, mid-list.
    broken = SimpleNamespace(
        flat_graph=candidates[0].flat_graph, rstep_map={}, output_tile=()
    )
    cands = [candidates[0], broken, *candidates[1:]]
    results = run_candidates(cands, DTYPE, max_parallel=4, tolerance=TOLERANCE)
    _check_shape(results, cands)

    assert results[1].stage == "codegen"
    assert isinstance(results[1].error, CodegenError)
    assert results[0].ok and results[0].latency_ms is None


def test_a_negative_tolerance_is_rejected_where_it_is_declared():
    # This used to reach the workers and fail every kernel by accident: torch raised
    # "atol must be greater than or equal to zero" inside the validate stage. A bound that
    # can never hold is now refused on purpose, before any candidate is built.
    with pytest.raises(SpecError, match="Tolerance.atol must be a finite number >= 0"):
        Tolerance(rtol=0.0, atol=-1.0)


def test_tolerance_is_threaded_into_workers(candidates):
    # The exact bound: an fp16 tensor-core matmul over K=512 is not bit-identical to torch's,
    # so each kernel fails validation as a mismatch -- by the comparison, at the tolerance
    # the worker was handed -- and reports an error the declared tolerance would accept.
    results = run_candidates(candidates[:2], DTYPE, tolerance=Tolerance(rtol=0.0, atol=0.0))
    _check_shape(results, candidates[:2])
    for result in results:
        assert result.stage == "validate"
        assert isinstance(result.error, OutputMismatch)
        assert isinstance(result.error, ValidationError)
        assert isinstance(result.error, AssertionError)
        assert "bound rtol=0 * rowmax|ref| + atol=0" in str(result.error)
        (measured,) = result.measured
        assert 0 < measured.rel <= TOLERANCE.rtol


def test_a_missing_tolerance_fails_the_candidate_at_validate(candidates):
    # run_candidates itself does not require one (its stages are injectable); the default
    # check does, and says so instead of falling back to a loose bound.
    (result,) = run_candidates(candidates[:1], DTYPE)
    assert result.stage == "validate" and type(result.error) is ValidationError
    assert "no validation tolerance declared" in str(result.error)
    assert result.measured is None


def test_timeout_is_typed(candidates):
    results = run_candidates(candidates[:2], DTYPE, tolerance=TOLERANCE, timeout_s=0.01)
    _check_shape(results, candidates[:2])
    for result in results:
        assert result.stage == "timeout"
        assert isinstance(result.error, CandidateTimeout)
        assert isinstance(result.error, TimeoutError)


def test_benchmark_in_worker_and_serial_retime(candidates):
    results = run_candidates(
        candidates, DTYPE, max_parallel=4, tolerance=TOLERANCE,
        benchmark=BenchOptions(warmup=4, iters=200),
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

    assert type(classify(stage="validate", error_type="AssertionError")[1]) is ValidationError
    # A validation failure keeps the kind the worker raised: wrong values, a NaN/inf, or a
    # kernel that could not be launched. Anything else at that stage is the plain base class.
    for kind in (OutputMismatch, NonFiniteOutput, LaunchCrash):
        stage, error = classify(stage="validate", error_type=kind.__name__)
        assert stage == "validate" and type(error) is kind
        assert isinstance(error, ValidationError) and isinstance(error, AssertionError)
    assert type(classify(stage="validate", error_type="ValueError")[1]) is ValidationError
    assert type(classify(stage="compile", error_type="OutputMismatch")[1]) is CompileError
    assert isinstance(classify(stage="bench", error_type="RuntimeError")[1], BenchmarkError)
    assert isinstance(classify(error_type="TVMError")[1], CompileError)
