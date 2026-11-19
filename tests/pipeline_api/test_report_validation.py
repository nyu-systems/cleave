"""CPU-only: reports carry what validation measured and how it failed.

A validate-stage failure is a mismatch, a non-finite output or a launch crash (anything else
there is a reference/layout problem); every candidate whose outputs were compared keeps the
error the comparator measured, and the report shows the worst of the passing ones.
"""

import json
from types import SimpleNamespace

import pytest

from cleave.core.errors import (
    CompileError,
    LaunchCrash,
    NonFiniteOutput,
    OutputMismatch,
    ValidationError,
)
from cleave.pipeline.dynamic import DynamicReport
from cleave.pipeline.report import Report, Result, format_measured, validation_kind
from cleave.runtime.autotune import WorkloadCandidate
from cleave.runtime.evaluate import CandidateResult
from cleave.runtime.validate import MeasuredError, Tolerance


def _result(i, *, error=None, stage=None, measured=None, latency_ms=None):
    plan = SimpleNamespace(label=f"p{i}", variant="solo", role="main")
    run = CandidateResult(None, "failed" if error else "ok", stage, error, latency_ms=latency_ms,
                          measured=measured)
    return Result(plan, run)


def _m(rel, abs_=0.0):
    return (MeasuredError(rel, abs_),)


def test_validation_failures_are_counted_by_kind():
    results = [
        _result(0, measured=_m(4e-4, 0.01), latency_ms=1.0),
        _result(1, measured=_m(9e-4, 0.03), latency_ms=2.0),
        _result(2, error=OutputMismatch("x"), stage="validate", measured=_m(1.2, 0.3)),
        _result(3, error=OutputMismatch("x"), stage="validate", measured=_m(0.4, 0.1)),
        _result(4, error=NonFiniteOutput("x"), stage="validate"),
        _result(5, error=LaunchCrash("x"), stage="validate"),
        _result(6, error=ValidationError("ambiguous layout"), stage="validate"),
        _result(7, error=CompileError("x"), stage="compile"),
    ]
    tolerance = Tolerance(rtol=5e-3)
    report = Report(results, tolerance=tolerance)
    assert report.failures.by_stage() == {"validate": 5, "compile": 1}
    assert report.failures.by_validation_kind() == {
        "mismatch": 2, "non_finite": 1, "launch_crash": 1, "other": 1}
    # the histogram splits them too: the kinds are distinct error types
    assert report.histogram() == {"OutputMismatch": 2, "NonFiniteOutput": 1, "LaunchCrash": 1,
                                  "ValidationError": 1, "CompileError": 1}
    assert all(isinstance(r.error, AssertionError) for r in report.failures if r.stage == "validate")

    # the worst error among the candidates that PASSED, never the failing ones
    assert report.measured_error() == [{"rel": 9e-4, "abs": 0.03}]
    validation = json.loads(json.dumps(report.to_dict()))["validation"]
    assert validation["tolerance"] == [{"rtol": 5e-3, "atol": 0.0}]
    assert validation["failures"] == report.failures.by_validation_kind()
    assert validation["max_error"] == [{"rel": 9e-4, "abs": 0.03}]
    # ... while every compared candidate, passing or not, lists what it measured
    assert [(m["plan"], m["ok"], m["error"], m["rel"]) for m in validation["measured"]] == [
        ("p0", True, None, [4e-4]), ("p1", True, None, [9e-4]),
        ("p2", False, "OutputMismatch", [1.2]), ("p3", False, "OutputMismatch", [0.4])]

    validate = dict((stage, text) for stage, text, _ in report.stage_lines())["validate"]
    assert validate.startswith("2 ok / 5 failed")
    assert "2x OutputMismatch" in validate and "1x LaunchCrash" in validate
    assert validate.endswith("max error rel 9.0e-04 of rtol 0.005")


def test_validation_kind_of_other_errors():
    assert validation_kind(ValidationError("x")) == "other"
    assert validation_kind(CompileError("x")) == "other"
    assert validation_kind(OutputMismatch("x")) == "mismatch"


def test_format_measured_names_the_governing_bound_per_output():
    worst = [{"rel": 7.8e-3, "abs": 1.6e-2}, {"rel": 2.7e-7, "abs": 1.9e-6}]
    tolerance = (Tolerance(rtol=5e-2), Tolerance(rtol=0.0, atol=1e-5))
    assert format_measured(worst, tolerance) == (
        "out0 rel 7.8e-03 of rtol 0.05; out1 abs 1.9e-06 of atol 1e-05")
    assert format_measured(worst[:1], Tolerance(rtol=1e-3, atol=1e-6)) == (
        "rel 7.8e-03 of rtol 0.001, abs 1.6e-02 of atol 1e-06")
    assert format_measured(worst[:1]) == "rel 7.8e-03 abs 1.6e-02"
    assert format_measured(None) == "" and format_measured([]) == ""


def test_report_without_measurements_stays_quiet():
    report = Report([_result(0, error=CompileError("x"), stage="compile")])
    assert report.measured_error() is None
    assert report.to_dict()["validation"] == {
        "tolerance": None, "failures": {}, "max_error": None, "measured": []}
    validate = dict((stage, text) for stage, text, _ in report.stage_lines())["validate"]
    assert "max error" not in validate


def test_dynamic_report_carries_measured_errors_and_failure_kinds():
    tolerance = (Tolerance(rtol=5e-2), Tolerance(rtol=0.0, atol=1e-5))
    good = WorkloadCandidate(label="good", source="", checked=3,
                             measured=(MeasuredError(7.8e-3, 0.02), MeasuredError(1e-7, 2e-6)))
    better = WorkloadCandidate(label="better", source="", checked=3,
                               measured=(MeasuredError(3.9e-3, 0.01), MeasuredError(1e-7, 4e-6)))
    wrong = WorkloadCandidate(label="wrong", source="", checked=1, failure="wl_1: OutputMismatch",
                              measured=(MeasuredError(7.0e-3, 0.02), MeasuredError(1e-7, 1e-6)))
    nan = WorkloadCandidate(label="nan", source="", failure="wl_0: NonFiniteOutput")
    broken = WorkloadCandidate(label="broken", source="", failure="compile: InternalError")
    errors = [("wrong", OutputMismatch("output 0: ...")), ("nan", NonFiniteOutput("output 1: ...")),
              ("broken", RuntimeError("no valid warp partition"))]
    plans = SimpleNamespace(refused=[])
    report = DynamicReport(plans, [good, better, wrong, nan, broken], errors, workloads=3,
                           timings={}, tolerance=tolerance)
    assert report.validation_failures() == {"mismatch": 1, "non_finite": 1}
    assert report.measured_error() == [{"rel": 7.8e-3, "abs": 0.02}, {"rel": 1e-7, "abs": 4e-6}]
    validation = json.loads(json.dumps(report.to_dict()))["validation"]
    assert validation["tolerance"] == [{"rtol": 5e-2, "atol": 0.0}, {"rtol": 0.0, "atol": 1e-5}]
    assert validation["failures"] == {"mismatch": 1, "non_finite": 1}
    assert [(m["plan"], m["ok"], m["workloads"]) for m in validation["measured"]] == [
        ("good", True, 3), ("better", True, 3), ("wrong", False, 1)]
    assert validation["max_error"][1]["abs"] == pytest.approx(4e-6)
