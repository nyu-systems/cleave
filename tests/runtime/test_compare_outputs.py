"""The comparator contract (CPU): row-relative ``rtol``, absolute ``atol``, strict non-finites.

``compare_outputs`` is the one comparison behind every validation (static workers, the paged
autotuner, the hand-written kernel tests), so its rules are pinned here once.
"""

import json
import math

import pytest

torch = pytest.importorskip("torch")

from cleave.core.errors import NonFiniteOutput, OutputMismatch, SpecError, ValidationError
from cleave.runtime.validate import MeasuredError, Tolerance, compare_outputs, worst

INF = float("inf")


def _row(*values, dtype=torch.float32):
    return torch.tensor([list(values)], dtype=dtype)


def _passes(actual, reference, tolerance):
    try:
        compare_outputs([actual], [reference], tolerance)
    except OutputMismatch:
        return False
    return True


def test_rtol_is_relative_to_the_row_max_not_to_the_entry():
    reference = _row(100.0, 1.0, 0.0)
    tol = Tolerance(rtol=1e-2)  # one unit, for this row
    # A small entry may be off by far more than its own size...
    assert _passes(_row(100.0, 1.9, 0.9), reference, tol)
    # ...and a large one by no more than the same absolute amount.
    assert _passes(_row(100.9, 1.0, 0.0), reference, tol)
    assert not _passes(_row(101.1, 1.0, 0.0), reference, tol)
    assert not _passes(_row(100.0, 1.0, 1.1), reference, tol)
    # Each row has its own scale: the same absolute error fails in a small row.
    reference = torch.tensor([[100.0, 1.0], [1.0, 0.5]])
    actual = reference.clone()
    actual[1, 1] += 0.9
    assert not _passes(actual, reference, tol)
    actual = reference.clone()
    actual[0, 1] += 0.9
    assert _passes(actual, reference, tol)


def test_measured_error_is_returned_per_output():
    reference = torch.tensor([[4.0, 2.0], [0.5, 0.25]])
    actual = reference + torch.tensor([[0.0, 0.2], [0.05, 0.0]])
    lse = torch.tensor([3.0, 5.0])
    measured = compare_outputs([actual, lse + 0.25], [reference, lse],
                               (Tolerance(rtol=0.2), Tolerance(rtol=0.0, atol=0.5)))
    assert measured[0].rel == pytest.approx(0.1)  # 0.05 / 0.5, not 0.2 / 4
    assert measured[0].abs == pytest.approx(0.2)
    assert measured[1].abs == pytest.approx(0.25)
    assert measured[1].rel == pytest.approx(0.25 / 5.0)  # a vector is one row
    assert worst(measured, None) == measured
    assert worst(measured, (MeasuredError(0.5, 0.01), MeasuredError(0.0, 1.0))) == (
        MeasuredError(0.5, pytest.approx(0.2)), MeasuredError(pytest.approx(0.05), 1.0))


def test_an_all_zero_row_demands_exact_zeros():
    reference = torch.zeros(2, 4)
    assert _passes(torch.zeros(2, 4), reference, Tolerance(rtol=1e-2))
    dirty = torch.zeros(2, 4)
    dirty[1, 2] = 1e-30
    # No division anywhere: a zero scale leaves only atol, and rtol cannot buy slack.
    assert not _passes(dirty, reference, Tolerance(rtol=1e3))
    assert _passes(dirty, reference, Tolerance(rtol=0.0, atol=1e-20))
    (measured,) = compare_outputs([dirty], [reference], Tolerance(rtol=0.0, atol=1e-20))
    assert measured == MeasuredError(0.0, pytest.approx(1e-30))  # no live row: rel is 0


def test_unit_axes_are_dropped_before_the_row_is_chosen():
    reference = torch.tensor([10.0, 1.0, 0.1])
    actual = reference + torch.tensor([0.0, 0.0, 0.5])
    tol = Tolerance(rtol=0.06)  # 0.6 against the global max 10; hopeless against 0.1
    # A vector is one row, however a kernel stores its unit axes.
    for shape in [(3,), (3, 1), (1, 3), (1, 3, 1)]:
        (measured,) = compare_outputs([actual.reshape(shape)], [reference.reshape(shape)], tol)
        assert measured.rel == pytest.approx(0.05), shape
    # With a real second axis the row is that axis again: (2, 3, 1) rows run along the 3.
    block = torch.stack([reference, reference * 1e-3]).reshape(2, 3, 1)
    off = block.clone()
    off[1, 2, 0] += 0.5  # row max 1e-2
    assert not _passes(off, block, tol)
    off = block.clone()
    off[0, 2, 0] += 0.5
    assert _passes(off, block, tol)


def test_shape_and_dtype_are_asserted_not_broadcast():
    reference = torch.ones(4, dtype=torch.float16)
    tol = Tolerance(rtol=1.0, atol=1.0)
    with pytest.raises(OutputMismatch, match=r"shape \(4, 1\) differs"):
        compare_outputs([torch.ones(4, 1, dtype=torch.float16)], [reference], tol)
    with pytest.raises(OutputMismatch, match="dtype torch.float32 differs"):
        compare_outputs([torch.ones(4)], [reference], tol)
    with pytest.raises(OutputMismatch, match="returned 2 outputs but the reference has 1"):
        compare_outputs([reference, reference], [reference], tol)


def test_any_nan_fails_even_against_a_nan_reference():
    nan = float("nan")
    tol = Tolerance(rtol=1.0, atol=1.0)
    for actual, reference in [(_row(1.0, nan), _row(1.0, 2.0)), (_row(1.0, nan), _row(1.0, nan)),
                              (_row(1.0, 2.0), _row(1.0, nan))]:
        with pytest.raises(NonFiniteOutput, match="NaN") as error:
            compare_outputs([actual], [reference], tol)
        assert isinstance(error.value, ValidationError) and error.value.measured is None


def test_non_finite_reference_entries_must_be_reproduced_exactly():
    reference = _row(1.0, -INF, 0.5)
    tol = Tolerance(rtol=0.1)
    assert _passes(_row(1.05, -INF, 0.5), reference, tol)  # -inf is a defined answer
    for wrong in (-3.0e38, INF, 0.0):
        with pytest.raises(NonFiniteOutput, match="not reproduced"):
            compare_outputs([_row(1.0, wrong, 0.5)], [reference], tol)
    with pytest.raises(NonFiniteOutput, match="infinite kernel entries"):
        compare_outputs([_row(INF, -INF, 0.5)], [reference], tol)
    # The row scale is 1.0, not inf: an infinite scale would accept this error.
    assert not _passes(_row(1.0, -INF, 0.9), reference, tol)
    (measured,) = compare_outputs([_row(1.05, -INF, 0.5)], [reference], tol)
    assert measured.rel == pytest.approx(0.05) and math.isfinite(measured.abs)
    # A row with nothing finite has no scale at all; matching it exactly is all there is.
    assert _passes(_row(-INF, -INF), _row(-INF, -INF), Tolerance(rtol=0.0))


def test_non_finite_takes_precedence_over_a_mismatch():
    tol = Tolerance(rtol=1e-3)
    with pytest.raises(NonFiniteOutput):
        compare_outputs([_row(5.0, 1.0), _row(float("nan"))], [_row(1.0, 1.0), _row(1.0)], tol)


def test_a_non_finite_failure_says_whether_the_rest_of_the_output_is_right():
    reference = torch.tensor([[4.0, 2.0, 1.0], [8.0, 1.0, INF]])
    tol = Tolerance(rtol=1e-2)
    # one stray infinity in an otherwise exact output ...
    stray = reference.clone()
    stray[0, 1] = INF
    with pytest.raises(NonFiniteOutput) as error:
        compare_outputs([stray], [reference], tol)
    assert str(error.value) == (
        "output 0: 1 infinite kernel entries where the reference is finite, 0 non-finite "
        "reference entries not reproduced (of 6); of the 4 entries finite on both sides 0 are "
        "out of tolerance (row-normalized error 0.000e+00)")
    # ... is not the same finding as a kernel that is also wrong where it is finite
    wrong = stray.clone()
    wrong[1, 0] = 4.0
    with pytest.raises(NonFiniteOutput, match=r"of the 4 entries finite on both sides 1 are "
                                              r"out of tolerance \(row-normalized error 5.000e-01\)"):
        compare_outputs([wrong], [reference], tol)
    # a NaN is reported as such, with the same tail
    nan = reference.clone()
    nan[0, 0] = float("nan")
    with pytest.raises(NonFiniteOutput, match=r"1 NaN in the kernel output, 0 in the reference "
                                              r"\(of 6 entries\); of the 4 entries finite"):
        compare_outputs([nan], [reference], tol)
    assert error.value.measured is None  # never a measurement of a valid kernel


def test_a_mismatch_reports_what_it_measured():
    reference = _row(2.0, 1.0)
    with pytest.raises(OutputMismatch, match="1 of 2 entries out of tolerance") as error:
        compare_outputs([_row(2.0, 1.5)], [reference], Tolerance(rtol=1e-2), what="kernel 7")
    assert str(error.value).startswith("kernel 7: output 0")
    assert "\n" not in str(error.value)  # one line: reports print the last line of an error
    assert error.value.measured == (MeasuredError(pytest.approx(0.25), pytest.approx(0.5)),)
    assert isinstance(error.value, AssertionError)


def test_tolerances_are_per_output_and_never_implied():
    out, lse = torch.ones(2, 3), torch.ones(2)
    both = (Tolerance(rtol=1e-2), Tolerance(rtol=0.0, atol=1e-2))
    compare_outputs([out, lse], [out, lse], both)
    compare_outputs([out, lse], [out, lse], list(both))
    compare_outputs([out, lse], [out, lse], Tolerance(rtol=1e-2))  # one for every output
    with pytest.raises(OutputMismatch, match="output 1"):  # the lse bound is its own
        compare_outputs([out, lse + 0.02], [out, lse], both)
    with pytest.raises(SpecError, match="1 tolerances were declared for 2 outputs"):
        compare_outputs([out, lse], [out, lse], both[:1])
    with pytest.raises(SpecError, match="no validation tolerance declared"):
        compare_outputs([out], [out], None)
    with pytest.raises(SpecError, match="must be a Tolerance"):
        compare_outputs([out], [out], (0.5,))


@pytest.mark.parametrize("fields", [dict(rtol=0.0, atol=-1.0), dict(rtol=-1e-3),
                                    dict(rtol=float("nan")), dict(rtol=1e-3, atol=INF),
                                    dict(rtol="1e-3"), dict(rtol=True)])
def test_a_tolerance_that_is_not_a_bound_is_rejected_where_it_is_declared(fields):
    with pytest.raises(SpecError, match="must be a finite number >= 0"):
        Tolerance(**fields)


def test_tolerance_has_no_equal_nan_and_no_default_rtol():
    assert Tolerance(rtol=1e-3) == Tolerance(1e-3, 0.0)
    with pytest.raises(TypeError):
        Tolerance()
    with pytest.raises(TypeError):
        Tolerance(rtol=1e-3, atol=0.0, equal_nan=False)


def test_empty_outputs_compare_equal():
    empty = torch.zeros(0, 4)
    assert compare_outputs([empty], [empty], Tolerance(rtol=0.0)) == (MeasuredError(0.0, 0.0),)


def test_validation_log_records_passes_and_failures(tmp_path, monkeypatch):
    log = tmp_path / "sub" / "validation.jsonl"
    monkeypatch.setenv("CLEAVE_VALIDATION_LOG", str(log))
    monkeypatch.setenv("CLEAVE_VALIDATION_CONTEXT", "unit")
    reference = _row(2.0, 1.0, dtype=torch.float16)
    tol = Tolerance(rtol=1e-2)
    compare_outputs([reference], [reference], tol, what="good")
    with pytest.raises(OutputMismatch):
        compare_outputs([_row(2.0, 1.5, dtype=torch.float16)], [reference], tol, what="bad")
    with pytest.raises(NonFiniteOutput):
        compare_outputs([_row(2.0, INF, dtype=torch.float16)], [reference], tol)
    good, bad, non_finite = [json.loads(line) for line in log.read_text().splitlines()]
    assert good == {"context": "unit", "what": "good", "kind": None, "outputs": [
        {"shape": [1, 2], "dtype": "float16", "rtol": 1e-2, "atol": 0.0, "kind": None,
         "rel": 0.0, "abs": 0.0}]}
    assert bad["kind"] == "mismatch" and bad["outputs"][0]["rel"] == pytest.approx(0.25)
    assert non_finite["kind"] == "non_finite" and non_finite["outputs"][0]["rel"] is None
    monkeypatch.delenv("CLEAVE_VALIDATION_LOG")
    compare_outputs([reference], [reference], tol)
    assert len(log.read_text().splitlines()) == 3
