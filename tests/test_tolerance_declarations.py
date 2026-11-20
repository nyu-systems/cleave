"""CPU-only: every operator that is validated declares its own bound, and that bound is
what the rule gives for the measurement pinned in ``tests/measured_errors.json``.

There is no default tolerance to fall back to, so a fixture whose operator is missing from
``tests/_tolerances.py`` cannot validate at all; these checks fail first and say which one.
A declared value that was edited by hand, in either direction, fails here: re-measure
(``python -m tests._measure_tolerances --out DIR --pin``) instead.
"""

import json

import pytest
import torch

from cleave.core.errors import OutputMismatch
from cleave.dynamic import OPERATORS
from cleave.dynamic.registry import LSE_ATOL
from cleave.runtime.validate import Tolerance, compare_outputs

from tests._measure_tolerances import (
    ATOL_FLOOR,
    MEASURED,
    expected_tolerance,
    expected_tolerances,
    recommend,
    step125,
)
from tests._tolerances import CEILING, FP32_CEILING, TOLERANCES, tolerance_for

PINNED = json.loads(MEASURED.read_text())


def test_declared_tolerances_are_row_relative_bounds_within_the_ceiling():
    assert (CEILING, FP32_CEILING) == (5e-2, 2e-5)
    for name, tolerance in TOLERANCES.items():
        assert isinstance(tolerance, Tolerance), name
        # fp16 outputs: relative to the row, no absolute slack, never the old 0.5
        assert 1e-3 <= tolerance.rtol <= CEILING and tolerance.atol == 0, name
    for name, operator in OPERATORS.items():
        out, lse = operator.tolerance
        assert 0 < out.rtol <= CEILING and out.atol == 0, name
        # The fp32 lse is held relative to its row too (its error scales with |lse|), at the
        # scale it measures, plus the one fixed floor for an lse of exactly 0.
        assert 0 < lse.rtol <= FP32_CEILING and lse.atol == LSE_ATOL == ATOL_FLOOR == 1e-6, name


def test_every_declared_tolerance_follows_from_the_pinned_measurement():
    """declared == rule(measured), for every test operator and every paged operator.

    (``tests/demo/test_demos.py`` holds the demos to the same file.) Nothing may be looser
    than its measurement supports, and nothing tighter than the rule's margin either.
    """
    unmeasured = set()
    for name, tolerance in TOLERANCES.items():
        outputs = PINNED.get(f"test:{name}")
        if outputs is None:
            unmeasured.add(name)
        else:
            assert tolerance == expected_tolerance(outputs), name
    # No kernel of this operator validates (its tests are deselected), so there is nothing
    # to measure: it declares the floor, one fp16 ulp of the row max.
    assert unmeasured == {"rmsnorm_matmul_split_combine_graph"}
    assert TOLERANCES["rmsnorm_matmul_split_combine_graph"] == Tolerance(
        rtol=recommend(0.0, "float16"))
    measured_tests = {key.removeprefix("test:") for key in PINNED if key.startswith("test:")}
    assert measured_tests == set(TOLERANCES) - unmeasured

    for name, operator in OPERATORS.items():
        outputs = PINNED[f"dynamic:{name}"]
        assert [o["dtype"] for o in outputs] == ["bfloat16", "float32"], name
        assert [o["absolute"] for o in outputs] == [False, True], name
        assert all(o["seeds"] >= 3 for o in outputs), name
        assert list(operator.tolerance) == expected_tolerances(outputs), name
    assert {key.removeprefix("dynamic:") for key in PINNED if key.startswith("dynamic:")} == set(
        OPERATORS)


@pytest.mark.parametrize("magnitude", [1.0, 16.0, 450.0, 1500.0])
def test_the_lse_bound_scales_with_the_lse(magnitude):
    """A correct lse may be a couple of fp32 ulps off at any magnitude, and no more than that.

    Logits x100 put |lse| in the hundreds, where one fp32 ulp (3e-5 at 450, 1.2e-4 at 1500)
    is larger than any absolute bound sized on the captured workloads (|lse| of 2 .. 25).
    """
    generator = torch.Generator().manual_seed(0)
    up = torch.tensor(float("inf"))
    for name, operator in OPERATORS.items():
        _, lse_tol = operator.tolerance
        heads = operator.dims[0]
        lse = magnitude * (1 + torch.rand(8, heads, generator=generator))
        lse[:, ::2] *= -1
        lse[3] = float("-inf")  # an empty request, reproduced exactly
        two_ulps = torch.where(torch.isfinite(lse),
                               torch.nextafter(torch.nextafter(lse, up), up), lse)
        (measured,) = compare_outputs([two_ulps], [lse], lse_tol)
        assert 0 < measured.rel < lse_tol.rtol, name
        # ... while an error of 1e-4 of the row max is a wrong kernel at every magnitude.
        with pytest.raises(OutputMismatch):
            compare_outputs([lse * (1 + 1e-4)], [lse], lse_tol)


def test_an_lse_of_zero_is_held_to_the_floor_not_to_an_exact_match():
    for name, operator in OPERATORS.items():
        _, lse_tol = operator.tolerance
        zeros = torch.zeros(2, operator.dims[0])
        assert compare_outputs([zeros + 0.5 * LSE_ATOL], [zeros], lse_tol), name
        with pytest.raises(OutputMismatch):
            compare_outputs([zeros + 2 * LSE_ATOL], [zeros], lse_tol)


def test_every_validating_fixture_has_a_declared_operator():
    from tests.graph.handwritten._cases import CASES as handwritten
    from tests.schedule.end2end.test_end2end import CASES as end2end

    for name, (build, *_rest) in end2end.items():
        assert tolerance_for(build) is TOLERANCES[build.__name__], name
    for name, case in handwritten.items():
        if case.out_idx is None:
            assert case.tolerance is tolerance_for(case.source), name
        else:  # a paged operator: the registry's (out, lse) pair
            assert case.tolerance is OPERATORS[name].tolerance, name


def test_the_rule_that_turns_a_measured_error_into_a_tolerance():
    assert [step125(x) for x in (9.0e-4, 1.0e-3, 1.1e-3, 2.0e-3, 3.9e-3, 5.1e-3, 0.0)] == [
        1e-3, 1e-3, 2e-3, 2e-3, 5e-3, 1e-2, 0.0]
    # 4x the largest valid row-relative error, rounded up to a 1-2-5 step ...
    assert recommend(9.8e-4, "float16") == 5e-3
    assert recommend(1.3e-3, "float16") == 1e-2
    assert recommend(3.2e-7, "float32") == 2e-6
    # ... never below one ulp of the output dtype (an exact operator is not held to 0) ...
    assert recommend(0.0, "float16") == 1e-3
    assert recommend(1e-4, "bfloat16") == 1e-2
    assert recommend(0.0, "float32") == 2e-7
    # ... and never above the dtype's ceiling.
    assert recommend(0.02, "float16") == CEILING
    assert recommend(1e-4, "float32") == FP32_CEILING
    # bf16 is the exception to 4x. Its kernels sit at one ulp, 7.8e-3 of the row max, so 4x
    # would be the ceiling itself: looser than the elementwise 0.01 + 0.01|ref| the paged
    # operators were held to before. It gets 2x, which is 2.56 ulps.
    assert recommend(7.8e-3, "bfloat16") == 2e-2
    assert recommend(9.4e-3, "bfloat16") == 2e-2
    assert recommend(0.02, "bfloat16") == CEILING
    # An operator declares one Tolerance: the loosest of what its outputs need.
    two = [dict(dtype="float16", rel=9.8e-4, absolute=False),
           dict(dtype="float16", rel=1.6e-3, absolute=False),
           dict(dtype="float16", rel=None, absolute=False)]  # never valid: nothing to add
    assert expected_tolerance(two) == Tolerance(rtol=1e-2)
    assert expected_tolerances(two) == [Tolerance(rtol=5e-3), Tolerance(rtol=1e-2), None]
    # An output with an absolute part gets the fixed floor; its measured part is the rtol.
    lse = [dict(dtype="float32", rel=5.7e-7, absolute=True)]
    assert expected_tolerances(lse) == [Tolerance(rtol=5e-6, atol=1e-6)]


def test_a_comparison_within_the_absolute_floor_does_not_feed_the_rtol():
    """The lse bound is ``rtol * rowmax + LSE_ATOL``. A comparison whose every entry is
    within the floor is inside it whatever the rtol: it is counted, and it is no input to
    the rtol. One entry over the floor and it is an input in full.

    The numbers are gqa_paged_decode_h32_kv8_d128_ps1's. Its rows with an lse near 0
    (logits that only fall along the keys) are off by 3.7e-7 .. 4.8e-7, which is 5.0e-7 ..
    6.8e-7 of a row max of 0.6; where the rtol is what bounds the output it measures
    4.1e-7. Taken as an rtol input the first would declare 5e-6 for an operator whose
    declared 2e-6 * rowmax + 1e-6 holds those very rows with 5x to spare.
    """
    from tests._measure_tolerances import feeds_rtol, measured_errors, rows

    def table(*lse, absolute=True):
        return {"dynamic:op": {0: [("float32", absolute, rel, abs_, None, "default")
                                   for rel, abs_ in lse]}}

    scaled, near_zero = (4.112e-7, 2.44e-4), (6.845e-7, 4.768e-7)
    assert recommend(scaled[0], "float32") == 2e-6 and recommend(near_zero[0], "float32") == 5e-6
    assert feeds_rtol(True, scaled[1]) and not feeds_rtol(True, near_zero[1])
    assert feeds_rtol(True, 1.01 * ATOL_FLOOR) and not feeds_rtol(True, ATOL_FLOOR)
    assert feeds_rtol(False, near_zero[1]), "no floor, nothing that covers it"

    (row,) = rows(table(scaled, near_zero))
    assert (row["largest_valid"], row["within_floor"], row["n"]) == (scaled[0], 1, 2)
    assert row["largest_valid_abs"] == scaled[1]  # abs and the count see every comparison
    assert (row["rtol"], row["atol"]) == (2e-6, ATOL_FLOOR)
    (entry,) = measured_errors(table(scaled, near_zero))["dynamic:op"]
    assert (entry["rel"], entry["abs"], entry["validated"]) == (scaled[0], scaled[1], 2)
    assert expected_tolerances([entry]) == [Tolerance(rtol=2e-6, atol=LSE_ATOL)]
    # the same row-relative error with one entry over the floor is an input like any other
    (row,) = rows(table(scaled, (near_zero[0], 1.2e-6)))
    assert (row["largest_valid"], row["within_floor"], row["rtol"]) == (near_zero[0], 0, 5e-6)
    # an output without an absolute part has no floor to be within
    (row,) = rows(table(scaled, near_zero, absolute=False))
    assert (row["largest_valid"], row["within_floor"], row["atol"]) == (near_zero[0], 0, 0.0)
    # only comparisons the floor covers: valid, and nothing left for the rtol to cover
    (row,) = rows(table(near_zero))
    assert (row["largest_valid"], row["within_floor"]) == (0.0, 1)
    assert row["rtol"] == recommend(0.0, "float32") == 2e-7
    # no valid comparison at all: nothing was measured
    (row,) = rows({"dynamic:op": {0: [("float32", True, None, None, "non_finite", "default")]}})
    assert row["largest_valid"] is None and row["rtol"] is None and row["within_floor"] == 0


def test_the_measurement_counts_hand_made_requests_and_not_a_kernel_broken_on_purpose(tmp_path):
    """``tests/graph/handwritten/test_paged_exact_softmax.py`` runs an operator's kernel on
    hand-made requests, which are that operator's validations like any other, and once more
    as it was before its softmax was exact, which is no kernel of the operator at all."""
    from tests._measure_tolerances import load, measured_errors

    name = "gqa_paged_decode_h32_kv8_d128_ps1"
    path = "tests/graph/handwritten/test_paged_exact_softmax.py"

    def record(test, lse_rel, lse_abs, kind=None):
        bad = kind is not None
        outputs = [
            dict(shape=[7, 32, 128], dtype="bfloat16", rtol=2e-2, atol=0.0, kind=kind,
                 rel=None if bad else 7.8e-3, abs=None if bad else 1.5e-2),
            dict(shape=[7, 32], dtype="float32", rtol=2e-6, atol=1e-6, kind=kind,
                 rel=lse_rel, abs=lse_abs),
        ]
        return json.dumps({"context": f"{path}::{test}[{name}] (call)", "what": "",
                           "kind": kind, "outputs": outputs})

    lines = [
        record("test_exact_softmax_on_hand_made_requests", 2.7e-7, 1.2e-4),
        record("test_exact_softmax_on_hand_made_requests", 6.845e-7, 4.768e-7),
        record("test_without_the_running_max_the_adversarial_workloads_fail", 9.0e-6, 3.0e-3),
        record("test_without_the_running_max_the_adversarial_workloads_fail", None, None,
               kind="non_finite"),
    ]
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "tests__default.jsonl").write_text("\n".join(lines) + "\n")
    table = load(tmp_path)
    assert set(table) == {f"dynamic:{name}"}
    out, lse = measured_errors(table)[f"dynamic:{name}"]
    assert (out["validated"], lse["validated"]) == (2, 2)
    assert (lse["rel"], lse["abs"], lse["absolute"]) == (2.7e-7, 1.2e-4, True)
    assert out["rel"] == 7.8e-3 and not out["absolute"]


def test_a_dense_programs_adversarial_cases_are_held_to_its_bound_and_do_not_feed_it(tmp_path):
    """A dense program whose softmax is exact is validated on the adversarial softmax inputs
    too, at the bound it declares from its normal inputs. Those comparisons are counted, and
    they are no input to the bound: it is never loosened for an adversarial case.

    The numbers are the exact MHA attention's: 9.8e-4 .. 1.2e-3 on normal inputs, up to
    1.63e-3 on logits that rise along the keys (two fp16 ulps of a row that its last keys
    dominate). As an input that last figure would turn the declared 5e-3 into 1e-2.
    """
    from tests._measure_tolerances import ADVERSARIAL_CASE, load, markdown, measured_errors, rows

    def record(what, rel, kind=None, context="demo:mha_decode_exact"):
        output = dict(shape=[32, 64, 64], dtype="float16", rtol=5e-3, atol=0.0, kind=kind,
                      rel=rel, abs=None if rel is None else rel / 4)
        return json.dumps({"context": context, "what": what, "kind": kind, "outputs": [output]})

    def table_of(*lines):
        directory = tmp_path / f"run{len(list(tmp_path.iterdir()))}"
        (directory / "validation").mkdir(parents=True)
        (directory / "validation" / "demo-x-0__default.jsonl").write_text("\n".join(lines) + "\n")
        return load(directory)

    assert recommend(1.2e-3, "float16") == 5e-3 and recommend(1.63e-3, "float16") == 1e-2
    table = table_of(
        record("", 9.8e-4), record("", 1.2e-3),
        record("increasing inputs", 1.63e-3), record("offset inputs", 9.77e-4),
        record("large_logits inputs", None, kind="non_finite"))
    assert [e[4] for e in table["demo:mha_decode_exact"][0]] == [
        None, None, ADVERSARIAL_CASE, ADVERSARIAL_CASE, "non_finite"]
    (row,) = rows(table)
    assert (row["n"], row["adversarial"], row["largest_adversarial"]) == (5, 2, 1.63e-3)
    assert (row["largest_valid"], row["rtol"]) == (1.2e-3, 5e-3)
    assert row["histogram"] == {"<1e-3": 2, "<1e-2": 2, "non-finite": 1}
    (entry,) = measured_errors(table)["demo:mha_decode_exact"]
    assert (entry["rel"], entry["abs"], entry["validated"]) == (1.2e-3, 3.0e-4, 5)
    assert expected_tolerance([entry]) == Tolerance(rtol=5e-3)
    assert "2 on adversarial inputs, largest 1.63e-03: held to the bound" in markdown(table)

    # an adversarial comparison over the bound is no input either: it is a finding
    table = table_of(record("", 9.8e-4), record("decreasing inputs", 7.0e-3, kind="mismatch"))
    (row,) = rows(table)
    assert (row["largest_valid"], row["rtol"], row["largest_adversarial"]) == (9.8e-4, 5e-3, 7e-3)
    assert "OVER THE BOUND" in markdown(table)

    # only adversarial comparisons: nothing was measured for the bound
    (row,) = rows(table_of(record("increasing inputs", 1.0e-3)))
    assert row["largest_valid"] is None and row["rtol"] is None and row["adversarial"] == 1

    # a dynamic operator names its adversarial workloads by key, not in ``what``, and counts
    # them; a ``what`` that is no adversarial case of the dense programs feeds as before
    table = table_of(record("increasing inputs", 3.0e-3, context="dynamic:op"),
                     record("some other inputs", 2.0e-3))
    assert [r["largest_valid"] for r in rows(table)] == [2.0e-3, 3.0e-3]
    assert not any(r["adversarial"] for r in rows(table))
