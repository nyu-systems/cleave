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
