"""CPU-only: every operator that is validated declares its own bound, within the ceiling.

There is no default tolerance to fall back to, so a fixture whose operator is missing from
``tests/_tolerances.py`` cannot validate at all; these checks fail first and say which one.
"""

from cleave.dynamic import OPERATORS
from cleave.runtime.validate import Tolerance

from tests._measure_tolerances import recommend, step125
from tests._tolerances import CEILING, TOLERANCES, tolerance_for


def test_declared_tolerances_are_row_relative_bounds_within_the_ceiling():
    assert CEILING == 5e-2
    for name, tolerance in TOLERANCES.items():
        assert isinstance(tolerance, Tolerance), name
        # fp16 outputs: relative to the row, no absolute slack, never the old 0.5
        assert 1e-3 <= tolerance.rtol <= CEILING and tolerance.atol == 0, name
    for name, operator in OPERATORS.items():
        out, lse = operator.tolerance
        assert 0 < out.rtol <= CEILING and out.atol == 0, name
        assert lse.rtol == 0 and 0 < lse.atol <= CEILING, name


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
    # 4x the largest valid error, rounded up to a 1-2-5 step ...
    assert recommend(9.8e-4, "float16", True) == 5e-3
    assert recommend(1.3e-3, "float16", True) == 1e-2
    assert recommend(2.9e-6, "float32", False) == 2e-5
    # ... never below one ulp of the output dtype (an exact operator is not held to 0) ...
    assert recommend(0.0, "float16", True) == 1e-3
    assert recommend(1e-4, "bfloat16", True) == 1e-2
    # ... and never above the ceiling: bf16 kernels sit at one ulp, 7.8e-3 of the row max
    assert recommend(7.8e-3, "bfloat16", True) == CEILING
    assert recommend(0.02, "bfloat16", True) == CEILING
