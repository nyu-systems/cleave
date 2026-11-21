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
    expected_tolerances_by_role,
    recommend,
    step125,
)
from tests._tolerances import CEILING, FP32_CEILING, TOLERANCES, tolerance_for

PINNED = json.loads(MEASURED.read_text())


def test_declared_tolerances_are_row_relative_bounds_within_the_ceiling():
    assert (CEILING, FP32_CEILING) == (5e-2, 2e-5)
    by_role = {name for name, tolerance in TOLERANCES.items() if isinstance(tolerance, dict)}
    assert by_role == {"exact_split_attention"}
    for name, tolerance in TOLERANCES.items():
        if name in by_role:
            # one bound per boundary role: the program's float16 outputs, and the float32
            # shift a split partial returns, which is held to a float32 bound of its own
            assert set(tolerance) == {"", "shift"}, name
            tolerance, shift = tolerance[""], tolerance["shift"]
            assert isinstance(shift, Tolerance), name
            assert 0 < shift.rtol <= FP32_CEILING < tolerance.rtol and shift.atol == 0, name
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
        elif isinstance(tolerance, dict):
            assert tolerance == expected_tolerances_by_role(outputs), name
            assert all(o["seeds"] >= 3 for o in outputs), name
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
    6.8e-7 of a row max of 0.6; where the rtol is what bounds the output (logits x100) it
    measures 4.1e-7. Taken as an rtol input the first would declare 5e-6 for an operator
    whose declared 1e-6 * rowmax + 1e-6 holds those very rows with 3x to spare.
    (Both are requests with transformed inputs, which are no input for that reason as
    well: the last test of this file. The floor is what leaves out the comparisons on
    normal inputs that have no entry over it, 430 of that operator's 798.)
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

    def record(test, lse_rel, lse_abs, kind=None, param=name):
        bad = kind is not None
        outputs = [
            dict(shape=[7, 32, 128], dtype="bfloat16", rtol=2e-2, atol=0.0, kind=kind,
                 rel=None if bad else 7.8e-3, abs=None if bad else 1.5e-2),
            dict(shape=[7, 32], dtype="float32", rtol=2e-6, atol=1e-6, kind=kind,
                 rel=lse_rel, abs=lse_abs),
        ]
        return json.dumps({"context": f"{path}::{test}[{param}] (call)", "what": "",
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

    # The split-KV pair of an operator is that operator's kernel too, whatever the combine
    # step its test id leads with; another operator's requests are another entry.
    other = "mla_paged_decode_h16_ckv512_kpe64_ps1"
    lines += [
        record("test_exact_split_pair_on_hand_made_requests", 3.1e-7, 1.4e-4,
               param=f"2-{name}"),
        record("test_exact_split_pair_on_hand_made_requests", 1.1e-6, 9.0e-4,
               param=f"1-{other}"),
    ]
    (tmp_path / "validation" / "tests__default.jsonl").write_text("\n".join(lines) + "\n")
    table = load(tmp_path)
    assert set(table) == {f"dynamic:{name}", f"dynamic:{other}"}
    out, lse = measured_errors(table)[f"dynamic:{name}"]
    assert (out["validated"], lse["validated"]) == (3, 3)
    assert (lse["rel"], lse["abs"]) == (3.1e-7, 1.4e-4)
    assert measured_errors(table)[f"dynamic:{other}"][1]["rel"] == 1.1e-6


def test_the_measurement_leaves_out_the_kernels_built_without_the_exact_softmax(tmp_path):
    """``tests/frontend/test_exact_softmax.py`` builds an attention without the stabilize
    request, and the exact program's tilings without the rewrite, to show that validation
    rejects them at x100. Those are no kernels of the operator: neither their passes on
    normal inputs nor their deliberate non-finite failures are part of its measurement, so
    a non-finite failure in its table is always a finding."""
    from tests._measure_tolerances import load, measured_errors, rows

    path = "tests/frontend/test_exact_softmax.py"

    def record(test, what, rel, kind=None):
        output = dict(shape=[32, 64, 64], dtype="float16", rtol=5e-3, atol=0.0, kind=kind,
                      rel=rel, abs=None if rel is None else rel / 4)
        return json.dumps({"context": f"{path}::{test} (call)", "what": what, "kind": kind,
                           "outputs": [output]})

    lines = [
        record("test_gpu_exact_softmax_attention", "", 9.4e-4),
        record("test_gpu_exact_attention_is_held_to_every_adversarial_case", "", 9.7e-4),
        record("test_gpu_exact_attention_is_held_to_every_adversarial_case",
               "large_logits inputs", 1.4e-3),
        record("test_gpu_unstabilized_twin_is_caught_at_x100", "", 9.9e-4),
        record("test_gpu_unstabilized_twin_is_caught_at_x100", "large_logits inputs", None,
               kind="non_finite"),
        record("test_gpu_unstabilized_kernel_fails_against_the_exact_reference", "", 9.8e-4),
        record("test_gpu_unstabilized_kernel_fails_against_the_exact_reference",
               "large_logits inputs", None, kind="non_finite"),
    ]
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "tests__default.jsonl").write_text("\n".join(lines) + "\n")
    table = load(tmp_path)
    assert set(table) == {"test:exact_attention"}
    (row,) = rows(table)
    assert row["n"] == 3 and "non-finite" not in row["histogram"]
    assert (row["largest_valid"], row["adversarial"]) == (9.7e-4, 1)
    (entry,) = measured_errors(table)["test:exact_attention"]
    assert (entry["rel"], entry["validated"]) == (9.7e-4, 3)


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

    # The name of a case marks a dense program's comparison only. A dynamic operator's
    # adversarial workloads are told by their key (``<workload>@<case>``, the test below),
    # so a dense case name in its ``what`` is an input like any other; so is a ``what``
    # that is no adversarial case of the dense programs.
    table = table_of(record("increasing inputs", 3.0e-3, context="dynamic:op"),
                     record("some other inputs", 2.0e-3))
    assert [r["largest_valid"] for r in rows(table)] == [2.0e-3, 3.0e-3]
    assert not any(r["adversarial"] for r in rows(table))
    # ... and the key of a dynamic workload marks nothing on a dense program
    (row,) = rows(table_of(record("wl_1_8_7@increasing", 2.0e-3)))
    assert (row["largest_valid"], row["adversarial"]) == (2.0e-3, 0)


def test_a_paged_operators_adversarial_workloads_are_held_to_its_bound_and_do_not_feed_it(
        tmp_path):
    """A paged operator's softmax is exact, so its kernels are validated on adversarial
    inputs too -- the registry's adversarial workloads and the hand-made requests that
    transform the inputs -- at the bound it declares from its normal inputs. Those
    comparisons are counted, and they are no input to the bound: it is never loosened for
    an adversarial case.

    The numbers are gqa_paged_decode_h32_kv4_d128_ps1's lse: 2.3e-7 on its captured
    workloads, 4.2e-7 at logits x100. As an input the second would turn the declared 1e-6
    into 2e-6. A workload is adversarial by its key, ``<workload>@<case>``, which every
    validation of a paged kernel logs as ``what``.
    """
    from tests._measure_tolerances import (ADVERSARIAL_CASE, is_adversarial, load, markdown,
                                           measured_errors, rows)

    name = "gqa_paged_decode_h32_kv4_d128_ps1"
    demo = f"dynamic:{name}"
    hand_made = ("tests/graph/handwritten/test_paged_exact_softmax.py::"
                 f"test_exact_softmax_on_hand_made_requests[{name}] (call)")
    pair = ("tests/graph/handwritten/test_splitkv_decode.py::"
            f"test_split_pair_matches_the_reference[{name}] (call)")

    def record(context, what, lse_rel, lse_abs, kind=None):
        bad = kind == "non_finite"
        outputs = [
            dict(shape=[16, 32, 128], dtype="bfloat16", rtol=2e-2, atol=0.0, kind=kind,
                 rel=None if bad else 7.8e-3, abs=None if bad else 1.5e-2),
            dict(shape=[16, 32], dtype="float32", rtol=1e-6, atol=1e-6, kind=kind,
                 rel=lse_rel, abs=lse_abs),
        ]
        return json.dumps({"context": context, "what": what, "kind": kind, "outputs": outputs})

    def table_of(*lines):
        directory = tmp_path / f"run{len(list(tmp_path.iterdir()))}"
        (directory / "validation").mkdir(parents=True)
        (directory / "validation" / "x__default.jsonl").write_text("\n".join(lines) + "\n")
        return load(directory)

    assert is_adversarial(demo, "wl_16_1070_1020@large_logits")
    assert is_adversarial(demo, "tile_edges@decreasing")
    assert not is_adversarial(demo, "wl_16_1070_1020") and not is_adversarial(demo, "")
    assert not is_adversarial("demo:mha_decode", "x@y")  # a dense program has no such key
    assert recommend(2.313e-7, "float32") == 1e-6 and recommend(4.208e-7, "float32") == 2e-6

    table = table_of(
        record(demo, "wl_1_8_7", 1.9e-7, 2.4e-6),
        record(demo, "wl_16_1070_1020", 2.313e-7, 2.86e-6),
        record(demo, "wl_16_1070_1020@large_logits", 4.208e-7, 2.44e-4),
        # the captured sample of a split pair, then one of its adversarial workloads
        record(pair, "wl_64_28831_28815", 2.1e-7, 2.6e-6),
        record(pair, "wl_64_28831_28815@increasing", 3.9e-7, 1.8e-4),
        # hand-made requests: as generated, and transformed. The last is within the floor
        # alone (an lse near 0): inside the bound whatever the rtol, as on normal inputs.
        record(hand_made, "tile_edges", 2.2e-7, 1.9e-6),
        record(hand_made, "tile_edges@dominant_last", 4.0e-7, 2.1e-4),
        record(hand_made, "tile_edges@decreasing", 6.845e-7, 4.768e-7),
        record(demo, "wl_1_8_7@large_scale", None, None, kind="non_finite"),
    )
    assert set(table) == {demo}
    assert [e[4] for e in table[demo][1]] == [
        None, None, ADVERSARIAL_CASE, None, ADVERSARIAL_CASE, None, ADVERSARIAL_CASE,
        ADVERSARIAL_CASE, "non_finite"]
    out, lse = rows(table)
    assert (lse["n"], lse["adversarial"], lse["largest_adversarial"]) == (9, 4, 4.208e-7)
    assert (lse["largest_valid"], lse["largest_valid_abs"]) == (2.313e-7, 2.86e-6)
    assert (lse["rtol"], lse["atol"], lse["within_floor"]) == (1e-6, ATOL_FLOOR, 0)
    assert (out["adversarial"], out["largest_valid"], out["rtol"]) == (4, 7.8e-3, 2e-2)
    pinned = measured_errors(table)[demo]
    assert (pinned[1]["rel"], pinned[1]["abs"], pinned[1]["validated"]) == (2.313e-7, 2.86e-6, 9)
    assert expected_tolerances(pinned) == [Tolerance(rtol=2e-2), Tolerance(rtol=1e-6, atol=1e-6)]
    assert expected_tolerances(pinned) == list(OPERATORS[name].tolerance)
    assert "4 on adversarial inputs, largest 4.21e-07: held to the bound" in markdown(table)

    # an adversarial comparison over the bound is no input either: it is a finding
    table = table_of(record(demo, "wl_1_8_7", 2.313e-7, 2.86e-6),
                     record(demo, "wl_1_8_7@increasing", 3.0e-6, 1.5e-3, kind="mismatch"))
    _, lse = rows(table)
    assert (lse["largest_valid"], lse["rtol"], lse["largest_adversarial"]) == (
        2.313e-7, 1e-6, 3.0e-6)
    assert "OVER THE BOUND" in markdown(table)

    # only adversarial comparisons: nothing was measured for the bound
    _, lse = rows(table_of(record(demo, "wl_1_8_7@increasing", 3.9e-7, 1.8e-4)))
    assert lse["largest_valid"] is None and lse["rtol"] is None and lse["adversarial"] == 1

    # without the key the comparison is an input like any other: what tells them apart
    # is the workload a validation names, so every paged validation has to name it
    _, lse = rows(table_of(record(demo, "wl_1_8_7", 2.313e-7, 2.86e-6),
                           record(demo, "", 4.208e-7, 2.44e-4)))
    assert (lse["largest_valid"], lse["rtol"], lse["adversarial"]) == (4.208e-7, 2e-6, 0)


def test_an_output_with_a_boundary_role_is_measured_apart(tmp_path):
    """The split partial of an exact softmax returns its float32 shift in front of its two
    float16 sums. The shift is one output of the operator, under its role, wherever a kernel
    returns it; the sums are numbered as in a partial without it, so the first is measured
    with the output of the solo and combine kernels as before. Each role gets the bound the
    rule gives its own outputs: nothing of one dtype is held to the error of another."""
    from tests._measure_tolerances import load, markdown, measured_errors, rows

    path = "tests/pipeline_api/test_exact_split_attention.py::test_gpu_exact_split_attention"

    def out(dtype, rel, role=None, shape=(32, 16, 64)):
        entry = dict(shape=list(shape), dtype=dtype, rtol=5e-3, atol=0.0, kind=None, rel=rel,
                     abs=rel / 4)
        return {**entry, "role": role} if role else entry

    def record(what, *outputs, kind="mha"):
        return json.dumps({"context": f"{path}[{kind}] (call)", "what": what, "kind": None,
                           "outputs": list(outputs)})

    partial = [out("float32", 9.4e-7, role="shift"), out("float16", 6.5e-4),
               out("float16", 1.1e-3, shape=(32, 16, 64, 64))]
    lines = [
        record("", *partial),
        record("", out("float16", 3.3e-4, shape=(32, 64, 64))),  # a combine
        record("", out("float16", 9.8e-4, shape=(32, 64, 64))),  # the chained Kernel
        # adversarial cases: held to the bounds, no input to them
        record("increasing inputs", out("float32", 1.9e-6, role="shift"),
               out("float16", 8.0e-4), out("float16", 1.6e-3, shape=(32, 16, 64, 64))),
    ]
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "tests__default.jsonl").write_text("\n".join(lines) + "\n")
    table = load(tmp_path)
    assert set(table) == {"test:exact_split_attention"}
    assert list(table["test:exact_split_attention"]) == ["shift", 0, 1]
    by_output = {r["output"]: r for r in rows(table)}
    assert list(by_output) == [0, 1, "shift"]  # the numbered outputs first
    assert (by_output[0]["dtype"], by_output[0]["n"], by_output[0]["largest_valid"]) == (
        "float16", 4, 9.8e-4)
    assert (by_output[1]["dtype"], by_output[1]["largest_valid"]) == ("float16", 1.1e-3)
    shift = by_output["shift"]
    assert (shift["dtype"], shift["n"], shift["largest_valid"]) == ("float32", 2, 9.4e-7)
    assert (shift["adversarial"], shift["largest_adversarial"], shift["rtol"]) == (1, 1.9e-6, 5e-6)

    entries = measured_errors(table)["test:exact_split_attention"]
    assert [e.get("role") for e in entries] == [None, None, "shift"]
    assert [e["dtype"] for e in entries] == ["float16", "float16", "float32"]
    assert expected_tolerances_by_role(entries) == {
        "": Tolerance(rtol=5e-3), "shift": Tolerance(rtol=5e-6)}
    # an operator with such an output has no single bound to declare
    with pytest.raises(AssertionError, match="declare this operator by role"):
        expected_tolerance(entries)
    assert "| test:exact_split_attention | shift | float32 |" in markdown(table)
    # ... and one without is not declared by role
    with pytest.raises(AssertionError):
        expected_tolerances_by_role(entries[:2])


def _measurement(directory, ran, *, tests=(), demos=()):
    """A measurement directory: validation logs of GPU tests and of demo runs (``(node id or
    demo name, row-relative error)`` each, on every seed) and the record of what was run."""
    (directory / "validation").mkdir(parents=True)

    def line(context, rel):
        entry = dict(shape=[8, 8], dtype="float16", rtol=5e-3, atol=0.0, kind=None, rel=rel,
                     abs=rel / 4)
        return json.dumps({"context": context, "what": "", "kind": None, "outputs": [entry]})

    for seed in ("default", "1", "2"):
        if tests:
            (directory / "validation" / f"tests__{seed}.jsonl").write_text(
                "".join(line(f"{node} (call)", rel) + "\n" for node, rel in tests))
        for name, rel in demos:
            (directory / "validation" / f"demo-{name}-0__{seed}.jsonl").write_text(
                line(f"demo:{name}", rel) + "\n")
    if ran is not None:
        (directory / "ran.json").write_text(json.dumps(ran) + "\n")


ALL_SEEDS = ["default", "1", "2"]
E2E = "tests/pipeline_api/test_e2e.py::test_compile_end_to_end"  # the operator "matmul"


def test_a_measurement_of_a_part_pins_the_operators_it_measured_in_full(tmp_path, monkeypatch,
                                                                         capsys):
    """``--only tests`` runs every GPU test, so it measures every test operator in full and
    none of the others: a dynamic operator is also fed by its ``demo.dynamic`` run and a
    demo's by the demo. ``--pin`` replaces the entries of the first kind and leaves the
    others as they were pinned, also a test operator none of whose kernels validated in
    this run (``rowsum_div_graph``: its test is deselected, its entry was measured by hand
    once). (A subset used to be pinned by a script that merged keys by hand; ``--pin`` on
    its directory would have overwritten every operator with the subset's counts.)"""
    import tests._measure_tolerances as tool

    old = [{"dtype": "float16", "rel": 1e-3, "abs": 1e-4, "absolute": False, "seeds": 3,
            "validated": 7}]
    pinned = tmp_path / "measured_errors.json"
    pinned.write_text(json.dumps({"demo:matmul": old, "dynamic:gqa_paged_decode_h32_kv8_d128_ps1": old,
                                  "test:matmul": old, "test:rowsum_div_graph": old}))
    monkeypatch.setattr(tool, "MEASURED", pinned)
    handwritten = "tests/graph/handwritten/test_splitkv_decode.py::test_splitkv_decode"
    run = tmp_path / "run"
    _measurement(run, {"seeds": ALL_SEEDS, "only": ["tests"], "names": None},
                 tests=[(E2E, 4e-4), (f"{handwritten}[2-gqa_paged_decode_h32_kv8_d128_ps1]", 5e-4)])
    tool.main(["--report", str(run), "--pin"])
    now = json.loads(pinned.read_text())
    assert sorted(now) == ["demo:matmul", "dynamic:gqa_paged_decode_h32_kv8_d128_ps1",
                           "test:matmul", "test:rowsum_div_graph"]
    assert now["demo:matmul"] == old and now["dynamic:gqa_paged_decode_h32_kv8_d128_ps1"] == old
    assert now["test:rowsum_div_graph"] == old
    assert "run in full, no kernel validated, entry kept: test:rowsum_div_graph" in (
        capsys.readouterr().err)
    assert (now["test:matmul"][0]["rel"], now["test:matmul"][0]["validated"]) == (4e-4, 3)
    # what the directory itself keeps is the whole table of what it measured
    assert sorted(json.loads((run / "measured_errors.json").read_text())) == [
        "dynamic:gqa_paged_decode_h32_kv8_d128_ps1", "test:matmul"]

    # demos by name: the named demo's operator, and no other demo's
    named = tmp_path / "named"
    _measurement(named, {"seeds": ALL_SEEDS, "only": ["demos"], "names": ["matmul"]},
                 demos=[("matmul", 6e-4)])
    tool.main(["--report", str(named), "--pin"])
    now = json.loads(pinned.read_text())
    assert now["demo:matmul"][0]["rel"] == 6e-4 and now["test:matmul"][0]["rel"] == 4e-4
    in_full = tool.measured_in_full({"only": ["demos"], "names": ["matmul"]})
    assert in_full("demo:matmul") and not in_full("demo:lora") and not in_full("test:matmul")
    # a dynamic operator needs its demo run and the GPU tests
    for only, complete in ((["dynamic"], False), (["tests"], False), (["dynamic", "tests"], True)):
        in_full = tool.measured_in_full({"only": only, "names": None})
        assert in_full("dynamic:gqa_paged_decode_h32_kv8_d128_ps1") is complete

    # everything: every operator it validated is replaced
    full = tmp_path / "full"
    _measurement(full, {"seeds": ALL_SEEDS, "only": ["demos", "dynamic", "tests"], "names": None},
                 tests=[(E2E, 3e-4)], demos=[("matmul", 2e-4)])
    tool.main(["--report", str(full), "--pin"])
    now = json.loads(pinned.read_text())
    assert (now["test:matmul"][0]["rel"], now["demo:matmul"][0]["rel"]) == (3e-4, 2e-4)
    assert now["test:rowsum_div_graph"] == old


@pytest.mark.parametrize(
    ("ran", "why"),
    [(None, "does not say what was run"),
     ({"seeds": ["default"], "only": ["tests"], "names": None}, "a pinned measurement covers")],
    ids=["no record of what was run", "one seed"],
)
def test_a_measurement_that_cannot_say_what_it_covers_pins_nothing(tmp_path, monkeypatch, ran, why):
    import tests._measure_tolerances as tool

    pinned = tmp_path / "measured_errors.json"
    pinned.write_text("{}\n")
    monkeypatch.setattr(tool, "MEASURED", pinned)
    _measurement(tmp_path / "run", ran, tests=[(E2E, 4e-4)])
    with pytest.raises(SystemExit, match=why):
        tool.main(["--report", str(tmp_path / "run"), "--pin"])
    assert pinned.read_text() == "{}\n"
    tool.main(["--report", str(tmp_path / "run")])  # the table alone needs no record


def test_a_tiling_test_of_a_file_with_several_builders_has_its_own_operator(tmp_path):
    """A validation a tiling test logs is measured under the graph builder that test passed
    to ``tolerance_for``. Two files validate at several builders (and import them in one
    statement): the tool could not load a measurement that ran them (a ``TypeError`` while
    looking for the one builder of the file)."""
    from tests._measure_tolerances import _test_operators, load

    operator_of = _test_operators()
    tiling = "tests/graph/apply_tiling/"
    single = tiling + "test_unit_extents.py::test_a_loop_over_a_single_row_block_validates"
    assert {name: operator_of(f"{single}[{name}]") for name in (
        "rowsum_one_row", "combine_one_row", "combine_one_column", "combine_one_element",
        "batched_combine_one_row")} == {
        "rowsum_one_row": "rowsum_graph",
        "combine_one_row": "matmul_splitk_combine_graph",
        "combine_one_column": "matmul_splitk_combine_graph",
        "combine_one_element": "matmul_splitk_combine_graph",
        "batched_combine_one_row": "batched_matmul_splitk_combine_graph"}
    assert operator_of(tiling + "test_unit_extents.py::"
                       "test_a_block_that_computes_one_element_validates") == "rowsum_graph"
    axes = tiling + "test_reduction_axes.py::"
    assert operator_of(axes + "test_unlooped_sum_over_a_front_named_axis_validates[rows128]") \
        == "batched_matmul_splitk_combine_graph"
    assert operator_of(axes + "test_unlooped_sum_over_a_loops_accumulator_validates[cols128]") \
        == "matmul_rowsum_graph"
    # a file with one builder is read as before
    assert operator_of(tiling + "test_matmul.py::test_apply_tiling_matmul[x]") == "matmul_graph"
    # a test of such a file that is not listed is named, not guessed
    with pytest.raises(AssertionError, match="say which one in SEVERAL_BUILDERS"):
        operator_of(tiling + "test_unit_extents.py::test_some_new_validation")

    # ... and a measurement that ran them loads, each under its own operator
    def record(node, rel):
        return json.dumps({"context": f"{node} (call)", "what": "", "kind": None, "outputs": [
            dict(shape=[4, 128], dtype="float16", rtol=1e-3, atol=0.0, kind=None, rel=rel,
                 abs=rel)]})

    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "tests__default.jsonl").write_text("\n".join([
        record(f"{single}[rowsum_one_row]", 1e-4),
        record(f"{single}[combine_one_row]", 0.0),
        record(axes + "test_unlooped_sum_over_a_loops_accumulator_validates[cols128]", 2e-4),
    ]) + "\n")
    assert set(load(tmp_path)) == {
        "test:rowsum_graph", "test:matmul_splitk_combine_graph", "test:matmul_rowsum_graph"}


def test_every_tiling_test_that_passes_a_tolerance_has_an_operator():
    """Every test under tests/graph/apply_tiling and tests/reference that passes a declared
    tolerance resolves to a declared operator: its file imports one graph builder on a line
    of its own, or names the builder of each such test in ``SEVERAL_BUILDERS``."""
    import ast
    import re
    from pathlib import Path

    from tests._measure_tolerances import ROOT, SEVERAL_BUILDERS, _test_operators

    operator_of = _test_operators()
    # a file pytest.ini leaves out of every run logs nothing
    ignored = re.findall(r"--ignore=(\S+)", (Path(ROOT) / "pytest.ini").read_text())
    seen = 0
    for directory in ("tests/graph/apply_tiling", "tests/reference"):
        for path in sorted((Path(ROOT) / directory).glob("test_*.py")):
            source = path.read_text()
            rel = f"{directory}/{path.name}"
            if rel in ignored:
                continue
            for node in ast.parse(source).body:
                if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                    continue
                if "tolerance_for(" not in ast.get_source_segment(source, node):
                    continue
                seen += 1
                if rel in SEVERAL_BUILDERS:
                    assert node.name in SEVERAL_BUILDERS[rel], (rel, node.name)
                else:
                    assert operator_of(f"{rel}::{node.name}") in TOLERANCES, (rel, node.name)
    assert seen >= 30
    # the listed tests exist, and name declared operators
    for rel, tests in SEVERAL_BUILDERS.items():
        source = (Path(ROOT) / rel).read_text()
        for name, builder in tests.items():
            assert f"def {name}(" in source, (rel, name)
            if isinstance(builder, str):
                assert builder in TOLERANCES and f"tolerance_for({builder})" in source
