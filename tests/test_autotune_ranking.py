"""The candidate-selection rules, without a GPU.

``cleave.util.autotune`` decides which of several correct kernels an operator ships. The
timing itself needs hardware and is covered by ``tests/perf``; what is unit-testable is the
part that turns latencies into a decision, and that part is where a wrong answer is silent:
a ranking that quietly prefers the wrong candidate still prints a winner.
"""

from __future__ import annotations

import pytest

from cleave.util.autotune import (
    TIE_BAND,
    WorkloadCandidate,
    format_ranking,
    pick_best,
    rank,
    sampled,
    tie_band_members,
    workload_key,
)


def candidate(label, us, *, tuned=False, origin=None, checked=4, eligible=True):
    """A candidate with its timings already filled in, as the benchmark pass leaves it."""
    made = WorkloadCandidate(
        label=label,
        source=f"# {label}",
        origin=origin or ("searched == tuned" if tuned else "searched"),
        tuned=tuned,
        eligible=eligible,
    )
    made.latency_us = {f"wl_{i}": value for i, value in enumerate(us)}
    made.checked = checked
    return made


def test_rank_orders_by_geomean_not_by_any_single_workload():
    """A candidate can win a workload and still rank second; the aggregate decides."""
    wins_small = candidate("kv=32", [1.0, 100.0])
    wins_large = candidate("kv=128", [2.0, 10.0])
    assert [c.label for _, c in rank([wins_small, wins_large])] == ["kv=128", "kv=32"]


def test_a_clear_margin_beats_the_tuned_point():
    """The tuned schedule has no privilege outside the tie band -- that is the whole point."""
    faster = candidate("kv=128", [10.0])
    tuned = candidate("kv=64", [12.0], tuned=True)
    assert pick_best(rank([faster, tuned])) is faster


def test_a_reference_candidate_cannot_win_however_fast_it_is():
    """The headline is "best kernel found", so only kernels the search found may win.

    A hand-written schedule the search cannot reach is entered for comparison; selecting it
    would report a kernel the pipeline is unable to produce.
    """
    reference = candidate("kv=16", [1.0], tuned=True, eligible=False)
    searched = candidate("kv=32", [10.0])
    assert pick_best(rank([reference, searched])) is searched
    # ... and it still ranks first, because the report has to state the gap.
    assert [c.label for _, c in rank([reference, searched])] == ["kv=16", "kv=32"]


def test_an_all_reference_field_has_no_winner():
    """Better to refuse than to quietly promote a reference kernel to a result."""
    with pytest.raises(ValueError, match="No eligible candidate"):
        pick_best(rank([candidate("kv=16", [1.0], tuned=True, eligible=False)]))


def test_the_tie_band_no_longer_decides_the_winner():
    """It reports margins too small to defend; it does not pick between candidates."""
    fastest = candidate("kv=32", [10.0])
    close = candidate("kv=64", [10.0 * (1 + TIE_BAND / 2)], tuned=True)
    assert pick_best(rank([fastest, close])) is fastest
    assert [c.label for c in tie_band_members(rank([fastest, close]))] == ["kv=32", "kv=64"]


def test_an_untimed_candidate_ranks_last_rather_than_first():
    """An empty timing dict must not read as zero latency."""
    timed = candidate("kv=64", [10.0])
    untimed = candidate("kv=32", [])
    assert untimed.geomean_us == float("inf")
    assert pick_best(rank([untimed, timed])) is timed


def test_a_zero_timing_does_not_crash_the_geomean():
    """A zero reading is not a 0us kernel, it is a failed measurement."""
    assert candidate("kv=64", [0.0]).geomean_us == float("inf")


def test_pick_best_refuses_an_empty_ranking():
    with pytest.raises(ValueError, match="No correct candidates"):
        pick_best([])


def test_the_report_names_the_same_winner_the_benchmark_returns():
    """The printed winner and the returned one are one decision, not two."""
    fastest = candidate("kv=32", [10.0])
    reference = candidate("kv=64", [10.0 * (1 + TIE_BAND / 2)], tuned=True, eligible=False)
    report = format_ranking([fastest, reference])
    assert f"winner: {pick_best(rank([fastest, reference])).label}" in report
    assert "reported, not selected" in report
    assert "tuned point (reference only)" in report


def test_the_report_accounts_for_excluded_candidates():
    """A candidate that compiled and then failed must not vanish from the table."""
    good = candidate("kv=64", [10.0], tuned=True)
    bad = WorkloadCandidate(label="kv=256", source="# bad")
    bad.failure = "wl_1_1_1: InternalError: dynamic shared memory"
    report = format_ranking([good], [bad])
    assert "kv=256" in report and "EXCLUDED" in report
    assert "dynamic shared memory" in report


def test_the_report_says_when_the_winner_does_not_dominate():
    """The geomean hides split results, and the captured workloads split routinely."""
    small = candidate("kv=32", [1.0, 100.0])
    large = candidate("kv=128", [2.0, 10.0])
    assert "does NOT dominate" in format_ranking([small, large])
    alone = candidate("kv=64", [1.0, 2.0], tuned=True)
    assert "dominates all 2" in format_ranking([alone])


def test_sampled_is_stratified_and_stable():
    """A prefix would sample only the degenerate shapes; ``WORKLOADS`` is smallest-first."""
    assert sampled(range(48), 4) == [0, 16, 31, 47]
    assert sampled(range(48), 4) == sampled(range(48), 4)
    assert sampled(range(3), 0) == [0, 1, 2]
    assert sampled(range(3), 9) == [0, 1, 2]
    with pytest.raises(AssertionError, match="must not be negative"):
        sampled(range(48), -1)


def test_workload_keys_are_stable_names():
    """Baselines are keyed by these, so inserting a workload must not re-map the others."""
    assert workload_key((16, 1069, 1034)) == "wl_16_1069_1034"
    assert workload_key((1.0, 2.0)) == "wl_1_2"


def test_the_report_states_the_direction_of_the_gap_correctly():
    """A sign error here silently reverses the headline claim of the whole benchmark."""
    slow_winner = candidate("kv=32", [12.0])
    fast_reference = candidate("kv=16", [10.0], tuned=True, eligible=False)
    report = format_ranking([slow_winner, fast_reference])
    assert "the winner is 1.200x slower" in report, report

    fast_winner = candidate("kv=128", [10.0])
    slow_tuned = candidate("kv=64", [12.0], tuned=True)
    report = format_ranking([fast_winner, slow_tuned])
    assert "the winner is 1.200x faster" in report, report
