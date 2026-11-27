"""Ranking of the rules that reproduce the flat graph, and the no-float64 guards.

The test is exact in a finite field, where ``exp(m - m')`` and ``exp(m) / exp(m')`` are the
same element; the float replay tells them apart. The grade is accuracy at the resolution of
the output, then the op count decides; only rules the assembly can emit are ranked; a value
that is not finite at ANY iteration rejects a combination, and so does a carried value that
is not the flat graph's within float32 rounding. CPU only (the MHA flash form is searched
first).
"""

import dataclasses
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cleave.ir.update_rule import show, update_of
from cleave.pipeline.update import rank, store, texts
from cleave.pipeline.update.rank import Ranked, Replay, order, replay

from ._attention import FACTOR
from ._programs import MHA, attention, derivation, flash, mha_derivation, searched

E = ("exp", ("sub", "m", "m'"))
CANONICAL = update_of(FACTOR)
QUOTIENT = update_of(("div", ("exp", "m"), ("exp", "m'")))  # the same field element
FP16_RHO = 2.0 ** -10 + 8 * 2.0 ** -23


def row(rules, errors, cost):
    return Ranked(rules, {}, {c: Replay(True, True, v) for c, v in errors.items()},
                  cost=cost, size=8, text=str(sorted(rules.items())))


def test_accuracy_first_then_the_op_count():
    """Within a grade the cheaper rule wins; a rule a grade more accurate wins whatever it
    costs."""
    cheap = row({"l": CANONICAL}, {"normal@32": 2.0 ** -10}, cost=2)
    dear = row({"l": QUOTIENT}, {"normal@32": 0.0}, cost=3)
    ranked, rejected = order([dear, cheap], FP16_RHO)
    assert ranked == [cheap, dear] and not rejected and (cheap.grade, dear.grade) == (0, 0)
    ranked, _ = order([dear, cheap], 1e-7)  # at float32's resolution the ulp counts
    assert ranked[0] is dear and cheap.grade > 0


def test_a_grade_counts_quarter_decades_above_the_resolution():
    assert rank.grade(0.0, 0.0, FP16_RHO) == 0
    assert rank.grade(4 * FP16_RHO, 0.0, FP16_RHO) == 0  # within 4x of what the output resolves
    assert rank.grade(4.01 * FP16_RHO, 0.0, FP16_RHO) == 1
    assert rank.grade(17 * FP16_RHO, 0.0, FP16_RHO) == 2
    assert rank.grade(1e-2, 1e-2, FP16_RHO) == 0  # the best on its case


def test_search_only_rules_the_assembly_can_emit_are_ranked():
    d = mha_derivation()
    ratio = ("add", "r", ("div", ("mul", "a", ("exp", "m")), ("exp", "m'")))  # no factor
    ranked, rejected = rank.rank(d.kernel, d.loops, {a: [ratio, CANONICAL] for a in d.loops},
                                 [(4, "normal")])
    assert [r.rules for r in ranked + rejected] == [{a: CANONICAL for a in d.loops}]
    assert rank.rank(d.kernel, d.loops, {a: [ratio] for a in d.loops}, [(4, "normal")]) == ([], [])


def test_search_the_replay_puts_the_stable_form_first():
    """``exp(m) / exp(m')`` reproduces the flat graph exactly in the field, as
    ``exp(m - m')`` does; in floating point its exp overflows on large logits, so the replay
    rejects it, and on the normal cases it costs one op more."""
    d = mha_derivation()
    ranked, rejected = rank.rank(d.kernel, d.loops, {a: [QUOTIENT, CANONICAL] for a in d.loops},
                                 [(32, "normal"), (32, "large_logits")])
    assert ranked[0].rules == {a: CANONICAL for a in d.loops}
    assert any("non-finite" in r.rejected for r in rejected)


# exp(m - m') + (1 / exp(3 m') - 1 / exp(3 m')), the ones written exp(m - m): the canonical
# factor plus a zero, and emittable. Its exp(3 m') overflows where the max passes
# ln(fp32 max) / 3 = 29.6, which the ``increasing`` case reaches only after its first tiles,
# and is divided away there: every value of the loop stays finite.
GROWING = ("exp", ("add", ("add", "m'", "m'"), "m'"))
ONE_OVER = ("div", ("exp", ("sub", "m", "m")), GROWING)
OVERFLOWING = ("add", E, ("sub", ONE_OVER, ONE_OVER))


def test_search_a_value_that_is_not_finite_at_an_inner_iteration_only_rejects():
    d = mha_derivation()
    update = update_of(OVERFLOWING)
    factors = {a: OVERFLOWING for a in d.loops}
    seen = replay(d.kernel, factors, 32, "increasing")
    tile = int(re.fullmatch(rf"non-finite {re.escape(show(GROWING))} at tile (\d+)",
                            seen.why).group(1))
    assert tile > 0  # finite at the first iteration ...
    assert not seen.finite and seen.same_value and np.isfinite(seen.error)  # ... and at the end
    canonical = replay(d.kernel, {a: FACTOR for a in d.loops}, 32, "increasing")
    assert canonical.why == "" and seen.error == canonical.error  # the same outputs
    ranked, rejected = rank.rank(d.kernel, d.loops, {a: [update] for a in d.loops},
                                 [(32, "normal"), (32, "increasing")])
    assert not ranked and [r.case for r in rejected] == ["increasing@32"]
    assert rejected[0].rejected == f"{seen.why} on increasing@32"


def _off(fold, by):
    def combine(a, b):
        return fold(a, b) * (1.0 + by) + by
    return combine


def test_search_a_carried_value_is_held_to_float32_rounding():
    """The carried value at the end is the flat graph's within the rounding of float32 in
    another order and another tiling (``reference.folds_to``): one ulp per tile is within
    it, for a max as for a sum; a fold off by more is rejected."""
    from cleave.ir.ops import ReduceSum

    from ._programs import identity, sum_shifted

    s = derivation(identity(sum_shifted, dict(R=16, C=64, D=8)))
    assert not s.refused and type(s.kernel.node(s.kernel.tile, s.full)) is ReduceSum
    m = mha_derivation()

    def ulp(fold):
        def combine(a, b):
            v = fold(a, b)
            return torch.nextafter(v, torch.full_like(v, float("inf")))
        return combine

    for d, fold in ((s, torch.add), (m, torch.maximum)):
        choices = {a: [CANONICAL] for a in d.loops}
        for combine in (None, ulp(fold)):
            ranked, rejected = rank.rank(d.kernel, d.loops, choices, [(32, "normal")],
                                         combine=combine)
            assert len(ranked) == 1 and not rejected
        ranked, rejected = rank.rank(d.kernel, d.loops, choices, [(32, "normal")],
                                     combine=_off(fold, 1e-2))
        assert not ranked
        assert rejected[0].rejected == f"the carried {d.full} is not the flat graph's on normal@32"
        assert rejected[0].replays["normal@32"].finite


def test_search_no_rule_survives_a_wrong_carry():
    _, graphs = searched(attention, MHA, 11)
    d = mha_derivation()
    refused = derivation(flash(graphs), combine=_off(torch.maximum, 1e-2))
    assert refused.refused == texts.NO_REPLAY.format(
        loops=", ".join(d.loops),
        case=f"the carried {d.full} is not the flat graph's on normal@32")


# ----------------------------------------------------------------------------- no float64


def _walk(x, found, seen):
    if id(x) in seen:
        return
    seen.add(id(x))
    if isinstance(x, np.ndarray):
        found.append(("array", x.dtype))
    elif isinstance(x, torch.Tensor):
        found.append(("tensor", x.dtype))
    elif isinstance(x, dict):
        for k, v in x.items():
            _walk(k, found, seen)
            _walk(v, found, seen)
    elif isinstance(x, (list, tuple, set, frozenset)):
        for v in x:
            _walk(v, found, seen)
    elif dataclasses.is_dataclass(x) and not isinstance(x, type):
        for f in dataclasses.fields(x):
            _walk(getattr(x, f.name), found, seen)


def test_search_nothing_a_derivation_holds_is_float64():
    """Every array and tensor reachable from the derivation (test data, holes, ranking
    replays) and its instances' caches: field elements (integers) and the program's own
    float dtypes, never float64."""
    found = []
    _walk(mha_derivation(), found, set())
    assert len(found) > 20
    assert not [k for k in found if k[1] in (np.float64, torch.float64)]
    assert {k[1] for k in found if k[0] == "tensor"} >= {torch.int64}


def test_the_engine_never_names_float64():
    for path in store.engine_sources():
        text = path.read_text()
        assert "float64" not in text and ".double(" not in text, path.name


_FLOAT64 = re.compile(r"\.double\(|\b(?:torch|np|numpy)\.(?:float64|double)\b")


def test_no_test_reference_or_library_computes_in_float64():
    """No float64 anywhere: references, search evaluation, ranking and tests use the
    program's own dtypes / float32, and the field's integers. A test that widened its error
    arithmetic to float64 (tests/schedule/end2end/test_absorbed_reduction.py did) is caught
    here."""
    root = store.engine_sources()[0].parents[3]
    offenders = []
    for top in ("python", "tests", "demo"):
        for path in sorted((root / top).rglob("*.py")):
            name = path.relative_to(root).as_posix()
            if name == "tests/update/test_ranking.py":
                continue
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if _FLOAT64.search(line):
                    offenders.append(f"{name}:{n}: {line.strip()}")
    assert offenders == []
    assert _FLOAT64.search("ref.double()") and _FLOAT64.search("x.to(torch.float64)")
    assert not _FLOAT64.search("csopt.float64") and not _FLOAT64.search('"float64": "f"')
