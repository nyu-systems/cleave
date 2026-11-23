"""Ranking of proven rules (DESIGN 8; 13.2 g) and the no-float64 guards (5.7; 13.2 h).

The grade is accuracy at the resolution of the output, so rounding noise cannot put an
unstable form first; only rules the assembly can emit are ranked; a value that is not
finite at ANY iteration rejects a combination, and so does a carried max that is not bit
for bit the flat graph's. CPU only (the MHA flash form is searched first).
"""

import dataclasses
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("z3")

from cleave.ir.update_rule import update_of, use_searched_updates
from cleave.pipeline.update import rank, search, store, texts
from cleave.pipeline.update.proof import prove
from cleave.pipeline.update.rank import Ranked, Replay, order, replay

from ._attention import FACTOR
from ._programs import MHA, attention, derivation, flash, mha_derivation, searched

E = ("exp", ("sub", "m", "m'"))
CANONICAL = update_of(FACTOR)
UNSTABLE = ("add", "r", ("div", "a", ("exp", ("sub", "m'", "m"))))
FP16_RHO = 2.0 ** -10 + 8 * 2.0 ** -23


def row(rules, errors, stable):
    return Ranked(rules, {}, {c: Replay(True, True, v) for c, v in errors.items()},
                  stable=stable, cost=2, size=8, text=str(sorted(rules.items())))


def test_one_ulp_of_the_output_does_not_put_an_unstable_form_first():
    """The judge's scenario: on a case where the unstable form is exact, the stable one is
    one float16 ulp off. At the output's resolution both are grade 0 and stability decides;
    at float32's resolution (the prototype's floor) the ulp would have won."""
    stable = row({"l": CANONICAL}, {"normal@32": 2.0 ** -10, "constant@32": 0.0}, True)
    unstable = row({"l": UNSTABLE}, {"normal@32": 0.0, "constant@32": 0.0}, False)
    ranked, rejected = order([unstable, stable], FP16_RHO)
    assert ranked == [stable, unstable] and not rejected
    assert (stable.grade, unstable.grade) == (0, 0)
    ranked, _ = order([unstable, stable], 1e-7)
    assert ranked[0] is unstable and stable.grade > 0


def test_a_grade_counts_quarter_decades_above_the_resolution():
    assert rank.grade(0.0, 0.0, FP16_RHO) == 0
    assert rank.grade(4 * FP16_RHO, 0.0, FP16_RHO) == 0  # within 4x of what the output resolves
    assert rank.grade(4.01 * FP16_RHO, 0.0, FP16_RHO) == 1
    assert rank.grade(17 * FP16_RHO, 0.0, FP16_RHO) == 2
    assert rank.grade(1e-2, 1e-2, FP16_RHO) == 0  # the best on its case


def test_search_only_rules_the_assembly_can_emit_are_ranked():
    d = mha_derivation()
    ratio = ("add", "r", ("div", ("mul", "a", ("exp", "m")), ("exp", "m'")))  # proved, no factor
    for acc in d.loops:
        assert prove(ratio, d.flat.program_graph, d.full, acc).proved
    ranked, rejected = rank.rank(d.kernel, d.loops, {a: [ratio, CANONICAL] for a in d.loops},
                                 [(4, "normal")])
    assert [r.rules for r in ranked + rejected] == [{a: CANONICAL for a in d.loops}]
    assert rank.rank(d.kernel, d.loops, {a: [ratio] for a in d.loops}, [(4, "normal")]) == ([], [])


# exp(m - m') + (1 / exp(3 m') - 1 / exp(3 m')), the ones written exp(m - m): the canonical
# factor plus a zero, proved and emittable. Its exp(3 m') overflows where the max passes
# ln(fp32 max) / 3 = 29.6, which the ``increasing`` case reaches only after its first tiles
# (at these small shapes), and is divided away there: every value of the loop stays finite.
GROWING = ("exp", ("add", ("add", "m'", "m'"), "m'"))
ONE_OVER = ("div", ("exp", ("sub", "m", "m")), GROWING)
OVERFLOWING = ("add", E, ("sub", ONE_OVER, ONE_OVER))


def test_search_a_value_that_is_not_finite_at_an_inner_iteration_only_rejects():
    d = mha_derivation()
    update = update_of(OVERFLOWING)
    for acc in d.loops:
        assert prove(update, d.flat.program_graph, d.full, acc).proved
    factors = {a: OVERFLOWING for a in d.loops}
    seen = replay(d.kernel, factors, 32, "increasing")
    from cleave.ir.update_rule import show

    tile = int(re.fullmatch(rf"non-finite {re.escape(show(GROWING))} at tile (\d+)",
                            seen.why).group(1))
    assert tile > 0  # finite at the first iteration ...
    assert not seen.finite and seen.same_max and np.isfinite(seen.error)  # ... and at the end
    canonical = replay(d.kernel, {a: FACTOR for a in d.loops}, 32, "increasing")
    assert canonical.why == "" and seen.error == canonical.error  # the same outputs
    ranked, rejected = rank.rank(d.kernel, d.loops, {a: [update] for a in d.loops},
                                 [(32, "normal"), (32, "increasing")])
    assert not ranked and [r.case for r in rejected] == ["increasing@32"]
    assert rejected[0].rejected == f"{seen.why} on increasing@32"


def one_ulp_up(a, b):
    m = torch.maximum(a, b)
    return torch.nextafter(m, torch.full_like(m, float("inf")))


def test_search_a_carried_max_one_ulp_off_rejects():
    d = mha_derivation()
    choices = {a: [CANONICAL] for a in d.loops}
    ranked, rejected = rank.rank(d.kernel, d.loops, choices, [(4, "normal")])
    assert len(ranked) == 1 and not rejected  # the control
    ranked, rejected = rank.rank(d.kernel, d.loops, choices, [(4, "normal")], combine=one_ulp_up)
    assert not ranked
    assert rejected[0].rejected == (f"the carried {d.full} is not bit-equal to the flat graph's "
                                    "on normal@4")
    assert rejected[0].replays["normal@4"].finite

    _, graphs = searched(attention, MHA, 11)
    with use_searched_updates():
        refused = derivation(flash(graphs), combine=one_ulp_up)
    assert refused.refused == texts.E12.format(
        loops=", ".join(d.loops),
        case=f"normal@32, where the carried {d.full} is not the flat graph's")


# ----------------------------------------------------------------------------- no float64


def test_every_search_level_is_float32():
    leaves = {"a": np.ones((3, 1), np.float32), "m": np.zeros((3, 1), np.float64)}
    with pytest.raises(AssertionError, match="float32"):
        search.enumerate_updates("x", leaves, {"a": False, "m": False},
                                 np.ones((3, 1), np.float32), budget=1)
    with pytest.raises(AssertionError, match="float32"):
        search.enumerate_updates("x", {"a": np.ones((3, 1), np.float32)}, {"a": False},
                                 np.ones((3, 1), np.float64), budget=1)


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
    """Every array and tensor reachable from the derivation (holes, references, ranking
    replays) and its instances' caches."""
    for d in (mha_derivation(),):
        found = []
        _walk(d, found, set())
        assert len(found) > 50
        assert not [k for k in found if k[1] in (np.float64, torch.float64)]
        assert {k[1] for k in found if k[0] == "array"} == {np.dtype(np.float32)}


def test_the_engine_never_names_float64():
    for path in store.engine_sources():
        text = path.read_text()
        assert "float64" not in text and ".double(" not in text, path.name
