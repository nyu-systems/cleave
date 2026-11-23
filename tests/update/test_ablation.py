"""Why the reference set is what it is (DESIGN 5.4, 5.5; 13.2 c), on MHA prefill's flash form.

On reference rows whose max did not move, the update "no repair" ``a + r`` is exact, so the
loose filter passes it; the proof rejects it, and the adequacy check refuses such a set
(E10) before anything is searched. The same holds for the first iteration alone, where the
old state is zero. CPU only (the superoptimizer runs first).
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("z3")

from cleave.pipeline.update import texts
from cleave.pipeline.update.engine import hole_blocks
from cleave.pipeline.update.proof import OBLIGATIONS, prove
from cleave.pipeline.update.reference import Refused, check_adequacy, discriminating
from cleave.pipeline.update.search import enumerate_updates

from ._attention import FACTOR
from ._programs import mha_derivation

NO_REPAIR = ("add", "a", "r")
RIGHT = ("add", "r", ("mul", "a", FACTOR))


def weak(d, which):
    """The cases of ``d``'s reference set that a weaker set would consist of."""
    full = d.full
    if which == "still":  # the normal prefixes, only the rows whose max did not move
        cases = [c.only(c.new[full].reshape(-1) == c.old[full].reshape(-1))
                 for c in d.references if c.label.startswith("normal")]
        return [c for c in cases if c.rows]
    return [c for c in d.references if c.prefix == 0]  # the first iterations alone


@pytest.mark.parametrize("which", ["still", "first"])
def test_search_a_weak_reference_set_lets_no_repair_through_and_is_refused(which):
    d = mha_derivation()
    cases = weak(d, which)
    assert cases and sum(c.rows for c in cases) >= d.references[0].rows
    for hole in d.holes:
        acc = hole.acc
        leaves, kinds, target, _ = hole_blocks(cases, d.full, acc)
        found = enumerate_updates(acc, leaves, kinds, target, budget=4, unary=d.unary)
        matches = [e for e, _ in found.matches]
        assert NO_REPAIR in matches and matches[0] == NO_REPAIR  # the smallest of them
        assert len(matches) > 50  # on the designed set there is one
        verdict = prove(NO_REPAIR, d.flat.program_graph, d.full, acc)
        assert not verdict.proved and verdict.results[OBLIGATIONS[2]] == "sat"
        assert not any(bool(discriminating(c, acc).any()) for c in cases)
    rows = d.references[0].rows
    with pytest.raises(Refused) as refusal:
        check_adequacy(cases, d.full, d.loops, rows)
    assert str(refusal.value) == texts.E10.format(full=d.full, acc=d.loops[0], k=0, n=rows)


def test_search_the_designed_set_passes_one_update_per_hole_and_is_adequate():
    d = mha_derivation()
    rows = d.references[0].rows
    assert check_adequacy(d.references, d.full, d.loops, rows) == d.adequacy
    assert all(n >= rows for n in d.adequacy.values())
    for hole in d.holes:
        assert [e for e, _ in hole.search.matches] == [RIGHT]
