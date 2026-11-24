"""The bounded search itself: exact merging by value, finiteness, every match
collected, determinism. Synthetic float32 blocks; CPU only, no superoptimizer.
"""

import numpy as np
import pytest

from cleave.ir.update_rule import show
from cleave.pipeline.update import search
from cleave.pipeline.update.search import _hash, _Seen, enumerate_updates


def blocks(seed=0, rows=6, width=1):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(rows, 1)).astype(np.float32)
    m1 = (m + np.abs(rng.normal(size=(rows, 1)))).astype(np.float32)
    a = rng.uniform(0.5, 2.0, size=(rows, width)).astype(np.float32)
    r = rng.uniform(0.5, 2.0, size=(rows, width)).astype(np.float32)
    target = (a * np.exp(m - m1) + r).astype(np.float32)
    leaves = {"a": a, "m": m, "m'": m1, "r": r}
    wide = {"a": width > 1, "m": False, "m'": False, "r": width > 1}
    return leaves, wide, target


def colliding(base: np.ndarray) -> np.ndarray:
    """Another float32 vector with the same fingerprint: bits moved by (0, +1, -2, +1, 0..)
    ulps keep both sums the fingerprint is made of (it is linear in the bits)."""
    bits = base.view(np.uint32).astype(np.int64)
    bits[1:4] += (1, -2, 1)
    return bits.astype(np.uint32).view(np.float32)


def test_a_fingerprint_is_shared_by_distinct_values_and_both_are_kept():
    base = np.linspace(1.0, 2.0, 6, dtype=np.float32)
    other = colliding(base)
    assert not np.array_equal(base, other)
    (h1,), (h2,) = _hash(base[None]).tolist(), _hash(other[None]).tolist()
    assert h1 == h2
    seen = _Seen()
    seen.add(h1, base.view(np.uint32).copy())
    assert seen.known(h1, base.view(np.uint32)) and not seen.known(h2, other.view(np.uint32))
    seen.add(h2, other.view(np.uint32).copy())
    assert seen.collisions == 1 and seen.known(h2, other.view(np.uint32))


@pytest.mark.parametrize("width", [1, 3])
def test_merging_by_value_does_not_depend_on_the_fingerprint(width, monkeypatch):
    """With every fingerprint equal the search keeps, forms and matches exactly what it
    does with the real one: equality is decided on the bits."""
    leaves, wide, target = blocks(width=width)
    real = enumerate_updates("x", leaves, wide, target, budget=3)
    monkeypatch.setattr(search, "_hash", lambda block: np.zeros(block.shape[0], np.uint64))
    same = enumerate_updates("x", leaves, wide, target, budget=3)
    assert (same.kept, same.formed, same.matches) == (real.kept, real.formed, real.matches)
    assert same.collisions > real.collisions


def test_every_match_is_collected_and_sorted():
    leaves, wide, target = blocks()
    res = enumerate_updates("x", leaves, wide, target, budget=4)
    texts = [show(e) for e, _ in res.matches]
    assert "(r + (a * exp((m - m'))))" in texts
    assert res.matches == sorted(res.matches, key=lambda m: (search.size(m[0]), show(m[0])))
    assert len(texts) == len(set(texts)) > 1  # not only the first
    assert all(err <= search.LOOSE_RTOL for _, err in res.matches)
    assert res.visited == sum(res.formed) and len(res.kept) == len(res.formed) == 4


def test_a_value_that_is_not_finite_is_neither_kept_nor_a_match():
    """At the first iteration the old max is the reducer's start value: ``exp(m' - m)``
    overflows there, so ``r + a / exp(m' - m)`` is never formed."""
    leaves, wide, target = blocks()
    leaves = dict(leaves)
    leaves["m"] = leaves["m"].copy()
    leaves["m"][0] = np.float32(-3.4028235e38)  # one first-iteration row
    leaves["a"] = leaves["a"].copy()
    leaves["a"][0] = 0.0
    target = (leaves["a"] * np.exp(leaves["m"] - leaves["m'"]) + leaves["r"]).astype(np.float32)
    res = enumerate_updates("x", leaves, wide, target, budget=4)
    texts = [show(e) for e, _ in res.matches]
    assert "(r + (a * exp((m - m'))))" in texts
    assert "(r + (a / exp((m' - m))))" not in texts
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.exp(leaves["m'"] - leaves["m"])).all()


def test_the_search_is_deterministic():
    leaves, wide, target = blocks(seed=3, width=2)
    one = enumerate_updates("x", leaves, wide, target, budget=3)
    two = enumerate_updates("x", leaves, wide, target, budget=3)
    assert (one.kept, one.formed, one.matches) == (two.kept, two.formed, two.matches)
