"""The bounded search itself: factors tested exactly in the field, every match collected,
the sample confirmed on every element, determinism. Synthetic field blocks; CPU only, no
superoptimizer.
"""

import numpy as np

from cleave.ir.update_rule import show
from cleave.pipeline.update import field, search
from cleave.pipeline.update.search import UNARY, enumerate_factors, evaluate


def blocks(factor, seed=0, cases=4, rows=5, width=1):
    """Random field leaves of ``cases`` cases and the target ``r + a * factor``."""
    rng = np.random.default_rng(seed)
    n = cases * rows
    leaves = {"a": rng.integers(0, field.PQ, (n, width)),
              "m": rng.integers(0, field.PQ, (n, 1)),
              "m'": rng.integers(0, field.PQ, (n, 1)),
              "r": rng.integers(0, field.PQ, (n, width))}
    target = (leaves["r"] + leaves["a"] * evaluate(factor, leaves)) % field.PQ
    return leaves, target


FACTOR = ("exp", ("sub", "m", "m'"))


def test_the_factor_is_found_and_every_match_is_collected_and_sorted():
    leaves, target = blocks(FACTOR, width=3)
    res = enumerate_factors("x", leaves, target, cases=4, budget=5)
    texts = [show(f) for f in res.matches]
    assert texts[0] == "exp((m - m'))"
    # exp(m) / exp(m') is the same field element: every factor that matches is listed
    assert "(exp(m) / exp(m'))" in texts
    assert res.matches == sorted(res.matches, key=lambda f: (search.size(f), show(f)))
    assert len(texts) == len(set(texts)) > 1
    assert res.visited == 2 + sum(res.formed) and len(res.kept) == len(res.formed) == 3


def test_a_match_on_the_sample_must_hold_on_every_element():
    """A factor that agrees on the sampled rows only is no match."""
    leaves, target = blocks(FACTOR, rows=8, width=2)
    rows, cols = search.sample(4, 8, 2)
    unsampled = sorted(set(range(32)) - set(rows.tolist()))
    target = target.copy()
    target[unsampled[0], 0] = (target[unsampled[0], 0] + 1) % field.PQ
    res = enumerate_factors("x", leaves, target, cases=4, budget=4)
    assert res.sampled > 0 and res.matches == []
    assert res.elements == (len(rows) * len(cols), 64)


def test_equality_is_exact():
    """One element off by one is a different field element: nothing matches."""
    leaves, target = blocks(FACTOR)
    target = target.copy()
    target[0, 0] = (target[0, 0] + 1) % field.PQ
    assert enumerate_factors("x", leaves, target, cases=4, budget=5).matches == []


def test_the_unary_ops_are_those_asked_for():
    leaves, target = blocks(("sqrt", ("div", "m", "m'")))
    with_sqrt = enumerate_factors("x", leaves, target, cases=4, budget=4, unary=("sqrt",))
    assert [show(f) for f in with_sqrt.matches][0] == "sqrt((m / m'))"
    assert enumerate_factors("x", leaves, target, cases=4, budget=4, unary=("exp",)).matches == []
    assert set(UNARY) == {"exp", "sqrt", "rsqrt", "silu"}


def test_the_search_is_deterministic():
    leaves, target = blocks(FACTOR, seed=3, width=2)
    one = enumerate_factors("x", leaves, target, cases=4, budget=5)
    two = enumerate_factors("x", leaves, target, cases=4, budget=5)
    assert (one.kept, one.formed, one.matches) == (two.kept, two.formed, two.matches)
