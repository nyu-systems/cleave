"""A program split over the dim its max reduces runs unsplit, and says so (CPU).

With the max of ``cl.softmax`` in the flat graph the superoptimizer returns no valid split
graph over the softmax dim: the max is an opaque reduction it neither moves nor splits. So
``superoptimize`` runs such a search unsplit (on the budget and with the cache entry of the
unsplit search) and the ``GraphSet`` keeps the split that was asked for
(``state_split``). The split itself is to be built from the loop state of each solo graph;
until it is, ``schedule`` refuses every split factor of every solo graph with one reason,
and the program's kernels are its solo ones. A program without such a max (split-K of a
matmul) is split by the search as it always was.

``test_search_*`` run the superoptimizer (root conftest marks).
"""

import pytest

pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import LoweringRejected
from cleave.frontend import trace
from cleave.pipeline.plans import NO_STATE_SPLIT
from cleave.pipeline.superopt import max_split_dims


def attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
              v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


def matmul(a: cl.Tensor["M", "K"], b: cl.Tensor["K", "N"]):
    return a @ b


SHAPES = dict(H=4, SQ=16, SKV=256, D=16)


def test_the_dims_a_max_reduces():
    prog = trace(attention, SHAPES, cl.f16)
    assert max_split_dims(prog, ("SKV",)) == ("SKV",)
    assert max_split_dims(prog, cl.Split("SKV", (4, 8))) == ("SKV",)
    assert max_split_dims(prog, ("SQ",)) == () and max_split_dims(prog, ()) == ()
    assert max_split_dims(trace(matmul, dict(M=64, N=64, K=1024), cl.f16), ("K",)) == ()


def test_search_a_split_over_the_softmax_dim_is_searched_unsplit(tmp_path, monkeypatch):
    """The search runs as the unsplit one -- same budget, same cache entry -- and every
    graph is solo; the split that was asked for travels with them."""
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    prog = trace(attention, SHAPES, cl.f16)
    split = P.superoptimize(prog, P.SuperoptConfig(split=("SKV",)))
    assert split.state_split == ("SKV",) and len(split) == 2
    assert all(not g.is_split and not g.invalid_split and g.split == ("SKV",) for g in split)
    # the budget of the search without the split (the auto rule adds 2 for split ops)
    assert P.SuperoptConfig(split=("SKV",)).resolve_max_ops(prog) == 15
    assert split.max_ops == P.SuperoptConfig().resolve_max_ops(prog) == 13
    solo = P.superoptimize(prog, P.SuperoptConfig())
    assert solo.from_cache and solo.checkpoint == split.checkpoint and solo.state_split == ()
    assert [g.structure for g in solo] == [g.structure for g in split]
    assert split.filter(lambda g: True).state_split == ("SKV",)


def test_search_every_split_factor_of_every_solo_graph_is_refused_with_the_reason():
    """The plans are the solo graphs' own; per solo graph and factor one refusal, which the
    summary carries (a decode demo runs unsplit and its report says so)."""
    prog = trace(attention, SHAPES, cl.f16)
    graphs = P.superoptimize(prog, P.SuperoptConfig(split=cl.Split("SKV", (4, 8)), cache=False))
    plans = P.schedule(graphs, P.ScheduleConfig(topk=2, split_k=(4, 8)))
    assert plans and {(p.variant, p.role) for p in plans} == {("solo", "main")}
    split = [r for r in plans.refused if r.variant != "solo"]
    assert sorted((r.graph_index, r.variant, r.role) for r in split) == sorted(
        (g.index, f"split-k={f}", "partial") for g in graphs for f in (4, 8))
    for refusal in split:
        assert isinstance(refusal.error, LoweringRejected)
        assert str(refusal.error) == (f"graph {refusal.graph_index}: split over SKV: "
                                      f"{NO_STATE_SPLIT}")
    assert NO_STATE_SPLIT == "no partial/combine pair is built for a graph with a max over that dim"
    assert f"LoweringRejected: graph 0: split over SKV: {NO_STATE_SPLIT}" in plans.summary()


def test_search_a_program_without_such_a_max_is_split_by_the_search():
    prog = trace(matmul, dict(M=64, N=64, K=1024), cl.f16)
    graphs = P.superoptimize(prog, P.SuperoptConfig(split=("K",), cache=False))
    assert graphs.state_split == ()
    assert any(g.is_split for g in graphs)
    plans = P.schedule(graphs, P.ScheduleConfig(topk=2, split_k=(4,)))
    assert {p.variant for p in plans} >= {"split-k=4"}
    assert not [r for r in plans.refused if NO_STATE_SPLIT in str(r.error)]
