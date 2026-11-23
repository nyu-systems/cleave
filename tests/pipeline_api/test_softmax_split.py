"""A program split over the dim its max reduces: the pair is built from the loop state (CPU).

With the max of ``cl.softmax`` in the flat graph the superoptimizer returns no valid split
graph over the softmax dim: the max is an opaque reduction it neither moves nor splits. So
``superoptimize`` runs such a search unsplit (on the budget and with the cache entry of the
unsplit search) and the ``GraphSet`` keeps the split that was asked for (``state_split``).
``schedule`` builds the split of each solo graph from its loop state instead
(``pipeline.plans.state_pair``): a partial that returns the accumulators and the max of
its own split, a combine that applies the proven repair and runs the solo graph's
epilogue. Where no pair is built the refusal says why, in its own words:

  P1  this graph carries no proven update over that dim[: {its streaming refusals}]
  P2  {op_type} on the way to the loop state reads no split input
  P3  no symbolic split factor for the looped dim
  P4  {acc} {dims} is not the full-extent value {rows} with at most one trailing axis
  P5  the dim has {n} elements, which {f} does not divide

and a few more that name what the construction does not take: a combine that would read
the program's inputs, two carried maxes over the dim, a split over two such dims, an
un-searched graph. No part of the split is dropped silently: a split without a factor is
refused as such, a dim of it that no max reduces is named (the search ran unsplit). A
program without such a max (split-K of a matmul) is split by the search as it always was.

``test_search_*`` run the superoptimizer (root conftest marks).
"""

import functools
import json

import pytest

pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import LoweringRejected, SpecError
from cleave.frontend import trace
from cleave.ir.graph_utils import boundary_signature
from cleave.lower.state_split import SplitRefused, combine_graph
from cleave.pipeline.plans import (
    NO_PROVEN_UPDATE,
    NO_SPLIT_DIM,
    NO_SPLIT_FACTOR,
    SEVERAL_DIMS,
    TWO_STATES,
    UNSEARCHED,
    bind_inputs,
    lower,
)
from cleave.pipeline.superopt import GraphSet, SearchedGraph, max_split_dims
from cleave.runtime.reference import output_order, output_roles, placeholder_order

FACTOR = ("exp", ("sub", "m", "m'"))
T3 = ("reduce_max_2_5: mma_10 reads reduction_2_8, a finished reduction over the dim "
      "reduce_max_2_5 reduces: that needs two passes over the row")


def attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
              v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


def exp_of_queries(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                   v: cl.Tensor["H", "SKV", "D"]):
    """An op on the queries alone ahead of the scores: no split of the keys reaches it."""
    return cl.softmax((cl.exp(q) @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


def plus_queries(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                 v: cl.Tensor["H", "SKV", "D"]):
    """Behind the loop state the output reads an input again."""
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v + q


def centered(x: cl.Tensor["R", "C"]):
    """A max over the split dim with nothing that loops under it: no obstacle at all."""
    return x - cl.max(x, -1, keepdim=True)


def two_softmaxes(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    return cl.softmax(x, -1) @ v + cl.softmax(y, -1) @ v


def dual(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1) * cl.softmax(x, 0)


def matmul(a: cl.Tensor["M", "K"], b: cl.Tensor["K", "N"]):
    return a @ b


SHAPES = dict(H=4, SQ=16, SKV=256, D=16)


@functools.lru_cache(maxsize=None)
def _graphs(fn=attention, shapes=tuple(SHAPES.items()), split=("SKV",)):
    prog = trace(fn, dict(shapes), cl.f16)
    return P.superoptimize(prog, P.SuperoptConfig(split=split, cache=False))


def _flash(graphs):
    (graph,) = [g for g in graphs if g.summary().endswith("sum>matmul>div")]
    return graph


def _traced(graphs):
    (graph,) = [g for g in graphs if g.summary().endswith("sum>div>matmul")]
    return graph


def _refused(graph, factor):
    with pytest.raises(LoweringRejected) as info:
        lower(graph, factor)
    return str(info.value)


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


def test_search_the_flash_graph_lowers_to_a_pair_built_from_its_loop_state():
    """``lower(solo graph, f)``: the partial returns each split's accumulators and max
    (float32, the max last, role "shift"), bound to the program's inputs; the combine
    reads exactly those tensors and returns the program's output. Each stage carries the
    update rule derived for it, so either may loop over the dim it reduces."""
    graphs = _graphs()
    graph = _flash(graphs)
    for factor in (4, 8):
        partial, combine = lower(graph, factor)
        crossing = boundary_signature(output_order(partial))
        assert crossing == boundary_signature(placeholder_order(combine)) == (
            (0, "", (4, factor, 16, 16), "float32"),
            (1, "", (4, factor, 16), "float32"),
            (2, "shift", (4, factor, 16), "float32"))
        assert output_roles(partial) == ("shift", "", "") and output_roles(combine) == ("",)
        assert bind_inputs(partial, graphs.program) is not None
        for stage in (partial, combine):
            assert stage.streaming_refusals == []
            (record,) = [n.streaming for n in stage.nodes
                         if getattr(n, "streaming", None) is not None]
            assert dict(record.rules) == {acc: FACTOR for acc in record.loops}
        (merged,) = [n for n in combine.nodes if n.name == "state_max"]
        assert merged.streaming is not None  # the combine streams over the split axis


def test_search_every_factor_of_the_flash_graph_has_partial_and_combine_plans():
    """``schedule``: per factor the flash graph's partial and combine plans; the traced
    graph has none, and says why (P1 with its T3 reason) once per factor."""
    graphs = _graphs(split=cl.Split("SKV", (4, 8)))
    plans = P.schedule(graphs, P.ScheduleConfig(topk=2, split_k=(4, 8)))
    flash, traced = _flash(graphs), _traced(graphs)
    assert {(p.graph_index, p.variant, p.role) for p in plans} == {
        (flash.index, "solo", "main"), (traced.index, "solo", "main")} | {
        (flash.index, f"split-k={f}", role) for f in (4, 8) for role in ("partial", "combine")}
    for plan in plans:
        assert plan.variant == "solo" or plan.graph is flash
        assert (plan.inputs is None) == (plan.role == "combine")
        assert plan.split_factor == (None if plan.variant == "solo" else int(plan.variant[8:]))
    split = [r for r in plans.refused if r.variant != "solo"]
    assert sorted((r.graph_index, r.variant, r.role) for r in split) == [
        (traced.index, f"split-k={f}", "partial") for f in (4, 8)]
    why = f"graph {traced.index}: split over SKV: {NO_PROVEN_UPDATE}: {T3}"
    assert NO_PROVEN_UPDATE == "this graph carries no proven update over that dim"
    for refusal in split:
        assert isinstance(refusal.error, LoweringRejected) and str(refusal.error) == why
    assert "LoweringRejected: graph" in plans.summary()


def test_a_split_dim_that_is_no_input_dim_is_refused_with_the_rest_of_the_split():
    """The whole split is checked, not only the part the search would run: a dim that is
    no input dim is a SpecError next to the softmax dim as well as alone."""
    prog = trace(attention, SHAPES, cl.f16)
    for split in (("BOGUS",), ("SKV", "BOGUS"), cl.Split("BOGUS", (4,))):
        with pytest.raises(SpecError, match=r"split dims \['BOGUS'\] are not dims of any input"):
            P.superoptimize(prog, P.SuperoptConfig(split=split, cache=False))


def test_search_no_part_of_the_split_is_dropped_without_a_reason():
    """Without a factor the split is refused once per solo graph (as a split graph is);
    a dim of the split that no max reduces is not split (the search ran unsplit), and one
    refusal per solo graph names it, next to the pairs over the softmax dim."""
    graphs = _graphs(split=("SKV", "SQ"))
    assert graphs.state_split == ("SKV", "SQ") and all(not g.is_split for g in graphs)
    flash, traced = _flash(graphs), _traced(graphs)
    unsplit = "the search ran unsplit, so SQ is not split"
    plans = P.schedule(graphs, P.ScheduleConfig(topk=2))
    assert plans and {(p.variant, p.role) for p in plans} == {("solo", "main")}
    split = sorted((r.graph_index, r.variant, str(r.error)) for r in plans.refused
                   if r.variant != "solo")
    assert split == sorted(
        [(g.index, "split", f"graph {g.index}: split over SKV: {unsplit}") for g in graphs]
        + [(g.index, "split", f"graph {g.index}: split over SKV: ScheduleConfig.split_k is "
                              "empty") for g in graphs])
    plans = P.schedule(graphs, P.ScheduleConfig(topk=2, split_k=(4,)))
    assert {(p.graph_index, p.variant, p.role) for p in plans if p.variant != "solo"} == {
        (flash.index, "split-k=4", "partial"), (flash.index, "split-k=4", "combine")}
    split = sorted((r.graph_index, r.variant, str(r.error)) for r in plans.refused
                   if r.variant != "solo")
    assert split == sorted(
        [(g.index, "split", f"graph {g.index}: split over SKV: {unsplit}") for g in graphs]
        + [(traced.index, "split-k=4",
            f"graph {traced.index}: split over SKV: {NO_PROVEN_UPDATE}: {T3}")])


def test_search_a_program_without_such_a_max_is_split_by_the_search():
    prog = trace(matmul, dict(M=64, N=64, K=1024), cl.f16)
    graphs = P.superoptimize(prog, P.SuperoptConfig(split=("K",), cache=False))
    assert graphs.state_split == ()
    assert any(g.is_split for g in graphs)
    plans = P.schedule(graphs, P.ScheduleConfig(topk=2, split_k=(4,)))
    assert {p.variant for p in plans} >= {"split-k=4"}
    assert not [r for r in plans.refused if "split over" in str(r.error)]


# ------------------------------------------------------------------- P1 .. P5, and the rest


def test_search_p1_a_graph_whose_max_carries_nothing_over_the_dim():
    """A max over the split dim that nothing loops under: no obstacle, so no refusal of
    one to quote."""
    (graph,) = _graphs(centered, (("R", 64), ("C", 256)), ("C",))
    assert _refused(graph, 4) == f"graph 0: split over C: {NO_PROVEN_UPDATE}"


def test_search_p1_the_traced_graph_says_why_its_max_is_not_carried():
    assert _refused(_traced(_graphs()), 4) == (
        f"graph {_traced(_graphs()).index}: split over SKV: {NO_PROVEN_UPDATE}: {T3}")


def test_search_p2_an_op_on_the_way_to_the_state_that_no_split_reaches():
    graph = _flash(_graphs(exp_of_queries))
    assert _refused(graph, 4) == (f"graph {graph.index}: split over SKV: kn_exp_op on the way "
                                  "to the loop state reads no split input")


def test_search_p3_a_looped_dim_no_factor_can_stand_for():
    """The partial splits the looped proxy symbol of the structure into a symbolic factor
    of 2, 3, 5 or 7 and the rest; a looped proxy that is a larger prime has none."""
    graph = _flash(_graphs())
    inputs = [op for op in graph.structure if op["op_type"] == "kn_input_op"]
    looped = inputs[1]["output_tensors"][0]["dim"][1]  # the keys of K
    tensors = [t for op in graph.structure for t in op["input_tensors"] + op["output_tensors"]]
    prime = next(p for p in (11, 13, 17, 19, 23) if p not in {d for t in tensors for d in t["dim"]})
    structure = json.loads(json.dumps(graph.structure))
    for op in structure:
        for t in op["input_tensors"] + op["output_tensors"]:
            t["dim"] = [prime if d == looped else d for d in t["dim"]]
    symbols = {(prime if s == looped else s): e for s, e in graph.symbol_map.items()}
    renamed = SearchedGraph(graph.index, structure, symbols, False, graph.program, graph.split)
    assert _refused(renamed, 4) == (f"graph {graph.index}: split over SKV: no symbolic split "
                                    "factor for the looped dim")


def test_search_p4_an_accumulator_that_is_not_the_rows_with_one_trailing_axis():
    """The combine repairs each accumulator by one weight per row: a state tensor that is
    not the max's rows with at most one more axis has none (here the keys, laid out
    (H, SKV, D) next to rows (H, SQ))."""
    graph = _flash(_graphs())
    (ir,) = lower(graph)
    (full,) = [n for n in ir.nodes if getattr(n, "streaming", None) is not None]
    keys = next(n for n in ir.nodes if n.is_placeholder() and n.input_index == 1)
    with pytest.raises(SplitRefused) as info:
        combine_graph(ir, full.name, ((keys.name, FACTOR),), 4, split_pos=1,
                      out_dtype=graph.dtype)
    assert str(info.value) == (f"{keys.name} (4, 4, 256, 16) is not the full-extent value "
                               "(4, 4, 16) with at most one trailing axis")


def test_search_p5_a_factor_that_does_not_divide_the_dim():
    graphs = _graphs()
    plans = P.schedule(graphs, P.ScheduleConfig(topk=2, split_k=(3,)))
    why = f"split over SKV: {NO_SPLIT_FACTOR.format(n=256, f=3)}"
    assert why == "split over SKV: the dim has 256 elements, which 3 does not divide"
    assert {(r.graph_index, r.variant, str(r.error)) for r in plans.refused
            if r.variant != "solo"} == {(g.index, "split-k=3", f"graph {g.index}: {why}")
                                        for g in graphs}


def test_search_a_combine_that_would_read_the_programs_inputs():
    (graph,) = [g for g in _graphs(plus_queries) if g.summary().endswith("sum>matmul>div>add")]
    assert _refused(graph, 4) == (
        f"graph {graph.index}: split over SKV: add_11, behind the loop state, read(s) "
        "PlaceHolder_input_10000001, an input of the program; the combine reads only what "
        "the partials return")


def test_a_split_over_no_dim_a_max_reduces_or_over_two():
    """``lower(solo graph, f)`` splits over the one dim of ``SearchedGraph.split`` that a max
    reduces: none, or two at once, is refused before anything is built."""
    prog = trace(attention, SHAPES, cl.f16)
    (graph,) = GraphSet.identity(prog)
    assert _refused(graph, 4) == f"graph 0: split over no dim: {NO_SPLIT_DIM}"
    graph.split = ("SQ",)
    assert _refused(graph, 4) == f"graph 0: split over SQ: {NO_SPLIT_DIM}"
    (twice,) = GraphSet.identity(trace(dual, dict(R=16, C=32), cl.f16))
    twice.split = ("R", "C")
    assert _refused(twice, 4) == f"graph 0: split over R,C: {SEVERAL_DIMS}"


def test_an_unsearched_graph_is_not_split():
    (graph,) = GraphSet.identity(trace(attention, SHAPES, cl.f16))
    graph.split = ("SKV",)
    assert _refused(graph, 4) == f"graph 0: split over SKV: {UNSEARCHED}"
    assert _refused(graph, 3) == f"graph 0: split over SKV: {NO_SPLIT_FACTOR.format(n=256, f=3)}"


def test_search_two_carried_maxes_over_the_dim():
    """The pair carries one loop state: the graph that serves both softmaxes of
    ``softmax(x) @ v + softmax(y) @ v`` over the split dim is refused; one that serves one
    of them reads the other half from the program's inputs behind the state. (About a
    minute of search: 18 ops.)"""
    graphs = _graphs(two_softmaxes, (("R", 64), ("C", 256), ("D", 64)), ("C",))
    (both,) = [g for g in graphs if g.summary().startswith(
        "solo max>max>sub>sub>exp>exp>sum>matmul>sum>matmul")]
    assert _refused(both, 4) == (f"graph {both.index}: split over C: " + TWO_STATES.format(
        n=2, names="reduce_max_1_3, reduce_max_1_4"))
    assert TWO_STATES.format(n=2, names="a, b") == (
        "2 maxes are carried across a loop over that dim (a, b); a pair is built from one")
    reasons = {_refused(g, 4).split(": ", 2)[2] for g in graphs if g is not both}
    assert any("an input of the program; the combine reads only" in r for r in reasons)
    assert any(r.startswith(NO_PROVEN_UPDATE) for r in reasons)
