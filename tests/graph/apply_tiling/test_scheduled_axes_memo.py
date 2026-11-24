"""``apply_tiling`` reads a flat graph's axis provenance once, not once per candidate (CPU).

The absorbed-reduction pin (``tiling._pin_absorbed_reduction_axes``) decides which tensor
dims an absorbed reduction holds whole from the axis provenance of the flat graph. That
provenance does not depend on the candidate, and every candidate of a search shares one
flat graph, but it was rebuilt for every candidate with an absorbed reduction: 1.93 of the
2.83 ms ``apply_tiling`` took per candidate of MLA decode, and so most of what the softmax
accept filter costs.
"""

import gc

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.core.errors import ScheduleRejected
from cleave.ir.analysis import build_output_axis_provenance, build_reduction_axis_provenance
from cleave.schedule import apply_tiling, search_tiling
from cleave.schedule import tiling

from tests import _graphs as g

# name -> (builder, dims, search options). Each has candidates the pin changes and
# candidates it leaves alone.
CASES = {
    "mha_bhsd": (g.mha_bhsd_graph, (1, 32, 4096, 4096, 64), {}),
    "matmul_matmul_small_n": (g.matmul_matmul_graph, (4096, 64, 64), {}),
    "mha_split_partial": (g.mha_split_partial_graph, (1, 32, 4, 4096, 1024, 64), {}),
    "qknorm": (g.qknorm_graph, (1, 64, 8, 4096, 4096, 128), {"topk": 32, "force_fusion": True}),
}


def _candidates(name):
    build, dims, options = CASES[name]
    graph = build(*dims)
    return graph, search_tiling(graph, **{"topk": 1000, **options})


def _maps(cand):
    return ({k.name: v for k, v in cand.tile_map.items()},
            {k.name: v for k, v in cand.itile_map.items()})


def _pin_from_scratch(cand):
    """The pin as it was computed before the memo: everything from this candidate's own
    clone of the flat graph. Returns the pinned iteration tiles, or the refusal's text."""
    tile_map, itile_map = _maps(cand)
    _, pruned = tiling._extract_split_extent(cand.flat_graph)
    nodes = [n for n in pruned.nodes if not n.is_placeholder() and not n.is_output()]
    if not tiling.absorbed_reductions(nodes, cand.rstep_map):
        return itile_map
    out_tokens = build_output_axis_provenance(nodes)
    red_tokens = build_reduction_axis_provenance(nodes, out_tokens)
    try:
        pinned = tiling._held_axis_tokens(nodes, cand.rstep_map, red_tokens)
    except ScheduleRejected as refusal:
        return str(refusal)
    for (node, out_id), dim_tokens in out_tokens.items():
        if out_id != 0 or node.name not in itile_map:
            continue
        tile, itile = tile_map[node.name], list(itile_map[node.name])
        if len(dim_tokens) != len(itile) or len(tile) != len(itile):
            continue
        for dim, tokens in enumerate(dim_tokens):
            if tokens & pinned:
                itile[dim] = tile[dim]
        itile_map[node.name] = itile
    return itile_map


def _pin(cand):
    tile_map, itile_map = _maps(cand)
    try:
        return tiling._pin_absorbed_reduction_axes(
            cand.flat_graph, cand.rstep_map, tile_map, itile_map
        )
    except ScheduleRejected as refusal:
        return str(refusal)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_pin_is_what_it_is_without_the_memo(name):
    _, cands = _candidates(name)
    changed = 0
    for cand in cands:
        expected = _pin_from_scratch(cand)
        assert _pin(cand) == expected
        changed += expected != _maps(cand)[1]
    assert 0 < changed < len(cands), "the case needs candidates the pin changes and others"


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_provenance_is_read_once_per_flat_graph(name, monkeypatch):
    calls = []

    def counted(nodes):
        calls.append(len(nodes))
        return build_output_axis_provenance(nodes)

    monkeypatch.setattr(tiling, "build_output_axis_provenance", counted)
    _, cands = _candidates(name)
    absorbed = 0
    for cand in cands:
        absorbed += bool(tiling.absorbed_reductions(list(cand.rstep_map), cand.rstep_map))
        try:
            apply_tiling(cand)
        except ScheduleRejected:
            pass
    # Once for the search and every tiling of it. It used to be once more per candidate
    # with an absorbed reduction.
    assert absorbed > 1
    assert len(calls) == 1, f"{len(calls)} reads for {len(cands)} candidates of one graph"


def test_a_graph_that_was_rewired_is_read_again():
    graph = g.matmul_rowsum_graph(4096, 4096, 64)
    first = tiling._scheduled_axes(graph)
    assert tiling._scheduled_axes(graph) is first
    assert {n.name for n in first.red_tokens} == {"mma_0", "rowsum_0"}

    # The same object with other nodes: what was read off it no longer describes it.
    graph.nodes = g.matmul_matmul_graph(4096, 64, 64).nodes
    second = tiling._scheduled_axes(graph)
    assert second is not first
    assert {n.name for n in second.red_tokens} == {n.name for n in second.nodes if n.raxis}
    assert {n.name for n in second.red_tokens} != {"mma_0", "rowsum_0"}
    assert tiling._scheduled_axes(graph) is second


def test_the_memo_goes_with_its_graph():
    graph = g.matmul_rowsum_graph(4096, 4096, 64)
    key = id(graph)
    tiling._scheduled_axes(graph)
    assert key in tiling._SCHEDULED_AXES
    del graph
    gc.collect()
    assert key not in tiling._SCHEDULED_AXES
