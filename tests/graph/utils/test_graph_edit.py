"""Editing a locked (tiled) graph: ``edit_graph`` / ``rekey_graph`` and friends (CPU).

``GraphNode.__hash__`` covers the loop's node list, so adding a node to a loop moves the
hash of the ``GraphNode`` that owns it. A dict keyed by that ``GraphNode`` -- the main
tier's ``tmap`` and ``tinfo_map`` of a hand-built or paged graph -- then still holds the
entry but can no longer find it. ``edit_graph`` is the way to make such an edit: it unlocks,
and on exit locks again and refiles every node-keyed dict under the current hashes.
"""

from dataclasses import fields

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.ir.graph import Graph, GraphNode, Iterator
from cleave.ir.graph_utils import (
    _NAME_KEYED_FIELDS,
    _NODE_KEYED_FIELDS,
    edit_graph,
    graph_tiers,
    lock_graph,
    rekey_graph,
    tmap_by_name,
    unlock_graph,
)
from cleave.ir.ops import Copy, Mul, wire_input
from cleave.schedule import apply_tiling

from tests._graphs import mha_bhsd_graph
from tests.graph.apply_tiling._test_utils import build_tiled_candidate
from tests.graph.handwritten._cases import CASES

# A stale key is still found when its old and new hash happen to probe the same slot of
# the dict (about one edit in eight on these 6-entry maps, whatever PYTHONHASHSEED is), so
# the hazard is shown over a series of edits that differ in the name of the added node.
_TRIALS = 24

_MHA_RSTEP = {
    "mma_0": {"v3": 64}, "mul_0": {}, "sub_0": {}, "exp_0": {}, "div_0": {},
    "mma_1": {"v3": 32}, "rowsum_0": {"v2": 32},
}


def _handwritten():
    return CASES["mha_bhsd"].graph()


def _tiled():
    return apply_tiling(
        build_tiled_candidate(mha_bhsd_graph(1, 32, 4096, 4096, 64), [1, 64, 64], _MHA_RSTEP))


def _loop(graph):
    return next((tier, owner) for tier, owner in graph_tiers(graph) if owner is not None)


def _add_lag(graph, name):
    """The smallest structural edit of a loop: a snapshot of its first accumulator."""
    tier, _ = _loop(graph)
    acc = tier.accum_nodes[0]
    lag = Copy(tuple(int(d) for d in acc.get_shape(0)), name=name, n_batch_dims=0)
    wire_input(acc, lag, 0)
    tier.nodes.append(lag)
    tier.lag_nodes.append(lag)
    tier.tmap[lag] = tier.tmap[acc]
    return lag


def _unfindable(graph):
    """``(tier, field, key)`` of every node-keyed entry its own dict cannot find."""
    out = []
    for tier, owner in graph_tiers(graph):
        for name in _NODE_KEYED_FIELDS:
            mapping = getattr(tier, name)
            out += [("main" if owner is None else owner.name, name, key.name)
                    for key in list(mapping) if key not in mapping]
    return out


def test_graph_tiers_lists_the_graph_and_its_loops_outermost_first():
    graph = _handwritten()
    tiers = graph_tiers(graph)
    assert [(tier is graph, owner) for tier, owner in tiers[:1]] == [(True, None)]
    assert [owner.name for _, owner in tiers[1:]] == ["GraphNode_0"]
    assert all(isinstance(owner, GraphNode) and owner.graph is tier for tier, owner in tiers[1:])
    assert all(any(owner is n for n in graph.nodes) for _, owner in tiers[1:])


def test_tmap_by_name_reads_a_tier_without_hashing_its_keys():
    graph = _handwritten()
    tier, owner = _loop(graph)
    unlock_graph(graph)  # GraphNode.__hash__ refuses to run now; names are still readable
    with pytest.raises(AssertionError, match="locked"):
        hash(owner)
    by_name = tmap_by_name(graph)
    lock_graph(graph)
    assert set(by_name) == {n.name for n in graph.tmap} and "GraphNode_0" in by_name
    assert all(by_name[n.name] is it for n, it in graph.tmap.items())
    assert tmap_by_name(tier)["rowsum_0"].tile_size == (64,)


def test_an_edit_outside_edit_graph_strands_the_entries_keyed_by_the_loop():
    stranded = set()
    for trial in range(_TRIALS):
        graph = _handwritten()
        _, owner = _loop(graph)
        assert not _unfindable(graph)
        before = hash(owner)
        unlock_graph(graph)
        _add_lag(graph, f"rowsum_0_prev_{trial}")
        lock_graph(graph)  # ... and no rekey
        # The cause, every time: the loop is filed under a hash it no longer has.
        assert hash(owner) != before
        stranded |= set(_unfindable(graph))
    # The consequence: the main tier still holds the loop's tile and buffer, unreachable.
    assert stranded == {("main", "tmap", "GraphNode_0"), ("main", "tinfo_map", "GraphNode_0")}


@pytest.mark.parametrize("build", [_handwritten, _tiled], ids=["handwritten", "apply_tiling"])
def test_edit_graph_refiles_every_key(build):
    for trial in range(_TRIALS if build is _handwritten else 2):
        graph = build()
        sizes = {(id(tier), name): len(getattr(tier, name))
                 for tier, _ in graph_tiers(graph) for name in _NODE_KEYED_FIELDS}
        dicts = {key: getattr(tier, key[1]) for tier, _ in graph_tiers(graph)
                 for key in [(id(tier), name) for name in _NODE_KEYED_FIELDS]}
        with edit_graph(graph) as editing:
            assert editing is graph and not graph.locked
            lag = _add_lag(graph, f"rowsum_0_prev_{trial}")
        assert graph.locked and all(tier.locked for tier, _ in graph_tiers(graph))
        assert not _unfindable(graph)
        tier, _ = _loop(graph)
        assert lag in tier.tmap and tier.lag_nodes == [lag]
        for tier, _ in graph_tiers(graph):
            for name in _NODE_KEYED_FIELDS:
                mapping = getattr(tier, name)
                # refilled in place (no alias of the dict is left behind), nothing dropped
                assert mapping is dicts[(id(tier), name)]
                grown = 1 if (name == "tmap" and lag in mapping) else 0
                assert len(mapping) == sizes[(id(tier), name)] + grown


def test_edit_graph_wants_a_locked_graph_and_relocks_on_failure():
    graph = _handwritten()
    unlock_graph(graph)
    with pytest.raises(AssertionError, match="locked"):
        with edit_graph(graph):
            pass
    lock_graph(graph)
    with pytest.raises(ValueError, match="refused"):
        with edit_graph(graph):
            raise ValueError("refused")  # how a rewrite refuses a candidate
    assert graph.locked and all(tier.locked for tier, _ in graph_tiers(graph))


def test_rekey_collapses_one_node_stored_twice_and_keeps_the_later_entry():
    stored_twice = 0
    for trial in range(_TRIALS):
        graph = _handwritten()
        _, owner = _loop(graph)
        unlock_graph(graph)
        _add_lag(graph, f"rowsum_0_prev_{trial}")
        lock_graph(graph)
        graph.tinfo_map[owner] = ("t_GraphNode_0_new", "r")  # filed again, under the new hash
        # Two entries for one node, unless both hashes probe the same slot (then the
        # first entry was overwritten in place).
        stored_twice += sum(k is owner for k in graph.tinfo_map) == 2
        rekey_graph(graph)
        assert sum(k is owner for k in graph.tinfo_map) == 1
        assert graph.tinfo_map[owner] == ("t_GraphNode_0_new", "r")
        assert not _unfindable(graph)
    assert stored_twice, "no edit left the loop in the map twice; the case tests nothing"


def test_rekey_refuses_to_merge_two_different_nodes():
    a, b = Mul((16, 32), const=2.0, name="a"), Mul((16, 32), const=2.0, name="b")
    tile = Iterator(lambda i: ("0", "0"), shape=(16, 32), tile_size=(16, 32))
    graph = Graph(nodes=[a, b], tmap={a: tile, b: tile})
    lock_graph(graph)
    rekey_graph(graph)  # distinct keys: nothing to say
    assert len(graph.tmap) == 2
    b.name = "a"  # an edit that makes two nodes of one tier indistinguishable
    with pytest.raises(AssertionError, match="two distinct `tmap` keys became equal"):
        rekey_graph(graph)


def test_every_dict_field_of_graph_is_classified_as_node_or_name_keyed():
    """A new dict-typed ``Graph`` field has to say which it is: a node-keyed one that
    ``rekey_graph`` does not know about would silently lose entries after an edit."""
    dict_fields = {
        f.name for f in fields(Graph)
        if f.default_factory is dict or "dict" in str(f.type).lower()
    }
    assert {"tmap", "tinfo_map", "tlayout_map", "binding", "smap", "accum_scale"} <= dict_fields
    assert not set(_NODE_KEYED_FIELDS) & set(_NAME_KEYED_FIELDS)
    assert len(set(_NODE_KEYED_FIELDS)) == len(_NODE_KEYED_FIELDS)
    assert len(set(_NAME_KEYED_FIELDS)) == len(_NAME_KEYED_FIELDS)
    assert set(_NODE_KEYED_FIELDS) | set(_NAME_KEYED_FIELDS) == dict_fields

    # ... and the classification is true of real graphs: what is called node-keyed is.
    for graph in (_handwritten(), CASES["gqa_paged_decode_h32_kv8_d128_ps1"].graph()):
        for tier, _ in graph_tiers(graph):
            for name in _NAME_KEYED_FIELDS:
                assert all(isinstance(k, str) for k in getattr(tier, name)), name
            for name in _NODE_KEYED_FIELDS:
                assert not any(isinstance(k, str) for k in getattr(tier, name)), name
