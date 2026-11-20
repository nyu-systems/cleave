"""Every tier's ``tmap`` holds each node once, under a hash that still finds it (CPU).

When ``apply_tiling`` merges a second reduction into an existing loop, the tensor map of the
loop is assigned again for every inner node. In MLA decode the merge also gives an inner
placeholder a second consumer: the latent kv feeds the score matmul and, once the value
matmul has joined the loop, that one too. While ``PlaceHolderNode.__hash__`` covered the
consumers, the placeholder's hash moved while it was a key of the map. The second assignment
then stored it twice -- the current entry, and a stale one no lookup could reach (600 of 600
MLA decode candidates) -- or, when the old and the new hash happened to probe the same slot,
which depends on ``PYTHONHASHSEED``, updated the entry in place under a hash the key no
longer had. Nothing iterates a ``tmap``, so nothing noticed. The hash is the name only now.

Two shapes, and only the first exercises the defect:

* ``mla_decode_bhsd`` -- the merge gives a placeholder a consumer. Pinned here: the cause
  (the hash of a key does not move between two assignments; holds or fails on every hash
  seed) and the consequence (one entry per node, each one found).
* ``mha_bhsd`` -- the merge only adds nodes; no placeholder is shared (K feeds the scores,
  V the values). It is the control: the same invariants on a merge that never had the defect.

``tests/schedule/search_tiling/test_shared_placeholder.py`` pins the same cause on LoRA.
"""

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.ir.graph_utils import graph_tiers
from cleave.schedule import apply_tiling
from cleave.schedule import tiling

from tests._graphs import mha_bhsd_graph, mla_decode_bhsd_graph
from tests.graph.apply_tiling._test_utils import build_tiled_candidate

_MLA_RSTEP = {
    "mma_0": {"v3": 128}, "mma_pe": {"v3": 64}, "add_0": {}, "mul_0": {}, "sub_0": {},
    "exp_0": {}, "div_0": {}, "mma_1": {"v3": 32}, "rowsum_0": {"v2": 32},
}
_MHA_RSTEP = {
    "mma_0": {"v3": 64}, "mul_0": {}, "sub_0": {}, "exp_0": {}, "div_0": {},
    "mma_1": {"v3": 32}, "rowsum_0": {"v2": 32},
}


def _watch_assignments(monkeypatch):
    """Record every ``assign_tensor_map`` call ``apply_tiling`` makes: the loop, and each
    inner placeholder with its consumer count and its hash once the call has returned."""
    calls = []
    assign = tiling.assign_tensor_map

    def watched(graph_node, *args, **kwargs):
        assign(graph_node, *args, **kwargs)
        placeholders = [
            (n, len(n.outputs), hash(n)) for n in graph_node.graph.nodes if n.is_placeholder()
        ]
        calls.append((graph_node, placeholders))

    monkeypatch.setattr(tiling, "assign_tensor_map", watched)
    return calls


@pytest.mark.parametrize(
    ("build", "tile", "rstep", "gains"),
    [
        (lambda: mla_decode_bhsd_graph(2, 128, 4096, 128, 64), [1, 64, 128], _MLA_RSTEP,
         ["PlaceHolder_BB"]),
        (lambda: mha_bhsd_graph(1, 32, 4096, 4096, 64), [1, 64, 64], _MHA_RSTEP, []),
    ],
    ids=["mla_decode_bhsd", "mha_bhsd"],
)
def test_tmap_holds_each_node_once(build, tile, rstep, gains, monkeypatch):
    calls = _watch_assignments(monkeypatch)
    tiled = apply_tiling(build_tiled_candidate(build(), tile, rstep))

    # The case: a loop whose map is assigned a second time (a merge), and, where the row
    # says so, a placeholder that was a key of that map and has a consumer more by then.
    loops = [loop for loop, _ in calls]
    assert any(sum(loop is other for other in loops) > 1 for loop in loops), (
        "no loop was merged into; the case tests nothing"
    )
    first, gained, moved = [], [], []
    for loop, placeholders in calls:
        for node, consumers, hashed in placeholders:
            seen = next((entry for entry in first if entry[0] is node), None)
            if seen is None:
                first.append((node, consumers, hashed))
                continue
            if consumers > seen[1] and node.name not in gained:
                gained.append(node.name)
            if hashed != seen[2]:
                moved.append(f"{loop.name}: {node.name}")
    assert gained == gains, "the merge no longer shares what this row was written for"

    # The cause: a key's hash is what it was when the key went into the map.
    assert not moved, f"hash moved while the node was a key of its loop's tmap: {moved}"

    # The consequence, on every tier.
    tiers = graph_tiers(tiled)
    assert len(tiers) > 1, "the candidate built no loop; the case tests nothing"
    for tier, owner in tiers:
        where = "main" if owner is None else owner.name
        keys = list(tier.tmap.keys())
        twice = sorted({k.name for k in keys if sum(k is o for o in keys) > 1})
        assert not twice, f"{where}: stored twice in tmap: {twice}"
        lost = [k.name for k in keys if k not in tier.tmap]
        assert not lost, f"{where}: tmap keys a lookup cannot find: {lost}"
    assert any(owner is not None and len(tier.accum_nodes) > 1 for tier, owner in tiers), (
        "no loop holds a merged reduction; the case tests nothing"
    )
