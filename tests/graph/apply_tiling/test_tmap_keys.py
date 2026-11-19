"""Every tier's ``tmap`` holds each node once, under a hash that still finds it (CPU).

When ``apply_tiling`` merges a second reduction into an existing loop, an inner placeholder
can gain a consumer (MLA decode: the latent kv feeds the score matmul and, after the merge,
the value matmul), and the tensor map is re-assigned for every inner node. While
``PlaceHolderNode.__hash__`` covered the consumers, the placeholder then sat in the map
twice: the current entry, and a stale one no lookup could reach (600 of 600 MLA decode
candidates). Nothing iterates a ``tmap``, so nothing noticed. The hash is the name only
now; this pins the consequence on the two attention shapes that merge.
"""

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.ir.graph import GraphNode
from cleave.schedule import apply_tiling

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


def _tiers(graph):
    """``graph`` and every loop tier nested in it, as ``(tier, owning GraphNode or None)``."""
    tiers = [(graph, None)]
    for tier, _ in tiers:
        tiers.extend((n.graph, n) for n in tier.nodes if isinstance(n, GraphNode))
    return tiers


@pytest.mark.parametrize(
    ("build", "tile", "rstep"),
    [
        (lambda: mla_decode_bhsd_graph(2, 128, 4096, 128, 64), [1, 64, 128], _MLA_RSTEP),
        (lambda: mha_bhsd_graph(1, 32, 4096, 4096, 64), [1, 64, 64], _MHA_RSTEP),
    ],
    ids=["mla_decode_bhsd", "mha_bhsd"],
)
def test_tmap_holds_each_node_once(build, tile, rstep):
    tiled = apply_tiling(build_tiled_candidate(build(), tile, rstep))
    tiers = _tiers(tiled)
    assert len(tiers) > 1, "the candidate built no loop; the case tests nothing"
    merged = False
    for tier, owner in tiers:
        where = "main" if owner is None else owner.name
        keys = list(tier.tmap.keys())
        twice = sorted({k.name for k in keys if sum(k is o for o in keys) > 1})
        assert not twice, f"{where}: stored twice in tmap: {twice}"
        lost = [k.name for k in keys if k not in tier.tmap]
        assert not lost, f"{where}: tmap keys a lookup cannot find: {lost}"
        merged |= owner is not None and len(tier.accum_nodes) > 1
    assert merged, "no loop holds a merged reduction; the case tests nothing"
