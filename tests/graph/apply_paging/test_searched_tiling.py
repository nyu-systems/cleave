"""The tiling the dense stages searched has to reach every layer's kernel.

``apply_paging`` rebuilds the attention body from the spec and keeps nothing of the dense
graph but its tiling, so a builder that writes the shipped operator's numbers instead of
reading them emits a kernel the schedule never described -- and inherits a measurement that
belongs to a different tile. Nothing else in this package notices: the goldens are compared
byte for byte at exactly the tuned tiling, which is the tiling a hardcode reproduces.

So every layer is built twice, at the tiling its tuned schedule was measured at and at one
other legal one, and the numbers the loop tier and the epilogue are made of are read back.
The second tiling is not free: each layer constrains what it can address -- the ragged
layer's P fragment needs whole 64-row blocks on a 128-thread block, a slab has to fit inside
one page, an MLA tile is one token's whole block of query heads -- and those constraints are
what ``test_apply_paging_dispatch.py`` pins.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tilelang")

from tests.graph.apply_paging._test_utils import build_mla_paged_graph, build_paged_graph
from cleave.graph import GraphNode, OutputNode
from cleave.paging import (
    MLA_PAGED_DECODE,
    MLA_PAGED_PREFILL_CAUSAL,
    PAGED_DECODE,
    PAGED_DECODE_SLAB,
    PAGED_PREFILL_CAUSAL,
    RAGGED_PREFILL_CAUSAL,
    MLASpec,
    PagingSpec,
)

# The layers whose block owns one request and one kv head, so that the head group -- and not
# the query tile -- is what their epilogue writes.
DECODE_KINDS = (PAGED_DECODE, PAGED_DECODE_SLAB)


def gqa(query_rows, head_dim, kv_tile, n_threads=128):
    return dict(
        query_rows=query_rows, head_dim=head_dim, kv_tile=kv_tile, n_threads=n_threads
    )


def mla(query_rows, ckv, kv_tile, n_threads):
    return dict(query_rows=query_rows, ckv=ckv, kv_tile=kv_tile, n_threads=n_threads)


def slab(num_kv_heads):
    return PagingSpec(
        num_qo_heads=32,
        num_kv_heads=num_kv_heads,
        kind=PAGED_DECODE_SLAB,
        page_size=64,
        pass_configs={},
    )


def mla_prefill(num_qo_heads):
    return MLASpec(
        num_qo_heads=num_qo_heads,
        kind=MLA_PAGED_PREFILL_CAUSAL,
        causal=True,
        index_window_tiles=4,
        prefetch=False,
    )


PREFILL = PagingSpec(
    num_qo_heads=32, num_kv_heads=8, kind=PAGED_PREFILL_CAUSAL, causal=True
)
RAGGED = PagingSpec(
    num_qo_heads=32,
    num_kv_heads=8,
    kind=RAGGED_PREFILL_CAUSAL,
    causal=True,
    rasterization=None,
    cache_inputs=("K", "V"),
)

# Two builds per layer: the tuned one, and a retiling that moves every number the builder
# could have baked in -- the KV tile, the query tile, the head dim and the block size.
CASES = {
    PAGED_DECODE: (PagingSpec(num_qo_heads=32, num_kv_heads=8), gqa(16, 128, 64)),
    "paged_decode_retiled": (
        PagingSpec(num_qo_heads=32, num_kv_heads=8),
        gqa(32, 64, 32, 256),
    ),
    # A slab has to lie inside one page, so the KV tile can only shrink here.
    PAGED_DECODE_SLAB: (slab(8), gqa(16, 128, 64)),
    "paged_decode_slab_retiled": (slab(8), gqa(16, 64, 32, 256)),
    # ... and a second head group, which is the one number a slab's addressing depends on
    # that its KV tile does not: the group is the rows of the query gather and of the store.
    "paged_decode_slab_kv4": (slab(4), gqa(16, 128, 64)),
    # A prefill block's M axis fuses (token, head slot), so its rows stay a multiple of the
    # group of 4. The tuned KV tile is the narrow one; widening it is the interesting way.
    PAGED_PREFILL_CAUSAL: (PREFILL, gqa(64, 128, 16)),
    "paged_prefill_causal_retiled": (PREFILL, gqa(32, 64, 32, 256)),
    # The ragged layer's fragment holds it to whole 64-row blocks on a 128-thread block, so
    # this is the one layer whose block size cannot be moved -- it refuses 256 outright.
    RAGGED_PREFILL_CAUSAL: (RAGGED, gqa(64, 128, 64)),
    "ragged_prefill_causal_retiled": (RAGGED, gqa(128, 64, 32)),
    # An MLA tile is one token's whole block of query heads, so retiling the query axis means
    # a different head count -- they are the same number.
    MLA_PAGED_DECODE: (MLASpec(num_qo_heads=16), mla(16, 512, 64, 256)),
    "mla_paged_decode_retiled": (MLASpec(num_qo_heads=32), mla(32, 256, 32, 256)),
    MLA_PAGED_PREFILL_CAUSAL: (mla_prefill(16), mla(16, 512, 64, 128)),
    # The one layer whose tuned block is the narrow one, so its retiling is the wide block.
    "mla_paged_prefill_causal_retiled": (mla_prefill(32), mla(32, 256, 32, 256)),
}


def _named(graph, name):
    return next(n for n in graph.nodes if n.name == name)


@pytest.mark.parametrize("case", sorted(CASES))
def test_honours_the_searched_tile(case):
    spec, tiling = CASES[case]
    is_mla = isinstance(spec, MLASpec)
    graph = (build_mla_paged_graph if is_mla else build_paged_graph)(spec, **tiling)
    rows, kv_tile = tiling["query_rows"], tiling["kv_tile"]
    width = tiling["ckv"] if is_mla else tiling["head_dim"]
    inner = next(n for n in graph.nodes if isinstance(n, GraphNode)).graph

    # One loop iteration per KV tile, and the KV tile is the score matmul's N extent.
    assert inner.loop_trip == f"T.ceildiv(valid_len, {kv_tile})"
    assert tuple(inner.tmap[_named(inner, "mma_0")].tile_size) == (rows, kv_tile)
    # The value matmul spans the whole head dim -- the recognizers refuse a split one -- so
    # its N extent is the operator's width rather than a block of it.
    assert tuple(inner.tmap[_named(inner, "mma_1")].tile_size) == (rows, width)
    # The epilogue is sized from the same tiling. For a decode block the rows that are real
    # are its kv head's group of query heads; every other layer writes the whole query tile.
    epi_rows = spec.group if spec.kind in DECODE_KINDS else rows
    assert tuple(_named(graph, "div_0").dims) == (epi_rows, width)
    assert tuple(_named(graph, "log2_0").dims) == (epi_rows,)
    # And the block size, which the policy searches alongside the tile rather than deriving.
    assert graph.n_threads == tiling["n_threads"]

    if spec.kind in DECODE_KINDS:
        # The group is also the head extent of the two tiles the block stores, which is a
        # second place to get it wrong: these come from the layer's tensor-map helper.
        outputs = [n for n in graph.args if isinstance(n, OutputNode)]
        assert tuple(graph.tmap[outputs[0]].tile_size) == (1, spec.group, width)
        assert tuple(graph.tmap[outputs[1]].tile_size) == (1, spec.group)
