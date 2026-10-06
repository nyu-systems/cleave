"""What the paging pass builds for the ps1 paged decode layer, node by node.

The emitted text is checked against the operator's golden in ``test_expected_kernels.py``,
which is the assertion that carries the tuned performance over; this file is the structural
half of that -- the graph the pass hands to codegen, including the parts of it a golden
cannot distinguish (the int32 page-table tile's layout label, which the pass has to
overwrite after annotation without changing a byte of this operator's source).

Two head counts are covered because they exercise different spec values -- the query
head group (4 vs 8) and, with it, the opposite grid order -- so passing both shows the
layer is parametric and not a template for one operator.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tilelang")

from tests.graph.apply_paging._test_utils import (
    OPERATOR_CASES,
    build_paged_graph,
    paged_kernel_source,
)
from cleave.graph import GraphNode, OutputNode, PlaceHolderNode
from cleave.ops import Div, Log2, MaskedExp, Matmul, PagedGather, Reduction

# Read out of the operator registry rather than restated here, so this file cannot hold a
# second copy of the specs that drifts from the one the goldens and the demos are built from.
CASES = {
    name: OPERATOR_CASES[name][0]
    for name in (
        "gqa_paged_decode_h32_kv8_d128_ps1",
        # kv4's schedule puts the kv head on the leading grid axis, a measured difference of
        # a few percent that the spec has to be able to express.
        "gqa_paged_decode_h32_kv4_d128_ps1",
    )
}


@pytest.mark.parametrize("operator", sorted(CASES))
def test_access_layer_structure(operator):
    spec = CASES[operator]
    graph = build_paged_graph(spec)

    # `args` is the emitted signature, and paged attention declares five inputs then two
    # outputs -- the dense graph had three and one.
    assert [n.name for n in graph.args] == [
        "PlaceHolder_Q",
        "PlaceHolder_K_CACHE",
        "PlaceHolder_V_CACHE",
        "PlaceHolder_KV_INDPTR",
        "PlaceHolder_KV_INDICES",
        "OutputNode_0",
        "OutputNode_1",
    ]
    inputs = [n for n in graph.args if isinstance(n, PlaceHolderNode)]
    indptr = next(n for n in inputs if n.name == "PlaceHolder_KV_INDPTR")
    assert indptr.index_only, "the indptr is read for values, so it takes no tile"

    outputs = [n for n in graph.args if isinstance(n, OutputNode)]
    # The lse is one float per (request, query head), and a block owns one request and one
    # kv head's group of them -- so both the extent and the tile are named by the spec. A
    # weaker assertion (that the leading extent is *some* symbolic name) let a swap to the
    # page pool's extent through, which only the goldens then caught.
    assert tuple(graph.tmap[outputs[1]].shape) == (spec.batch_dim, spec.num_qo_heads)
    assert tuple(graph.tmap[outputs[1]].tile_size) == (1, spec.group)

    # Symbolic extents, and a grid over the two axes the spec asked for.
    assert graph.dyn_dims == ["batch", "num_pages", "len_indptr", "n_idx"]
    assert graph.scalar_params == [("sm_scale", "float32")]
    heads, batch = str(spec.num_kv_heads), "batch"
    assert graph.extent == (
        (batch, heads) if spec.grid_order == ("batch", "head") else (heads, batch)
    )

    # The epilogue divides by a floored denominator and takes the log of it.
    assert isinstance(next(n for n in graph.nodes if n.name == "div_0"), Div)
    assert isinstance(next(n for n in graph.nodes if n.name == "log2_0"), Log2)

    inner = next(n for n in graph.nodes if isinstance(n, GraphNode)).graph
    # The KV tiles are gathered, the softmax is masked, and the scale is a runtime value:
    # the dense Mul/Sub/Exp chain is gone.
    gathers = [n for n in inner.nodes if isinstance(n, PagedGather)]
    assert [n.name for n in gathers] == ["paged_gather_k", "paged_gather_v"]
    assert gathers[0].inputs[1].src_node is gathers[1].inputs[1].src_node, (
        "one page table feeds both gathers"
    )
    exp = next(n for n in inner.nodes if isinstance(n, MaskedExp))
    assert exp.scale is None, "the softmax scale arrives as a runtime scalar"
    assert not [n for n in inner.nodes if type(n).__name__ in ("Mul", "Sub", "Exp")]

    # The loop's bound is runtime, not the proxy extent the scheduler tiled against.
    assert inner.extent == 1
    assert inner.loop_trip == "T.ceildiv(valid_len, 64)"
    assert inner.loop_kind == "serial", "a gather loop must not be pipelined"
    assert [n.name for n in inner.prefetch_nodes] == ["paged_gather_k", "paged_gather_v"]
    assert [n.name for n in inner.accum_nodes] == ["rowsum_0", "mma_1"]
    assert len([n for n in inner.nodes if isinstance(n, Matmul)]) == 2
    assert isinstance(next(n for n in inner.nodes if n.name == "rowsum_0"), Reduction)

    # The int32 page-table tile must not be labelled a tensor-core layout: it is read one
    # entry per row. `annotate_tc_layout` reaches it through the gathers and marks it, so
    # the pass has to correct it. The correction does not change this operator's emitted
    # text (the tile is global-scope), so only this assertion catches its loss.
    index_tile = inner.binding["PlaceHolder_KV_INDICES"]
    assert inner.tlayout_map[index_tile] == "linear", (
        "the page table is labelled a tensor-core layout: annotate_tc_layout reaches the "
        "int32 tile through the gathers, and the paging pass has to overwrite that. The page "
        "table is read one entry per row, not in tensor-core fragments, and a tile that ever "
        "moves to shared -- as the MLA layer's page-table window does -- would silently get a "
        "swizzled layout and gather the wrong pages"
    )


def test_pass_configs_disable_bounds_predication():
    graph = build_paged_graph(CASES["gqa_paged_decode_h32_kv8_d128_ps1"])
    assert graph.pass_configs == {"tl.disable_safe_memory_legalize": True}


def test_proxy_extent_does_not_reach_the_kernel():
    """The KV length the scheduler tiled against must not appear in the emitted kernel."""
    spec = CASES["gqa_paged_decode_h32_kv8_d128_ps1"]
    source = paged_kernel_source(spec, kv_proxy=4096)
    assert "4096" not in source, "the proxy KV extent leaked into the kernel"
    # ... and a different proxy gives the same kernel, since only the tile survives.
    assert paged_kernel_source(spec, kv_proxy=8192) == source


SPEC = CASES["gqa_paged_decode_h32_kv8_d128_ps1"]


@pytest.mark.parametrize("kv_tile", [32, 64, 128])
def test_honours_the_searched_kv_tile(kv_tile):
    """The KV tile the dense stages chose has to reach the kernel, not a baked-in one.

    Without this the pass could ignore the schedule entirely and hardcode the shipped
    operator's numbers, and every other test here would still pass.
    """
    inner = next(
        n
        for n in build_paged_graph(SPEC, kv_tile=kv_tile).nodes
        if isinstance(n, GraphNode)
    ).graph
    assert inner.loop_trip == f"T.ceildiv(valid_len, {kv_tile})"
    assert inner.prefetch_guard == f"tile_base + {kv_tile} < valid_len"
    gather = next(n for n in inner.nodes if isinstance(n, PagedGather))
    assert tuple(inner.tmap[gather].tile_size) == (kv_tile, 128)
    score = next(n for n in inner.nodes if n.name == "mma_0")
    assert tuple(inner.tmap[score].tile_size) == (16, kv_tile)


@pytest.mark.parametrize("query_rows", [16, 32])
def test_honours_the_searched_query_tile(query_rows):
    inner = next(
        n
        for n in build_paged_graph(SPEC, query_rows=query_rows).nodes
        if isinstance(n, GraphNode)
    ).graph
    score = next(n for n in inner.nodes if n.name == "mma_0")
    assert tuple(inner.tmap[score].tile_size) == (query_rows, 64)
    value = next(n for n in inner.nodes if n.name == "mma_1")
    assert tuple(inner.tmap[value].tile_size) == (query_rows, 128)


def test_honours_the_schedules_thread_count():
    from tests.graph.apply_paging._test_utils import tiled_dense_attention
    from tests._graphs import flash_attention_sd_graph
    from cleave.paging import apply_paging

    dense = flash_attention_sd_graph(16, 4096, 128, dtype="bfloat16", out_dtype="bfloat16")
    scheduled = tiled_dense_attention(dense, 16, 128, 64)
    assert scheduled.n_threads == 128
    scheduled.n_threads = 256
    assert apply_paging(scheduled, SPEC).n_threads == 256
