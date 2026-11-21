"""What the paging pass builds for the ps1 paged decode layer, node by node.

The emitted text is checked against the operator's expected in ``test_expected_kernels.py``,
which is the assertion that carries the tuned performance over; this file is the structural
half of that -- the graph the pass hands to codegen, including the parts of it a expected
cannot distinguish (the int32 page-table tile's layout label, which the pass has to
overwrite after annotation without changing a byte of this operator's source).

Two head counts are covered because they exercise different spec values -- the query
head group (4 vs 8) and, with it, the opposite grid order -- so passing both shows the
layer is parametric and not a template for one operator.

The softmax is exact in every layer: the loop carries a running row max of its masked,
scaled scores, takes each tile relative to it and rescales what it has accumulated when
the max moves. The second half of this file pins that for every operator of the registry
and for the split-KV partial, which stores the max as its third partial.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tilelang")

from tests.graph.apply_paging._test_utils import OPERATOR_CASES
from cleave.codegen import TilelangCodeGen
from cleave.dynamic import OPERATORS
from cleave.dynamic.dense import build_mla_paged_graph, build_paged_graph, paged_kernel_source
from cleave.dynamic.paging import DenseAttention, MLASpec, build_split_decode
from cleave.dynamic.paging.specs import CombineTiling
from cleave.ir.graph import AccumEdge, GraphNode, OutputNode, PlaceHolderNode
from cleave.ir.ops import (
    NEG_FLT_MAX,
    Copy,
    Div,
    Exp2,
    Lse2,
    MaskedExp,
    MaskedLse2,
    MaskedScore,
    Matmul,
    PagedGather,
    ReduceMax,
    ReduceSum,
)

# Read out of the operator registry rather than restated here, so this file cannot hold a
# second copy of the specs that drifts from the one the expected kernels and the demos are built from.
CASES = {
    name: OPERATOR_CASES[name][0]
    for name in (
        "gqa_paged_decode_h32_kv8_d128_ps1",
        # kv4's schedule puts the kv head on the leading grid axis, a measured difference of
        # a few percent that the spec has to be able to express.
        "gqa_paged_decode_h32_kv4_d128_ps1",
    )
}


def _loop(graph):
    return next(n for n in graph.nodes if isinstance(n, GraphNode))


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
    # page pool's extent through, which only the expected kernels then caught.
    assert tuple(graph.tmap[outputs[1]].shape) == (spec.batch_dim, spec.num_qo_heads)
    assert tuple(graph.tmap[outputs[1]].tile_size) == (1, spec.group)

    # Symbolic extents, and a grid over the two axes the spec asked for.
    assert graph.dyn_dims == ["batch", "num_pages", "len_indptr", "n_idx"]
    assert graph.scalar_params == [("sm_scale", "float32")]
    heads, batch = str(spec.num_kv_heads), "batch"
    assert graph.extent == (
        (batch, heads) if spec.grid_order == ("batch", "head") else (heads, batch)
    )

    # The epilogue divides by a floored denominator, and its lse is the log of the row sum
    # plus the carried max: the row sum is a sum of exp2(s - max).
    div = next(n for n in graph.nodes if n.name == "div_0")
    assert type(div) is Div and div.denom_floor == 1e-30
    lse = outputs[1].inputs[0].src_node
    assert (type(lse), lse.name) == (Lse2, "lse_0")
    assert not [n.name for n in graph.nodes if n.name == "log2_0"]

    loop = _loop(graph)
    inner = loop.graph
    # The KV tiles are gathered, the softmax is masked, and the scale is a runtime value:
    # the dense Mul/Sub/Exp chain is gone.
    gathers = [n for n in inner.nodes if isinstance(n, PagedGather)]
    assert [n.name for n in gathers] == ["paged_gather_k", "paged_gather_v"]
    assert gathers[0].inputs[1].src_node is gathers[1].inputs[1].src_node, (
        "one page table feeds both gathers"
    )
    exp = next(n for n in inner.nodes if isinstance(n, MaskedExp))
    assert exp.scale is None, "the softmax scale arrives as a runtime scalar"
    assert exp.shift is True
    assert not [n for n in inner.nodes if type(n).__name__ in ("Mul", "Exp")]
    subs = [n.name for n in inner.nodes if type(n).__name__ == "Sub"]
    # The exp is the shifted form: it reads the masked, scaled score and that score's
    # running row max, and takes the mask scalars alone -- the scale is the score's.
    score = next(n for n in inner.nodes if isinstance(n, MaskedScore))
    row_max = next(n for n in inner.nodes if isinstance(n, ReduceMax))
    assert (score.name, row_max.name) == ("score_0", "exp_0_max")
    assert score.inputs[0].src_node.name == "mma_0"
    assert [e.src_node for e in exp.inputs] == [score, row_max]
    assert row_max.inputs[0].src_node is score, "the max is taken after scaling"
    assert score.scalar_args == ("tile_base", "valid_len", "scale2")
    assert exp.scalar_args == ("tile_base", "valid_len")
    # float32 statistics; the max starts at the most negative finite float32
    assert row_max.accum_init == NEG_FLT_MAX and row_max.axis == 1
    assert {score.out_dtype, row_max.out_dtype} == {"float32"}
    # The rolling update is declared on the tier: a snapshot of the max, and one factor
    # exp2(m_prev - m) for both accumulators. Its only subtraction is that step.
    assert subs == ["exp_0_max_step"]
    assert [n.name for n in inner.lag_nodes] == ["exp_0_max_prev"]
    assert inner.lag_nodes[0].inputs[0].src_node is row_max
    alpha = next(n for n in inner.nodes if n.name == "exp_0_alpha")
    assert type(alpha) is Exp2 and alpha.inputs[0].src_node.name == "exp_0_max_step"
    assert {acc.name: f for acc, f in inner.accum_scale.items()} == {
        "rowsum_0": alpha, "mma_1": alpha}
    # The lse reads the row sum and the max through the loop's accumulators, by index.
    assert all(isinstance(e, AccumEdge) and e.src_node is loop for e in lse.inputs)
    assert [(e.accum_node.name, e.src_id) for e in lse.inputs] == [
        ("rowsum_0", 0), ("exp_0_max", 2)]
    assert [(e.accum_node.name, e.src_id) for e in div.inputs] == [
        ("mma_1", 1), ("rowsum_0", 0)]

    # The loop's bound is runtime, not the proxy extent the scheduler tiled against.
    assert inner.extent == 1
    assert inner.loop_trip == "T.ceildiv(valid_len, 64)"
    assert inner.loop_kind == "serial", "a gather loop must not be pipelined"
    assert [n.name for n in inner.prefetch_nodes] == ["paged_gather_k", "paged_gather_v"]
    # The carried max is appended behind the automatic accumulators, whose indices stay.
    assert [n.name for n in inner.accum_nodes] == ["rowsum_0", "mma_1", "exp_0_max"]
    assert len(graph.tmap[loop]) == len(inner.accum_nodes)
    assert [tuple(it.tile_size) for it in graph.tmap[loop]] == [(16,), (16, 128), (16,)]
    assert len([n for n in inner.nodes if isinstance(n, Matmul)]) == 2
    assert isinstance(next(n for n in inner.nodes if n.name == "rowsum_0"), ReduceSum)

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
    from cleave.dynamic.dense import flash_attention_sd_graph, tiled_dense_attention
    from cleave.dynamic.paging import apply_paging

    dense = flash_attention_sd_graph(16, 4096, 128, dtype="bfloat16", out_dtype="bfloat16")
    scheduled = tiled_dense_attention(dense, 16, 128, 64)
    assert scheduled.n_threads == 128
    scheduled.n_threads = 256
    assert apply_paging(scheduled, SPEC).n_threads == 256


# ------------------------------------------------------------------ the exact softmax
def _exact(graph) -> bool:
    """Whether a paged graph's loop tier carries a running max (and then all of it does)."""
    inner = _loop(graph).graph
    exp = next(n for n in inner.nodes if isinstance(n, MaskedExp))
    carried = [n.name for n in inner.accum_nodes if isinstance(n, ReduceMax)]
    assert bool(carried) == exp.shift == bool(inner.lag_nodes) == bool(inner.accum_scale)
    return exp.shift


@pytest.mark.parametrize("operator", sorted(OPERATORS))
def test_every_operator_has_an_exact_softmax(operator):
    """Every layer: GQA and MLA, decode and causal prefill, gathered, slab and ragged keys."""
    op = OPERATORS[operator]
    is_mla, causal, tiling = isinstance(op.spec, MLASpec), op.spec.causal, op.tuned_tiling
    graph = (build_mla_paged_graph if is_mla else build_paged_graph)(op.spec, **tiling)
    assert _exact(graph)
    loop = _loop(graph)
    inner = loop.graph
    by_name = {n.name: n for n in inner.nodes}
    exp, score, row_max = by_name["exp_0"], by_name["score_0"], by_name["exp_0_max"]
    # The max is taken of the masked, scaled score -- of the summed score for MLA -- and the
    # exp and the score share one mask: the same predicate, scalars and row grouping.
    assert score.inputs[0].src_node.name == ("add_0" if is_mla else "mma_0")
    assert [e.src_node for e in exp.inputs] == [score, row_max]
    assert row_max.inputs[0].src_node is score and row_max.accum_init == NEG_FLT_MAX
    assert type(exp).__name__ == ("CausalMaskedExp" if causal else "MaskedExp")
    assert type(score).__name__ == ("CausalMaskedScore" if causal else "MaskedScore")
    assert score.scalar_args == (*exp.scalar_args, "scale2")
    assert score.row_group == exp.row_group
    # Both sums of exp2(s - m) terms are rescaled by the one factor, and the max is carried
    # behind them. MLA's denominator is the whole (rows, kv) tile, folded after the loop.
    denom, alpha = ("psum_0" if is_mla else "rowsum_0"), by_name["exp_0_alpha"]
    assert {acc.name: f for acc, f in inner.accum_scale.items()} == {
        denom: alpha, "mma_1": alpha}
    assert [n.name for n in inner.lag_nodes] == ["exp_0_max_prev"]
    assert [n.name for n in inner.accum_nodes] == [denom, "mma_1", "exp_0_max"]
    rows, kv = tiling["query_rows"], tiling["kv_tile"]
    width = tiling["ckv"] if is_mla else tiling["head_dim"]
    assert [tuple(it.tile_size) for it in graph.tmap[loop]] == [
        (rows, kv) if is_mla else (rows,), (rows, width), (rows,)]
    assert tuple(inner.tmap[score].tile_size) == (rows, kv)
    # The lse is log2(l) + m, row-predicated in the prefill layers; it reads the max
    # through the loop's accumulators. It is the only node that takes a log: the plain
    # log2 ops the layers used before are gone from the IR.
    lse = next(n for n in graph.nodes if n.name == "lse_0")
    assert type(lse) is (MaskedLse2 if causal else Lse2)
    assert [n.name for n in graph.nodes if "log2" in n.name or "lse" in n.name] == ["lse_0"]
    row_sum, carried = lse.inputs
    assert isinstance(carried, AccumEdge) and carried.src_node is loop
    assert (carried.accum_node.name, carried.src_id) == ("exp_0_max", 2)
    if is_mla:
        assert type(row_sum.src_node) is ReduceSum and row_sum.src_node.name == "rowsum_0"
        assert row_sum.src_node.inputs[0].accum_node.name == "psum_0"
    else:
        assert (row_sum.accum_node.name, row_sum.src_id) == ("rowsum_0", 0)


def test_the_split_partial_is_exact_and_stores_its_max():
    """A partial's row sums and values are relative to its own running max, so it hands that
    max to the combine as a third partial (``test_split_decode.py`` pins the pair)."""
    spec = CASES["gqa_paged_decode_h32_kv8_d128_ps1"]
    dense = DenseAttention(query_rows=16, kv_tile=64, head_dim=128, n_threads=128,
                           dtype="bfloat16", accum_dtype="float32", out_dtype="bfloat16")
    partial, _combine = build_split_decode(
        spec, dense, 8, CombineTiling(split=8, split_step=1, cols=128, n_threads=128))
    assert _exact(partial)
    loop = _loop(partial)
    assert [n.name for n in loop.graph.accum_nodes] == ["rowsum_0", "mma_1", "exp_0_max"]
    stores = {n.name: n for n in partial.nodes if isinstance(n, Copy)}
    assert sorted(stores) == ["partial_max", "partial_rowsum", "partial_value"]
    read = {name: store.inputs[0] for name, store in stores.items()}
    assert all(isinstance(e, AccumEdge) and e.src_node is loop for e in read.values())
    assert {name: (e.accum_node.name, e.src_id) for name, e in read.items()} == {
        "partial_rowsum": ("rowsum_0", 0), "partial_value": ("mma_1", 1),
        "partial_max": ("exp_0_max", 2)}


@pytest.mark.parametrize("tiling", [dict(kv_tile=32), dict(kv_tile=128), dict(query_rows=32),
                                    dict(head_dim=64)])
def test_the_exact_softmax_at_any_tiling(tiling):
    """The softmax belongs to the layer, not to a tuned operator: any tiling, any head dim."""
    graph = build_paged_graph(SPEC, **tiling)
    assert _exact(graph)
    inner = _loop(graph).graph
    rows, kv = tiling.get("query_rows", 16), tiling.get("kv_tile", 64)
    by_name = {n.name: n for n in inner.nodes}
    assert tuple(inner.tmap[by_name["score_0"]].tile_size) == (rows, kv)
    for name in ("exp_0_max", "exp_0_max_prev", "exp_0_max_step", "exp_0_alpha"):
        assert tuple(inner.tmap[by_name[name]].tile_size) == (rows,), name
        assert inner.tinfo_map[by_name[name]] == (f"t_{name}", "r")
    assert tuple(graph.tmap[_loop(graph)][2].tile_size) == (rows,)


def test_exact_kernel_text():
    """What the exact softmax adds to the kernel, statement by statement."""
    source = paged_kernel_source(SPEC)
    body = source[source.index("def GraphNode_0_iter("):source.index("def GraphNode_0(")]
    flat = "".join(body.split()).replace(",)", ")")
    statements = [
        "ukernel_mma_ssr_transB_clear(PlaceHolder_QQ,t_paged_gather_k,t_mma_0)",
        "ukernel_maskedscore_scale_rt2_2dims_rr(t_mma_0,tile_base,valid_len,scale2,t_score_0)",
        "ukernel_copy_1dims_rr(t_exp_0_max,t_exp_0_max_prev)",
        "ukernel_reduce_max_axis1_rr(t_score_0,t_exp_0_max)",
        "ukernel_maskedexp_shift2_2dims_rrr(t_score_0,t_exp_0_max,tile_base,valid_len,t_exp_0)",
        "ukernel_sub_1dims_rrr(t_exp_0_max_prev,t_exp_0_max,t_exp_0_max_step)",
        "ukernel_exp2_1dims_rr(t_exp_0_max_step,t_exp_0_alpha)",
        "ukernel_rescale_1dims_rr(t_exp_0_alpha,t_rowsum_0)",
        "ukernel_reduction_axis1_rr(t_exp_0,t_rowsum_0)",
        "T.copy(t_exp_0,t_exp_0_cast)",
        "ukernel_rescale_lead1_2dims_rr(t_exp_0_alpha,t_mma_1)",
        "ukernel_mma_ssr(t_exp_0_cast,t_paged_gather_v,t_mma_1)",
    ]
    at = [flat.index(s) for s in statements]
    assert at == sorted(at)
    # The max is carried: allocated and filled outside the loop, never inside it.
    assert "t_exp_0_max=T.alloc_fragment" not in flat
    assert "T.fill(t_exp_0_max, -3.4028234663852886e38)" in source
    assert "T.clear(t_rowsum_0)" in source and "T.clear(t_mma_1)" in source
    # A masked score is that same finite value and a masked P the literal 0: no -inf.
    assert source.count("-3.4028234663852886e38") == 2 and "infinity" not in source
    assert "T.exp2(A[m0, m1] - M[m0]), 0.0)" in source
    assert "T.exp2(A[m0, m1] * scale2)" not in source
    # lse = log2(l) + m with the row sum unfloored; the floor guards the divide alone.
    assert "ukernel_lse2_1dims_rrg(t_rowsum_0, t_exp_0_max, OutputNode_1)" in source
    assert "B[i0] = T.log2(A[i0]) + M[i0]" in source
    assert source.count("T.max(") == 1 and "T.max(B[i0], 1e-30)" in source


def test_exact_kernel_without_the_lse():
    """With no lse the carried max has no reader after the loop, and the kernel still emits."""
    from dataclasses import replace

    graph = build_paged_graph(replace(SPEC, emit_lse=False))
    assert _exact(graph)
    assert not [n for n in graph.nodes if isinstance(n, Lse2) or "log2" in n.name]
    assert [n.name for n in graph.args if isinstance(n, OutputNode)] == ["OutputNode_0"]
    source = TilelangCodeGen().gen(graph)
    assert "lse2" not in source and "T.fill(t_exp_0_max, -3.4028234663852886e38)" in source
