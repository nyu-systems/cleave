"""What the paging pass builds for split-KV decode: the pair, its structure, its expected kernels.

Split-KV is the decode layers' answer to a grid of fewer blocks than the GPU has SMs: the
KV window of one (request, kv head) is dealt out over ``split`` blocks that each store a
raw partial numerator and denominator and the running max both are relative to, and a
second kernel weighs the partials by those maxima, sums them and applies the epilogue. The
pass builds the pair from the same tiling decisions it reads for the single kernel plus the
combine's own (``CombineTiling``), and codegen emits one prim_func of two kernels sharing
the partial tensors.

Three kinds of assertion here. The structural half pins the graph the pass hands to
codegen -- the extra grid axis, the three ``Copy`` stores, the combine's grid, its weights
and its three carried values. The expected kernels pin the emitted text, which is what the
measured numbers belong to (``tests/graph/handwritten/test_splitkv_decode.py`` runs them).
The arithmetic half checks the tile-range formulas the partial kernel is built on by
evaluating the emitted expressions in Python: every KV tile of every request must be owned
by exactly one split.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tilelang")

from tests.graph.apply_paging._test_utils import OPERATOR_CASES
from tests.graph.handwritten._test_utils import assert_expected_tilelang_kernel
from cleave.codegen import TilelangCodeGen
from cleave.ir.graph import AccumEdge, GraphNode, OutputNode, PlaceHolderNode
from cleave.ir.ops import (
    NEG_FLT_MAX,
    Accumulate,
    AccumulateMax,
    Copy,
    Div,
    Exp2,
    Log2,
    Lse2,
    Mul,
    ReduceMax,
    ReduceSum,
    Sub,
)
from cleave.dynamic.paging import (
    PARTIAL_MAX,
    PARTIAL_ROWSUM,
    PARTIAL_VALUE,
    SPLIT_KINDS,
    CombineTiling,
    DenseAttention,
    DenseMLA,
    MLASpec,
    build_split_decode,
)
from cleave.dynamic.paging.stages import _split_range_scalars

# The four decode operators, at their tuned partial tiling and the combine tiling the
# searches rank first (one partial per step, the head dim in 128-wide column blocks).
SPLIT = 8
CASES = {
    op: OPERATOR_CASES[op]
    for op in (
        "gqa_paged_decode_h32_kv8_d128_ps1",
        "gqa_paged_decode_h32_kv4_d128_ps1",
        "gqa_paged_decode_h32_kv8_d128_ps64",
        "mla_paged_decode_h16_ckv512_kpe64_ps1",
    )
}


def dense_of(spec, tiling):
    if isinstance(spec, MLASpec):
        return DenseMLA(
            query_rows=tiling["query_rows"],
            kv_tile=tiling["kv_tile"],
            head_dim_ckv=tiling["ckv"],
            head_dim_kpe=tiling["kpe"],
            n_threads=tiling.get("n_threads", 128),
            dtype="bfloat16",
            accum_dtype="float32",
            out_dtype="bfloat16",
        )
    return DenseAttention(
        query_rows=tiling["query_rows"],
        kv_tile=tiling["kv_tile"],
        head_dim=tiling["head_dim"],
        n_threads=tiling.get("n_threads", 128),
        dtype="bfloat16",
        accum_dtype="float32",
        out_dtype="bfloat16",
    )


def combine_of(spec, tiling, split=SPLIT, step=1):
    head_dim = tiling.get("head_dim") or tiling["ckv"]
    return CombineTiling(split=split, split_step=step, cols=min(128, head_dim), n_threads=128)


def build(op, split=SPLIT, step=1):
    spec, tiling = CASES[op]
    return build_split_decode(spec, dense_of(spec, tiling), split, combine_of(spec, tiling, split, step))


def split_kernel_source(op, split=SPLIT, step=1) -> str:
    return TilelangCodeGen().gen(build(op, split, step))


def test_split_kinds_are_the_decode_layers():
    assert set(SPLIT_KINDS) == {"paged_decode", "paged_decode_slab", "mla_paged_decode"}
    assert all(CASES[op][0].kind in SPLIT_KINDS for op in CASES)


@pytest.mark.parametrize("operator", sorted(CASES))
def test_reproduces_the_expected_split_kernel(operator):
    """The emitted pair, byte for byte, against ``expected_kernels/splitkv/<op>_splitkv8.py``."""
    assert_expected_tilelang_kernel(f"splitkv/{operator}_splitkv{SPLIT}", split_kernel_source(operator))


@pytest.mark.parametrize("operator", sorted(CASES))
def test_partial_structure(operator):
    spec, tiling = CASES[operator]
    partial, combine = build(operator)
    is_mla = isinstance(spec, MLASpec)
    heads = spec.num_qo_heads
    group = heads if is_mla else spec.group
    head_dim = tiling.get("head_dim") or tiling["ckv"]

    # The grid: the single kernel's axes plus the split, trailing.
    extent = partial.extent
    assert extent[-1] == str(SPLIT), extent
    if is_mla:
        assert extent == (spec.batch_dim, str(SPLIT))
    else:
        lead = (spec.batch_dim, str(spec.num_kv_heads))
        if spec.grid_order != ("batch", "head"):
            lead = tuple(reversed(lead))
        assert extent == lead + (str(SPLIT),)

    # Three raw stores in place of the divide and the lse: row sums first, then the values,
    # then the running max both are relative to; all in the accumulation precision and
    # indexed by split first. The max is one value per row and has the row sum's tile.
    outputs = [n for n in partial.args if isinstance(n, OutputNode)]
    assert [n.name for n in outputs] == ["OutputNode_0", "OutputNode_1", "OutputNode_2"]
    stores = [o.inputs[0].src_node for o in outputs]
    assert [n.name for n in stores] == ["partial_rowsum", "partial_value", "partial_max"]
    assert {n.name for n in partial.nodes if isinstance(n, Copy)} == {n.name for n in stores}
    assert all(type(n) is Copy for n in stores)
    assert not [n for n in partial.nodes
                if isinstance(n, (Div, Log2, Lse2)) or "log2" in n.name]
    assert all(str(o.get_dtype(0)) == "float32" for o in outputs)
    assert tuple(partial.tmap[outputs[0]].shape) == (SPLIT, spec.batch_dim, heads)
    assert tuple(partial.tmap[outputs[0]].tile_size) == (1, 1, group)
    assert tuple(partial.tmap[outputs[1]].shape) == (SPLIT, spec.batch_dim, heads, head_dim)
    assert tuple(partial.tmap[outputs[1]].tile_size) == (1, 1, group, head_dim)
    assert tuple(partial.tmap[outputs[2]].shape) == (SPLIT, spec.batch_dim, heads)
    assert tuple(partial.tmap[outputs[2]].tile_size) == (1, 1, group)
    assert [partial.tinfo_map[s] for s in stores] == [
        (f"OutputNode_{i}", "g") for i in range(3)]

    # The loop runs the block's own tile count, and every scalar it needs is derivable.
    loop = next(n for n in partial.nodes if isinstance(n, GraphNode))
    inner = loop.graph
    # What it stores is what the loop carried: the max as it stands after the split's last
    # tile, or at its finite start if the split ran none.
    carried = stores[2].inputs[0]
    assert isinstance(carried, AccumEdge) and carried.src_node is loop
    assert (carried.accum_node.name, carried.src_id) == ("exp_0_max", 2)
    assert carried.accum_node.accum_init == NEG_FLT_MAX
    assert inner.loop_trip == "my_trip"
    for name in ("tile0", "my_trip"):
        assert name in dict(inner.sargs) and name in partial.smap
    # Only a layer that reads ahead needs to know where its own range ends: the slab layer's
    # pipelined loop has no look-ahead of its own to bound.
    if inner.prefetch_nodes:
        assert "split_end" in dict(inner.sargs) and "split_end" in partial.smap
        assert "split_end" in inner.prefetch_guard and "valid_len" not in inner.prefetch_guard
    else:
        assert "split_end" not in dict(inner.sargs)
    if inner.window_nodes:
        assert "split_base" in inner.window_guard

    # The operator's inputs and the scale are exactly the single kernel's.
    inputs = [n.name for n in partial.args if isinstance(n, PlaceHolderNode)]
    assert f"PlaceHolder_{spec.index_input}" in inputs
    assert f"PlaceHolder_{spec.indptr_input}" in inputs
    if spec.kind == "paged_decode_slab":
        assert inputs[-1] == f"PlaceHolder_{spec.last_page_len_input}"
    assert partial.scalar_params == [(spec.scale_input, "float32")]
    assert partial.dyn_dims == [spec.batch_dim, spec.pages_dim, spec.indptr_dim, spec.index_dim]


@pytest.mark.parametrize("operator", sorted(CASES))
def test_combine_structure(operator):
    spec, tiling = CASES[operator]
    partial, combine = build(operator)
    is_mla = isinstance(spec, MLASpec)
    heads = spec.num_qo_heads
    group = heads if is_mla else spec.group
    head_dim = tiling.get("head_dim") or tiling["ckv"]
    comb = combine_of(spec, tiling)

    inputs = [n for n in combine.args if isinstance(n, PlaceHolderNode)]
    outputs = [n for n in combine.args if isinstance(n, OutputNode)]
    # The partials are the combine's inputs, in the partial's output order, whole along the
    # split axis so the loop tier can step through it.
    assert [n.name for n in inputs] == [
        f"PlaceHolder_{PARTIAL_ROWSUM}",
        f"PlaceHolder_{PARTIAL_VALUE}",
        f"PlaceHolder_{PARTIAL_MAX}",
    ]
    for ph, store in zip(inputs, partial.args[-3:]):
        assert tuple(combine.tmap[ph].shape) == tuple(partial.tmap[store].shape)
        assert str(ph.get_dtype(0)) == str(store.get_dtype(0)) == "float32"
    assert tuple(combine.tmap[inputs[0]].tile_size) == (SPLIT, 1, group)
    assert tuple(combine.tmap[inputs[1]].tile_size) == (SPLIT, 1, group, comb.cols)
    assert tuple(combine.tmap[inputs[2]].tile_size) == (SPLIT, 1, group)
    # The operator's real outputs, at the single kernel's tiles.
    assert [n.name for n in outputs] == ["OutputNode_0", "OutputNode_1"]
    assert str(outputs[0].get_dtype(0)) == "bfloat16"
    assert tuple(combine.tmap[outputs[0]].shape) == (spec.batch_dim, heads, head_dim)
    assert tuple(combine.tmap[outputs[0]].tile_size) == (1, group, comb.cols)
    assert tuple(combine.tmap[outputs[1]].tile_size) == (1, group)
    # No scale, no page table: the combine reads only the partials.
    assert combine.scalar_params == []
    assert combine.dyn_dims == [spec.batch_dim]
    # The grid: the partial's without its split axis, plus a column axis when a block owns
    # less than the head dim.
    n_cols = head_dim // comb.cols
    expected = (spec.batch_dim,) if is_mla else partial.extent[:-1]
    if n_cols > 1:
        expected = tuple(expected) + (str(n_cols),)
    assert combine.extent == expected
    # One partial per step here, so the two sums and the max over the splits are the loop
    # itself: elementwise accumulates, the max carried behind the sums.
    loop = next(n for n in combine.nodes if isinstance(n, GraphNode))
    inner = loop.graph
    assert inner.extent == SPLIT // comb.split_step and inner.loop_kind == "serial"
    carried = list(inner.accum_nodes)
    assert [n.name for n in carried] == ["rowsum_combine_0", "value_combine_0", "combine_max"]
    assert [type(n) for n in carried] == [Accumulate, Accumulate, AccumulateMax]
    assert {type(n) for n in carried} == {Accumulate, AccumulateMax} and len(carried) == 3
    assert [tuple(it.tile_size) for it in combine.tmap[loop]] == [
        (group,), (group, comb.cols), (group,)]
    # The single kernel's epilogue on the sums: the floored divide, and log2(L) + M.
    div = next(n for n in combine.nodes if n.name == "div_0")
    lse = next(n for n in combine.nodes if n.name == "lse_0")
    assert type(div) is Div and div.denom_floor == 1e-30 and type(lse) is Lse2
    assert not [n.name for n in combine.nodes if type(n) is Log2 or "log2" in n.name]
    assert [(e.accum_node.name, e.src_id) for e in div.inputs] == [
        ("value_combine_0", 1), ("rowsum_combine_0", 0)]
    assert [(e.accum_node.name, e.src_id) for e in lse.inputs] == [
        ("rowsum_combine_0", 0), ("combine_max", 2)]


@pytest.mark.parametrize("operator", sorted(CASES))
@pytest.mark.parametrize("step", [1, 2, SPLIT])
def test_combine_weighs_every_split_by_its_max(operator, step):
    """The combine is the online softmax one level up: a split where the single kernel has
    a KV tile. ``M`` is the running max of the splits' maxima, every slice is weighed by
    ``exp2(m_i - M)`` before it is summed, and the sums so far are rescaled when ``M`` moves.
    """
    spec, tiling = CASES[operator]
    group = spec.num_qo_heads if isinstance(spec, MLASpec) else spec.group
    cols = combine_of(spec, tiling).cols
    _, combine = build(operator, step=step)
    loop = next(n for n in combine.nodes if isinstance(n, GraphNode))
    inner = loop.graph
    by_name = {n.name: n for n in inner.nodes}
    rows, tile = ((step, group), (step, group, cols)) if step > 1 else ((group,), (group, cols))
    ph_d, ph_o, ph_m = (inner.binding[f"PlaceHolder_{p}"]
                        for p in (PARTIAL_ROWSUM, PARTIAL_VALUE, PARTIAL_MAX))
    assert inner.extent == SPLIT // step

    # M: carried from the most negative finite float32, a fold of the slice's split axis
    # where the slice has one and an elementwise max where it does not.
    m = by_name["combine_max"]
    assert type(m) is (ReduceMax if step > 1 else AccumulateMax)
    assert m.accum_init == NEG_FLT_MAX and m.out_dtype == "float32"
    assert m.running_max and m.inputs[0].src_node is ph_m
    assert tuple(inner.tmap[m].tile_size) == (group,)
    if step > 1:
        assert m.axis == 0 and tuple(m.dims) == rows
    # w = exp2(m_i - M): one weight per row of the slice, never above 1.
    shift, weight = by_name["combine_shift"], by_name["combine_weight"]
    assert type(shift) is Sub and [e.src_node for e in shift.inputs] == [ph_m, m]
    assert shift.broadcast == (1 if step > 1 else -1)
    assert type(weight) is Exp2 and weight.inputs[0].src_node is shift
    assert tuple(inner.tmap[shift].tile_size) == tuple(inner.tmap[weight].tile_size) == rows
    # The weighted slices are what is summed; the values take the weight along their rows.
    rowsum_w, value_w = by_name["rowsum_weighted_0"], by_name["value_weighted_0"]
    assert type(rowsum_w) is Mul and [e.src_node for e in rowsum_w.inputs] == [ph_d, weight]
    assert type(value_w) is Mul and [e.src_node for e in value_w.inputs] == [ph_o, weight]
    assert rowsum_w.broadcast == -1 and value_w.broadcast == 0
    assert tuple(inner.tmap[rowsum_w].tile_size) == rows
    assert tuple(inner.tmap[value_w].tile_size) == tile
    rowsum_c, value_c = by_name["rowsum_combine_0"], by_name["value_combine_0"]
    assert rowsum_c.inputs[0].src_node is rowsum_w and value_c.inputs[0].src_node is value_w
    assert {type(rowsum_c), type(value_c)} == {ReduceSum if step > 1 else Accumulate}
    if step > 1:
        assert rowsum_c.axis == value_c.axis == 0
    # The rolling update, declared on the tier exactly as in the partial's loop.
    assert [n.name for n in inner.lag_nodes] == ["combine_max_prev"]
    assert inner.lag_nodes[0].inputs[0].src_node is m
    alpha = by_name["combine_alpha"]
    assert type(alpha) is Exp2 and alpha.inputs[0].src_node.name == "combine_max_step"
    assert [e.src_node.name for e in by_name["combine_max_step"].inputs] == [
        "combine_max_prev", "combine_max"]
    assert {acc.name: f for acc, f in inner.accum_scale.items()} == {
        "rowsum_combine_0": alpha, "value_combine_0": alpha}
    assert [n.name for n in inner.accum_nodes] == [
        "rowsum_combine_0", "value_combine_0", "combine_max"]
    # The max slice has two readers, so it is loaded once into a fragment; the other two
    # slices are staged where the combine tiling says.
    assert inner.tinfo_map[ph_m] == ("PlaceHolder_MAXX", "r")
    assert inner.tinfo_map[ph_d][1] == inner.tinfo_map[ph_o][1] == "s"
    for name in ("combine_max", "combine_max_prev", "combine_max_step", "combine_alpha",
                 "combine_shift", "combine_weight", "rowsum_weighted_0", "value_weighted_0"):
        assert inner.tinfo_map[by_name[name]] == (f"t_{name}", "r"), name


def test_combine_with_a_wider_step_reduces():
    """A step above one folds a slice of the split axis with reductions, not accumulates:
    two sums and the max, all over the leading (split) axis."""
    _, combine = build("gqa_paged_decode_h32_kv8_d128_ps1", step=2)
    inner = next(n for n in combine.nodes if isinstance(n, GraphNode)).graph
    assert inner.extent == SPLIT // 2
    assert [type(n) for n in inner.accum_nodes] == [ReduceSum, ReduceSum, ReduceMax]
    assert all(n.axis == 0 for n in inner.accum_nodes)


def test_combine_text():
    """The combine's loop body, statement by statement, at one partial per step."""
    source = split_kernel_source("gqa_paged_decode_h32_kv8_d128_ps1")
    body = source[source.index("def combine_GraphNode_0_iter("):
                  source.index("def combine_GraphNode_0(")]
    flat = "".join(body.split()).replace(",)", ")")
    statements = [
        "T.copy(PlaceHolder_ROWSUM,PlaceHolder_ROWSUMM)",
        "PlaceHolder_MAXX=T.alloc_fragment((4),\"float32\")",
        "T.copy(PlaceHolder_MAX,PlaceHolder_MAXX)",
        "combine_ukernel_copy_1dims_rr(t_combine_max,t_combine_max_prev)",
        "combine_ukernel_accumulate_max_1dims_rr(PlaceHolder_MAXX,t_combine_max)",
        "combine_ukernel_sub_1dims_rrr(PlaceHolder_MAXX,t_combine_max,t_combine_shift)",
        "combine_ukernel_exp2_1dims_rr(t_combine_shift,t_combine_weight)",
        "combine_ukernel_mul_1dims_srr(PlaceHolder_ROWSUMM,t_combine_weight,t_rowsum_weighted_0)",
        "combine_ukernel_sub_1dims_rrr(t_combine_max_prev,t_combine_max,t_combine_max_step)",
        "combine_ukernel_exp2_1dims_rr(t_combine_max_step,t_combine_alpha)",
        "combine_ukernel_rescale_1dims_rr(t_combine_alpha,t_rowsum_combine_0)",
        "combine_ukernel_accumulate_1dims_rr(t_rowsum_weighted_0,t_rowsum_combine_0)",
        "T.copy(PlaceHolder_VALUE,PlaceHolder_VALUEE)",
        "combine_ukernel_mul_broadcast_0_2dims_srr(PlaceHolder_VALUEE,t_combine_weight,"
        "t_value_weighted_0)",
        "combine_ukernel_rescale_lead1_2dims_rr(t_combine_alpha,t_value_combine_0)",
        "combine_ukernel_accumulate_2dims_rr(t_value_weighted_0,t_value_combine_0)",
    ]
    at = [flat.index(s) for s in statements]
    assert at == sorted(at)
    # The max over the splits is carried: filled before the loop, never inside it; the two
    # sums start cleared. Every start value and every masked score is the finite sentinel.
    assert "T.fill(t_combine_max, -3.4028234663852886e38)" in source
    assert "T.clear(t_rowsum_combine_0)" in source and "T.clear(t_value_combine_0)" in source
    assert "infinity" not in source
    assert "B[i0] = T.max(B[i0], A[i0])" in source
    # The partial stores its max third, and the combine reads it under the same name.
    assert "split_ukernel_copy_1dims_rg(t_exp_0_max, OutputNode_2)" in source
    for name in ("Partial_ROWSUM", "Partial_VALUE", "Partial_MAX"):
        assert f"{name}_split: T.Tensor(" in source and f"{name}_combine: T.Tensor(" in source
    assert source.index("Partial_VALUE_combine:") < source.index("Partial_MAX_split:")
    # lse = log2(L) + M on the combined sums, the row sum unfloored.
    assert (
        "combine_ukernel_lse2_1dims_rrg(t_rowsum_combine_0, t_combine_max, OutputNode_1)"
        in source
    )
    assert "ukernel_log2" not in source and "T.exp2(A[m0, m1] * scale2)" not in source


@pytest.mark.parametrize("operator", sorted(CASES))
@pytest.mark.parametrize("step", [1, 2, SPLIT])
def test_split_pair_lowers_without_a_gpu(operator, step):
    """TileLang lowers the pair at every combine step: its layout inference has to place
    the weights and the rescale factor against both carried sums."""
    from tests._lowering import lower_to_cuda

    cuda = lower_to_cuda(split_kernel_source(operator, SPLIT, step))
    # base-2 exponentials, and the start of both carried maxima: exactly -FLT_MAX
    assert "exp2f(" in cuda and "-0x1.fffffep+127f" in cuda


@pytest.mark.parametrize(
    "bad",
    [
        dict(split=1),
        dict(split_step=3),
        dict(split_step=16),
        dict(cols=96),
        dict(cols=256),
        dict(loop_kind="unrolled"),
        dict(input_scope="r"),
    ],
)
def test_combine_tiling_refuses_what_it_cannot_build(bad):
    good = dict(split=SPLIT, split_step=1, cols=128, n_threads=128)
    with pytest.raises(AssertionError):
        CombineTiling(**{**good, **bad}).validate(128)


def test_split_and_combine_must_agree():
    spec, tiling = CASES["gqa_paged_decode_h32_kv8_d128_ps1"]
    with pytest.raises(AssertionError, match="splits"):
        build_split_decode(spec, dense_of(spec, tiling), 4, combine_of(spec, tiling, split=8))


def test_split_is_a_decode_transformation():
    spec, tiling = OPERATOR_CASES["gqa_paged_prefill_causal_h32_kv8_d128_ps1"]
    with pytest.raises(AssertionError, match="decode"):
        build_split_decode(spec, dense_of(spec, tiling), SPLIT, combine_of(spec, tiling))


# ------------------------------------------------------------- the tile-range arithmetic
def _evaluate(expr: str, **names) -> int:
    """Evaluate one emitted scalar expression with TileLang's intrinsics in Python."""
    env = {
        "T": type("T", (), {
            "ceildiv": staticmethod(lambda a, b: -(-a // b)),
            "min": staticmethod(min),
            "max": staticmethod(max),
        }),
        **names,
    }
    return eval(expr, {"__builtins__": {}}, env)  # noqa: S307 - our own emitted text


@pytest.mark.parametrize("split", [2, 3, 4, 8, 16])
@pytest.mark.parametrize("bn", [16, 64])
def test_every_tile_is_owned_by_exactly_one_split(split, bn):
    """The three range scalars deal every KV tile out once, whatever the request length."""
    scalars = _split_range_scalars(split, bn, lambda b: "valid_len", lambda build: build)
    for valid_len in [0, 1, bn - 1, bn, bn + 1, 3 * bn, 7 * bn + 5, 64 * bn, 1000 * bn + 1]:
        n_tiles = -(-valid_len // bn)
        owned = []
        for s in range(split):
            tile0 = _evaluate(scalars["tile0"]("b", "h", str(s)), valid_len=valid_len)
            trip = _evaluate(scalars["my_trip"]("b", "h", str(s)), valid_len=valid_len)
            end = _evaluate(scalars["split_end"]("b", "h", str(s)), valid_len=valid_len)
            assert trip >= 0
            owned += range(tile0, tile0 + trip)
            # The look-ahead bound is the token past this block's last tile, clamped.
            if trip:
                assert end == min(valid_len, bn * (tile0 + trip)) or end == min(
                    valid_len, bn * (tile0 + -(-n_tiles // split))
                )
                assert bn * (tile0 + trip - 1) < end <= valid_len
        assert sorted(owned) == list(range(n_tiles)), (split, bn, valid_len, owned)
