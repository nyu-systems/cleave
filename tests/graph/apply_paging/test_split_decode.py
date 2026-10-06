"""What the paging pass builds for split-KV decode: the pair, its structure, its goldens.

Split-KV is the decode layers' answer to a grid of fewer blocks than the GPU has SMs: the
KV window of one (request, kv head) is dealt out over ``split`` blocks that each store a
raw partial numerator and denominator, and a second kernel sums the partials and applies
the epilogue. The pass builds the pair from the same tiling decisions it reads for the
single kernel plus the combine's own (``CombineTiling``), and codegen emits one prim_func
of two kernels sharing the partial tensors.

Three kinds of assertion here. The structural half pins the graph the pass hands to
codegen -- the extra grid axis, the two ``Copy`` stores, the combine's grid and its two
reductions. The goldens pin the emitted text, which is what the measured numbers belong to
(``tests/graph/handwritten/test_splitkv_decode.py`` runs them). The arithmetic half checks
the tile-range formulas the partial kernel is built on by evaluating the emitted expressions
in Python: every KV tile of every request must be owned by exactly one split.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tilelang")

from tests.graph.apply_paging._test_utils import OPERATOR_CASES
from tests.graph.handwritten._test_utils import assert_expected_tilelang_kernel
from cleave.codegen import TilelangCodeGen
from cleave.graph import GraphNode, OutputNode, PlaceHolderNode
from cleave.ops import Accumulate, Copy, Div, Log2, Reduction
from cleave.paging import (
    PARTIAL_ROWSUM,
    PARTIAL_VALUE,
    SPLIT_KINDS,
    CombineTiling,
    DenseAttention,
    DenseMLA,
    MLASpec,
    _split_range_scalars,
    build_split_decode,
)

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

    # Two raw stores in place of the divide and the log: row sums first, then the values,
    # both in the accumulation precision and indexed by split first.
    outputs = [n for n in partial.args if isinstance(n, OutputNode)]
    assert [n.name for n in outputs] == ["OutputNode_0", "OutputNode_1"]
    stores = [n for n in partial.nodes if isinstance(n, Copy)]
    assert {n.name for n in stores} == {"partial_rowsum", "partial_value"}
    assert not [n for n in partial.nodes if isinstance(n, (Div, Log2))]
    assert all(str(o.get_dtype(0)) == "float32" for o in outputs)
    assert tuple(partial.tmap[outputs[0]].shape) == (SPLIT, spec.batch_dim, heads)
    assert tuple(partial.tmap[outputs[0]].tile_size) == (1, 1, group)
    assert tuple(partial.tmap[outputs[1]].shape) == (SPLIT, spec.batch_dim, heads, head_dim)
    assert tuple(partial.tmap[outputs[1]].tile_size) == (1, 1, group, head_dim)

    # The loop runs the block's own tile count, and every scalar it needs is derivable.
    inner = next(n for n in partial.nodes if isinstance(n, GraphNode)).graph
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
    ]
    assert tuple(combine.tmap[inputs[0]].shape) == tuple(partial.tmap[partial.args[-2]].shape)
    assert tuple(combine.tmap[inputs[1]].shape) == tuple(partial.tmap[partial.args[-1]].shape)
    assert tuple(combine.tmap[inputs[1]].tile_size) == (SPLIT, 1, group, comb.cols)
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
    # One partial per step here, so the sum is an elementwise accumulate across the loop.
    inner = next(n for n in combine.nodes if isinstance(n, GraphNode)).graph
    assert inner.extent == SPLIT // comb.split_step and inner.loop_kind == "serial"
    sums = [n for n in inner.accum_nodes]
    assert {type(n) for n in sums} == {Accumulate} and len(sums) == 2
    assert isinstance(next(n for n in combine.nodes if n.name == "div_0"), Div)
    assert isinstance(next(n for n in combine.nodes if n.name == "log2_0"), Log2)


def test_combine_with_a_wider_step_reduces():
    """A step above one folds a slice of the split axis with a reduction, not an accumulate."""
    _, combine = build("gqa_paged_decode_h32_kv8_d128_ps1", step=2)
    inner = next(n for n in combine.nodes if isinstance(n, GraphNode)).graph
    assert inner.extent == SPLIT // 2
    assert {type(n) for n in inner.accum_nodes} == {Reduction}
    assert all(n.axis == 0 for n in inner.accum_nodes)


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
