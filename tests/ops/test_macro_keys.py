"""macro_key(): two ops with the same key emit the same macro name and body."""

import math
from collections import defaultdict

import pytest

pytest.importorskip("tilelang")

from cleave.ir.graph import GraphNode, TensorOp
from cleave.ir.ops import (
    NEG_FLT_MAX,
    Accumulate,
    AccumulateMax,
    Add,
    CausalMaskedExp,
    CausalMaskedScore,
    ConstFill,
    Copy,
    Div,
    Exp,
    Exp2,
    IndexGather,
    IndptrTileScan,
    Log2,
    Lse2,
    MacroOp,
    MaskedDiv,
    MaskedExp,
    MaskedLog2,
    MaskedLse2,
    MaskedScore,
    Matmul,
    ReduceMax,
    Mul,
    PagedGather,
    RangeGather,
    ReduceSum,
    Rescale,
    RowGather,
    RSqrt,
    Silu,
    Sqrt,
    Sub,
)
from cleave.ir.update_rule import Streaming
from tests.graph.handwritten._cases import CASES


def _variants():
    for cls in (Add, Sub, Mul, Div):
        for const in (None, 1, 1.0, -math.inf):
            for name in ("a", "b"):
                yield cls((16, 32), const=const, name=name)
        yield cls((16, 32), broadcast=0)
        op = cls((2, 16, 32), broadcast=1, dtype="float16")
        op.resolve_operand_layout = "rhs"
        yield op
    yield Div((16, 32), const=2.0, const_on_left=True, denom_floor=1e-6)
    for cls in (Silu, Exp, Exp2, RSqrt, Sqrt, Log2, Copy):
        yield cls((16, 32))
        yield cls((16, 32), name="other", dtype="float16")
        yield cls((2, 16, 32), n_batch_dims=0)
    yield MaskedDiv((16, 32), row_group=2)
    yield MaskedLog2((16, 32), row_group=2, fuse_dst=False)
    yield MaskedExp((16, 32), 0.5)
    yield MaskedExp((16, 32), None, scalar_args=("a", "b", "c"))
    yield CausalMaskedExp((16, 32), 0.5, row_group=2)
    # the exact paged softmax: masked score, shifted exp, two-operand lse
    for name in ("a", "b"):
        yield MaskedExp((16, 32), None, scalar_args=("a", "b"), shift=True, name=name)
        yield MaskedScore((16, 32), scalar_args=("a", "b", "c"), name=name)
        yield Lse2((16,), name=name)
    yield MaskedExp((2, 16, 32), None, scalar_args=("a", "b"), shift=True, n_batch_dims=0)
    yield CausalMaskedExp((16, 32), None, scalar_args=("a", "b", "c"), row_group=2, shift=True)
    yield CausalMaskedExp((16, 32), None, scalar_args=("a", "b", "c"), shift=True)
    yield MaskedScore((16, 32))
    yield CausalMaskedScore((16, 32), row_group=2)
    yield CausalMaskedScore((16, 32))
    yield Lse2((16, 32))
    yield MaskedLse2((16,))
    yield MaskedLse2((16, 32), row_group=2, fuse_dst=False)
    for policy in ("rows", "cols"):
        for dtype in ("float16", "bfloat16"):
            yield Matmul((64, 64, 32), warp_policy=policy, dtype=dtype)
    yield ReduceSum((16, 32), axis=-1)
    yield ReduceSum((16, 32), axis=1)
    yield ReduceMax((16, 32), axis=-1)
    yield ReduceMax((16, 32), axis=1, name="other")
    yield ReduceMax((16, 32), axis=-1, name="carried", accum_init=NEG_FLT_MAX)
    # the split combine: the elementwise running max next to the elementwise sum, and the
    # weight broadcast over the last axis of a three-dim tile next to the two-dim form
    for dims in ((16,), (16, 32)):
        yield Accumulate(dims)
        yield AccumulateMax(dims)
        yield AccumulateMax(dims, name="carried", accum_init=NEG_FLT_MAX)
    for cls in (Mul, Sub):
        yield cls((2, 16, 32), broadcast=0, n_batch_dims=0)
        yield cls((2, 16, 32), broadcast=0, n_batch_dims=0, name="other")
        yield cls((2, 16), broadcast=1, n_batch_dims=0)
    marked = ReduceMax((16, 32), axis=-1, name="marked")
    marked.streaming = Streaming(("rowsum",), (("rowsum", ("exp", ("sub", "m", "m'"))),), "k")
    yield marked
    for rank, lead in ((1, 1), (2, 1), (2, 1), (2, 2), (3, 1)):
        yield Rescale(rank, lead)
    yield PagedGather((16, 64), source_rows=128, index_extent=8, page_size=4)
    yield IndexGather((16,), index_extent=8, row_stride=1, clamp_index=7)
    yield RowGather((16, 64), source_rows=128, n_valid=4)
    yield RangeGather((16, 64), source_rows=128, row_group=2, src_dims=3)
    yield ConstFill((16, 32), -math.inf)
    yield IndptrTileScan(16)


def _graph_ops(graph):
    for node in list(graph.nodes) + list(graph.else_nodes):
        if isinstance(node, GraphNode):
            yield from _graph_ops(node.graph)
        elif isinstance(node, (TensorOp, MacroOp)):
            yield node
    for scan in graph.prologue:
        yield scan.op


def _case_ops():
    for case in CASES.values():
        graphs = case.graph()
        for graph in graphs if isinstance(graphs, list) else [graphs]:
            yield from _graph_ops(graph)


def _emit(op, mem_scope, clear):
    try:
        kw = dict(mem_scope=mem_scope, clear=clear)
        return op.get_macro_name(**kw), op.gen_macro(**kw)
    except (AssertionError, NotImplementedError, ValueError):
        return None


def test_equal_keys_mean_equal_macros():
    groups = defaultdict(list)
    for op in list(_variants()) + list(_case_ops()):
        groups[op.macro_key()].append(op)
    assert len(groups) > 40
    for key, ops in groups.items():
        # Rescale is no node (no `inputs`); its one operand besides the tile is the factor.
        n_in = 1 if isinstance(ops[0], Rescale) else len(getattr(ops[0], "inputs", ()))
        for scope in ("g" * n_in + "r", "s" * n_in + "r", "r" * (n_in + 1), "g" * n_in + "s"):
            for clear in (False, True):
                emitted = {_emit(op, scope, clear) for op in ops}
                assert len(emitted) == 1, (key, scope, clear)


def test_key_tells_apart_literals_and_policies():
    assert Mul((16, 32), const=1).macro_key() != Mul((16, 32), const=1.0).macro_key()
    a, b = (Matmul((64, 64, 32), warp_policy=p) for p in ("rows", "cols"))
    assert a.macro_key() != b.macro_key()
    assert ReduceSum((16, 32)).macro_key() != ReduceMax((16, 32)).macro_key()
    assert Accumulate((16,)).macro_key() != AccumulateMax((16,)).macro_key()
    assert Rescale(2, 1).macro_key() != Rescale(2, 2).macro_key()
    assert Rescale(2, 1).macro_key() != Rescale(3, 1).macro_key()


def test_shift_and_mask_are_part_of_the_key():
    plain = MaskedExp((16, 32), None, scalar_args=("tile_base", "valid_len", "scale2"))
    shifted = MaskedExp((16, 32), None, scalar_args=("tile_base", "valid_len"), shift=True)
    assert "shift" in MaskedExp._macro_attrs
    assert plain.macro_key() != shifted.macro_key()
    assert _emit(plain, "rr", True)[0] != _emit(shifted, "rrr", True)[0]
    causal = CausalMaskedExp(
        (16, 32), None, scalar_args=("tile_base", "valid_len", "diag"), shift=True)
    assert causal.macro_key() != shifted.macro_key()
    assert _emit(causal, "rrr", True)[0] != _emit(shifted, "rrr", True)[0]
    assert MaskedScore((16, 32)).macro_key() != CausalMaskedScore((16, 32)).macro_key()
    assert Lse2((16,)).macro_key() != MaskedLse2((16,)).macro_key()
    assert Exp((16,)).macro_key() != Exp2((16,)).macro_key()
    assert Lse2((16,)).macro_key() != Log2((16,)).macro_key()


def test_rescale_is_emitted_for_register_operands():
    """The equal-keys loop skips scopes an op refuses; make sure it does not skip them all."""
    a, b = Rescale(2, 1), Rescale(2, 1)
    assert a.macro_key() == b.macro_key()
    assert _emit(a, "rr", False) == _emit(b, "rr", False) is not None


def test_annotations_are_not_part_of_the_macro_key():
    """A `Streaming` record (the update rule of a max) and a per-instance `accum_init` change
    what is built around the op, never the op's own macro: marked and plain share one name
    and one body."""
    plain, marked = ReduceMax((16, 32)), ReduceMax((16, 32), name="marked")
    marked.streaming = Streaming(("rowsum",), (("rowsum", ("exp", ("sub", "m", "m'"))),), "k")
    assert marked.macro_key() == plain.macro_key()
    assert _emit(marked, "rr", True) == _emit(plain, "rr", True) is not None
    assert "streaming" not in ReduceMax._macro_attrs

    carried = ReduceMax((16, 32), accum_init=NEG_FLT_MAX)
    assert carried.macro_key() == ReduceMax((16, 32)).macro_key()
    assert "accum_init" not in ReduceMax._macro_attrs
