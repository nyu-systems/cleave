"""macro_key(): two ops with the same key emit the same macro name and body."""

import math
from collections import defaultdict

import pytest

pytest.importorskip("tilelang")

from cleave.ir.graph import GraphNode, TensorOp
from cleave.ir.ops import (
    Add,
    CausalMaskedExp,
    ConstFill,
    Copy,
    Div,
    Exp,
    IndexGather,
    IndptrTileScan,
    Log2,
    MacroOp,
    MaskedDiv,
    MaskedExp,
    MaskedLog2,
    Matmul,
    ReduceMax,
    Mul,
    PagedGather,
    RangeGather,
    ReduceSum,
    RowGather,
    RSqrt,
    Silu,
    Sqrt,
    Sub,
)
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
    for cls in (Silu, Exp, RSqrt, Sqrt, Log2, Copy):
        yield cls((16, 32))
        yield cls((16, 32), name="other", dtype="float16")
        yield cls((2, 16, 32), n_batch_dims=0)
    yield MaskedDiv((16, 32), row_group=2)
    yield MaskedLog2((16, 32), row_group=2, fuse_dst=False)
    yield MaskedExp((16, 32), 0.5)
    yield MaskedExp((16, 32), None, scalar_args=("a", "b", "c"))
    yield CausalMaskedExp((16, 32), 0.5, row_group=2)
    for policy in ("rows", "cols"):
        for dtype in ("float16", "bfloat16"):
            yield Matmul((64, 64, 32), warp_policy=policy, dtype=dtype)
    yield ReduceSum((16, 32), axis=-1)
    yield ReduceSum((16, 32), axis=1)
    yield ReduceMax((16, 32), axis=-1)
    yield ReduceMax((16, 32), axis=1, name="other")
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
        n_in = len(getattr(ops[0], "inputs", ()))
        for scope in ("g" * n_in + "r", "s" * n_in + "r", "r" * (n_in + 1), "g" * n_in + "s"):
            for clear in (False, True):
                emitted = {_emit(op, scope, clear) for op in ops}
                assert len(emitted) == 1, (key, scope, clear)


def test_key_tells_apart_literals_and_policies():
    assert Mul((16, 32), const=1).macro_key() != Mul((16, 32), const=1.0).macro_key()
    a, b = (Matmul((64, 64, 32), warp_policy=p) for p in ("rows", "cols"))
    assert a.macro_key() != b.macro_key()
    assert ReduceSum((16, 32)).macro_key() != ReduceMax((16, 32)).macro_key()
