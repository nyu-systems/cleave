"""Superoptimizer graph structure (``KernelGraph`` or its op dicts) -> Cleave ``Graph``.

Each op is built by a rule from ``_RULES``, keyed on ``(kind, form)``; ``fuse_rsqrt_ops``
is a rewrite pass over the ops that runs before conversion.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import tilelang  # noqa: F401
import tvm
from tvm import te

from ..core.schema import KernelGraph, KGOp, KGTensor, parse_op_type
from ..ir.graph import Graph, OutputNode, PlaceHolderNode
from ..ir.ops import (Add, Div, Exp, Matmul, ReduceMax, Mul, ReduceSum, RSqrt, Silu, SplitView,
                   Sqrt, Sub)
from ..ir.ops.base import wire_input
from ..ir.graph_utils import find_topo_sort, gather_all_nodes

_MAX_AXIS = 4
IndexedOps = List[Tuple[int, KGOp]]  # (index in the source structure, op); names use the index


def _infer_broadcast_index(dims: Tuple[int, ...], other_shape: Tuple[int, ...]) -> int:
    if tuple(dims) == tuple(other_shape):
        return -1
    if len(dims) < 2:
        return -1
    batch_dims = dims[:-2]
    m_dim, n_dim = dims[-2], dims[-1]
    if tuple(other_shape) == tuple(batch_dims + (m_dim,)):
        return 0
    if tuple(other_shape) == tuple(batch_dims + (n_dim,)):
        return 1
    return -1


def _infer_matmul_config(
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
    transpose_a: bool,
    transpose_b: bool,
) -> Tuple[Tuple[int, ...], bool, bool, Tuple[int, ...], Tuple[int, ...]]:
    def _normalize_input_batch_dims(
        input_batch_dims: Tuple[int, ...],
        output_batch_dims: Tuple[int, ...],
        input_name: str,
    ) -> Tuple[int, ...]:
        if len(input_batch_dims) > len(output_batch_dims):
            raise ValueError(
                f"Invalid matmul input shapes: input {input_name} has batch rank "
                f"{len(input_batch_dims)} > output batch rank {len(output_batch_dims)}"
            )
        if not output_batch_dims:
            return ()

        n_input = len(input_batch_dims)
        n_output = len(output_batch_dims)
        if n_input == 0:
            return (1,) * n_output

        compatible_candidates = []

        def _build_candidate(
            selected_positions: Tuple[int, ...],
        ) -> Optional[Tuple[int, ...]]:
            candidate = [1] * n_output
            for input_idx, out_idx in enumerate(selected_positions):
                candidate[out_idx] = int(input_batch_dims[input_idx])

            for out_dim, in_dim in zip(output_batch_dims, candidate):
                out_dim_i = int(out_dim)
                in_dim_i = int(in_dim)
                if in_dim_i <= 0 or out_dim_i % in_dim_i != 0:
                    return None
            return tuple(candidate)

        def _search(
            next_out_idx: int,
            next_input_idx: int,
            selected_positions: Tuple[int, ...],
        ) -> None:
            remaining_inputs = n_input - next_input_idx
            remaining_outputs = n_output - next_out_idx
            if remaining_outputs < remaining_inputs:
                return
            if next_input_idx == n_input:
                candidate = _build_candidate(selected_positions)
                if candidate is None:
                    return
                exact_matches = sum(
                    int(output_batch_dims[out_idx]) == int(input_batch_dims[input_idx])
                    for input_idx, out_idx in enumerate(selected_positions)
                )
                movement_cost = sum(
                    abs(out_idx - input_idx)
                    for input_idx, out_idx in enumerate(selected_positions)
                )
                compatible_candidates.append(
                    (
                        exact_matches,
                        -movement_cost,
                        selected_positions,
                        candidate,
                    )
                )
                return

            max_out_idx = n_output - remaining_inputs
            for out_idx in range(next_out_idx, max_out_idx + 1):
                _search(
                    out_idx + 1,
                    next_input_idx + 1,
                    selected_positions + (out_idx,),
                )

        _search(0, 0, ())
        if compatible_candidates:
            compatible_candidates.sort(reverse=True)
            return compatible_candidates[0][3]

        if len(input_batch_dims) < len(output_batch_dims):
            padded_dims = (1,) * (
                len(output_batch_dims) - len(input_batch_dims)
            ) + tuple(int(v) for v in input_batch_dims)
        else:
            padded_dims = tuple(int(v) for v in input_batch_dims)

        for out_dim, in_dim in zip(output_batch_dims, padded_dims):
            out_dim_i = int(out_dim)
            in_dim_i = int(in_dim)
            if in_dim_i <= 0 or out_dim_i % in_dim_i != 0:
                raise ValueError(
                    f"Invalid matmul input shapes: input {input_name} batch dim "
                    f"{in_dim_i} is incompatible with output batch dim {out_dim_i}"
                )
        return padded_dims

    # Support matvec-style broadcast: (K,) x (K, N) -> (N,)
    if len(a_shape) == 1 and len(b_shape) >= 2:
        batch_dims = b_shape[:-2]
        k_dim = b_shape[-1] if transpose_b else b_shape[-2]
        n_dim = b_shape[-2] if transpose_b else b_shape[-1]
        if a_shape[-1] != k_dim:
            raise ValueError(f"Invalid matmul input shapes: {a_shape}, {b_shape}")
        m_dim = 1
        dims = (*batch_dims, m_dim, n_dim, k_dim)
        return dims, transpose_a, transpose_b, batch_dims, batch_dims
    if len(a_shape) >= 2 and len(b_shape) == 1:
        batch_dims = a_shape[:-2]
        m_dim = a_shape[-1] if transpose_a else a_shape[-2]
        k_dim = a_shape[-2] if transpose_a else a_shape[-1]
        if b_shape[-1] != k_dim:
            raise ValueError(f"Invalid matmul input shapes: {a_shape}, {b_shape}")
        n_dim = 1
        dims = (*batch_dims, m_dim, n_dim, k_dim)
        return dims, transpose_a, transpose_b, batch_dims, batch_dims
    if len(a_shape) < 2 or len(b_shape) < 2:
        raise ValueError(f"Invalid matmul input shapes: {a_shape}, {b_shape}")

    k_dim = a_shape[-2] if transpose_a else a_shape[-1]
    b_k_dim = b_shape[-1] if transpose_b else b_shape[-2]
    if k_dim != b_k_dim:
        raise ValueError(f"Invalid matmul input shapes: {a_shape}, {b_shape}")

    if len(out_shape) >= 2:
        batch_dims = out_shape[:-2]
        m_dim, n_dim = out_shape[-2], out_shape[-1]
    else:
        batch_dims = a_shape[:-2]
        m_dim = a_shape[-1] if transpose_a else a_shape[-2]
        n_dim = b_shape[-2] if transpose_b else b_shape[-1]
    a_batch_dims = _normalize_input_batch_dims(a_shape[:-2], batch_dims, "A")
    b_batch_dims = _normalize_input_batch_dims(b_shape[:-2], batch_dims, "B")
    dims = (*batch_dims, m_dim, n_dim, k_dim)
    return dims, transpose_a, transpose_b, a_batch_dims, b_batch_dims


def _dims(t: KGTensor) -> Tuple[int, ...]:
    return t.dims or t.dim_padding  # num_dims <= 0 keeps every dim


class _Ctx:
    """Conversion state: producer node, known shape and metadata dtype per tensor guid."""

    def __init__(self, ops: IndexedOps, dtype, accum_dtype, out_dtype, symbol_map, fuse_rsqrt):
        self.dtype, self.accum_dtype, self.out_dtype = dtype, accum_dtype, out_dtype
        self.fuse_rsqrt = fuse_rsqrt
        # the constant dim '1' always maps to 1
        self.symbol_map = None if symbol_map is None else {**symbol_map, 1: 1}
        self.nodes: Dict[int, Tuple[object, int]] = {}
        self.shapes: Dict[int, Tuple[int, ...]] = {}
        self.dtypes: Dict[int, str] = {}
        self.consumers: Dict[int, List[str]] = {}
        self.outputs: List[OutputNode] = []
        for _, op in ops:
            for t in op.inputs:
                if t.dtype is not None:
                    self.dtypes[t.guid] = str(t.dtype)
                self.consumers.setdefault(t.guid, []).append(op.kind)
            for t in op.outputs:
                if t.dtype is not None:
                    self.dtypes[t.guid] = str(t.dtype)

    def shape(self, t: KGTensor) -> Tuple[int, ...]:
        dims = tuple(int(v) for v in _dims(t))
        if not self.symbol_map:
            return dims
        for v in dims:
            assert v in self.symbol_map, (
                f"Dimension {v} not found in symbol_map while mapping shape {dims}. "
                f"Available symbol_map keys: {sorted(self.symbol_map.keys())}"
            )
        return tuple(self.symbol_map[v] for v in dims)

    def known_shape(self, t: KGTensor) -> Tuple[int, ...]:
        shape = self.shapes.get(t.guid)
        return shape if shape is not None else self.shape(t)

    def producer(self, guid: int):
        return self.nodes.get(guid, (None, None))[0]

    def out_dtype_of(self, op: KGOp) -> str:
        """Metadata dtype of the op's output, else ``out_dtype`` if only outputs read it."""
        if not op.outputs:
            return self.accum_dtype
        guid = op.outputs[0].guid
        if guid in self.dtypes:
            return self.dtypes[guid]
        consumers = self.consumers.get(guid, [])
        if consumers and all(k == "output" for k in consumers):
            return self.out_dtype
        return self.accum_dtype

    def in_dtype(self, guid: int) -> str:
        src, src_id = self.nodes.get(guid, (None, 0))
        if src is None:
            return self.dtypes.get(guid, self.dtype)
        for attr in ("out_dtype", "dtype"):
            if getattr(src, attr, None) is not None:
                return str(getattr(src, attr))
        dtypes = getattr(src, "_dtypes", None)
        if dtypes and src_id < len(dtypes) and dtypes[src_id] is not None:
            return str(dtypes[src_id])
        return self.dtype

    def common_dtype(self, *guids: int) -> str:
        dtypes = [self.in_dtype(g) for g in guids]
        if not dtypes:
            return self.dtype
        return dtypes[0] if len(set(dtypes)) == 1 else self.accum_dtype

    def batch_dims(self, guid: int, out_dims) -> Optional[int]:
        """Keep a producer's all-but-innermost batch dims (a ReduceSum tiles its rank-2
        (batch, seq) output as a 1-D buffer, so same-rank consumers must match); None: default."""
        src, src_shape = self.producer(guid), self.shapes.get(guid)
        if src is None or src_shape is None:
            return None
        nbd = getattr(src, "n_batch_dims", None)
        if nbd is None:
            return None
        rank = len(src_shape)
        if rank >= 2 and nbd == rank - 1 and len(out_dims) == rank:
            return len(out_dims) - 1
        return None

    def placeholder(self, t: KGTensor) -> PlaceHolderNode:
        node = PlaceHolderNode(name=f"input_{t.guid}")
        shape = self.shape(t)
        dtype = self.dtype if t.dtype is None else str(t.dtype)
        node.set_shape(shape)
        node.set_dtype(tvm.DataType(dtype))
        self.nodes[t.guid] = (node, 0)
        self.shapes[t.guid] = shape
        self.dtypes[t.guid] = dtype
        return node

    def wire(self, node, inputs: Sequence[KGTensor]) -> None:
        for dst_id, t in enumerate(inputs):
            if t.guid not in self.nodes:
                self.placeholder(t)
            src, src_id = self.nodes[t.guid]
            wire_input(src, node, dst_id, src_id=src_id)
            if isinstance(src, PlaceHolderNode) and dst_id < len(node.input_buffers):
                self.shapes[t.guid] = tuple(src.get_shape(src_id))

    def register(self, node, outputs: Sequence[KGTensor]) -> None:
        for i, t in enumerate(outputs):
            self.nodes[t.guid] = (node, i)
            self.shapes[t.guid] = tuple(node.get_shape(i))


Built = Tuple[Optional[object], Sequence[KGTensor]]  # (node or None, the inputs to wire)
Rule = Callable[[_Ctx, int, KGOp], Built]


def _input(ctx: _Ctx, idx: int, op: KGOp) -> Built:
    if op.outputs:
        ctx.placeholder(op.outputs[0])
    return None, ()


def _output(ctx: _Ctx, idx: int, op: KGOp) -> Built:
    if op.inputs:
        guid = op.inputs[0].guid
        if guid not in ctx.nodes:
            raise ValueError(f"Output tensor guid {guid} has no producer.")
        ctx.outputs.append(OutputNode(ctx.producer(guid), dtype=ctx.dtypes.get(guid, ctx.out_dtype)))
    return None, ()


def _split(ctx: _Ctx, idx: int, op: KGOp) -> Built:
    if len(op.inputs) != 1 or len(op.outputs) != 1:
        raise ValueError(f"{op.op_type} expects one input and one output tensor.")
    axis, t = op.axis, op.inputs[0]
    src_shape = ctx.known_shape(t)
    out_shape = ctx.shape(op.outputs[0])
    if len(out_shape) != len(src_shape) + 1:
        raise ValueError(f"Invalid split output rank: src {src_shape}, out {out_shape}, axis={axis}")
    factor, supported = None, []
    for position in (0, len(src_shape) - 2):
        f = int(out_shape[position])
        if f <= 0 or src_shape[axis] % f != 0:
            continue
        candidate = list(src_shape)
        candidate[axis] = src_shape[axis] // f
        candidate.insert(position, f)
        supported.append(tuple(candidate))
        if tuple(candidate) == tuple(out_shape):
            factor = f
            break
    if factor is None:
        raise ValueError(
            "Unsupported split layout: expected one of the supported raw split layouts; "
            f"src={src_shape}, axis={axis}, out={out_shape}, supported={supported}"
        )
    node = SplitView(src_shape, axis=axis, factor=factor, name=f"split_{axis}_{idx}",
                     dtype=ctx.in_dtype(t.guid), split_dim_is_batch=True)
    return node, op.inputs


def _matmul(ctx: _Ctx, idx: int, op: KGOp) -> Built:
    a, b = op.inputs[0], op.inputs[1]
    dims, ta, tb, a_batch, b_batch = _infer_matmul_config(
        ctx.known_shape(a), ctx.known_shape(b), ctx.shape(op.outputs[0]),
        bool(op.attrs.get("trans_A", False)), bool(op.attrs.get("trans_B", False)))
    node = Matmul(dims, input_batch_dims_A=a_batch, input_batch_dims_B=b_batch, name=f"mma_{idx}",
                  transpose_A=ta, transpose_B=tb, warp_policy="rows", dtype=ctx.dtype,
                  out_dtype=ctx.accum_dtype)
    return node, op.inputs


def _reduction(cls) -> Rule:
    def rule(ctx: _Ctx, idx: int, op: KGOp) -> Built:
        t, axis = op.inputs[0], op.axis
        shape = ctx.known_shape(t)
        # Named by the pinned macro stem (cls.kind): node names reach generated kernels.
        node = cls(shape, axis=-1 if axis == len(shape) - 1 else axis,
                   name=f"{cls.kind}_{axis}_{idx}", dtype=ctx.in_dtype(t.guid),
                   out_dtype=ctx.accum_dtype)
        return node, op.inputs
    return rule


def _unary(cls) -> Rule:
    def rule(ctx: _Ctx, idx: int, op: KGOp) -> Built:
        t = op.inputs[0]
        shape = ctx.known_shape(t)
        node = cls(shape, name=f"{op.kind}_{idx}", dtype=ctx.in_dtype(t.guid),
                   out_dtype=ctx.out_dtype_of(op), n_batch_dims=ctx.batch_dims(t.guid, shape))
        return node, op.inputs
    return rule


def _rsqrt(ctx: _Ctx, idx: int, op: KGOp) -> Built:
    t = op.inputs[0]
    shape = ctx.known_shape(t)
    nbd = ctx.batch_dims(t.guid, shape)
    # dims typed as the fused Sqrt's input buffer shape
    node = RSqrt(tuple(te.placeholder(shape).shape), name=f"rsqrt_{idx}", dtype=ctx.accum_dtype,
                 out_dtype=ctx.out_dtype_of(op), n_batch_dims=nbd)
    return node, op.inputs


def _is_reciprocal(op: KGOp) -> bool:
    scalar = op.attrs.get("scalar")
    return (op.kind == "div" and scalar is not None
            and bool(op.attrs.get("scalar_on_left", False)) and float(scalar) == 1.0)


def _scalar(cls) -> Rule:
    def rule(ctx: _Ctx, idx: int, op: KGOp) -> Built:
        out_dtype = ctx.out_dtype_of(op)
        if not op.inputs:
            raise ValueError(f"{op.op_type} with scalar requires one input tensor.")
        t = op.inputs[0]
        shape = ctx.known_shape(t)
        kw, dtype = {}, ctx.in_dtype(t.guid)
        if cls is Div:
            kw["const_on_left"] = bool(op.attrs.get("scalar_on_left", False))
            if ctx.fuse_rsqrt and _is_reciprocal(op):  # a 1/x left unfused
                dtype = ctx.accum_dtype
        node = cls(shape, const=float(op.attrs["scalar"]), name=f"{op.kind}_{idx}", dtype=dtype,
                   out_dtype=out_dtype, n_batch_dims=ctx.batch_dims(t.guid, shape), **kw)
        return node, op.inputs
    return rule


def _binary(cls, out_is_in_dtype: bool = False, silu_first: bool = False) -> Rule:
    def rule(ctx: _Ctx, idx: int, op: KGOp) -> Built:
        ins = list(op.inputs)
        if silu_first:  # deterministic operand order for swiglu-like patterns
            if len(ins) < 2:
                raise ValueError(f"{op.op_type} requires two input tensors.")
            if isinstance(ctx.producer(ins[1].guid), Silu) and not isinstance(
                    ctx.producer(ins[0].guid), Silu):
                ins = [ins[1], ins[0]]
        lhs, rhs = ins[0], ins[1]
        ls, rs = ctx.known_shape(lhs), ctx.known_shape(rhs)
        dtype = ctx.common_dtype(lhs.guid, rhs.guid)
        node = cls(ls, broadcast=_infer_broadcast_index(ls, rs), name=f"{op.kind}_{idx}",
                   dtype=dtype, out_dtype=dtype if out_is_in_dtype else ctx.out_dtype_of(op),
                   n_batch_dims=ctx.batch_dims(lhs.guid, ls))
        return node, ins
    return rule


_RULES: Dict[Tuple[str, str], Rule] = {
    ("input", ""): _input,
    ("output", ""): _output,
    ("split", ""): _split,
    ("matmul", ""): _matmul,
    ("reduce_sum", ""): _reduction(ReduceSum),
    ("reduce_max", ""): _reduction(ReduceMax),
    ("exp", ""): _unary(Exp),
    ("silu", ""): _unary(Silu),
    ("sqrt", ""): _unary(Sqrt),
    ("rsqrt", ""): _rsqrt,
    ("add", ""): _binary(Add),
    ("sub", ""): _binary(Sub),
    ("mul", ""): _binary(Mul, silu_first=True),
    ("div", ""): _binary(Div, out_is_in_dtype=True),
    ("add", "scalar"): _scalar(Add),
    ("mul", "scalar"): _scalar(Mul),
    ("div", "scalar"): _scalar(Div),
}
_AXIS_KINDS = ("split", "reduce_sum", "reduce_max")
_KINDS = {k for k, _ in _RULES}
_SOURCE_KINDS = _KINDS - {"rsqrt"}  # rsqrt only comes from fuse_rsqrt_ops


def _rule(op: KGOp, idx: int, kinds=_KINDS) -> Rule:
    if op.kind in _AXIS_KINDS:
        axis_ok = op.axis is not None and 0 <= op.axis <= _MAX_AXIS
    else:
        axis_ok = op.axis is None
    if op.kind not in kinds or not axis_ok:
        raise ValueError(f"Unsupported op_type '{op.op_type}' at index {idx}")
    form = "scalar" if op.attrs.get("scalar") is not None and (op.kind, "scalar") in _RULES else ""
    return _RULES[(op.kind, form)]


def fuse_rsqrt_ops(ops: IndexedOps) -> IndexedOps:
    """``div(1.0, sqrt(x))`` -> ``rsqrt(x)`` where the div is the sqrt's only consumer."""
    uses = Counter(t.guid for _, op in ops for t in op.inputs)
    producer: Dict[int, int] = {}
    out, drop = list(ops), set()
    for pos, (idx, op) in enumerate(ops):
        if _is_reciprocal(op) and op.inputs:
            guid = op.inputs[0].guid
            src = producer.get(guid)
            sqrt = None if src is None else ops[src][1]
            if sqrt is not None and sqrt.kind == "sqrt" and sqrt.inputs and uses[guid] == 1:
                drop.add(src)
                out[pos] = (idx, KGOp("rsqrt", None, [sqrt.inputs[0]], list(op.outputs)))
        for t in op.outputs:
            producer[t.guid] = pos
    return [x for pos, x in enumerate(out) if pos not in drop]


def as_kernel_graph(graph_structure: Union[KernelGraph, List[Dict]]) -> KernelGraph:
    if isinstance(graph_structure, KernelGraph):
        return graph_structure
    for idx, d in enumerate(graph_structure):
        try:
            parse_op_type(d.get("op_type", ""))
        except ValueError:
            raise ValueError(f"Unsupported op_type '{d.get('op_type', '')}' at index {idx}") from None
    return KernelGraph.from_dicts(graph_structure)


def to_ir(
    graph_structure: Union[KernelGraph, List[Dict]],
    dtype: str = "float16",
    accum_dtype: str = "float32",
    out_dtype: str = "float16",
    symbol_map: Optional[Dict[int, int]] = None,
    fuse_rsqrt: bool = True,
) -> Graph:
    ops: IndexedOps = list(enumerate(as_kernel_graph(graph_structure).ops))
    for idx, op in ops:
        _rule(op, idx, _SOURCE_KINDS)
    if fuse_rsqrt:
        ops = fuse_rsqrt_ops(ops)
    ctx = _Ctx(ops, dtype, accum_dtype, out_dtype, symbol_map, fuse_rsqrt)
    for idx, op in ops:
        node, inputs = _rule(op, idx)(ctx, idx, op)
        if node is not None:
            ctx.wire(node, inputs)
            ctx.register(node, op.outputs)
    if not ctx.outputs:
        raise ValueError("No output nodes found in graph structure.")
    nodes = gather_all_nodes(*ctx.outputs)
    return Graph(nodes=find_topo_sort([n for n in nodes if n.is_output()]))
