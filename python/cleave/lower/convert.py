"""Superoptimizer graph structure (``KernelGraph`` or its op dicts) -> Cleave ``Graph``.

Each op is built by a rule from ``_RULES``, keyed on ``(kind, form)``; ``fuse_rsqrt_ops``
is a rewrite pass over the ops that runs before conversion.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import tilelang  # noqa: F401
import tvm
from tvm import te

from ..core.schema import KernelGraph, KGOp, KGTensor, parse_op_type
from ..ir.graph import Boundary, Graph, OutputNode, PlaceHolderNode
from ..ir.ops import (Add, Div, Exp, Matmul, ReduceMax, Mul, ReduceSum, RSqrt, Silu, SplitView,
                   Sqrt, Sub)
from ..ir.ops.base import wire_input
from ..ir.graph_utils import find_topo_sort, gather_all_nodes

_MAX_AXIS = 4
IndexedOps = List[Tuple[int, KGOp]]  # (index in the source structure, op); names use the index


def _broadcast_fits(dims: Tuple[int, ...], other_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """The ``broadcast`` values under which a binary op on ``dims`` reads ``other_shape``.

    -1: the same shape. 0: ``dims`` without its last axis (the operand spans the
    second-to-last and is broadcast along the last). 1: ``dims`` without its second-to-last.
    Value shapes only, so equal trailing extents give ``(0, 1)`` and the structure has to
    choose. Empty: the op cannot read that operand at all.
    """
    dims, other_shape = tuple(dims), tuple(other_shape)
    if dims == other_shape:
        return (-1,)
    if len(dims) < 2:
        return ()
    return tuple(i for i in (0, 1) if other_shape == dims[:-2] + (dims[i - 2],))


def _broadcast_from_symbols(lhs: Tuple[int, ...], rhs: Tuple[int, ...], rank: int) -> Optional[int]:
    """The broadcast index read off the raw dims of the two structure tensors, else None.

    Before ``symbol_map`` two axes of one extent are still different symbols, and a
    reduction leaves its axis behind as a literal ``1``: the axis the operand does not span.
    ``rank`` is the rank of the lhs *value*, which has dropped the axes reduced before it
    (a split combine divides ``(1, BH, SQ, D)`` by ``(1, BH, SQ, 1)``: values of rank 3 and 2).
    """
    lhs, rhs = tuple(lhs), tuple(rhs)
    if len(lhs) != rank and len(lhs) == len(rhs):
        kept = [i for i in range(len(lhs)) if not lhs[i] == rhs[i] == 1]
        lhs, rhs = tuple(lhs[i] for i in kept), tuple(rhs[i] for i in kept)
    if len(lhs) != rank or rank < 2:
        return None
    if len(rhs) == len(lhs):  # keepdim form
        if rhs[:-2] != lhs[:-2]:
            return None
        keeps_m, keeps_n = rhs[-2] == lhs[-2], rhs[-1] == lhs[-1]
        if keeps_m and not keeps_n and rhs[-1] == 1:
            return 0
        if keeps_n and not keeps_m and rhs[-2] == 1:
            return 1
        return None
    if len(rhs) == len(lhs) - 1 and rhs[:-1] == lhs[:-2]:
        spans_m, spans_n = rhs[-1] == lhs[-2], rhs[-1] == lhs[-1]
        if spans_m != spans_n:
            return 0 if spans_m else 1
    return None


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


def _live_ops(ops: IndexedOps) -> set:
    """Indices of the ops that some output reads, directly or through other ops."""
    producer = {t.guid: idx for idx, op in ops for t in op.outputs}
    reads = {idx: [t.guid for t in op.inputs] for idx, op in ops}
    live, stack = set(), [idx for idx, op in ops if op.kind == "output"]
    while stack:
        idx = stack.pop()
        if idx not in live:
            live.add(idx)
            stack += [producer[guid] for guid in reads[idx] if guid in producer]
    return live


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
        # guid -> (structure rank, structure axis behind each value axis), where a value
        # has dropped axes its structure tensor keeps as unit dims: a reduction's output
        # and what shape-preserving ops make of it. Absent: the value has every axis.
        self.kept: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
        # Inputs: the operand shape each consumer declares (name, shape), and the guids of
        # those a binary op takes with unit axes dropped (``broadcast_index``).
        self.read_as: Dict[int, List[Tuple[str, Tuple[int, ...]]]] = {}
        self.reshaped: set = set()
        # Input op -> its position among the structure's input ops: the program's order
        # of its inputs, which the graph rebuilt from the outputs forgets. Every input op
        # is counted, also one no output reads: it is not converted and gets no
        # placeholder, and the inputs declared after it keep their place. This is the one
        # definition of that position; both records of it (``Boundary.index``, which
        # pairs split stages, and ``input_index``, which binds a program's inputs) are
        # written from it in ``placeholder``.
        self.input_position: Dict[int, int] = {
            idx: position
            for position, idx in enumerate(
                idx for idx, op in ops if op.kind == "input" and op.outputs
            )
        }
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

    def placeholder(self, t: KGTensor, position: Optional[int] = None) -> PlaceHolderNode:
        """The placeholder for ``t``; ``position`` (``input_position``) when an input op
        declares it: the node's ``boundary`` index and its ``input_index``."""
        node = PlaceHolderNode(name=f"input_{t.guid}")
        shape = self.shape(t)
        dtype = self.dtype if t.dtype is None else str(t.dtype)
        node.set_shape(shape)
        node.set_dtype(tvm.DataType(dtype))
        node.boundary = None if position is None else Boundary(position)
        if position is not None:
            node.input_index = position
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
                declared = tuple(int(d) for d in node.input_buffers[dst_id].shape)
                self.read_as.setdefault(t.guid, []).append((node.name, declared))

    def register(self, node, outputs: Sequence[KGTensor]) -> None:
        for i, t in enumerate(outputs):
            self.nodes[t.guid] = (node, i)
            self.shapes[t.guid] = tuple(node.get_shape(i))

    def extents(self, t: KGTensor) -> tuple:
        """The dims of the structure tensor ``t`` as extents (None for an unknown symbol)."""
        dims = tuple(int(v) for v in _dims(t))
        return tuple(self.symbol_map.get(v) for v in dims) if self.symbol_map else dims

    def kept_axes(self, t: KGTensor) -> Tuple[int, Tuple[int, ...], bool]:
        """``(rank, axes, keepdim)``: the structure axis behind each axis of the value of
        ``t``, and whether its structure tensor still lists the dropped axes (as units).

        The identity unless a reduction was recorded for ``t`` and the structure dims of
        ``t`` agree with the record: a rewritten split structure numbers axes its own way.
        """
        shape = tuple(self.known_shape(t))
        identity = (len(shape), tuple(range(len(shape))), False)
        if t.guid not in self.kept:
            return identity
        rank, axes = self.kept[t.guid]
        dims = self.extents(t)
        if len(axes) != len(shape):
            return identity
        if dims == shape:
            return rank, axes, False
        if len(dims) == rank and all(
                dims[a] == (shape[axes.index(a)] if a in axes else 1) for a in range(rank)):
            return rank, axes, True
        return identity

    def forward_kept(self, src: KGTensor, op: KGOp) -> None:
        """A shape-preserving op keeps the meaning of its input's axes."""
        if src.guid in self.kept:
            for t in op.outputs:
                self.kept[t.guid] = self.kept[src.guid]

    def value_axis(self, idx: int, op: KGOp, t: KGTensor) -> int:
        """``op.axis`` (an axis of the structure tensor ``t``) as an axis of its value.

        The two differ once a reduction has run: the structure keeps the reduced axis as
        a unit dim, the value drops it, so ``sum(sum(x, 0), 1)`` reduces value axis 0.
        """
        _, axes, keepdim = self.kept_axes(t)
        if not keepdim:
            return op.axis
        if op.axis not in axes:
            raise ValueError(f"{op.op_type} at index {idx}: axis {op.axis} of "
                             f"{tuple(_dims(t))} was already reduced away")
        return axes.index(op.axis)

    def positions(self, t: KGTensor) -> Optional[Tuple[int, ...]]:
        """Where each axis of the value of ``t`` sits in the dims of ``t``, if the dims say.

        They do for a value that has every axis its tensor lists, and for one whose record
        of reduced axes the dims confirm (they keep those axes as units). The tensors of
        one op are aligned position by position when they have the same rank, so with both
        answers known the dims say exactly along which axes the program broadcasts.
        """
        _, axes, keepdim = self.kept_axes(t)
        if keepdim:
            return axes
        shape = tuple(self.known_shape(t))
        return tuple(range(len(shape))) if self.extents(t) == shape else None

    def is_input(self, t: KGTensor) -> bool:
        """A graph input: it takes the shape its consumer declares (``wire_input``)."""
        return t.guid not in self.nodes or isinstance(self.producer(t.guid), PlaceHolderNode)

    def broadcast_index(self, idx: int, op: KGOp, lhs: KGTensor, rhs: KGTensor) -> int:
        """``broadcast`` of the binary op ``lhs <op> rhs`` (see ``_broadcast_fits``).

        The value shapes say which indices can read ``rhs``; the structure says which one
        the program means, and the two must agree. Where both tensors list the same axes
        and the dims confirm what each value has kept of them, the reading is exact: the
        program broadcasts ``rhs`` along the axes where its extent is 1 and the lhs's is
        not, and an operand that fits an index only because two extents happen to be equal
        is refused, not misread. Elsewhere the symbolic dims of the two tensors choose, and
        between two fitting indices the axis the producer of ``rhs`` reduced; never which
        shape test runs first.
        """
        ls, rs = tuple(self.known_shape(lhs)), tuple(self.known_shape(rhs))
        where = f"{op.op_type} at index {idx}"
        dims = f"dims {tuple(_dims(lhs))} and {tuple(_dims(rhs))}"
        rule = ("a binary op keeps the shape of its first operand and broadcasts the second "
                "along one of its last two axes")
        fits = _broadcast_fits(ls, rs)
        lext, rext = self.extents(lhs), self.extents(rhs)
        lpos, rpos = self.positions(lhs), self.positions(rhs)
        if len(lext) == len(rext) and lpos is not None and rpos is not None:
            for ours, theirs in zip(lext, rext):
                if theirs not in (ours, 1):  # the result would not have the lhs's shape
                    raise ValueError(f"{where}: the operand {rs} has extent {theirs} where "
                                     f"the first operand {ls} has {ours} ({dims}); {rule}")
            # axes of the lhs value along which the program broadcasts the operand
            along = [lpos.index(p) for p in range(len(lext)) if rext[p] == 1 != lext[p]]
            if along:
                index = {len(ls) - 1: 0, len(ls) - 2: 1}.get(along[0])
                if len(along) > 1 or len(ls) < 2 or index is None:
                    axes = ("axis " if len(along) == 1 else "axes ") + ", ".join(map(str, along))
                    raise ValueError(f"{where}: the operand {rs} is broadcast along {axes} of "
                                     f"the first operand {ls} ({dims}); {rule}")
                if index not in fits:
                    reads = ls[:along[0]] + ls[along[0] + 1:]
                    raise ValueError(f"{where}: the operand {rs} is broadcast along axis "
                                     f"{along[0]} of the first operand {ls}, which reads an "
                                     f"operand of shape {reads} there ({dims})")
                return index
            # No broadcast: both values hold the same elements, give or take unit axes.
            if fits == (0, 1):  # two unit axes it may lack: the one the operand has dropped
                return 1 if lpos[-2] not in rpos and lpos[-1] in rpos else 0
            if fits:
                return fits[0]
            if not self.is_input(rhs):
                raise ValueError(f"{where}: the operand {rs} and the first operand {ls} hold "
                                 f"the same elements with different unit axes ({dims})")
            self.reshaped.add(rhs.guid)  # an input: wiring declares it like the first operand
            return -1
        if not fits:
            if self.is_input(rhs) and math.prod(rs) == math.prod(ls):
                self.reshaped.add(rhs.guid)  # declared with the shape of the first operand
                return -1
            raise ValueError(f"{where}: an operand of shape {rs} is neither the shape {ls} of "
                             f"the first operand nor that shape without one of its last two "
                             f"axes ({dims})")
        if fits == (-1,):
            return -1
        index = _broadcast_from_symbols(tuple(_dims(lhs)), tuple(_dims(rhs)), len(ls))
        if index is not None and index not in fits:
            raise ValueError(f"{where}: by its dims the operand {rs} is broadcast along axis "
                             f"{len(ls) - 1 - index} of {ls}, where its shape does not fit ({dims})")
        if index is None and len(fits) > 1:
            (lrank, laxes, _), (rrank, raxes, _) = self.kept_axes(lhs), self.kept_axes(rhs)
            if lrank == rrank and set(raxes) <= set(laxes):
                spans = tuple(laxes.index(a) for a in raxes)  # axes of the lhs value
                batch = tuple(range(len(ls) - 2))
                index = {batch + (len(ls) - 2,): 0, batch + (len(ls) - 1,): 1}.get(spans)
        if index is None:
            # One index reads the operand, or the trailing axes hold one element each (both
            # indices then read the same value): there is nothing to choose.
            if len(fits) > 1 and ls[-1] != 1:
                raise ValueError(
                    f"{where}: cannot tell which axis of {ls} the operand {rs} is broadcast "
                    f"along (equal extents; {dims}, and the operand's producer, do not say)")
            index = fits[0]
        return index


Built = Tuple[Optional[object], Sequence[KGTensor]]  # (node or None, the inputs to wire)
Rule = Callable[[_Ctx, int, KGOp], Built]


def _input(ctx: _Ctx, idx: int, op: KGOp) -> Built:
    # The k-th input op is input k of the graph. A graph is rebuilt in discovery order
    # from its outputs below, so this is the only place the declaration order is known.
    if op.outputs:
        ctx.placeholder(op.outputs[0], ctx.input_position[idx])
    return None, ()


def _output(ctx: _Ctx, idx: int, op: KGOp) -> Built:
    if op.inputs:
        guid = op.inputs[0].guid
        if guid not in ctx.nodes:
            raise ValueError(f"Output tensor guid {guid} has no producer.")
        node = OutputNode(ctx.producer(guid), dtype=ctx.dtypes.get(guid, ctx.out_dtype))
        node.boundary = Boundary(len(ctx.outputs))  # the k-th output op is output k
        ctx.outputs.append(node)
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
    for which, t in (("first", a), ("second", b)):
        # The config below reads a value of rank 1 as a vector and the last two axes of any
        # other as its matrix axes. A reduction drops its axis from the value, so a batched
        # vector would be read as a matrix whose rows are its last batch axis.
        rank, axes, keepdim = ctx.kept_axes(t)
        if keepdim and len(axes) >= 2 and not {rank - 2, rank - 1} <= set(axes):
            raise ValueError(
                f"{op.op_type} at index {idx}: the {which} operand {tuple(ctx.known_shape(t))} "
                f"has lost a matrix axis to a reduction (dims {tuple(_dims(t))}); only a "
                "vector without batch axes can stand in for a matrix")
    shapes = {"first": tuple(ctx.known_shape(a)), "second": tuple(ctx.known_shape(b))}
    for (which, shape), (_, other) in zip(shapes.items(), reversed(shapes.items())):
        # A vector stands in for a one-row or one-column matrix. Next to a batched matrix it
        # would be that row for every batch entry, which the config below declares as a
        # vector per batch entry: an operand of B times its size, which nothing can run.
        if len(shape) == 1 and len(other) > 2:
            raise ValueError(
                f"{op.op_type} at index {idx}: the {which} operand is a vector {shape} and "
                f"the other a batched matrix {other} (dims {tuple(_dims(a))} and "
                f"{tuple(_dims(b))}); only a matmul without batch axes takes a vector")
    dims, ta, tb, a_batch, b_batch = _infer_matmul_config(
        shapes["first"], shapes["second"], ctx.shape(op.outputs[0]),
        bool(op.attrs.get("trans_A", False)), bool(op.attrs.get("trans_B", False)))
    batch_entries = math.prod(int(extent) for extent in dims[:-3])  # of the output
    for (which, shape), t in zip(shapes.items(), (a, b)):
        # The op declares both operands with the batch axes of its output. An input takes
        # the declared shape (unit extents where it has no batch axis); a computed value
        # keeps its own, and one that has fewer axes lost a batch axis to a reduction.
        # While it still has every batch entry of the output -- the output has one entry
        # along each axis the value lacks, because the other operand lost it too or is
        # broadcast over it -- the declared shape is the stored one with unit axes, which
        # every tile squeezes. Where the output has more entries, the value is read once
        # per entry and is tiled as the matmul reads it, not as it is stored: no tiling of
        # such a graph builds.
        if (len(shape) >= 2 and len(shape) != len(dims) - 1 and not ctx.is_input(t)
                and math.prod(shape[:-2]) != batch_entries):
            raise ValueError(
                f"{op.op_type} at index {idx}: the {which} operand {shape} is computed "
                f"without the batch axes of its matmul, whose output is {tuple(dims[:-1])} "
                f"(dims {tuple(_dims(t))}); only an input is read over batch axes it does "
                "not have")
    node = Matmul(dims, input_batch_dims_A=a_batch, input_batch_dims_B=b_batch, name=f"mma_{idx}",
                  transpose_A=ta, transpose_B=tb, warp_policy="rows", dtype=ctx.dtype,
                  out_dtype=ctx.accum_dtype)
    return node, op.inputs


def _reduction(cls) -> Rule:
    def rule(ctx: _Ctx, idx: int, op: KGOp) -> Built:
        t, axis = op.inputs[0], op.axis
        shape = ctx.known_shape(t)
        value_axis = ctx.value_axis(idx, op, t)
        # Named by the pinned macro stem (cls.kind): node names reach generated kernels.
        node = cls(shape, axis=-1 if value_axis == len(shape) - 1 else value_axis,
                   name=f"{cls.kind}_{axis}_{idx}", dtype=ctx.in_dtype(t.guid),
                   out_dtype=ctx.accum_dtype)
        rank, axes, _ = ctx.kept_axes(t)
        for out in op.outputs:
            ctx.kept[out.guid] = (rank, axes[:node.axis] + axes[node.axis + 1:])
        return node, op.inputs
    return rule


def _unary(cls) -> Rule:
    def rule(ctx: _Ctx, idx: int, op: KGOp) -> Built:
        t = op.inputs[0]
        shape = ctx.known_shape(t)
        node = cls(shape, name=f"{op.kind}_{idx}", dtype=ctx.in_dtype(t.guid),
                   out_dtype=ctx.out_dtype_of(op), n_batch_dims=ctx.batch_dims(t.guid, shape))
        ctx.forward_kept(t, op)
        return node, op.inputs
    return rule


def _rsqrt(ctx: _Ctx, idx: int, op: KGOp) -> Built:
    t = op.inputs[0]
    shape = ctx.known_shape(t)
    nbd = ctx.batch_dims(t.guid, shape)
    # dims typed as the fused Sqrt's input buffer shape
    node = RSqrt(tuple(te.placeholder(shape).shape), name=f"rsqrt_{idx}", dtype=ctx.accum_dtype,
                 out_dtype=ctx.out_dtype_of(op), n_batch_dims=nbd)
    ctx.forward_kept(t, op)
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
        ctx.forward_kept(t, op)
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
        ls = ctx.known_shape(lhs)
        dtype = ctx.common_dtype(lhs.guid, rhs.guid)
        broadcast, n_batch_dims = ctx.broadcast_index(idx, op, lhs, rhs), ctx.batch_dims(lhs.guid, ls)
        if broadcast != -1 and n_batch_dims is not None:
            # every axis but the last is a batch axis of the lhs (see ``_Ctx.batch_dims``),
            # and a binary op broadcasts along a non-batch axis only
            raise ValueError(
                f"{op.op_type} at index {idx}: an operand of shape {tuple(ctx.known_shape(rhs))} "
                f"cannot be broadcast over {tuple(ls)}, a value with {n_batch_dims} batch axes")
        node = cls(ls, broadcast=broadcast, name=f"{op.kind}_{idx}",
                   dtype=dtype, out_dtype=dtype if out_is_in_dtype else ctx.out_dtype_of(op),
                   n_batch_dims=n_batch_dims)
        ctx.forward_kept(lhs, op)  # the result has the lhs's shape
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
    live = _live_ops(ops)
    for idx, op in ops:
        if idx not in live:
            # No output reads it, so there is nothing to build -- and an op that is built is
            # wired: it would hand its shape to the inputs it reads and could refuse the graph.
            continue
        node, inputs = _rule(op, idx)(ctx, idx, op)
        if node is not None:
            ctx.wire(node, inputs)
            ctx.register(node, op.outputs)
    if not ctx.outputs:
        raise ValueError("No output nodes found in graph structure.")
    for guid in sorted(ctx.reshaped):
        # ``wire_input`` gives an input the shape of the last consumer wired; that is only a
        # reshape of the input if every consumer reads it the same way.
        (name, shape), *others = ctx.read_as[guid]
        other = next(((n, s) for n, s in others if s != shape), None)
        if other is not None:
            raise ValueError(f"input {guid} is read as {shape} by {name} and as {other[1]} by "
                             f"{other[0]}; an input has one layout")
    nodes = gather_all_nodes(*ctx.outputs)
    return Graph(nodes=find_topo_sort([n for n in nodes if n.is_output()]))
