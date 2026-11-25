"""The finite field of the first pass's fingerprint, and a flat graph evaluated in it.

The superoptimizer accepts a graph when its fingerprint equals the program's: every op
evaluated in ``Z_PQ`` (``P = 167``, ``Q = 83``, kept as the pair of residues) on random
inputs (``src/superoptimizer/kernel/fingerprint_cpu.cc``). The update search tests its
candidates in the same field with the same semantics, on the lowered flat graph:
:func:`values` is the counterpart of ``runtime.reference.values``.

    add, sub, mul       mod PQ
    div                 by the inverse of each residue (0 has inverse 1, as there)
    exp(x)              3 ** (x mod Q) mod P, in the P residue only (exp(a + b) = exp(a) exp(b))
    sqrt(x)             x ** ((r + 1) / 4) mod r, per residue r
    rsqrt(x)            1.0 / sqrt(x)  (the lowering fuses that pair into one op)
    silu(x)             exp(x) * (x mod Q), in the P residue only
    a constant c        the bits of float32(c) mod PQ; x - c is x + (-c), c - x is
                        x * (-1) + c (the tile graph writes them so)
    sum, contraction    sums of products mod PQ
    max                 opaque: a hash of the reduced slice, in order (no field has an order)

A division by an element that is zero in a residue takes the native convention there, and
an identity such as ``x / x = 1`` fails on it: the value tests nothing in that residue.
:func:`values` says where, when asked (``with_zeros``): per element, bit 1 where the value
is computed from a division by zero in P, bit 2 in Q (:func:`zero_bits` of every divisor,
as the value reads it: operand by operand, over a reduction or a contraction).

An op without semantics here (one the first pass does not fingerprint either) raises
:class:`NoField`; the caller has no test then. ``tests/update/test_field.py`` holds the
evaluator to the native fingerprint (``CyKNGraph.output_fingerprints``) op by op.
"""

from __future__ import annotations

import struct

import torch

P, Q = 167, 83
PQ = P * Q  # 13861
EXP_BASE = 3
_UNIT_P = 13695  # 1 mod P, 0 mod Q
_UNIT_Q = 167  # 1 mod Q, 0 mod P
_MASK = 0xFFFFFFFF


class NoField(Exception):
    """A node of the graph has no semantics in the field."""


def _inverses(r: int) -> torch.Tensor:
    return torch.tensor([1] + [pow(i, r - 2, r) for i in range(1, r)], dtype=torch.int64)


_EXP = torch.tensor([pow(EXP_BASE, i, P) for i in range(Q)], dtype=torch.int64)
_INV_P, _INV_Q = _inverses(P), _inverses(Q)
_SQRT_P = torch.tensor([pow(i, (P + 1) // 4, P) for i in range(P)], dtype=torch.int64)
_SQRT_Q = torch.tensor([pow(i, (Q + 1) // 4, Q) for i in range(Q)], dtype=torch.int64)


def crt(p, q):
    """The element of ``Z_PQ`` with residues ``p`` mod P and ``q`` mod Q."""
    return (p * _UNIT_P + q * _UNIT_Q) % PQ


def scalar(c: float) -> int:
    """A constant: the bits of ``float32(c)`` mod PQ."""
    return struct.unpack("<I", struct.pack("<f", float(c)))[0] % PQ


def add(x, y):
    return (x + y) % PQ


def sub(x, y):
    return (x % PQ - y % PQ) % PQ


def mul(x, y):
    return ((x % PQ) * (y % PQ)) % PQ


def div(x, y):
    return crt((x % P) * _INV_P[y % P] % P, (x % Q) * _INV_Q[y % Q] % Q)


def exp(x):
    return _EXP[x % Q] * _UNIT_P % PQ


def sqrt(x):
    return crt(_SQRT_P[x % P], _SQRT_Q[x % Q])


def rsqrt(x):
    return div(torch.full_like(x, scalar(1.0)), sqrt(x))


def zero_bits(y):
    """Where ``y`` is zero in each residue: bit 1 in P, bit 2 in Q (a division by it there
    takes the native convention and tests nothing)."""
    y = torch.as_tensor(y)
    return (y % P == 0).long() | ((y % Q == 0).long() << 1)


def or_bits(z, axis: int):
    """The bits of ``z`` or-ed over ``axis``."""
    return (z & 1).amax(dim=axis) | (z & 2).amax(dim=axis)


def silu(x):
    q = x % Q
    return _EXP[q] * q * _UNIT_P % PQ


def _mul32(x, c: int):
    """``x * c`` mod 2**32 for ``x`` < 2**32, without leaving int64."""
    lo, hi = x & 0xFFFF, x >> 16
    return (lo * c + (((hi * c) & 0xFFFF) << 16)) & _MASK


def _rand(x):
    """The native ``fingerprint_rand`` on uint32 values held in int64."""
    x = (x + 0x9E3779B9) & _MASK
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    return x ^ (x >> 16)


def reduce_sum(x, axis: int):
    return x.sum(dim=axis) % PQ


def reduce_max(x, axis: int):
    """Opaque: a hash of each reduced slice, element by element in order."""
    x = x.movedim(axis, -1)
    h = torch.full(x.shape[:-1], 0x6D6178, dtype=torch.int64)
    for k in range(x.shape[-1]):
        h = _rand(h ^ x[..., k])
    return h % PQ


def native_inputs(shapes) -> list:
    """The inputs the first pass fingerprints: element ``i`` of input ``k`` is
    ``fingerprint_rand(k + i mod PQ) & 0xFFFF``."""
    out = []
    for k, shape in enumerate(shapes):
        n = 1
        for d in shape:
            n *= int(d)
        seeds = (torch.arange(n, dtype=torch.int64) + k) % PQ
        out.append((_rand(seeds) & 0xFFFF).reshape(tuple(int(d) for d in shape)))
    return out


def random_inputs(shapes, seed: int) -> list:
    """Uniform random elements of the field, one tensor per shape."""
    g = torch.Generator().manual_seed(int(seed))
    return [torch.randint(0, PQ, tuple(int(d) for d in s), generator=g, dtype=torch.int64)
            for s in shapes]


# ------------------------------------------------------------------- one node of the IR


def _matmul(node, a, b):
    from ...runtime.reference import _broadcast_batch

    a, b = _matmul_operands(node, a, b)
    a = _broadcast_batch(a % PQ, node.batch_dims, node.input_batch_dims_A)
    b = _broadcast_batch(b % PQ, node.batch_dims, node.input_batch_dims_B)
    return (a @ b) % PQ


def _matmul_operands(node, a, b):
    m, n, k = node.M, node.N, node.K
    a = a.reshape((*node.input_batch_dims_A, k, m) if node.transpose_A
                  else (*node.input_batch_dims_A, m, k))
    b = b.reshape((*node.input_batch_dims_B, n, k) if node.transpose_B
                  else (*node.input_batch_dims_B, k, n))
    if node.transpose_A:
        a = a.transpose(-2, -1)
    if node.transpose_B:
        b = b.transpose(-2, -1)
    return a, b


def _matmul_zeros(node, za, zb):
    """Element ``(i, j)`` of a product reads row ``i`` of ``A`` and column ``j`` of ``B``."""
    from ...runtime.reference import _broadcast_batch

    za, zb = _matmul_operands(node, za, zb)
    rows = _broadcast_batch(or_bits(za, -1).unsqueeze(-1), node.batch_dims,
                            node.input_batch_dims_A)
    cols = _broadcast_batch(or_bits(zb, -2).unsqueeze(-2), node.batch_dims,
                            node.input_batch_dims_B)
    return rows | cols


def _operand_b(node, b):
    if node.broadcast == 0:
        return b.unsqueeze(-1)
    if node.broadcast == 1:
        return b.unsqueeze(-2)
    return b


def _binary_zeros(node, args, zs):
    from ...ir.ops import Div

    z = zs[0]
    if node.const is None:
        z = z | _operand_b(node, zs[1])
    if type(node) is Div:
        if node.const is None:
            divisor = _operand_b(node, args[1])
        elif node.const_on_left:
            divisor = args[0]
        else:
            divisor = torch.tensor(scalar(float(node.const)))
        z = z | zero_bits(divisor)
    return z


def _binary(node, args):
    from ...ir.ops import Add, Div, Mul, Sub

    if getattr(node, "denom_floor", None) is not None:
        raise NoField(f"{node.name}: a floored denominator has no field semantics")
    cls, a = type(node), args[0]
    if node.const is not None:
        c = float(node.const)
        if cls is Sub:
            if node.const_on_left:
                return add(mul(a, scalar(-1.0)), scalar(c))
            return add(a, scalar(-c))
        if cls is Div and node.const_on_left:
            return div(torch.full_like(a, scalar(c)), a)
        return {Add: add, Mul: mul, Div: div}[cls](a, torch.full_like(a, scalar(c)))
    return {Add: add, Sub: sub, Mul: mul, Div: div}[cls](a, _operand_b(node, args[1]))


def _splitview(node, x):
    shape = list(x.shape)
    axis = node.axis
    x = x.reshape(tuple(shape[:axis]) + (node.factor, node.inner_extent) + tuple(shape[axis + 1:]))
    return x.movedim(axis, node.split_pos)


def _semantics():
    from ...ir.ops import (Accumulate, AccumulateMax, Add, Copy, Div, Exp, Matmul, Mul,
                           PermuteView, ReduceMax, ReduceSum, RSqrt, Silu, SplitView, Sqrt,
                           Sub)

    return {
        SplitView: lambda node, args: _splitview(node, args[0]),
        PermuteView: lambda node, args: args[0].permute(*node.perm).contiguous(),
        Matmul: lambda node, args: _matmul(node, args[0], args[1]),
        ReduceSum: lambda node, args: reduce_sum(args[0], node.axis),
        ReduceMax: lambda node, args: reduce_max(args[0], node.axis),
        Add: _binary, Sub: _binary, Mul: _binary, Div: _binary,
        Exp: lambda node, args: exp(args[0]),
        Sqrt: lambda node, args: sqrt(args[0]),
        RSqrt: lambda node, args: rsqrt(args[0]),
        Silu: lambda node, args: silu(args[0]),
        Copy: lambda node, args: args[0],
        Accumulate: lambda node, args: args[0],
        AccumulateMax: lambda node, args: args[0],
    }


_SEMANTICS: dict = {}


def semantics(node):
    """The field semantics of ``node``'s exact class, or None."""
    if not _SEMANTICS:
        _SEMANTICS.update(_semantics())
    return _SEMANTICS.get(type(node))


def _zeros(node, args, zs):
    """The zero-divisor bits of ``node``'s value (``args`` its operands' values, ``zs``
    their bits): the operands' as the value reads them, and its own division's."""
    from ...ir.ops import (Add, Div, Matmul, Mul, PermuteView, ReduceMax, ReduceSum, RSqrt,
                           SplitView, Sub)

    cls = type(node)
    if cls in (ReduceSum, ReduceMax):
        return or_bits(zs[0], node.axis)
    if cls is Matmul:
        return _matmul_zeros(node, zs[0], zs[1])
    if cls is SplitView:
        return _splitview(node, zs[0])
    if cls is PermuteView:
        return zs[0].permute(*node.perm).contiguous()
    if cls in (Add, Sub, Mul, Div):
        return _binary_zeros(node, args, zs)
    if cls is RSqrt:
        return zs[0] | zero_bits(args[0])
    return zs[0]  # elementwise: exp, sqrt, silu, copies


def values(graph, inputs, given=None, given_zeros=None, with_zeros=False):
    """``({node name: value}, outputs)`` of ``graph`` in the field for ``inputs`` (one
    int64 tensor per placeholder, ``placeholder`` order): the counterpart of
    ``runtime.reference.values``. ``given`` (``{node name: tensor}``) supplies the value of
    the named compute nodes instead of evaluating them (``given_zeros`` their zero-divisor
    bits; none where it does not say). ``with_zeros``: a third result, ``{node name: bits}``
    (:func:`zero_bits`, module docstring). :class:`NoField` for a node the field gives no
    meaning."""
    from ...ir.graph import OutputNode, PlaceHolderNode
    from ...ir.graph_utils import find_topo_sort

    given, given_zeros = given or {}, given_zeros or {}
    order = find_topo_sort(graph.nodes)
    placeholders = [n for n in order if isinstance(n, PlaceHolderNode)]
    if len(inputs) != len(placeholders):
        raise ValueError(f"expected {len(placeholders)} inputs, got {len(inputs)}")
    env = {(p, 0): t for p, t in zip(placeholders, inputs)}
    zeros = {(p, 0): torch.zeros_like(t) for p, t in zip(placeholders, inputs)}
    outputs = []
    for node in order:
        if isinstance(node, PlaceHolderNode):
            continue
        if isinstance(node, OutputNode):
            edge = node.inputs[0]
            outputs.append(env[(edge.src_node, edge.src_id)])
            continue
        if node.name in given:
            value = env[(node, 0)] = given[node.name]
            zeros[(node, 0)] = given_zeros.get(node.name, torch.zeros_like(value))
            continue
        fn = semantics(node)
        if fn is None:
            raise NoField(f"{node.name} ({type(node).__name__}) has no field semantics")
        edges = sorted(node.inputs, key=lambda e: e.dst_id)
        args = [env[(e.src_node, e.src_id)] for e in edges]
        env[(node, 0)] = fn(node, args)
        if with_zeros:
            zs = [zeros[(e.src_node, e.src_id)] for e in edges]
            zeros[(node, 0)] = _zeros(node, args, zs).expand(env[(node, 0)].shape).contiguous()
    named = {node.name: value for (node, _), value in env.items()}
    if with_zeros:
        return named, outputs, {node.name: z for (node, _), z in zeros.items()}
    return named, outputs
