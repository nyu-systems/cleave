"""Bounded bottom-up enumeration of update rules, tested exactly in the field.

The loop the assembly builds updates an accumulator one way: ``acc *= F``, then the
accumulator's own macro adds the tile (``update_rule.update_of``: ``a * F + r``). So what is
searched is the factor ``F``, an expression (``cleave.ir.update_rule``) over

    leaves   the full-extent value before and after the iteration, ``m`` / ``m'``;
    ops      + - * /  and the unary functions the flat graph applies between the
             full-extent value and its accumulators (``UNARY``: exp, sqrt, rsqrt, silu).
             No constant, no reduction, no matmul.

and a factor is a match when ``r + a * F`` equals the program's own new state, where ``a``
is the accumulator's old value and ``r`` the tile's result. Every value is a field element
(``field``, the first pass's fingerprint semantics) of the test data (``reference``): the
program's states on prefixes of the looped dim, every row of every case. The enumeration
runs on a SAMPLE of them -- ``ROWS_PER_CASE`` rows of every case, ``SAMPLE_COLS`` columns of
a wide accumulator -- and a factor whose sampled update equals the target is CONFIRMED on
every element; only a confirmed one is a match. Equality is exact: no tolerance anywhere.

Equality is per residue. Where a value is computed from a division by zero in a residue
(``field.zero_bits``: the native convention stands in for the quotient), it tests nothing in
that residue: an element is compared in the residues in which neither the program's test
data (``excluded``) nor the update divides by zero, and an update that leaves an element the
data tests compared in neither residue is no match (:func:`tested`, :func:`agrees`). So
``exp(m) / exp(m')``, whose divisor is zero mod Q everywhere (``exp`` lives in P), is
compared mod P, and ``m / (m - m)`` is no match.

A factor is a *row* value (the shape of the full-extent value): the repair multiplies every
element of an accumulator's row by the row's factor.

Size = number of op nodes of the factor (the update has two more: ``a * F + r``). Level
``k`` holds the values of every factor of size ``k``: unary on level ``k-1``, binary on
levels ``a + b = k - 1`` (commutative ops form each unordered pair once). Every formed factor
is tested BEFORE the merge, and every one that matches is collected: the search never stops
at a match. Intermediates are merged by value (equal on every sampled element; the first
expression wins).

Deterministic: fixed leaf order (``m, m'``), op order, level order, sample.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dfield

import numpy as np

from ...ir.update_rule import show, size, update_of
from . import field as F

BINARY = ("add", "sub", "mul", "div")
COMMUTATIVE = ("add", "mul")
LEAF_ORDER = ("m", "m'")
ROWS_PER_CASE = 6
SAMPLE_COLS = 4

PQ, P, Q = F.PQ, F.P, F.Q
_EXP = F._EXP.numpy()
_INV_P, _INV_Q = F._INV_P.numpy(), F._INV_Q.numpy()
_SQRT_P, _SQRT_Q = F._SQRT_P.numpy(), F._SQRT_Q.numpy()
_ONE = F.scalar(1.0)


def _crt(p, q):
    return (p * F._UNIT_P + q * F._UNIT_Q) % PQ


def _div(x, y):
    return _crt((x % P) * _INV_P[y % P] % P, (x % Q) * _INV_Q[y % Q] % Q)


def _sqrt(x):
    return _crt(_SQRT_P[x % P], _SQRT_Q[x % Q])


def _silu(x):
    q = x % Q
    return _EXP[q] * q * F._UNIT_P % PQ


OPS = {
    "add": lambda x, y: (x + y) % PQ,
    "sub": lambda x, y: (x - y) % PQ,
    "mul": lambda x, y: (x * y) % PQ,
    "div": _div,
}
UNARY = {
    "exp": lambda x: _EXP[x % Q] * F._UNIT_P % PQ,
    "sqrt": _sqrt,
    "rsqrt": lambda x: _div(np.full_like(x, _ONE), _sqrt(x)),
    "silu": _silu,
}


@dataclass
class Result:
    hole: str
    budget: int  # ops of the update ``a * F + r``
    unary: tuple
    visited: int = 0  # factors formed and tested
    kept: list = dfield(default_factory=list)  # distinct factor values kept per level
    formed: list = dfield(default_factory=list)  # factors formed per level
    sampled: int = 0  # factors whose update matched on the sample
    matches: list = dfield(default_factory=list)  # confirmed factors, smallest first
    elements: tuple = (0, 0)  # (sampled, all) elements of the test data
    seconds: float = 0.0


def evaluate(expr, env):
    """A searched expression on field blocks; a row operand broadcasts over a wide one."""
    if isinstance(expr, str):
        return env[expr]
    op, *args = expr
    vals = [evaluate(a, env) for a in args]
    if op in UNARY:
        return UNARY[op](vals[0])
    return OPS[op](vals[0], vals[1])


def _zero_bits(y):
    return (y % P == 0).astype(np.int64) | ((y % Q == 0).astype(np.int64) << 1)


def evaluate_zeros(expr, env):
    """``(value, bits)`` of :func:`evaluate`: ``bits`` where the expression itself divides by
    zero (a ``div`` or ``rsqrt`` of an element that is zero in a residue)."""
    if isinstance(expr, str):
        return env[expr], np.zeros_like(env[expr])
    op, *args = expr
    pairs = [evaluate_zeros(a, env) for a in args]
    vals = [v for v, _ in pairs]
    bits = pairs[0][1] if len(pairs) == 1 else pairs[0][1] | pairs[1][1]
    if op == "div":
        bits = bits | _zero_bits(vals[1])
    elif op == "rsqrt":
        bits = bits | _zero_bits(vals[0])
    value = UNARY[op](vals[0]) if op in UNARY else OPS[op](vals[0], vals[1])
    return value, bits


def tested(got, want, excluded, bits) -> np.ndarray:
    """Per element: ``got`` equals ``want`` in every residue in which neither the data
    (``excluded``) nor the update (``bits``) divides by zero (bit 1: P, bit 2: Q), and the
    update leaves the element compared in a residue wherever the data does."""
    skip = excluded | bits
    diff = (np.asarray(got) - np.asarray(want)) % PQ
    equal = (((diff % P) == 0) | ((skip & 1) != 0)) & (((diff % Q) == 0) | ((skip & 2) != 0))
    return equal & ((skip != 3) | (excluded == 3))


def agrees(update, leaves: dict, target, excluded) -> bool:
    """Whether ``update`` gives ``target`` on every element of the test data, as
    :func:`tested` compares them."""
    target = np.asarray(target)
    got, bits = evaluate_zeros(update, leaves)
    got = np.broadcast_to(got, target.shape)
    bits = np.broadcast_to(bits, target.shape)
    return bool(tested(got, target, excluded, bits).all())


def sample(cases: int, rows: int, width: int):
    """``(row indices, column indices)`` of the sample: ``ROWS_PER_CASE`` rows spread over
    each of ``cases`` cases of ``rows`` rows, ``SAMPLE_COLS`` columns spread over ``width``."""
    per = np.unique(np.linspace(0, rows - 1, min(rows, ROWS_PER_CASE)).round().astype(int))
    idx = np.concatenate([c * rows + per for c in range(cases)])
    cols = np.unique(np.linspace(0, width - 1, min(width, SAMPLE_COLS)).round().astype(int))
    return idx, cols


def enumerate_factors(hole: str, leaves: dict, target: np.ndarray, *, cases: int,
                      budget: int = 4, unary=("exp",), excluded=None) -> Result:
    """Every factor ``F`` of at most ``budget - 2`` ops over ``m`` and ``m'`` whose update
    ``r + a * F`` equals ``target`` on every element of the test data.

    ``leaves``: int64 field blocks ``a`` and ``r`` ``(n, w)``, ``m`` and ``m'`` ``(n, 1)``, the
    rows of ``cases`` cases stacked; ``target`` is ``(n, w)``; ``excluded`` ``(n, w)`` the
    residues each element tests nothing in (module docstring; none by default).
    """
    t0 = time.perf_counter()
    target = np.asarray(target, dtype=np.int64)
    n_all, width = target.shape
    excluded = (np.zeros_like(target) if excluded is None
                else np.asarray(excluded, dtype=np.int64))
    rows, cols = sample(cases, n_all // cases, width)
    full = {k: np.asarray(v, dtype=np.int64) for k, v in leaves.items()}
    a, r = full["a"][rows][:, cols], full["r"][rows][:, cols]
    goal, skip = target[rows][:, cols], excluded[rows][:, cols]
    res = Result(hole, budget, tuple(unary), elements=(goal.size, target.size))

    def check(exprs_fn, vals, bits):
        """Test factors ``vals`` ``(count, n, 1)`` (dividing by zero at ``bits``); record
        every confirmed one."""
        res.visited += vals.shape[0]
        update = (r[None] + a[None] * vals) % PQ
        hit = tested(update, goal[None], skip[None], bits).reshape(vals.shape[0], -1).all(axis=1)
        for i in np.nonzero(hit)[0]:
            expr = exprs_fn(int(i))
            res.sampled += 1
            if agrees(update_of(expr), full, target, excluded):
                res.matches.append(expr)

    seen = set()
    level0 = np.stack([full[k][rows] for k in LEAF_ORDER])  # (2, n, 1)
    groups = [(list(LEAF_ORDER), level0, np.zeros_like(level0))]
    for v in level0:
        seen.add(v.astype(np.int16).tobytes())
    check(lambda i: LEAF_ORDER[i], level0, groups[0][2])

    def keep(exprs, vals, zeros, exprs_fn, out, out_bits):
        for i in range(out.shape[0]):
            key = out[i].astype(np.int16).tobytes() + out_bits[i].astype(np.int8).tobytes()
            if key not in seen:
                seen.add(key)
                exprs.append(exprs_fn(i))
                vals.append(out[i])
                zeros.append(out_bits[i])

    ops_budget = budget - 2
    empty = np.zeros((0,) + level0.shape[1:], dtype=np.int64)
    for k in range(1, ops_budget + 1):
        exprs, vals, zeros, formed = [], [], [], 0
        last = k == ops_budget
        prev_exprs, prev_vals, prev_bits = groups[k - 1]
        for op in unary:  # unary on level k-1
            if not prev_exprs:
                break
            out = UNARY[op](prev_vals)
            out_bits = prev_bits | _zero_bits(prev_vals) if op == "rsqrt" else prev_bits
            fn = lambda i, op=op, e=prev_exprs: (op, e[i])  # noqa: E731
            formed += out.shape[0]
            check(fn, out, out_bits)
            if not last:
                keep(exprs, vals, zeros, fn, out, out_bits)
        for x in range(0, k):  # binary: sizes x + y = k - 1
            y = k - 1 - x
            ea, va, za = groups[x]
            eb, vb, zb = groups[y]
            if not ea or not eb:
                continue
            for op in BINARY:
                if op in COMMUTATIVE and x > y:
                    continue  # the unordered pair of levels once
                out = OPS[op](va[:, None], vb[None])  # (ca, cb, n, 1)
                out_bits = za[:, None] | zb[None]
                if op == "div":
                    out_bits = out_bits | _zero_bits(vb[None])
                ca, cb = out.shape[:2]
                if op in COMMUTATIVE and x == y:  # i <= j only
                    ii, jj = np.nonzero(np.arange(ca)[:, None] <= np.arange(cb)[None, :])
                else:
                    ii, jj = np.divmod(np.arange(ca * cb), cb)
                block, block_bits = out[ii, jj], np.broadcast_to(out_bits, out.shape)[ii, jj]
                fn = lambda i, op=op, ea=ea, eb=eb, ii=ii, jj=jj: (  # noqa: E731
                    op, ea[int(ii[i])], eb[int(jj[i])])
                formed += block.shape[0]
                check(fn, block, block_bits)
                if not last:
                    keep(exprs, vals, zeros, fn, block, block_bits)
        groups.append((exprs, np.stack(vals) if vals else empty,
                       np.stack(zeros) if zeros else empty))
        res.formed.append(formed)
        res.kept.append(len(exprs))
    res.seconds = time.perf_counter() - t0
    res.matches.sort(key=lambda m: (size(m), show(m)))
    return res
