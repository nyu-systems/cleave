"""Bounded bottom-up enumeration of update expressions for one accumulator hole (DESIGN 6).

An expression (``cleave.ir.update_rule``) is a tree over

    leaves   the hole's own old value ``a``, the full-extent value before and after the
             iteration ``m`` / ``m'``, and the tile's finished result for this hole ``r``;
    ops      + - * /  and the unary functions the flat graph applies between the
             full-extent value and its accumulators (``exp`` for a softmax). No ``max``, no
             constants, no reduction, no matmul: the tile computation is not searched.

Typing: a value is a *row* block (the shape of the full-extent value) or a *wide* one (the
hole's shape, rows first). ``row (op) wide`` broadcasts the row over the trailing axes
(what ``BinaryElementwise(broadcast=0)`` and ``Rescale(lead=...)`` do in a kernel); nothing
else broadcasts. The root has the hole's shape.

Size = number of op nodes of the tree. Level ``k`` holds the values of every tree of size
``k``: unary on level ``k-1``, binary on levels ``a + b = k - 1`` (commutative ops form each
unordered pair once). Every formed candidate is compared with the target BEFORE the merge,
and every one within the loose filter is collected: the search never stops at a match.

Intermediates are merged BY VALUE: equal float32 bits on every reference row. A 64-bit
fingerprint of the bits selects a bucket, and equality is then confirmed on the stored bits
(a fingerprint collision never drops a distinct value); the first expression wins. A value
with a non-finite entry on any reference row is neither kept nor a match (D3).

Deterministic: fixed leaf order (``a, m, m', r``), op order, level order; numpy float32
elementwise on the CPU. Everything is float32.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ...ir.update_rule import show, size

BINARY = ("add", "sub", "mul", "div")
COMMUTATIVE = ("add", "mul")
_NP = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}
_UNARY = {"exp": np.exp}
LEAF_ORDER = ("a", "m", "m'", "r")

# The first filter: deliberately loose. A wrong update that slips through is dropped by the
# proof; a right one that fails here is lost. A search filter, not a declared tolerance.
LOOSE_RTOL = 2e-2


@dataclass
class Result:
    hole: str
    budget: int
    leaves: list
    unary: tuple
    visited: int = 0  # candidate expressions formed and compared with the target
    kept: list = field(default_factory=list)  # distinct values kept per level
    formed: list = field(default_factory=list)  # candidates formed per level
    matches: list = field(default_factory=list)  # (expr, max relative error), sorted
    collisions: int = 0  # fingerprints shared by distinct values (each kept)
    seconds: float = 0.0


def _hash(block: np.ndarray) -> np.ndarray:
    """A 64-bit fingerprint per row of a 2-D float32 block (of its exact bits)."""
    bits = np.ascontiguousarray(block).view(np.uint32).astype(np.uint64)
    weights = (np.arange(1, bits.shape[1] + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
               + np.uint64(0xD6E8FEB86659FD93))
    return (bits * weights).sum(axis=1) ^ (bits[:, 0] << np.uint64(17))


class _Seen:
    """The distinct values of one type (row / wide): fingerprint -> the stored bits of
    every value with that fingerprint."""

    def __init__(self):
        self.buckets: dict = {}
        self.collisions = 0

    def known(self, h: int, bits: np.ndarray) -> bool:
        """Whether a value with exactly these bits is already in (fingerprint ``h``)."""
        return any(np.array_equal(other, bits) for other in self.buckets.get(h, ()))

    def add(self, h: int, bits: np.ndarray) -> None:
        """Record a value that is not :meth:`known` (``bits`` is kept: pass a copy)."""
        bucket = self.buckets.setdefault(h, [])
        self.collisions += bool(bucket)
        bucket.append(bits)


def _f32(x) -> np.ndarray:
    assert x.dtype == np.float32, f"the search computes in float32, got {x.dtype}"
    return x


def enumerate_updates(hole: str, leaves: dict, wide: dict, target: np.ndarray, *,
                      budget: int = 4, unary=("exp",), rtol: float = LOOSE_RTOL,
                      chunk_bytes: int = 256 << 20) -> Result:
    """Every expression of at most ``budget`` ops over ``leaves`` whose value is ``target``
    within ``rtol`` (relative to the largest target entry of its row) on every row.

    ``leaves``: name -> float32 array ``(n, 1)`` (row) or ``(n, w)`` (wide), in
    ``LEAF_ORDER``; ``wide``: name -> bool. ``target`` is ``(n, w)`` or ``(n, 1)``.
    """
    t0 = time.perf_counter()
    target = _f32(np.asarray(target))
    n, width = target.shape
    scale = np.abs(target).max(axis=1, keepdims=True) + np.float32(1e-30)
    _f32(scale)
    res = Result(hole, budget, list(leaves), tuple(unary))
    order = [k for k in LEAF_ORDER if k in leaves] + [k for k in leaves if k not in LEAF_ORDER]

    def as_block(v, is_wide):
        v = _f32(np.asarray(v))
        if v.shape[1] == (width if is_wide else 1):
            return v
        return np.ascontiguousarray(np.broadcast_to(v, (n, width)))

    # groups[size][is_wide] = (exprs, values (count, n, width or 1))
    groups = [{}]
    seen = {False: _Seen(), True: _Seen()}
    for is_wide in (False, True):
        names = [k for k in order if bool(wide[k]) == is_wide]
        if not names:
            continue
        vals = np.stack([as_block(leaves[k], is_wide) for k in names])
        _f32(vals)
        groups[0][is_wide] = (list(names), vals)
        flat = vals.reshape(len(names), -1)
        for i, h in enumerate(_hash(flat).tolist()):
            if not seen[is_wide].known(h, flat[i].view(np.uint32)):
                seen[is_wide].add(h, flat[i].copy().view(np.uint32))

    def check(exprs_fn, vals, out_wide):
        """Compare formed candidates with the target; record every match."""
        res.visited += vals.shape[0]
        if out_wide != (width > 1):
            return
        with np.errstate(all="ignore"):
            err = np.abs(vals - target[None]) / scale[None]
        flat = err.reshape(vals.shape[0], -1)
        worst = flat.max(axis=1)
        finite = np.isfinite(vals.reshape(vals.shape[0], -1)).all(axis=1)
        for i in np.nonzero(finite & (worst <= rtol))[0]:
            res.matches.append((exprs_fn(int(i)), float(worst[i])))

    def keep(new, out_wide, exprs_fn, vals):
        """Merge by value into ``new[out_wide]`` (first expression wins)."""
        flat = vals.reshape(vals.shape[0], -1)
        finite = np.isfinite(flat).all(axis=1)
        hashes = _hash(flat)
        store = new.setdefault(out_wide, ([], []))
        bucket = seen[out_wide]
        for i in np.nonzero(finite)[0]:
            h = int(hashes[i])
            if bucket.known(h, flat[i].view(np.uint32)):
                continue
            value = vals[i].copy()
            bucket.add(h, value.reshape(-1).view(np.uint32))
            store[0].append(exprs_fn(int(i)))
            store[1].append(value)

    for k in range(1, budget + 1):
        new, formed = {}, 0
        last = k == budget
        for is_wide, (exprs, vals) in groups[k - 1].items():  # unary on level k-1
            for op in unary:
                with np.errstate(all="ignore"):
                    out = _f32(_UNARY[op](vals))
                fn = lambda i, op=op, exprs=exprs: (op, exprs[i])  # noqa: E731
                formed += out.shape[0]
                check(fn, out, is_wide)
                if not last:
                    keep(new, is_wide, fn, out)
        for a in range(0, k):  # binary: sizes a + b = k - 1
            b = k - 1 - a
            for wa, (ea, va) in groups[a].items():
                for wb, (eb, vb) in groups[b].items():
                    out_wide = wa or wb
                    for op in BINARY:
                        commutative = op in COMMUTATIVE
                        if commutative and (a, wa) > (b, wb):
                            continue  # the unordered pair of groups once
                        same = commutative and (a, wa) == (b, wb)
                        w = width if out_wide else 1
                        per = max(1, chunk_bytes // (vb.shape[0] * n * w * 4))
                        for lo in range(0, va.shape[0], per):
                            xa = va[lo:lo + per]
                            with np.errstate(all="ignore"):
                                out = _f32(_NP[op](xa[:, None], vb[None]))  # (ca, cb, n, w)
                            ca, cb = out.shape[:2]
                            if same:  # i <= j only
                                ii, jj = np.nonzero(np.arange(lo, lo + ca)[:, None]
                                                    <= np.arange(cb)[None, :])
                            else:
                                ii, jj = np.divmod(np.arange(ca * cb), cb)
                            block = out[ii, jj]
                            fn = lambda i, op=op, ea=ea, eb=eb, ii=ii, jj=jj, lo=lo: (  # noqa: E731
                                op, ea[lo + int(ii[i])], eb[int(jj[i])])
                            formed += block.shape[0]
                            check(fn, block, out_wide)
                            if not last:
                                keep(new, out_wide, fn, block)
        level = {}
        for is_wide, (exprs, vals) in new.items():
            if exprs:
                level[is_wide] = (exprs, _f32(np.stack(vals)))
        groups.append(level)
        res.formed.append(formed)
        res.kept.append(sum(len(e) for e, _ in level.values()))
    res.collisions = seen[False].collisions + seen[True].collisions
    res.seconds = time.perf_counter() - t0
    res.matches.sort(key=lambda m: (size(m[0]), show(m[0])))
    return res
