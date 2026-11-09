"""``cl.*`` ops. Each dispatches on its argument: ``cl.Tensor`` records IR, torch runs eagerly."""

from __future__ import annotations

from .program import grouped_matmul
from .tensor import Tensor


def _traced(x):
    return isinstance(x, Tensor)


def exp(x):
    return x.exp() if _traced(x) else __import__("torch").exp(x)


def sqrt(x):
    return x.sqrt() if _traced(x) else __import__("torch").sqrt(x)


def silu(x):
    return x.silu() if _traced(x) else __import__("torch").nn.functional.silu(x)


def rsqrt(x):
    """``1 / sqrt(x)`` (the tile IR has no rsqrt; lowering re-fuses ``div(1, sqrt)``)."""
    return 1.0 / sqrt(x)


def matmul(a, b):
    """``a @ b``; batch dims broadcast grouped (GQA: 32 query heads over 8 kv heads)."""
    return a @ b if _traced(a) or _traced(b) else grouped_matmul(a, b)


def sum(x, dim: int = -1, keepdim: bool = False):  # noqa: A001
    return x.sum(dim, keepdim=keepdim)


def max(x, dim: int = -1, keepdim: bool = False):  # noqa: A001
    """Max over ``dim`` (values only); traced, it needs ``keepdim=True``."""
    return x.max(dim, keepdim=keepdim) if _traced(x) else x.amax(dim, keepdim=keepdim)


def mean(x, dim: int = -1, keepdim: bool = False):
    return x.mean(dim, keepdim=keepdim)


def rms_norm(x, dim: int = -1, eps: float = 0.0):
    """``x * rsqrt(mean(x*x) + eps)`` -- the demos' primitive chain, without weight."""
    ms = mean(x * x, dim, keepdim=True)
    if eps:
        ms = ms + eps
    return x * rsqrt(ms)


def softmax(x, dim: int = -1, shift: float | None = None):
    """``shift=None``: the exact ``exp(x - max(x)) / sum(...)``. A constant
    ``shift`` is the fixed-shift ``exp(x + shift) / sum(exp(x + shift))`` (no max op)."""
    if shift is None:
        e = exp(x - max(x, dim, keepdim=True))
    else:
        e = exp(x + shift) if shift else exp(x)
    return e / sum(e, dim, keepdim=True)
