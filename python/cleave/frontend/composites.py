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


def log(x):
    """Natural log; traced, it is reference-only (a ``cl.Sample`` fn), never lowered to a kernel."""
    return x.log() if _traced(x) else __import__("torch").log(x)


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


def softmax(x, dim: int = -1):
    """The exact softmax, ``exp(x - max(x)) / sum(...)`` over ``dim``.

    Traced and eager alike it is the max written out: it records ``reduce_max, sub, exp,
    reduce_sum, div``, exactly what ``cl.exp(x - cl.max(x, dim, keepdim=True))`` divided by
    its sum records by hand (one program, one fingerprint). The max is an ordinary reduction
    of the searched flat graph. Where a tiling loops over the softmax dim under it, the loop
    carries the max across its iterations and repairs every sum and matmul it accumulates
    with an update rule that is searched for and tested on the program's own flat graph
    (``cleave.pipeline.update``, ``cleave.schedule.streaming``). (``cl.exp(x) /
    cl.sum(cl.exp(x), dim, keepdim=True)`` without the max is the plain exp, and overflows
    where its logits do.)

    A row of any length is served where the softmax feeds a sum or a matmul over that row
    (attention's ``softmax(...) @ v``). A softmax that is returned itself -- a standalone
    softmax, attention weights, a softmax next to another ``exp`` of its logits -- is served
    while its row fits one block, and so is a second softmax of ``x`` over another axis, which
    needs a row and a column in one block."""
    e = exp(x - max(x, dim, keepdim=True))
    return e / sum(e, dim, keepdim=True)
