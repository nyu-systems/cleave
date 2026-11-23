"""``cl.*`` ops. Each dispatches on its argument: ``cl.Tensor`` records IR, torch runs eagerly."""

from __future__ import annotations

from ..ir.update_rule import searched_updates
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

    Traced, it records ``exp, sum, div`` and asks for the exp to be stabilized
    (``Program._stable_exp``): the search sees no max, and each schedule gets the shift it
    needs -- the row max where a tile holds the whole row, a running max where the row is
    looped over, the max of each split where the row is split. Eagerly it is the max written
    out. (``cl.exp(x) / cl.sum(cl.exp(x), dim, keepdim=True)`` written by hand makes no such
    request: it is the plain exp, and overflows where its logits do.)

    A row of any length is served where the softmax feeds a sum or a matmul over that row
    (attention's ``softmax(...) @ v``). A softmax that is returned itself -- a standalone
    softmax, attention weights -- is served while its row fits one block. Next to another
    ``exp`` of the same tensor the search computes that ``exp`` once and the lowering
    computes it again for the softmax (``cleave.lower.unmerge``); the softmax is a returned
    one there too. ``cl.softmax(x) + cl.exp(x)`` has plans for float16 rows of 4096 and none
    at 8192 (where a softmax alone still has one). A second softmax of ``x`` over another
    axis needs a row and a column in one block, which is the whole tensor: 16x32 and 64x128
    have plans, 64x256 has none."""
    if _traced(x) and not searched_updates():
        e = x.program._stable_exp(x, dim)
    else:
        # Dormant until the switch-over (``CLEAVE_SEARCHED_UPDATES``): the max is recorded,
        # an ordinary reduction of the searched flat graph, and a loop over the softmax
        # dim carries it with an update rule that is searched and proven
        # (``cleave.pipeline.update``, ``cleave.schedule.streaming``).
        e = exp(x - max(x, dim, keepdim=True))
    return e / sum(e, dim, keepdim=True)
