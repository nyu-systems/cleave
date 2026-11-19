"""The statistic nodes of a rolling (online) softmax, built in one place.

A softmax whose axis is tiled by a loop keeps a running max ``m`` and, every iteration,
rescales what the loop has accumulated so far by ``alpha = exp(m_prev - m)``. The static
rewrite and the paged builders both build these nodes here, so their names, dtypes and
wiring cannot drift apart.
"""

from __future__ import annotations

from .ops import Copy, Sub, wire_input

STAT_DTYPE = "float32"  # m, m_prev, the step and alpha: always float32


def rolling_nodes(max_node, *, stem: str, rows: tuple, n_batch_dims: int, exp_cls):
    """``(m_prev, step, alpha)`` for the carried max ``max_node``, all float32 and wired:

        m_prev = Copy(rows,    name=f"{stem}_max_prev")  <- max_node            (lag node)
        step   = Sub(rows,     name=f"{stem}_max_step")  <- (m_prev, max_node)
        alpha  = exp_cls(rows, name=f"{stem}_alpha")     <- step

    ``rows`` is the shape of the max (the exp's shape without its softmax axis) and
    ``exp_cls`` the class of the exp being stabilized, so ``alpha`` and ``P`` always use
    the same base. The caller adds the three nodes to its tier, gives them tiles, and
    declares ``lag_nodes.append(m_prev)`` and ``accum_scale[acc] = alpha``.
    """
    assert max_node.out_dtype == STAT_DTYPE, (
        f"carried max {max_node.name!r} is {max_node.out_dtype}; its start value "
        f"is a {STAT_DTYPE} number"
    )
    kw = dict(dtype=STAT_DTYPE, out_dtype=STAT_DTYPE, n_batch_dims=n_batch_dims)
    m_prev = Copy(tuple(rows), name=f"{stem}_max_prev", **kw)
    step = Sub(tuple(rows), name=f"{stem}_max_step", **kw)
    alpha = exp_cls(tuple(rows), name=f"{stem}_alpha", **kw)
    wire_input(max_node, m_prev, 0)
    wire_input(m_prev, step, 0)
    wire_input(max_node, step, 1)
    wire_input(step, alpha, 0)
    return m_prev, step, alpha
