"""The statistic nodes of a rolling (online) softmax, built in one place.

A softmax whose axis is tiled by a loop keeps a running max ``m`` and, every iteration,
rescales what the loop has accumulated so far by ``alpha = exp(m_prev - m)``. The paged
builders build these nodes here (``rolling_nodes``), and the assembly of a searched update
rule builds its repair factor here (``build_expression``).
"""

from __future__ import annotations

from .ops import Add, Copy, Div, Mul, Sub, wire_input

STAT_DTYPE = "float32"  # m, m_prev, the step and alpha: always float32

_BINARY = {"add": Add, "sub": Sub, "mul": Mul, "div": Div}


def build_expression(expr, env, made, *, stem: str, rows: tuple, n_batch_dims: int, exp_cls,
                     narrow=()):
    """The IR nodes of a searched repair factor (``cleave.ir.update_rule``): ``expr`` is a
    nested tuple over the leaves of ``env`` (leaf name -> node), ``("exp", ("sub", "m", "m'"))``.

    One float32 node of shape ``rows`` per distinct subterm. ``made`` memoises (expression
    -> node), so the accumulators of one loop share their subterms; the nodes are named
    ``{stem}_upd{k}`` in the order they are first needed (``update_rule.subterms``). ``exp``
    is built with ``exp_cls``, the class of the exp the repaired values were computed with.

    A leaf named in ``narrow`` is a value WITHOUT the first non-batch axis of ``rows`` (the
    merged max of a split combine, next to the max of each split). It may only be the
    second operand of a binary op, which then broadcasts it across that axis
    (``broadcast=1``).

    ``ValueError`` for an expression no tile op computes; nothing has been built for it
    then. The caller gives the new nodes their tiles and declares what they are for
    (``Graph.lag_nodes``, ``Graph.accum_scale``).
    """
    narrow = frozenset(narrow)
    check_expression(expr, stem=stem, narrow=narrow)
    return _build(expr, env, made, stem, tuple(rows), n_batch_dims, exp_cls, narrow)


def check_expression(expr, *, stem: str, narrow=()) -> None:
    """``ValueError`` if :func:`build_expression` has no tile op for ``expr``, with the
    reason; for a caller that must know before it touches a graph."""
    if isinstance(expr, str):
        return
    op, *args = expr
    if op != "exp" and op not in _BINARY:
        raise ValueError(f"{stem}: the repair uses {op!r}, which has no tile op")
    for position, arg in enumerate(args):
        if isinstance(arg, str) and arg in narrow and (op == "exp" or position == 0):
            how = "of" if op == "exp" else "on the left of"
            raise ValueError(
                f"{stem}: the reduced value {arg!r} {how} {op!r} has no tile op")
        check_expression(arg, stem=stem, narrow=narrow)


def _build(expr, env, made, stem, rows, n_batch_dims, exp_cls, narrow):
    if isinstance(expr, str):
        return env[expr]
    if expr in made:
        return made[expr]
    op, *args = expr
    operands = [_build(a, env, made, stem, rows, n_batch_dims, exp_cls, narrow) for a in args]
    kw = dict(name=f"{stem}_upd{len(made)}", dtype=STAT_DTYPE, out_dtype=STAT_DTYPE,
              n_batch_dims=n_batch_dims)
    if op == "exp":
        node = exp_cls(rows, **kw)
    else:
        if isinstance(args[1], str) and args[1] in narrow:
            kw["broadcast"] = 1  # the reduced value, across the first non-batch axis
        node = _BINARY[op](rows, **kw)
    for position, operand in enumerate(operands):
        wire_input(operand, node, position)
    made[expr] = node
    return node


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
