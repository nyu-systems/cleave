"""Proof of an update rule over the reals (Python z3).

Float matching cannot tell a right update from one that is merely close on the sampled
prefixes, so every survivor of the search is proved, or dropped.

Symbols. ``m`` / ``m1``: the full-extent value before / after an iteration, free reals (no
``m1 >= m`` is assumed). ``a``, ``b``: accumulator values. ``r``: the tile's finished result.
One free real per value of the flat graph that one element's contribution is made of and
that does not depend on the full-extent value (a score, a value-matrix entry).

The contribution of ONE element to an accumulator, ``c(x.., m)``, is read off the flat
graph (:func:`contribution`): the operand of the looped reduction, as a scalar term in the
full-extent value. For a softmax row sum that is ``exp(s - m)``; for the value matmul
``exp(s - m) * v``. It is never written down by hand.

``exp`` is an uninterpreted function with the law ``exp(x + y) = exp(x) * exp(y)`` and
``exp(0) = 1``. z3 does not instantiate a quantified law modulo arithmetic, so the instances
are generated here: for every two exp arguments ``x``, ``y`` that occur (and 0),
``exp(x) = exp(x - y) * exp(y)``; and ``exp(t) > 0`` for every argument (a consequence of
the two axioms: ``exp(t) = exp(t/2)^2`` and ``exp(t) * exp(-t) = 1``).

Obligations for an update ``U(a, m, m1, r)``, with ``G(a) := U(a, m, m1, r) - r``:

    (0) accumulate form     G does not depend on r           U(a,..,r) - r = U(a,..,r2) - r2
    (i) one element         G(c(x.., m)) = c(x.., m1)
    (ii) distributes        G(a + b) = G(a) + G(b)

(0) says the update is "repair, then accumulate"; (i) and (ii) are the two identities of the
plan. Together: ``G(sum_j c(x_j, m)) = sum_j c(x_j, m1)`` for any number of elements, hence
``U(F(H), T) = F(H ++ T)`` for any prefix, and by induction over the loop for any number of
tiles. ``G(0) = 0`` follows from (ii), which is the first iteration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

TIMEOUT_MS = 5000


@dataclass
class Verdict:
    proved: bool
    results: dict  # obligation -> "unsat" (proved) | "sat" | "unknown"
    seconds: float
    why: str = ""


class NotTranslatable(Exception):
    pass


def _z3():
    import z3

    return z3


def _exp():
    z3 = _z3()
    return z3.Function("exp", z3.RealSort(), z3.RealSort())


def contribution(flat_graph, full: str, acc: str):
    """``(build, leaves)``: ``build(m)`` is the z3 term of one element's contribution to
    ``acc`` under the full-extent value ``m``; ``leaves`` the free symbols it is made of.

    Walks the flat graph from the looped reduction back. A node that is not computed from
    the full-extent node is one free real (one element of it). Elementwise ops become
    arithmetic; ``Exp`` the uninterpreted ``exp``; a ``ReduceSum`` contributes its operand,
    a ``Matmul`` the product of its operands (the contraction is the sum that is looped).
    Anything else is not translatable, and the hole has no proof.
    """
    from ...ir.ops import Add, Div, Exp, Matmul, Mul, ReduceSum, Sub

    z3 = _z3()
    by_name = {n.name: n for n in flat_graph.nodes}
    full_node, node = by_name[full], by_name[acc]

    below = set()  # names computed from the full-extent node

    def mark(n):
        for e in n.outputs:
            d = e.dst_node
            if d.name not in below and d.name in by_name:
                below.add(d.name)
                mark(d)

    mark(full_node)
    leaves = {}

    def term(n, m):
        if n is full_node:
            return m
        if n.name not in below:
            return leaves.setdefault(n.name, z3.Real(f"x_{n.name}"))
        args = [term(e.src_node, m) for e in sorted(n.inputs, key=lambda e: e.dst_id)]
        if type(n) in (Add, Sub, Mul, Div):
            if n.const is not None:
                c = z3.RealVal(repr(float(n.const)))
                if type(n) is Div and n.const_on_left:
                    return c / args[0]
                return type(n).py_op(args[0], c)
            return type(n).py_op(args[0], args[1])
        if isinstance(n, Exp):
            return _exp()(args[0])
        raise NotTranslatable(f"{type(n).__name__} {n.name} has no term")

    def build(m):
        if type(node) is ReduceSum:
            (e,) = node.inputs
            return term(e.src_node, m)
        if type(node) is Matmul:
            a, b = (e.src_node for e in sorted(node.inputs, key=lambda e: e.dst_id))
            return term(a, m) * term(b, m)
        raise NotTranslatable(f"{type(node).__name__} {node.name} is not a sum over the looped dim")

    build(z3.Real("m"))  # fill ``leaves``; raises NotTranslatable early
    return build, leaves


def translate(expr, env):
    """A searched expression as a z3 term; ``env`` maps leaf names to z3 terms."""
    if isinstance(expr, str):
        return env[expr]
    op, *args = expr
    a = [translate(x, env) for x in args]
    if op == "exp":
        return _exp()(a[0])
    return {"add": lambda x, y: x + y, "sub": lambda x, y: x - y,
            "mul": lambda x, y: x * y, "div": lambda x, y: x / y}[op](*a)


def _exp_args(term, found):
    z3 = _z3()
    if z3.is_app(term):
        if term.decl().name() == "exp":
            found[z3.simplify(term.arg(0)).sexpr()] = z3.simplify(term.arg(0))
        for child in term.children():
            _exp_args(child, found)


def _axioms(*terms):
    """Ground instances of the exp law for the arguments that occur in ``terms``."""
    z3 = _z3()
    exp = _exp()
    args = {}
    for t in terms:
        _exp_args(t, args)
    zero = z3.RealVal(0)
    axioms = [exp(zero) == 1]
    pos = dict(args)
    items = list(args.values())
    for x in items + [zero]:
        for y in items:
            if x.sexpr() == y.sexpr():
                continue
            d = z3.simplify(x - y)
            axioms.append(exp(x) == exp(d) * exp(y))
            pos[d.sexpr()] = d
    axioms += [exp(t) > 0 for t in pos.values()]
    return axioms


def _check(goal, extra=()):
    """"unsat" iff ``goal`` (a z3 Bool) follows from the exp-law instances."""
    z3 = _z3()
    s = z3.Solver()
    s.set("timeout", TIMEOUT_MS)
    s.add(*_axioms(goal, *extra))
    s.add(*extra)
    s.add(z3.Not(goal))
    return str(s.check())


def prove(expr, flat_graph, full: str, acc: str, *, old="a", shared=None) -> Verdict:
    """Prove the update ``expr`` for the accumulator ``acc``.

    Leaves of ``expr``: ``old`` (the accumulator before), ``"m"``, ``"m'"``, ``"r"`` (the
    tile's result), and the names in ``shared`` (name -> expression over ``m`` / ``m'``).
    """
    z3 = _z3()
    t0 = time.perf_counter()
    try:
        build, _ = contribution(flat_graph, full, acc)
    except NotTranslatable as e:
        return Verdict(False, {}, time.perf_counter() - t0, str(e))
    m, m1, a, b, r, r2 = z3.Reals("m m1 a b r r2")
    base = {"m": m, "m'": m1}
    for name, sub in (shared or {}).items():
        base[name] = translate(sub, base)

    def U(acc_value, tile):
        return translate(expr, {**base, old: acc_value, "r": tile})

    def G(acc_value):
        return U(acc_value, r) - r

    goals = {
        "(0) accumulate form": U(a, r) - r == U(a, r2) - r2,
        "(i) one element": G(build(m)) == build(m1),
        "(ii) distributes over sum": G(a + b) == G(a) + G(b),
    }
    results = {}
    for name, goal in goals.items():
        try:
            results[name] = _check(goal)
        except Exception as e:  # noqa: BLE001 - z3 refuses a term (division by a literal 0)
            results[name] = f"error: {type(e).__name__}"
    proved = all(v == "unsat" for v in results.values())
    return Verdict(proved, results, time.perf_counter() - t0)


def nonpositive_exp_args(expr, shared=None) -> bool:
    """Structural stability: every ``exp`` argument of ``expr`` is provably ``<= 0`` given
    ``m <= m1`` (the full-extent value only grows)."""
    z3 = _z3()
    m, m1, a, r = z3.Reals("m m1 a r")
    base = {"m": m, "m'": m1}
    for name, sub in (shared or {}).items():
        base[name] = translate(sub, base)
    term = translate(expr, {**base, "a": a, "r": r})
    args = {}
    _exp_args(term, args)
    for arg in args.values():
        s = z3.Solver()
        s.set("timeout", TIMEOUT_MS)
        s.add(m <= m1, z3.Not(arg <= 0))
        if str(s.check()) != "unsat":
            return False
    return True
