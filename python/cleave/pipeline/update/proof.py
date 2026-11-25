"""Proof of an update rule over the reals (Python z3).

Float matching cannot tell a right update from one that is merely close on the sampled
prefixes, so every survivor of the search is proved, or dropped.

Symbols. ``m`` / ``m1``: the full-extent value before / after an iteration, FREE reals (no
``m1 >= m`` is assumed, so a proof also serves a split's merge). ``a``, ``b``: accumulator
values; ``r``, ``r2``: tile results. One free real ``x_<name>`` per value of the flat graph
that one element's contribution is made of and that is not computed from the full-extent
value (a score, a value-matrix entry).

The contribution of ONE element to an accumulator, ``c(x.., m)``, is read off the program's
own flat graph (:func:`contribution`). Only a fragment of the IR is read (``E7``-``E9``,
``T5``): exact ``Add`` / ``Sub`` / ``Mul`` / ``Div`` / ``Exp`` between the full-extent value
and an accumulator that sums from zero (``update_rule.accumulates``: a sum or a matmul);
constants exactly, as the rational the float is. Nothing is assumed of the full-extent
value's own reducer: ``m`` and ``m1`` are free.

``exp`` is an uninterpreted function. Every lemma below is a theorem about the real ``exp``
(``exp(x + y) = exp(x) exp(y)``, ``exp(0) = 1``), so a proof from them is a proof with the
law; they are ground instances for the set ``A`` of (simplified) exp arguments in a goal:

    L0  exp(0) = 1
    L1  exp(x) = exp(x - y) * exp(y)                 x in A + {0}, y in A, x != y
    L3  exp(t) * prod_{k<0} exp(s)^|k| = prod_{k>0} exp(s)^k * exp(c)
                                                     t in A linear with integer coefficients
    L2  exp(t) > 0                                   every argument above

Obligations for a candidate ``U(a, m, m1, r)``, with ``G(a) := U(a, m, m1, r) - r``; each is
valid iff ``lemmas and not goal`` is unsat:

    (D)  defined          for every division p / q in U: q != 0   (z3's division is total)
    (0)  accumulate form  U(a, .., r) - r == U(a, .., r2) - r2
    (i)  one element      G(c(x.., m)) == c(x.., m1), assuming q != 0 for every divisor q
                          of c not computed from the full-extent value (``fixed_divisors``)
    (ii) distributes      G(a + b) == G(a) + G(b)

Together: ``G(sum_j c(x_j, m)) = sum_j c(x_j, m1)`` for any number of elements, hence
``U(F(H), T) = F(H ++ T)`` for any prefix and, by induction, any number of tiles; ``G(0) = 0``
(from (ii)) is the first iteration, whatever the start value is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction

from .texts import E7, E8, E9, E14, E16, T5, Refused

TIMEOUT_MS = 5000
OBLIGATIONS = ("(D) defined", "(0) accumulate form", "(i) one element",
               "(ii) distributes over sum")


@dataclass
class Verdict:
    proved: bool
    results: dict = field(default_factory=dict)  # obligation -> unsat | sat | unknown | error: ..
    seconds: float = 0.0
    why: str = ""

    def failed(self) -> list:
        """The obligations that were not proved."""
        return [k for k, v in self.results.items() if v != "unsat"]


_Z3 = {}


def _z3():
    import z3

    return z3


def _exp():
    """The uninterpreted ``exp`` (one per process)."""
    if "exp" not in _Z3:
        z3 = _z3()
        _Z3["exp"] = z3.Function("exp", z3.RealSort(), z3.RealSort())
    return _Z3["exp"]


def const(c) -> object:
    """A float constant as the exact rational it is: ``Fraction(float(c))``, never a
    rounded decimal (``0.1`` is ``3602879701896397 / 2**55``)."""
    f = Fraction(float(c))
    return _z3().Q(f.numerator, f.denominator)


# ------------------------------------------------------------------- one element's term


def _between(by_name, full_node, acc_node) -> set:
    """Names of the nodes computed from ``full_node`` that ``acc_node`` reads (strictly
    between the two)."""
    below = set()
    stack = [full_node]
    while stack:
        n = stack.pop()
        for e in n.outputs:
            d = e.dst_node
            if d.name in by_name and d.name not in below:
                below.add(d.name)
                stack.append(d)
    above, stack = set(), [acc_node]
    while stack:
        n = stack.pop()
        for e in n.inputs:
            s = e.src_node
            if s.name not in above:
                above.add(s.name)
                stack.append(s)
    return (below & above) - {full_node.name, acc_node.name}


def _unmodelled(node) -> tuple | None:
    """``(attr, value)`` of the first attribute of ``node`` the proof does not model."""
    from ...ir.update_rule import EXTENTS, SEMANTIC

    if getattr(node, "denom_floor", None) is not None:
        return "denom_floor", node.denom_floor
    for attr in getattr(node, "_macro_attrs", ()):
        if attr not in SEMANTIC and attr not in EXTENTS:
            return attr, getattr(node, attr, None)
    return None


def check_contribution(flat_graph, full: str, acc: str) -> None:
    """The refusal (T5, E7, E8, E9) if the contribution of one element to ``acc`` cannot
    be read off the flat graph; nothing otherwise."""
    from ...ir.ops import Add, Div, Exp, Mul, Sub
    from ...ir.update_rule import accumulates

    by_name = {n.name: n for n in flat_graph.nodes}
    full_node, acc_node = by_name[full], by_name[acc]
    if not accumulates(acc_node):
        raise Refused(T5.format(d=acc, cls=type(acc_node).__name__, full=full))
    between = _between(by_name, full_node, acc_node)
    computed = between | {full}
    for node in [n for n in flat_graph.nodes if n.name in between] + [acc_node]:
        if node is not acc_node and type(node) not in (Add, Sub, Mul, Div, Exp):
            raise Refused(E7.format(node=node.name, cls=type(node).__name__, full=full, acc=acc))
        attr = _unmodelled(node)
        if attr is not None:
            raise Refused(E8.format(node=node.name, attr=attr[0], value=attr[1]))
        operands = [e.src_node.name for e in node.inputs]
        both = sum(name in computed for name in operands) > 1
        if both and (node is acc_node or getattr(node, "broadcast", -1) != -1):
            raise Refused(E9.format(node=node.name, full=full))


def check_leaves(flat_graph, full: str, acc: str) -> None:
    """E14 unless every value the contribution of one element to ``acc`` reads as a free
    real is the same on every prefix of the loop: computed from no other reduction over the
    looped dim (a second softmax's max, in ``sum(pa * pb)``). The proof's free reals are
    per-element constants; such a value is not, and the references then say so too, by
    chance of the data. Read off the provenance the scheduler pins by."""
    from ...schedule.tiling import scheduled_reductions

    nodes, red = scheduled_reductions(flat_graph)
    scheduled = {n.name: n for n in nodes}
    tokens = set()  # the looped dim, as every input that carries it into the loop names it
    for name in (full, acc):
        node = scheduled.get(name)
        if node in red and red[node]:
            tokens |= set(red[node][0])
    looping = {n.name for n in nodes if n in red and n.name not in (full, acc)
               and any(set(t) & tokens for t in red[n] if t)}
    if not looping:
        return
    _, leaves = contribution(flat_graph, full, acc)
    by_name = {n.name: n for n in flat_graph.nodes}
    for leaf in sorted(leaves):
        stack, seen = [by_name[leaf]], set()
        while stack:
            n = stack.pop()
            if n.name in looping:
                raise Refused(E14.format(node=leaf, acc=acc, other=n.name, full=full))
            for e in n.inputs:
                if e.src_node.name not in seen:
                    seen.add(e.src_node.name)
                    stack.append(e.src_node)


def contribution(flat_graph, full: str, acc: str):
    """``(build, leaves)``: ``build(m)`` is the z3 term of one element's contribution to
    ``acc`` under the full-extent value ``m``; ``leaves`` the free symbols it is made of.

    Walks the flat graph from the accumulator back. A node not computed from the
    full-extent node is one free real (one element of it); ``Add`` / ``Sub`` / ``Mul`` /
    ``Div`` become arithmetic (``const_on_left`` honoured for every class, the constant
    exact), ``Exp`` the uninterpreted ``exp``; a ``ReduceSum`` contributes its operand, a
    ``Matmul`` the product of its two operands (the contraction is the looped sum).
    Raises :class:`Refused` (T5, E7, E8, E9) for anything else.
    """
    build, leaves, _, _ = _walk(flat_graph, full, acc)
    return build, leaves


def fixed_divisors(flat_graph, full: str, acc: str) -> list:
    """The divisors of the contribution to ``acc`` that are NOT computed from the full-extent
    value: the free reals a ``Div`` between the two divides by (``exp(x - max(x)) / w``).

    Where one of them is 0 the flat graph itself is undefined, and the kernel computes the
    same inf / nan, so ``q != 0`` is a sound hypothesis of the one-element obligation (i):
    z3's division is total, and ``c / 0`` would otherwise be an arbitrary value no repair
    can relate. A divisor computed from the full-extent value (``exp(x - m) / (x - m)``) is
    never assumed: the flat graph evaluates it at the final value only, the kernel at every
    running one (:func:`check_running_divisors`)."""
    return _walk(flat_graph, full, acc)[2]


def check_running_divisors(flat_graph, full: str, acc: str) -> None:
    """E16 unless every divisor of the contribution of one element to ``acc`` that IS
    computed from the full-extent value is provably nonzero for every value of it (an
    ``exp`` of it is; the value itself, or ``x - m``, is not). The flat graph divides by such
    a divisor at the finished value only, a loop at every running one, and any update that
    repairs the accumulator would divide by it as well, so (D) of :func:`prove` holds for
    none: ``x / sum(x)``, whose running sum may be zero. Read off the flat graph alone,
    before anything is searched."""
    for node, divisor in _walk(flat_graph, full, acc)[3]:
        if valid(divisor != 0) != "unsat":
            raise Refused(E16.format(node=node, full=full, acc=acc))


def _walk(flat_graph, full: str, acc: str):
    from ...ir.ops import Div, Exp, Matmul

    z3 = _z3()
    check_contribution(flat_graph, full, acc)
    by_name = {n.name: n for n in flat_graph.nodes}
    full_node, acc_node = by_name[full], by_name[acc]
    computed = _between(by_name, full_node, acc_node) | {full}
    leaves = {}
    divisors = {}  # sexpr -> z3 term of a divisor not computed from the full-extent value
    running = {}  # sexpr -> (node, z3 term) of a divisor that is computed from it

    def term(n, m):
        if n is full_node:
            return m
        if n.name not in computed:
            return leaves.setdefault(n.name, z3.Real(f"x_{n.name}"))
        operands = [e.src_node for e in sorted(n.inputs, key=lambda e: e.dst_id)]
        args = [term(o, m) for o in operands]
        if type(n) is Exp:
            return _exp()(args[0])
        if n.const is not None:
            c = const(n.const)
            if type(n) is Div and n.const_on_left:  # c / x: x is computed from it
                running.setdefault(args[0].sexpr(), (n.name, args[0]))
            return type(n).py_op(c, args[0]) if n.const_on_left else type(n).py_op(args[0], c)
        if type(n) is Div and operands[1].name not in computed:
            divisors.setdefault(args[1].sexpr(), args[1])
        elif type(n) is Div:
            running.setdefault(args[1].sexpr(), (n.name, args[1]))
        return type(n).py_op(args[0], args[1])

    def build(m):
        operands = [e.src_node for e in sorted(acc_node.inputs, key=lambda e: e.dst_id)]
        if type(acc_node) is Matmul:
            return term(operands[0], m) * term(operands[1], m)
        return term(operands[0], m)

    build(z3.Real("m"))  # fill ``leaves``, ``divisors`` and ``running``
    return build, leaves, list(divisors.values()), list(running.values())


# --------------------------------------------------------------------- the update's term


def translate(expr, env):
    """A searched expression as a z3 term; ``env`` maps leaf names to z3 terms. The prover
    also reads ``("const", c)`` (the search never forms one)."""
    if isinstance(expr, str):
        return env[expr]
    op, *args = expr
    if op == "const":
        return const(args[0])
    a = [translate(x, env) for x in args]
    if op == "exp":
        return _exp()(a[0])
    if op not in ("add", "sub", "mul", "div"):
        raise Refused(f"the update uses {op!r}, for which no law is supplied")
    return {"add": lambda x, y: x + y, "sub": lambda x, y: x - y,
            "mul": lambda x, y: x * y, "div": lambda x, y: x / y}[op](*a)


def _divisors(expr, env, out):
    if isinstance(expr, str) or expr[0] == "const":
        return
    if expr[0] == "div":
        out.append(translate(expr[2], env))
    for x in expr[1:]:
        _divisors(x, env, out)


def _exp_args(term, found):
    z3 = _z3()
    if z3.is_app(term):
        if term.decl().name() == "exp" and term.num_args() == 1:
            arg = z3.simplify(term.arg(0))
            found[arg.sexpr()] = arg
        for child in term.children():
            _exp_args(child, found)


def linear(term):
    """``({symbol sexpr: (symbol, Fraction)}, constant)`` of a z3 term built of ``+``,
    ``-``, unary minus, multiplication by numerals, numerals and uninterpreted constants;
    None for anything else."""
    z3 = _z3()
    if z3.is_rational_value(term):
        return {}, Fraction(term.numerator_as_long(), term.denominator_as_long())
    if z3.is_const(term) and term.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        return {term.sexpr(): (term, Fraction(1))}, Fraction(0)
    if not z3.is_app(term):
        return None
    kind, kids = term.decl().kind(), term.children()
    if kind in (z3.Z3_OP_ADD, z3.Z3_OP_SUB):
        form, k = {}, Fraction(0)
        for i, child in enumerate(kids):
            sub = linear(child)
            if sub is None:
                return None
            sign = -1 if kind == z3.Z3_OP_SUB and i > 0 else 1
            for key, (sym, coef) in sub[0].items():
                form[key] = (sym, form.get(key, (sym, Fraction(0)))[1] + sign * coef)
            k += sign * sub[1]
        return form, k
    if kind == z3.Z3_OP_UMINUS:
        sub = linear(kids[0])
        return None if sub is None else (
            {key: (s, -c) for key, (s, c) in sub[0].items()}, -sub[1])
    if kind == z3.Z3_OP_MUL:
        nums = [c for c in kids if z3.is_rational_value(c)]
        rest = [c for c in kids if not z3.is_rational_value(c)]
        if len(rest) > 1:
            return None
        scale = Fraction(1)
        for n in nums:
            scale *= Fraction(n.numerator_as_long(), n.denominator_as_long())
        if not rest:
            return {}, scale
        sub = linear(rest[0])
        return None if sub is None else (
            {key: (s, c * scale) for key, (s, c) in sub[0].items()}, sub[1] * scale)
    return None


def lemmas(*terms) -> list:
    """L0-L3 for the exp arguments of ``terms``."""
    z3 = _z3()
    exp = _exp()
    args = {}
    for t in terms:
        _exp_args(t, args)
    zero = z3.RealVal(0)
    out = [exp(zero) == 1]
    positive = dict(args)
    items = list(args.values())
    for x in items + [zero]:  # L1: pairwise differences
        for y in items:
            if x.sexpr() == y.sexpr():
                continue
            d = z3.simplify(x - y)
            out.append(exp(x) == exp(d) * exp(y))
            positive[d.sexpr()] = d
    for t in items:  # L3: decomposition into the symbols, integer coefficients only
        form = linear(t)
        if form is None or not form[0]:
            continue
        coefs, k = form
        if any(c.denominator != 1 for _, c in coefs.values()):
            continue
        lhs, rhs = exp(t), z3.RealVal(1)
        if k != 0:
            kc = z3.Q(k.numerator, k.denominator)
            rhs = rhs * exp(kc)
            positive[kc.sexpr()] = kc
        for sym, c in coefs.values():
            if c == 0:
                continue
            positive[sym.sexpr()] = sym
            for _ in range(abs(int(c))):
                if c > 0:
                    rhs = rhs * exp(sym)
                else:
                    lhs = lhs * exp(sym)
        out.append(lhs == rhs)
    out += [exp(t) > 0 for t in positive.values()]  # L2: positivity
    return out


def valid(goal, *context) -> str:
    """``"unsat"`` iff ``goal`` follows from the lemmas of ``goal`` and ``context``."""
    z3 = _z3()
    s = z3.Solver()
    s.set("timeout", TIMEOUT_MS)
    s.add(*lemmas(goal, *context))
    s.add(z3.Not(goal))
    return str(s.check())


def _assuming(divisors, goal):
    """``goal`` under ``q != 0`` for every divisor (the goal itself when there is none)."""
    z3 = _z3()
    if not divisors:
        return goal
    return z3.Implies(z3.And(*[q != 0 for q in divisors]), goal)


def prove(update, flat_graph, full: str, acc: str) -> Verdict:
    """Prove the update ``update`` (leaves ``a``, ``m``, ``m'``, ``r``) of the accumulator
    ``acc`` under the full-extent node ``full`` of ``flat_graph``: (D), (0), (i), (ii)."""
    z3 = _z3()
    t0 = time.perf_counter()
    try:
        build, _, fixed, _ = _walk(flat_graph, full, acc)
        m, m1, a, b, r, r2 = z3.Reals("m m1 a b r r2")
        base = {"m": m, "m'": m1}

        def U(acc_value, tile):
            return translate(update, {**base, "a": acc_value, "r": tile})

        def G(acc_value):
            return U(acc_value, r) - r

        here = U(a, r)
        divisors = []
        _divisors(update, {**base, "a": a, "r": r}, divisors)
        goals = {
            OBLIGATIONS[0]: [(q != 0, (here,)) for q in divisors],
            OBLIGATIONS[1]: [(here - r == U(a, r2) - r2, ())],
            OBLIGATIONS[2]: [(_assuming(fixed, G(build(m)) == build(m1)), ())],
            OBLIGATIONS[3]: [(G(a + b) == G(a) + G(b), ())],
        }
    except Refused as e:
        return Verdict(False, {}, time.perf_counter() - t0, str(e))
    results = {}
    for name, checks in goals.items():
        verdict = "unsat"
        for goal, context in checks:
            try:
                got = valid(goal, *context)
            except Exception as e:  # noqa: BLE001 - z3 refuses a term (division by literal 0)
                got = f"error: {type(e).__name__}"
            if got != "unsat":
                verdict = got
                break
        results[name] = verdict
    proved = all(v == "unsat" for v in results.values())
    return Verdict(proved, results, time.perf_counter() - t0)


def nonpositive_exp_args(update, combine: str) -> bool:
    """Structural stability: every ``exp`` argument of ``update`` is provably ``<= 0``
    under what the full-extent value's reducer gives (``m <= m'`` for a max; nothing for
    another reducer). One query per argument."""
    z3 = _z3()
    m, m1, a, r = z3.Reals("m m1 a r")
    term = translate(update, {"m": m, "m'": m1, "a": a, "r": r})
    args = {}
    _exp_args(term, args)
    for arg in args.values():
        s = z3.Solver()
        s.set("timeout", TIMEOUT_MS)
        if combine == "max":
            s.add(m <= m1)
        s.add(z3.Not(arg <= 0))
        if str(s.check()) != "unsat":
            return False
    return True
