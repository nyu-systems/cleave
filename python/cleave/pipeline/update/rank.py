"""Ranking of proven, emittable update rules: accuracy, stability, cost.

A proof is over the reals; two rules that are the same real function can differ by orders
of magnitude in floating point (``a * exp(m - m')`` against ``a * exp(m) / exp(m')``), so
every combination of the holes' rules is replayed as the kernel runs it:

  * the flat graph re-instantiated in the program's dtype (a float16 operand is rounded
    where the kernel rounds it), a float32 carried state starting at the reducer's start
    value and zeros, per tile ``m' = combine(m, max of the tile)``, the tile computation by
    the flat graph with ``m'`` given, each accumulator updated as the kernel does it,
    ``a * F + r`` with every subterm of ``F`` evaluated as a node, and at the end the flat
    graph's own epilogue on the final state, compared with the flat graph on the whole input;
  * on ``normal @ 32`` (8 tiles), ``normal @ 4`` (the first iteration alone) and every
    adversarial case of the graph at 32 (``runtime.inputs.adversarial_cases``).

A combination is REJECTED when, on any case, a subterm of a factor, a carried accumulator
(after any iteration) or an output is non-finite, or the carried full-extent value at the
end is not bit-equal to the flat graph's. The rest are ordered by the key
``(grade, not stable, cost, size, text)``, smallest first:

  grade   accuracy at the output's resolution: with ``rho = ulp(output dtype) + n_tiles *
          2^-23`` and ``f = max(best error on the case, rho)``, an error ``v`` has grade
          ``max(0, ceil(log4(max(v, f) / f)) - 1)``; the combination's is its largest.
          Grade 0 = within 4x of what the output can resolve.
  stable  every ``exp`` argument provably ``<= 0``, no division by and no subtraction of
          a value that uses ``a`` or ``r``
  cost    distinct op nodes of the factors taken together (what the loop pays per iteration)
  size    sum of the tree sizes of the updates
  text    the updates' texts in accumulator-name order (the deterministic tie-break)

No FP64: the replay runs in the program's dtype with a float32 state.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from ...ir.update_rule import repair_of, show, size, subterms, uses
from ...runtime.reference import values
from .reference import Flat, combine_of

ULP = {"float16": 2.0 ** -10, "bfloat16": 2.0 ** -7, "float32": 2.0 ** -23}


def evaluate(expr, env, memo=None):
    """A searched expression on torch tensors; a row operand broadcasts over the trailing
    axes of a wide one. ``memo`` (expression -> value) shares subterms."""
    import torch

    if isinstance(expr, str):
        return env[expr]
    if memo is not None and expr in memo:
        return memo[expr]
    op, *args = expr
    vals = [evaluate(a, env, memo) for a in args]
    if op == "exp":
        out = torch.exp(vals[0])
    else:
        x, y = vals
        if x.dim() < y.dim():
            x = x.reshape(x.shape + (1,) * (y.dim() - x.dim()))
        elif y.dim() < x.dim():
            y = y.reshape(y.shape + (1,) * (x.dim() - y.dim()))
        out = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}[op](x, y)
    if memo is not None:
        memo[expr] = out
    return out


@dataclass
class Replay:
    finite: bool  # every factor subterm, carried accumulator and output, at every iteration
    same_max: bool  # the carried full-extent value at the end is bit-equal to the graph's
    error: float  # largest output difference relative to the largest reference entry
    why: str = ""  # what failed first: "non-finite <what> at tile <k>" or the max


def replay(kernel: Flat, factors: dict, n: int, distribution: str, *, seed: int = 0,
           combine=None) -> Replay:
    """The loop the factors define, run over ``n // kernel.tile`` tiles of one sample."""
    import torch

    t = kernel.tile
    combine = combine or combine_of(kernel.node(t, kernel.full))
    inputs = kernel.sample(n, seed, distribution)
    ref_env, reference = values(kernel.graph(n), inputs)
    state = kernel.start({k: ref_env[k].float() for k in (kernel.full, *kernel.accs)})
    why = ""
    for k, p in enumerate(range(0, n, t)):
        own = kernel.state(inputs, p, t)
        m_old = state[kernel.full]
        m_new = combine(m_old, own[kernel.full])
        tile = kernel.state(inputs, p, t, given={kernel.full: m_new})
        env, memo = {"m": m_old, "m'": m_new}, {}
        new = {kernel.full: m_new}
        for acc in kernel.accs:
            for sub in subterms(factors[acc]):
                value = evaluate(sub, env, memo)
                if not why and not bool(torch.isfinite(value).all()):
                    why = f"non-finite {show(sub)} at tile {k}"
            f = evaluate(factors[acc], env, memo).float()
            a = state[acc]
            f = f.reshape(f.shape + (1,) * (a.dim() - f.dim()))
            new[acc] = (a * f + tile[acc]).float()
            if not why and not bool(torch.isfinite(new[acc]).all()):
                why = f"non-finite {acc} after tile {k}"
        state = new
    same_max = bool(torch.equal(state[kernel.full], ref_env[kernel.full].float()))
    _, outputs = values(kernel.graph(n), inputs, given=state)
    worst = 0.0
    for out, ref in zip(outputs, reference):
        if not why and not bool(torch.isfinite(out.float()).all()):
            why = "non-finite output"
        scale = float(ref.float().abs().max()) or 1.0
        worst = max(worst, float((out.float() - ref.float()).abs().max()) / scale)
    if not why and not same_max:
        why = f"the carried {kernel.full} is not bit-equal to the flat graph's"
    return Replay(not why.startswith("non-finite"), same_max, worst, why)


def resolution(kernel: Flat, cases) -> float:
    """``rho``: one ulp of the widest-ulp output dtype plus a float32 ulp per tile."""
    graph = kernel.graph(max(n for n, _ in cases))
    dtypes = {str(o.get_dtype(0)) for o in graph.nodes if o.is_output()}
    ulp = max(ULP.get(d, ULP["float32"]) for d in dtypes) if dtypes else ULP["float32"]
    tiles = max(n for n, _ in cases) // kernel.tile
    return ulp + tiles * ULP["float32"]


def grade(v: float, best: float, rho: float) -> int:
    """The grade of an error ``v`` on a case whose best finite error is ``best``."""
    f = max(best, rho)
    return max(0, math.ceil(math.log(max(v, f) / f) / math.log(4.0)) - 1)


def structurally_stable(update, combine: str = "max") -> bool:
    from .proof import nonpositive_exp_args

    def bad(e):
        if isinstance(e, str):
            return False
        op = e[0]
        if op == "div" and (uses(e[2], "a") or uses(e[2], "r")):
            return True  # division by an accumulator or a tile result
        if op == "sub" and any(uses(x, "a") or uses(x, "r") for x in e[1:]):
            return True  # subtraction of accumulators cancels
        return any(bad(x) for x in e[1:])

    return not bad(update) and nonpositive_exp_args(update, combine)


@dataclass
class Ranked:
    rules: dict  # accumulator -> update expression
    factors: dict  # accumulator -> factor expression (``repair_of``)
    replays: dict  # case label -> Replay
    rejected: str = ""  # "" = ranked; else "<why> on <case>"
    case: str = ""  # the first case it was rejected on
    grade: int = 0
    stable: bool = False
    cost: int = 0
    size: int = 0
    text: str = ""

    @property
    def errors(self) -> dict:
        return {c: r.error for c, r in self.replays.items()}

    @property
    def key(self) -> tuple:
        return (self.grade, not self.stable, self.cost, self.size, self.text)


def order(rows: list, rho: float) -> list:
    """Grade the rows that are not rejected and sort them by the key; the rejected ones
    are returned separately (in their input order): ``(ranked, rejected)``."""
    kept = [r for r in rows if not r.rejected]
    cases = list(rows[0].replays) if rows else []
    best = {c: min((r.replays[c].error for r in kept), default=float("inf")) for c in cases}
    for r in kept:
        r.grade = max((grade(r.replays[c].error, best[c], rho) for c in cases), default=0)
    return sorted(kept, key=lambda r: r.key), [r for r in rows if r.rejected]


def rank(kernel: Flat, loops: tuple, choices: dict, cases: list, *, combine=None,
         reducer: str = "max"):
    """Every combination of the holes' rules, ``(ranked best first, rejected)``.

    ``choices``: accumulator -> proven updates; only those the assembly can emit
    (``repair_of`` is not None) take part. ``cases``: ``(n, distribution)``."""
    emittable = {acc: [u for u in choices[acc] if repair_of(u) is not None] for acc in loops}
    rows = []
    for combo in itertools.product(*(emittable[acc] for acc in loops)):
        rules = dict(zip(loops, combo))
        factors = {acc: repair_of(u) for acc, u in rules.items()}
        replays = {f"{d}@{n}": replay(kernel, factors, n, d, combine=combine) for n, d in cases}
        case = next((c for c, r in replays.items() if r.why), "")
        rows.append(Ranked(
            rules, factors, replays, f"{replays[case].why} on {case}" if case else "", case,
            stable=all(structurally_stable(u, reducer) for u in rules.values()),
            cost=len(subterms(*factors.values())),
            size=sum(size(u) for u in rules.values()),
            text="; ".join(f"{acc}' = {show(rules[acc])}" for acc in sorted(rules))))
    return order(rows, resolution(kernel, cases))
