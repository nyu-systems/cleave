"""Ranking of proven update rules: accuracy in the kernel's own precision, then a
structural stability rule, then cost.

Accuracy is measured by running the LOOP, not one step: the flat graph re-instantiated in
the program's dtype (so a float16 operand is rounded where the kernel rounds it), a float32
carried state that starts at the sentinel and zeros, many tiles, and the adversarial input
cases the validation uses (``cleave.runtime.inputs``). The final state is put back into the
flat graph and its outputs are compared with the flat graph's own. Any non-finite output
rejects the candidate. No FP64.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...ir.update_rule import show, size, subterms, uses
from ...runtime.reference import values
from .reference import Flat


def evaluate(expr, env):
    """A searched expression on torch tensors; a row operand broadcasts over the trailing
    axes of a wide one."""
    import torch

    if isinstance(expr, str):
        return env[expr]
    op, *args = expr
    vals = [evaluate(a, env) for a in args]
    if op == "exp":
        return torch.exp(vals[0])
    x, y = vals
    if x.dim() < y.dim():
        x = x.reshape(x.shape + (1,) * (y.dim() - x.dim()))
    elif y.dim() < x.dim():
        y = y.reshape(y.shape + (1,) * (x.dim() - y.dim()))
    return {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}[op](x, y)


def run_loop(flat: Flat, rules: dict, inputs, n: int, shared=None):
    """The outputs of the loop that carries ``flat.full`` and updates each accumulator with
    its rule, over ``n // flat.tile`` tiles; and the flat graph's own outputs."""
    import torch

    t = flat.tile
    reference_env, reference = values(flat.graph(n), inputs)
    state = flat.start({k: reference_env[k].float() for k in (flat.full, *flat.accs)})
    for p in range(0, n, t):
        own, _ = flat.state(inputs, p, t)
        m_old = state[flat.full]
        m_new = torch.maximum(m_old, own[flat.full])
        tile, _ = flat.state(inputs, p, t, given={flat.full: m_new})
        env = {"m": m_old, "m'": m_new}
        for name, sub in (shared or {}).items():
            env[name] = evaluate(sub, env)
        new = {flat.full: m_new}
        for acc in flat.accs:
            new[acc] = evaluate(rules[acc], {**env, "a": state[acc], "r": tile[acc]}).float()
        state = new
    _, outputs = values(flat.graph(n), inputs, given=state)
    return outputs, reference


def loop_error(flat: Flat, rules: dict, n: int, distribution: str, seed: int = 0, shared=None):
    """Largest error of the loop's outputs relative to the largest reference entry; ``inf``
    when an output is not finite."""
    import torch

    inputs = flat.sample(n, seed, distribution)
    outputs, reference = run_loop(flat, rules, inputs, n, shared)
    worst = 0.0
    for out, ref in zip(outputs, reference):
        if not bool(torch.isfinite(out.float()).all()):
            return float("inf")
        scale = float(ref.float().abs().max()) or 1.0
        worst = max(worst, float((out.float() - ref.float()).abs().max()) / scale)
    return worst


def dag_ops(*exprs) -> int:
    """Distinct op nodes of several expressions taken together (shared subterms once)."""
    return len(subterms(*exprs))


@dataclass
class Ranked:
    rules: dict  # accumulator -> update expression
    errors: dict  # distribution -> error
    finite: bool
    accuracy: int  # 0 = within 2x of the best candidate on every case; larger = worse
    stable: bool  # every exp argument <= 0, no division by / subtraction of an accumulator
    cost: int  # distinct op nodes of the combination
    text: str = ""
    key: tuple = field(default=())


def _structurally_stable(expr, shared) -> bool:
    from .proof import nonpositive_exp_args

    def bad(e):
        if isinstance(e, str):
            return False
        op = e[0]
        if op == "div" and (uses(e[2], "a") or uses(e[2], "r")):
            return True  # division by an accumulator or a tile sum
        if op == "sub" and any(uses(x, "a") or uses(x, "r") for x in e[1:]):
            return True  # subtraction of accumulators cancels
        return any(bad(x) for x in e[1:])

    return not bad(expr) and nonpositive_exp_args(expr, shared)


def rank(flat: Flat, combos: list, cases: list, shared=None) -> list:
    """``combos``: list of ``{accumulator: expression}``; ``cases``: ``(n, distribution)``.
    Returns ``Ranked`` best first. Deterministic."""
    import math

    rows = []
    for rules in combos:
        errors = {f"{d}@{n}": loop_error(flat, rules, n, d, shared=shared) for n, d in cases}
        rows.append((rules, errors))
    best = {c: min((e[c] for _, e in rows if math.isfinite(e[c])), default=float("inf"))
            for c in rows[0][1]} if rows else {}
    ranked = []
    for rules, errors in rows:
        finite = all(math.isfinite(v) for v in errors.values())
        accuracy = 0
        if finite:
            for c, v in errors.items():
                floor = max(best[c], 1e-7)  # below float32 resolution every error is "the best"
                accuracy = max(accuracy, max(0, math.ceil(math.log2(max(v, floor) / floor)) - 1))
        stable = all(_structurally_stable(e, shared) for e in rules.values())
        cost = dag_ops(*rules.values())
        text = "; ".join(f"{k}' = {show(v)}" for k, v in sorted(rules.items()))
        key = (not finite, accuracy, not stable, cost, sum(size(v) for v in rules.values()), text)
        ranked.append(Ranked(rules, errors, finite, accuracy, stable, cost, text, key))
    ranked.sort(key=lambda r: r.key)
    return ranked
