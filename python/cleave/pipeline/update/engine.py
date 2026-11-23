"""The derivation of one loop's update rules: references -> search -> proof -> ranking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ...ir.update_rule import Streaming, repair_of, show, size
from .proof import prove
from .rank import rank
from .reference import Flat, NotStreamable, extension, moving_inputs
from .search import enumerate_updates

PREFIX_N = 16  # looped extent of the search references: tile 4, prefixes 4 / 8 / 12
RANK_CASES = ((32, "normal"), (4, "normal"))  # plus the adversarial cases of the graph, at 32


@dataclass
class Hole:
    acc: str
    search: object  # search.Result
    verdicts: list  # (expr, loose error, proof.Verdict)
    proven: list  # expressions
    shared_search: object = None  # search with the earlier holes' intermediates as leaves


@dataclass
class Derivation:
    key: str
    full: str
    loops: tuple
    holes: list = field(default_factory=list)
    ranked: list = field(default_factory=list)  # rank.Ranked, best first
    chosen: dict = field(default_factory=dict)  # accumulator -> update expression
    rules: tuple = ()  # ((accumulator, factor expression), ...) for the assembly
    refused: str = ""  # why no rule was derived ("" = derived)
    seconds: float = 0.0
    timings: dict = field(default_factory=dict)
    cases: list = field(default_factory=list)
    unary: tuple = ()

    def record(self) -> Streaming:
        return Streaming(self.loops, self.rules, self.key)


def _unary_ops(flat_graph, full, accs) -> tuple:
    """The unary functions the flat graph applies between the full-extent value and the
    accumulators under it: the search may use those, and + - * /."""
    from ...ir.ops import Exp

    by_name = {n.name: n for n in flat_graph.nodes}
    found, stack, seen = [], [by_name[full]], set()
    stop = set(accs)
    while stack:
        n = stack.pop()
        for e in n.outputs:
            d = e.dst_node
            if id(d) in seen or d.name in stop or d.name not in by_name:
                continue
            seen.add(id(d))
            if isinstance(d, Exp) and "exp" not in found:
                found.append("exp")
            stack.append(d)
    return tuple(found)


def _block(cases, pick, row_shape):
    """One float32 block ``(cases * rows, width)`` from a tensor per case."""
    out = []
    n_rows = int(np.prod(row_shape))
    for c in cases:
        x = pick(c).float().numpy().astype(np.float32)
        out.append(x.reshape(n_rows, -1))
    return np.concatenate(out, axis=0)


def _subterms(expr, out):
    """Sub-expressions of ``expr`` that read only ``m`` / ``m'`` (what a later hole may
    reuse for free)."""
    def only_m(e):
        return e in ("m", "m'") if isinstance(e, str) else all(only_m(x) for x in e[1:])

    if isinstance(expr, str):
        return
    if only_m(expr) and expr not in out:
        out.append(expr)
    for x in expr[1:]:
        _subterms(x, out)


def run(flat_graph, obstacle, rebuild, symbols, *, budget: int = 4, key: str = "",
        shared_budget: int = 2, rank_cases=None, looped=None) -> Derivation:
    from ...runtime.inputs import adversarial_cases

    d = Derivation(key, obstacle.full, tuple(obstacle.loops))
    t0 = time.perf_counter()
    try:
        flat = Flat(rebuild, symbols, obstacle.full, tuple(obstacle.loops), dtype="float32",
                    looped=looped)
    except NotStreamable as e:
        d.refused = str(e)
        return d
    if flat.structure(flat.graph(flat.tile)) != flat.structure(flat_graph):
        d.refused = "the flat graph does not re-instantiate at small shapes with the same nodes"
        return d
    d.unary = _unary_ops(flat_graph, obstacle.full, obstacle.loops)

    # ---- references: benign prefixes in which the max moves (every row / some rows)
    try:
        _, _, cases = moving_inputs(flat, PREFIX_N)
        for seed in (1, 2):
            inputs = flat.sample(PREFIX_N, seed)
            cases += [extension(flat, inputs, p, f"normal seed={seed} prefix={p}")
                      for p in range(flat.tile, PREFIX_N, flat.tile)]
    except NotStreamable as e:
        d.refused = str(e)
        d.seconds = time.perf_counter() - t0
        return d
    d.cases = [(c.label, c.moved) for c in cases]
    row_shape = tuple(cases[0].old[obstacle.full].shape)
    m_old = _block(cases, lambda c: c.old[obstacle.full], row_shape)
    m_new = _block(cases, lambda c: c.new[obstacle.full], row_shape)
    d.timings["references"] = time.perf_counter() - t0

    # ---- holes, narrowest first (a later hole may reuse an earlier one's intermediates)
    order = sorted(obstacle.loops, key=lambda a: (cases[0].old[a].dim(), a))
    shared_exprs = []
    for acc in order:
        t1 = time.perf_counter()
        wide = cases[0].old[acc].dim() > len(row_shape)
        leaves = {
            "a": _block(cases, lambda c: c.old[acc], row_shape),
            "m": m_old, "m'": m_new,
            "r": _block(cases, lambda c: c.tile[acc], row_shape),
        }
        kinds = {"a": wide, "m": False, "m'": False, "r": wide}
        target = _block(cases, lambda c: c.new[acc], row_shape)
        found = enumerate_updates(acc, leaves, kinds, target, budget=budget, unary=d.unary)
        d.timings[f"search {acc}"] = time.perf_counter() - t1
        t1 = time.perf_counter()
        verdicts = [(e, err, prove(e, flat_graph, obstacle.full, acc)) for e, err in found.matches]
        d.timings[f"proof {acc}"] = time.perf_counter() - t1
        hole = Hole(acc, found, verdicts, [e for e, _, v in verdicts if v.proved])
        # with the earlier holes' m-only intermediates as free leaves (cost sharing)
        if shared_exprs:
            t1 = time.perf_counter()
            from .rank import evaluate
            import torch

            env = {"m": torch.from_numpy(m_old), "m'": torch.from_numpy(m_new)}
            extra = {f"u{i}": evaluate(e, env).numpy() for i, e in enumerate(shared_exprs)}
            hole.shared_search = enumerate_updates(
                acc, {**leaves, **extra}, {**kinds, **{k: False for k in extra}}, target,
                budget=shared_budget, unary=d.unary)
            d.timings[f"shared search {acc}"] = time.perf_counter() - t1
        d.holes.append(hole)
        if not hole.proven:
            d.refused = f"no proven update for accumulator {acc} within {budget} ops"
            d.seconds = time.perf_counter() - t0
            return d
        for e in hole.proven:
            _subterms(e, shared_exprs)

    # ---- ranking of the combinations, in the program's own dtype
    t1 = time.perf_counter()
    dtype = getattr(flat_graph, "program_dtype", None) or _program_dtype(flat_graph)
    kernel = Flat(rebuild, symbols, obstacle.full, tuple(obstacle.loops), dtype=dtype,
                  looped=flat.looped)
    if rank_cases is None:
        rank_cases = list(RANK_CASES) + [(32, c) for c in adversarial_cases(kernel.graph(32))]
    combos = [{}]
    for hole in d.holes:
        combos = [{**c, hole.acc: e} for c in combos for e in hole.proven]
    d.ranked = rank(kernel, combos, rank_cases)
    d.timings["ranking"] = time.perf_counter() - t1
    best = d.ranked[0]
    if not best.finite:
        d.refused = "every proven combination is non-finite on an adversarial case"
        d.seconds = time.perf_counter() - t0
        return d
    d.chosen = dict(best.rules)
    rules = []
    for acc in obstacle.loops:
        factor = repair_of(d.chosen[acc])
        if factor is None:
            d.refused = (f"the chosen update of {acc}, {show(d.chosen[acc])}, is not "
                         "'accumulator * factor + tile'")
            d.seconds = time.perf_counter() - t0
            return d
        rules.append((acc, factor))
    d.rules = tuple(rules)
    d.seconds = time.perf_counter() - t0
    return d


def _program_dtype(flat_graph) -> str:
    """The narrowest float dtype a placeholder of the flat graph is stored in."""
    from ...ir.graph import PlaceHolderNode

    dtypes = {str(n.get_dtype(0)) for n in flat_graph.nodes if isinstance(n, PlaceHolderNode)}
    for name in ("float16", "bfloat16", "float32"):
        if name in dtypes:
            return name
    return "float32"


def describe(d: Derivation, out=print) -> None:
    """The whole derivation, for a log."""
    out(f"loop: full-extent value {d.full}; accumulators {list(d.loops)}; key {d.key[:16]}")
    out(f"unary ops read off the flat graph: {list(d.unary)}; binary: + - * /")
    out("reference cases (label, share of rows whose max moved):")
    for label, moved in d.cases:
        out(f"    {label}: {moved:.2f}")
    for hole in d.holes:
        s = hole.search
        out(f"\nhole {hole.acc}: leaves {s.leaves}, budget {s.budget} ops")
        out(f"    candidates formed per level {s.formed} (total {s.visited}), distinct values kept "
            f"per level {s.kept}, {s.seconds:.2f} s")
        out(f"    {len(s.matches)} expression(s) pass the loose filter:")
        for e, err, v in hole.verdicts:
            verdict = "PROVED " if v.proved else "rejected"
            detail = "" if v.proved else "  [" + ", ".join(f"{k.split()[0]} {r}" for k, r in v.results.items()) + (f" {v.why}" if v.why else "") + "]"
            out(f"      {verdict} {size(e)} ops  loose err {err:.1e}  z3 {v.seconds * 1e3:6.1f} ms  "
                f"{hole.acc}' = {show(e)}{detail}")
        if hole.shared_search is not None:
            s2 = hole.shared_search
            out(f"    with earlier holes' intermediates as leaves {s2.leaves}: budget {s2.budget}, "
                f"{s2.visited} candidates, {s2.seconds:.2f} s, matches:")
            for e, err in s2.matches:
                out(f"      {size(e)} ops  loose err {err:.1e}  {hole.acc}' = {show(e)}")
    if d.ranked:
        out(f"\nranking of {len(d.ranked)} proven combination(s) "
            "(finite, accuracy class, structurally stable, cost in distinct ops):")
        for i, r in enumerate(d.ranked):
            worst = max(r.errors.values())
            out(f"  #{i} finite={r.finite} accuracy={r.accuracy} stable={r.stable} cost={r.cost} "
                f"worst err {worst:.2e}   {r.text}")
        out("  errors of the best per case: " + ", ".join(f"{k} {v:.1e}" for k, v in d.ranked[0].errors.items()))
    if d.refused:
        out(f"\nREFUSED: {d.refused}")
    else:
        out("\nchosen: " + "; ".join(f"{k}' = {show(v)}" for k, v in d.chosen.items()))
        out("factor per accumulator (acc *= factor, then its own macro adds the tile): "
            + "; ".join(f"{k}: {show(f)}" for k, f in d.rules))
    out("timings (s): " + ", ".join(f"{k} {v:.2f}" for k, v in d.timings.items())
        + f"; total {d.seconds:.2f}")
