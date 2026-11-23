"""The derivation of one loop's update rules: references -> search -> proof -> ranking.

``run`` never raises: every reason an obstacle keeps its pin is a typed refusal
(``texts``), and an unexpected exception is ``E13``, which names itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ...ir.update_rule import Streaming, repair_of, show, size
from . import proof as proof_module
from . import rank as rank_module
from . import reference as ref
from . import search as search_module
from .texts import E1, E11, E12, Refused, bug

BUDGETS = (4, 5)  # 5 only for a hole with no proven emittable rule at 4
RANK_CASES = ((32, "normal"), (4, "normal"))  # plus the adversarial cases of the graph, at 32
RANK_N = 32


@dataclass
class Hole:
    acc: str
    wide: bool
    leaves: dict  # name -> float32 block
    target: np.ndarray  # float32 block of F(H ++ T)[acc]
    search: object = None  # search.Result of the last budget tried
    verdicts: list = field(default_factory=list)  # (expr, loose error, proof.Verdict)
    proven: list = field(default_factory=list)  # expressions
    emittable: list = field(default_factory=list)  # proven, of the form a * F + r
    tried: list = field(default_factory=list)  # (search.Result, verdicts) per budget


@dataclass
class Derivation:
    key: str
    full: str
    loops: tuple
    cases: list = field(default_factory=list)  # (label, share of rows whose max moved)
    adequacy: dict = field(default_factory=dict)  # acc -> discriminating rows
    holes: list = field(default_factory=list)  # Hole
    ranked: list = field(default_factory=list)  # rank.Ranked, best first
    rejected: list = field(default_factory=list)  # (combination text, why)
    chosen: dict = field(default_factory=dict)  # accumulator -> update expression
    rules: tuple = ()  # ((accumulator, factor expression), ...), in ``loops`` order
    refused: str = ""  # why no rule was derived ("" = derived)
    timings: dict = field(default_factory=dict)
    budget: int = 0  # the largest budget a hole was searched at
    unary: tuple = ()
    shaped: str = ""  # the shaped sample of the reference set
    seconds: float = 0.0
    from_store: bool = False  # read back from ``store`` (re-proved), not derived here
    bug: bool = False  # refused with E13: an unexpected exception (never stored)
    report_text: str = ""  # the report of a derivation read from the store
    references: list = field(default_factory=list, repr=False)  # reference.Case
    flat: object = field(default=None, repr=False)  # the float32 reference instance
    kernel: object = field(default=None, repr=False)  # the program-dtype ranking instance

    def record(self) -> Streaming:
        return Streaming(self.loops, self.rules, self.key)


def unary_ops(flat_graph, full: str, accs) -> tuple:
    """The unary functions the flat graph applies between the full-extent value and the
    accumulators under it: the search may use those, and + - * /."""
    from ...ir.ops import Exp

    by_name = {n.name: n for n in flat_graph.nodes}
    found = []
    for acc in accs:
        for name in sorted(proof_module._between(by_name, by_name[full], by_name[acc])):
            if type(by_name[name]) is Exp and "exp" not in found:
                found.append("exp")
    return tuple(found)


def hole_blocks(cases, full: str, acc: str):
    """``(leaves, kinds, target)`` of the hole of ``acc``: float32 blocks over the rows of
    every case (``a``, ``m``, ``m'``, ``r``) and whether each is wide."""
    wide = cases[0].view("old", acc).shape[1] > 1
    leaves = {
        "a": ref.block(cases, "old", acc),
        "m": ref.block(cases, "old", full),
        "m'": ref.block(cases, "new", full),
        "r": ref.block(cases, "tile", acc),
    }
    kinds = {"a": wide, "m": False, "m'": False, "r": wide}
    return leaves, kinds, ref.block(cases, "new", acc), wide


def _looped(recipe, flat_graph, full: str):
    looped = recipe.looped(flat_graph, full)
    if looped is None:
        looped = ref.probe_looped(recipe, full)
    if looped is None:
        raise Refused(E1.format(full=full))
    return looped


def run(flat_graph, obstacle, recipe, *, key: str = "", budgets=BUDGETS, rank_cases=None,
        combine=None) -> Derivation:
    """The derivation for one obstacle of ``flat_graph``; ``recipe`` builds the graph
    again at other extents (``instantiate.Recipe``)."""
    d = Derivation(key, obstacle.full, tuple(obstacle.loops))
    t0 = time.perf_counter()
    try:
        _run(d, flat_graph, recipe, budgets, rank_cases, combine)
    except Refused as e:
        d.refused = str(e)
    except Exception as e:  # noqa: BLE001 - a bug in the engine keeps the pin and says so
        d.refused, d.bug = bug(e), True
    d.seconds = time.perf_counter() - t0
    return d


def _run(d, flat_graph, recipe, budgets, rank_cases, combine):
    full, loops = d.full, d.loops
    t0 = time.perf_counter()
    # ---- what the proof can read, before anything is built (T5, E7, E8, E9, E14)
    for acc in loops:
        proof_module.check_contribution(flat_graph, full, acc)
    for acc in loops:
        proof_module.check_leaves(flat_graph, full, acc)
    d.unary = unary_ops(flat_graph, full, loops)

    # ---- references (E1-E4 instantiation, E6 rows, E5 cases, E10 adequacy)
    looped = _looped(recipe, flat_graph, full)
    flat = d.flat = ref.Flat(recipe, flat_graph, full, loops, looped, dtype="float32")
    ref.row_dependency(flat)
    refs = ref.reference_set(flat, check=False)
    d.references, d.adequacy, d.shaped = refs.cases, refs.adequacy, refs.shaped
    d.cases = [(c.label, c.moved) for c in refs.cases]
    ref.check_adequacy(refs.cases, full, loops, refs.rows)
    d.timings["references"] = time.perf_counter() - t0

    # ---- one hole per accumulator, narrowest first; independent of each other
    order = sorted(loops, key=lambda a: (refs.cases[0].view("old", a).numel(), a))
    holes = {}
    for acc in order:
        leaves, kinds, target, wide = hole_blocks(refs.cases, full, acc)
        hole = Hole(acc, wide, leaves, target)
        for budget in budgets:
            t1 = time.perf_counter()
            found = search_module.enumerate_updates(acc, leaves, kinds, target, budget=budget,
                                                    unary=d.unary)
            d.timings[f"search {acc} @{budget}"] = time.perf_counter() - t1
            t1 = time.perf_counter()
            verdicts = [(e, err, proof_module.prove(e, flat_graph, full, acc))
                        for e, err in found.matches]
            d.timings[f"proof {acc} @{budget}"] = time.perf_counter() - t1
            hole.tried.append((found, verdicts))
            hole.search, hole.verdicts = found, verdicts
            hole.proven = [e for e, _, v in verdicts if v.proved]
            hole.emittable = [e for e in hole.proven if repair_of(e) is not None]
            d.budget = max(d.budget, budget)
            if hole.emittable:
                break
        holes[acc] = hole
        d.holes.append(hole)
        if not hole.emittable:
            raise Refused(E11.format(acc=acc, budget=hole.search.budget,
                                     matched=len(hole.search.matches), proved=len(hole.proven)))

    # ---- ranking of the combinations, in the program's own dtype
    t1 = time.perf_counter()
    kernel = d.kernel = ref.Flat(recipe, flat_graph, full, loops, looped, dtype=recipe.dtype,
                                 small=dict(flat.small))
    if rank_cases is None:
        from ...runtime.inputs import adversarial_cases

        rank_cases = list(RANK_CASES) + [(RANK_N, c) for c in adversarial_cases(kernel.graph(RANK_N))]
    reducer = type(flat.node(flat.tile, full)).combine
    d.ranked, rejected = rank_module.rank(kernel, loops, {a: holes[a].emittable for a in loops},
                                          rank_cases, combine=combine, reducer=reducer)
    d.rejected = [(r.text, r.rejected) for r in rejected]
    d.timings["ranking"] = time.perf_counter() - t1
    if not d.ranked:
        first = rejected[0]
        case = first.case
        if first.replays[case].finite:  # rejected for its max, not for a value
            case += f", where the carried {full} is not the flat graph's"
        raise Refused(E12.format(loops=", ".join(loops), case=case))
    best = d.ranked[0]
    d.chosen = dict(best.rules)
    d.rules = tuple((acc, best.factors[acc]) for acc in loops)


# --------------------------------------------------------------------------------- report


def describe(d: Derivation, out=print, *, timings: bool = True) -> None:
    """The whole derivation, for a log; without ``timings`` it is the same text on every
    run (``report``)."""
    if d.from_store:
        for line in d.report_text.splitlines():
            out(line)
        if timings:
            out(f"(read from the update store and re-proved, {d.seconds:.2f} s)")
        return
    out(f"loop: full-extent value {d.full}; accumulators {list(d.loops)}; key {d.key[:16]}")
    out(f"unary ops read off the flat graph: {list(d.unary)}; binary: + - * /")
    if d.cases:
        out(f"reference cases (label, share of rows whose full-extent value moved); "
            f"shaped sample: {d.shaped}")
        for label, moved in d.cases:
            out(f"    {label}: {moved:.2f}")
        out("discriminating rows per hole: "
            + ", ".join(f"{a} {n}" for a, n in sorted(d.adequacy.items())))
    for hole in d.holes:
        for s, verdicts in hole.tried:
            out(f"\nhole {hole.acc} ({'wide' if hole.wide else 'row'}): leaves {s.leaves}, "
                f"budget {s.budget} ops")
            out(f"    candidates formed per level {s.formed} (total {s.visited}), distinct "
                f"values kept per level {s.kept}, fingerprint collisions {s.collisions}")
            out(f"    {len(s.matches)} expression(s) pass the loose filter:")
            for e, err, v in verdicts:
                verdict = "PROVED  " if v.proved else "rejected"
                emit = (f"emittable, factor {show(repair_of(e))}" if repair_of(e) is not None
                        else "not emittable")
                detail = ("" if v.proved else "  [" + ", ".join(
                    f"{k.split()[0]} {r}" for k, r in v.results.items()) + (
                    f" {v.why}" if v.why else "") + "]")
                out(f"      {verdict} {size(e)} ops  loose err {err:.1e}  {hole.acc}' = "
                    f"{show(e)}  {emit if v.proved else ''}{detail}".rstrip())
    if d.ranked or d.rejected:
        out(f"\nranking of {len(d.ranked) + len(d.rejected)} combination(s) of emittable "
            "proven rules (grade, stable, cost, size):")
        for text, why in d.rejected:
            out(f"  REJECTED ({why}): {text}")
        for i, r in enumerate(d.ranked):
            out(f"  #{i} grade={r.grade} stable={r.stable} cost={r.cost} size={r.size}  {r.text}")
            out("       errors: " + ", ".join(f"{c} {v:.2e}" for c, v in r.errors.items()))
    if d.refused:
        out(f"\nREFUSED: {d.refused}")
    else:
        out("\nchosen: " + "; ".join(f"{k}' = {show(v)}" for k, v in sorted(d.chosen.items())))
        out("factor per accumulator (acc *= factor, then its own macro adds the tile): "
            + "; ".join(f"{k}: {show(f)}" for k, f in d.rules))
    if timings:
        out("timings (s): " + ", ".join(f"{k} {v:.2f}" for k, v in d.timings.items())
            + f"; total {d.seconds:.2f}")


def report(d: Derivation) -> str:
    """``describe`` without timings: equal for equal derivations."""
    lines = []
    describe(d, lines.append, timings=False)
    return "\n".join(lines)
