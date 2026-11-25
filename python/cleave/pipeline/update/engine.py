"""The derivation of one loop's update rules: test data -> enumeration -> float replay.

``run`` never raises: every reason a conflict keeps its pin is one of the refusals of
``texts`` (no test, no rule, no rule survives the replay), and an unexpected exception is
``BUG``, which names itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ...ir.rolling import check_expression
from ...ir.update_rule import Streaming, repair_of, show, size, update_of
from . import rank as rank_module
from . import reference as ref
from . import search as search_module
from .texts import NO_REPLAY, NO_RULE, Refused, bug

BUDGETS = (4, 5)  # 5 only for an accumulator with no emittable match at 4
RANK_CASES = ((32, "normal"), (4, "normal"))  # plus the adversarial cases of the graph, at 32
RANK_N = 32


@dataclass
class Hole:
    acc: str
    wide: bool
    search: object = None  # search.Result of the last budget tried
    matches: list = field(default_factory=list)  # updates a * F + r that reproduce the graph
    emittable: list = field(default_factory=list)  # of those, the ones the assembly can emit
    tried: list = field(default_factory=list)  # search.Result per budget


@dataclass
class Derivation:
    key: str
    full: str
    loops: tuple
    cases: list = field(default_factory=list)  # labels of the test cases
    holes: list = field(default_factory=list)  # Hole
    ranked: list = field(default_factory=list)  # rank.Ranked, best first
    rejected: list = field(default_factory=list)  # (combination text, why)
    chosen: dict = field(default_factory=dict)  # accumulator -> update expression
    rules: tuple = ()  # ((accumulator, factor expression), ...), in ``loops`` order
    refused: str = ""  # why no rule was derived ("" = derived)
    timings: dict = field(default_factory=dict)
    budget: int = 0  # the largest budget an accumulator was searched at
    unary: tuple = ()
    seconds: float = 0.0
    from_store: bool = False  # read back from ``store`` (tested again), not derived here
    bug: bool = False  # refused with BUG: an unexpected exception (never stored)
    # ``Recipe.key`` of a refusal: it is kept for that recipe only ("" for a rule, which
    # holds for the loop structure).
    recipe_key: str = ""
    report_text: str = ""  # the report of a derivation read from the store
    references: list = field(default_factory=list, repr=False)  # reference.Case
    flat: object = field(default=None, repr=False)  # the instances of the test data
    kernel: object = field(default=None, repr=False)  # the program-dtype replay instances

    def record(self) -> Streaming:
        return Streaming(self.loops, self.rules, self.key)

    def light(self) -> "Derivation":
        """The derivation without its instances and test data: what the process keeps per
        loop structure (the report needs none of them)."""
        from dataclasses import replace

        return replace(self, references=[], flat=None, kernel=None)


# The unary ops of the IR, by the name the search and the assembly give them.
def _unary_classes():
    from ...ir.ops import Exp, RSqrt, Silu, Sqrt

    return {Exp: "exp", Sqrt: "sqrt", RSqrt: "rsqrt", Silu: "silu"}


def between(by_name, full_node, acc_node) -> set:
    """Names of the nodes computed from ``full_node`` that ``acc_node`` reads (strictly
    between the two)."""
    below, stack = set(), [full_node]
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


def unary_ops(flat_graph, full: str, accs) -> tuple:
    """The unary functions the flat graph applies between the full-extent value and the
    accumulators under it: the search may use those, and + - * /."""
    classes = _unary_classes()
    by_name = {n.name: n for n in flat_graph.nodes}
    found = set()
    for acc in accs:
        for name in between(by_name, by_name[full], by_name[acc]):
            op = classes.get(type(by_name[name]))
            if op is not None:
                found.add(op)
    return tuple(op for op in search_module.UNARY if op in found)


_PARTS = (("a", "old", False), ("m", "old", True), ("m'", "new", True), ("r", "tile", False))


def hole_blocks(cases, full: str, acc: str):
    """``(leaves, target, wide, excluded)`` of the hole of ``acc``: field blocks over the rows
    of every case (``a``, ``m``, ``m'``, ``r``), the new state, whether the accumulator is
    wide, and per element the residues its test data is computed from a division by zero
    in (``search`` skips those). ``no_test`` where that is every residue of every element."""
    wide = cases[0].view("old", acc).shape[1] > 1
    leaves, excluded = {}, ref.zeros_block(cases, "new", acc)
    for leaf, part, of_full in _PARTS:
        name = full if of_full else acc
        leaves[leaf] = ref.block(cases, part, name)
        excluded = excluded | ref.zeros_block(cases, part, name)
    if (excluded == 3).all():
        raise ref.no_test(full, f"every element of the test data of {acc} is computed from a "
                          "division by zero in the field")
    return leaves, ref.block(cases, "new", acc), wide, excluded


def emittable(update, full: str) -> bool:
    """Whether the loop can emit ``update``: the form ``a * F + r`` with a factor the
    assembly has tile ops for."""
    factor = repair_of(update)
    if factor is None:
        return False
    try:
        check_expression(factor, stem=full)
    except ValueError:
        return False
    return True


def looped_dim(recipe, flat_graph, full: str):
    looped = recipe.looped(flat_graph, full)
    if looped is None:
        looped = ref.probe_looped(recipe, full)
    if looped is None:
        raise ref.no_test(full, "the dim it reduces has no symbol: the graph cannot be built "
                          "again on a prefix of it")
    return looped


def run(flat_graph, obstacle, recipe, *, key: str = "", budgets=BUDGETS, rank_cases=None,
        combine=None) -> Derivation:
    """The derivation for one conflict of ``flat_graph``; ``recipe`` builds the graph again
    at other extents (``instantiate.Recipe``)."""
    d = Derivation(key, obstacle.full, tuple(obstacle.loops))
    t0 = time.perf_counter()
    try:
        _run(d, flat_graph, recipe, budgets, rank_cases, combine, t0)
    except Refused as e:
        d.refused = str(e)
        d.recipe_key = recipe.key
    except Exception as e:  # noqa: BLE001 - a bug in the engine keeps the pin and says so
        d.refused, d.bug = bug(e), True
    d.seconds = time.perf_counter() - t0
    return d


def _run(d, flat_graph, recipe, budgets, rank_cases, combine, t0):
    full, loops = d.full, d.loops
    looped = looped_dim(recipe, flat_graph, full)
    flat = d.flat = ref.Flat(recipe, flat_graph, full, loops, looped, dtype="float32")
    cases = d.references = ref.reference_set(flat)
    d.cases = [c.label for c in cases]
    d.timings["test data"] = time.perf_counter() - t0
    d.unary = unary_ops(flat_graph, full, loops)

    # ---- one hole per accumulator, narrowest first; independent of each other
    order = sorted(loops, key=lambda a: (cases[0].view("old", a).size, a))
    holes = {}
    for acc in order:
        leaves, target, wide, excluded = hole_blocks(cases, full, acc)
        hole = Hole(acc, wide)
        for budget in budgets:
            t1 = time.perf_counter()
            found = search_module.enumerate_factors(acc, leaves, target, budget=budget,
                                                    unary=d.unary, cases=len(cases),
                                                    excluded=excluded)
            d.timings[f"search {acc} @{budget}"] = time.perf_counter() - t1
            hole.tried.append(found)
            hole.search = found
            hole.matches = [update_of(f) for f in found.matches]
            hole.emittable = [e for e in hole.matches if emittable(e, full)]
            d.budget = max(d.budget, budget)
            if hole.emittable:
                break
        holes[acc] = hole
        d.holes.append(hole)
        if not hole.emittable:
            raise Refused(NO_RULE.format(acc=acc, budget=hole.search.budget))

    # ---- ranking of the combinations, replayed in floating point in the program's dtype
    t1 = time.perf_counter()
    kernel = d.kernel = ref.Flat(recipe, flat_graph, full, loops, looped, dtype=recipe.dtype)
    if rank_cases is None:
        from ...runtime.inputs import adversarial_cases

        rank_cases = list(RANK_CASES) + [(RANK_N, c) for c in adversarial_cases(kernel.graph(RANK_N))]
    d.ranked, rejected = rank_module.rank(kernel, loops, {a: holes[a].emittable for a in loops},
                                          rank_cases, combine=combine)
    d.rejected = [(r.text, r.rejected) for r in rejected]
    d.timings["ranking"] = time.perf_counter() - t1
    if not d.ranked:
        raise Refused(NO_REPLAY.format(loops=", ".join(loops), case=rejected[0].rejected))
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
            out(f"(read from the update store and tested again, {d.seconds:.2f} s)")
        return
    out(f"loop: full-extent value {d.full}; accumulators {list(d.loops)}; key {d.key[:16]}")
    out(f"unary ops read off the flat graph: {list(d.unary)}; binary: + - * /")
    if d.cases:
        out(f"test data: {len(d.cases)} cases ({', '.join(d.cases)}), in the field of the "
            "first pass's fingerprint")
    for hole in d.holes:
        for s in hole.tried:
            out(f"\nhole {hole.acc} ({'wide' if hole.wide else 'row'}): factors over m, m', "
                f"updates of at most {s.budget} ops")
            out(f"    factors formed per level {s.formed} (total {s.visited}), distinct "
                f"values kept per level {s.kept}; sampled {s.elements[0]} of "
                f"{s.elements[1]} elements")
            out(f"    {s.sampled} match the sample, {len(s.matches)} every element:")
            for f in s.matches:
                e = update_of(f)
                emit = "emittable" if emittable(e, d.full) else "not emittable"
                out(f"      {size(e)} ops  {hole.acc}' = {show(e)}  {emit}")
    if d.ranked or d.rejected:
        out(f"\nranking of {len(d.ranked) + len(d.rejected)} combination(s) of emittable "
            "rules (grade, cost, size):")
        for text, why in d.rejected:
            out(f"  REJECTED ({why}): {text}")
        for i, r in enumerate(d.ranked):
            out(f"  #{i} grade={r.grade} cost={r.cost} size={r.size}  {r.text}")
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

