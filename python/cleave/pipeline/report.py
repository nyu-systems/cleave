"""Per-candidate results, failure histograms, measured errors, the split-K pairing rule and
``best()``."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core.errors import CleaveError, LaunchCrash, NonFiniteOutput, OutputMismatch

# How a kernel can fail the validate stage; "other" is a reference or layout problem.
VALIDATION_KINDS = {OutputMismatch: "mismatch", NonFiniteOutput: "non_finite",
                    LaunchCrash: "launch_crash"}


def matmul_operands(graph) -> tuple:
    """``(format, significant bits)`` of every format the matmuls of a lowered graph
    multiply their operands in that is not the operands' own dtype, sorted.

    Today that is ``("tf32", 11)`` for a float32 matmul (``Matmul.tensor_core_format``):
    the one case in which a kernel computes with fewer bits than its program's dtype has.
    """
    from ..ir.ops import Matmul

    found = set()
    for node in getattr(graph, "nodes", ()):
        if isinstance(node, Matmul) and node.tensor_core_format != node.dtype:
            fmt = node.tensor_core_format
            found.add((fmt, Matmul.SIGNIFICANT_BITS[fmt]))
    return tuple(sorted(found))


def validation_kind(error) -> str:
    return next((kind for cls, kind in VALIDATION_KINDS.items() if isinstance(error, cls)),
                "other")


def tolerance_dicts(tolerance) -> list | dict | None:
    """``[{"rtol", "atol"}, ...]`` for a Tolerance or a sequence of them, ``{role: {"rtol",
    "atol"}}`` for a declaration keyed by boundary role (None stays None, also as an entry
    of a sequence: an output no single bound governs)."""
    def entry(t):
        return None if t is None else {"rtol": t.rtol, "atol": t.atol}

    if tolerance is None:
        return None
    if isinstance(tolerance, Mapping):
        return {role: entry(t) for role, t in tolerance.items()}
    items = tolerance if isinstance(tolerance, (tuple, list)) else (tolerance,)
    return [entry(t) for t in items]


def bounds_by_output(tolerance, roles_of_results):
    """What governs each output index of a report: ``tolerance`` itself, or for a
    declaration keyed by role one entry per output index -- the bound of the role that
    output has in the plans measured (``roles_of_results``: the output roles of each), None
    where plans disagree on it."""
    if not isinstance(tolerance, Mapping):
        return tolerance
    at = {}
    for roles in roles_of_results:
        for i, role in enumerate(roles):
            at.setdefault(i, set()).add(role)
    return [tolerance.get(next(iter(roles))) if len(roles) == 1 else None
            for _, roles in sorted(at.items())]


def worst_measured(measurements) -> list | None:
    """Per output index, ``{"rel", "abs"}``: the largest of each over ``measurements``.

    As long as the longest measurement: a split partial has more outputs than a solo plan.
    """
    out = None
    for measured in measurements:
        if measured is None:
            continue
        out = out or []
        out += [{"rel": 0.0, "abs": 0.0} for _ in range(len(measured) - len(out))]
        for worst, m in zip(out, measured):
            worst["rel"], worst["abs"] = max(worst["rel"], m.rel), max(worst["abs"], m.abs)
    return out


def format_measured(worst, tolerance=None) -> str:
    """``rel 6.1e-04 of rtol 0.005`` per output, for the stage report: the measured error
    beside the bound that governs it. Both errors when the tolerance is not known, and when
    it has both terms: neither error alone is then a fraction of its term."""
    if not worst:
        return ""
    if isinstance(tolerance, Mapping):  # by role: which output has which is not known here
        tolerance = None
    bounds = tolerance_dicts(tolerance) or [None] * len(worst)
    if len(bounds) == 1:
        bounds = bounds * len(worst)
    parts = []
    for i, (w, bound) in enumerate(zip(worst, bounds)):
        both = f"rel {w['rel']:.1e} abs {w['abs']:.1e}"
        if bound is None:
            text = both
        elif bound["rtol"] and bound["atol"]:
            text = f"{both} of rtol {bound['rtol']:g} * rowmax + atol {bound['atol']:g}"
        elif bound["rtol"]:
            text = f"rel {w['rel']:.1e} of rtol {bound['rtol']:g}"
        elif bound["atol"]:
            text = f"abs {w['abs']:.1e} of atol {bound['atol']:g}"
        else:
            text = f"{both} of an exact bound"
        parts.append((f"out{i} " if len(worst) > 1 else "") + text)
    return "; ".join(parts)


@dataclass(eq=False)
class Result:
    plan: Any  # pipeline.plans.Plan
    run: Any  # cleave.runtime.evaluate.CandidateResult
    kernel: Any = None  # pipeline.build.BuiltKernel, when codegen produced source
    # The input cases the reference of its graph was built for (cleave.runtime.inputs): what
    # a kernel had to pass to be ok. ``cases_note`` says why a program whose softmax is exact
    # was held to fewer than every adversarial case ("" when it was held to all of them).
    cases: tuple | None = None
    cases_note: str = ""

    @property
    def ok(self) -> bool:
        return self.run.ok

    @property
    def latency_ms(self) -> float | None:
        return self.run.latency_ms

    @property
    def latency_us(self) -> float | None:
        return None if self.run.latency_ms is None else self.run.latency_ms * 1e3

    @property
    def error(self):
        return self.run.error

    @property
    def measured(self):
        """``validate.MeasuredError`` per output, when the comparison ran to the end."""
        return getattr(self.run, "measured", None)

    @property
    def stage(self) -> str | None:
        return self.run.stage

    @property
    def error_type(self) -> str | None:
        if self.ok:
            return None
        return type(self.run.error).__name__ if self.run.error is not None else str(self.run.stage)

    @property
    def source(self) -> str | None:
        return self.kernel.source if self.kernel is not None else self.run.code

    @property
    def cuda_src(self) -> str | None:
        return self.run.cuda_src

    @property
    def variant(self) -> str:
        return self.plan.variant

    @property
    def role(self) -> str:
        return self.plan.role

    @property
    def label(self) -> str:
        return self.plan.label

    def __repr__(self):
        state = f"{self.latency_us:.2f} us" if self.latency_us is not None else (
            "ok" if self.ok else f"{self.error_type} at {self.stage}")
        return f"Result({self.label}: {state})"


class Failures(list):
    def histogram(self) -> dict[str, int]:
        return dict(Counter(r.error_type for r in self).most_common())

    def by_stage(self) -> dict[str, int]:
        return dict(Counter(r.stage for r in self).most_common())

    def by_validation_kind(self) -> dict[str, int]:
        """The validate-stage failures as mismatch / non_finite / launch_crash / other."""
        return dict(Counter(validation_kind(r.error) for r in self
                            if r.stage == "validate").most_common())

    def __str__(self):
        return ", ".join(f"{n}x {k}" for k, n in self.histogram().items()) or "none"


@dataclass
class VariantResult:
    variant: str
    latency_us: float
    members: list[Result] = field(default_factory=list)  # [main] or [partial, combine]


class _Results(Sequence):
    results: list[Result]

    def __len__(self):
        return len(self.results)

    def __getitem__(self, i):
        return self.results[i]

    @property
    def correct(self) -> list[Result]:
        return [r for r in self.results if r.ok]

    @property
    def failures(self) -> Failures:
        return Failures(r for r in self.results if not r.ok)

    def input_cases(self) -> list[tuple]:
        """``(cases, kernels, note)`` for each distinct set of input cases the candidates
        were held to (``Result.cases``), in first-seen order; empty when none is recorded."""
        groups = {}
        for r in self.results:
            if r.cases is not None:
                groups[(r.cases, r.cases_note)] = groups.get((r.cases, r.cases_note), 0) + 1
        return [(cases, n, note) for (cases, note), n in groups.items()]

    def histogram(self) -> dict[str, int]:
        return self.failures.histogram()

    def matmul_operands(self) -> tuple:
        """``(format, significant bits)`` of the reduced formats the candidates' matmuls
        multiply in (``matmul_operands``), over every lowered graph; empty when every
        matmul multiplies its operands as they are."""
        found, seen = set(), set()
        for r in self.results:
            graph = getattr(getattr(r.plan, "tile", None), "flat_graph", None)
            if graph is not None and id(graph) not in seen:
                seen.add(id(graph))
                found.update(matmul_operands(graph))
        return tuple(sorted(found))

    def _matmul_text(self) -> str:
        """What the validate line and a failed compile say about reduced matmul operands."""
        return "".join(
            f"  matmul operands multiplied as {fmt} ({bits} significant bits): the result "
            "is not the product at the program's dtype"
            for fmt, bits in self.matmul_operands())


class Report(_Results):
    def __init__(self, results: list[Result], plans=None, timings: dict | None = None,
                 tolerance=None):
        self.results, self.plans = list(results), plans
        self.timings: dict[str, float] = dict(timings or {})
        self.tolerance = tolerance  # what the candidates were validated at
        self._best = None

    @property
    def program(self):
        return None if self.plans is None else self.plans.program

    @property
    def timed(self) -> list[Result]:
        return [r for r in self.results if r.ok and r.latency_ms is not None]

    def _resolve(self) -> tuple[dict[str, VariantResult], dict[str, str]]:
        timed: dict[str, dict[str, list[Result]]] = {}
        for r in self.timed:
            timed.setdefault(r.variant, {}).setdefault(r.role, []).append(r)
        out, unpaired = {}, {}
        for variant, roles in timed.items():
            if "main" in roles:
                members = [_fastest(roles["main"])]
            else:
                members, why = _fastest_pair(roles.get("partial", ()), roles.get("combine", ()))
                if why is not None:
                    unpaired[variant] = why
            if members:
                out[variant] = VariantResult(variant, sum(m.latency_us for m in members), members)
        return out, unpaired

    def variants(self) -> dict[str, VariantResult]:
        """Best latency per variant: solo = best main; split-k=N = the fastest partial +
        combine pair that pass the same tensors across the cut (see ``unpaired``).

        A split variant's latency is the sum of its two stage timings. The copy that hands
        a partial's transposed outputs to the combine is in the partial's timing
        (``runtime.reference.with_hand_off``), so the sum covers the stages and the
        hand-off of a chained call; it is not a timing of the chained Kernel itself."""
        return self._resolve()[0]

    def unpaired(self) -> dict[str, str]:
        """Split variants that have timed partial and combine kernels but no pair that fits
        together, with the reason. Such a variant is not in ``variants()``; the stage
        report, ``to_dict()`` and ``best(variant)`` say why instead of dropping it."""
        return self._resolve()[1]

    def best(self, variant: str | None = None):
        """The fastest Kernel overall, or of one ``variant`` (``"solo"``, ``"split-k=8"``)."""
        if variant is not None or self._best is None:
            variants = self.variants()
            if variant is not None:
                variants = {k: v for k, v in variants.items() if k == variant}
            if not variants:
                refused = getattr(self.plans, "refused", None) if not len(self) else None
                why = f"; refused: {refused[0].reason}" if refused else ""
                unpaired = self.unpaired()
                for name in [variant] if variant is not None else unpaired:
                    if name in unpaired:
                        why += f"; {name} unpaired: {unpaired[name]}"
                if any(isinstance(r.error, OutputMismatch) for r in self.failures):
                    # a bound set for the program's dtype is one no such kernel can meet
                    why += self._matmul_text().replace("  matmul", "; matmul", 1)
                raise CleaveError(f"no {variant or 'kernel'} compiled, validated and timed "
                                  f"({len(self)} candidates; failures: {self.failures}{why})")
            win = min(variants.values(), key=lambda v: v.latency_us)
            if variant is not None:
                return _make_kernel(win, self)
            self._best = _make_kernel(win, self)
        return self._best

    def table(self, top: int = 10, by: str | None = None) -> str:
        if by == "variant":
            rows = [(v.variant, f"{v.latency_us:.2f}", " + ".join(m.label for m in v.members))
                    for v in sorted(self.variants().values(), key=lambda v: v.latency_us)][:top]
            return _fmt(rows, ("variant", "us", "kernels"))
        rows = sorted(self.timed, key=lambda r: r.latency_ms)[:top]
        return _fmt([(i, r.variant, r.role, f"{r.latency_us:.2f}", r.label) for i, r in enumerate(rows)],
                    ("#", "variant", "role", "us", "plan"))

    def measured_error(self) -> list | None:
        """Per output, the largest error measured over the candidates that passed."""
        return worst_measured(r.measured for r in self.correct)

    def _bounds(self):
        """The bound beside each entry of ``measured_error()``."""
        from ..runtime.reference import output_roles

        return bounds_by_output(self.tolerance, (
            output_roles(r.plan.tile.flat_graph) for r in self.correct if r.measured is not None))

    def to_dict(self) -> dict:
        variants = self.variants()
        return {
            "candidates": len(self), "correct": len(self.correct),
            "failures": self.histogram(), "failures_by_stage": self.failures.by_stage(),
            "validation": {
                "tolerance": tolerance_dicts(self.tolerance),
                "failures": self.failures.by_validation_kind(),
                "max_error": self.measured_error(),
                # every candidate whose outputs were compared, passing or not
                "measured": [{"plan": r.label, "ok": r.ok, "error": r.error_type,
                              "rel": [m.rel for m in r.measured],
                              "abs": [m.abs for m in r.measured]}
                             for r in self.results if r.measured is not None],
                **self._inputs_entry(),
                **self._matmul_entry(),
            },
            "timings_s": self.timings,
            "variants": {k: {"latency_us": v.latency_us, "kernels": [m.label for m in v.members]}
                         for k, v in variants.items()},
            # split variants whose timed partial and combine kernels do not fit together
            "unpaired": self.unpaired(),
            "best": None if not variants else min(variants.values(), key=lambda v: v.latency_us).variant,
        }

    def _inputs_entry(self) -> dict:
        """``{"inputs": [...]}`` for ``to_dict()["validation"]``: the input cases the
        candidates were validated on, when the build recorded them."""
        held = self.input_cases()
        if not held:
            return {}
        return {"inputs": [{"cases": list(cases), "kernels": n, "note": note}
                           for cases, n, note in held]}

    def _matmul_entry(self) -> dict:
        """``{"matmul_operands": [...]}`` for ``to_dict()["validation"]``: the formats the
        matmuls multiply in that keep fewer bits than the program's dtype, when there are
        any."""
        reduced = self.matmul_operands()
        if not reduced:
            return {}
        return {"matmul_operands": [{"format": fmt, "significant_bits": bits}
                                    for fmt, bits in reduced]}

    def _inputs_text(self) -> str:
        """What the validate line says about the input cases: nothing while every candidate
        was held to the normal inputs alone and none of them claims more."""
        held = self.input_cases()
        if all(cases == ("normal",) and not note for cases, _, note in held):
            return ""
        parts = []
        for cases, n, note in held:
            text = ", ".join(cases) + (f" ({n} kernels)" if len(held) > 1 else "")
            if note:
                text += f" [exact softmax not held to every adversarial case: {note}]"
            parts.append(text)
        return "  inputs: " + "; ".join(parts)

    def stage_lines(self) -> list[tuple[str, str, float | None]]:
        """``(stage, text, seconds)`` rows: superoptimize / schedule / build / validate / benchmark."""
        rows = []
        graphs = getattr(self.plans, "graphs", None)
        if graphs is not None:
            src = "cache hit" if graphs.from_cache else "searched" if graphs.config else "identity"
            kinds = Counter(g.kind for g in graphs)
            rows.append(("superoptimize", f"{len(graphs)} graph(s) [{_hist(kinds)}] "
                         f"max_ops={graphs.max_ops} {src}", self.timings.get("superoptimize")))
        if self.plans is not None:
            text = self.plans.summary().replace("\n  ", "  ")
            rows.append(("schedule", text, self.timings.get("schedule")))
        built = [r for r in self.results if r.ok or r.stage not in ("validate", "bench")]
        build_fail = Failures(r for r in built if not r.ok)
        text = f"{len(self)} kernels: {len(self) - len(build_fail)} compiled, {len(build_fail)} failed"
        rows.append(("build", text + (f"  [{build_fail}]" if build_fail else ""), self.timings.get("build")))
        val_fail = Failures(r for r in self.results if not r.ok and r.stage == "validate")
        n_val = len(self) - len(build_fail)
        worst = format_measured(self.measured_error(), self._bounds())
        rows.append(("validate", f"{n_val - len(val_fail)} ok / {len(val_fail)} failed"
                     + (f"  [{val_fail}]" if val_fail else "")
                     + (f"  max error {worst}" if worst else "") + self._inputs_text()
                     + self._matmul_text(), None))
        bench_fail = Failures(r for r in self.results if not r.ok and r.stage == "bench")
        variants, unpaired = self._resolve()
        if variants:
            win = min(variants.values(), key=lambda v: v.latency_us)
            others = ", ".join(f"{v.variant} {v.latency_us:.2f}" for v in
                               sorted(variants.values(), key=lambda v: v.latency_us) if v is not win)
            text = f"best {win.latency_us:.2f} us  {' + '.join(m.label for m in win.members)}"
            if others:
                text += f"  (others: {others} us)"
        else:
            text = "no complete variant"
        for name, why in unpaired.items():
            text += f"  [{name} unpaired: {why}]"
        if bench_fail:
            text += f"  [{bench_fail}]"
        rows.append(("benchmark", text, self.timings.get("benchmark")))
        return rows

    def summary(self) -> str:
        rows = self.stage_lines()
        out = []
        for i, (stage, text, secs) in enumerate(rows):
            lead = "┌" if i == 0 else "└" if i == len(rows) - 1 else "├"
            t = "" if secs is None else f"  {secs:.2f} s"
            out.append(f"{lead} {stage:<14}{text}{t}")
        return "\n".join(out)

    def __str__(self):
        return (f"Report: {len(self)} candidates, {len(self.correct)} correct, "
                f"failures: {self.failures}")


def _fastest(results) -> Result:
    return min(results, key=lambda r: r.latency_ms)


def _fastest_pair(partials, combines) -> tuple[list[Result] | None, str | None]:
    """``([partial, combine], None)``: the pair with the lowest total latency among those
    whose boundary signatures are equal. ``(None, why)`` when there are partials and
    combines but none fit together, ``(None, None)`` when one of the two is missing.

    The combine of a factor is searched once and shared by every split graph of the program,
    so the fastest partial and the fastest combine need not come from the same cut. Each is
    validated alone, on its own random inputs, so no validation sees whether a partial
    writes what a combine reads; the boundary records are the check there is. They compare
    index, role, shape and dtype of every tensor crossing the cut -- not what the tensors
    hold: two cuts that pass equally shaped tensors with different contents would be
    paired. (Every cut of the programs searched today has one boundary and one combine.)

    This is the pairing ``kernel.stages_for`` accepts. Plans without records (hand-built
    graphs) all compare equal, which is the pairing there was before the records. A side
    with records never pairs with one without, and a plan whose records cannot be read
    (set on some of its tensors only, or one index twice) pairs with nothing: either is a
    graph built without saying which tensor is which, and the reason is returned so that
    the variant is reported as unpaired instead of vanishing.
    """
    from .kernel import _describe, boundary_of

    read: dict[int, tuple] = {}  # plans of one lowered graph share the answer

    def signature(result):
        graph = getattr(getattr(result.plan, "tile", None), "flat_graph", None)
        if id(graph) not in read:
            try:
                read[id(graph)] = (boundary_of(result.plan), None)
            except ValueError as error:
                read[id(graph)] = (None, f"{result.label}: {error}")
        return read[id(graph)]

    sides: dict[object, dict[str, list[Result]]] = {}
    unreadable = []
    for role, results in (("partial", partials), ("combine", combines)):
        for r in results:
            found, error = signature(r)
            if error is None:
                sides.setdefault(found, {}).setdefault(role, []).append(r)
            elif error not in unreadable:
                unreadable.append(error)
    pairs = [[_fastest(side["partial"]), _fastest(side["combine"])]
             for side in sides.values() if len(side) == 2]
    if pairs:
        return min(pairs, key=lambda pair: sum(m.latency_ms for m in pair)), None
    if not partials or not combines:
        return None, None
    if unreadable:
        return None, "; ".join(unreadable)

    def tensors(role):
        return " or ".join("tensors without boundary records" if found is None
                           else _describe(found) for found, side in sides.items() if role in side)

    return None, (f"no partial writes what a combine reads: the partials write "
                  f"{tensors('partial')}, the combines read {tensors('combine')}")


def _hist(counter) -> str:
    return ", ".join(f"{n} {k}" for k, n in counter.most_common())


def _fmt(rows, header) -> str:
    rows = [tuple(str(c) for c in r) for r in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(header)]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()  # noqa: E731
    return "\n".join([line(tuple(map(str, header))), line(tuple("-" * w for w in widths))]
                     + [line(r) for r in rows])


def _make_kernel(win: VariantResult, report: Report):
    from .kernel import Kernel, stages_for

    plans = [m.plan for m in win.members]
    program = plans[0].program
    stages, problem = stages_for([(m.plan, m.run, m.kernel) for m in win.members], program)
    inputs = list(getattr(program, "inputs", []) or [])
    outputs = getattr(program, "outputs", None)
    out_shapes = None
    if outputs:
        out_shapes = [tuple(int(s) for s in getattr(o, "shape", None) or o.tensor.shape) for o in outputs]
    if problem is None and len(win.members) > 1 and out_shapes is None:
        problem = "split kernel without Program output shapes"
    meta = {"fingerprint": program.fingerprint() if hasattr(program, "fingerprint") else None,
            "graph_index": plans[0].graph_index, "dtype": str(getattr(program, "dtype", ""))}
    reduced = sorted({found for plan in plans
                      for found in matmul_operands(getattr(plan.tile, "flat_graph", None))})
    if reduced:  # what this kernel's matmuls multiply in, where that is not its dtype
        meta["matmul_operands"] = [fmt for fmt, _ in reduced]
    from . import cache

    meta["build_id"] = cache.build_id()
    label = " + ".join(p.label for p in plans)
    return Kernel(stages, name=getattr(program, "name", "kernel"), input_names=[s.name for s in inputs],
                  input_shapes=[tuple(int(x) for x in s.shape) for s in inputs], output_shapes=out_shapes,
                  variant=win.variant, label=label, latency_us=win.latency_us, report=report,
                  plans=plans, unrunnable=problem, meta=meta)
