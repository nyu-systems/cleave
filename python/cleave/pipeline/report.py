"""Per-candidate results, failure histograms, the split-K pairing rule and ``best()``."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core.errors import CleaveError


@dataclass(eq=False)
class Result:
    plan: Any  # pipeline.plans.Plan
    run: Any  # cleave.runtime.evaluate.CandidateResult
    kernel: Any = None  # pipeline.build.BuiltKernel, when codegen produced source

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

    def histogram(self) -> dict[str, int]:
        return self.failures.histogram()


class Report(_Results):
    def __init__(self, results: list[Result], plans=None, timings: dict | None = None):
        self.results, self.plans = list(results), plans
        self.timings: dict[str, float] = dict(timings or {})
        self._best = None

    @property
    def program(self):
        return None if self.plans is None else self.plans.program

    @property
    def timed(self) -> list[Result]:
        return [r for r in self.results if r.ok and r.latency_ms is not None]

    def variants(self) -> dict[str, VariantResult]:
        """Best latency per variant: solo = best main; split-k=N = best partial + best combine."""
        best: dict[tuple[str, str], Result] = {}
        for r in self.timed:
            key = (r.variant, r.role)
            if key not in best or r.latency_ms < best[key].latency_ms:
                best[key] = r
        out = {}
        for variant in dict.fromkeys(v for v, _ in best):
            roles = ("main",) if (variant, "main") in best else ("partial", "combine")
            if all((variant, role) in best for role in roles):
                members = [best[(variant, role)] for role in roles]
                out[variant] = VariantResult(variant, sum(m.latency_us for m in members), members)
        return out

    def best(self, variant: str | None = None):
        """The fastest Kernel overall, or of one ``variant`` (``"solo"``, ``"split-k=8"``)."""
        if variant is not None or self._best is None:
            variants = self.variants()
            if variant is not None:
                variants = {k: v for k, v in variants.items() if k == variant}
            if not variants:
                refused = getattr(self.plans, "refused", None) if not len(self) else None
                why = f"; refused: {refused[0].reason}" if refused else ""
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

    def to_dict(self) -> dict:
        variants = self.variants()
        return {
            "candidates": len(self), "correct": len(self.correct),
            "failures": self.histogram(), "failures_by_stage": self.failures.by_stage(),
            "timings_s": self.timings,
            "variants": {k: {"latency_us": v.latency_us, "kernels": [m.label for m in v.members]}
                         for k, v in variants.items()},
            "best": None if not variants else min(variants.values(), key=lambda v: v.latency_us).variant,
        }

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
        rows.append(("validate", f"{n_val - len(val_fail)} ok / {len(val_fail)} failed"
                     + (f"  [{val_fail}]" if val_fail else ""), None))
        bench_fail = Failures(r for r in self.results if not r.ok and r.stage == "bench")
        variants = self.variants()
        if variants:
            win = min(variants.values(), key=lambda v: v.latency_us)
            others = ", ".join(f"{v.variant} {v.latency_us:.2f}" for v in
                               sorted(variants.values(), key=lambda v: v.latency_us) if v is not win)
            text = f"best {win.latency_us:.2f} us  {' + '.join(m.label for m in win.members)}"
            if others:
                text += f"  (others: {others} us)"
        else:
            text = "no complete variant"
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
    from . import cache

    meta["build_id"] = cache.build_id()
    label = " + ".join(p.label for p in plans)
    return Kernel(stages, name=getattr(program, "name", "kernel"), input_names=[s.name for s in inputs],
                  input_shapes=[tuple(int(x) for x in s.shape) for s in inputs], output_shapes=out_shapes,
                  variant=win.variant, label=label, latency_us=win.latency_us, report=report,
                  plans=plans, unrunnable=problem, meta=meta)
