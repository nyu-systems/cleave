"""Stages 3-4: build (codegen, then compile + validate + time in workers) and evaluate."""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import EvalConfig
from .plans import Plan, PlanSet
from .report import Report, Result, _Results

ENTRY = "main_kernel"


@dataclass(frozen=True)
class KernelParam:
    name: str
    dtype: str
    dims: tuple[int | str, ...]
    role: str  # "input" | "output"


@dataclass(eq=False)
class BuiltKernel:
    """One generated TileLang kernel: its source plus the call signature it declares."""

    source: str = field(repr=False)
    entry: str = ENTRY
    params: tuple[KernelParam, ...] = ()  # inputs in call order, then outputs
    out_idx: tuple[int, ...] = ()
    pass_configs: dict = field(default_factory=dict)
    dyn_dims: tuple[str, ...] = ()
    plan: Any = field(default=None, repr=False)

    @property
    def inputs(self) -> list[KernelParam]:
        return [p for p in self.params if p.role == "input"]

    @property
    def outputs(self) -> list[KernelParam]:
        return [p for p in self.params if p.role == "output"]

    @classmethod
    def from_plan(cls, plan, source: str, signature=None) -> "BuiltKernel":
        params, out_idx, dyn = signature or _signature(plan.tile.flat_graph)
        return cls(source, ENTRY, params, out_idx, pass_configs_of(source), dyn, plan)

    def to_dict(self) -> dict:
        return {"entry": self.entry, "params": [[p.name, p.dtype, list(p.dims), p.role] for p in self.params],
                "out_idx": list(self.out_idx), "pass_configs": dict(self.pass_configs),
                "dyn_dims": list(self.dyn_dims)}

    @classmethod
    def from_dict(cls, d: dict, source: str, plan=None) -> "BuiltKernel":
        params = tuple(KernelParam(n, t, tuple(s), r) for n, t, s, r in d.get("params", ()))
        return cls(source, d.get("entry", ENTRY), params, tuple(d.get("out_idx", ())),
                   dict(d.get("pass_configs") or {}), tuple(d.get("dyn_dims", ())), plan)

    def compile(self):
        """``tl.compile`` the entry under its pass configs (JIT; needs a GPU)."""
        from ..runtime.compile import _compile, _load_main_kernel_from_code

        return _compile(_load_main_kernel_from_code(self.source), list(self.out_idx))


def pass_configs_of(source: str) -> dict:
    """The ``PASS_CONFIGS = {...}`` literal a generated kernel declares, else ``{}``."""
    if "PASS_CONFIGS" not in source:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "PASS_CONFIGS" for t in node.targets):
            try:
                return dict(ast.literal_eval(node.value))
            except ValueError:
                return {}
    return {}


def _dtype(node) -> str:
    try:
        return str(node.get_dtype(0))
    except (IndexError, AttributeError):
        return str(getattr(node, "dtype", ""))


def _signature(graph):
    from ..runtime.reference import graph_out_indices, output_order, placeholder_order

    def extent(s):  # a symbolic extent is kept by name
        try:
            return int(s)
        except Exception:  # noqa: BLE001 - a tir.Var
            return str(s)

    dims = lambda n: tuple(extent(s) for s in n.get_shape(0))  # noqa: E731

    params = tuple([KernelParam(n.name, _dtype(n), dims(n), "input") for n in placeholder_order(graph)]
                   + [KernelParam(n.name, _dtype(n), dims(n), "output") for n in output_order(graph)])
    dyn = tuple(str(d) for d in getattr(graph, "dyn_dims", None) or ())
    return params, tuple(graph_out_indices(graph)), dyn


def built_kernels(plans: Sequence[Plan], runs) -> list[BuiltKernel | None]:
    """A ``BuiltKernel`` per run that produced source (signatures shared per lowered graph)."""
    sigs: dict[int, tuple] = {}
    out = []
    for plan, run in zip(plans, runs):
        if not getattr(run, "code", None):
            out.append(None)
            continue
        graph = plan.tile.flat_graph
        if id(graph) not in sigs:
            sigs[id(graph)] = _signature(graph)
        out.append(BuiltKernel.from_plan(plan, run.code, sigs[id(graph)]))
    return out


class BuildSet(_Results):
    def __init__(self, results: list[Result], plans: PlanSet | None = None, seconds: float = 0.0):
        self.results, self.plans, self.seconds = list(results), plans, seconds

    @property
    def built(self) -> list[Result]:
        return self.correct

    @property
    def kernels(self) -> list[BuiltKernel]:
        return [r.kernel for r in self.results if r.kernel is not None]

    def __str__(self):
        return f"BuildSet: {len(self)} candidates, {len(self.correct)} ok, failures: {self.failures}"


def _program_dtype(plans) -> str:
    program = getattr(plans, "program", None)
    if program is None and len(plans):
        program = plans[0].program
    return str(getattr(program, "dtype", None) or "float16")


def build(plans: PlanSet | Sequence[Plan], workers: int | None = None, *,
          config: EvalConfig | None = None, compiler_pass_options=None) -> BuildSet:
    """Compile, validate and time every plan in its own worker (a crash fails one candidate).

    Parallelism: ``workers``, else the active ``Session.max_parallel``, else ``config``.
    """
    from ..core.session import current_session
    from ..runtime.validate import BenchOptions
    from ..runtime.evaluate import run_candidates

    config = config or EvalConfig()
    session = current_session()
    timeout = config.timeout_s
    if timeout is None:
        timeout = session.settings.compile_timeout_secs or 0  # None in Settings: disabled
    plan_list = list(plans)
    t0 = time.perf_counter()
    runs = run_candidates([p.tile for p in plan_list], _program_dtype(plans),
                          max_parallel=workers or session.max_parallel or config.max_parallel,
                          tolerance=config.tolerance, timeout_s=timeout,
                          compiler_pass_options=compiler_pass_options,
                          benchmark=BenchOptions(config.warmup, config.iters))
    kernels = built_kernels(plan_list, runs)
    results = [Result(p, r, k) for p, r, k in zip(plan_list, runs, kernels)]
    return BuildSet(results, plans if isinstance(plans, PlanSet) else None,
                    time.perf_counter() - t0)


def evaluate(built: BuildSet | PlanSet | Sequence[Plan], config: EvalConfig | None = None) -> Report:
    """The ``Report`` of a ``BuildSet`` (``config`` is only used to build plans first)."""
    if not isinstance(built, BuildSet):
        built = build(built, config=config)
    return Report(built.results, built.plans, {"build": built.seconds})
