"""Stages 3-4: build (codegen, then compile + validate + time in workers) and evaluate."""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core.errors import SpecError
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
    def from_plan(cls, plan, source: str, signature=None, stored_layouts=None) -> "BuiltKernel":
        """``stored_layouts`` (``CandidateResult.stored_layouts``) gives the outputs the dims
        the kernel declares; without it they keep the lowered graph's logical dims."""
        params, out_idx, dyn = signature or _signature(plan.tile.flat_graph)
        if stored_layouts is not None:
            params = _stored_output_dims(params, stored_layouts)
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


def _stored_output_dims(params, stored_layouts):
    """``params`` with each output's dims as the kernel stores them (one layout per output,
    in order): the signature of a lowered graph has them in the graph's logical layout."""
    layouts = list(stored_layouts)
    if len(layouts) != sum(p.role == "output" for p in params):
        raise ValueError(f"{len(layouts)} stored layouts for the outputs of "
                         f"{[p.name for p in params if p.role == 'output']}")
    out = []
    for p in params:
        if p.role == "output":
            axes, pad = layouts.pop(0)
            p = KernelParam(p.name, p.dtype, tuple(p.dims[a] for a in axes) + (1,) * pad, p.role)
        out.append(p)
    return tuple(out)


def built_kernels(plans: Sequence[Plan], runs) -> list[BuiltKernel | None]:
    """A ``BuiltKernel`` per run that produced source (signatures shared per lowered graph;
    the output dims are each candidate's own, since tilings store outputs differently)."""
    sigs: dict[int, tuple] = {}
    out = []
    for plan, run in zip(plans, runs):
        if not getattr(run, "code", None):
            out.append(None)
            continue
        graph = plan.tile.flat_graph
        if id(graph) not in sigs:
            sigs[id(graph)] = _signature(graph)
        out.append(BuiltKernel.from_plan(plan, run.code, sigs[id(graph)],
                                         getattr(run, "stored_layouts", None)))
    return out


class BuildSet(_Results):
    def __init__(self, results: list[Result], plans: PlanSet | None = None, seconds: float = 0.0,
                 tolerance=None):
        self.results, self.plans, self.seconds = list(results), plans, seconds
        self.tolerance = tolerance  # what the kernels were validated at

    @property
    def built(self) -> list[Result]:
        return self.correct

    @property
    def kernels(self) -> list[BuiltKernel]:
        return [r.kernel for r in self.results if r.kernel is not None]

    def __str__(self):
        return f"BuildSet: {len(self)} candidates, {len(self.correct)} ok, failures: {self.failures}"


def _program(plans):
    program = getattr(plans, "program", None)
    if program is None and len(plans):
        program = plans[0].program
    return program


def _program_dtype(plans) -> str:
    return str(getattr(_program(plans), "dtype", None) or "float16")


def _reference_fn(distributions, program, publish):
    """``publish(cache_dir, graph, dtype)`` over the input cases the kernels are held to.

    ``distributions`` is what the caller asked for. A program whose softmax is exact
    (``cl.softmax`` recorded the stabilize request) is right at any logit scale, so its
    kernels are also held to every adversarial softmax case its inputs can be shaped into
    (``cleave.runtime.inputs.adversarial_cases``), whether asked or not; they are timed on
    the first case, as always. An ``exp / sum`` written out by hand makes no such
    claim (its reference overflows on those inputs) and is validated on what was asked.

    The returned function remembers what each graph was held to: ``held`` maps
    ``id(flat graph)`` to ``(cases, note)``, ``note`` saying why a graph of an exact program
    got fewer than every adversarial case (``""`` when it got them all, and for a program
    that makes no claim).
    """
    exact = bool(getattr(program, "stabilize_requests", ()))
    held = {}

    def reference(reference_cache_dir, graph, dtype):
        cases, note = distributions, ""
        if exact:
            from ..runtime.inputs import adversarial_cases, why_not_adversarial

            cases += tuple(d for d in adversarial_cases(graph) if d not in distributions)
            note = why_not_adversarial(graph)
        held[id(graph)] = (cases, note)
        return publish(reference_cache_dir, graph, dtype, distributions=cases)

    reference.held = held
    return reference


def _check_tolerance_fits(plan_list, tolerance) -> None:
    """Refuse a ``tolerance`` that leaves an output of some plan without a bound, or that
    does not fit every plan, before any runs.

    One ``Tolerance`` covers every output the program or its cut defines. An output that a
    rewrite added to the cut carries a boundary role (the float32 shift a split partial of
    an exact softmax returns), and is only covered by a declaration keyed by role: a mapping
    from role to ``Tolerance``, which fits every plan. A sequence is one per output, by
    position, and the plans of one program need not have the same outputs: a split partial
    has more than its combine and than the solo plans. All candidates share the
    declaration, so a sequence is only valid when every plan has exactly that many outputs.
    """
    from collections.abc import Mapping

    from ..runtime import reference
    from ..runtime.validate import Tolerance, tolerances_for, tolerances_of

    if not plan_list:
        return  # nothing to validate
    if isinstance(tolerance, (Tolerance, Mapping)):
        checked = set()
        for plan in plan_list:
            graph = plan.tile.flat_graph
            if id(graph) in checked:
                continue
            checked.add(id(graph))
            try:
                tolerances_of(tolerance, reference.output_roles(graph))
            except SpecError as error:
                raise SpecError(f"build(): plan {plan.label!r} ({plan.role}): {error}") from None
        return
    try:
        declared = len(tuple(tolerance))
    except TypeError:
        declared = 1
    tolerances_for(tolerance, declared)  # a SpecError unless these are Tolerances
    counted = {}  # id(flat graph) -> number of outputs
    for plan in plan_list:
        graph = plan.tile.flat_graph
        if id(graph) not in counted:
            counted[id(graph)] = len(reference.output_order(graph))
        if counted[id(graph)] != declared:
            raise SpecError(
                f"build(): {declared} tolerances were declared, but plan {plan.label!r} "
                f"({plan.role}) has {counted[id(graph)]} output(s). A per-output tolerance "
                "must fit every plan, and a split partial has more outputs than its combine "
                "and the solo plans; declare one Tolerance for all outputs, or one per "
                "boundary role")


def _plans_by_graph(plan_list) -> dict:
    """id(flat graph) -> the plan it was lowered for (plans of one graph share it)."""
    return {id(p.tile.flat_graph): p for p in plan_list}


def derive_for(plan, graph):
    """The ``derive`` of ``make_inputs`` for ``plan``'s ``graph``: every input its Program
    declares (``equals=`` / ``sample=``) recomputed from the drawn ones, else None.

    A combine stage reads the partial's outputs, no Program input, so it draws them all.
    """
    from ..runtime.reference import declared_inputs_hook

    if plan is None or plan.role == "combine":
        return None
    return declared_inputs_hook(plan.program, plan.inputs, graph)


def _publish_reference(publish, distributions, plans_by_graph, cache_dir, graph, dtype):
    """``publish_reference`` with the declared inputs of the plan the graph belongs to."""
    derive = derive_for(plans_by_graph.get(id(graph)), graph)
    return publish(cache_dir, graph, dtype, distributions=distributions, derive=derive)


def build(plans: PlanSet | Sequence[Plan], workers: int | None = None, *,
          config: EvalConfig | None = None, compiler_pass_options=None) -> BuildSet:
    """Compile, validate and time every plan in its own worker (a crash fails one candidate).

    Parallelism: ``workers``, else the active ``Session.max_parallel``, else ``config``.
    ``config.tolerance`` is required as soon as there is a plan to validate: one
    ``Tolerance`` for every output, one per boundary role, or one per output when every
    plan has that many (``_check_tolerance_fits``).
    """
    from ..core.session import current_session
    from ..runtime.inputs import check_distributions
    from ..runtime.validate import BenchOptions, publish_reference
    from ..runtime.evaluate import run_candidates

    config = config or EvalConfig()
    session = current_session()
    timeout = config.timeout_s
    if timeout is None:
        timeout = session.settings.compile_timeout_secs or 0  # None in Settings: disabled
    plan_list = list(plans)
    if plan_list and config.tolerance is None:
        raise SpecError("build(): no validation tolerance declared; there is no default bound. "
                        "Pass config=EvalConfig(tolerance=cl.Tolerance(rtol=...))")
    distributions = check_distributions(config.distributions)
    plans_by_graph = _plans_by_graph(plan_list)

    def publish(reference_cache_dir, graph, dtype, distributions):
        return _publish_reference(publish_reference, distributions, plans_by_graph,
                                  reference_cache_dir, graph, dtype)

    reference_fn = _reference_fn(distributions, _program(plans), publish)
    _check_tolerance_fits(plan_list, config.tolerance)
    t0 = time.perf_counter()
    runs = run_candidates([p.tile for p in plan_list], _program_dtype(plans),
                          max_parallel=workers or session.max_parallel or config.max_parallel,
                          tolerance=config.tolerance, timeout_s=timeout,
                          compiler_pass_options=compiler_pass_options,
                          benchmark=BenchOptions(config.warmup, config.iters),
                          reference_fn=reference_fn)
    kernels = built_kernels(plan_list, runs)
    held = [reference_fn.held.get(id(p.tile.flat_graph), (None, "")) for p in plan_list]
    results = [Result(p, r, k, cases, note)
               for p, r, k, (cases, note) in zip(plan_list, runs, kernels, held)]
    return BuildSet(results, plans if isinstance(plans, PlanSet) else None,
                    time.perf_counter() - t0, config.tolerance)


def evaluate(built: BuildSet | PlanSet | Sequence[Plan], config: EvalConfig | None = None) -> Report:
    """The ``Report`` of a ``BuildSet`` (``config`` is only used to build plans first)."""
    if not isinstance(built, BuildSet):
        built = build(built, config=config)
    return Report(built.results, built.plans, {"build": built.seconds}, built.tolerance)
