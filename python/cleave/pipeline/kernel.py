"""The compiled result: a callable ``Kernel`` (one stage, or split-K partial -> combine)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FORMAT = 1


@dataclass
class Param:
    """One kernel input: placeholder ``name`` fed from Program input / previous-stage output ``index``."""

    name: str
    source: str  # "input" | "stage"
    index: int
    shape: tuple[int, ...]
    # For a "stage" param, how the previous stage's kernel stores that output: the
    # ``(axes, pad)`` its tiling recorded (``Graph.stored_layouts``). ``shape`` is the
    # logical shape this kernel reads, so the tensor is brought back with ``to_logical``.
    # None for a pair of hand-built graphs (no boundary records): the tensor is then
    # reshaped to ``shape``, which is only right when the two differ by unit axes.
    layout: tuple | None = None


@dataclass
class Stage:
    role: str  # "main" | "partial" | "combine"
    source: str = field(repr=False)
    out_idx: list[int]
    params: list[Param]  # in the lowered graph's placeholder order
    label: str = ""
    latency_us: float | None = None
    built: Any = field(default=None, repr=False, compare=False)  # pipeline.build.BuiltKernel


class Kernel:
    def __init__(self, stages: list[Stage], *, name: str, input_names: list[str],
                 input_shapes: list[tuple], output_shapes: list[tuple] | None = None,
                 variant: str = "solo", label: str = "", latency_us: float | None = None,
                 report=None, plans=(), unrunnable: str | None = None, meta: dict | None = None,
                 output_order: list[int] | None = None):
        self.stages, self.name, self.variant, self.label = stages, name, variant, label
        self.input_names, self.input_shapes = list(input_names), [tuple(s) for s in input_shapes]
        self.output_shapes = None if output_shapes is None else [tuple(s) for s in output_shapes]
        # Where each declared output is among the outputs the last stage's kernel returns
        # (``declared_output_order``). None: a kernel of hand-built graphs, returned as it is.
        self.output_order = None if output_order is None else [int(i) for i in output_order]
        self.latency_us, self.report, self.plans = latency_us, report, list(plans)
        self.unrunnable, self.meta = unrunnable, dict(meta or {})
        self._compiled = None

    @property
    def source(self) -> str:
        if len(self.stages) == 1:
            return self.stages[0].source
        return "\n\n".join(f"# ---- stage {i}: {s.role} ----\n{s.source}" for i, s in enumerate(self.stages))

    @property
    def plan(self):
        return self.plans[0] if len(self.plans) == 1 else tuple(self.plans)

    @property
    def kernels(self) -> list:
        """The ``BuiltKernel`` of each stage (None for a stage saved without one)."""
        return [s.built for s in self.stages]

    def __repr__(self):
        lat = "?" if self.latency_us is None else f"{self.latency_us:.2f} us"
        return f"Kernel({self.name!r}, {self.variant}, {lat}, {self.label!r})"

    # ------------------------------------------------------------------ call
    def _bind(self, args, named) -> list:
        if args and named:
            raise TypeError("pass inputs either positionally or by name, not both")
        if named:
            missing = [n for n in self.input_names if n not in named]
            extra = sorted(set(named) - set(self.input_names))
            if missing or extra:
                raise TypeError(f"{self.name}: missing inputs {missing}, unexpected {extra}")
            args = [named[n] for n in self.input_names]
        if len(args) != len(self.input_names):
            raise TypeError(f"{self.name}: expected {len(self.input_names)} inputs "
                            f"({', '.join(self.input_names)}), got {len(args)}")
        for n, s, x in zip(self.input_names, self.input_shapes, args):
            if tuple(x.shape) != s:
                raise ValueError(f"{self.name}: input {n!r} has shape {tuple(x.shape)}, expected {s}")
        return list(args)

    def compile(self) -> "Kernel":
        """JIT-compile the TileLang sources (no search); done lazily by ``__call__``."""
        if self.unrunnable:
            raise NotImplementedError(self.unrunnable)
        if self._compiled is None:
            from ..runtime.compile import _compile, _load_main_kernel_from_code

            self._compiled = [_compile(_load_main_kernel_from_code(s.source), s.out_idx)
                              for s in self.stages]
        return self

    def __call__(self, *args, **named):
        from ..runtime.reference import order_inputs_for_kernel, to_logical

        inputs = self._bind(args, named)
        self.compile()
        outs: list = []
        for stage, kernel in zip(self.stages, self._compiled):
            feed = {}
            for p in stage.params:
                x = inputs[p.index] if p.source == "input" else outs[p.index]
                # A partial stores its outputs split axis first; the combine reads them in
                # the logical layout. That is a transposition, which a reshape scrambles.
                feed[p.name] = (_fit(x, p.shape) if p.layout is None
                                else to_logical(x, p.layout, p.shape))
            res = kernel(*order_inputs_for_kernel(feed, kernel))
            outs = list(res) if isinstance(res, (list, tuple)) else [res]
        if self.output_order is not None:
            # A kernel returns its outputs in the order its graph lists them, which is the
            # order a walk from the first output finds them, not the order the program
            # returns them in.
            outs = [outs[i] for i in self.output_order]
        if self.output_shapes is not None:
            outs = [_fit(o, s) for o, s in zip(outs, self.output_shapes)]
        return outs[0] if len(outs) == 1 else tuple(outs)

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> Path:
        p = Path(path).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        stages = []
        for i, s in enumerate(self.stages):
            fname = f"kernel_{i}_{s.role}.py" if len(self.stages) > 1 else "kernel.py"
            (p / fname).write_text(s.source)
            d = {"role": s.role, "out_idx": s.out_idx, "params": [asdict(q) for q in s.params],
                 "label": s.label, "latency_us": s.latency_us, "file": fname}
            if s.built is not None:
                d["kernel"] = s.built.to_dict()
            stages.append(d)
        meta = {"format": FORMAT, "name": self.name, "variant": self.variant, "label": self.label,
                "latency_us": self.latency_us, "input_names": self.input_names,
                "input_shapes": self.input_shapes, "output_shapes": self.output_shapes,
                "output_order": self.output_order,
                "unrunnable": self.unrunnable, "stages": stages, "meta": self.meta}
        (p / "kernel.json").write_text(json.dumps(meta, indent=1, default=str))
        if self.report is not None and hasattr(self.report, "to_dict"):
            (p / "report.json").write_text(json.dumps(self.report.to_dict(), indent=1, default=str))
        return p


def load(path) -> Kernel:
    """Read a saved Kernel back; the TileLang sources are recompiled on first call (no search)."""
    p = Path(path).expanduser()
    m = json.loads((p / "kernel.json").read_text())
    if m.get("format") != FORMAT:
        raise ValueError(f"{p}: unsupported kernel format {m.get('format')!r}")
    if m.get("paged"):
        from .dynamic import PagedKernel

        return PagedKernel._load(p, m)
    from .build import BuiltKernel

    stages = []
    for d in m["stages"]:
        params = [Param(q["name"], q["source"], q["index"], tuple(q["shape"]),
                        _as_layout(q.get("layout"))) for q in d["params"]]
        source = (p / d["file"]).read_text()
        built = None if "kernel" not in d else BuiltKernel.from_dict(d["kernel"], source)
        stages.append(Stage(d["role"], source, list(d["out_idx"]), params,
                            d.get("label", ""), d.get("latency_us"), built))
    if "output_order" not in m and stages and len(stages[-1].out_idx) > 2:
        # Without the record the outputs come back in the kernel's own order, which for
        # three or more outputs need not be the program's.
        raise ValueError(f"{p}: saved without the order of its {len(stages[-1].out_idx)} "
                         "outputs; compile the program again")
    return Kernel(stages, name=m["name"], input_names=m["input_names"],
                  input_shapes=m["input_shapes"], output_shapes=m.get("output_shapes"),
                  variant=m["variant"], label=m["label"], latency_us=m["latency_us"],
                  unrunnable=m.get("unrunnable"), meta=dict(m.get("meta") or {}, loaded_from=str(p)),
                  output_order=m.get("output_order"))


def _fit(x, shape):
    """View ``x`` as ``shape`` when only the layout of unit/batch axes differs."""
    shape = tuple(shape)
    if tuple(x.shape) == shape or math.prod(x.shape) != math.prod(shape):
        return x
    return x.reshape(shape)


def _as_layout(value):
    """A stored layout read back from JSON: ``(axes, pad)`` with ``axes`` a tuple again."""
    return None if value is None else (tuple(int(a) for a in value[0]), int(value[1]))


def boundary_of(plan):
    """What a split plan passes across the cut: ``boundary_signature`` of a partial's
    outputs or of a combine's placeholders. None for any other plan, and for a graph
    that carries no boundary records (a hand-built one)."""
    from ..ir.graph_utils import boundary_signature
    from ..runtime.reference import output_order, placeholder_order

    graph = getattr(getattr(plan, "tile", None), "flat_graph", None)
    order = {"partial": output_order, "combine": placeholder_order}.get(plan.role)
    return None if graph is None or order is None else boundary_signature(order(graph))


def declared_output_order(plan, program=None):
    """``(order, problem)`` for the plan whose kernel returns a Kernel's outputs (a solo
    plan, or the combine of a split pair): ``order[k]`` is the place of the program's
    output ``k`` among the outputs that kernel returns.

    The kernel's outputs come in the order its graph lists them (``to_ir`` lists them in the
    structure's order; a graph built another way need not). Every ``OutputNode`` ``to_ir``
    makes records the position of its output op (``Boundary.index``): that is the declared
    place. A graph
    without any record (a hand-built one) has no declared order, and its outputs are
    returned as the kernel returns them (``order`` is None). Records that are not exactly
    the program's outputs are a problem: the Kernel is unrunnable instead of guessing.
    """
    from ..runtime.reference import output_order

    outs = output_order(plan.tile.flat_graph)
    records = [getattr(o, "boundary", None) for o in outs]
    if all(r is None for r in records):
        return None, None
    declared = getattr(program, "outputs", None)
    count = len(declared) if declared else len(outs)
    if (any(r is None or r.role for r in records)
            or sorted(r.index for r in records) != list(range(count))):
        found = ", ".join("none" if r is None else f"#{r.index}{':' + r.role if r.role else ''}"
                          for r in records)
        return None, (f"{plan.label}: its kernel returns the outputs [{found}], not the "
                      f"program's {count} output(s) once each")
    place = {r.index: i for i, r in enumerate(records)}
    return [place[k] for k in range(count)], None


def _combine_params(plan, phs, partial_plan, partial_layouts):
    """``(params, problem)`` for a combine: each placeholder fed from the partial output
    that is the same boundary tensor, with that output's stored layout."""
    from ..runtime.reference import output_order

    outputs = output_order(partial_plan.tile.flat_graph)  # as the partial kernel returns them
    unfed = [Param(ph.name, "stage", -1, _shape(ph)) for ph in phs]
    try:
        want, have = boundary_of(plan), boundary_of(partial_plan)
    except ValueError as error:  # records on some tensors of one side only, or an index twice
        return unfed, f"{plan.label}: {error}"
    if want is None and have is None:
        return _combine_params_by_position(plan, phs, outputs)
    if want is None or have is None:
        # Records on one side only: a graph built outside ``to_ir`` next to a lowered one.
        # The positional pairing is for graphs that have no records at all; here it would
        # be the guess the records replace, so the pair is refused.
        bare, tagged = (plan, partial_plan) if want is None else (partial_plan, plan)
        return unfed, (f"{bare.label}: carries no boundary records while {tagged.label} "
                       "does; every tensor crossing the cut needs a Boundary on both sides")
    problem = None
    if want != have:
        # Index, role, logical shape and dtype of every tensor, compared exactly: two
        # tensors of one shape are told apart by index and role, never by position.
        problem = (f"{plan.label}: the combine reads {_describe(want)} but the partial kernel "
                   f"{partial_plan.label} writes {_describe(have)}")
    layouts = partial_layouts
    if layouts is None or len(layouts) != len(outputs):
        # Without the layouts the hand-off would be a reshape, which hands a split-first
        # output over transposed -- silently so when the shapes coincide.
        found = "no" if layouts is None else str(len(layouts))
        problem = problem or (
            f"{partial_plan.label}: the partial's result carries {found} stored layouts "
            f"for its {len(outputs)} outputs, so they cannot be handed to the combine")
        layouts = None
    position = {o.boundary.index: i for i, o in enumerate(outputs)}
    params = []
    for ph in phs:
        i = position.get(ph.boundary.index, -1)
        layout = None if layouts is None or i < 0 else layouts[i]
        params.append(Param(ph.name, "stage", i, _shape(ph), layout))
    return params, problem


def _combine_params_by_position(plan, phs, outputs):
    """The pairing for graphs without boundary records: the combine's placeholders, in the
    order its node list has them, against the partial's outputs; guarded by element count."""
    graph = plan.tile.flat_graph
    ids = {id(ph) for ph in phs}
    decl = [n for n in graph.nodes if id(n) in ids]
    pos = {ph.name: i for i, ph in enumerate(decl)}
    prev_outputs = [_shape(o) for o in outputs]
    params, problem = [], None
    for ph in phs:
        i = pos[ph.name]
        if i >= len(prev_outputs) or math.prod(prev_outputs[i]) != math.prod(_shape(ph)):
            problem = problem or (f"{plan.label}: combine inputs do not line up with "
                                  "the partial kernel's outputs")
        params.append(Param(ph.name, "stage", i, _shape(ph)))
    return params, problem


def _describe(signature) -> str:
    return "[" + ", ".join(f"#{i}{':' + role if role else ''} {shape} {dtype}"
                           for i, role, shape, dtype in signature) + "]"


def stages_for(plans_and_results, program) -> tuple[list[Stage], str | None]:
    """Call stages from ``[(plan, result[, BuiltKernel])]`` (one solo, or partial then combine).

    A combine's placeholders are paired with the partial's outputs by boundary record
    (``Boundary``: index and role, with logical shape and dtype compared exactly), and each
    takes the stored layout the partial's result carries, so ``Kernel.__call__`` can undo it.
    A pair with records is a problem (the Kernel is unrunnable) when the two sides differ,
    when only one side has records, or when the partial's result has no stored layouts.
    Only a pair with no records at all (hand-built graphs) is paired by position and handed
    over by reshape, as before the records.
    """
    from ..runtime.reference import graph_out_indices, placeholder_order

    stages, problem = [], None
    prev_plan, prev_layouts = None, None
    for plan, res, *built in plans_and_results:
        built = built[0] if built else None
        graph = plan.tile.flat_graph
        phs = placeholder_order(graph)
        params = []
        if plan.role in ("main", "partial"):
            if plan.inputs is None:
                problem = problem or (f"{plan.label}: could not bind kernel placeholders "
                                      "to the Program inputs")
                bind = {}
            else:
                bind = {name: i for i, name in enumerate(plan.inputs)}
            for ph in phs:
                params.append(Param(ph.name, "input", bind.get(ph.name, -1), _shape(ph)))
        elif prev_plan is None:
            problem = problem or f"{plan.label}: a combine needs the partial stage before it"
        else:
            params, mismatch = _combine_params(plan, phs, prev_plan, prev_layouts)
            problem = problem or mismatch
        prev_plan, prev_layouts = plan, getattr(res, "stored_layouts", None)
        lat = None if res is None or res.latency_ms is None else res.latency_ms * 1e3
        source = built.source if built is not None else res.code if res is not None else ""
        stages.append(Stage(plan.role, source, graph_out_indices(graph), params, plan.label, lat,
                            built))
    return stages, problem


def _shape(node) -> tuple[int, ...]:
    return tuple(int(s) for s in node.get_shape(0))
