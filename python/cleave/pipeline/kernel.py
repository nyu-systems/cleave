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
                 report=None, plans=(), unrunnable: str | None = None, meta: dict | None = None):
        self.stages, self.name, self.variant, self.label = stages, name, variant, label
        self.input_names, self.input_shapes = list(input_names), [tuple(s) for s in input_shapes]
        self.output_shapes = None if output_shapes is None else [tuple(s) for s in output_shapes]
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
        from ..runtime.reference import order_inputs_for_kernel

        inputs = self._bind(args, named)
        self.compile()
        outs: list = []
        for stage, kernel in zip(self.stages, self._compiled):
            feed = {}
            for p in stage.params:
                x = inputs[p.index] if p.source == "input" else outs[p.index]
                feed[p.name] = _fit(x, p.shape)
            res = kernel(*order_inputs_for_kernel(feed, kernel))
            outs = list(res) if isinstance(res, (list, tuple)) else [res]
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
        params = [Param(q["name"], q["source"], q["index"], tuple(q["shape"])) for q in d["params"]]
        source = (p / d["file"]).read_text()
        built = None if "kernel" not in d else BuiltKernel.from_dict(d["kernel"], source)
        stages.append(Stage(d["role"], source, list(d["out_idx"]), params,
                            d.get("label", ""), d.get("latency_us"), built))
    return Kernel(stages, name=m["name"], input_names=m["input_names"],
                  input_shapes=m["input_shapes"], output_shapes=m.get("output_shapes"),
                  variant=m["variant"], label=m["label"], latency_us=m["latency_us"],
                  unrunnable=m.get("unrunnable"), meta=dict(m.get("meta") or {}, loaded_from=str(p)))


def _fit(x, shape):
    """View ``x`` as ``shape`` when only the layout of unit/batch axes differs."""
    shape = tuple(shape)
    if tuple(x.shape) == shape or math.prod(x.shape) != math.prod(shape):
        return x
    return x.reshape(shape)


def stages_for(plans_and_results, program) -> tuple[list[Stage], str | None]:
    """Call stages from ``[(plan, result[, BuiltKernel])]`` (one solo, or partial then combine)."""
    from ..runtime.reference import graph_out_indices, output_order, placeholder_order

    stages, problem = [], None
    prev_outputs: list[tuple] = []
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
        else:  # combine: its placeholders, in declaration order, are the partial's outputs
            ids = {id(ph) for ph in phs}
            decl = [n for n in graph.nodes if id(n) in ids]
            pos = {ph.name: i for i, ph in enumerate(decl)}
            for ph in phs:
                i = pos[ph.name]
                if i >= len(prev_outputs) or math.prod(prev_outputs[i]) != math.prod(_shape(ph)):
                    problem = problem or (f"{plan.label}: combine inputs do not line up with "
                                          "the partial kernel's outputs")
                params.append(Param(ph.name, "stage", i, _shape(ph)))
        prev_outputs = [_shape(o) for o in output_order(graph)]
        lat = None if res is None or res.latency_ms is None else res.latency_ms * 1e3
        source = built.source if built is not None else res.code if res is not None else ""
        stages.append(Stage(plan.role, source, graph_out_indices(graph), params, plan.label, lat,
                            built))
    return stages, problem


def _shape(node) -> tuple[int, ...]:
    return tuple(int(s) for s in node.get_shape(0))
