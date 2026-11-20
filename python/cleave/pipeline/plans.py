"""Stage 2: lower each searched graph and search tilings -> uniform Plans (solo / split-K)."""

from __future__ import annotations

import inspect
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..core.errors import LoweringRejected, ScheduleRejected
from .config import ScheduleConfig
from .superopt import GraphSet, SearchedGraph


@dataclass(eq=False)
class Plan:
    graph_index: int
    variant: str  # "solo" | "split-k=N"
    role: str  # "main" | "partial" | "combine"
    tile: Any  # cleave.schedule.policy.hint.TileDict
    label: str
    rank: int = 0
    split_factor: int | None = None
    graph: SearchedGraph | None = field(default=None, repr=False)
    inputs: tuple[str, ...] | None = None  # placeholder name per Program input (main/partial)

    @property
    def output_tile(self) -> list[int]:
        return list(self.tile.output_tile)

    @property
    def n_threads(self) -> int | None:
        return getattr(self.tile, "n_threads", None)

    @property
    def program(self):
        return None if self.graph is None else self.graph.program


@dataclass
class Refusal:
    graph_index: int
    variant: str
    role: str
    error: Exception

    @property
    def reason(self) -> str:
        return f"{type(self.error).__name__}: {str(self.error).splitlines()[0][:160] if str(self.error) else ''}"


class PlanSet(Sequence):
    def __init__(self, plans, graphs: GraphSet | None = None, config: ScheduleConfig | None = None,
                 refused=(), *, forced_fusion: bool = False, seconds: float = 0.0):
        self.plans, self.graphs, self.config = list(plans), graphs, config
        self.refused: list[Refusal] = list(refused)
        self.forced_fusion, self.seconds = forced_fusion, seconds

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, i):
        return self.plans[i]

    @property
    def program(self):
        return None if self.graphs is None else self.graphs.program

    def filter(self, fn: Callable[[Plan], bool]) -> "PlanSet":
        return PlanSet([p for p in self.plans if fn(p)], self.graphs, self.config, self.refused,
                       forced_fusion=self.forced_fusion, seconds=self.seconds)

    def variants(self) -> dict[str, dict[str, list[Plan]]]:
        out: dict[str, dict[str, list[Plan]]] = {}
        for p in self.plans:
            out.setdefault(p.variant, {}).setdefault(p.role, []).append(p)
        return out

    def summary(self) -> str:
        parts = []
        for variant, roles in self.variants().items():
            parts.append(f"{variant}: " + ", ".join(f"{r}={len(ps)}" for r, ps in roles.items()))
        line = f"{len(self)} plans ({'; '.join(parts) or 'none'})"
        if self.forced_fusion:
            line += " [fusion forced]"
        if self.refused:
            hist = Counter(r.reason for r in self.refused)
            line += "\n  refused: " + "; ".join(f"{n}x {k}" for k, n in hist.most_common())
        return line

    def __str__(self):
        return self.summary()


def lower(graph: SearchedGraph, split_factor: int | None = None) -> list:
    """``[graph]`` for solo, ``[partial, combine]`` for a split graph at ``split_factor``."""
    from ..lower.process import to_bundles

    dtype = graph.dtype
    try:
        bundles = to_bundles([graph.structure], symbol_map=graph.symbol_map,
                             split_factors=None if split_factor is None else [split_factor],
                             dtype=dtype, out_dtype=dtype)
    except Exception as e:  # noqa: BLE001
        raise LoweringRejected(f"graph {graph.index}: {type(e).__name__}: {e}") from e
    if len(bundles) != 1:
        raise LoweringRejected(f"graph {graph.index}: lowering produced {len(bundles)} bundles")
    bundle = list(bundles[0])
    if len(bundle) != (2 if graph.is_split else 1):
        raise LoweringRejected(f"graph {graph.index}: expected {2 if graph.is_split else 1} "
                               f"stage(s), got {len(bundle)}")
    if getattr(graph.program, "stabilize_requests", ()):
        _annotate_softmax(graph, bundle)
    return bundle


SPLIT_SOFTMAX = "exact softmax across a split is not available yet"


def _annotate_softmax(graph: SearchedGraph, bundle: list) -> None:
    """Mark the softmax exps of a freshly lowered graph whose program asked for them.

    ``cl.softmax`` records a plain ``exp`` plus a request; which Exp of *this* graph the
    request means is derived by the scale analysis. The flat graph stays shift-free (the
    tiling policy must not see a max ahead of the sums): the mark is what the reference
    honours and what the per-schedule stabilization consumes after tiling.
    """
    from ..lower.scale import ScaleError, annotate

    try:
        if graph.is_split:
            raise ScaleError(SPLIT_SOFTMAX)
        annotate(bundle[0], graph.program)
    except ScaleError as e:
        raise LoweringRejected(f"graph {graph.index}: softmax cannot be stabilized: {e}") from e


def bind_inputs(ir_graph, program) -> tuple[str, ...] | None:
    """Placeholder name for each Program input.

    Paired by the position lowering recorded on each graph input (``input_index``: the
    structure declares its inputs in the program's order). ``ir_graph.nodes`` is a
    topological order, so two inputs of one shape may come the other way round there;
    a graph without the record is only bound where the shapes leave no choice.

    A recorded input is bound when the graph declares it with the program's shape give
    or take unit axes: lowering declares an input written in keepdim form next to a
    reduced value (``rowsum(x) / y`` with ``y`` of shape ``(R, 1)``) the way its consumer
    reads it, ``(R,)``. Those are the same elements in the same order, and
    ``Kernel.__call__`` views each input as its placeholder's shape.
    """
    from ..ir.graph import PlaceHolderNode

    if program is None or not hasattr(program, "inputs"):
        return None
    phs = [n for n in ir_graph.nodes if isinstance(n, PlaceHolderNode)]
    want = [tuple(int(s) for s in spec.shape) for spec in program.inputs]
    if len(phs) != len(want):
        return None
    declared = [getattr(n, "input_index", None) for n in phs]
    if None not in declared and sorted(declared) == list(range(len(phs))):
        phs = [n for _, n in sorted(zip(declared, phs), key=lambda pair: pair[0])]
        if all(_without_units(_shape(n)) == _without_units(shape)
               for n, shape in zip(phs, want)):
            return tuple(n.name for n in phs)
    ph_shapes = [_shape(n) for n in phs]
    taken, names = set(), []
    for shape in want:
        hits = [i for i, s in enumerate(ph_shapes) if s == shape and i not in taken]
        if len(hits) != 1:
            return None
        taken.add(hits[0])
        names.append(phs[hits[0]].name)
    return tuple(names)


def _shape(node) -> tuple[int, ...]:
    return tuple(int(s) for s in node.get_shape(0))


def _without_units(shape) -> tuple[int, ...]:
    """``shape`` without its unit axes: equal for two layouts of the same elements in the
    same order."""
    return tuple(s for s in shape if s != 1)


def describe_tile(t) -> str:
    rsteps = []
    for v in (getattr(t, "rstep_map", None) or {}).values():
        vals = v.values() if isinstance(v, dict) else v if isinstance(v, (list, tuple)) else [v]
        rsteps.extend(int(x) for x in vals)
    text = f"tile={list(t.output_tile)}"
    if rsteps:
        text += f" rstep={rsteps}"
    if getattr(t, "n_threads", None):
        text += f" threads={t.n_threads}"
    return text


MAX_ROW_LIMIT = ("a max written by hand needs its whole row in one block, and this row is longer; "
                 "cl.softmax carries a running max across the tiles of a row of any length")


def _search(ir_graph, topk: int, force: bool, hints: dict | None,
            refused: Counter | None = None):
    """The top ``topk`` tilings of one lowered graph.

    A graph with a marked softmax exp keeps only the tilings ``apply_tiling`` builds,
    stabilization included, decided before the top-k cut; the reasons of the others are
    added to ``refused``. For any other graph nothing changes.
    """
    from ..schedule import search_tiling
    from ..schedule.stabilize import stabilize_accept

    kw = dict(hints or {})
    if kw:
        accepted = inspect.signature(search_tiling).parameters
        unknown = [k for k in kw if k not in accepted]
        if unknown:
            raise ScheduleRejected(f"search_tiling does not accept hints {unknown}")
    stable = stabilize_accept(ir_graph)
    if stable is not None:
        user = kw.get("accept")
        kw["accept"] = stable if user is None else (lambda cand: user(cand) and stable(cand))
    try:
        return list(search_tiling(ir_graph, topk=topk, force_fusion=force, **kw))
    except ScheduleRejected:
        raise  # the policy's own refusal of the whole graph, with its reason
    except Exception as e:  # noqa: BLE001
        # Not a refused candidate: the filter drops those itself. Whatever else it or the
        # search raises is a bug, and names itself in the refusal of the whole graph.
        raise ScheduleRejected(f"{type(e).__name__}: {e}") from e
    finally:
        if stable is not None and refused is not None:
            refused.update(stable.reasons)


def _schedule_once(graphs: GraphSet, config: ScheduleConfig, force: bool):
    plans, refused = [], []
    program = graphs.program
    topk, hints = config.topk, config.hints

    def add(g, variant, role, ir, factor, inputs=None):
        reasons = Counter()
        try:
            tiles = _search(ir, topk, force, hints, reasons)
        except ScheduleRejected as e:
            refused.append(Refusal(g.index, variant, role, e))
            return
        if not tiles:
            why = MAX_ROW_LIMIT if "max" in g.ops else "no tiling candidates"
            if reasons:  # every tiling was refused by the softmax filter: say why
                why += f" ({sum(reasons.values())} refused: " + "; ".join(
                    f"{n}x {reason}" for reason, n in reasons.most_common()) + ")"
            refused.append(Refusal(g.index, variant, role, ScheduleRejected(why)))
        for rank, t in enumerate(tiles):
            label = f"g{g.index} {variant} {role} #{rank} {describe_tile(t)}"
            plans.append(Plan(g.index, variant, role, t, label, rank, factor, g, inputs))

    combined = set()
    for g in graphs:
        if g.invalid_split:
            why = f" ({g.split_rejection})" if g.split_rejection else ""
            refused.append(Refusal(g.index, "split", "partial", LoweringRejected(
                f"graph {g.index}: split ops admit no partial/combine cut{why}")))
            continue
        if not g.is_split:
            try:
                (ir,) = lower(g)
            except LoweringRejected as e:
                refused.append(Refusal(g.index, "solo", "main", e))
                continue
            add(g, "solo", "main", ir, None, bind_inputs(ir, program))
            continue
        if not config.split_k:
            refused.append(Refusal(g.index, "split", "partial",
                                   LoweringRejected("split graph but ScheduleConfig.split_k is empty")))
            continue
        for f in config.split_k:
            variant = f"split-k={f}"
            try:
                partial, combine = lower(g, f)
            except LoweringRejected as e:
                refused.append(Refusal(g.index, variant, "partial", e))
                continue
            add(g, variant, "partial", partial, f, bind_inputs(partial, program))
            if f not in combined:  # the combine is searched once per factor
                combined.add(f)
                add(g, variant, "combine", combine, f)
    return plans, refused


def schedule(graphs: GraphSet | Sequence[SearchedGraph], config: ScheduleConfig | None = None) -> PlanSet:
    config = config or ScheduleConfig()
    if not isinstance(graphs, GraphSet):
        graphs = list(graphs)
        graphs = GraphSet(graphs, graphs[0].program if graphs else None)
    t0 = time.perf_counter()
    force = config.fusion == "force"
    plans, refused = _schedule_once(graphs, config, force)
    if not plans and config.fusion == "auto":
        force = True
        plans, retry_refused = _schedule_once(graphs, config, True)
        refused += retry_refused
    return PlanSet(plans, graphs, config, refused, forced_fusion=force,
                   seconds=time.perf_counter() - t0)
