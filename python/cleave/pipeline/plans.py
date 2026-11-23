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
    return bundle


def bind_inputs(ir_graph, graph: SearchedGraph) -> tuple[str | None, ...] | None:
    """Placeholder name for each Program input, by input identity; None for an unread input.

    The k-th ``kn_input_op`` of the structure is Program input k, and its tensor's guid names
    the lowered placeholder (``input_<guid>``). Shapes cannot decide: same-shaped inputs would
    bind in placeholder (first-use) order. None (unbindable) if a placeholder is not a
    structure input or the input counts differ.
    """
    from ..ir.graph import PlaceHolderNode
    from ..lower.convert import input_name

    program = graph.program
    if program is None or not hasattr(program, "inputs"):
        return None
    guids = [op["output_tensors"][0]["guid"] for op in graph.structure
             if op["op_type"] == "kn_input_op"]
    if len(guids) != len(program.inputs):
        return None
    names = {n.name: n for n in ir_graph.nodes if isinstance(n, PlaceHolderNode)}
    bound = tuple(PlaceHolderNode.full_name(input_name(g)) for g in guids)
    if not set(names) <= set(bound):
        return None
    return tuple(b if b in names else None for b in bound)


def describe_tile(t) -> str:
    rsteps = []
    for v in (getattr(t, "rstep_map", None) or {}).values():
        vals = v.values() if isinstance(v, dict) else v if isinstance(v, (list, tuple)) else [v]
        rsteps.extend(int(x) for x in vals)
    text = f"tile={list(t.output_tile)}"
    if rsteps:
        text += f" rstep={rsteps}"
    accumulation = getattr(t, "accumulation", None)
    if accumulation:  # per output, in graph order: which one the CTA loop accumulates
        outputs = [n for n in t.flat_graph.nodes if n.is_output()]
        contexts = [accumulation.get(o.inputs[0].src_node, "?") for o in outputs]
        text += f" acc=[{','.join(contexts)}]"
    if getattr(t, "n_threads", None):
        text += f" threads={t.n_threads}"
    return text


MAX_ROW_LIMIT = "a max over a row longer than one block needs online softmax, which is not implemented"


def _search(ir_graph, topk: int, force: bool, hints: dict | None):
    from ..schedule import search_tiling

    kw = dict(hints or {})
    if kw:
        accepted = inspect.signature(search_tiling).parameters
        unknown = [k for k in kw if k not in accepted]
        if unknown:
            raise ScheduleRejected(f"search_tiling does not accept hints {unknown}")
    try:
        return list(search_tiling(ir_graph, topk=topk, force_fusion=force, **kw))
    except Exception as e:  # noqa: BLE001
        raise ScheduleRejected(f"{type(e).__name__}: {e}") from e


def _schedule_once(graphs: GraphSet, config: ScheduleConfig, force: bool):
    plans, refused = [], []
    program = graphs.program
    topk, hints = config.topk, config.hints

    def add(g, variant, role, ir, factor, inputs=None):
        try:
            tiles = _search(ir, topk, force, hints)
        except ScheduleRejected as e:
            refused.append(Refusal(g.index, variant, role, e))
            return
        if not tiles:
            why = MAX_ROW_LIMIT if "max" in g.ops else "no tiling candidates"
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
            add(g, "solo", "main", ir, None, bind_inputs(ir, g))
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
            add(g, variant, "partial", partial, f, bind_inputs(partial, g))
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
