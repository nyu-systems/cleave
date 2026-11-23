"""Stage 2: lower each searched graph and search tilings -> uniform Plans (solo / split-K)."""

from __future__ import annotations

import inspect
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..core.errors import LoweringRejected, ScheduleRejected
from ..ir.update_rule import searched_updates
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
    bundle = _lower_structure(graph, graph.structure, split_factor)
    if getattr(graph.program, "stabilize_requests", ()):
        bundle = _annotate_softmax(graph, bundle)
    elif searched_updates() and not graph.is_split:
        _derive_updates(graph, bundle[0])
    return bundle


def _looped_symbol(graph: SearchedGraph, nodes, red_tokens, full: str):
    """The symbolic dim of ``graph.structure`` that the reduction ``full`` runs over: read
    off the reduction-axis provenance the scheduler pins by. Its tokens name the input
    axes the dim comes from; lowering recorded which input of the structure each
    placeholder is (``input_index``). None when the tokens do not name one symbol."""
    node = next(n for n in nodes if n.name == full)
    tokens = red_tokens[node][0]
    placeholders = {id(e.src_node): e.src_node
                    for n in nodes for e in n.inputs if e.src_node.is_placeholder()}
    inputs = [op for op in graph.structure if op["op_type"] == "kn_input_op"]
    symbols = set()
    for node_id, _, dim in tokens:
        index = getattr(placeholders.get(node_id), "input_index", None)
        if index is None or index >= len(inputs):
            return None
        symbols.add(inputs[index]["output_tensors"][0]["dim"][dim])
    return symbols.pop() if len(symbols) == 1 else None


def _derive_updates(graph: SearchedGraph, ir) -> None:
    """Give every streaming obstacle of a freshly lowered flat graph its proven update
    rule, or the reason it has none.

    The obstacles are the scheduler's own (``schedule.streaming.obstacles``, from the
    reduction-axis provenance the policy pins by). For each one that can be streamed, the
    rules are derived once per loop structure (``pipeline.update.derive``: references from
    this very graph at small shapes, search, z3 proof, ranking) and written on the
    full-extent node as a ``Streaming`` record. The policy lets a loop step under a max
    only where it finds that record; without one the pin and the refusal stay.

    ``ir.streaming_refusals`` (``["{full}: {reason}", ...]``) exists exactly when the graph
    has an obstacle; ``ir.derivations`` holds the derivations that gave a record.
    """
    from ..schedule.streaming import obstacles
    from ..schedule.tiling import reduction_ancestors, scheduled_reductions

    try:
        nodes, red = scheduled_reductions(ir)
    except NotImplementedError:
        return
    found = obstacles(nodes, red, reduction_ancestors(nodes))
    if not found:
        return
    from .update import derive, rebuilder

    ir.streaming_refusals = []
    for ob in found:
        why = ob.refused
        if not why and graph.symbol_map is None:
            why = "this graph has no symbolic dims to re-instantiate it at small shapes"
        if not why:
            derivation = derive(ir, ob, rebuilder(graph), graph.symbol_map,
                                looped=_looped_symbol(graph, nodes, red, ob.full))
            why = derivation.refused
            if not why:
                next(n for n in ir.nodes if n.name == ob.full).streaming = derivation.record()
                ir.derivations = getattr(ir, "derivations", []) + [derivation]
        if why:
            ir.streaming_refusals.append(f"{ob.full}: {why}")


def _lower_structure(graph: SearchedGraph, structure, split_factor: int | None = None) -> list:
    from ..lower.process import to_bundles

    dtype = graph.dtype
    try:
        bundles = to_bundles([structure], symbol_map=graph.symbol_map,
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


def _annotate_softmax(graph: SearchedGraph, bundle: list) -> list:
    """The freshly lowered graph(s) of a program that asked for a softmax, with its exps
    marked.

    ``cl.softmax`` records a plain ``exp`` plus a request; which Exp of *this* graph the
    request means is derived by the scale analysis. The flat graph stays shift-free (the
    tiling policy must not see a max ahead of the sums): the mark is what the reference
    honours and what the per-schedule stabilization consumes after tiling.

    A split pair keeps a softmax on either side of the cut: the partial's, whose shift it
    exports as one more output, and the combine's over the split axis, which weights the
    partials by that shift (``cleave.lower.scale.stabilize_split``).

    The search computes a value once: where the program wrote several exps of one tensor
    (``cl.softmax(x) + cl.exp(x)``, two softmaxes of ``x`` over different axes) the solo
    graph it returns has one, whose factor does not cancel. Such a graph is lowered with
    that exp computed again for a part of its readers (``cleave.lower.unmerge``): the first
    such structure the analysis accepts, in a fixed order, so lowering the graph again
    gives the same one. The structure of the searched graph itself is left as it is.
    """
    from ..lower.scale import ScaleError, annotate, stabilize_split, written_exps
    from ..lower.unmerge import exp_ops, unmerged

    try:
        if graph.is_split:
            return list(stabilize_split(*bundle, graph.program))
        annotate(bundle[0], graph.program)
    except ScaleError as e:
        merged = 0 if graph.is_split else written_exps(graph.program) - exp_ops(graph.structure)
        for structure in unmerged(graph.structure, merged):
            try:
                served = _lower_structure(graph, structure)
                annotate(served[0], graph.program)
            except (LoweringRejected, ScaleError):
                continue
            return served
        why = ("; the search merged exps the program wrote apart, and computing the merged "
               "one again for a part of its readers does not serve the softmax either"
               if merged > 0 else "")
        raise LoweringRejected(
            f"graph {graph.index}: softmax cannot be stabilized: {e}{why}") from e
    return bundle


def bind_inputs(ir_graph, program) -> tuple[str | None, ...] | None:
    """Placeholder name for each Program input; None for an input the graph does not read.

    Paired by the position lowering recorded on each graph input (``input_index``: the
    structure declares its inputs in the program's order, and an input no output reads
    keeps its position but has no placeholder -- a per-output search, or the merge of
    several, leaves such inputs). ``ir_graph.nodes`` is a topological order, so two inputs
    of one shape may come the other way round there; a graph without the record is only
    bound where the shapes leave no choice, and only when it reads every input.

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
    declared = [getattr(n, "input_index", None) for n in phs]
    if (None not in declared and len(set(declared)) == len(declared)
            and set(declared) <= set(range(len(want)))):
        by_index = dict(zip(declared, phs))
        if all(_without_units(_shape(n)) == _without_units(want[i])
               for i, n in by_index.items()):
            return tuple(by_index[i].name if i in by_index else None for i in range(len(want)))
    if len(phs) != len(want):
        return None
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
    accumulation = getattr(t, "accumulation", None)
    if accumulation:  # per output, in graph order: which one the CTA loop accumulates
        outputs = [n for n in t.flat_graph.nodes if n.is_output()]
        contexts = [accumulation.get(o.inputs[0].src_node, "?") for o in outputs]
        text += f" acc=[{','.join(contexts)}]"
    if getattr(t, "n_threads", None):
        text += f" threads={t.n_threads}"
    return text


# (Short enough for a refusal's one-line reason. A softmax that is returned itself -- a
# standalone softmax, attention weights -- needs its row in one block with cl.softmax too:
# the advice is for a softmax that is summed or multiplied over its row.)
MAX_ROW_LIMIT = ("a max written by hand needs its whole row in one block, and this row is longer; "
                 "cl.softmax carries a running max over any row it feeds to a sum or a matmul")


def _search(ir_graph, topk: int, force: bool, hints: dict | None,
            refused: Counter | None = None):
    """The top ``topk`` tilings of one lowered graph.

    A graph with a marked softmax exp keeps only the tilings ``apply_tiling`` builds,
    stabilization included, decided before the top-k cut; the reasons of the others are
    added to ``refused``. The same holds for a graph with a streaming obstacle, served or
    not (``schedule.streaming.streaming_accept``). For any other graph nothing changes.
    """
    from ..schedule import search_tiling
    from ..schedule.stabilize import stabilize_accept
    from ..schedule.streaming import streaming_accept

    kw = dict(hints or {})
    if kw:
        accepted = inspect.signature(search_tiling).parameters
        unknown = [k for k in kw if k not in accepted]
        if unknown:
            raise ScheduleRejected(f"search_tiling does not accept hints {unknown}")
    stable = stabilize_accept(ir_graph)
    if stable is None:
        stable = streaming_accept(ir_graph)
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
            unstreamed = getattr(ir, "streaming_refusals", None)
            if unstreamed:  # why the max is not carried across a loop instead
                why = ("a max needs its whole row in one block, and this row is longer; "
                       "it is not carried across a loop: " + "; ".join(unstreamed))
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
