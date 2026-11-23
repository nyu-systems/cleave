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
    """``[ir]`` for a solo graph; ``[partial, combine]`` at ``split_factor`` for a split
    graph the search returned, and for a solo graph whose program asked for a split over
    the dim a max of it reduces (``SearchedGraph.split``): that pair is built from the
    loop state of the solo graph (:func:`state_pair`)."""
    if graph.is_split:
        return _lower_structure(graph, graph.structure, split_factor)
    from .update.instantiate import Recipe

    (ir,) = _lower_structure(graph, graph.structure)
    _derive_updates(ir, Recipe.of(graph))
    if split_factor is None:
        return [ir]
    return state_pair(graph, ir, split_factor)


# Why a solo graph has no pair built from its loop state (``state_pair``), after
# ``graph {i}: split over {dim}: ``.
NO_PROVEN_UPDATE = "this graph carries no proven update over that dim"  # P1 (+ ": {refusals}")
NO_SPLIT_FACTOR = "the dim has {n} elements, which {f} does not divide"  # P5
TWO_STATES = "{n} maxes are carried across a loop over that dim ({names}); a pair is built from one"
NO_SPLIT_DIM = "a solo graph is split only over a dim a max of its program reduces; none is asked for"
SEVERAL_DIMS = "a pair is built from the loop state over one dim at a time"
UNSEARCHED = "the graph was not searched; a pair is built on the symbolic dims of a searched graph"


def state_pair(graph: SearchedGraph, ir, split_factor: int) -> list:
    """``[partial, combine]``: the split of solo graph ``graph`` (lowered to ``ir``, with
    its records derived) over the dim of ``graph.split`` that a max of the program reduces,
    built from the loop state of the streaming obstacle served over that dim
    (``lower.state_split``; DESIGN 11).

    The partial is the flat graph up to the loop state on a split of that dim: each split
    returns its accumulators and its own max, float32 at the cut. The combine takes the max
    over the splits, applies the proven repair of each accumulator with ``m := m_i``,
    ``m' := max_i m_i``, sums over the splits and runs the solo graph's own epilogue. Both
    are flat graphs with a max in front of their sums, so each gets its own derivation.

    ``LoweringRejected("graph {i}: split over {dim}: {reason}")`` when no pair is built.
    """
    from ..frontend.tensor import split_names
    from ..ir.graph_utils import boundary_signature
    from ..lower.state_split import (SplitRefused, combine_graph, lower_partial, op_indices,
                                     split_position)
    from ..runtime.reference import output_order, placeholder_order
    from .superopt import max_split_dims
    from .update.instantiate import Recipe

    dims = max_split_dims(graph.program, graph.split)
    if len(dims) != 1:
        names = ",".join(dims or split_names(graph.split)) or "no dim"
        why = SEVERAL_DIMS if len(dims) > 1 else NO_SPLIT_DIM
        raise LoweringRejected(f"graph {graph.index}: split over {names}: {why}")
    (dim,) = dims
    prefix = f"graph {graph.index}: split over {dim}: "
    f = int(split_factor)
    extent = (getattr(graph.program, "sizes", None) or {}).get(dim)
    if f < 1 or (extent is not None and int(extent) % f):
        raise LoweringRejected(prefix + NO_SPLIT_FACTOR.format(n=extent, f=f))
    if graph.symbol_map is None:
        raise LoweringRejected(prefix + UNSEARCHED)
    served = _served_over(graph, ir, dim)
    if not served:
        refusals = "; ".join(getattr(ir, "streaming_refusals", None) or ())
        raise LoweringRejected(prefix + NO_PROVEN_UPDATE + (f": {refusals}" if refusals else ""))
    if len(served) > 1:
        raise LoweringRejected(prefix + TWO_STATES.format(
            n=len(served), names=", ".join(n.name for n in served)))
    (node,) = served
    full, record = node.name, node.streaming
    accs, rules = tuple(record.loops), tuple(record.rules)
    try:
        full_op, acc_ops = op_indices(ir, full, accs)
        partial = lower_partial(graph, full_op, acc_ops, f)
        split_pos = split_position(ir, full, partial)
        combine = combine_graph(ir, full, rules, f, split_pos, out_dtype=graph.dtype)
    except SplitRefused as e:
        raise LoweringRejected(prefix + str(e)) from e
    except Exception as e:  # noqa: BLE001 - a lowering error names itself, as for any graph
        raise LoweringRejected(prefix + f"{type(e).__name__}: {e}") from e
    crossing = boundary_signature(output_order(partial))
    if crossing != boundary_signature(placeholder_order(combine)):
        raise LoweringRejected(  # a bug of the construction: never handed to the build
            prefix + f"the partial returns {crossing} but the combine reads "
            f"{boundary_signature(placeholder_order(combine))}")
    _derive_updates(partial, Recipe.partial(graph, full, accs, solo=ir, split_factor=f))
    _derive_updates(combine, Recipe.combine(graph, full, rules, split_pos, graph.dtype, solo=ir,
                                            split_factor=f))
    return [partial, combine]


def _served_over(graph: SearchedGraph, ir, dim: str) -> list:
    """The nodes of ``ir`` whose streaming obstacle is served (a record) and loops over the
    program's dim ``dim``: the symbol of the searched structure at that dim's place in the
    program's inputs."""
    from ..frontend.tensor import Dim
    from .update.engine import _looped
    from .update.instantiate import Recipe

    recorded = [n for n in ir.nodes if getattr(n, "streaming", None) is not None]
    if not recorded:
        return []
    inputs = [op for op in graph.structure if op["op_type"] == "kn_input_op"]
    wanted = set()
    for spec, op in zip(graph.program.inputs, inputs):
        t = op["output_tensors"][0]
        symbols = t["dim"][: int(t.get("num_dims", len(t["dim"])))]
        if len(symbols) == len(spec.dims):
            wanted |= {s for d, s in zip(spec.dims, symbols)
                       if isinstance(d, Dim) and d.name == dim}
    recipe = Recipe.searched(graph)
    served = []
    for node in recorded:
        try:
            looped = _looped(recipe, ir, node.name)
        except Exception:  # noqa: BLE001 - a looped dim that cannot be named is not this one
            continue
        found = set(looped) if isinstance(looped, frozenset) else {looped}
        if found & wanted:
            served.append(node)
    return served


def _derive_updates(ir, recipe) -> None:
    """Give every streaming obstacle of a freshly lowered flat graph its proven update
    rule, or the reason it has none.

    The obstacles are the scheduler's own (``schedule.streaming.obstacles``, from the
    reduction-axis provenance the policy pins by). For each one that can be streamed, the
    rules are derived once per loop structure (``pipeline.update.derive``: references from
    this very graph, built again at small shapes by its recipe, search, z3 proof, ranking)
    and written on the full-extent node as a ``Streaming`` record. The policy lets a loop
    step under a max only where it finds that record; without one the pin and the refusal
    stay. Nothing the engine does raises out of here (an unexpected error is ``E13``).

    ``ir.streaming_refusals`` (``["{full}: {reason}", ...]``) exists exactly when the graph
    has an obstacle; ``ir.derivations`` holds every derivation the engine made for it.
    ``recipe`` (``pipeline.update.instantiate.Recipe``) builds ``ir`` again at other extents.
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
    from .update import derive

    ir.streaming_refusals = []
    for ob in found:
        why = ob.refused
        if not why:
            derivation = derive(ir, ob, recipe)
            ir.derivations = getattr(ir, "derivations", []) + [derivation]
            why = derivation.refused
            if not why:
                next(n for n in ir.nodes if n.name == ob.full).streaming = derivation.record()
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


# Why a graph with a max has no tiling at all. A max is a max, whoever wrote it, and a loop
# carries it wherever its streaming obstacle is served; a graph with an obstacle that is not
# served says why after this text (``_schedule_once``). Only where it is true: not when the
# softmax filter refused a tiling that held every such row in one block (then the row was not
# longer, and the filter's reasons are the reason).
MAX_ROW_LIMIT = "a max needs its whole row in one block, and this row is longer"
NO_TILING = "no tiling candidates"


def _search(ir_graph, topk: int, force: bool, hints: dict | None,
            refused: Counter | None = None, whole: Counter | None = None):
    """The top ``topk`` tilings of one lowered graph.

    A graph with a streaming obstacle, served or not, keeps only the tilings the build
    schedules (``schedule.streaming.streaming_accept``), decided before the top-k cut; the
    reasons of the others are added to ``refused``, and those of the refused tilings that
    reduce every max without a record in one step to ``whole``. For any other graph
    nothing changes.
    """
    from ..schedule import search_tiling
    from ..schedule.streaming import streaming_accept

    kw = dict(hints or {})
    if kw:
        accepted = inspect.signature(search_tiling).parameters
        unknown = [k for k in kw if k not in accepted]
        if unknown:
            raise ScheduleRejected(f"search_tiling does not accept hints {unknown}")
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
        if stable is not None and whole is not None:
            whole.update(stable.whole)


def _no_tiling(g, ir, reasons: Counter, whole: Counter) -> str:
    """Why lowered graph ``ir`` of searched graph ``g`` has no tiling.

    The row limit only where no tiling the policy made held the row of every max without a
    record in one step (``whole`` is empty): the policy made none, or the softmax filter
    refused only tilings that step such a max. Then, for a max whose streaming obstacle is
    not served, why it is not carried across a loop instead. Otherwise the graph has no
    tiling for the reasons the filter counted (``reasons``), whatever its maxes.
    """
    unstreamed = "; ".join(getattr(ir, "streaming_refusals", None) or ())
    if "max" in g.ops and not whole:
        why = MAX_ROW_LIMIT
        if unstreamed:
            why += f"; it is not carried across a loop: {unstreamed}"
    else:
        why = NO_TILING
        if unstreamed:
            why += f"; a max is not carried across a loop: {unstreamed}"
    if reasons:  # every tiling was refused by the softmax filter: say why
        why += f" ({sum(reasons.values())} refused: " + "; ".join(
            f"{n}x {reason}" for reason, n in reasons.most_common()) + ")"
    return why


def _schedule_once(graphs: GraphSet, config: ScheduleConfig, force: bool):
    plans, refused = [], []
    program = graphs.program
    topk, hints = config.topk, config.hints

    def add(g, variant, role, ir, factor, inputs=None):
        reasons, whole = Counter(), Counter()
        try:
            tiles = _search(ir, topk, force, hints, reasons, whole)
        except ScheduleRejected as e:
            refused.append(Refusal(g.index, variant, role, e))
            return
        if not tiles:
            refused.append(Refusal(g.index, variant, role,
                                   ScheduleRejected(_no_tiling(g, ir, reasons, whole))))
        for rank, t in enumerate(tiles):
            label = f"g{g.index} {variant} {role} #{rank} {describe_tile(t)}"
            plans.append(Plan(g.index, variant, role, t, label, rank, factor, g, inputs))

    combined = set()
    state_split = getattr(graphs, "state_split", ())
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
            else:
                add(g, "solo", "main", ir, None, bind_inputs(ir, program))
            if state_split:
                _state_split_plans(g, graphs, config, add, refused)
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


def _state_split_plans(g, graphs, config, add, refused) -> None:
    """The split plans of solo graph ``g`` of a ``GraphSet`` with ``state_split``: per
    factor of ``config.split_k`` the partial and the combine built from its loop state
    (``lower(g, f)``), or the reason there is no pair (``Refusal(..., "partial", ...)``,
    as for a searched split graph that has no cut). A split that names no factor is refused
    once. No part of the split is dropped silently: a dim of it that no max reduces is not
    split (the search ran unsplit), and one refusal per solo graph says so."""
    from ..frontend.tensor import split_names
    from .superopt import max_split_dims

    maxed = max_split_dims(graphs.program, graphs.state_split)
    others = [d for d in split_names(graphs.state_split) if d not in maxed]
    head = f"graph {g.index}: split over {','.join(maxed)}: "
    if others:
        refused.append(Refusal(g.index, "split", "partial", LoweringRejected(
            head + f"the search ran unsplit, so {','.join(others)} is not split")))
    if not config.split_k:
        refused.append(Refusal(g.index, "split", "partial",
                               LoweringRejected(head + "ScheduleConfig.split_k is empty")))
        return
    for f in config.split_k:
        variant = f"split-k={f}"
        try:
            partial, combine = lower(g, f)
        except LoweringRejected as e:
            refused.append(Refusal(g.index, variant, "partial", e))
            continue
        add(g, variant, "partial", partial, f, bind_inputs(partial, graphs.program))
        add(g, variant, "combine", combine, f)  # one combine per solo graph and factor


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
