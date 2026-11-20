"""Static scale analysis: which ``Exp`` nodes of a lowered graph are softmax exps.

The search never sees a max: ``cl.softmax`` is ``exp(x) / sum(exp(x))`` plus a program-level
request. Which IR node that request refers to cannot be carried through the search (graphs
are re-enumerated with fresh ids, exps are merged and duplicated), so it is *derived* per
lowered graph, without numerics:

    replace the output of one Exp by ``c * exp(x)`` with an unknown positive ``c`` that
    may vary along the Exp's axes, and propagate the *degree* in ``c`` through the graph.

A value is ``Scale(deg, ax)``: its degree in ``c`` and, per tensor axis, the Exp axis ``c``
varies with along it (``None``: constant along that axis). Rules per consumer class, matched
on the exact type:

    the analysed Exp          input degree 0; output degree 1, ax = (0 .. rank-1)
    all inputs of degree 0    degree 0, whatever the op
    ReduceSum / ReduceMax     degree unchanged; a scaled axis that is reduced joins ``const``
    Matmul, one scaled op.    its degree; the contracted axis, if scaled, joins ``const``;
                              a grouped (batch-broadcast) scaled operand is refused
    Matmul, two scaled ops.   refused
    Mul / Div of tensors      degrees add / subtract; two scaled operands must carry ``c``
                              on the same output axes
    Add / Sub of tensors      equal degrees and equal ``c`` axes
    Mul / Div by a constant   degree unchanged (constant / x: negated)
    Add / Sub of a constant, Exp, Silu, Sqrt, RSqrt, SplitView
                              refused on a scaled value
    any other class           refused on a scaled value: there is no identity fall-through

``const`` -- the Exp axes along which ``c`` must be constant -- is iterated to a fixed point.
When the graph is legal and every output has degree 0, ``c = exp(-max over the const axes)``
changes no output: the shift cancels. Each Exp is analysed alone with its own symbol; one
shared symbol is unsound (``exp(x) / exp(y)`` would pass).

``record_for`` turns a report into the plain-data ``Stabilize`` record of the Exp, or says
why not; ``annotate`` applies it to a whole solo graph on behalf of a requesting program.

A split pair cannot cancel the factor inside the partial: every output of a partial keeps
degree 1, each split relative to its own max. ``stabilize_split`` makes the pair exact
instead. The partial exports that max as one more output (``export_shift``); the combine,
which the same analysis proves to cancel the factor when its inputs are seeded with the
partial's output scales (``analyse_split``), weights every partial by ``exp(m_i)`` before
it sums them over the split axis (``exact_combine``): a softmax over the split axis, marked
and served like any other.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..ir.analysis import _map_input_axes_to_consumer_axes
from ..ir.graph import Boundary, ComputeOpNode, Graph, OutputNode, PlaceHolderNode
from ..ir.graph_utils import boundary_signature, find_topo_sort, gather_all_nodes
from ..ir.ops import (Add, Div, Exp, Matmul, Mul, ReduceMax, ReduceSum, RSqrt, Silu, SplitView,
                      Sqrt, Stabilize, Sub, wire_input)
from ..ir.rolling import STAT_DTYPE


class ScaleError(Exception):
    """The scale of one Exp does not cancel; the message names the node where it breaks."""


@dataclass(frozen=True)
class Scale:
    deg: int
    ax: tuple  # per tensor axis: the Exp axis c varies with along it, or None

    def restricted(self, const) -> "Scale":
        return Scale(self.deg, tuple(None if a in const else a for a in self.ax))


@dataclass(frozen=True)
class Reduction:
    """A node that sums a scaled value over an Exp axis (so ``c`` is constant along it)."""

    node: str
    kind: str  # class name
    operand: int
    exp_axis: int
    degree: int


@dataclass
class ScaleReport:
    exp: str | None  # None: a seeded analysis
    rank: int
    const: tuple  # Exp axes (front-relative) along which c must be constant
    outputs: dict  # (OutputNode name, producer name) -> Scale, in graph order
    reductions: tuple  # Reduction, ... in first-seen order
    scaled: dict = field(default_factory=dict)  # node name -> Scale, deg != 0, no placeholders


_RULED = (Matmul, ReduceSum, ReduceMax, Add, Sub, Mul, Div)
_REDUCING = (Matmul, ReduceSum, ReduceMax)
_REFUSED = (Exp, Silu, Sqrt, RSqrt, SplitView)


def _zero(rank) -> Scale:
    return Scale(0, (None,) * rank)


def _shape(node, i=0):
    return tuple(int(s) for s in node.get_shape(i))


def _degree(n, ins, mapped) -> int:
    """Output degree of ``n``: ``ins`` are its operand scales, ``mapped`` the scaled ones."""
    kind = type(n)
    if kind is Matmul:
        if len(mapped) == 2:
            raise ScaleError(f"{n.name}: Matmul of two scaled values")
        return mapped[0][0].deg
    if kind in (ReduceSum, ReduceMax):
        return ins[0].deg
    name = kind.__name__
    if n.const is not None:
        if kind in (Add, Sub):
            raise ScaleError(f"{n.name}: {name} of a constant and a scaled value")
        return -ins[0].deg if (kind is Div and n.const_on_left) else ins[0].deg
    if kind in (Add, Sub):
        if ins[0].deg != ins[1].deg:
            raise ScaleError(f"{n.name}: {name} of degrees {ins[0].deg} and {ins[1].deg}")
        return ins[0].deg
    return ins[0].deg + ins[1].deg if kind is Mul else ins[0].deg - ins[1].deg


def _propagate(graph, exp, const, seeds):
    """One pass with ``const`` fixed.

    Returns ``(scales, outputs, found, reductions, pending)``: ``found`` are the Exp axes
    this pass saw reduced, ``pending`` the refusals that depend on where ``c`` varies. Those
    only count once ``const`` is complete (an axis that turns out constant carries nothing),
    so they are collected; a refusal that does not depend on ``const`` is raised at once.
    """
    scales, outs, found, reds, pending = {}, {}, set(), [], []
    for n in graph.nodes:
        if isinstance(n, PlaceHolderNode):
            rank = len(_shape(n))
            seed = (seeds or {}).get(n.name, _zero(rank))
            if len(seed.ax) != rank:
                raise ScaleError(f"{n.name}: seeded with the scale of a value of rank "
                                 f"{len(seed.ax)}, the input has rank {rank}")
            scales[id(n)] = seed.restricted(const)
            continue
        edges = sorted(n.inputs, key=lambda e: e.dst_id)
        ins = [scales[id(e.src_node)] for e in edges]
        if isinstance(n, OutputNode):
            outs[(n.name, edges[0].src_node.name)] = ins[0]
            continue
        rank = len(_shape(n))
        if n is exp:
            if ins[0].deg:
                raise ScaleError(f"{n.name}: the analysed exp reads a scaled value")
            scales[id(n)] = Scale(1, tuple(range(rank))).restricted(const)
            continue
        if not any(s.deg for s in ins):
            scales[id(n)] = _zero(rank)
            continue
        kind = type(n)
        if kind in _REFUSED:
            raise ScaleError(f"{n.name}: {kind.__name__} of a scaled value")
        if kind not in _RULED:
            raise ScaleError(f"{n.name}: {kind.__name__} has no scale rule")
        mapped = []  # (operand scale, its c axes placed on this node's output axes)
        for e, s in zip(edges, ins):
            if not s.deg:
                continue
            # The axis maps below read the op's own layout: they hold for the operand the
            # op declares, not for a value of another shape wired into it (a rank-0 value
            # has no axis to misplace).
            actual = _shape(e.src_node, e.src_id)
            declared = tuple(int(d) for d in n.input_buffers[e.dst_id].shape)
            if actual and actual != declared:
                raise ScaleError(f"{n.name}: scaled operand {e.dst_id} has shape {actual}, "
                                 f"{kind.__name__} takes {declared}")
            if kind is Matmul:
                own = n.input_batch_dims_A if e.dst_id == 0 else n.input_batch_dims_B
                if len(s.ax) - 2 != n.n_batch_dims or tuple(own) != tuple(n.batch_dims):
                    raise ScaleError(
                        f"{n.name}: Matmul broadcasts its scaled operand {e.dst_id} over the "
                        f"batch ({tuple(own)} -> {tuple(n.batch_dims)})")
            ax = [None] * rank
            for p, cons in enumerate(_map_input_axes_to_consumer_axes(n, e.dst_id, len(s.ax))):
                if s.ax[p] is None:
                    continue
                if not cons:  # reduced / contracted away: c must be constant along it
                    if kind not in _REDUCING:
                        pending.append(f"{n.name}: {kind.__name__} drops a scale axis")
                        continue
                    found.add(s.ax[p])
                    reds.append(Reduction(n.name, kind.__name__, e.dst_id, s.ax[p], s.deg))
                    continue
                if len(cons) != 1:
                    pending.append(f"{n.name}: a scale axis fans out")
                    continue
                (c,) = cons
                if ax[c] is not None and ax[c] != s.ax[p]:
                    pending.append(f"{n.name}: two scale axes land on output axis {c}")
                    continue
                ax[c] = s.ax[p]
            mapped.append((s, tuple(ax)))
        deg = _degree(n, ins, mapped)
        if len(mapped) == 2 and mapped[0][1] != mapped[1][1]:
            pending.append(f"{n.name}: {kind.__name__} of two scaled values on different axes "
                           f"{mapped[0][1]} vs {mapped[1][1]}")
        ax = tuple(next((m[i] for _, m in mapped if m[i] is not None), None) for i in range(rank))
        scales[id(n)] = Scale(deg, ax if deg else (None,) * rank)
    return scales, outs, found, reds, pending


def analyse(graph, exp=None, seeds=None) -> ScaleReport:
    """Propagate the scale of ``exp`` (or of seeded placeholders) to a fixed point of ``const``.

    ``seeds`` maps placeholder names to a ``Scale`` (inputs that already carry the factor).
    Raises ``ScaleError`` naming the first node where the scale cannot be carried.
    """
    const, reductions = set(), []
    while True:
        scales, outs, found, reds, pending = _propagate(graph, exp, const, seeds)
        reductions += [r for r in reds if r not in reductions]
        if found <= const:
            break
        const |= found
    if pending:
        raise ScaleError(pending[0])
    return ScaleReport(
        exp=None if exp is None else exp.name,
        rank=len(_shape(exp)) if exp is not None else 0,
        const=tuple(sorted(const)),
        outputs=outs,
        reductions=tuple(reductions),
        scaled={n.name: scales[id(n)] for n in graph.nodes
                if not isinstance(n, (PlaceHolderNode, OutputNode)) and scales[id(n)].deg},
    )


def exps(graph) -> list:
    """Exact-type match, as ``AXIS_MAP`` does: masked / paged exps are other classes."""
    return [n for n in graph.nodes if type(n) is Exp]


def _rolling_obstacle(exp, report: ScaleReport, reducers) -> str:
    """Why the softmax axis of ``exp`` may not be tiled by a loop (``""`` if it may).

    The rolling form keeps one running max per softmax row and multiplies every scaled
    accumulator by ``alpha[row]`` before it accumulates. That needs every value that still
    has the softmax axis to be computable from one tile, the row axes leading on every
    accumulator, and accumulators that start at zero.
    """
    (axis,) = report.const
    reached, stack = [], [exp]
    while stack:  # nodes that see the exp without crossing a reduction over the softmax axis
        for edge in stack.pop().outputs:
            node = edge.dst_node
            if node.name in reducers or isinstance(node, OutputNode) or node in reached:
                continue
            reached.append(node)
            stack.append(node)
    for node in reached:
        done = [a.name for a in gather_all_nodes(node, descending=False, change_name=False)
                if a.name in reducers]
        if done:
            return f"{node.name} reads the exp next to the finished reduction {done[0]}"
    rows = tuple(a for a in range(report.rank) if a != axis)
    for name in reducers:
        ax = report.scaled[name].ax
        if ax != rows + (None,) * (len(ax) - len(rows)):
            return f"{name} carries the factor on axes {ax}, not on its leading axes"
    for r in report.reductions:
        if r.kind == ReduceMax.__name__:
            return (f"{r.node} takes a max of the scaled value; a rescaled accumulator must "
                    "start at zero")
    return ""


def record_for(graph, exp) -> Stabilize:
    """The ``Stabilize`` record of ``exp`` in a solo graph, else ``ScaleError`` with the reason.

    The Exp is a softmax exp iff the propagation is legal, something sums it, every graph
    output has degree 0, and the shape is one a tiling can serve: exactly one softmax axis,
    a non-batch axis of an Exp with two non-batch axes, every sum of degree 1.
    """
    report = analyse(graph, exp)
    if not report.const:
        raise ScaleError(f"{exp.name}: nothing sums it, so there is no softmax axis")
    for (_, src), s in report.outputs.items():
        if s.deg:
            raise ScaleError(f"{exp.name}: output of {src} keeps degree {s.deg} in its scale")
    return _record(exp, report)


def _softmax_axis(exp, report: ScaleReport) -> int:
    """The softmax axis of ``exp`` (counted from the front) when its shape is one a tiling
    can serve, else ``ScaleError``: exactly one softmax axis, a non-batch axis of an Exp
    with two non-batch axes, every sum of degree 1."""
    if len(report.const) != 1:
        raise ScaleError(f"{exp.name}: the factor must be constant along axes "
                         f"{list(report.const)}; one softmax axis is supported")
    (axis,) = report.const
    rank, n_batch = report.rank, exp.n_batch_dims
    if axis < n_batch:
        raise ScaleError(f"{exp.name}: softmax axis {axis} is a batch axis of the exp")
    if rank - n_batch != 2:
        raise ScaleError(f"{exp.name}: the exp has {rank - n_batch} non-batch axes; the "
                         "softmax axis and one row axis are supported")
    for r in report.reductions:
        if r.degree != 1:
            raise ScaleError(f"{exp.name}: {r.node} accumulates a value of degree {r.degree}")
    return axis


def _record(exp, report: ScaleReport, export=None) -> Stabilize:
    """The record a legal report gives ``exp`` (``_softmax_axis`` says when it gives one)."""
    axis = _softmax_axis(exp, report)
    reducers = tuple(dict.fromkeys(r.node for r in report.reductions))
    return Stabilize(axes=(axis - report.rank,), scaled=tuple((name, 1) for name in reducers),
                     rolling=_rolling_obstacle(exp, report, reducers), export=export)


def annotate(graph, program) -> list:
    """Mark the softmax exps of a solo graph for a program that requested them.

    Every Exp whose scale cancels gets its ``Stabilize`` record; the others stay plain.
    ``ScaleError`` (and nothing is marked) when the graph cannot honour the program's
    requests: no Exp cancels, fewer cancel than were requested, or an Exp is left plain
    although every ``exp`` of the program is a softmax exp.
    """
    records, reasons = [], []
    for exp in exps(graph):
        try:
            records.append((exp, record_for(graph, exp)))
        except ScaleError as e:
            reasons.append(str(e))
    requests = len(program.stabilize_requests)
    plain_exp = any(r.kind == "exp" and r.attr("stabilize") is None for r in program.ops)
    if not records:
        raise ScaleError(reasons[0] if reasons else "the lowered graph has no exp")
    if len(records) < requests:
        why = f" ({'; '.join(reasons)})" if reasons else ""
        raise ScaleError(f"{requests} softmax request(s), {len(records)} exp(s) of this graph "
                         f"can be stabilized{why}")
    if reasons and not plain_exp:
        raise ScaleError(reasons[0])
    # A reduction scaled by two softmaxes would need both factors: no rolling form.
    owners: dict = {}
    for exp, record in records:
        for name, _ in record.scaled:
            owners.setdefault(name, []).append(exp.name)
    marked = []
    for exp, record in records:
        if not record.rolling:
            shared = next(((name, other) for name, _ in record.scaled
                           for other in owners[name] if other != exp.name), None)
            if shared is not None:
                record = replace(record, rolling=(
                    f"{shared[0]} is also scaled by {shared[1]}; the rolling form keeps one "
                    "factor per accumulator"))
        exp.stabilize = record
        marked.append(exp)
    return marked


# ------------------------------------------------------------------------------ split pairs

SHIFT = "shift"  # Boundary.role of the max a partial exports and its combine reads


def _outputs(graph) -> list:
    return [n for n in graph.nodes if isinstance(n, OutputNode)]


def _placeholders(graph) -> list:
    return [n for n in graph.nodes if isinstance(n, PlaceHolderNode)]


def _partial_report(partial, exp) -> ScaleReport:
    """The report of ``exp`` as the softmax exp of a split partial, else ``ScaleError``.

    The solo rule (``record_for``) with one difference: a partial returns the statistics of
    its own split, so nothing cancels the factor there. Every output keeps degree 1, and
    carries the factor on its leading axes, the axes of the Exp without the softmax axis:
    the shape of the shift every one of them is relative to.
    """
    report = analyse(partial, exp)
    if not report.const:
        raise ScaleError(f"{exp.name}: nothing sums it, so there is no softmax axis")
    for (_, src), s in report.outputs.items():
        if s.deg != 1:
            raise ScaleError(f"{exp.name}: partial output of {src} has degree {s.deg} in its "
                             "scale; every output of a split partial keeps degree 1")
    axis = _softmax_axis(exp, report)
    rows = tuple(a for a in range(report.rank) if a != axis)
    for (_, src), s in report.outputs.items():
        if s.ax != rows + (None,) * (len(s.ax) - len(rows)):
            raise ScaleError(f"{exp.name}: partial output of {src} carries the factor on axes "
                             f"{s.ax}, not on its leading axes {rows}")
    return report


def _crossing(signature) -> str:
    return "[" + ", ".join(f"#{index}{':' + role if role else ''} {shape} {dtype}"
                           for index, role, shape, dtype in signature) + "]"


def analyse_split(partial, combine):
    """``(exp, partial report, combine report)`` of a lowered split pair, else ``ScaleError``.

    Partial: exactly one Exp passes ``_partial_report``. Combine: every placeholder is
    seeded with the scale of the partial output that is the same tensor -- paired by
    boundary record (index, role, logical shape and dtype, compared exactly), never by
    position -- and has to cancel the factor: legal, with outputs of degree 0. The scale
    axes are numbered as the axes of the partial's Exp on both sides, so ``const`` of the
    combine report is the split axis in that numbering: the one axis the partials are
    summed over.
    """
    found, reasons = [], []
    for exp in exps(partial):
        try:
            found.append((exp, _partial_report(partial, exp)))
        except ScaleError as e:
            reasons.append(str(e))
    if not found:
        raise ScaleError(reasons[0] if reasons else "the partial has no exp")
    if len(found) != 1:
        raise ScaleError(f"{[exp.name for exp, _ in found]} are softmax exps of one partial; "
                         "a split carries one softmax")
    exp, report = found[0]

    outs, phs = _outputs(partial), _placeholders(combine)
    try:
        wrote, read = boundary_signature(outs), boundary_signature(phs)
    except ValueError as e:  # records on some tensors of one side only, or an index twice
        raise ScaleError(f"split boundary: {e}") from e
    if wrote is None or read is None:
        bare = "partial's outputs" if wrote is None else "combine's inputs"
        raise ScaleError(f"split boundary: the {bare} carry no boundary records, so the two "
                         "sides of the cut cannot be paired")
    if wrote != read:
        raise ScaleError(f"split boundary: the combine reads {_crossing(read)} but the "
                         f"partial writes {_crossing(wrote)}")
    scale_of = {out: s for (out, _), s in report.outputs.items()}
    written = {out.boundary.index: scale_of[out.name] for out in outs}
    seeds = {ph.name: written[ph.boundary.index] for ph in phs}
    combined = analyse(combine, None, seeds)
    for (_, src), s in combined.outputs.items():
        if s.deg:
            raise ScaleError(f"combine: output of {src} keeps degree {s.deg} in the scale of "
                             f"{exp.name}")
    if len(combined.const) != 1:
        raise ScaleError(f"combine: the partials of {exp.name} are summed over axes "
                         f"{list(combined.const)} of its scale; one split axis is supported")
    return exp, report, combined


def _stored_dtype(edge) -> str:
    """The dtype the value ``edge`` reads is stored in."""
    src = edge.src_node
    return src.out_dtype if isinstance(src, ComputeOpNode) else str(src.get_dtype(edge.src_id))


def export_shift(partial, exp, report: ScaleReport):
    """``(partial, m)``: the partial with its shift as one more output.

    ``m = ReduceMax(scores)`` over the softmax axis reads what the Exp reads, in float32 (a
    max of stored values is exact there), and feeds nothing but its own ``OutputNode``. It is
    a sibling of the Exp, not an ancestor of the sums: the tiling policy steps it with them
    and has nothing to pin. The Exp's record names it (``Stabilize.export``), so the rewrite
    reuses it as the shift -- the carried max, where the softmax axis is looped -- and what
    the kernel returns is the max its sums are relative to.

    The outputs the cut produced keep their names and their boundary index; the shift takes
    the next index and the role ``"shift"``. The kernel returns it first, in front of the
    sums that are relative to it: the tiling policy walks a graph in the order of its
    outputs, and its memory model frees a tile when the last reader has run. The kernel
    takes the max before the exp, and listed there the max lets the scores go when the exp
    has read them, as in the two-output partial. Listed behind the sums it would hold the
    scores across the whole softmax, one score tile more than the kernel ever has, and the
    tilings with the widest key steps would go over the shared-memory cap (8 of 32 for MHA
    decode split 16 ways, 32 of 80 for GQA and MQA decode at every factor).
    """
    (axis,) = report.const
    scores = exp.inputs[0]
    dims = _shape(exp)
    m = ReduceMax(dims, axis=-1 if axis == len(dims) - 1 else axis,
                  name=f"reduce_max_{axis}_{exp.name}", dtype=_stored_dtype(scores),
                  out_dtype=STAT_DTYPE)
    wire_input(scores.src_node, m, 0, scores.src_id)
    outs = _outputs(partial)
    shift = OutputNode(m, dtype=STAT_DTYPE)
    shift.boundary = Boundary(1 + max(out.boundary.index for out in outs), SHIFT)
    while shift.name in {out.name for out in outs}:  # as ``to_ir`` names a further output
        shift.name = shift.name[:-1] + chr(ord(shift.name[-1]) + 1)
    exp.stabilize = _record(exp, report, export=m.name)
    return Graph(nodes=find_topo_sort([shift] + outs)), m


def exact_combine(combine, shift_shape, split_axis: int) -> Graph:
    """The lowered ``combine`` as a softmax over the split axis: ``w = exp(m)`` weights
    every partial before the partials are summed.

    Derived from the combine the cut produced, not rebuilt from a template: every
    ``ReduceSum`` over ``split_axis`` that reads a boundary input ``p`` becomes
    ``ReduceSum(p * w)`` with a float32 product, where ``w = Exp(m)`` reads one more
    boundary input, the float32 shift of shape ``shift_shape`` (role ``"shift"``).
    Everything behind the sums is kept as it is. ``w`` is then held to the solo rule
    (``record_for``), which gives it its record: the scale analysis proves that the max of
    ``m`` over the split axis cancels, and the rewrite turns ``w`` into ``exp(m - M)`` per
    schedule, ``M`` a running max where the split axis is looped. The graph itself stays
    shift-free: a ``ReduceMax`` ahead of the sums would pin the split axis to one tile.

    A boundary input that anything but such a sum reads has no place for its weight; the
    combine is refused, with the reader named.
    """
    shift_shape = tuple(int(s) for s in shift_shape)
    rank = len(shift_shape)
    inputs = _placeholders(combine)
    sums = []
    for p in inputs:
        for edge in p.outputs:
            reader = edge.dst_node
            if type(reader) is not ReduceSum or reader.flat_axis != split_axis:
                raise ScaleError(
                    "combine: not every boundary input is summed over the split axis "
                    f"({p.name} is read by {type(reader).__name__} {reader.name})")
            if not any(reader is s for s in sums):
                sums.append(reader)
    if not sums:
        raise ScaleError("combine: no boundary input is summed over the split axis")
    if any(p.boundary is None for p in inputs):
        raise ScaleError("combine: its inputs carry no boundary records, so the shift has no "
                         "place among them")

    w = Exp(shift_shape, name="exp_shift", dtype=STAT_DTYPE, out_dtype=STAT_DTYPE,
            n_batch_dims=max(rank - 2, 0))
    m = w.inputs[0].src_node
    m.name = "PlaceHolder_shift"
    m.boundary = Boundary(1 + max(p.boundary.index for p in inputs), SHIFT)
    for old in sums:
        p = old.inputs[0].src_node
        dims = _shape(p)
        if dims == shift_shape:
            broadcast = -1
        elif dims[:-1] == shift_shape:
            broadcast = 0  # B is A without its last axis: one weight per row of the tile
        else:
            raise ScaleError(f"combine: {p.name} {dims} is not the shift {shift_shape} with "
                             "at most one trailing axis")
        weighted = Mul(dims, broadcast=broadcast, name=f"{old.name}_weighted",
                       dtype=STAT_DTYPE, out_dtype=STAT_DTYPE, n_batch_dims=w.n_batch_dims)
        new = ReduceSum(dims, axis=old.flat_axis_arg, name=old.name, dtype=STAT_DTYPE,
                        out_dtype=old.out_dtype)
        p._out_edges[:] = [e for e in p._out_edges if e is not old.inputs[0]]
        wire_input(p, weighted, 0)
        wire_input(w, weighted, 1)
        wire_input(weighted, new, 0)
        for edge in list(old.outputs):
            wire_input(new, edge.dst_node, edge.dst_id)
    exact = Graph(nodes=find_topo_sort(_outputs(combine)))
    # Held to the rule it was derived from; the record is the analysis's own.
    record = record_for(exact, w)
    if record.axes != (split_axis - rank,) or {name for name, _ in record.scaled} != {
            old.name for old in sums}:
        raise ScaleError(f"combine: the weights cancel over axes {record.axes} in "
                         f"{[name for name, _ in record.scaled]}, not over the split axis "
                         f"{split_axis - rank} in {[old.name for old in sums]}")
    w.stabilize = record
    return exact


def stabilize_split(partial, combine, program):
    """``(partial, combine)`` of a lowered split pair, made exact for a program that
    requested a softmax; ``ScaleError`` (and the pair is refused) when it cannot be.

    A split carries one softmax: the program made one request, and unless it also wrote a
    plain ``exp`` the softmax exp is the only Exp of the pair.
    """
    requests = len(program.stabilize_requests)
    if requests != 1:
        raise ScaleError(f"{requests} softmax requests; a split carries one softmax")
    exp, report, combined = analyse_split(partial, combine)
    if not any(r.kind == "exp" and r.attr("stabilize") is None for r in program.ops):
        plain = [n.name for graph in (partial, combine) for n in exps(graph) if n is not exp]
        if plain:
            raise ScaleError(f"{plain} would stay plain exps, and the program wrote none")
    (axis,), (split,) = report.const, combined.const
    rows = tuple(a for a in range(report.rank) if a != axis)  # the Exp axes of the shift
    dims = _shape(exp)
    partial, _ = export_shift(partial, exp, report)
    combine = exact_combine(combine, tuple(dims[a] for a in rows), rows.index(split))
    return partial, combine
