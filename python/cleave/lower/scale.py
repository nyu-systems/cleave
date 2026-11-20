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
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..ir.analysis import _map_input_axes_to_consumer_axes
from ..ir.graph import OutputNode, PlaceHolderNode
from ..ir.graph_utils import gather_all_nodes
from ..ir.ops import (Add, Div, Exp, Matmul, Mul, ReduceMax, ReduceSum, RSqrt, Silu, SplitView,
                      Sqrt, Stabilize, Sub)


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
            seed = (seeds or {}).get(n.name, _zero(len(_shape(n))))
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
    reducers = tuple(dict.fromkeys(r.node for r in report.reductions))
    return Stabilize(axes=(axis - rank,), scaled=tuple((name, 1) for name in reducers),
                     rolling=_rolling_obstacle(exp, report, reducers))


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
