"""Streaming obstacles: a loop that sits under a value which needs the whole looped dim.

A flat graph can keep the stable softmax as it is written, ``exp(x - max(x)) / sum(...)``.
A reduction over a dim whose operand is computed from another reduction over that same dim
(the row max) cannot simply be looped: every iteration would need the finished max. The
policy pins such a reduction to its full extent, and a row longer than one tile has no
tiling.

The pair is a *streaming obstacle* when all of this holds (:func:`obstacles`):

  1. the ancestor reduction is a running reducer (``running_max``): its value over a prefix
     of the dim is the value the flat graph computes for that prefix;
  2. it and every descendant reduction it overlaps reduce the same tensor dim (one axis
     each, of one extent, the descendant's over at least the ancestor's provenance), so
     one loop counter steps all of them;
  3. no overlapped descendant is itself an ancestor of another one over that dim (that
     needs two passes: the traced order ``(exp / sum) @ v``);
  4. nothing computed from the ancestor on the way to the descendants is read elsewhere.

It is *served* where, in addition, an update rule was found and PROVEN for every
overlapped descendant (``cleave.pipeline.update``): the full-extent node carries a
``Streaming`` record (``cleave.ir.update_rule``) whose key is the key of the very graph it
sits on (:func:`streamed`). Then the policy lets the descendants step (no pin), steps the
max with them, and ``apply_tiling`` builds the one loop it always built. :func:`assemble`,
called at the tail of ``apply_tiling``, makes the max carried state and puts the proven
repair in front of each accumulator through the loop-carried mechanism
(``GraphNode.carry``, ``Graph.lag_nodes``, ``Graph.accum_scale``). Every other overlap
keeps its pin and its refusal.

A graph without a record never gets past the first line of :func:`streamed`: every hook of
this module is inert for it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..core.errors import ScheduleRejected
from ..ir.graph import ComputeOpNode, Graph, Iterator
from ..ir.graph_utils import edit_graph, graph_tiers, tmap_by_name
from ..ir.rolling import STAT_DTYPE, build_expression, check_expression
from ..ir.update_rule import Obstacle, Streaming, loop_key

__all__ = [
    "FOREIGN_RECORD", "Obstacle", "Site", "Streaming", "StreamingRefused", "assemble",
    "carry_early", "obstacles", "recorded", "sites", "streamed", "streaming_accept",
]

# A record is honoured only on the graph it was derived for (``update_rule.loop_key``).
FOREIGN_RECORD = "its update rule was derived for another graph"


class StreamingRefused(ScheduleRejected):
    """This tiling cannot carry the full-extent value; the candidate is dropped."""


# ----------------------------------------------------------------------------- detection


def _axis_tokens(red_tokens, node):
    tokens = red_tokens.get(node)
    return tokens[0] if tokens else set()


def obstacles(ordered_nodes, red_tokens, ancestors) -> list:
    """Every ancestor/descendant reduction overlap of a flat graph whose ancestor is a
    running reducer, classified (``Obstacle.refused`` says why one is not a streaming
    obstacle). ``red_tokens`` is ``build_reduction_axis_provenance``, ``ancestors`` is
    ``tiling.reduction_ancestors``; both are what the policy already has.
    """
    reductions = [n for n in ordered_nodes if n in red_tokens and len(n.raxis) > 0]
    found = []
    for full in reductions:
        if not getattr(full, "running_max", False):
            continue
        tokens = _axis_tokens(red_tokens, full)
        if not tokens:
            continue
        loops = [
            d for d in reductions
            if d is not full and full in ancestors.get(d, ()) and any(
                t & tokens for t in red_tokens[d] if t)
        ]
        if not loops:
            continue
        refused = ""
        if len(full.raxis) != 1:
            refused = f"{full.name} reduces {len(full.raxis)} axes; one is carried"
        for d in loops:
            if refused:
                break
            # A contraction ties the dim of its other operand to the same axis (the rows of
            # V under the keys of the scores), so the descendant's axis may run over more
            # source dims than the max's: it has to run over ALL of the max's.
            if len(d.raxis) != 1 or not tokens <= _axis_tokens(red_tokens, d) \
                    or int(d.raxis[0].dom.extent) != int(full.raxis[0].dom.extent):
                refused = (f"{d.name} does not reduce the whole dim {full.name} reduces "
                           "along one axis of the same extent; one loop counter cannot "
                           "step both")
        for d in loops:
            if refused:
                break
            inner = [a for a in loops if a is not d and a in ancestors.get(d, ())]
            if inner:
                refused = (f"{d.name} reads {inner[0].name}, a finished reduction over the "
                           f"dim {full.name} reduces: that needs two passes over the row")
        if not refused:
            refused = _read_outside(full, loops, ancestors)
        found.append(Obstacle(full.name, tuple(d.name for d in loops), refused))
    return found


def _read_outside(full, loops, ancestors) -> str:
    """Why a value computed from ``full`` on the way to ``loops`` cannot stay inside their
    loop, or ``""``: it has a reader that is not on the way to any of them. That reader
    would need every per-iteration value at once (a softmax that is returned itself: the
    exp is divided by its own finished sum). Readers of ``full`` itself are allowed: the
    finished max may be returned."""
    inside = set()
    for d in loops:
        inside.update(n.name for n in ancestors.get(d, ()))
        inside.add(d.name)
    stack, seen = [full], set()
    while stack:
        n = stack.pop()
        for e in n.outputs:
            reader = e.dst_node
            if n is not full and reader.name not in inside:
                return (f"{n.name}, computed from {full.name} for every element of the "
                        f"looped dim, is also read by {reader.name}, which is not summed "
                        "over that dim: that needs two passes over the row")
            if reader.name in inside and id(reader) not in seen and not any(
                    reader is d for d in loops):
                seen.add(id(reader))
                stack.append(reader)
    return ""


def _honoured(record, flat_graph, full: str, loops: tuple) -> bool:
    """Whether ``record`` was derived for ``loops`` under ``full`` on ``flat_graph``."""
    return (flat_graph is not None and tuple(record.loops) == tuple(loops)
            and record.key == loop_key(flat_graph, full, tuple(loops)))


def streamed(ordered_nodes, red_tokens, ancestors, flat_graph=None) -> dict:
    """``{full-extent node: [looped descendants]}`` for the obstacles of a flat graph that
    are served: classified as streaming AND carrying a proven rule for every descendant,
    derived for this very graph.

    ``ordered_nodes`` are the op nodes of ``flat_graph`` (or of a clone of it); the record
    is honoured only where its loops are the obstacle's and its key is
    ``loop_key(flat_graph, ...)``. Anything else keeps the pin, and ``flat_graph`` is told
    why (``streaming_refusals``). Without ``flat_graph`` (a policy built by hand) nothing
    is served.
    """
    if flat_graph is None or not any(
            getattr(n, "streaming", None) is not None for n in ordered_nodes):
        return {}
    by_name = {n.name: n for n in ordered_nodes}
    served = {}
    for ob in obstacles(ordered_nodes, red_tokens, ancestors):
        record = getattr(by_name[ob.full], "streaming", None)
        if ob.refused or record is None:
            continue
        if not _honoured(record, flat_graph, ob.full, ob.loops):
            refusals = getattr(flat_graph, "streaming_refusals", None)
            if refusals is None:
                refusals = flat_graph.streaming_refusals = []
            if f"{ob.full}: {FOREIGN_RECORD}" not in refusals:
                refusals.append(f"{ob.full}: {FOREIGN_RECORD}")
            continue
        served[by_name[ob.full]] = [by_name[name] for name in ob.loops]
    return served


# ------------------------------------------------------------------------------ assembly
#
# Everything below edits a finished tiled graph. It is the mechanical half of what
# ``schedule/stabilize.py`` does for the hand-derived pass and reuses that module's helpers.
# It is much smaller: the max and the shifted scores already exist in the flat graph, so
# only the carried state and the repair are added.


@dataclass(frozen=True)
class Site:
    """Where a full-extent value ended up looped, and what it was given."""

    full: str
    tier: str
    axis: int  # the looped dim in the stored tile of the max's operand
    extent: int  # of that dim in the flat graph
    block: int  # ... held by one grid block
    step: int  # ... held by one iteration
    repaired: tuple  # names of the accumulators that got a repair


def _refused(node, text) -> StreamingRefused:
    return StreamingRefused(f"{node.name}: {text}")


def _locate(tier, owner, full, record):
    """Classify one recorded full-extent node from the tier that was built (read-only).

    ``None``: its dim is whole in the tile it is computed from (the main tier, or a loop
    that steps another dim): the flat graph's own max is right as it stands. A ``Site``:
    the tier's loop steps its dim. Anything a tiling can be refused for is decided here,
    before a node is touched.
    """
    from .stabilize import _descends, _is, _squeezed, _stored_tile

    if owner is None:
        return None
    where = owner.name
    edge = full.inputs[0]
    operand = _stored_tile(tier, edge)
    dims = tuple(int(s) for s in edge.src_node.get_shape(edge.src_id))
    axis = operand.position(full.flat_axis, len(dims))
    if axis is None:
        raise _refused(full, f"the dim it reduces is not in the tile "
                             f"{tuple(operand.tile_size)} of its operand")
    extent = dims[full.flat_axis]
    block, step = int(operand.shape[axis]), int(operand.tile_size[axis])
    if step == block:
        return None
    if block < extent:
        raise _refused(full, f"the dim it reduces is spread over grid blocks ({block} of "
                             f"{extent} per block); a value cannot be carried across blocks")
    if step == 1:
        raise _refused(full, "one element of the reduced dim per iteration; the tile "
                             "would lose that axis")
    if not isinstance(tier.extent, int) or tier.extent * step != block:
        raise _refused(full, f"{where} iterates {tier.extent} times but the reduced dim is "
                             f"stepped {step} of {block}")
    if full.out_dtype != STAT_DTYPE:
        raise _refused(full, f"it is {full.out_dtype}; a carried value starts at a "
                             f"{STAT_DTYPE} number")
    tmaps = tmap_by_name(tier)
    rows = _squeezed(tmaps[full.name].tile_size)
    if not rows:
        raise _refused(full, f"its tile {tuple(tmaps[full.name].tile_size)} holds a single row")
    rules = dict(record.rules)
    under = [n for n in tier.accum_nodes if n is not full and _descends(n, full, tier)]
    missing = sorted(set(rules) - {n.name for n in under})
    if missing:
        raise _refused(full, f"{missing} read(s) it but not as accumulator(s) of the loop "
                             f"{where} that steps its dim")
    for acc in under:
        if acc.name not in rules:
            raise _refused(full, f"accumulator {acc.name} of {where} is computed from it but "
                                 "has no proven update rule")
        if getattr(acc, "accum_init", None) is not None:
            raise _refused(full, f"accumulator {acc.name} starts at {acc.accum_init}; a "
                                 "repaired accumulator must start at zero")
        if any(a is acc for a in tier.accum_scale):
            raise _refused(full, f"accumulator {acc.name} is already repaired; one repair "
                                 "per accumulator")
        acc_tile = tuple(int(t) for t in tmaps[acc.name].tile_size)
        if _squeezed(acc_tile)[: len(rows)] != rows:
            raise _refused(full, f"accumulator {acc.name} tile {acc_tile} does not lead with "
                                 f"the row tile {tuple(rows)}")
        if len(_squeezed(acc_tile)) != len(acc_tile):
            raise _refused(full, f"accumulator {acc.name} has a unit extent in its tile "
                                 f"{acc_tile}")
    for node in tier.nodes:  # nothing computed from the per-iteration max leaves the loop
        if isinstance(node, ComputeOpNode) and not _is(node, tier.accum_nodes) \
                and _descends(node, full, tier):
            outside = [e.dst_node.name for e in node.outputs if not _is(e.dst_node, tier.nodes)]
            if outside:
                raise _refused(full, f"{node.name}, computed from it per iteration, is read "
                                     f"by {outside} outside {where}")
    for _, factor in record.rules:  # every repair has tile ops
        try:
            check_expression(factor, stem=full.name)
        except ValueError as e:
            raise StreamingRefused(str(e)) from e
    return Site(full.name, where, axis, extent, block, step, tuple(n.name for n in under))


def _exp_class(full, tier):
    """The exp between the max and what it feeds, or None: the repair uses the same base
    as the values it repairs."""
    from ..ir.ops import Exp
    from .stabilize import _is

    stack, seen = [full], set()
    while stack:
        n = stack.pop()
        for e in n.outputs:
            d = e.dst_node
            if id(d) in seen or not _is(d, tier.nodes):
                continue
            seen.add(id(d))
            if isinstance(d, Exp):
                return d
            stack.append(d)
    return None


def carry_early(main, rstep_map) -> None:
    """Register every recorded full-extent value that its loop steps as carried state of
    that loop, while ``apply_tiling`` is still building (before the main node list is
    walked from the outputs): a value that is also read OUTSIDE the loop -- the max a split
    partial returns next to its sums -- has to leave the loop as an accumulator, like any
    other value that does. Appended after the loop's own accumulators; ``assemble`` adds
    the lag copy and the repairs at the tail."""
    for tier, owner in graph_tiers(main):
        if owner is None:
            continue
        for node in list(tier.nodes):
            if getattr(node, "streaming", None) is None or node not in rstep_map:
                continue
            steps = list(rstep_map[node].values())
            if steps and int(steps[0]) < int(node.raxis[0].dom.extent):
                owner.carry(node)


def _carry(tier, owner, full, record, site, taken):
    """Make ``full`` carried state of its loop and repair what was accumulated under it.
    Only called inside ``edit_graph``."""
    from ..ir.ops import Copy, Exp, wire_input
    from .stabilize import _weighted_slices

    rows = tuple(int(s) for s in full.get_shape())
    nb = full.n_batch_dims
    it = tmap_by_name(tier)[full.name]
    exp = _exp_class(full, tier)
    lag = Copy(rows, name=f"{full.name}_prev", dtype=STAT_DTYPE, out_dtype=STAT_DTYPE,
               n_batch_dims=nb)
    wire_input(full, lag, 0)
    made = {}
    factors = {
        name: build_expression(expr, {"m": lag, "m'": full}, made, stem=full.name, rows=rows,
                               n_batch_dims=nb, exp_cls=Exp if exp is None else type(exp))
        for name, expr in record.rules
    }
    new_nodes = [lag] + list(made.values())
    clash = sorted({n.name for n in new_nodes} & taken)
    assert not clash, f"{full.name}: node name(s) {clash} are already taken in this graph"
    taken.update(n.name for n in new_nodes)
    for n in new_nodes:
        tier.tmap[n] = Iterator(it.offset, it.shape, it.tile_size, it.axes)
    tier.nodes.extend(new_nodes)
    # Where the carried value starts (this instance only; the class default stays the
    # identity of the reducer).
    full.accum_init = type(full).carry_init
    owner.carry(full)  # appended after the loop's own accumulators
    tier.lag_nodes.append(lag)
    accs = [acc for acc in tier.accum_nodes if acc.name in site.repaired]
    for acc in accs:
        tier.accum_scale[acc] = factors[acc.name]
    if exp is not None and _weighted_slices(tier, exp, accs):
        tier.loop_kind = "serial"
    full.streaming, full.streamed = None, site  # served on this (tiled) copy


def recorded(graph) -> list:
    """Full-extent nodes of a flat or tiled graph that carry a record: ``(tier, owner, node)``."""
    return [
        (tier, owner, n)
        for tier, owner in graph_tiers(graph)
        for n in tier.nodes
        if getattr(n, "streaming", None) is not None
    ]


def sites(tiled) -> list:
    """The ``Site`` of every full-extent value of a tiled graph that :func:`assemble` served."""
    return [n.streamed for tier, _ in graph_tiers(tiled) for n in tier.nodes
            if getattr(n, "streamed", None) is not None]


def assemble(tiled: Graph, flat_graph=None) -> Graph:
    """Serve every recorded full-extent value of a tiled graph, in place.

    ``flat_graph`` is the graph the tiling is of. A record is consumed only where it was
    derived for that graph (the check ``streamed`` makes for the policy, made again here
    before anything is edited). ``StreamingRefused`` if a value cannot be served by this
    tiling; nothing has been edited then. A value whose dim is whole where it is computed
    needs nothing.
    """
    plan = []
    for tier, owner, full in recorded(tiled):
        record = full.streaming
        site = _locate(tier, owner, full, record)
        if site is None:
            continue
        if not _honoured(record, flat_graph, full.name, record.loops):
            raise _refused(full, FOREIGN_RECORD)
        plan.append((tier, owner, full, site))
    if not plan:
        return tiled
    taken = {n.name for tier, _ in graph_tiers(tiled) for n in tier.nodes}
    with edit_graph(tiled):
        for tier, owner, full, site in plan:
            _carry(tier, owner, full, full.streaming, site, taken)
    return tiled


def streaming_accept(flat_graph):
    """``search_tiling(accept=...)`` filter for a graph with a recorded full-extent value,
    else ``None``: a candidate is kept iff the default build schedules it (the same
    ``schedule_kernel`` the build calls), decided before the top-k cut. ``accept.reasons``
    counts the refusals by reason.

    Also for a graph whose obstacle is NOT served (``streaming_refusals`` lists why): the
    policy still steps such a max on its own while its descendants are pinned, and every
    one of those tilings is refused by ``apply_tiling`` (the scores would exist per
    iteration and whole at once).
    """
    if not recorded(flat_graph) and not getattr(flat_graph, "streaming_refusals", None):
        return None
    from .stabilize import _reason_key
    from .tiling import schedule_kernel

    def accept(cand) -> bool:
        try:
            schedule_kernel(cand)
        except ScheduleRejected as e:
            accept.reasons[_reason_key(e)] += 1
            return False
        return True

    accept.reasons = Counter()
    return accept
