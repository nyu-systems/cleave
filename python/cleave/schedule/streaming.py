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
  4. nothing computed from the ancestor on the way to the descendants is read elsewhere;
  5. every overlapped descendant is a plain sum or matmul (by exact type) that starts at
     zero: the repair multiplies what such an accumulator holds, and nothing else is
     modelled by the proof (a max of the scaled value starts at ``-inf``).

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

import re
from collections import Counter
from dataclasses import dataclass

from ..core.errors import ScheduleRejected
from ..ir.graph import AccumEdge, ComputeOpNode, Graph, Iterator
from ..ir.graph_utils import edit_graph, graph_tiers, tmap_by_name
from ..ir.ops import Copy, Exp, Matmul, ReduceSum, wire_input
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
        for d in loops:
            if refused:
                break
            refused = _not_a_sum(full, d)
        found.append(Obstacle(full.name, tuple(d.name for d in loops), refused))
    return found


def _not_a_sum(full, d) -> str:
    """Why ``d`` cannot accumulate under a carried ``full``, or ``""``: the repair rescales
    what an accumulator holds, which is right for a sum (or a contraction) that starts at
    zero and nothing else the proof models. A subclass may accumulate otherwise (a max is
    a ``ReduceSum``), so the type is asked exactly."""
    if type(d) in (ReduceSum, Matmul) and getattr(d, "accum_init", None) is None:
        return ""
    return (f"{d.name} is a {type(d).__name__}: the reductions that loop under {full.name} "
            "must be sums or matmuls that start at zero")


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
# Everything below edits a finished tiled graph: the max and the shifted scores already
# exist in the flat graph, so only the carried state and the repair are added.


def _stored_tile(tier, edge) -> Iterator:
    """The tile of the value ``edge`` reads, in the tier that stores it: an accumulator's
    buffer is the one its loop's own tier allocates."""
    if isinstance(edge, AccumEdge):
        return tmap_by_name(edge.src_node.graph)[edge.accum_node.name]
    return tmap_by_name(tier)[edge.src_node.name]


def _is(node, nodes) -> bool:
    return any(node is n for n in nodes)


def _squeezed(tile) -> list:
    return [int(t) for t in tile if int(t) != 1]


def _descends(node, ancestor, tier) -> bool:
    """Whether ``node`` is computed from ``ancestor`` inside ``tier``."""
    stack, seen = [node], set()
    while stack:
        n = stack.pop()
        if n is ancestor:
            return True
        if id(n) in seen:
            continue
        seen.add(id(n))
        stack.extend(e.src_node for e in n.inputs if _is(e.src_node, tier.nodes))
    return False


def _weighted_slices(tier, exp, accs) -> list:
    """The slices the loop of ``tier`` loads that are multiplied into the softmax's output:
    placeholders the loop cuts per iteration from which a rescaled accumulator of ``accs``
    is computed without passing through ``exp``, read by something other than a matmul.

    A pipelined loop keeps the slices of several iterations in one buffer and loads the
    next ones while the body runs; the body says when it is done with a slice, and the
    slot is loaded again. A matmul reads its slice in place, before that. Any other op
    first copies the slice into registers, and what tells the loader that the slot is
    free is not something the device compiler treats as touching memory: it may read the
    slot later, where the values are used. Here they are used last by the sums over the
    softmax axis, one cross-thread reduction per element, behind the running max and the
    rescale. Measured on a split combine (8 partials, 64 value columns a thread, slots
    reused after 3 iterations): columns 24 and up of every row came from a later
    iteration's partials, a different count of them on every launch.

    A plain loop loads one slice, waits for it and uses it; nothing is loaded behind its
    back. The slices of the scores (which pass through the exp) and of a value matmul are
    not in this list: every attention kernel keeps its pipeline.
    """
    tmaps = tmap_by_name(tier)
    found, stack, seen = [], list(accs), set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for edge in node.inputs:
            src = edge.src_node
            if src is exp or not _is(src, tier.nodes):
                continue
            if not src.is_placeholder():
                stack.append(src)
                continue
            it = tmaps[src.name]
            cut = any(int(whole) != int(part) for whole, part in zip(it.shape, it.tile_size))
            if cut and type(node) is not Matmul and not _is(src, found):
                found.append(src)
    return found


def _reason_key(error) -> str:
    """A refusal's first line with bare numbers folded, so one reason counts once whatever
    the tile sizes were. Digits that belong to a name (``exp_5``) are kept."""
    line = (str(error).splitlines() or [""])[0]
    return re.sub(r"(?<![\w.\-])\d+(?![\w.])", "N", line)[:200]


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


def _other_lead(flat_graph, full, acc):
    """Why the leading dims of ``acc`` are not the dims of ``full``, or None.

    The repair multiplies ``acc`` by a factor tiled like ``full`` (``Rescale(lead)``):
    element ``[i, ..., j]`` of the accumulator by element ``[i, ...]`` of the factor. That
    is right only where the leading dims of the accumulator ARE the dims of ``full``, which
    their tile extents cannot tell when two of them are equal (``(v @ e) / sum(e, 0)``
    leads with the rows of ``v``; its max is indexed by its columns). It is read off the
    output-axis provenance of the flat graph (which input dims each output dim is), as the
    policy reads it: dim ``k`` of ``acc`` must come from where dim ``k`` of ``full`` comes
    from, and no other dim of ``acc`` may. Where one input dim is both (``(x^T @ e) /
    sum(e, 0)``: the rows of the accumulator and the columns it sums under the max are the
    columns of ``x``), provenance cannot say which dim the factor belongs to, and it is
    the trailing one.

    A dim with no provenance on either side is left to the tile check before this one: a
    unit dim comes from nowhere (one head, MLA decode's batch of one), and so does a dim
    the analysis does not follow through a view (the heads of a split partial)."""
    if flat_graph is None:
        return None  # nothing is served without its flat graph anyway (R1)
    from .tiling import _scheduled_axes

    tokens = {(n.name, out): t for (n, out), t in _scheduled_axes(flat_graph).out_tokens.items()}
    rows, lead = tokens.get((full.name, 0)), tokens.get((acc.name, 0))
    if rows is None or lead is None:
        return f"its dim 0 is not dim 0 of {full.name}"
    for k, dim in enumerate(rows):
        if k >= len(lead) or bool(dim) != bool(lead[k]) or (dim and not dim & lead[k]):
            return f"its dim {k} is not dim {k} of {full.name}"
        also = next((j for j, other in enumerate(lead) if j != k and dim & other), None)
        if also is not None:
            return f"dim {k} of {full.name} is also its dim {also}"
    return None


def _locate(tier, owner, full, record, flat_graph=None):
    """Classify one recorded full-extent node from the tier that was built (read-only).

    ``None``: its dim is whole in the tile it is computed from (the main tier, or a loop
    that steps another dim): the flat graph's own max is right as it stands. A ``Site``:
    the tier's loop steps its dim. Anything a tiling can be refused for is decided here,
    before a node is touched. ``flat_graph`` is the graph the tiling is of.
    """
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
        other = _other_lead(flat_graph, full, acc)
        if other is not None:
            raise _refused(full, f"accumulator {acc.name} tile {acc_tile} does not lead with "
                                 f"the row tile {tuple(rows)}: {other}")
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
    plan, claimed = [], set()
    for tier, owner, full in recorded(tiled):
        record = full.streaming
        site = _locate(tier, owner, full, record, flat_graph)
        if site is None:
            continue
        if not _honoured(record, flat_graph, full.name, record.loops):
            raise _refused(full, FOREIGN_RECORD)
        # One factor per accumulator (``Graph.accum_scale``): a sum that two carried values
        # repair (``sum(pa * pb)`` under the maxes of two softmaxes) would keep the factor
        # of the last one only. ``_locate`` reads the tier as it is, before either is added.
        for name in site.repaired:
            if (id(tier), name) in claimed:
                raise _refused(full, f"accumulator {name} is already repaired; one repair "
                                     "per accumulator")
            claimed.add((id(tier), name))
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
    counts the refusals by reason; ``accept.whole`` counts, by reason, the refused tilings
    that reduce every max without a record in one step (its row fitted one block, so the
    tiling was refused for something else).

    Also for a graph whose obstacle is NOT served (``streaming_refusals`` lists why): the
    policy still steps such a max on its own while its descendants are pinned, and every
    one of those tilings is refused by ``apply_tiling`` (the scores would exist per
    iteration and whole at once).
    """
    if not recorded(flat_graph) and not getattr(flat_graph, "streaming_refusals", None):
        return None
    from .tiling import _iterative_reductions, schedule_kernel

    held = [n for tier, _ in graph_tiers(flat_graph) for n in tier.nodes
            if getattr(n, "running_max", False) and getattr(n, "streaming", None) is None
            and len(n.raxis) > 0]

    def accept(cand) -> bool:
        try:
            schedule_kernel(cand)
        except ScheduleRejected as e:
            accept.reasons[_reason_key(e)] += 1
            if not _iterative_reductions(held, cand.rstep_map):
                accept.whole[_reason_key(e)] += 1
            return False
        return True

    accept.reasons, accept.whole = Counter(), Counter()
    return accept
