"""Exact softmax per schedule: the stabilization rewrite on the *tiled* graph.

The search and the tiling policy see ``exp(x) / sum(exp(x))``: no max, so nothing pins the
softmax axis to one tile. Lowering marks each ``Exp`` whose unknown positive scale provably
cancels with a ``Stabilize`` record (``Exp.stabilize``). ``apply_tiling`` calls
:func:`stabilize` as its last statement when the graph carries such a record, and each
marked ``Exp`` gets the shift its tiling needs, decided from the tier that was actually
built (its ``tmap`` and ``extent``):

    softmax axis whole in the Exp's tile  ->  in-tile:  m = max(s);  P = exp(s - m)
    softmax axis tiled by the tier's loop ->  rolling:  m carried across iterations,
                                              alpha = exp(m_prev - m), every accumulator
                                              that sums the Exp is multiplied by alpha
                                              right before it accumulates
    softmax axis spread over grid blocks  ->  refused (``StabilizeRefused``)

A rolling Exp whose accumulators are too wide for a thread's registers gets the *lazy* form
(``Site.lazy``): the shift is the running max rounded down to a grid, which moves only when
the max crosses a multiple of it, and the rescale is skipped while it does not
(``cleave.ir.rolling.lazy_rolling_nodes``). Same softmax, far fewer multiplies.

A tiling the pass cannot serve is refused with a reason that names the Exp; the candidate
is dropped, not the program. A graph without a marked Exp never reaches this module: it is
returned by ``apply_tiling`` as before, so every non-softmax kernel and every graph built
without the frontend request (paged dense bodies, IR fixtures, hand-written schedules)
generates the text it always did.

The pass works on the finished, locked graph, whose node-keyed dicts may hold keys hashed
by structure. It therefore mutates only inside ``edit_graph`` (unlock, edit by name, lock,
rebuild the dicts).
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from dataclasses import dataclass

from tilelang.carver.roller.node import Edge

from ..core.errors import ScheduleRejected
from ..core.target import current_target
from ..ir.graph import AccumEdge, ComputeOpNode, Graph, Iterator
from ..ir.graph_utils import edit_graph, graph_tiers, tmap_by_name
from ..ir.offsets import TileOffset
from ..ir.ops import NEG_FLT_MAX, Exp, ReduceMax, Sub, wire_input
from ..ir.rolling import STAT_DTYPE, lazy_rolling_nodes, rolling_nodes
from .tiling import rebase_reduction_axis


# The lazy rolling form is used when one thread holds at least this many elements of a
# rescaled accumulator. A thread has at most 255 registers, so such an accumulator lives in
# memory, and rescaling it is a load, a multiply and a store per element on every iteration
# (measured on MLA decode, 64 x 512 over 128 threads: 107 of 282 us). Below that the
# accumulator is in registers, the multiply is nearly free, and the guard and the grid cost
# more than they save (measured at 128, 64 and 32 elements per thread: 6% to 13% slower).
LAZY_MIN_ELEMENTS = 256


class StabilizeRefused(ScheduleRejected):
    """This tiling cannot carry an exact softmax; the candidate is dropped, not the program."""


@dataclass(frozen=True)
class Site:
    """Where one marked Exp ended up in a tiled graph, and what it was given."""

    exp: str
    tier: str  # "main" or the owning GraphNode's name
    case: str  # "in-tile" | "rolling"
    axis: int  # softmax axis in the tier-local tile (index into Iterator.tile_size)
    full: int  # extent of the softmax axis in the flat graph
    block: int  # ... held by one grid block
    step: int  # ... held by one iteration of the tier
    scaled: tuple = ()  # rolling: names of the rescaled accumulators
    lazy: bool = False  # rolling: shift on a grid, rescale only when it moved


def annotated(graph) -> list:
    """Marked Exps of a flat or tiled graph, as ``(tier, owner, exp)``."""
    return [
        (tier, owner, n)
        for tier, owner in graph_tiers(graph)
        for n in tier.nodes
        if type(n) is Exp and n.stabilize is not None
    ]


def sites(tiled) -> list:
    """The ``Site`` of every Exp of a tiled graph that :func:`stabilize` served."""
    return [
        n.stabilized
        for tier, _ in graph_tiers(tiled)
        for n in tier.nodes
        if getattr(n, "stabilized", None) is not None
    ]


def _refused(exp, text) -> StabilizeRefused:
    return StabilizeRefused(f"{exp.name}: {text}")


def _stored_dtype(edge) -> str:
    """The dtype the value ``edge`` reads is stored in (what codegen casts from)."""
    src = edge.accum_node if isinstance(edge, AccumEdge) else edge.src_node
    return src.out_dtype if isinstance(src, ComputeOpNode) else str(src.get_dtype(0))


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


# --------------------------------------------------------------------------- classification


def _lazy(accs, tmaps, threads) -> bool:
    """Whether a thread holds too much of one of the rescaled accumulators ``accs`` for its
    registers (``LAZY_MIN_ELEMENTS``): the rescale is then worth skipping."""
    if not threads or not accs:
        return False
    widest = 1
    for n in accs:
        elements = 1
        for t in tmaps[n.name].tile_size:
            elements *= int(t)
        widest = max(widest, elements)
    return widest >= LAZY_MIN_ELEMENTS * int(threads)


def _locate(tier, owner, exp, threads=None) -> Site:
    """Classify one marked Exp from its tier's own ``tmap`` and ``extent`` (read-only).

    Everything a tiling can be refused for is decided here, before anything is edited.
    ``threads`` is the kernel's thread count, when known: it decides between the eager and
    the lazy rolling form.
    """
    st = exp.stabilize
    where = "main" if owner is None else owner.name
    if len(st.axes) != 1:
        raise _refused(exp, f"softmax over {len(st.axes)} axes; one softmax axis is supported")
    (a,) = st.axes
    dims = tuple(int(s) for s in exp.get_shape())
    nb = exp.n_batch_dims
    if not -(len(dims) - nb) <= a < 0:
        raise _refused(exp, f"softmax axis {a} is not a non-batch dim of the exp {dims}")
    if len(dims) - nb != 2:
        raise _refused(
            exp,
            f"{len(dims) - nb} non-batch dims; the shift is broadcast over exactly one "
            "other axis",
        )
    if _is(exp, tier.accum_nodes):
        raise _refused(exp, f"the exp is itself an accumulator of {where}")
    tmaps = tmap_by_name(tier)
    it = tmaps[exp.name]
    tile = tuple(int(t) for t in it.tile_size)
    # A loop tier carries only the dims its block stores (``Iterator.axes``), so the place
    # of the softmax axis is asked of the tile, not counted from its end.
    flat = len(dims) + a
    axis = it.position(flat, len(dims))
    if axis is None:
        raise _refused(exp, f"softmax axis {a} is not in its tile {tile}")
    full = dims[a]
    if owner is None:
        block = step = tile[axis]
    else:
        block, step = int(it.shape[axis]), tile[axis]
    if block < full:
        raise _refused(
            exp,
            f"the softmax axis is spread over grid blocks ({block} of {full} per block); "
            "a max cannot be carried across blocks",
        )
    if all(t == 1 for i, t in enumerate(tile) if i != axis):
        # m / alpha would be all-unit tiles, which the emitter squeezes to nothing.
        raise _refused(exp, f"its tile {tile} holds a single row")

    if step == block:
        if a != -1 and (owner is not None or any(t == 1 for t in tile[:axis])):
            # An un-looped reduction over a non-last axis is emitted with the flat-graph
            # `dim`; once a unit dim in front of it is squeezed out of the tile that is
            # the wrong axis. A max would have the same defect.
            raise _refused(exp, f"softmax axis {a} is not the last axis and is not looped")
        if st.export is not None:
            if owner is not None:
                raise _refused(
                    exp,
                    f"exported max {st.export!r} would accumulate over a loop that does "
                    "not tile the softmax axis",
                )
            _exported_max(tier, where, exp, flat, carried=False)
        return Site(exp.name, where, "in-tile", axis, full, block, step)

    # rolling: the tier's loop is the loop over the softmax axis
    if step == 1:
        # Unit extents are squeezed out of a tile, so every macro over it loses an index.
        raise _refused(
            exp,
            "one element of the softmax axis per iteration; the tile would lose that axis",
        )
    if st.rolling:
        raise _refused(exp, f"no rolling form: {st.rolling}")
    if not isinstance(tier.extent, int) or tier.extent * step != block:
        raise _refused(
            exp,
            f"{where} iterates {tier.extent} times but the softmax axis is stepped "
            f"{step} of {block}",
        )
    outside = [e.dst_node.name for e in exp.outputs if not _is(e.dst_node, tier.nodes)]
    if outside:
        raise _refused(
            exp, f"read by {outside} outside the loop that tiles its softmax axis"
        )
    want = dict(st.scaled)
    accs = [n for n in tier.accum_nodes if n.name in want]
    missing = sorted(set(want) - {n.name for n in accs})
    if missing:
        raise _refused(
            exp,
            f"{missing} sum(s) the softmax axis but not as accumulator(s) of the loop "
            f"{where} that tiles it",
        )
    for n in tier.accum_nodes:
        if n.name not in want and n.name != st.export and _descends(n, exp, tier):
            raise _refused(
                exp,
                f"accumulator {n.name} depends on the exp but is not one of its "
                "analysed reductions",
            )
    rows = _squeezed(t for i, t in enumerate(tile) if i != axis)
    for n in accs:
        if want[n.name] != 1:
            raise _refused(exp, f"accumulator {n.name} has degree {want[n.name]}")
        if getattr(n, "accum_init", None) is not None:
            raise _refused(
                exp,
                f"accumulator {n.name} starts at {n.accum_init}; a rescaled accumulator "
                "must start at zero",
            )
        factor = next((f for acc, f in tier.accum_scale.items() if acc is n), None)
        if factor is not None:
            raise _refused(
                exp,
                f"accumulator {n.name} is already rescaled by {factor.name}; the rolling "
                "form keeps one factor per accumulator",
            )
        acc_tile = tuple(int(t) for t in tmaps[n.name].tile_size)
        if _squeezed(acc_tile)[: len(rows)] != rows:
            raise _refused(
                exp,
                f"accumulator {n.name} tile {acc_tile} does not lead with the row tile "
                f"{tuple(rows)}",
            )
        if len(_squeezed(acc_tile)) != len(acc_tile):
            # Its buffer is allocated unsqueezed, the rescale indexes the squeezed rank.
            raise _refused(exp, f"accumulator {n.name} has a unit extent in its tile {acc_tile}")
    if st.export is not None:
        _exported_max(tier, where, exp, flat, carried=True)
    # An exported max is the kernel's shift as the reference has it: what such a kernel
    # accumulates must stay in the frame of the max itself.
    lazy = st.export is None and _lazy(accs, tmaps, threads)
    return Site(
        exp.name, where, "rolling", axis, full, block, step, tuple(n.name for n in accs), lazy
    )


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


def _exported_max(tier, where, exp, flat, *, carried: bool):
    """The ``ReduceMax`` a split partial exports as its shift, found by its recorded name
    and verified; never recognised by shape.

    ``flat`` is the softmax axis among the dims of the exp, counted from the front. It is
    compared with the axis the max was built with (``flat_axis``): ``axis_arg`` is the
    ``dim`` the tiler gave the emitted macro, an index into the stored tile, and differs
    from one tiling to the next.
    """
    name = exp.stabilize.export
    m = next((n for n in tier.nodes if n.name == name), None)
    if m is None:
        raise _refused(exp, f"exported max {name!r} is not in {where}, the tier of the exp")
    if type(m) is not ReduceMax:
        raise _refused(exp, f"exported max {name!r} is a {type(m).__name__}, not a ReduceMax")
    own, other = exp.inputs[0], m.inputs[0]
    same = (
        own.src_node is other.src_node
        and own.src_id == other.src_id
        and getattr(own, "accum_node", None) is getattr(other, "accum_node", None)
    )
    if not same:
        raise _refused(exp, f"exported max {name!r} does not read the operand of the exp")
    if m.flat_axis != flat:
        raise _refused(
            exp,
            f"exported max {name!r} reduces axis {m.flat_axis}, the softmax axis is {flat}",
        )
    if m.out_dtype != STAT_DTYPE:
        raise _refused(exp, f"exported max {name!r} is {m.out_dtype}, not {STAT_DTYPE}")
    if carried and not _is(m, tier.accum_nodes):
        raise _refused(exp, f"exported max {name!r} is not an accumulator of {where}")
    return m


# --------------------------------------------------------------------------------- rewrite


def _build_parts(exp_cls, name, dims, nb, a, scores_dtype, out_dtype, rolling, fresh_exp,
                 lazy=False):
    """The nodes one marked Exp gains, wired among themselves, as a small ``Graph``.

        scores (a stand-in placeholder)
        m       = ReduceMax(scores)     {name}_max       float32
        shifted = Sub(scores, m)        {name}_shifted   float32
        exp     = exp_cls(shifted)      {name}           (only when the Exp read storage)
        m_prev, step, alpha             ``ir.rolling.rolling_nodes`` (only when rolling)

    ``lazy`` (rolling only): ``ir.rolling.lazy_rolling_nodes`` instead, with one more node,
    ``{name}_max_grid = g(m)``, which ``shifted`` then subtracts in place of ``m``.
    """
    flat_axis = len(dims) + a
    pos = flat_axis - nb  # the softmax axis among the two non-batch dims
    rows = dims[:flat_axis] + dims[flat_axis + 1 :]
    # Named and typed as lowering types a reduction: it reads the scores as they are
    # stored (a max of stored values is exact in float32).
    m = ReduceMax(
        dims,
        axis=-1 if a == -1 else flat_axis,
        name=f"{name}_max",
        dtype=scores_dtype,
        out_dtype=STAT_DTYPE,
    )
    scores = m.inputs[0].src_node
    shifted = Sub(
        dims,
        broadcast=1 - pos,
        name=f"{name}_shifted",
        dtype=STAT_DTYPE,
        out_dtype=STAT_DTYPE,
        n_batch_dims=nb,
    )
    wire_input(scores, shifted, 0)
    if not (rolling and lazy):
        wire_input(m, shifted, 1)
    nodes = [scores, m, shifted]
    if fresh_exp:
        exp = exp_cls(dims, name=name, dtype=STAT_DTYPE, out_dtype=out_dtype, n_batch_dims=nb)
        wire_input(shifted, exp, 0)
        nodes.append(exp)
    if rolling and lazy:
        stats = lazy_rolling_nodes(m, stem=name, rows=rows, n_batch_dims=nb, exp_cls=exp_cls)
        wire_input(stats[1], shifted, 1)  # s - g(m)
        nodes += stats
    elif rolling:
        nodes += rolling_nodes(m, stem=name, rows=rows, n_batch_dims=nb, exp_cls=exp_cls)
    return Graph(nodes=nodes)


@functools.lru_cache(maxsize=64)
def _part_templates(target, *key):
    """``_build_parts(*key)``, built once per Exp and target.

    Building the five ops costs several times what tiling a candidate does (each one
    derives a prim_func, for the current target), and the accept filter rewrites every raw
    candidate of a graph. The parts depend on the Exp only, never on the tiling, so a
    rewrite clones them instead.
    """
    return _build_parts(*key)


def _parts(*key, cache: bool = True) -> dict:
    """Fresh, unshared copies of the parts for one rewrite, by role."""
    graph = _part_templates(current_target(), *key).clone() if cache else _build_parts(*key)
    name = key[1]
    by_name = {n.name: n for n in graph.nodes[1:]}
    roles = {
        "max": f"{name}_max",
        "shifted": f"{name}_shifted",
        "exp": name,
        "max_prev": f"{name}_max_prev",
        "max_grid": f"{name}_max_grid",
        "max_step": f"{name}_max_step",
        "alpha": f"{name}_alpha",
    }
    return {role: by_name[n] for role, n in roles.items() if n in by_name}


def _read(edge, dst, dst_id: int):
    """Make ``dst`` input ``dst_id`` read what ``edge`` reads: same source, same kind.

    An accumulator of a loop is read through an ``AccumEdge``; a plain ``Edge`` to the
    ``GraphNode`` would not say which accumulator.
    """
    _detach(dst.inputs[dst_id])
    if isinstance(edge, AccumEdge):
        new = AccumEdge(edge.src_node, dst, edge.src_id, dst_id, accum_node=edge.accum_node)
    else:
        new = Edge(edge.src_node, dst, edge.src_id, dst_id)
    edge.src_node._out_edges.append(new)
    dst.set_inputs(dst_id, new)


def _detach(edge):
    """Remove ``edge`` from its source's readers (by identity)."""
    out = edge.src_node._out_edges
    for i, e in enumerate(out):
        if e is edge:
            del out[i]
            return


def _wire(src, dst, dst_id: int):
    """``dst`` input ``dst_id`` <- the compute node ``src``, replacing what it read."""
    _detach(dst.inputs[dst_id])
    edge = Edge(src, dst, 0, dst_id)
    src._out_edges.append(edge)
    dst.set_inputs(dst_id, edge)


class _DroppedAxis:
    """A hand-written offset function without one of its axes (same arity)."""

    def __init__(self, offset, axis):
        self.offset, self.axis = offset, axis

    def __call__(self, *args):
        return tuple(v for i, v in enumerate(self.offset(*args)) if i != self.axis)

    @property
    def __code__(self):
        return self.offset.__code__


def _drop_axis(it: Iterator, axis: int) -> Iterator:
    """``it`` without ``axis``: the tile of ``m`` / ``m_prev`` / ``alpha``."""

    def drop(seq):
        return type(seq)(v for i, v in enumerate(seq) if i != axis)

    offset = it.offset
    if isinstance(offset, TileOffset):
        offset = TileOffset(drop(offset.terms), offset.num_args)
    else:
        offset = _DroppedAxis(offset, axis)
    # The row vector has the exp's dims without the softmax axis: the dims the tile
    # carries are the exp's, renumbered past the one that is gone.
    axes = it.axes
    if axes is not None:
        gone = axes[axis]
        axes = tuple(d - (d > gone) for d in axes if d != gone)
    return Iterator(offset, shape=drop(it.shape), tile_size=drop(it.tile_size), axes=axes)


def _rewrite(tier, owner, exp, site: Site, taken: set, cache: bool = True):
    """Give ``exp`` the shift ``site`` calls for. Only called inside ``edit_graph``."""
    st = exp.stabilize
    (a,) = st.axes
    dims = tuple(int(s) for s in exp.get_shape())
    nb = exp.n_batch_dims
    rolling = site.case == "rolling"
    scores_edge = exp.inputs[0]
    tmaps = tmap_by_name(tier)  # by name: a structure-hashed key is unfindable mid-edit
    exp_it = tmaps[exp.name]
    vec_it = _drop_axis(exp_it, site.axis)
    fresh_exp = exp.dtype != STAT_DTYPE  # it read fp16 storage; now the float32 difference

    names = [f"{exp.name}_shifted"] + ([] if st.export is not None else [f"{exp.name}_max"])
    if rolling:
        names += [f"{exp.name}_max_prev", f"{exp.name}_max_step", f"{exp.name}_alpha"]
    if site.lazy:
        names.append(f"{exp.name}_max_grid")
    clash = sorted(set(names) & taken)
    assert not clash, f"{exp.name}: node name(s) {clash} are already taken in this graph"
    taken.update(names)

    parts = _parts(
        type(exp), exp.name, dims, nb, a, _stored_dtype(scores_edge), exp.out_dtype,
        rolling, fresh_exp, *((True,) if site.lazy else ()), cache=cache,
    )
    shifted = parts["shifted"]
    new_nodes = []

    # --- the max: the one a split partial already exports, else the new node ------------
    if st.export is not None:
        m = next(n for n in tier.nodes if n.name == st.export)
        for edge in list(parts["max"].outputs):  # whatever read the stand-in reads `m`
            _wire(m, edge.dst_node, edge.dst_id)
    else:
        m = parts["max"]
        _read(scores_edge, m, 0)
        # Its emitted `dim` indexes the stored scores, as for every reduction the tiler
        # scheduled (`_normalize_inner_reduction_axes`, `_normalize_main_reduction_axes`).
        rebase_reduction_axis(m, _stored_tile(tier, scores_edge))
        tier.tmap[m] = vec_it
        new_nodes.append(m)

    # --- P = exp(s - m) -----------------------------------------------------------------
    _read(scores_edge, shifted, 0)
    tier.tmap[shifted] = Iterator(exp_it.offset, exp_it.shape, exp_it.tile_size, exp_it.axes)
    new_nodes.append(shifted)
    if fresh_exp:
        # A node whose dtype attribute and prim_func disagree is not left behind: the Exp
        # is replaced by one built for float32 input. Same class and name, so the same
        # identity, buffer name and macro.
        new_exp = parts["exp"]
        for key, value in exp.__dict__.items():
            if key not in new_exp.__dict__ and not key.startswith("_memo_"):
                new_exp.__dict__[key] = value
        _detach(scores_edge)
        for edge in exp._out_edges:
            edge.src_node = new_exp
        new_exp._out_edges, exp._out_edges, exp._in_edges = list(exp._out_edges), [], []
        tier.nodes[next(i for i, n in enumerate(tier.nodes) if n is exp)] = new_exp
        del tier.tmap[exp]
        tier.tmap[new_exp] = exp_it
        exp = new_exp
    else:
        _wire(shifted, exp, 0)

    # --- rolling: carry m, lag it, rescale what the loop has accumulated so far ----------
    if rolling:
        m.accum_init = NEG_FLT_MAX  # this instance only; the class default stays -inf
        stats = [parts["max_prev"], parts["max_step"], parts["alpha"]]
        if site.lazy:
            assert st.export is None and getattr(parts["alpha"], "unit_guard", False)
            stats.insert(1, parts["max_grid"])
        for n in stats:
            tier.tmap[n] = Iterator(vec_it.offset, vec_it.shape, vec_it.tile_size, vec_it.axes)
        new_nodes += stats

    assert sorted(n.name for n in new_nodes) == sorted(names)
    tier.nodes.extend(new_nodes)
    if rolling:
        owner.carry(m)  # appended after the existing accumulators; a no-op for an export
        tier.lag_nodes.append(parts["max_prev"])
        for acc in tier.accum_nodes:
            if acc.name in site.scaled:
                tier.accum_scale[acc] = parts["alpha"]
    # Served: the request is consumed on this (tiled) copy of the node, so running the pass
    # again is a no-op. The flat graph's node -- the one the reference reads -- is another
    # object and keeps its record.
    exp.stabilize, exp.stabilized = None, site


def stabilize(tiled: Graph, cache: bool = True) -> Graph:
    """Give every marked Exp of a tiled graph its exact shift, in place.

    ``StabilizeRefused`` if any of them cannot be served by this tiling; nothing has been
    edited then. A graph without a marked Exp is returned as it came.
    """
    found = annotated(tiled)
    if not found:
        return tiled
    threads = getattr(tiled, "n_threads", None)
    plan = [(tier, owner, exp, _locate(tier, owner, exp, threads)) for tier, owner, exp in found]
    claimed = {}
    for tier, _, exp, site in plan:
        for name in site.scaled:
            other = claimed.setdefault((id(tier), name), exp.name)
            if other != exp.name:
                raise _refused(
                    exp,
                    f"accumulator {name} is also scaled by {other}; the rolling form "
                    "keeps one factor per accumulator",
                )
    taken = {n.name for tier, _ in graph_tiers(tiled) for n in tier.nodes}
    with edit_graph(tiled):
        for tier, owner, exp, site in plan:
            _rewrite(tier, owner, exp, site, taken, cache)
    return tiled


# ------------------------------------------------------------------------------ diagnostics


def overhead_bytes(tiled: Graph) -> int:
    """Bytes of the row vectors the rewrite added to one block: ``m``, plus ``m_prev`` /
    ``step`` / ``alpha`` when rolling (and ``g(m)`` in the lazy form), 4 bytes a row each.

    A diagnostic, not a gate: they are register fragments, and charging them to the
    shared-memory model removed no candidate on any measured operator. The shifted-score
    tile is not counted: the policy's allocator frees a tile once its consumers ran, so in
    ``scores -> shifted -> exp`` at most two score-sized tiles are live with or without the
    extra link.
    """
    total = 0
    tiers = {("main" if owner is None else owner.name): tier for tier, owner in graph_tiers(tiled)}
    for site in sites(tiled):
        tile = tmap_by_name(tiers[site.tier])[site.exp].tile_size
        rows = 1
        for i, t in enumerate(tile):
            rows *= 1 if i == site.axis else int(t)
        total += 4 * rows * ((5 if site.lazy else 4) if site.case == "rolling" else 1)
    return total


# ---------------------------------------------------------------------------- accept filter


def _reason_key(error) -> str:
    """A refusal's first line with bare numbers folded, so one reason counts once whatever
    the tile sizes were. Digits that belong to a name (``exp_5``) are kept."""
    line = (str(error).splitlines() or [""])[0]
    return re.sub(r"(?<![\w.\-])\d+(?![\w.])", "N", line)[:200]


def stabilize_accept(flat_graph):
    """``search_tiling(accept=...)`` filter for a graph with marked Exps, else ``None``.

    A candidate is kept iff ``apply_tiling`` builds it, rewrite included: the filter is the
    build, so the two cannot disagree. It runs before the top-k cut, because once the max is
    gone from the searched graph the policy ranks tilings the rewrite has to refuse above
    the ones it can serve, and they would fill the top-k.

    Only ``ScheduleRejected`` (``StabilizeRefused`` and the tiler's own typed refusals)
    drops a candidate; ``accept.reasons`` counts them by reason. Any other exception is a
    bug and propagates.
    """
    if not annotated(flat_graph):
        return None
    from .tiling import apply_tiling

    def accept(cand) -> bool:
        try:
            apply_tiling(cand)
        except ScheduleRejected as e:
            accept.reasons[_reason_key(e)] += 1
            return False
        return True

    accept.reasons = Counter()
    return accept
