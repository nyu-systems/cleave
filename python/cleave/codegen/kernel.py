from typing import List, Sequence

from ..ir.graph import (
    AccumEdge,
    ComputeOpNode,
    Graph,
    GraphNode,
    Node,
    OutputNode,
    PlaceHolderNode,
)
from ..ir.ops.base import const_literal
from ..ir.ops.reduction import Rescale
from ..ir.graph_utils import find_topo_sort
from ..schedule.policy.hint import INTER
from .tensor_codegen import (
    INTER_BLOCK,
    gen_atomic_add,
    gen_copy,
    gen_tensor_allocation_str,
    gen_tensor_type,
)
from .buffer import BASE_INDENT, CodeBuffer
from .macros import qualify_name
from .operands import HOISTED_SUFFIX, emit_alloc, hoisted_nodes, kept_extents, stored_extents, tc_layout


def _bind_node(graph, arg, is_main):
    if isinstance(arg, (PlaceHolderNode, OutputNode)) and not is_main:
        return graph.binding[arg.name]
    return arg


def _tile_slices(offset, tile_size):
    """``o : o + ts`` per axis, and the unit axes to squeeze (all but one if every one is)."""
    slices = [f"{o} : {o} + {ts}" for o, ts in zip(offset, tile_size)]
    squeeze = [i for i, ts in enumerate(tile_size) if ts == 1]
    if len(squeeze) == len(tile_size):
        squeeze = squeeze[:-1]
    return slices, squeeze


def _call(func: str, args: Sequence[str]) -> str:
    return f"{func}({', '.join(args)})\n"


def _iter_sargs(graph):
    return graph.sargs if graph.iter_sargs is None else graph.iter_sargs


# An inter output is summed across CTAs in a global buffer of this dtype; the runtime
# zeroes it before every launch and casts it to the output's dtype afterwards.
INTER_DTYPE = "float32"
# ``INTER_LAYOUTS`` value of a buffer stored in 8x8 blocks (``inter_blocked_width``).
BLOCKED_LAYOUT = "blocked8x8"


def inter_nodes(graph) -> set:
    """The nodes ``graph.accumulation`` sums across CTAs: OutputNodes of the main graph,
    accum nodes of a loop graph. Empty for every graph without accumulation contexts."""
    return {n for n, context in graph.accumulation.items() if context == INTER}


def inter_blocked_width(tmap):
    """The last extent N when an inter accum node's buffer is stored 8x8-blocked, else None.

    Blocked (see ``tensor_codegen.gen_atomic_add``) when every iteration adds whole rows
    (the tile spans the block's last dim), 8-row aligned (tile rows a multiple of 8 that
    divides the block's rows), N a multiple of 8, all static. The CTA block of an inter
    output spans its whole ``[M, N]`` planes, so the layout is the buffer's own and the
    runtime undoes it while casting (``INTER_LAYOUTS``).
    """
    try:
        shape = [int(d) for d in tmap.shape]
        tile = [int(d) for d in tmap.tile_size]
    except TypeError:
        return None
    if len(shape) < 2 or len(kept_extents(tile)) < 2 or kept_extents(tile)[-2:] != tile[-2:]:
        return None
    (rows, width), (tile_rows, tile_width) = shape[-2:], tile[-2:]
    if (
        tile_width != width
        or width % INTER_BLOCK
        or tile_rows % INTER_BLOCK
        or rows % tile_rows
    ):
        return None
    return width


def inter_output_layout(out) -> str | None:
    """``INTER_LAYOUTS`` entry of an inter OutputNode: its loop accumulator's layout."""
    edge = out.inputs[0]
    if not isinstance(edge, AccumEdge):
        return None
    tmap = edge.src_node.graph.tmap[edge.accum_node]
    return BLOCKED_LAYOUT if inter_blocked_width(tmap) is not None else None


def _inter_dst(graph, node) -> str:
    """The global view an inter accum node's per-iteration tile is atomically added to."""
    return f"{graph.tinfo_map[node][0]}_global"


def _arg_dtype(graph, arg, is_main) -> str:
    if is_main and arg in inter_nodes(graph):
        return INTER_DTYPE
    return arg.get_dtype(0)


def emission_order(graph) -> List[Node]:
    """The order one tier's nodes are emitted in.

    Dataflow order (``find_topo_sort``) for a graph that declares no lag node and no
    rescaled accumulator: the same list, so the same text. With them, two orderings no
    edge expresses are added. A lag node is emitted before the accumulator it snapshots
    (it reads the value the *previous* iteration left, so its own input edge is not a
    dependency), and a rescale factor is emitted before the accumulator it multiplies.
    """
    if not graph.lag_nodes and not graph.accum_scale:
        return _work_behind_loops(find_topo_sort(graph.nodes))
    lag_ids = {id(n) for n in graph.lag_nodes}
    lags_of = {}
    for lag in graph.lag_nodes:
        lags_of.setdefault(id(lag.inputs[0].src_node), []).append(lag)
    factor_of = {id(acc): factor for acc, factor in graph.accum_scale.items()}

    def deps(node):
        if id(node) in lag_ids:
            return []
        found = [e.src_node for e in node.inputs] + lags_of.get(id(node), [])
        factor = factor_of.get(id(node))
        return found if factor is None else found + [factor]

    order, state = [], {}  # state: 1 = on the stack, 2 = emitted

    def visit(node):
        seen = state.get(id(node))
        if seen == 2:
            return
        assert seen is None, (
            f"{node.name} would have to be emitted before itself: a lag node or a rescale "
            "factor depends on the accumulator it is ordered in front of."
        )
        state[id(node)] = 1
        for dep in deps(node):
            visit(dep)
        state[id(node)] = 2
        order.append(node)

    for node in graph.nodes:
        visit(node)
    return _work_behind_loops(order)


def _work_behind_loops(order: List[Node]) -> List[Node]:
    """``order`` with every compute node that no loop reads moved behind the last loop.

    Dataflow leaves a node that neither feeds a loop nor reads one free to be emitted on
    either side of it: the value sum of a split combine whose step is the whole split
    axis, next to the loop of the row sum. TileLang (0.1.13) cannot take it in front. It
    lowers a tier with a pipelined loop by splitting the block's threads at the loop into
    a group that loads and a group that computes, and runs whatever stands in front of
    the loop on both, without the thread offset it gives the computing group behind the
    split (in the lowered CUDA the fragment is filled at ``threadIdx.x * 8`` in front and
    stored at ``threadIdx.x * 8 - 1024`` behind). That kernel does not return: it spins
    on the GPU until it is killed. Behind the loop the same statement belongs to the
    computing group alone. A tier without such a node keeps its order, and so its text.
    """
    loops = [i for i, node in enumerate(order) if isinstance(node, GraphNode)]
    if not loops:
        return order
    read_by_a_loop, stack = set(), [order[i] for i in loops]
    while stack:
        for edge in stack.pop().inputs:
            if id(edge.src_node) not in read_by_a_loop:
                read_by_a_loop.add(id(edge.src_node))
                stack.append(edge.src_node)
    moved = set()
    for node in order[: loops[-1]]:
        if isinstance(node, ComputeOpNode) and id(node) not in read_by_a_loop:
            moved.add(id(node))
        elif isinstance(node, OutputNode) and id(node.inputs[0].src_node) in moved:
            moved.add(id(node))  # the store of a moved node goes with it
    if not moved:
        return order
    front = [node for node in order[: loops[-1] + 1] if id(node) not in moved]
    behind = [node for node in order[: loops[-1]] if id(node) in moved]
    return front + behind + order[loops[-1] + 1 :]


class KernelEmitter:
    # ---------------------------------------------------------------- kernel scope
    def _emit_prologue(self, graph, it, name_prefix, arg_name_map=None) -> str:
        """Allocations and scan calls for ``graph.prologue``, at kernel scope, so ``smap``
        and ``block_guard`` can read values the device just computed."""
        buf = CodeBuffer()
        seen: set[str] = set()
        for scan in getattr(graph, "prologue", ()):
            assert (
                scan.buffer not in seen
            ), f"Two prologue scans write the buffer {scan.buffer!r}."
            seen.add(scan.buffer)
            assert scan.size >= scan.op.min_size, (
                f"{type(scan.op).__name__} indexes at least {scan.op.min_size} "
                f"elements, but {scan.buffer} is sized {scan.size}."
            )
            buf.raw(
                gen_tensor_allocation_str(
                    scan.buffer, (scan.size,), scan.dtype, scope=scan.scope
                )
            )
            # Inputs are named as this kernel sees them (split-k renames partials).
            operands = [
                (arg_name_map.get(t, t.name) if arg_name_map else t.name)
                for t in scan.inputs
            ]
            sargs = list(scan.scalar_args(*it)) if scan.scalar_args is not None else []
            func = self.emit_op_node(
                scan.op,
                mem_scope="g" * len(operands) + scan.scope,
                clear=False,
                name_prefix=name_prefix,
            )
            buf.raw(_call(func, operands + sargs + [scan.buffer]))
        return str(buf)

    def _emit_else_nodes(self, graph, it, name_prefix, arg_name_map=None) -> str:
        """Macro calls for ``graph.else_nodes``: scalar-only stores into an output tile."""
        buf = CodeBuffer()
        for node in graph.else_nodes:
            tmap = graph.tmap[node]
            sl, squeeze = _tile_slices(tmap.offset(*it), tmap.tile_size)
            dst_buf = graph.tinfo_map[node][0]
            # The parameter name (split-k renames it) is found through the output it writes.
            if arg_name_map:
                for arg, name in arg_name_map.items():
                    if graph.tinfo_map.get(arg, (None,))[0] == dst_buf:
                        dst_buf = name
                        break
            dst = f"_S({dst_buf}[{', '.join(sl)}], squeeze_dim={squeeze})"
            func = self.emit_op_node(
                node,
                mem_scope=graph.tinfo_map[node][-1],
                clear=False,
                name_prefix=name_prefix,
            )
            # The body's own extents must match its tile, or part of it is skipped or overrun.
            tile_elems = 1
            for extent in tmap.tile_size:
                tile_elems *= extent if isinstance(extent, int) else 1
            assert node.min_size == tile_elems, (
                f"else node {node.name!r} writes {node.min_size} elements but its tile "
                f"{tuple(tmap.tile_size)} holds {tile_elems}."
            )
            kept = len(kept_extents(tmap.tile_size))
            if hasattr(node, "dst_rank"):
                assert node.dst_rank() == kept, (
                    f"else node {node.name!r} indexes a rank-{node.dst_rank()} "
                    f"destination, but its tile reduces to rank {kept}."
                )
            sargs = [graph.smap[s](*it) for s in getattr(node, "scalar_args", ())]
            buf.raw(_call(func, sargs + [dst]))
        return str(buf)

    def _emit_loop_header(self, graph, is_main, buf: CodeBuffer) -> list[str]:
        """``with T.Kernel`` (main) or the tier's ``for`` loop; returns the index names."""
        if is_main:
            assert hasattr(graph, "n_threads"), "Main graph must define n_threads."
            assert isinstance(
                graph.extent, (int, tuple)
            ), "Main graphs only support int or tuple extent."
            if isinstance(graph.extent, tuple):
                assert (
                    1 <= len(graph.extent) <= 3
                ), "Only supports up to a 3D grid for now."
                grid_str, n_it = ", ".join(map(str, graph.extent)), len(graph.extent)
            else:
                grid_str, n_it = f"{graph.extent}", 1
            it = [f"i{d}" for d in range(n_it)]
            buf.line(
                f"with T.Kernel({grid_str}, threads=({graph.n_threads})) as ({', '.join(it)}):"
            )
            rast = (
                self.apply_rasterization
                if graph.rasterization is None
                else graph.rasterization
            )
            if rast > 1:
                with buf.indented():
                    buf.line(f"T.use_swizzle({rast})")
            return it
        assert not hasattr(
            graph, "n_threads"
        ), "Inner graphs must not define n_threads."
        # Kernel-scope buffers and grid-slot guards have no meaning one tier down.
        assert (
            not graph.prologue and graph.block_guard is None
        ), "prologue/block_guard are main-graph only."
        assert isinstance(
            graph.extent, int
        ), "Iterative graphs only support int extent."
        trip = graph.loop_trip if graph.loop_trip is not None else f"{graph.extent}"
        if graph.loop_kind == "serial":
            buf.line(f"for i0 in T.serial({trip}):")
        else:
            stages = (
                graph.loop_stages
                if graph.loop_stages is not None
                else min(3, graph.extent)
            )
            buf.line(f"for i0 in T.Pipelined({trip}, num_stages={stages}):")
        return ["i0"]

    def _bind_call_args(
        self, graph, it, is_main, name_prefix, arg_name_map, dep, kwargs
    ):
        """One ``*_iter`` keyword per graph argument: a slice of the caller's tensor, or a
        whole-tile buffer filled here (copy or producer) when the tile is the full shape.
        """
        region_args = set(graph.region_args)
        for arg in graph.args:
            if getattr(arg, "index_only", False):
                continue
            bind_node = _bind_node(graph, arg, is_main)
            assert bind_node in graph.tmap, f"Argument {bind_node} not in tensor map."
            tmap = graph.tmap[bind_node]
            offset = tmap.offset(*it)
            arg_name = arg_name_map.get(arg, arg.name) if arg_name_map else arg.name
            if tmap.shape == tmap.tile_size and not is_main:
                assert all(
                    [i == "0" for i in offset]
                ), f"Expect all zero offset, but got {offset=}"
                tname, scope = graph.tinfo_map[bind_node]
                src_name, src_scope = graph.tinfo_map[arg]
                same_level = scope == src_scope
                if not same_level:
                    emit_alloc(
                        dep,
                        graph,
                        bind_node,
                        kept_extents(tmap.shape),
                        arg.get_dtype(0),
                    )
                producer = getattr(bind_node, "producer_op", None)
                if producer is not None:
                    dep.block(
                        self._emit_producer_call(
                            graph, producer, tname, scope, name_prefix
                        )
                    )
                else:
                    dep.block(
                        gen_copy(
                            src=src_name,
                            dst=tname,
                            same_level=same_level,
                            is_tc_layout=tc_layout(graph, bind_node),
                            movement=(src_scope, scope),
                            tile_shape=tmap.shape,
                            dtype=arg.get_dtype(0),
                        )
                    )
                kwargs.append(f"{tname}={tname}")
                continue
            s, squeeze_dims = _tile_slices(offset, tmap.tile_size)
            if bind_node in region_args or arg in region_args:
                # A BufferRegion drops a unit rank with a scalar subscript, keeping
                # tilelang's own view of the buffer so it can predicate/bulk-copy.
                subs = [
                    r if i not in squeeze_dims else o
                    for i, (o, r) in enumerate(zip(offset, s))
                ]
                kwargs.append(f"{arg.name}={arg_name}[{', '.join(subs)}]")
            else:
                kwargs.append(
                    f"{arg.name}=_S({arg_name}[{', '.join(s)}], squeeze_dim={squeeze_dims})"
                )

    def _bind_call_scalars(self, graph, it, kwargs):
        shifted, shared_shifted = self._prefetch_plan(graph)
        for sname, _ in _iter_sargs(graph):
            resolve = graph.smap.get(sname)
            assert resolve is not None, f"Scalar argument {sname} has no smap entry."
            if sname not in shifted or sname in shared_shifted:
                kwargs.append(f"{sname}={resolve(*it)}")
            if sname in shifted:
                ahead = [f"({v} + 1)" if i == 0 else v for i, v in enumerate(it)]
                kwargs.append(f"{sname}_next={resolve(*ahead)}")

    def _bind_hoisted_buffers(self, graph, name_prefix, dep, kwargs):
        """Window and prefetch buffers live outside the loop and are threaded through it."""
        for node in graph.window_nodes:
            tname, _ = graph.tinfo_map[node]
            emit_alloc(
                dep,
                graph,
                node,
                kept_extents(graph.tmap[node].tile_size),
                node.out_dtype,
            )
            kwargs.append(f"{tname}={tname}")
        # A prefetch fragment is seeded with iteration 0's tile, except off a window, which
        # does not exist out here; the body seeds those when it fills the window.
        for node in graph.prefetch_nodes:
            tname = self._prefetch_buffer_name(graph, node)
            dep.block(
                gen_tensor_allocation_str(
                    tname,
                    kept_extents(graph.tmap[node].tile_size),
                    node.out_dtype,
                    scope="r",
                )
            )
            if not any(e.src_node in graph.window_nodes for e in node.inputs):
                seed = [here for _, here, _ in self._prefetch_scalars(graph, node, "0")]
                dep.block(self._emit_prefetch_read(graph, node, seed, name_prefix))
            kwargs.append(f"{tname}={tname}")
        # Loop-invariant values: filled on the first iteration, read by every iteration.
        _, kept = hoisted_nodes(graph)
        for node in kept:
            name = graph.tinfo_map[node][0] + HOISTED_SUFFIX
            dep.block(
                gen_tensor_allocation_str(
                    name,
                    kept_extents(graph.tmap[node].tile_size),
                    node.out_dtype,
                    scope="s",
                    is_tc_layout=True,
                )
            )
            kwargs.append(f"{name}={name}")
        if kept:
            kwargs.append("i0=i0")

    def _build_graph_call(
        self,
        graph: Graph,
        name: str,
        is_main: bool = True,
        name_prefix: str = "",
        arg_name_map: dict[Node, str] | None = None,
    ) -> tuple[str, str]:
        name_prefix = name_prefix or ""
        dep, body = CodeBuffer(BASE_INDENT), CodeBuffer(BASE_INDENT)
        it = self._emit_loop_header(graph, is_main, body)

        kwargs: list[str] = []
        self._bind_call_args(graph, it, is_main, name_prefix, arg_name_map, dep, kwargs)
        self._bind_call_scalars(graph, it, kwargs)
        self._bind_hoisted_buffers(graph, name_prefix, dep, kwargs)
        if not is_main:
            inter = inter_nodes(graph)
            for n in graph.accum_nodes:
                if n not in inter:
                    kwargs.append(f"{graph.tinfo_map[n][0]}={graph.tinfo_map[n][0]}")
                    continue
                # This iteration's tile of the CTA's block: the loop counter's offset.
                tmap, dst = graph.tmap[n], _inter_dst(graph, n)
                s, squeeze_dims = _tile_slices(tmap.offset(*it), tmap.tile_size)
                kwargs.append(f"{dst}=_S({dst}[{', '.join(s)}], squeeze_dim={squeeze_dims})")
        call = f"{qualify_name(name, name_prefix)}_iter({''.join(k + ', ' for k in kwargs)})\n"

        with body.indented():
            if is_main:
                body.block(self._emit_prologue(graph, it, name_prefix, arg_name_map))
                if graph.block_guard is not None:
                    body.line(f"if {graph.block_guard(*it)}:")
                    body.level += 1
                if graph.work_guard is not None:
                    body.line(f"if {graph.work_guard(*it)}:")
                    body.level += 1
            body.block(call)
            if is_main and graph.else_nodes:
                assert (
                    graph.work_guard is not None
                ), "else_nodes need a work_guard to be the alternative to."
                body.level -= 1
                body.line("else:")
                body.level += 1
                body.block(self._emit_else_nodes(graph, it, name_prefix, arg_name_map))
        return str(dep), str(body)

    # ---------------------------------------------------------------- *_iter macro
    def _iter_signature(self, graph, qualified_name, args, is_main_iter) -> str:
        params = []
        for arg in args:
            if getattr(arg, "index_only", False):
                continue
            bind_node = _bind_node(graph, arg, is_main_iter)
            tmap = graph.tmap[bind_node]
            # Unit extents are squeezed, but an all-unit tile stays 1-D, not a scalar.
            tile_size = kept_extents(tmap.tile_size) or [1]
            # A full-shape tile was copied in by the caller under the bound node's name.
            arg_name = bind_node.name if tmap.shape == tmap.tile_size else arg.name
            params.append(
                f"{arg_name}: {gen_tensor_type(tile_size, _arg_dtype(graph, arg, is_main_iter))}"
            )
        shifted, shared_shifted = self._prefetch_plan(graph)
        for sname, sdtype in _iter_sargs(graph):
            if sname not in shifted or sname in shared_shifted:
                params.append(f"{sname}: T.{sdtype}")
            if sname in shifted:
                params.append(f"{sname}_next: T.{sdtype}")
        for node in graph.prefetch_nodes:
            ty = gen_tensor_type(
                kept_extents(graph.tmap[node].tile_size), node.out_dtype
            )
            params.append(f"{self._prefetch_buffer_name(graph, node)}: {ty}")
        for node in graph.window_nodes:
            ty = gen_tensor_type(
                kept_extents(graph.tmap[node].tile_size), node.out_dtype
            )
            params.append(f"{graph.tinfo_map[node][0]}: {ty}")
        _, kept = hoisted_nodes(graph)
        for node in kept:
            ty = gen_tensor_type(kept_extents(graph.tmap[node].tile_size), node.out_dtype)
            params.append(f"{graph.tinfo_map[node][0]}{HOISTED_SUFFIX}: {ty}")
        if kept:
            params.append("i0: T.int32")
        if not is_main_iter:
            params += self._accum_params(graph)
        return f"\n@T.macro\ndef {qualified_name}_iter({''.join(p + ', ' for p in params)}):\n"

    @staticmethod
    def _accum_params(graph, whole: bool = False) -> list[str]:
        """One parameter per accum node: its register accumulator, or for an inter node
        the global view it is added into (the CTA's block of the output when ``whole``,
        in the loop's wrapper; this iteration's tile otherwise, in the ``*_iter`` macro).
        """
        inter = inter_nodes(graph)
        params = []
        for n in graph.accum_nodes:
            tmap = graph.tmap[n]
            if n in inter:
                extents = kept_extents(tmap.shape if whole else tmap.tile_size) or [1]
                ty = gen_tensor_type(extents, INTER_DTYPE)
                params.append(f"{_inter_dst(graph, n)}: {ty}")
            else:
                ty = gen_tensor_type(tmap.tile_size, n.out_dtype)
                params.append(f"{graph.tinfo_map[n][0]}: {ty}")
        return params

    def _emit_window_block(self, graph, name_prefix, buf: CodeBuffer):
        """Window fills lead the body; a prefetch off a window is re-seeded from it."""
        if not graph.window_nodes:
            return
        assert graph.window_guard is not None, (
            "window_nodes without a window_guard would refill every iteration, which "
            "is what hoisting the buffer is meant to avoid."
        )
        buf.line(f"if {graph.window_guard}:")
        with buf.indented():
            for node in graph.window_nodes:
                buf.block(self._emit_window_fill(graph, node, name_prefix))
            for node in graph.prefetch_nodes:
                if any(e.src_node in graph.window_nodes for e in node.inputs):
                    scalars = list(getattr(node, "scalar_args", ()))
                    buf.block(
                        self._emit_prefetch_read(graph, node, scalars, name_prefix)
                    )

    def _emit_prefetch_block(self, graph, name_prefix, buf: CodeBuffer):
        """Hand every prefetched tile back, then issue every look-ahead read, all before
        this tile's first consumer so the loads overlap the math below."""
        for node in graph.prefetch_nodes:
            assert node not in graph.accum_nodes, (
                f"{node.name} accumulates across iterations, so its buffer cannot be "
                "refilled from a prefetch."
            )
            tname, scope = graph.tinfo_map[node]
            shape = kept_extents(graph.tmap[node].tile_size)
            emit_alloc(buf, graph, node, shape, node.out_dtype)
            buf.block(
                gen_copy(
                    src=self._prefetch_buffer_name(graph, node),
                    dst=tname,
                    same_level=False,
                    movement=("r", scope),
                    tile_shape=shape,
                    dtype=node.out_dtype,
                )
            )
        if not graph.prefetch_nodes:
            return
        if graph.prefetch_guard is not None:
            buf.line(f"if {graph.prefetch_guard}:")
        with buf.indented(int(graph.prefetch_guard is not None)):
            for node in graph.prefetch_nodes:
                scalars = [
                    sname if ahead is None else f"{sname}_next"
                    for sname, _, ahead in self._prefetch_scalars(graph, node, "i0")
                ]
                buf.block(self._emit_prefetch_read(graph, node, scalars, name_prefix))

    def _emit_hoisted_block(self, graph, name_prefix, buf: CodeBuffer) -> set:
        """Loop-invariant nodes, evaluated on the first iteration only; the values later
        iterations read are copied into the buffers the loop's caller allocated.

        On the first iteration rather than before the loop: TileLang 0.1.13 mis-handles
        fragment math in front of a warp-specialized ``T.Pipelined`` loop (wrong results
        at 128 threads, "no available layout found" at 256).
        """
        nodes, kept = hoisted_nodes(graph)
        if not nodes:
            return set()
        buf.line("if i0 == 0:")
        with buf.indented():
            for node in nodes:
                self._emit_compute(graph, node, buf, False, name_prefix)
            for node in kept:
                tname = graph.tinfo_map[node][0]
                buf.block(
                    gen_copy(src=tname, dst=tname + HOISTED_SUFFIX, same_level=False)
                )
        return set(nodes)

    def _emit_placeholder(self, graph, node, buf, is_main_iter, name_prefix):
        if is_main_iter:
            return
        tmap = graph.tmap[node]
        tname, scope = graph.tinfo_map[node]
        src_name, src_scope = graph.tinfo_map[graph.binding[node.name]]
        # A full-shape tile was already loaded by the caller.
        if tmap.shape == tmap.tile_size:
            return
        same_level = scope == src_scope
        if not same_level:
            emit_alloc(
                buf, graph, node, kept_extents(tmap.tile_size), node.get_dtype(0)
            )
        buf.block(
            gen_copy(
                src=src_name,
                dst=tname,
                same_level=same_level,
                is_tc_layout=tc_layout(graph, node),
                movement=(src_scope, scope),
                tile_shape=tmap.tile_size,
                dtype=node.get_dtype(0),
            )
        )
        if (src_scope, scope) == ("g", "r") and tc_layout(graph, node):
            # gen_copy took the slice through a shared buffer in the tensor-core layout
            self._loaded_through[(id(graph), tname)] = src_name + "_intermediate"

    def _emit_compute(self, graph, node, buf, is_main_iter, name_prefix):
        tname, scope = graph.tinfo_map[node]
        writes_to_arg = tname in {
            graph.tinfo_map[a][0] for a in graph.args if a in graph.tinfo_map
        }
        # An inter accum node is a fresh partial every iteration, added to global memory
        # below: it is allocated and cleared here, in the accumulation dtype.
        inter = node in inter_nodes(graph)
        accumulates = node in graph.accum_nodes and not inter
        # Only non-accumulating nodes are allocated inside the loop.
        if not accumulates and not writes_to_arg:
            emit_alloc(
                buf,
                graph,
                node,
                stored_extents(graph.tmap[node].tile_size),
                INTER_DTYPE if inter else node.out_dtype,
            )
        staged = set(getattr(node, "stage_fragment_operands_in_shared", ()))
        call_args, mem_scope = [], ""
        for input_idx, edge in enumerate(node.inputs):
            operand, operand_scope = self._resolve_operand(
                buf, graph, node, input_idx, edge, staged
            )
            call_args.append(operand)
            mem_scope += operand_scope
        call_args += list(getattr(node, "scalar_args", ())) + [tname]
        func_name = self.emit_op_node(
            node,
            mem_scope=mem_scope + scope,
            clear=not accumulates,
            name_prefix=name_prefix,
        )
        if graph.accum_scale and node in graph.accum_scale:
            buf.block(self._emit_rescale(graph, node, name_prefix))
        buf.block(_call(func_name, call_args))
        if inter:
            tile = kept_extents(graph.tmap[node].tile_size)
            width = inter_blocked_width(graph.tmap[node])
            buf.block(gen_atomic_add(tname, _inter_dst(graph, node), width, len(tile)))

    def _emit_rescale(self, graph, node, name_prefix) -> str:
        """``acc *= factor`` on the accumulator's own buffer, right before it accumulates."""
        factor = graph.accum_scale[node]
        tname, scope = graph.tinfo_map[node]
        fname, fscope = graph.tinfo_map[factor]
        op = Rescale(
            rank=len(kept_extents(graph.tmap[node].tile_size)),
            lead=len(kept_extents(graph.tmap[factor].tile_size)),
        )
        func_name = self.emit_op_node(
            op, mem_scope=fscope + scope, clear=False, name_prefix=name_prefix
        )
        return _call(func_name, [fname, tname])

    def _emit_subgraph(self, graph, node, buf, is_main_iter, name_prefix):
        sub = node.graph
        self.gen_graph(sub, node.name, is_main=False, name_prefix=name_prefix)
        inter = inter_nodes(sub)
        for accum_node in sub.accum_nodes:
            if accum_node in inter:
                continue  # added straight into the output's global buffer
            assert (
                accum_node in sub.tmap
            ), f"Accumulation node {accum_node} not in tensor map."
            tname, scope = sub.tinfo_map[accum_node]
            buf.block(
                gen_tensor_allocation_str(
                    tname, sub.tmap[accum_node].tile_size, accum_node.out_dtype, scope
                )
            )
            init = getattr(accum_node, "accum_init", None)
            buf.block(f"T.clear({tname})\n" if init is None
                      else f"T.fill({tname}, {const_literal(init)})\n")
        call_args = []
        for edge in node.inputs:
            if isinstance(edge.src_node, GraphNode):
                assert isinstance(
                    edge, AccumEdge
                ), f"Expect accumulation edge, but got {edge=}"
                call_args.append(edge.src_node.graph.tinfo_map[edge.accum_node][0])
            else:
                call_args.append(graph.tinfo_map[edge.src_node][0])
        call_args += [sname for sname, _ in sub.sargs]
        call_args += [
            self._inter_output_name(graph, node, n) if n in inter else sub.tinfo_map[n][0]
            for n in sub.accum_nodes
        ]
        buf.block(_call(qualify_name(node.name, name_prefix), call_args))

    @staticmethod
    def _inter_output_name(graph, gnode, accum_node) -> str:
        """The (global) output buffer of ``graph`` that ``accum_node`` of loop ``gnode``
        accumulates into."""
        (name,) = [
            graph.tinfo_map[out][0]
            for out in inter_nodes(graph)
            if isinstance(out.inputs[0], AccumEdge)
            and out.inputs[0].src_node is gnode
            and out.inputs[0].accum_node is accum_node
        ]
        return name

    def _emit_output(self, graph, node, buf, is_main_iter, name_prefix):
        assert (
            len(node.inputs) == 1
        ), f"output node should have only one input, but got {node.inputs=}"
        edge = node.inputs[0]
        if node in inter_nodes(graph):
            return  # the loop already added every partial into this buffer
        if isinstance(edge.src_node, GraphNode):
            src_node, _, _ = edge.src_node._resolve_output_binding(edge.src_id)
            src_name, src_scope = edge.src_node.graph.tinfo_map[src_node]
        else:
            src_name, src_scope = graph.tinfo_map[edge.src_node]
        dst_name, dst_scope = graph.tinfo_map[node]
        # TODO: may be redundant when a subgraph output is moved to global/shared memory.
        if src_scope != dst_scope:
            buf.block(
                gen_copy(
                    src=src_name,
                    dst=dst_name,
                    same_level=False,
                    is_tc_layout=tc_layout(graph, node),
                    movement=(src_scope, dst_scope),
                    tile_shape=[i for i in graph.tmap[node].tile_size if i != 1],
                    dtype=node.get_dtype(0),
                )
            )

    # Checked in order, so a subclass is dispatched like its first listed base.
    _NODE_EMITTERS = (
        (PlaceHolderNode, "_emit_placeholder"),
        (ComputeOpNode, "_emit_compute"),
        (GraphNode, "_emit_subgraph"),
        (OutputNode, "_emit_output"),
    )

    def _gen_graph_iter(
        self,
        graph: Graph,
        name: str,
        args: List[Node],
        is_main_iter: bool = False,
        name_prefix: str = "",
    ):
        buf = CodeBuffer(BASE_INDENT)
        buf.raw(
            self._iter_signature(
                graph, qualify_name(name, name_prefix), args, is_main_iter
            )
        )
        self._emit_window_block(graph, name_prefix, buf)
        self._emit_prefetch_block(graph, name_prefix, buf)
        hoisted = self._emit_hoisted_block(graph, name_prefix, buf)
        for node in emission_order(graph):
            if node in graph.prefetch_nodes or node in graph.window_nodes:
                continue
            if node in hoisted:
                continue
            emitter = next(
                (attr for kind, attr in self._NODE_EMITTERS if isinstance(node, kind)),
                None,
            )
            if emitter is None:
                raise NotImplementedError(f"Unknown node type: {node}")
            getattr(self, emitter)(graph, node, buf, is_main_iter, name_prefix)
        buf.raw("\n")
        self.emit(str(buf))

    def gen_graph(
        self,
        graph: Graph,
        name: str,
        is_main: bool = True,
        name_prefix: str = "",
        emit_wrapper: bool = True,
    ):
        self._gen_graph_iter(
            graph, f"{name}", graph.args, is_main_iter=is_main, name_prefix=name_prefix
        )
        if not emit_wrapper:
            return
        self.emit("@T.prim_func\n" if is_main else "@T.macro\n")
        self.emit(f"def {qualify_name(name, name_prefix)}(")
        self.emit("".join(p + ", " for p in self._wrapper_params(graph, is_main)))
        self.emit("):\n")
        dep_code, body_code = self._build_graph_call(
            graph, name, is_main=is_main, name_prefix=name_prefix
        )
        self.emit(dep_code), self.emit(body_code)

    def _wrapper_params(self, graph, is_main) -> list[str]:
        # Declared scalar inputs sit between the tensor inputs and the outputs, matching
        # the operator's own parameter order.
        scalar_params = (
            [f"{n}: T.{t}" for n, t in graph.scalar_params] if is_main else []
        )
        first_output = len(graph.args)
        if scalar_params:
            last_input = max(
                (i for i, a in enumerate(graph.args) if isinstance(a, PlaceHolderNode)),
                default=-1,
            )
            first_output = next(
                (i for i, a in enumerate(graph.args) if isinstance(a, OutputNode)),
                len(graph.args),
            )
            assert last_input < first_output, (
                "scalar_params are emitted between the inputs and the outputs, which "
                "only has one meaning if every input precedes every output; got "
                f"{[a.name for a in graph.args]}"
            )
        params = []
        for idx, arg in enumerate(graph.args):
            if idx == first_output:
                params += scalar_params
                scalar_params = []
            # A parameter filled by a gathering producer is the whole source buffer, whose
            # size only an entry keyed on the argument itself records.
            shape_node = arg if arg in graph.tmap else _bind_node(graph, arg, is_main)
            params.append(
                f"{arg.name}: "
                f"{gen_tensor_type(graph.tmap[shape_node].shape, _arg_dtype(graph, arg, is_main))}"
            )
        params += scalar_params  # no outputs to sit in front of
        if not is_main:
            params += [f"{n}: T.{t}" for n, t in graph.sargs]
            params += self._accum_params(graph, whole=True)
        return params
