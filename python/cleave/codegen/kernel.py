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
from ..ir.graph_utils import find_topo_sort
from .tensor_codegen import gen_copy, gen_tensor_allocation_str, gen_tensor_type
from .buffer import BASE_INDENT, CodeBuffer
from .macros import qualify_name
from .operands import emit_alloc, kept_extents, tc_layout


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
            kwargs += [
                f"{graph.tinfo_map[n][0]}={graph.tinfo_map[n][0]}"
                for n in graph.accum_nodes
            ]
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
            params.append(f"{arg_name}: {gen_tensor_type(tile_size, arg.get_dtype(0))}")
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
        if not is_main_iter:
            params += self._accum_params(graph)
        return f"\n@T.macro\ndef {qualified_name}_iter({''.join(p + ', ' for p in params)}):\n"

    @staticmethod
    def _accum_params(graph) -> list[str]:
        return [
            f"{graph.tinfo_map[n][0]}: {gen_tensor_type(graph.tmap[n].tile_size, n.out_dtype)}"
            for n in graph.accum_nodes
        ]

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

    def _emit_compute(self, graph, node, buf, is_main_iter, name_prefix):
        tname, scope = graph.tinfo_map[node]
        writes_to_arg = tname in {
            graph.tinfo_map[a][0] for a in graph.args if a in graph.tinfo_map
        }
        # Only non-accumulating nodes are allocated inside the loop.
        if node not in graph.accum_nodes and not writes_to_arg:
            emit_alloc(
                buf,
                graph,
                node,
                kept_extents(graph.tmap[node].tile_size),
                node.out_dtype,
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
            clear=node not in graph.accum_nodes,
            name_prefix=name_prefix,
        )
        buf.block(_call(func_name, call_args))

    def _emit_subgraph(self, graph, node, buf, is_main_iter, name_prefix):
        sub = node.graph
        self.gen_graph(sub, node.name, is_main=False, name_prefix=name_prefix)
        for accum_node in sub.accum_nodes:
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
        call_args += [sub.tinfo_map[n][0] for n in sub.accum_nodes]
        buf.block(_call(qualify_name(node.name, name_prefix), call_args))

    def _emit_output(self, graph, node, buf, is_main_iter, name_prefix):
        assert (
            len(node.inputs) == 1
        ), f"output node should have only one input, but got {node.inputs=}"
        edge = node.inputs[0]
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
        for node in find_topo_sort(graph.nodes):
            if node in graph.prefetch_nodes or node in graph.window_nodes:
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
                f"{arg.name}: {gen_tensor_type(graph.tmap[shape_node].shape, arg.get_dtype(0))}"
            )
        params += scalar_params  # no outputs to sit in front of
        if not is_main:
            params += [f"{n}: T.{t}" for n, t in graph.sargs]
            params += self._accum_params(graph)
        return params
