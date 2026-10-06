from textwrap import indent
from typing import List, Sequence
import black

from .graph import Graph, Node, PlaceHolderNode, GraphNode, OutputNode, AccumEdge
from .util import gen_copy, gen_tensor_type, gen_tensor_allocation_str, find_topo_sort
from .graph import ComputeOpNode
from .ops.base import MacroOp


TAB = "    "
BASE_INDENT = 1

# TODO: instead of generating the code for each macro
# we should consider importing the macros so that we only generate the graphs (used macros only)
IMPORT_CODE = """from math import gcd
from typing import Optional
import tilelang as tl
import tilelang.language as T
import tvm
"""

UTIL_CODE = """VEC_WIDTH = 4
DEBUG = False
def _S(A: tvm.tirx.stmt.BufferRegion, new_offset_factor: Optional[int]=None, squeeze_dim=None): # convert a sliced bufferregion to a tensor
    ranges = A.region
    shape = [r.extent for r in ranges]
    strides = A.buffer.strides
    if len(strides) == 0: # not specified
        # reversed cumprod on shapes. e.g., shape=[4, 2], strides=[2, 1]
        strides = A.buffer.shape[1:] + [1]
        for i in reversed(range(len(strides) - 1)):
            strides[i] *= strides[i + 1]
    offset = sum([r.min * s for s, r in zip(strides, ranges)])
    # TODO: infer buffer_type from original tensor
    # elem_offset: the starting offset of the first element.
    # offset_factor: a factor that divides elem_offset (though not sure where it's used in TVM?)
    if new_offset_factor is not None:
        offset_factor = gcd(A.buffer.offset_factor, new_offset_factor)
    else:
        offset_factor = A.buffer.offset_factor

    if squeeze_dim is not None:
        for i in squeeze_dim:
            assert shape[i] == 1, f"Can only squeeze dimension of size 1, but got shape {shape} and squeeze_dim {squeeze_dim}"
        shape = [s for i, s in enumerate(shape) if i not in squeeze_dim]
        strides = [s for i, s in enumerate(strides) if i not in squeeze_dim]
        if DEBUG:
            print(f"Squeezed shape: {shape}, strides: {strides}")

    return T.Buffer(shape,
                    dtype=A.buffer.dtype,
                    strides=strides,
                    elem_offset=A.buffer.elem_offset + offset, offset_factor=offset_factor,
                    buffer_type="default", align=A.buffer.data_alignment, data=A.buffer.data,
                    scope=A.buffer.scope())
"""


class TilelangCodeGen:
    def __init__(self, apply_rasterization: bool = 16, debug: bool = False):
        self.prim_code = ""
        self.graph_code = ""
        self.symbol_code = ""
        self.pass_code = ""
        self.all_names = set()
        self.generated_macros = {}
        self.apply_rasterization = apply_rasterization
        self.debug = debug

    def _qualify_name(self, name: str, prefix: str = "") -> str:
        return f"{prefix}_{name}" if prefix else name

    @staticmethod
    def _kept_extents(dims):
        """Extents with unit dims squeezed out. Symbolic extents are always kept."""
        return [d for d in dims if not isinstance(d, int) or d > 1]

    def _get_partial_name(self, node: Node) -> str:
        if node.name.startswith("PlaceHolder_"):
            return "Partial_" + node.name[len("PlaceHolder_") :]
        return f"Partial_{node.name}"

    def _replace_func_name(self, code: str, src_name: str, dst_name: str) -> str:
        if src_name == dst_name:
            return code
        return code.replace(f"def {src_name}(", f"def {dst_name}(", 1)


    def _emit_producer_call(
        self, graph, producer, dst_name, dst_scope, name_prefix: str = ""
    ) -> str:
        """Macro call that fills ``dst_name`` from ``producer``'s operands.

        Operands are the producer's tensor inputs -- inner placeholders whose OUTER binding is
        what the enclosing macro received -- followed by the op's scalar arguments.
        """
        operand_names = list(getattr(producer, "source_args", ()))
        if not operand_names:
            operand_names = [
                graph.binding.get(e.src_node.name, e.src_node).name
                for e in sorted(producer.inputs, key=lambda e: e.dst_id)
            ]
        mem_scope = "g" * len(operand_names) + dst_scope
        func_name = producer.get_macro_name(mem_scope=mem_scope, clear=False)
        self.emit_op_node(
            producer, mem_scope=mem_scope, clear=False, name_prefix=name_prefix
        )
        func_name = self._qualify_name(func_name, name_prefix)
        call = ", ".join(
            operand_names + list(getattr(producer, "scalar_args", ())) + [dst_name]
        )
        return f"{func_name}({call})\n"

    @staticmethod
    def _prefetch_buffer_name(graph, node) -> str:
        return graph.tinfo_map[node][0] + "_pf"

    @staticmethod
    def _prefetch_scalars(graph, node, it: str):
        """The node's scalar arguments, split into those the look-ahead has to re-derive.

        Yields ``(param_name, value_at_i, value_at_i_plus_1)``. A scalar whose ``smap``
        expression does not mention the loop index is the same in both iterations, so it is
        threaded through under its own name; one that does needs a second parameter, since
        the ``*_iter`` macro cannot see the index itself.
        """
        out = []
        for sname in getattr(node, "scalar_args", ()):
            resolve = graph.smap.get(sname)
            assert resolve is not None, (
                f"Prefetched node {node.name} reads scalar {sname!r}, which has no smap "
                "entry, so its value one iteration ahead cannot be derived."
            )
            here, ahead = resolve(it), resolve(f"({it} + 1)")
            out.append((sname, here, ahead if ahead != here else None))
        return out

    def _prefetch_plan(self, graph):
        """``(shifted, shared)`` for an iterative graph's prefetched reads.

        ``shifted`` names the scalars whose value the look-ahead read needs one iteration
        on; those are threaded into the ``*_iter`` macro as ``<name>_next``. ``shared``
        names the subset of them that some node in the body still reads for the CURRENT
        tile, which therefore has to be passed under both names.
        """
        shifted, readers = set(), set()
        for node in graph.prefetch_nodes:
            for sname, _, ahead in self._prefetch_scalars(graph, node, "i0"):
                if ahead is not None:
                    shifted.add(sname)
            # A prefetch off a windowed operand is re-seeded, in the body, at the CURRENT
            # iteration, so it needs its scalars under their own names as well.
            if any(e.src_node in graph.window_nodes for e in node.inputs):
                readers.update(getattr(node, "scalar_args", ()))
        for node in graph.nodes:
            if node in graph.prefetch_nodes:
                continue
            readers.update(getattr(node, "scalar_args", ()))
        return shifted, shifted & readers

    def _emit_window_fill(self, graph, node, name_prefix: str = "") -> str:
        """Macro call refilling a windowed node's hoisted buffer.

        A windowed node is a plain gather off the graph's arguments -- no dtype cast and no
        staged operand -- so this is the simple case of the node emission in the body, minus
        the allocation, which lives in the enclosing tier.
        """
        call_args, mem_scope = [], ""
        for edge in node.inputs:
            tinfo = graph.tinfo_map[edge.src_node]
            call_args.append(tinfo[0])
            mem_scope += tinfo[1]
        tname, scope = graph.tinfo_map[node]
        mem_scope += scope
        assert node not in graph.accum_nodes, (
            f"{node.name} is windowed, so its buffer is refilled, not accumulated into."
        )
        func_name = node.get_macro_name(mem_scope=mem_scope, clear=True)
        self.emit_op_node(node, mem_scope=mem_scope, clear=True, name_prefix=name_prefix)
        call = ", ".join(call_args + list(getattr(node, "scalar_args", ())) + [tname])
        return f"{self._qualify_name(func_name, name_prefix)}({call})\n"

    def _emit_prefetch_read(
        self, graph, node, scalars, name_prefix: str = ""
    ) -> str:
        """Macro call issuing ``node``'s global read into its prefetch fragment.

        ``scalars`` are the already-resolved argument expressions, in the op's own order.
        """
        call_args, mem_scope = [], ""
        for edge in node.inputs:
            tinfo = graph.tinfo_map[edge.src_node]
            call_args.append(tinfo[0])
            mem_scope += tinfo[1]
        mem_scope += "r"
        for edge, scope in zip(node.inputs, mem_scope):
            # A shared operand is only safe to read ahead if it is a windowed buffer, which
            # holds still for the whole run of iterations the look-ahead stays inside. Any
            # other shared operand is rewritten every iteration, and hoisting the read past
            # the barrier that protects it would read the wrong tile.
            assert scope != "s" or edge.src_node in graph.window_nodes, (
                f"Prefetched node {node.name} reads a shared operand ({mem_scope}) that is "
                "not windowed; its read cannot be hoisted past that operand's barrier."
            )
        func_name = node.get_macro_name(mem_scope=mem_scope, clear=False)
        self.emit_op_node(node, mem_scope=mem_scope, clear=False, name_prefix=name_prefix)
        call = ", ".join(
            call_args + list(scalars) + [self._prefetch_buffer_name(graph, node)]
        )
        return f"{self._qualify_name(func_name, name_prefix)}({call})\n"

    def _emit_prologue(
        self,
        graph: Graph,
        it: Sequence[str],
        name_prefix: str,
        arg_name_map: dict[Node, str] | None = None,
    ) -> str:
        """Buffer allocations and scan calls for ``graph.prologue``, at kernel scope.

        Emitting these before the ``*_iter`` call is what lets an ``smap`` expression or
        a ``block_guard`` read a value the device just computed, instead of one the host
        had to precompute into a work-list tensor.
        """
        scans = getattr(graph, "prologue", ())
        if not scans:
            return ""
        code = ""
        seen: set[str] = set()
        for scan in scans:
            assert (
                scan.buffer not in seen
            ), f"Two prologue scans write the buffer {scan.buffer!r}."
            seen.add(scan.buffer)
            assert scan.size >= scan.op.min_size, (
                f"{type(scan.op).__name__} indexes at least {scan.op.min_size} "
                f"elements, but {scan.buffer} is sized {scan.size}."
            )
            # gen_tensor_allocation_str asserts the extent is a compile-time constant,
            # which is exactly the constraint any kernel-scope buffer is under.
            code += gen_tensor_allocation_str(
                scan.buffer, (scan.size,), scan.dtype, scope=scan.scope
            )
            # An input has to be named the way this kernel sees it: split-k renames a
            # partial tensor's parameter, and the prologue reads that parameter.
            operands = [
                (arg_name_map.get(t, t.name) if arg_name_map else t.name)
                for t in scan.inputs
            ]
            sargs = list(scan.scalar_args(*it)) if scan.scalar_args is not None else []
            mem_scope = "g" * len(operands) + scan.scope
            self.emit_op_node(
                scan.op, mem_scope=mem_scope, clear=False, name_prefix=name_prefix
            )
            func = self._qualify_name(
                scan.op.get_macro_name(mem_scope=mem_scope, clear=False), name_prefix
            )
            code += f"{func}({', '.join(operands + sargs + [scan.buffer])})\n"
        return code

    def _emit_else_nodes(
        self,
        graph: Graph,
        it: Sequence[str],
        name_prefix: str,
        arg_name_map: dict[Node, str] | None = None,
    ) -> str:
        """Macro calls for `graph.else_nodes`, each writing one tile of a declared output.

        These take no tensor input, so the whole call is the destination slice plus the
        op's scalars -- which is the point: it is a store, not a computation.
        """
        code = ""
        for node in graph.else_nodes:
            tmap = graph.tmap[node]
            offset = tmap.offset(*it)
            sl = [f"{o} : {o} + {ts}" for o, ts in zip(offset, tmap.tile_size)]
            squeeze = [i for i, ts in enumerate(tmap.tile_size) if ts == 1]
            if len(squeeze) == len(tmap.tile_size):
                squeeze = squeeze[:-1]
            buf = graph.tinfo_map[node][0]
            # The parameter name, which split-k rewrites: look it up through the OUTPUT
            # this fill writes, not through the fill itself (which is never an arg, so
            # keying the map by it silently found nothing).
            if arg_name_map:
                for arg, name in arg_name_map.items():
                    if graph.tinfo_map.get(arg, (None,))[0] == buf:
                        buf = name
                        break
            dst = f"_S({buf}[{', '.join(sl)}], squeeze_dim={squeeze})"
            mem_scope = graph.tinfo_map[node][-1]
            self.emit_op_node(
                node, mem_scope=mem_scope, clear=False, name_prefix=name_prefix
            )
            func = self._qualify_name(
                node.get_macro_name(mem_scope=mem_scope, clear=False), name_prefix
            )
            # The tile says how much room there is; `min_size`/`dst_rank` say what the
            # hand-written body actually touches. They are independent, so a fill whose
            # own extents disagree with its tile would either leave part of the tile
            # unwritten or run past it -- both silent.
            tile_elems = 1
            for extent in tmap.tile_size:
                tile_elems *= extent if isinstance(extent, int) else 1
            assert node.min_size == tile_elems, (
                f"else node {node.name!r} writes {node.min_size} elements but its tile "
                f"{tuple(tmap.tile_size)} holds {tile_elems}."
            )
            kept = len(self._kept_extents(tmap.tile_size))
            if hasattr(node, "dst_rank"):
                assert node.dst_rank() == kept, (
                    f"else node {node.name!r} indexes a rank-{node.dst_rank()} "
                    f"destination, but its tile reduces to rank {kept}."
                )
            sargs = [graph.smap[s](*it) for s in getattr(node, "scalar_args", ())]
            code += f"{func}({', '.join(sargs + [dst])})\n"
        return code

    def _build_graph_call(
        self,
        graph: Graph,
        name: str,
        is_main: bool = True,
        name_prefix: str = "",
        arg_name_map: dict[Node, str] | None = None,
    ) -> tuple[str, str]:
        dep_code, body_code, call_code = "", "", ""
        name_prefix = name_prefix or ""
        it = ["i0"]
        qualified_name = self._qualify_name(name, name_prefix)
        if is_main:
            assert hasattr(graph, "n_threads"), "Main graph must define n_threads."
            assert isinstance(
                graph.extent, (int, tuple)
            ), "Main graphs only support int or tuple extent."
            if isinstance(graph.extent, tuple):
                assert (
                    1 <= len(graph.extent) <= 3
                ), "Only supports up to a 3D grid for now."
                grid_dims = graph.extent
                grid_str = ", ".join(map(str, grid_dims))
                n_it = len(grid_dims)
            else:
                grid_str = f"{graph.extent}"
                n_it = 1
            it = [f"i{d}" for d in range(n_it)]
            body_code += (
                TAB * BASE_INDENT
                + f"with T.Kernel({grid_str}, threads=({graph.n_threads})) as ({', '.join(it)}):\n"
            )
            rast = (
                self.apply_rasterization
                if graph.rasterization is None
                else graph.rasterization
            )
            if rast > 1:
                body_code += TAB * (BASE_INDENT + 1) + f"T.use_swizzle({rast})\n"
        else:
            assert not hasattr(graph, "n_threads"), "Inner graphs must not define n_threads."
            # A prologue buffer is allocated at kernel scope and a guard wraps a whole
            # grid slot; neither has a meaning one tier down, and no operator needs an
            # inner-tier scan, so refuse them rather than emit something unreachable.
            assert not graph.prologue and graph.block_guard is None, (
                "prologue/block_guard are main-graph only."
            )
            assert isinstance(
                graph.extent, int
            ), "Iterative graphs only support int extent."
            trip = graph.loop_trip if graph.loop_trip is not None else f"{graph.extent}"
            if graph.loop_kind == "serial":
                body_code += TAB * BASE_INDENT + f"for {it[0]} in T.serial({trip}):\n"
            else:
                stages = (
                    graph.loop_stages
                    if graph.loop_stages is not None
                    else min(3, graph.extent)
                )
                body_code += (
                    TAB * BASE_INDENT
                    + f"for {it[0]} in T.Pipelined({trip}, num_stages={stages}):\n"
                )

        # The call is accumulated separately so a `block_guard` can indent all of it by
        # one more tab without disturbing the header or the `dep_code` above it.
        call_code += f"{qualified_name}_iter("
        region_args = set(graph.region_args)
        for arg in graph.args:
            if getattr(arg, "index_only", False):
                continue
            bind_node = (
                graph.binding[arg.name]
                if isinstance(arg, (PlaceHolderNode, OutputNode)) and not is_main
                else arg
            )
            assert bind_node in graph.tmap, f"Argument {bind_node} not in tensor map."
            tmap = graph.tmap[bind_node]
            offset = tmap.offset(*it)
            arg_name = arg_name_map.get(arg, arg.name) if arg_name_map else arg.name
            if tmap.shape == tmap.tile_size and not is_main:
                # no need to slice
                assert all(
                    [i == "0" for i in offset]
                ), f"Expect all zero offset, but got {offset=}"

                tinfo = graph.tinfo_map[bind_node]
                tinfo_arg = graph.tinfo_map[arg]
                same_level = tinfo[-1] == tinfo_arg[-1]
                if not same_level:
                    dep_code += indent(
                        gen_tensor_allocation_str(
                            tinfo[0],
                            self._kept_extents(tmap.shape),
                            arg.get_dtype(0),
                            scope=tinfo[-1],
                            is_tc_layout=(graph.tlayout_map.get(bind_node, "") == "tc"),
                        ),
                        TAB * BASE_INDENT,
                    )
                producer = getattr(bind_node, "producer_op", None)
                if producer is not None:
                    dep_code += indent(
                        self._emit_producer_call(
                            graph, producer, tinfo[0], tinfo[-1], name_prefix
                        ),
                        TAB * BASE_INDENT,
                    )
                else:
                    dep_code += indent(
                        gen_copy(
                            src=tinfo_arg[0],
                            dst=tinfo[0],
                            same_level=same_level,
                            is_tc_layout=(graph.tlayout_map.get(bind_node, "") == "tc"),
                            movement=(tinfo_arg[1], tinfo[1]),
                            tile_shape=tmap.shape,
                            dtype=arg.get_dtype(0),
                        ),
                        TAB * BASE_INDENT,
                    )
                call_code += f"{tinfo[0]}={tinfo[0]}, "
            else:
                s = [f"{o} : {o} + {ts}" for o, ts in zip(offset, tmap.tile_size)]
                squeeze_dims = [i for i, ts in enumerate(tmap.tile_size) if ts == 1]
                # Squeezing every axis would leave a scalar; a single-element tile still
                # has to be passed as a 1-D tensor.
                if len(squeeze_dims) == len(tmap.tile_size):
                    squeeze_dims = squeeze_dims[:-1]
                if bind_node in region_args or arg in region_args:
                    # A unit extent becomes a scalar subscript, which is how a
                    # BufferRegion drops a rank -- `_S`'s squeeze_dim has no equivalent
                    # here, and the point of the region is to keep tilelang's own view of
                    # the buffer so it can predicate and bulk-copy the access.
                    subs = [
                        r if i not in squeeze_dims else o
                        for i, (o, r) in enumerate(zip(offset, s))
                    ]
                    call_code += f"{arg.name}={arg_name}[{', '.join(subs)}], "
                else:
                    call_code += f"{arg.name}=_S({arg_name}[{', '.join(s)}], squeeze_dim={squeeze_dims}), "
        shifted, shared_shifted = self._prefetch_plan(graph)
        for sname, _ in (graph.sargs if graph.iter_sargs is None else graph.iter_sargs):
            resolve = graph.smap.get(sname)
            assert resolve is not None, f"Scalar argument {sname} has no smap entry."
            if sname not in shifted or sname in shared_shifted:
                call_code += f"{sname}={resolve(*it)}, "
            if sname in shifted:
                ahead = [f"({v} + 1)" if i == 0 else v for i, v in enumerate(it)]
                call_code += f"{sname}_next={resolve(*ahead)}, "
        # A windowed node's buffer is allocated once, outside the loop; the first iteration
        # of every window fills it and the rest read what it left.
        for node in graph.window_nodes:
            tname, scope = graph.tinfo_map[node]
            dep_code += indent(
                gen_tensor_allocation_str(
                    tname,
                    self._kept_extents(graph.tmap[node].tile_size),
                    node.out_dtype,
                    scope=scope,
                    is_tc_layout=(graph.tlayout_map.get(node, "") == "tc"),
                ),
                TAB * BASE_INDENT,
            )
            call_code += f"{tname}={tname}, "
        # A prefetch fragment is allocated once, outside the loop, and seeded with the tile
        # iteration 0 will consume; from then on each iteration hands the buffer back and
        # refills it for its successor.
        for node in graph.prefetch_nodes:
            tname = self._prefetch_buffer_name(graph, node)
            dep_code += indent(
                gen_tensor_allocation_str(
                    tname,
                    self._kept_extents(graph.tmap[node].tile_size),
                    node.out_dtype,
                    scope="r",
                ),
                TAB * BASE_INDENT,
            )
            # ... except when the read is off a windowed buffer, which does not exist yet
            # out here. The body seeds those on the iteration that fills the window, and
            # iteration 0 is always such an iteration.
            if not any(e.src_node in graph.window_nodes for e in node.inputs):
                dep_code += indent(
                    self._emit_prefetch_read(
                        graph,
                        node,
                        [here for _, here, _ in self._prefetch_scalars(graph, node, "0")],
                        name_prefix,
                    ),
                    TAB * BASE_INDENT,
                )
            call_code += f"{tname}={tname}, "
        if not is_main:
            for accum_node in graph.accum_nodes:
                tinfo = graph.tinfo_map[accum_node]
                call_code += f"{tinfo[0]}={tinfo[0]}, "
        call_code += ")\n"

        call_indent = BASE_INDENT + 1
        if is_main:
            body_code += indent(
                self._emit_prologue(graph, it, name_prefix, arg_name_map),
                TAB * (BASE_INDENT + 1),
            )
            if graph.block_guard is not None:
                body_code += (
                    TAB * (BASE_INDENT + 1) + f"if {graph.block_guard(*it)}:\n"
                )
                call_indent = BASE_INDENT + 2
            if graph.work_guard is not None:
                body_code += (
                    TAB * call_indent + f"if {graph.work_guard(*it)}:\n"
                )
                call_indent += 1
        body_code += indent(call_code, TAB * call_indent)
        if is_main and graph.else_nodes:
            assert (
                graph.work_guard is not None
            ), "else_nodes need a work_guard to be the alternative to."
            body_code += TAB * (call_indent - 1) + "else:\n"
            body_code += indent(
                self._emit_else_nodes(graph, it, name_prefix, arg_name_map),
                TAB * call_indent,
            )
        return dep_code, body_code

    def _gen_graph_iter(
        self,
        graph: Graph,
        name: str,
        args: List[Node],
        is_main_iter: bool = False,
        name_prefix: str = "",
    ):
        # generate the iteration function
        qualified_name = self._qualify_name(name, name_prefix)
        iter_code = "\n"
        iter_code += f"@T.macro\ndef {qualified_name}_iter("
        for arg in args:
            if getattr(arg, "index_only", False):
                continue
            bind_node = (
                graph.binding[arg.name]
                if isinstance(arg, (PlaceHolderNode, OutputNode)) and not is_main_iter
                else arg
            )
            tmap = graph.tmap[bind_node]
            # Unit extents are squeezed out of the declared type, but a tile whose every
            # extent is 1 must still be declared 1-D rather than collapsing to a scalar.
            tile_size = self._kept_extents(tmap.tile_size) or [1]
            # check if tensor was copied in from main function
            arg_name = bind_node.name if tmap.shape == tmap.tile_size else arg.name
            iter_code += f"{arg_name}: {gen_tensor_type(tile_size, arg.get_dtype(0))}, "
        shifted, shared_shifted = self._prefetch_plan(graph)
        for sname, sdtype in (
            graph.sargs if graph.iter_sargs is None else graph.iter_sargs
        ):
            if sname not in shifted or sname in shared_shifted:
                iter_code += f"{sname}: T.{sdtype}, "
            if sname in shifted:
                iter_code += f"{sname}_next: T.{sdtype}, "
        for node in graph.prefetch_nodes:
            tmap = graph.tmap[node]
            iter_code += (
                f"{self._prefetch_buffer_name(graph, node)}: "
                f"{gen_tensor_type(self._kept_extents(tmap.tile_size), node.out_dtype)}, "
            )
        for node in graph.window_nodes:
            tmap = graph.tmap[node]
            iter_code += (
                f"{graph.tinfo_map[node][0]}: "
                f"{gen_tensor_type(self._kept_extents(tmap.tile_size), node.out_dtype)}, "
            )
        if not is_main_iter:
            for accum_node in graph.accum_nodes:
                tmap = graph.tmap[accum_node]
                tname, _ = graph.tinfo_map[accum_node]
                iter_code += f"{tname}: {gen_tensor_type(tmap.tile_size, accum_node.out_dtype)}, "
        iter_code += "):\n"

        # A window fill leads everything, because a prefetch reading that window has to be
        # re-seeded from it on the iteration that refills it: its predecessor's look-ahead
        # stopped at the window's edge, so this tile was never fetched.
        windowed_prefetch = [
            n
            for n in graph.prefetch_nodes
            if any(e.src_node in graph.window_nodes for e in n.inputs)
        ]
        if graph.window_nodes:
            assert graph.window_guard is not None, (
                "window_nodes without a window_guard would refill every iteration, which "
                "is what hoisting the buffer is meant to avoid."
            )
            iter_code += TAB * BASE_INDENT + f"if {graph.window_guard}:\n"
            for node in graph.window_nodes:
                iter_code += indent(
                    self._emit_window_fill(graph, node, name_prefix),
                    TAB * (BASE_INDENT + 1),
                )
            for node in windowed_prefetch:
                iter_code += indent(
                    self._emit_prefetch_read(
                        graph,
                        node,
                        list(getattr(node, "scalar_args", ())),
                        name_prefix,
                    ),
                    TAB * (BASE_INDENT + 1),
                )

        # Prefetched tiles lead the body: every hand-back is issued before any look-ahead
        # read, so the fragments are free the moment the reads need them, and every
        # look-ahead is issued before this tile's first consumer, which is the whole point
        # -- the loads are then in flight across all of the math below.
        for node in graph.prefetch_nodes:
            assert node not in graph.accum_nodes, (
                f"{node.name} accumulates across iterations, so its buffer cannot be "
                "refilled from a prefetch."
            )
            iter_code += indent(
                gen_tensor_allocation_str(
                    graph.tinfo_map[node][0],
                    self._kept_extents(graph.tmap[node].tile_size),
                    node.out_dtype,
                    scope=graph.tinfo_map[node][-1],
                    is_tc_layout=(graph.tlayout_map.get(node, "") == "tc"),
                ),
                TAB * BASE_INDENT,
            )
            iter_code += indent(
                gen_copy(
                    src=self._prefetch_buffer_name(graph, node),
                    dst=graph.tinfo_map[node][0],
                    same_level=False,
                    movement=("r", graph.tinfo_map[node][-1]),
                    tile_shape=self._kept_extents(graph.tmap[node].tile_size),
                    dtype=node.out_dtype,
                ),
                TAB * BASE_INDENT,
            )
        if graph.prefetch_nodes:
            read_indent = BASE_INDENT
            if graph.prefetch_guard is not None:
                iter_code += TAB * BASE_INDENT + f"if {graph.prefetch_guard}:\n"
                read_indent += 1
            for node in graph.prefetch_nodes:
                iter_code += indent(
                    self._emit_prefetch_read(
                        graph,
                        node,
                        [
                            sname if ahead is None else f"{sname}_next"
                            for sname, _, ahead in self._prefetch_scalars(
                                graph, node, "i0"
                            )
                        ],
                        name_prefix,
                    ),
                    TAB * read_indent,
                )

        for node in find_topo_sort(graph.nodes):
            if node in graph.prefetch_nodes or node in graph.window_nodes:
                continue
            if isinstance(node, PlaceHolderNode):
                if not is_main_iter:
                    tmap = graph.tmap[node]
                    tinfo = graph.tinfo_map[node]
                    tinfo_arg = graph.tinfo_map[graph.binding[node.name]]

                    # if tmap.shape equals tmap.tile_size,
                    # no need to slice or load because already loaded in the main function
                    copied_before = tmap.shape == tmap.tile_size
                    same_level = tinfo[-1] == tinfo_arg[-1]
                    if not same_level and not copied_before:
                        iter_code += indent(
                            gen_tensor_allocation_str(
                                graph.tinfo_map[node][0],
                                self._kept_extents(tmap.tile_size),
                                node.get_dtype(0),
                                scope=tinfo[-1],
                                is_tc_layout=(graph.tlayout_map.get(node, "") == "tc"),
                            ),
                            TAB * BASE_INDENT,
                        )
                    if not copied_before:
                        iter_code += indent(
                            gen_copy(
                                src=tinfo_arg[0],
                                dst=tinfo[0],
                                same_level=same_level,
                                is_tc_layout=(graph.tlayout_map.get(node, "") == "tc"),
                                movement=(tinfo_arg[1], tinfo[1]),
                                tile_shape=tmap.tile_size,
                                dtype=node.get_dtype(0),
                            ),
                            TAB * BASE_INDENT,
                        )
                continue

            elif isinstance(node, ComputeOpNode):
                writes_to_arg = graph.tinfo_map[node][0] in {
                    graph.tinfo_map[a][0] for a in graph.args if a in graph.tinfo_map
                }
                # only non-accumulation nodes are allocated inside the loop
                if not node in graph.accum_nodes and not writes_to_arg:
                    iter_code += indent(
                        gen_tensor_allocation_str(
                            graph.tinfo_map[node][0],
                            self._kept_extents(graph.tmap[node].tile_size),
                            node.out_dtype,
                            scope=graph.tinfo_map[node][-1],
                            is_tc_layout=(graph.tlayout_map.get(node, "") == "tc"),
                        ),
                        TAB * BASE_INDENT,
                    )

                # set call arguments and memory placement
                call_args, mem_scope = "", ""
                stage_input_indices = set(
                    getattr(node, "stage_fragment_operands_in_shared", ())
                )
                for input_idx, edge in enumerate(node.inputs):
                    if isinstance(edge.src_node, GraphNode):
                        assert isinstance(
                            edge, AccumEdge
                        ), f"Expect accumulation edge, but got {edge=}"
                        assert isinstance(
                            edge.accum_node, ComputeOpNode
                        ), f"Accumulation node must be ComputeOpNode, but got {edge.accum_node=}"

                        input_out_dtype = edge.accum_node.out_dtype
                        tinfo = edge.src_node.graph.tinfo_map[edge.accum_node]
                        operand_tile_shape = self._kept_extents(
                            edge.src_node.graph.tmap[edge.accum_node].tile_size
                        )

                        # cast if different dtype
                        if input_out_dtype != node.dtype and (
                            input_idx not in getattr(node, "index_inputs", ())
                        ):
                            fuse_stage = input_idx in stage_input_indices
                            tinfo = (tinfo[0] + "_cast", "s" if fuse_stage else "r")
                            iter_code += indent(
                                gen_tensor_allocation_str(
                                    tinfo[0],
                                    operand_tile_shape,
                                    node.dtype,
                                    scope=tinfo[1],
                                    is_tc_layout=fuse_stage,
                                ),
                                TAB * BASE_INDENT,
                            )
                            iter_code += indent(
                                gen_copy(
                                    src=tinfo[0][:-5],
                                    dst=tinfo[0],
                                    same_level=False,
                                ),
                                TAB * BASE_INDENT,
                            )
                    else:
                        input_out_dtype = (
                            edge.src_node.out_dtype
                            if isinstance(edge.src_node, ComputeOpNode)
                            else str(edge.src_node.get_dtype(0))
                        )
                        tinfo = graph.tinfo_map[edge.src_node]
                        operand_tile_shape = [
                            *self._kept_extents(graph.tmap[edge.src_node].tile_size)
                        ]

                        # cast if different dtype
                        if input_out_dtype != node.dtype and (
                            input_idx not in getattr(node, "index_inputs", ())
                        ):
                            fuse_stage = input_idx in stage_input_indices
                            tinfo = (tinfo[0] + "_cast", "s" if fuse_stage else "r")
                            iter_code += indent(
                                gen_tensor_allocation_str(
                                    tinfo[0],
                                    operand_tile_shape,
                                    node.dtype,
                                    scope=tinfo[1],
                                    is_tc_layout=fuse_stage,
                                ),
                                TAB * BASE_INDENT,
                            )
                            iter_code += indent(
                                gen_copy(
                                    src=tinfo[0][:-5],
                                    dst=tinfo[0],
                                    same_level=False,
                                ),
                                TAB * BASE_INDENT,
                            )
                    operand_tname = tinfo[0]
                    operand_dtype = (
                        node.dtype if input_out_dtype != node.dtype else input_out_dtype
                    )
                    if input_idx in stage_input_indices and tinfo[1] == "r":
                        operand_tname = f"{tinfo[0]}_intermediate"
                        iter_code += indent(
                            gen_tensor_allocation_str(
                                operand_tname,
                                operand_tile_shape,
                                operand_dtype,
                                scope="s",
                                is_tc_layout=True,
                            ),
                            TAB * BASE_INDENT,
                        )
                        iter_code += indent(
                            gen_copy(
                                src=tinfo[0],
                                dst=operand_tname,
                                same_level=False,
                            ),
                            TAB * BASE_INDENT,
                        )
                    call_args += f"{operand_tname}, "
                    mem_scope += tinfo[1]
                for sname in getattr(node, "scalar_args", ()):
                    call_args += f"{sname}, "
                call_args += f"{graph.tinfo_map[node][0]}, "
                mem_scope += graph.tinfo_map[node][1]
                func_name = node.get_macro_name(
                    mem_scope=mem_scope, clear=node not in graph.accum_nodes
                )
                self.emit_op_node(
                    node,
                    mem_scope=mem_scope,
                    clear=node not in graph.accum_nodes,
                    name_prefix=name_prefix,
                )
                func_name = self._qualify_name(func_name, name_prefix)
                iter_code += indent(
                    f"{func_name}({call_args[:-2]})\n", TAB * BASE_INDENT
                )

            elif isinstance(node, GraphNode):
                self.gen_graph(
                    node.graph,
                    node.name,
                    is_main=False,
                    name_prefix=name_prefix,
                )

                for accum_node in node.graph.accum_nodes:
                    assert (
                        accum_node in node.graph.tmap
                    ), f"Accumulation node {accum_node} not in tensor map."
                    tmap = node.graph.tmap[accum_node]
                    tname, scope = node.graph.tinfo_map[accum_node]
                    iter_code += indent(
                        gen_tensor_allocation_str(
                            tname,
                            tmap.tile_size,
                            accum_node.out_dtype,
                            scope,
                        ),
                        TAB * BASE_INDENT,
                    )
                    iter_code += indent(f"T.clear({tname})\n", TAB * BASE_INDENT)

                # set call arguments
                call_args = ""
                for edge in node.inputs:
                    if isinstance(edge.src_node, GraphNode):
                        assert isinstance(
                            edge, AccumEdge
                        ), f"Expect accumulation edge, but got {edge=}"
                        tinfo = edge.src_node.graph.tinfo_map[edge.accum_node]
                    else:
                        tinfo = graph.tinfo_map[edge.src_node]
                    call_args += f"{tinfo[0]}, "
                for sname, _ in node.graph.sargs:
                    call_args += f"{sname}, "
                for accum_node in node.graph.accum_nodes:
                    call_args += f"{node.graph.tinfo_map[accum_node][0]}, "
                func_name = self._qualify_name(node.name, name_prefix)
                iter_code += indent(
                    f"{func_name}({call_args[:-2]})\n", TAB * BASE_INDENT
                )

            elif isinstance(node, OutputNode):
                assert (
                    len(node.inputs) == 1
                ), f"output node should have only one input, but got {node.inputs=}"
                is_subgraph_output = isinstance(node.inputs[0].src_node, GraphNode)

                if is_subgraph_output:
                    output_edge = node.inputs[0]
                    src_graph_node = output_edge.src_node
                    src_node, _, _ = src_graph_node._resolve_output_binding(
                        output_edge.src_id
                    )
                    tinfo_map = src_graph_node.graph.tinfo_map
                    tinfo_src = tinfo_map[src_node]
                    tinfo_dst = graph.tinfo_map[node]
                else:
                    src_node = node.inputs[0].src_node
                    tinfo_src = graph.tinfo_map[src_node]
                    tinfo_dst = graph.tinfo_map[node]

                # TODO: this may cause redundancy sometimes if the output of a sub graph is moved to global memory
                # or even shared memory. may need careful optimization later
                if tinfo_src[1] != tinfo_dst[1]:
                    iter_code += indent(
                        gen_copy(
                            src=tinfo_src[0],
                            dst=tinfo_dst[0],
                            same_level=False,
                            is_tc_layout=(graph.tlayout_map.get(node, "") == "tc"),
                            movement=(tinfo_src[1], tinfo_dst[1]),
                            tile_shape=[
                                i for i in graph.tmap[node].tile_size if i != 1
                            ],
                            dtype=node.get_dtype(0),
                        ),
                        TAB * BASE_INDENT,
                    )
                continue

            else:
                raise NotImplementedError(f"Unknown node type: {node}")

            # if isinstance(node.outputs[0].dst_node, OutputNode):
            #     out_node = node.outputs[0].dst_node # so that it reuses the args buffer # TODO: enable this optimization

        iter_code += "\n"
        self.emit(iter_code)

    def gen_graph(
        self,
        graph: Graph,
        name: str,
        is_main: bool = True,
        name_prefix: str = "",
        emit_wrapper: bool = True,
    ):
        self._gen_graph_iter(
            graph,
            f"{name}",
            graph.args,
            is_main_iter=is_main,
            name_prefix=name_prefix,
        )
        if not emit_wrapper:
            return

        # 1. generate the function signature
        # 1.1 generate decorator and name
        qualified_name = self._qualify_name(name, name_prefix)
        self.emit(f"@T.prim_func\n") if is_main else self.emit(f"@T.macro\n")
        self.emit(f"def {qualified_name}(")

        # 1.2 generate parameters
        # A declared scalar input sits between the tensor inputs and the outputs, so the
        # emitted signature matches the operator's own parameter order.
        scalar_params = list(graph.scalar_params) if is_main else []
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
        else:
            first_output = len(graph.args)
        for idx, arg in enumerate(graph.args):
            if scalar_params and idx == first_output:
                for sname, sdtype in scalar_params:
                    self.emit(f"{sname}: T.{sdtype}, ")
                scalar_params = []
            bind_node = (
                graph.binding[arg.name]
                if isinstance(arg, (PlaceHolderNode, OutputNode)) and not is_main
                else arg
            )
            # Normally the parameter IS the bound tensor. It is not when the bound tensor
            # is filled by a producer that gathers from the caller's buffer: then the
            # parameter is that whole source buffer, and only the graph itself can say how
            # big it is, so an entry keyed on the argument wins over the bound node's tile.
            shape_node = arg if arg in graph.tmap else bind_node
            self.emit(
                f"{arg.name}: {gen_tensor_type(graph.tmap[shape_node].shape, arg.get_dtype(0))}, "
            )
        for sname, sdtype in scalar_params:  # no outputs to sit in front of
            self.emit(f"{sname}: T.{sdtype}, ")
        if not is_main:
            for sname, sdtype in graph.sargs:
                self.emit(f"{sname}: T.{sdtype}, ")
        if not is_main:
            for accum_node in graph.accum_nodes:
                tmap = graph.tmap[accum_node]
                tname, _ = graph.tinfo_map[accum_node]
                self.emit(
                    f"{tname}: {gen_tensor_type(tmap.tile_size, accum_node.out_dtype)}, "
                )
        self.emit("):\n")

        dep_code, body_code = self._build_graph_call(
            graph,
            name,
            is_main=is_main,
            name_prefix=name_prefix,
        )
        self.emit(dep_code), self.emit(body_code)

    def _emit_combined_main_kernel(self, split_graph: Graph, combine_graph: Graph):
        split_outputs = [arg for arg in split_graph.args if isinstance(arg, OutputNode)]
        combine_inputs = [
            arg for arg in combine_graph.args if isinstance(arg, PlaceHolderNode)
        ]
        assert (
            len(split_outputs) == len(combine_inputs) and len(split_outputs) > 0
        ), "Split-k combined codegen expects matching split outputs and combine inputs."
        for g in (split_graph, combine_graph):
            # Both halves share one prim_func body, so two prologues would have to agree
            # on buffer names; no operator needs a scan under split-k, so forbid it.
            assert not g.prologue and g.block_guard is None, (
                "Split-k combined codegen does not support prologue/block_guard."
            )
            # An else node names its destination through `tinfo_map`, but split-k renames
            # the partial output's parameter, so the fill would emit the pre-rename name
            # and fail to resolve. Refuse rather than emit that.
            assert not g.else_nodes and g.work_guard is None, (
                "Split-k combined codegen does not support work_guard/else_nodes."
            )

        partial_bindings = []
        for split_output, combine_input in zip(split_outputs, combine_inputs):
            split_tmap = split_graph.tmap[split_output]
            combine_tmap = combine_graph.tmap[combine_input]
            assert (
                split_tmap.shape == combine_tmap.shape
            ), f"Partial tensor shape mismatch: {split_tmap.shape} vs {combine_tmap.shape}"
            assert split_output.get_dtype(0) == combine_input.get_dtype(
                0
            ), "Partial tensor dtype mismatch."
            partial_name = self._get_partial_name(combine_input)
            partial_bindings.append(
                (
                    split_output,
                    combine_input,
                    f"{partial_name}_split",
                    f"{partial_name}_combine",
                )
            )

        # A scalar kernel input (the softmax scale of a paged attention) is declared once
        # and shared by both halves, so the two may only disagree by one of them not
        # reading it at all. It sits where the single-kernel emitter puts it -- after the
        # tensor inputs -- so the operator's parameter order is preserved up to the
        # partials, which are the one thing a split signature adds.
        scalar_params = []
        for g in (split_graph, combine_graph):
            for sname, sdtype in g.scalar_params:
                known = dict(scalar_params).get(sname)
                assert known is None or known == sdtype, (
                    f"scalar param {sname!r} is declared as {known} by one half and "
                    f"{sdtype} by the other."
                )
                if known is None:
                    scalar_params.append((sname, sdtype))

        self.emit("@T.prim_func\n")
        self.emit("def main_kernel(")
        for arg in split_graph.args:
            if isinstance(arg, OutputNode):
                continue
            self.emit(
                f"{arg.name}: {gen_tensor_type(split_graph.tmap[arg].shape, arg.get_dtype(0))}, "
            )
        for sname, sdtype in scalar_params:
            self.emit(f"{sname}: T.{sdtype}, ")
        for (
            _,
            combine_input,
            split_partial_name,
            combine_partial_name,
        ) in partial_bindings:
            self.emit(
                f"{split_partial_name}: {gen_tensor_type(combine_graph.tmap[combine_input].shape, combine_input.get_dtype(0))}, "
            )
            self.emit(
                f"{combine_partial_name}: {gen_tensor_type(combine_graph.tmap[combine_input].shape, combine_input.get_dtype(0))}, "
            )
        for arg in combine_graph.args:
            if isinstance(arg, OutputNode):
                self.emit(
                    f"{arg.name}: {gen_tensor_type(combine_graph.tmap[arg].shape, arg.get_dtype(0))}, "
                )
        self.emit("):\n")

        split_arg_name_map = {
            split_output: split_partial_name
            for split_output, _, split_partial_name, _ in partial_bindings
        }
        dep_code, body_code = self._build_graph_call(
            split_graph,
            name="main_kernel",
            is_main=True,
            name_prefix="split",
            arg_name_map=split_arg_name_map,
        )
        self.emit(dep_code)
        self.emit(body_code)

        combine_arg_name_map = {
            combine_input: combine_partial_name
            for _, combine_input, _, combine_partial_name in partial_bindings
        }
        dep_code, body_code = self._build_graph_call(
            combine_graph,
            name="main_kernel",
            is_main=True,
            name_prefix="combine",
            arg_name_map=combine_arg_name_map,
        )
        self.emit(dep_code)
        self.emit(body_code)

    def _gen_split_kernel(self, graphs: Sequence[Graph]):
        assert (
            len(graphs) == 2
        ), "Split-k combined codegen currently supports exactly two graphs."
        split_graph, combine_graph = graphs
        split_codegen = TilelangCodeGen(
            apply_rasterization=self.apply_rasterization,
            debug=self.debug,
        )
        split_codegen.gen_graph(
            split_graph,
            name="main_kernel",
            is_main=True,
            name_prefix="split",
            emit_wrapper=False,
        )
        self.emit(split_codegen.prim_code + split_codegen.graph_code)

        combine_codegen = TilelangCodeGen(
            apply_rasterization=self.apply_rasterization,
            debug=self.debug,
        )
        combine_codegen.gen_graph(
            combine_graph,
            name="main_kernel",
            is_main=True,
            name_prefix="combine",
            emit_wrapper=False,
        )
        self.emit(combine_codegen.prim_code + combine_codegen.graph_code)
        self._emit_combined_main_kernel(split_graph, combine_graph)

    def gen(self, graph: Graph | Sequence[Graph]):
        self.prim_code = ""
        self.graph_code = ""
        self.symbol_code = ""
        self.pass_code = ""
        self.generated_macros = {}
        for g in graph if isinstance(graph, (list, tuple)) else [graph]:
            self._validate_graph(g, is_main=True)
        self._declare_dyn_dims(graph)
        self._declare_pass_configs(graph)

        if isinstance(graph, (list, tuple)):
            if len(graph) == 1:
                graph = graph[0]
            else:
                self._gen_split_kernel(graph)
                return self.format_code()

        self.gen_graph(graph, name="main_kernel", is_main=True)
        return self.format_code()

    def _validate_graph(self, graph, *, is_main: bool, seen=None):
        """Reject a graph that does not have the shape the emitter assumes.

        Everything below is an invariant the emitter already relies on; without these
        checks a schedule that breaks one produces malformed TileLang or an error from
        deep inside TVM, attributed to the wrong place. Each escape-hatch field added for
        a performance reason (`prologue`, `region_args`, `scalar_params`, `work_guard` /
        `else_nodes`) is checked here too, so it stays a controlled extension of the
        structure rather than a way around it.
        """
        seen = seen if seen is not None else set()
        if id(graph) in seen:
            return
        seen.add(id(graph))
        where = "main graph" if is_main else "subgraph"

        assert graph.nodes, f"{where}: no nodes."

        # --- args: only declared IO, inputs first, each fully described -----------------
        seen_output = False
        for arg in graph.args:
            assert isinstance(
                arg, (PlaceHolderNode, OutputNode)
            ), f"{where}: arg {arg.name!r} is a {type(arg).__name__}, not declared IO."
            if isinstance(arg, OutputNode):
                seen_output = True
            elif graph.scalar_params:
                # Only load-bearing when scalars are emitted at the input/output boundary;
                # `annotate_args` appends in topological order, so a legitimate
                # multi-output graph can interleave and must not be rejected for it.
                assert not seen_output, (
                    f"{where}: input {arg.name!r} comes after an output in `args`. The "
                    "emitted signature keeps this order, and scalar params are inserted "
                    "at the boundary, so inputs must all precede outputs."
                )
        tmap = graph.tmap
        if isinstance(tmap, dict):
            for arg in graph.args:
                # A subgraph argument may instead be bound from the parent (`binding`),
                # in which case the parent's tmap is what slices it and the subgraph has
                # no entry of its own.
                assert (
                    arg in tmap or not is_main
                ), f"{where}: arg {arg.name!r} has no tmap entry."
                assert (
                    arg in graph.tinfo_map
                ), f"{where}: arg {arg.name!r} has no tinfo_map entry."
                if arg in tmap:
                    it = tmap[arg]
                    assert len(it.tile_size) == len(it.shape), (
                        f"{where}: arg {arg.name!r} tile_size {it.tile_size} and shape "
                        f"{it.shape} disagree in rank."
                    )

        # --- accumulators and regions must be part of this graph -----------------------
        for node in graph.accum_nodes:
            assert (
                node in graph.nodes
            ), f"{where}: accum node {node.name!r} is not in `nodes`."
        for node in graph.region_args:
            # Membership in `args` is deliberately NOT the invariant: a subgraph's region
            # args are its own inner placeholders, which are not in `args` at the point
            # this runs. What the emitter actually depends on is that a region arg is an IO
            # node whose tile is known, since the region's scalar subscripts come from it.
            assert isinstance(node, (PlaceHolderNode, OutputNode)), (
                f"{where}: region arg {node.name!r} is a {type(node).__name__}. Only a "
                "graph argument can be passed as a region."
            )
            assert not isinstance(tmap, dict) or node in tmap, (
                f"{where}: region arg {node.name!r} has no tmap entry, so its region "
                "subscripts cannot be built."
            )

        # --- scalars: every name an op or a guard reads must be derivable ---------------
        for name, _dtype in graph.sargs:
            assert name in graph.smap, f"{where}: sarg {name!r} has no smap expression."
        if graph.iter_sargs is not None:
            sarg_names = {n for n, _ in graph.sargs}
            for name, _dtype in graph.iter_sargs:
                # Either a scalar this graph derives, or one the loop supplies (a tile
                # base, say), which is resolved through `smap` at the call site.
                assert name in sarg_names or name in graph.smap, (
                    f"{where}: iter_sarg {name!r} is neither one of `sargs` nor "
                    "resolvable through `smap`."
                )

        # --- fields that only mean anything for the outermost graph ---------------------
        for field_name in ("prologue", "scalar_params", "else_nodes"):
            if not is_main:
                assert not getattr(graph, field_name, None), (
                    f"{where}: `{field_name}` is set, but it is only emitted for the main "
                    "graph -- a subgraph setting it would be silently ignored."
                )
        if not is_main:
            assert (
                getattr(graph, "rasterization", None) is None
            ), f"{where}: `rasterization` is only emitted for the main graph."
            for field_name in ("block_guard", "work_guard"):
                assert (
                    getattr(graph, field_name, None) is None
                ), f"{where}: `{field_name}` is only emitted for the main graph."

        # --- prefetched and windowed nodes stay part of the dataflow --------------------
        for field_name in ("prefetch_nodes", "window_nodes"):
            nodes = getattr(graph, field_name, ())
            assert not (nodes and is_main), (
                f"{where}: `{field_name}` only means something inside a loop, and the main "
                "graph's body runs once per grid slot."
            )
            for node in nodes:
                assert node in graph.nodes, (
                    f"{field_name} entry {node.name!r} is not in `nodes`. It is still an "
                    "ordinary node of the dataflow; only where its buffer lives changes."
                )
                assert isinstance(node, ComputeOpNode), (
                    f"{field_name} entry {node.name!r} is a {type(node).__name__}; only a "
                    "compute node has a read to move."
                )
                assert node not in graph.accum_nodes, (
                    f"{field_name} entry {node.name!r} accumulates across iterations, so "
                    "its buffer is carried, not refilled."
                )
        assert not (graph.window_nodes and graph.window_guard is None), (
            f"{where}: `window_nodes` without a `window_guard` would refill every "
            "iteration, which is what hoisting the buffer is meant to avoid."
        )
        for node in graph.prefetch_nodes:
            for sname in getattr(node, "scalar_args", ()):
                assert sname in graph.smap, (
                    f"prefetched node {node.name!r} reads scalar {sname!r}, which has no "
                    "`smap` entry, so its value one iteration ahead cannot be derived."
                )

        # --- the else branch has to be an alternative producer of the same tile ---------
        if graph.work_guard is not None or graph.else_nodes:
            assert graph.work_guard is not None, (
                "else_nodes need a work_guard to be the alternative to; without one they "
                "would run unconditionally alongside the body they are meant to replace."
            )
            out_tiles = {}
            if isinstance(tmap, dict):
                for arg in graph.args:
                    if isinstance(arg, OutputNode):
                        out_tiles.setdefault(graph.tinfo_map[arg][0], []).append(
                            tuple(tmap[arg].tile_size)
                        )
            for node in graph.else_nodes:
                assert not any(n is node for n in graph.nodes), (
                    f"else node {node.name!r} is also in `nodes`. It replaces part of the "
                    "dataflow, so it must not also be part of it."
                )
                assert not node.inputs, (
                    f"else node {node.name!r} has tensor inputs. The else branch runs "
                    "precisely when the body that would produce them is skipped, so it "
                    "can only depend on scalars."
                )
                assert (
                    node in graph.tinfo_map
                ), f"else node {node.name!r} has no tinfo_map entry."
                assert node in tmap, f"else node {node.name!r} has no tmap entry."
                buf = graph.tinfo_map[node][0]
                assert buf in out_tiles, (
                    f"else node {node.name!r} writes {buf!r}, which is not one of this "
                    f"graph's outputs ({sorted(out_tiles)}). The else branch may only "
                    "produce declared outputs."
                )
                tile = tuple(tmap[node].tile_size)
                shapes = out_tiles[buf]
                assert tile in shapes, (
                    f"else node {node.name!r} writes a {tile} tile of {buf!r}, but the "
                    f"body writes {shapes}. The two paths must cover the same tile, or "
                    "the outputs disagree depending on which branch ran."
                )
                for name in getattr(node, "scalar_args", ()):
                    assert name in graph.smap, (
                        f"else node {node.name!r} reads scalar {name!r}, which has no "
                        "smap expression."
                    )
            # Coverage, which is the one way this mechanism can silently produce a wrong
            # answer: an output the body writes but the else branch does not is simply
            # never written for the slots that take the else branch, and no other slot
            # owns that region.
            filled = {graph.tinfo_map[n][0] for n in graph.else_nodes}
            # An empty else branch is included deliberately: `work_guard` with no fills
            # just skips the body, which is what `block_guard` already means, and leaves
            # every output unwritten for the slots it skips.
            assert filled == set(out_tiles), (
                "the else branch writes "
                f"{sorted(filled)} but the body writes {sorted(out_tiles)}. Every output "
                "needs a producer on both paths, or it is left uninitialised whenever the "
                "guard is false."
            )

        for node in graph.nodes:
            if isinstance(node, GraphNode):
                self._validate_graph(node.graph, is_main=False, seen=seen)

    def _declare_pass_configs(self, graph):
        """Emit the tilelang pass configs this kernel must be compiled under.

        They travel in the generated module rather than in the caller, because a kernel
        that only works with (say) TMA lowering off is not correct otherwise, and every
        caller -- tests, the benchmark harness -- has to agree.
        """
        graphs = graph if isinstance(graph, (list, tuple)) else [graph]
        merged = {}
        for g in graphs:
            for key, value in getattr(g, "pass_configs", {}).items():
                assert merged.get(key, value) == value, (
                    f"Split graphs disagree on pass config {key!r}: "
                    f"{merged[key]} vs {value}"
                )
                merged[key] = value

        # Two tilelang defaults changed between 0.1.5 and 0.1.13, and both cost cleave
        # roughly 24% on every operator. Neither is a scheduling preference -- they restore
        # what the compiler used to do -- so they are seeded for every kernel rather than
        # per schedule.
        #
        # `enable_fast_math`: 0.1.5 passed `--use_fast_math` to ptxas unconditionally; 0.1.13
        # defaults it off. Without it the accurate libdevice path is called instead of the
        # inline approximations. This restores 0.1.5's numerics exactly, so it does not lean
        # on the 1e-2 tolerance these operators are checked at.
        #
        # `ptxas_register_usage_level`: new key in 0.1.13. At ptxas's own default budget
        # (100 registers here) the predicated `exp2` in the masked softmax --
        # `if_then_else(mask, exp2(x * s), 0)` -- costs a few extra live registers and ptxas
        # stops keeping the global loads in flight: measured `stall long_scoreboard` +30% and
        # DRAM throughput 17.5% -> 14.0% of peak at IDENTICAL DRAM bytes, MMA count, shared
        # traffic and occupancy. Level 10 raises the budget to 118 registers and recovers it.
        # The construct is what matters: 0.1.13 is FASTER than 0.1.5 on the same kernel with
        # either the mask or the exp2 removed, and slower only with both together.
        merged.setdefault("tl.enable_fast_math", True)
        merged.setdefault("tl.ptxas_register_usage_level", 10)

        if not merged:
            return
        entries = "".join(f'    "{k}": {v!r},\n' for k, v in sorted(merged.items()))
        self.pass_code = f"PASS_CONFIGS = {{\n{entries}}}\n"

    def _declare_dyn_dims(self, graph):
        """Emit ``name = T.symbolic("name")`` for every symbolic extent in the graph.

        A symbolic extent must appear bare in at least one tensor shape so the runtime
        can infer it from the bound tensor; once inferred it may be used inside extent
        and grid expressions, and is in scope inside the emitted macros.
        """
        roots = list(graph) if isinstance(graph, (list, tuple)) else [graph]
        names, pending = [], list(roots)
        while pending:
            g = pending.pop(0)
            for name in getattr(g, "dyn_dims", ()):
                if name not in names:
                    names.append(name)
            pending.extend(n.graph for n in g.nodes if isinstance(n, GraphNode))
        for name in names:
            assert any(
                name in tuple(g.tmap[arg].shape)
                for g in roots
                for arg in g.args
                if isinstance(arg, PlaceHolderNode) and arg in g.tmap
            ), (
                f"symbolic extent {name!r} never appears bare in an input tensor shape, "
                "so the runtime has nothing to infer it from"
            )
            self.symbol_code += f'{name} = T.symbolic("{name}")\n'

    def emit(self, code: str):
        self.graph_code += code

    def emit_op_node(
        self,
        node: Node,
        mem_scope: str,
        clear: bool,
        name_prefix: str = "",
    ):
        assert isinstance(
            node, (ComputeOpNode, MacroOp)
        ), f"Only ComputeOpNode or MacroOp can generate prim, but got {node=}"
        macro_name = node.get_macro_name(mem_scope=mem_scope, clear=clear)
        qualified_name = self._qualify_name(macro_name, name_prefix)
        macro = node.gen_macro(mem_scope=mem_scope, clear=clear)
        macro = self._replace_func_name(macro, macro_name, qualified_name)
        emitted = self.generated_macros.get(qualified_name)
        if emitted is None:
            self.generated_macros[qualified_name] = macro
            self.prim_code += macro
        else:
            assert emitted == macro, (
                f"Two ops emit different bodies under the macro name {qualified_name!r}; "
                "the second would shadow the first. get_macro_name must encode every "
                "attribute that changes the body."
            )

    def format_code(self):
        util_code = UTIL_CODE.replace("DEBUG = False", f"DEBUG = {self.debug}")
        full_code = (
            IMPORT_CODE
            + util_code
            + self.pass_code
            + self.symbol_code
            + self.prim_code
            + self.graph_code
        )
        mode = black.Mode(target_versions={black.TargetVersion.PY311}, line_length=100)
        formatted_code = black.format_str(full_code, mode=mode)
        return formatted_code
