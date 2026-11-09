from ..ir.graph import AccumEdge, ComputeOpNode, GraphNode
from .tensor_codegen import gen_copy, gen_tensor_allocation_str
from .buffer import CodeBuffer


def kept_extents(dims):
    """Extents with unit dims squeezed out. Symbolic extents are always kept."""
    return [d for d in dims if not isinstance(d, int) or d > 1]


def tc_layout(graph, node) -> bool:
    return graph.tlayout_map.get(node, "") == "tc"


def emit_alloc(buf: CodeBuffer, graph, node, shape, dtype, scope=None) -> None:
    """Allocate ``node``'s tile buffer under its ``tinfo_map`` name."""
    tname, node_scope = graph.tinfo_map[node]
    buf.block(
        gen_tensor_allocation_str(
            tname,
            shape,
            dtype,
            scope=scope or node_scope,
            is_tc_layout=tc_layout(graph, node),
        )
    )


def emit_staged_copy(
    buf: CodeBuffer, src: str, dst: str, shape, dtype, scope, tc
) -> None:
    """Allocate ``dst`` and copy ``src`` into it across levels."""
    buf.block(
        gen_tensor_allocation_str(dst, shape, dtype, scope=scope, is_tc_layout=tc)
    )
    buf.block(gen_copy(src=src, dst=dst, same_level=False))


class OperandResolver:
    def _resolve_operand(self, buf: CodeBuffer, graph, node, input_idx, edge, staged):
        """The (buffer name, scope) a compute node reads input ``input_idx`` from.

        Emits the dtype cast and the fragment-to-shared staging the operand needs first.
        """
        src = edge.src_node
        if isinstance(src, GraphNode):
            assert isinstance(
                edge, AccumEdge
            ), f"Expect accumulation edge, but got {edge=}"
            assert isinstance(
                edge.accum_node, ComputeOpNode
            ), f"Accumulation node must be ComputeOpNode, but got {edge.accum_node=}"
            in_dtype = edge.accum_node.out_dtype
            tname, scope = src.graph.tinfo_map[edge.accum_node]
            shape = kept_extents(src.graph.tmap[edge.accum_node].tile_size)
        else:
            in_dtype = (
                src.out_dtype
                if isinstance(src, ComputeOpNode)
                else str(src.get_dtype(0))
            )
            tname, scope = graph.tinfo_map[src]
            shape = kept_extents(graph.tmap[src].tile_size)

        stage = input_idx in staged
        if in_dtype != node.dtype and input_idx not in getattr(
            node, "index_inputs", ()
        ):
            cast, scope = tname + "_cast", "s" if stage else "r"
            emit_staged_copy(buf, tname, cast, shape, node.dtype, scope, tc=stage)
            tname = cast
        if stage and scope == "r":
            dtype = node.dtype if in_dtype != node.dtype else in_dtype
            emit_staged_copy(
                buf, tname, f"{tname}_intermediate", shape, dtype, "s", tc=True
            )
            tname = f"{tname}_intermediate"
        return tname, scope

    def _input_operands(self, graph, node) -> tuple[list[str], str]:
        """Names and concatenated scopes of ``node``'s inputs, as ``tinfo_map`` has them."""
        names, mem_scope = [], ""
        for edge in node.inputs:
            tname, scope = graph.tinfo_map[edge.src_node]
            names.append(tname)
            mem_scope += scope
        return names, mem_scope

    def _emit_producer_call(self, graph, producer, dst_name, dst_scope, name_prefix=""):
        """Macro call that fills ``dst_name`` from ``producer``'s operands.

        Operands are the producer's tensor inputs (inner placeholders whose outer binding
        the enclosing macro received), then the op's scalar arguments.
        """
        operand_names = list(getattr(producer, "source_args", ()))
        if not operand_names:
            operand_names = [
                graph.binding.get(e.src_node.name, e.src_node).name
                for e in sorted(producer.inputs, key=lambda e: e.dst_id)
            ]
        mem_scope = "g" * len(operand_names) + dst_scope
        func_name = self.emit_op_node(
            producer, mem_scope=mem_scope, clear=False, name_prefix=name_prefix
        )
        call = operand_names + list(getattr(producer, "scalar_args", ())) + [dst_name]
        return f"{func_name}({', '.join(call)})\n"

    def _emit_window_fill(self, graph, node, name_prefix=""):
        """Macro call refilling a windowed node's hoisted buffer (a plain gather)."""
        call_args, mem_scope = self._input_operands(graph, node)
        tname, scope = graph.tinfo_map[node]
        assert (
            node not in graph.accum_nodes
        ), f"{node.name} is windowed, so its buffer is refilled, not accumulated into."
        func_name = self.emit_op_node(
            node, mem_scope=mem_scope + scope, clear=True, name_prefix=name_prefix
        )
        call = call_args + list(getattr(node, "scalar_args", ())) + [tname]
        return f"{func_name}({', '.join(call)})\n"

    @staticmethod
    def _prefetch_buffer_name(graph, node) -> str:
        return graph.tinfo_map[node][0] + "_pf"

    @staticmethod
    def _prefetch_scalars(graph, node, it: str):
        """``(param_name, value_at_i, value_at_i_plus_1 or None if unchanged)`` per scalar."""
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
        """``(shifted, shared)``: scalars the look-ahead needs one iteration on (passed
        as ``<name>_next``), and the subset the body still reads for the current tile.
        """
        shifted, readers = set(), set()
        for node in graph.prefetch_nodes:
            for sname, _, ahead in self._prefetch_scalars(graph, node, "i0"):
                if ahead is not None:
                    shifted.add(sname)
            # A prefetch off a windowed operand is re-seeded at the current iteration.
            if any(e.src_node in graph.window_nodes for e in node.inputs):
                readers.update(getattr(node, "scalar_args", ()))
        for node in graph.nodes:
            if node in graph.prefetch_nodes:
                continue
            readers.update(getattr(node, "scalar_args", ()))
        return shifted, shifted & readers

    def _emit_prefetch_read(self, graph, node, scalars, name_prefix=""):
        """Macro call issuing ``node``'s global read into its prefetch fragment."""
        call_args, mem_scope = self._input_operands(graph, node)
        mem_scope += "r"
        for edge, scope in zip(node.inputs, mem_scope):
            # Only a windowed shared operand holds still across the look-ahead; any other
            # is rewritten every iteration behind a barrier the read would be hoisted past.
            assert scope != "s" or edge.src_node in graph.window_nodes, (
                f"Prefetched node {node.name} reads a shared operand ({mem_scope}) that is "
                "not windowed; its read cannot be hoisted past that operand's barrier."
            )
        func_name = self.emit_op_node(
            node, mem_scope=mem_scope, clear=False, name_prefix=name_prefix
        )
        call = call_args + list(scalars) + [self._prefetch_buffer_name(graph, node)]
        return f"{func_name}({', '.join(call)})\n"
