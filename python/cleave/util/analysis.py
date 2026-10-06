from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from tvm import tirx

from ..graph import Graph, OutputNode
from ..ops import Add, Div, Exp, Matmul, Mul, RSqrt, Reduction, Silu, SplitView, Sqrt, Sub


def expr_uses_var(expr, var) -> bool:
    if expr is None:
        return False
    found = False

    def _visitor(node):
        nonlocal found
        if found:
            return
        if isinstance(node, tirx.Var) and (
            node.same_as(var) or (node.name == var.name and node.dtype == var.dtype)
        ):
            found = True

    tirx.stmt_functor.post_order_visit(expr, _visitor)
    return found


def _identity_axis_map(
    producer_rank: int, consumer_rank: Optional[int] = None
) -> List[Set[int]]:
    if consumer_rank is None:
        return [{axis} for axis in range(producer_rank)]
    return [{axis} if axis < consumer_rank else set() for axis in range(producer_rank)]


def _map_input_axes_to_consumer_axes(
    consumer,
    input_id: int,
    producer_rank: int,
) -> List[Set[int]]:
    consumer_rank = len(consumer.get_shape()) if hasattr(consumer, "get_shape") else 0

    if isinstance(consumer, OutputNode):
        return _identity_axis_map(producer_rank, consumer_rank=consumer_rank)

    if isinstance(consumer, Matmul):
        n_batch = consumer.n_batch_dims
        axis_map = [set() for _ in range(producer_rank)]
        producer_batch_rank = max(0, producer_rank - 2)
        input_batch_dims = (
            consumer.input_batch_dims_A
            if input_id == 0
            else consumer.input_batch_dims_B
        )
        consumer_batch_axes = []
        remaining_producer_axes = producer_batch_rank
        for axis, input_extent in enumerate(input_batch_dims):
            remaining_candidate_axes = n_batch - axis
            if remaining_candidate_axes <= remaining_producer_axes or int(input_extent) != 1:
                consumer_batch_axes.append(axis)
                remaining_producer_axes -= 1
            if remaining_producer_axes == 0:
                break

        if len(consumer_batch_axes) != producer_batch_rank:
            batch_axis_offset = max(0, n_batch - producer_batch_rank)
            consumer_batch_axes = list(
                range(batch_axis_offset, batch_axis_offset + producer_batch_rank)
            )

        for producer_batch_axis, output_batch_axis in enumerate(consumer_batch_axes):
            if output_batch_axis < n_batch:
                axis_map[producer_batch_axis].add(output_batch_axis)
        if input_id == 0:
            m_axis = producer_rank - (1 if consumer.transpose_A else 2)
            if 0 <= m_axis < producer_rank:
                axis_map[m_axis].add(n_batch)
        elif input_id == 1:
            n_axis = producer_rank - (2 if consumer.transpose_B else 1)
            if 0 <= n_axis < producer_rank:
                axis_map[n_axis].add(n_batch + 1)
        return axis_map

    if isinstance(consumer, SplitView):
        axis_map = [set() for _ in range(producer_rank)]
        for producer_axis in range(producer_rank):
            if producer_axis == consumer.axis:
                if 0 <= consumer.split_pos < consumer_rank:
                    axis_map[producer_axis].add(consumer.split_pos)
                if 0 <= consumer.inner_pos < consumer_rank:
                    axis_map[producer_axis].add(consumer.inner_pos)
                continue

            consumer_axis = producer_axis + (1 if consumer.split_pos <= producer_axis else 0)
            if 0 <= consumer_axis < consumer_rank:
                axis_map[producer_axis].add(consumer_axis)
        return axis_map

    if isinstance(consumer, Reduction):
        axis_map = [set() for _ in range(producer_rank)]
        for producer_axis in range(producer_rank):
            if producer_axis == consumer.axis:
                continue
            consumer_axis = (
                producer_axis if producer_axis < consumer.axis else producer_axis - 1
            )
            if 0 <= consumer_axis < consumer_rank:
                axis_map[producer_axis].add(consumer_axis)
        return axis_map

    if isinstance(consumer, (Add, Sub, Mul, Div)):
        if input_id == 0 or getattr(consumer, "const", None) is not None:
            return _identity_axis_map(producer_rank, consumer_rank=consumer_rank)
        if consumer.broadcast == -1:
            return _identity_axis_map(producer_rank, consumer_rank=consumer_rank)

        n_batch = consumer.n_batch_dims
        axis_map = [set() for _ in range(producer_rank)]
        for batch_axis in range(min(n_batch, producer_rank)):
            axis_map[batch_axis].add(batch_axis)
        if producer_rank > n_batch:
            consumer_axis = n_batch + consumer.broadcast
            if 0 <= consumer_axis < consumer_rank:
                axis_map[producer_rank - 1].add(consumer_axis)
        return axis_map

    if isinstance(consumer, (Exp, RSqrt, Sqrt, Silu)):
        return _identity_axis_map(producer_rank, consumer_rank=consumer_rank)

    return _identity_axis_map(producer_rank, consumer_rank=consumer_rank)


def infer_output_axis_dependencies(output_node: OutputNode) -> Dict[int, List[Set[int]]]:
    if len(output_node.inputs) == 0:
        return {}
    output_src = output_node.inputs[0].src_node
    output_rank = len(output_src.get_shape(output_node.inputs[0].src_id))
    output_identity = [{axis} for axis in range(output_rank)]

    deps: Dict[int, List[Set[int]]] = {
        output_node.name: [set(s) for s in output_identity],
        output_src.name: [set(s) for s in output_identity],
    }
    worklist = [output_src]

    while worklist:
        consumer = worklist.pop()
        consumer_deps = deps.get(consumer.name)
        if consumer_deps is None:
            continue

        for input_edge in consumer.inputs:
            producer = input_edge.src_node
            producer_rank = len(producer.get_shape(input_edge.src_id))
            axis_map = _map_input_axes_to_consumer_axes(
                consumer, input_edge.dst_id, producer_rank
            )
            propagated = [set() for _ in range(producer_rank)]

            for producer_axis, consumer_axes in enumerate(axis_map):
                for consumer_axis in consumer_axes:
                    if 0 <= consumer_axis < len(consumer_deps):
                        propagated[producer_axis].update(consumer_deps[consumer_axis])

            existing = deps.get(producer.name)
            if existing is None or len(existing) != producer_rank:
                deps[producer.name] = propagated
                worklist.append(producer)
                continue

            changed = False
            for axis, propagated_axes in enumerate(propagated):
                prev_len = len(existing[axis])
                existing[axis].update(propagated_axes)
                if len(existing[axis]) != prev_len:
                    changed = True
            if changed:
                worklist.append(producer)

    return deps


def get_output_axis_tokens(
    provenance: Dict[Tuple[object, int], List[Set[Tuple[int, int, int]]]],
    node,
    output_id: int,
    dim_id: int,
):
    token_map = provenance.get((node, output_id))
    if token_map is None or dim_id < 0 or dim_id >= len(token_map):
        return set()
    return token_map[dim_id]


def get_placeholder_axis_token(node, output_id: int, dim_id: int):
    return (id(node), output_id, dim_id)


def match_statement(node, block):
    if block is None:
        return None
    block_name = node.block_analyzer.get_block_name(block)
    for dep in getattr(node.ana, "deps", []):
        if node.block_analyzer.get_block_name(dep.block) == block_name:
            return dep
    return None


def build_output_axis_provenance(ordered_nodes):
    provenance = {}
    for node in ordered_nodes:
        for edge in node.inputs:
            key = (edge.src_node, edge.src_id)
            if key in provenance:
                continue
            if not edge.src_node.is_placeholder():
                continue
            src_ndim = len(edge.src_node.get_shape(edge.src_id))
            provenance[key] = [
                {
                    get_placeholder_axis_token(
                        edge.src_node, edge.src_id, dim_id
                    )
                }
                for dim_id in range(src_ndim)
            ]

        if len(node.output_buffers) == 0:
            continue

        output_id = 0
        out_shape = node.get_shape(output_id)
        out_tokens = [set() for _ in out_shape]

        target_block = (
            node.reduction_block
            if node.reduction_block is not None
            else node.schedule_stages[0]
        )
        block_info = node.block_analyzer.get_block_info(target_block)
        spatial_iters = [it for it in block_info.iters if it.kind == "S"]

        input_buffers = node.block_analyzer.get_input_buffers(target_block)
        output_buffer = node.block_analyzer.get_output_buffers(target_block)[0]
        if spatial_iters:
            out_axis_vars = [it.var for it in spatial_iters]
            out_dim_offset = len(out_shape) - len(out_axis_vars)
            try:
                input_expr_map = node.ana.get_input_exprs(
                    {output_buffer.name: out_axis_vars}
                )
            except Exception:
                input_expr_map = {}

            for axis_id, axis_var in enumerate(out_axis_vars):
                out_dim_id = out_dim_offset + axis_id
                if out_dim_id < 0 or out_dim_id >= len(out_tokens):
                    continue
                axis_tokens = set(out_tokens[out_dim_id])
                for edge in node.inputs:
                    if edge.dst_id >= len(input_buffers):
                        continue
                    input_buffer = input_buffers[edge.dst_id]
                    input_expr = input_expr_map.get(input_buffer.name)
                    if input_expr is None:
                        continue
                    src_shape = edge.src_node.get_shape(edge.src_id)
                    dim_offset = len(src_shape) - len(input_expr)
                    for local_dim, expr in enumerate(input_expr):
                        src_dim = dim_offset + local_dim
                        if src_dim < 0 or src_dim >= len(src_shape):
                            continue
                        if expr_uses_var(expr, axis_var):
                            axis_tokens.update(
                                get_output_axis_tokens(
                                    provenance, edge.src_node, edge.src_id, src_dim
                                )
                            )
                out_tokens[out_dim_id] = axis_tokens

        provenance[(node, output_id)] = out_tokens

        for extra_output_id in range(1, node.num_outputs()):
            if (node, extra_output_id) in provenance:
                continue
            out_shape = node.get_shape(extra_output_id)
            provenance[(node, extra_output_id)] = [
                set() for _ in range(len(out_shape))
            ]

    return provenance


def build_reduction_axis_provenance(ordered_nodes, output_axis_provenance):
    reduction_axis_provenance = {}
    for node in ordered_nodes:
        if node.reduction_block is None or not node.raxis:
            continue

        stmt = match_statement(node, node.reduction_block)
        axis_tokens = [set() for _ in node.raxis]
        if stmt is None:
            reduction_axis_provenance[node] = axis_tokens
            continue

        reduction_input_buffers = node.block_analyzer.get_input_buffers(
            node.reduction_block
        )
        for ax_id, raxis in enumerate(node.raxis):
            for edge in node.inputs:
                buffer_name = reduction_input_buffers[edge.dst_id].name
                input_regions = stmt.dependent_region.get(buffer_name, [])
                if not input_regions:
                    continue
                src_shape = edge.src_node.get_shape(edge.src_id)
                for region in input_regions:
                    dim_offset = len(src_shape) - len(region)
                    for local_dim, expr in enumerate(region):
                        src_dim = dim_offset + local_dim
                        if src_dim < 0 or src_dim >= len(src_shape):
                            continue
                        if expr_uses_var(expr, raxis.var):
                            axis_tokens[ax_id].update(
                                get_output_axis_tokens(
                                    output_axis_provenance,
                                    edge.src_node,
                                    edge.src_id,
                                    src_dim,
                                )
                            )
        reduction_axis_provenance[node] = axis_tokens

    return reduction_axis_provenance
