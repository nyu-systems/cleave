from collections import Counter
from dataclasses import dataclass
from math import prod
from typing import Dict, List, Optional, Set
from tilelang.carver.roller.node import Edge
from .graph import (
    Graph,
    OutputNode,
    GraphNode,
    PlaceHolderNode,
    Iterator,
    ViewOpNode,
)
from .policy.default import DefaultPolicy, TCPolicy
from .ops import Matmul, Reduction
from .util.analysis import (
    infer_output_axis_dependencies,
)
from .arch import get_arch
from .util import (
    get_op_ancestors,
    get_offset_lambda,
    gather_all_nodes,
    find_topo_sort,
    clone_graph,
    lock_graph,
    unlock_graph,
)


def _get_placeholder_iter_repeat_factors(
    node,
    shape,
    tile_size,
    threadblock_extent,
    node_axis_deps: Optional[List[Set[int]]] = None,
    output_threadblock_extent: Optional[tuple[int, ...]] = None,
):
    iter_repeat_factors = [1] * len(shape)
    if not isinstance(node, PlaceHolderNode):
        return iter_repeat_factors
    if (
        len(shape) != len(tile_size)
        or len(shape) != len(threadblock_extent)
        or not hasattr(node, "is_batch_dim")
        or node.is_batch_dim is None
        or len(node.is_batch_dim) != len(shape)
    ):
        return iter_repeat_factors

    for dim_id, is_batch_dim in enumerate(node.is_batch_dim):
        if not is_batch_dim:
            continue
        if tile_size[dim_id] == 0:
            continue
        src_extent = shape[dim_id] // tile_size[dim_id]
        dst_extent = None
        if (
            node_axis_deps is not None
            and output_threadblock_extent is not None
            and dim_id < len(node_axis_deps)
        ):
            dep_extents = {
                output_threadblock_extent[axis]
                for axis in node_axis_deps[dim_id]
                if 0 <= axis < len(output_threadblock_extent)
            }
            if len(dep_extents) == 1:
                dst_extent = dep_extents.pop()
        if dst_extent is None:
            dst_extent = threadblock_extent[dim_id]
        if src_extent <= 0 or dst_extent % src_extent != 0:
            continue
        repeat_factor = dst_extent // src_extent
        if repeat_factor <= 1:
            continue

        iter_repeat_factors[dim_id] = repeat_factor

    return iter_repeat_factors


def _get_schedule_shape_owner(node):
    if not node.is_output():
        return node
    src_node = node.inputs[0].src_node
    return node if isinstance(src_node, GraphNode) else src_node


def _expand_tile_to_shape_rank(node, tile_size):
    shape_owner = _get_schedule_shape_owner(node)
    raw_shape = list(shape_owner.get_shape())
    tile_size = list(tile_size)
    if len(tile_size) == len(raw_shape):
        return tile_size

    batch_flags = list(getattr(shape_owner, "is_batch_dim", []) or [])
    if len(batch_flags) != len(raw_shape):
        return tile_size

    n_non_batch_dims = sum(1 for is_batch_dim in batch_flags if not is_batch_dim)
    visible_batch_dims = len(tile_size) - n_non_batch_dims
    if visible_batch_dims < 0:
        return tile_size

    collapsed_batch_tiles = tile_size[:visible_batch_dims]
    if any(tile != 1 for tile in collapsed_batch_tiles):
        raise NotImplementedError(
            "Collapsed batch tiles greater than 1 are not supported."
        )

    expanded_tile = []
    non_batch_idx = visible_batch_dims
    for is_batch_dim in batch_flags:
        if is_batch_dim:
            expanded_tile.append(1)
            continue
        if non_batch_idx >= len(tile_size):
            return tile_size
        expanded_tile.append(tile_size[non_batch_idx])
        non_batch_idx += 1

    return expanded_tile


def _get_batch_axes_and_extents(node):
    shape_owner = _get_schedule_shape_owner(node)
    batch_flags = list(getattr(shape_owner, "is_batch_dim", []) or [])
    shape = list(shape_owner.get_shape())
    if len(batch_flags) != len(shape):
        return []
    return [
        (axis, int(extent))
        for axis, (extent, is_batch_dim) in enumerate(zip(shape, batch_flags))
        if is_batch_dim
    ]


@dataclass(frozen=True)
class _OutputAxisCoordExpr:
    expr: str
    omit_unit_scale: bool = False


@dataclass(frozen=True)
class _LaunchAxisPlan:
    extent: tuple[int, ...]
    axis_coord_exprs: Dict[int, _OutputAxisCoordExpr]


def _get_output_batch_dim_flags(node, rank: int) -> List[bool]:
    shape_owner = _get_schedule_shape_owner(node)
    batch_flags = list(getattr(shape_owner, "is_batch_dim", []) or [])
    if len(batch_flags) != rank:
        return [False] * rank
    return [bool(flag) for flag in batch_flags]


def _get_output_binding_node(output_node):
    src_node = output_node.inputs[0].src_node
    if isinstance(src_node, GraphNode):
        bound_node, _, _ = src_node._resolve_output_binding(output_node.inputs[0].src_id)
        return bound_node
    return src_node


def _select_canonical_output_node(output_nodes):
    if not output_nodes:
        raise ValueError("Expected at least one output node.")

    canonical_output = output_nodes[0]
    canonical_rank = len(_get_output_binding_node(canonical_output).get_shape())
    for output_node in output_nodes[1:]:
        output_rank = len(_get_output_binding_node(output_node).get_shape())
        if output_rank > canonical_rank:
            canonical_output = output_node
            canonical_rank = output_rank
    return canonical_output


def _get_rstep_for_axis(node, rstep_map: dict, axis_id: int = 0):
    axis_name = node.raxis[axis_id].var.name
    if axis_name in rstep_map:
        return rstep_map[axis_name]

    if len(node.raxis) == 1 and len(rstep_map) == 1:
        return next(iter(rstep_map.values()))

    values = list(rstep_map.values())
    if len(values) == len(node.raxis):
        return values[axis_id]

    raise KeyError(
        f"Unable to resolve reduction step for axis {axis_name} on node {node.name}."
    )


def _get_same_launch_output_axis_map(canonical_node, output_node):
    canonical_shape = list(canonical_node.get_shape())
    output_shape = list(output_node.get_shape())
    canonical_batch_flags = list(getattr(canonical_node, "is_batch_dim", []) or [])
    output_batch_flags = list(getattr(output_node, "is_batch_dim", []) or [])
    if len(canonical_batch_flags) != len(canonical_shape) or len(output_batch_flags) != len(
        output_shape
    ):
        raise NotImplementedError(
            "Same-launch multi-output requires explicit batch flags for output mapping."
        )

    canonical_batch_axes = [
        axis
        for axis, is_batch_dim in enumerate(canonical_batch_flags)
        if is_batch_dim
    ]
    output_batch_axes = [
        axis for axis, is_batch_dim in enumerate(output_batch_flags) if is_batch_dim
    ]
    if len(canonical_batch_axes) != len(output_batch_axes):
        raise NotImplementedError(
            "Same-launch multi-output requires matching batch-axis rank."
        )

    axis_map = [None] * len(output_shape)
    for output_axis, canonical_axis in zip(output_batch_axes, canonical_batch_axes):
        if int(output_shape[output_axis]) != int(canonical_shape[canonical_axis]):
            raise NotImplementedError(
                "Same-launch multi-output requires matching batch extents in order."
            )
        axis_map[output_axis] = canonical_axis

    canonical_non_batch_axes = [
        axis
        for axis, is_batch_dim in enumerate(canonical_batch_flags)
        if not is_batch_dim
    ]
    output_non_batch_axes = [
        axis
        for axis, is_batch_dim in enumerate(output_batch_flags)
        if not is_batch_dim
    ]
    if len(output_non_batch_axes) > len(canonical_non_batch_axes):
        raise NotImplementedError(
            "Sibling outputs cannot have more non-batch axes than the canonical output."
        )

    for output_axis, canonical_axis in zip(
        output_non_batch_axes, canonical_non_batch_axes
    ):
        if int(output_shape[output_axis]) != int(canonical_shape[canonical_axis]):
            raise NotImplementedError(
                "Sibling outputs must drop only trailing non-batch axes from the canonical output."
            )
        axis_map[output_axis] = canonical_axis

    if any(axis is None for axis in axis_map):
        raise NotImplementedError(
            "Unable to map sibling output axes onto the canonical launch axes."
        )
    return axis_map


def _build_group_axis_coord_expr(
    arg_name: str, group_extents: List[int], axis_pos: int
) -> _OutputAxisCoordExpr:
    axis_extent = group_extents[axis_pos]
    if axis_extent <= 1:
        return _OutputAxisCoordExpr("0")

    trailing_extent = prod(group_extents[axis_pos + 1 :])
    leading_extent = prod(group_extents[:axis_pos])

    expr = arg_name if trailing_extent == 1 else f"({arg_name} // {trailing_extent})"
    if axis_pos > 0 and leading_extent > 1:
        expr = f"({expr} % {axis_extent})"

    # Keep inner flattened batch axes as direct coordinates when the tile size is 1.
    omit_unit_scale = axis_pos > 0 and leading_extent > 1
    return _OutputAxisCoordExpr(expr, omit_unit_scale=omit_unit_scale)


def _build_launch_axis_plan(
    output_node,
    output_threadblock_extent: tuple[int, ...],
) -> _LaunchAxisPlan:
    rank = len(output_threadblock_extent)
    batch_flags = _get_output_batch_dim_flags(output_node, rank)
    axis_coord_exprs = {axis: _OutputAxisCoordExpr("0") for axis in range(rank)}
    launch_groups: List[tuple[int, ...]] = []

    batch_axes = tuple(
        axis for axis, is_batch_dim in enumerate(batch_flags) if is_batch_dim
    )
    if batch_axes:
        batch_extent = prod(output_threadblock_extent[axis] for axis in batch_axes)
        if batch_extent > 1:
            launch_groups.append(batch_axes)

    for axis, extent in enumerate(output_threadblock_extent):
        if batch_flags[axis] or extent <= 1:
            continue
        launch_groups.append((axis,))

    launch_extent: List[int] = []
    for arg_idx, group_axes in enumerate(launch_groups):
        group_extents = [int(output_threadblock_extent[axis]) for axis in group_axes]
        launch_extent.append(prod(group_extents))
        arg_name = f"i{arg_idx}"
        for axis_pos, axis in enumerate(group_axes):
            axis_coord_exprs[axis] = _build_group_axis_coord_expr(
                arg_name, group_extents, axis_pos
            )

    return _LaunchAxisPlan(
        extent=tuple(launch_extent) if launch_extent else (1,),
        axis_coord_exprs=axis_coord_exprs,
    )


def _resolve_iter_coord_exprs(
    iter_dims: List[bool],
    node_axis_deps: Optional[List[Set[int]]],
    output_axis_coord_exprs: Dict[int, _OutputAxisCoordExpr],
    split_dim_flags: Optional[List[bool]] = None,
    output_split_axis: Optional[int] = None,
) -> List[Optional[_OutputAxisCoordExpr]]:
    coord_exprs: List[Optional[_OutputAxisCoordExpr]] = [None] * len(iter_dims)
    for dim_idx, is_iter in enumerate(iter_dims):
        if not is_iter:
            continue

        dep_axes = []
        if node_axis_deps is not None and dim_idx < len(node_axis_deps):
            dep_axes = sorted(
                axis
                for axis in node_axis_deps[dim_idx]
                if axis in output_axis_coord_exprs
            )

        coord_expr = None
        if len(dep_axes) == 1:
            coord_expr = output_axis_coord_exprs[dep_axes[0]]
        elif (
            split_dim_flags is not None
            and split_dim_flags[dim_idx]
            and output_split_axis is not None
            and output_split_axis in output_axis_coord_exprs
        ):
            coord_expr = output_axis_coord_exprs[output_split_axis]

        if coord_expr is None or coord_expr.expr == "0":
            raise ValueError(
                f"Expected exactly one launch-axis mapping for iter dim {dim_idx}, got {dep_axes}."
            )
        coord_exprs[dim_idx] = coord_expr

    return coord_exprs


def _build_offset_lambda_from_coord_exprs(
    tile_shape: List[int],
    coord_exprs: List[Optional[_OutputAxisCoordExpr]],
    num_args: int,
    iter_repeat_factors: Optional[List[int]] = None,
):
    assert len(coord_exprs) == len(
        tile_shape
    ), "coord_exprs must have the same length as tile_shape"
    assert num_args >= 1, "num_args must be positive"

    lambda_args = [f"i{i}" for i in range(num_args)]
    iter_repeat_factors = iter_repeat_factors or [1] * len(tile_shape)
    assert len(iter_repeat_factors) == len(
        tile_shape
    ), "iter_repeat_factors must have the same length as tile_shape"

    offset_strs = []
    for tile, coord_expr, repeat_factor in zip(
        tile_shape, coord_exprs, iter_repeat_factors
    ):
        if coord_expr is None or coord_expr.expr == "0":
            offset_strs.append('"0"')
            continue

        expr = coord_expr.expr
        if repeat_factor > 1:
            expr = f"({expr} // {repeat_factor})"

        if int(tile) == 1 and coord_expr.omit_unit_scale and repeat_factor == 1:
            offset_strs.append(f'f"{expr}"')
        else:
            offset_strs.append(f'f"{int(tile)} * {expr}"')

    lambda_body = f"lambda {', '.join(lambda_args)}: ({', '.join(offset_strs)},)"
    return eval(lambda_body)


def _extract_split_extent(
    # remove split ops and annotate producer and consumer nodes
    flat_graph: Graph,
) -> tuple[Optional[int], Graph]:
    pruned_graph = clone_graph(flat_graph)
    split_nodes = [
        node
        for node in pruned_graph.nodes
        if isinstance(node, ViewOpNode) and hasattr(node, "factor")
    ]
    if not split_nodes:
        return None, pruned_graph

    split_factors = {int(node.factor) for node in split_nodes}
    if len(split_factors) != 1:
        raise NotImplementedError(
            "Multiple split extents in the same main graph are not supported."
        )
    split_extent = split_factors.pop()
    if split_extent <= 1:
        split_extent = None

    def _infer_consumer_split_axis(split_node, consumer, default_axis: int) -> int:
        src_node = split_node.inputs[0].src_node
        src_batch_counts = Counter(
            extent for _, extent in _get_batch_axes_and_extents(src_node)
        )
        consumer_batch_axes = _get_batch_axes_and_extents(consumer)
        if not consumer_batch_axes:
            return default_axis

        candidate_axes = []
        remaining = src_batch_counts.copy()
        for axis, extent in consumer_batch_axes:
            if remaining[extent] > 0:
                remaining[extent] -= 1
            else:
                candidate_axes.append(axis)

        if len(candidate_axes) == 1:
            return candidate_axes[0]

        split_axes = [
            axis for axis, extent in consumer_batch_axes if extent == split_extent
        ]
        if len(split_axes) == 1:
            return split_axes[0]

        return default_axis

    def _propagate_split_axis_to_consumers(start_node, split_output_axis: int):
        stack = [start_node]
        visited = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            existing_axis = getattr(node, "_split_axis", None)
            if existing_axis is not None and existing_axis != split_output_axis:
                raise NotImplementedError(
                    f"Conflicting split axes for node {node.name}: {existing_axis} vs {split_output_axis}"
                )
            setattr(node, "_split_axis", split_output_axis)
            for edge in node.outputs:
                stack.append(edge.dst_node)

    for split_node in split_nodes:
        if len(split_node.inputs) != 1:
            raise NotImplementedError(
                f"Split view node {split_node.name} is expected to have exactly one input."
            )
        in_edge = split_node.inputs[0]
        src_node = in_edge.src_node
        src_id = in_edge.src_id

        split_axis = getattr(split_node, "axis", None)
        if isinstance(split_axis, int):
            # TODO: support producer propagation
            setattr(src_node, "_split_axis", split_axis)
        split_output_axis = getattr(split_node, "split_pos", None)
        if not isinstance(split_output_axis, int):
            split_output_axis = 0

        for out_edge in list(split_node.outputs):
            consumer_split_axis = _infer_consumer_split_axis(
                split_node, out_edge.dst_node, split_output_axis
            )
            _propagate_split_axis_to_consumers(out_edge.dst_node, consumer_split_axis)
            new_edge = Edge(src_node, out_edge.dst_node, src_id, out_edge.dst_id)
            src_node._out_edges.append(new_edge)
            out_edge.dst_node.set_inputs(out_edge.dst_id, new_edge)
            split_node._out_edges.remove(out_edge)

        if in_edge in src_node._out_edges:
            src_node._out_edges.remove(in_edge)
        split_node._in_edges = []
        split_node._out_edges = []

    output_nodes = [node for node in pruned_graph.nodes if isinstance(node, OutputNode)]
    pruned_graph.nodes = find_topo_sort(output_nodes)
    return split_extent, pruned_graph


def _get_split_dim_flags(
    node, shape: list, split_extent: Optional[int]
) -> Optional[List[bool]]:
    if split_extent is None:
        return None

    split_dim_flags = [False] * len(shape)

    explicit_split_axis = getattr(node, "_split_axis", None)
    if isinstance(explicit_split_axis, int) and 0 <= explicit_split_axis < len(shape):
        split_dim_flags[explicit_split_axis] = True

    return split_dim_flags


def search_tiling(
    main_graph: Graph, topk: int, force_fusion: bool = False
) -> List[Graph]:
    arch = get_arch("cuda")  # TODO: make arch configurable
    is_tensorcore = False
    output_nodes = []
    for node in main_graph.nodes:
        if isinstance(node, Matmul):
            is_tensorcore = True
            node.set_tag("tensorcore_config", True)
        elif isinstance(node, OutputNode):
            output_nodes.append(node)
    policy = (
        TCPolicy(output_nodes, arch, force_fusion=force_fusion, flat_graph=main_graph)
        if is_tensorcore
        else DefaultPolicy(
            output_nodes, arch, force_fusion=force_fusion, flat_graph=main_graph
        )
    )
    candidates = policy.emit_config(topk)
    return candidates


def assign_tensor_map(
    graph_node: GraphNode,
    tile_map: dict,
    itile_map: dict,
    node_to_graph: dict,
    graphs: dict,
    top_bind_node: dict,
    parent_graph_name: str,
):
    assert not hasattr(
        graph_node.graph, "n_threads"
    ), "Inner graphs must not define n_threads."
    for n in graph_node.graph.nodes:
        # remove nodes from the parent graph
        if node_to_graph.get(n.name, None) == parent_graph_name:
            graphs[parent_graph_name].nodes.remove(n)
        node_to_graph[n.name] = graph_node.name

        # set binding between inner and outer nodes
        if isinstance(n, PlaceHolderNode):
            outer_node = graph_node.graph.binding[n.name]
            top_bind_node[n.name] = top_bind_node[outer_node.name]
        bind_node = top_bind_node[n.name] if isinstance(n, PlaceHolderNode) else n

        # get tile shape and tile size from candidate
        shape = [
            t
            for i, t in enumerate(tile_map[bind_node.name])
            if not n.is_batch_dim[i] or itile_map[bind_node.name][i] > 1
        ]
        tile_size = [
            t
            for i, t in enumerate(itile_map[bind_node.name])
            if not n.is_batch_dim[i] or t > 1
        ]
        iter_dims = [i // j > 1 for i, j in zip(shape, tile_size)]
        num_args = (
            len(graph_node.graph.extent)
            if isinstance(graph_node.graph.extent, tuple)
            else 1
        )
        arg_indices = [0 if is_iter else None for is_iter in iter_dims]

        # set tmap for the inner graph
        graph_node.graph.tmap[n] = Iterator(
            get_offset_lambda(tile_size, arg_indices, num_args),
            shape=shape,
            tile_size=tile_size,
        )


def _normalize_inner_reduction_axes(graph_node: GraphNode, itile_map: dict):
    for node in graph_node.graph.nodes:
        if (
            not isinstance(node, Reduction)
            or len(node.inputs) != 1
            # Negative axes remain stable when leading batch dims are dropped.
            # Positive axes need to be rebased onto the visible tiled input rank.
            or getattr(node, "axis_arg", -1) < 0
        ):
            continue

        input_edge = node.inputs[0]
        src_node = input_edge.src_node
        bind_node = graph_node.graph.binding.get(src_node.name, src_node)
        full_tile = itile_map.get(bind_node.name)
        batch_flags = list(getattr(src_node, "is_batch_dim", []) or [])
        if full_tile is None or len(batch_flags) != len(full_tile):
            continue

        visible_axes = [
            axis
            for axis, (is_batch_dim, tile_extent) in enumerate(zip(batch_flags, full_tile))
            if not is_batch_dim or tile_extent > 1
        ]
        if node.axis not in visible_axes:
            continue

        visible_axis = visible_axes.index(node.axis)
        node.axis = visible_axis
        node.axis_arg = visible_axis


def apply_tiling(cand: List) -> List[Graph]:
    output_nodes = [n for n in cand.flat_graph.nodes if n.is_output()]
    output_node = _select_canonical_output_node(output_nodes)

    # this needs to be called before any graph modifications
    # to correctly infer output axis dependencies for the
    # original output node
    output_axis_deps = infer_output_axis_dependencies(output_node)

    # remove the split ops and extract the split extent if applicable
    split_extent, flat_graph = _extract_split_extent(cand.flat_graph)

    # extract flat graph info and initialize the main graph
    graphs = {"main": flat_graph}
    graphs["main"].n_threads = cand.n_threads
    ordered_nodes = [
        n for n in graphs["main"].nodes if not n.is_placeholder() and not n.is_output()
    ]
    output_nodes = [n for n in graphs["main"].nodes if n.is_output()]
    main_output_nodes = list(output_nodes)
    output_node = _select_canonical_output_node(output_nodes)
    node_to_graph, graph_to_node = {
        node.name: "main" for node in gather_all_nodes(*ordered_nodes)
    }, {}
    top_bind_node = {
        node.name: node
        for node in graphs["main"].nodes
        if isinstance(node, PlaceHolderNode)
    }

    # create tile_map and itile_map with string keys for
    # serialization (avoid mutation node issues)
    tile_map = {k.name: v for k, v in cand.tile_map.items()}
    itile_map = {k.name: v for k, v in cand.itile_map.items()}

    # set top level threadblock extent and tmap based on the
    # candidate tile and reduction step choices
    threadblock_extent_map = {}
    for node in graphs["main"].nodes:
        shape_owner = _get_schedule_shape_owner(node)
        tile = tile_map[
            _get_output_binding_node(node).name if node.is_output() else node.name
        ]
        expanded_tile = _expand_tile_to_shape_rank(node, tile)
        threadblock_extent_map[node.name] = tuple(
            i // j for i, j in zip(shape_owner.get_shape(), expanded_tile)
        )
    dshape = threadblock_extent_map[output_node.name]

    # get output split provenance
    output_split_dim_flags = _get_split_dim_flags(
        _get_output_binding_node(output_node), list(dshape), split_extent
    )
    output_split_axis = (
        output_split_dim_flags.index(True)
        if output_split_dim_flags is not None and any(output_split_dim_flags)
        else None
    )

    launch_axis_plan = _build_launch_axis_plan(output_node, dshape)
    graphs["main"].extent = launch_axis_plan.extent

    for node in reversed(ordered_nodes):
        if (
            len(node.raxis) > 0
            and node in cand.rstep_map
            and _get_rstep_for_axis(node, cand.rstep_map[node]) < node.raxis[0].dom.extent
        ):
            # check if we can merge with an existing inner graph
            found_compatible_graph = False
            for gnode in graphs["main"].nodes:
                if isinstance(gnode, GraphNode):
                    accum_node = gnode.graph.accum_nodes[0]  # assume one accum node
                    same_rstep = (
                        _get_rstep_for_axis(accum_node, cand.rstep_map[accum_node])
                        == _get_rstep_for_axis(node, cand.rstep_map[node])
                    )

                    # merge node and its ancestors into an existing graph when
                    # they already share inner nodes/placeholders, even if the
                    # reduction-step choice differs. In that case the existing
                    # inner graph extent wins and the incompatible candidate
                    # choice is ignored instead of failing the whole schedule.
                    lock_graph(gnode.graph)
                    new_nodes = gather_all_nodes(node, descending=False)
                    unlock_graph(gnode.graph)

                    top_bind_inner_nodes = [
                        top_bind_node[n.name]
                        for n in gnode.graph.nodes
                        if isinstance(n, PlaceHolderNode)
                    ]
                    shared_inner_nodes = [
                        n
                        for n in new_nodes
                        if not isinstance(n, PlaceHolderNode) and n in gnode.graph.nodes
                    ]
                    existing_placeholders = [
                        n
                        for n in new_nodes
                        if isinstance(n, PlaceHolderNode)
                        and top_bind_node[n.name] in top_bind_inner_nodes
                    ]

                    if (
                        not same_rstep
                        and not shared_inner_nodes
                        and not existing_placeholders
                    ):
                        continue

                    found_compatible_graph = True

                    new_nodes = [n for n in new_nodes if n not in gnode.graph.nodes]
                    new_nodes = [n for n in new_nodes if n not in existing_placeholders]

                    # reconnect existing placeholder nodes
                    # TODO merge this code with assign_placeholder_nodes
                    if existing_placeholders:
                        inner_placeholders_map = {}
                        for n in gnode.graph.nodes:
                            if isinstance(n, PlaceHolderNode):
                                inner_placeholders_map[top_bind_node[n.name].name] = n
                        for ph in existing_placeholders:
                            inner_node = inner_placeholders_map[
                                top_bind_node[ph.name].name
                            ]
                            for edge in list(ph.outputs):
                                if edge.dst_node in new_nodes:
                                    edge.src_node = inner_node
                                    inner_node._out_edges.append(edge)
                                    ph._out_edges.remove(edge)

                    # add new nodes to the graph
                    prev_nodes = list(gnode.graph.nodes)
                    gnode.graph.nodes.extend(new_nodes)
                    gnode._assign_placeholder_nodes(skip=prev_nodes)
                    gnode._assign_accum_nodes()
                    assign_tensor_map(
                        gnode,
                        tile_map,
                        itile_map,
                        node_to_graph,
                        graphs,
                        top_bind_node,
                        node_to_graph[gnode.name],
                    )
                    _normalize_inner_reduction_axes(gnode, itile_map)

                    break

            if found_compatible_graph:
                continue

            # only ancestor nodes will be moved to the inner graph
            ancestors = get_op_ancestors(node, ordered_nodes)
            inner_op_nodes = [node] + ancestors

            # create a new inner graph
            inner_graph_name = f"inner_graph_{node.name}"
            inner_graph = Graph(
                nodes=gather_all_nodes(*inner_op_nodes, descending=False)
            )
            raxis = node.raxis[0]
            node_rstep = _get_rstep_for_axis(node, cand.rstep_map[node])
            assert raxis.dom.extent % node_rstep == 0
            inner_graph.extent = int(
                raxis.dom.extent // node_rstep
            )
            graphs[inner_graph_name] = inner_graph

            # create a GraphNode to represent the inner graph in the parent graph
            graph_node = GraphNode(name=f"GraphNode_{node.name}", graph=inner_graph)
            graph_to_node[inner_graph_name] = graph_node

            # the new GraphNode belongs to the same graph as the current reduction node
            parent_graph_name = node_to_graph[node.name]
            graphs[parent_graph_name].nodes.append(graph_node)
            node_to_graph[graph_node.name] = parent_graph_name

            assign_tensor_map(
                graph_node,
                tile_map,
                itile_map,
                node_to_graph,
                graphs,
                top_bind_node,
                parent_graph_name,
            )
            _normalize_inner_reduction_axes(graph_node, itile_map)
    lock_graph(graphs["main"])  # graph structure is fixed after this point

    main_nodes = [
        n for n in graphs["main"].nodes if not n.is_placeholder() and not n.is_output()
    ]
    graphs["main"].nodes = find_topo_sort(main_output_nodes)
    canonical_output_binding = _get_output_binding_node(output_node)
    for node in graphs["main"].nodes:
        tile_size = []
        if isinstance(node, GraphNode):
            continue
        elif isinstance(node, OutputNode):
            src_node = node.inputs[0].src_node
            tile_size = (
                tile_map[_get_output_binding_node(node).name]
                if isinstance(src_node, GraphNode)
                else tile_map[src_node.name]
            )
        else:
            tile_size = tile_map[node.name]

        if isinstance(node, OutputNode) and isinstance(node.inputs[0].src_node, GraphNode):
            bound_node = _get_output_binding_node(node)
            logical_shape = list(bound_node.get_shape())
            logical_tile_size = list(tile_size)
            axis_map = _get_same_launch_output_axis_map(canonical_output_binding, bound_node)
            logical_coord_exprs = [
                launch_axis_plan.axis_coord_exprs[canonical_axis]
                for canonical_axis in axis_map
            ]

            physical_shape = logical_shape
            physical_tile_size = logical_tile_size
            physical_coord_exprs = logical_coord_exprs
            split_axis = getattr(bound_node, "_split_axis", None)
            if (
                len(main_output_nodes) > 1
                and split_extent is not None
                and isinstance(split_axis, int)
                and 0 <= split_axis < len(logical_shape)
            ):
                reorder = [split_axis] + [
                    axis for axis in range(len(logical_shape)) if axis != split_axis
                ]
                physical_shape = [logical_shape[axis] for axis in reorder]
                physical_tile_size = [logical_tile_size[axis] for axis in reorder]
                physical_coord_exprs = [logical_coord_exprs[axis] for axis in reorder]

            while len(physical_shape) < len(canonical_output_binding.get_shape()):
                physical_shape.append(1)
                physical_tile_size.append(1)
                physical_coord_exprs.append(None)

            num_args = (
                len(graphs["main"].extent)
                if isinstance(graphs["main"].extent, tuple)
                else 1
            )
            graphs["main"].tmap[node] = Iterator(
                _build_offset_lambda_from_coord_exprs(
                    physical_tile_size, physical_coord_exprs, num_args
                ),
                shape=tuple(physical_shape),
                tile_size=tuple(physical_tile_size),
            )
            continue

        tile_size = _expand_tile_to_shape_rank(node, tile_size)
        shape = _get_schedule_shape_owner(node).get_shape()
        iter_dims = [i // j > 1 for i, j in zip(shape, tile_size)]
        num_args = (
            len(graphs["main"].extent)
            if isinstance(graphs["main"].extent, tuple)
            else 1
        )

        split_dim_flags = _get_split_dim_flags(node, shape, split_extent)
        coord_exprs = _resolve_iter_coord_exprs(
            iter_dims=iter_dims,
            node_axis_deps=output_axis_deps.get(node.name),
            output_axis_coord_exprs=launch_axis_plan.axis_coord_exprs,
            split_dim_flags=split_dim_flags,
            output_split_axis=output_split_axis,
        )
        graphs["main"].tmap[node] = Iterator(
            _build_offset_lambda_from_coord_exprs(
                tile_size,
                coord_exprs,
                num_args,
                iter_repeat_factors=_get_placeholder_iter_repeat_factors(
                    node,
                    shape,
                    tile_size,
                    threadblock_extent_map[node.name],
                    node_axis_deps=output_axis_deps.get(node.name),
                    output_threadblock_extent=dshape,
                ),
            ),
            shape=shape,
            tile_size=tile_size,
        )

    return graphs["main"]
