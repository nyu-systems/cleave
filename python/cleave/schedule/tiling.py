import weakref
from collections import Counter
from dataclasses import dataclass
from math import prod
from typing import Dict, List, NamedTuple, Optional, Set
from tilelang.carver.roller.node import Edge
from ..core.errors import ScheduleRejected
from ..ir.graph import (
    Graph,
    OutputNode,
    GraphNode,
    PlaceHolderNode,
    Iterator,
    TensorOp,
    ViewOpNode,
)
from ..ir.ops import ReduceSum
from ..ir.offsets import OffsetTerm, TileOffset, get_offset_lambda
from ..ir.analysis import (
    build_output_axis_provenance,
    build_reduction_axis_provenance,
    launch_axis_dependencies,
)
from ..ir.graph_utils import (
    get_op_ancestors,
    gather_all_nodes,
    find_topo_sort,
    clone_graph,
    graph_tiers,
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


def _sibling_launch_axes(canonical_output, output_nodes):
    """``(output, [canonical axis per axis of it])`` for the outputs next to the canonical one.

    The same map the outputs of a loop are stored by (``_get_same_launch_output_axis_map``):
    batch axes in order, the other axes onto the canonical output's leading non-batch axes.
    An output that has no such map is left out; its own ancestors then depend on no launch
    axis, and a tiling whose grid would have to walk them is refused as before.

    The map pairs axes by position and extent, not by what indexes them: whether an axis
    of a sibling really is the canonical axis it was paired with is checked against the
    graph, per tiling, by ``_refuse_siblings_on_another_axis``.
    """
    canonical = _get_output_binding_node(canonical_output)
    siblings = []
    for output in output_nodes:
        if output is canonical_output:
            continue
        try:
            axis_map = _get_same_launch_output_axis_map(
                canonical, _get_output_binding_node(output)
            )
        except NotImplementedError:
            continue
        siblings.append((output, axis_map))
    return siblings


def _refuse_siblings_on_another_axis(disagreements, axis_coord_exprs):
    """Refuse a grid that walks an axis on which a sibling output is wrongly launched.

    A sibling output is launched and stored on the canonical axes its extents were paired
    with (``_get_same_launch_output_axis_map``; the policy tiles it by the same pairing).
    ``disagreements`` (``ir.analysis.launch_axis_dependencies``) are the dims of the
    ancestors it shares with the canonical output that this pairing would have indexed by
    other launch axes than the canonical output does. A tensor has one tile and one offset,
    the canonical output's, so wherever the grid walks an axis the two views do not share,
    the sibling's own ops are cut along an axis their operand is not cut along:
    ``exp(x) * 2`` next to ``exp(x) @ b`` took its column block ``i1`` from the first
    columns of an ``exp(x)`` held whole, and ``sum(w, 0)`` next to ``x @ w`` stored the
    column sums of column block ``i1`` at row block ``i0`` -- kernels that were built,
    lowered, and wrote wrong values (several blocks to one slice). While the grid stands
    still on those axes, both views slice every tensor alike and the tiling is served.
    """

    def walked(axes):
        return {
            axis
            for axis in axes
            if axis in axis_coord_exprs and axis_coord_exprs[axis].expr != "0"
        }

    for sibling, node, dim, axes, sibling_axes in disagreements:
        if walked(axes) != walked(sibling_axes):
            raise ScheduleRejected(
                f"{sibling} is launched on the axes of the canonical output by extent, but "
                f"dim {dim} of {node} is indexed by launch axes {sorted(axes)} for the "
                f"canonical output and {sorted(sibling_axes)} for {sibling}: the grid walks "
                f"{sorted(walked(axes) ^ walked(sibling_axes))}, where the two are not the "
                "same axis."
            )


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
            # The block holds this dim whole. That serves every reader only while the
            # grid stands still on the launch axes that index it: a dim that two launch
            # axes index has two readers (``x @ x.mT`` reads the rows of ``x`` as rows and
            # as columns of the output), and when the grid walks one of them that reader
            # needs a slice while the other needs the whole. A tensor has one tile, the
            # last reader's; the kernel was built and failed TileLang's gemm shape check.
            dep_axes = (
                sorted(node_axis_deps[dim_idx])
                if node_axis_deps is not None and dim_idx < len(node_axis_deps)
                else []
            )
            walked = [
                axis
                for axis in dep_axes
                if axis in output_axis_coord_exprs
                and output_axis_coord_exprs[axis].expr != "0"
            ]
            if len(dep_axes) > 1 and walked:
                raise ScheduleRejected(
                    f"Dim {dim_idx} is held whole but launch axes {dep_axes} index it and "
                    f"the grid walks {walked}: one tile cannot be a slice and the whole."
                )
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
            # A refusal of this tiling, not a bug: the grid would have to walk a dim that
            # no launch axis (or more than one) indexes.
            raise ScheduleRejected(
                f"Expected exactly one launch-axis mapping for iter dim {dim_idx}, got {dep_axes}."
            )
        coord_exprs[dim_idx] = coord_expr

    return coord_exprs


def _build_offset_lambda_from_coord_exprs(
    tile_shape: List[int],
    coord_exprs: List[Optional[_OutputAxisCoordExpr]],
    num_args: int,
    iter_repeat_factors: Optional[List[int]] = None,
) -> TileOffset:
    assert len(coord_exprs) == len(
        tile_shape
    ), "coord_exprs must have the same length as tile_shape"
    assert num_args >= 1, "num_args must be positive"
    iter_repeat_factors = iter_repeat_factors or [1] * len(tile_shape)
    assert len(iter_repeat_factors) == len(
        tile_shape
    ), "iter_repeat_factors must have the same length as tile_shape"

    terms = []
    for tile, coord_expr, repeat_factor in zip(
        tile_shape, coord_exprs, iter_repeat_factors
    ):
        if coord_expr is None or coord_expr.expr == "0":
            terms.append(None)
            continue
        omit_scale = int(tile) == 1 and coord_expr.omit_unit_scale and repeat_factor == 1
        terms.append(
            OffsetTerm(
                coord_expr.expr,
                scale=None if omit_scale else int(tile),
                divisor=repeat_factor,
            )
        )
    return TileOffset(tuple(terms), num_args)


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

        # get tile shape and tile size from candidate. The loop is handed the buffer its
        # parent stores, which has no dim the block holds a single element of (codegen's
        # ``kept_extents``; an all-unit tile stays 1-D), so the tier carries exactly the
        # other dims -- whether or not they are batch dims, and also where an iteration
        # takes one element of them: that extent is squeezed when the slice is loaded.
        block_tile = tile_map[bind_node.name]
        axes = [i for i, t in enumerate(block_tile) if t > 1] or [len(block_tile) - 1]
        shape = [block_tile[i] for i in axes]
        tile_size = [itile_map[bind_node.name][i] for i in axes]
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
            axes=tuple(axes),
        )


def _is_unit(extent) -> bool:
    """Whether codegen squeezes ``extent`` out of a stored tile (its ``kept_extents``)."""
    return isinstance(extent, int) and extent <= 1


def _rebase_reduction_axis(node: ReduceSum, tile):
    """Point the ``dim`` that ``node`` emits at its reduced axis in the stored input tile.

    ``tile`` is the tile of the node's input over the node's full ``dims``. A kernel
    stores a tile without its unit extents, so the reduced axis has as many dims in
    front of it as ``tile`` has non-unit extents there; an axis given from the end is
    kept in that form and counted from the end. The answer is derived from the axis the
    node was built with (``flat_axis``), never from an earlier answer: a loop's nodes
    are visited again on every merge into that loop.
    """
    axis = node.flat_axis
    node.axis, node.axis_arg = axis, node.flat_axis_arg
    node.stored_axis = axis
    if tile is None or len(tile) != len(node.dims) or _is_unit(tile[axis]):
        # A unit extent on the reduced axis itself (a loop that walks the axis one
        # element at a time) leaves the stored tile without that axis: there is no
        # ``dim`` to give, and such a tiling does not compile.
        return
    # Where the axis sits in the stored tile, whichever end it was named from.
    node.stored_axis = sum(1 for extent in tile[:axis] if not _is_unit(extent))
    if node.flat_axis_arg < 0:
        node.axis_arg = -1 - sum(1 for extent in tile[axis + 1 :] if not _is_unit(extent))
    else:
        node.axis = node.axis_arg = node.stored_axis


def _normalize_inner_reduction_axes(graph_node: GraphNode, itile_map: dict):
    """Rebase the reduction axes of a loop tier onto the iteration tiles they read."""
    for node in graph_node.graph.nodes:
        if not isinstance(node, ReduceSum) or len(node.inputs) != 1:
            continue
        src_node = node.inputs[0].src_node
        bind_node = graph_node.graph.binding.get(src_node.name, src_node)
        tile = itile_map.get(bind_node.name)
        _rebase_reduction_axis(node, None if tile is None else list(tile))


def rebase_reduction_axis(node: ReduceSum, it: Iterator):
    """``_rebase_reduction_axis`` from ``it``, the tile of the node's input in its tier.

    ``apply_tiling`` does this for every reduction it schedules. A reduction added to a
    scheduled graph afterwards has to be given its ``dim`` the same way, from the
    ``Iterator`` of the tensor it reads.
    """
    rank = len(node.dims)
    tile = [1] * rank  # a dim the tier does not carry is one the tile holds one element of
    for dim in range(rank):
        position = it.position(dim, rank)
        if position is not None:
            tile[dim] = it.tile_size[position]
    _rebase_reduction_axis(node, tile)


def _normalize_main_reduction_axes(main: Graph):
    """Rebase the reduction axes of the main tier onto the tiles they read.

    A reduction that is not looped stays in the main tier, where a batch dim is a unit
    extent of every tile and is squeezed like any other: without this, a sum over a
    non-leading axis is emitted with its flat ``dim`` and reduces the next axis over.
    """
    for node in main.nodes:
        if not isinstance(node, ReduceSum) or len(node.inputs) != 1:
            continue
        edge = node.inputs[0]
        if isinstance(edge.src_node, GraphNode):
            # A loop's accumulator: the buffer is the one the loop's own tier allocates.
            accum_node = getattr(edge, "accum_node", None)
            it = None if accum_node is None else edge.src_node.graph.tmap.get(accum_node)
            if isinstance(it, Iterator):
                rebase_reduction_axis(node, it)
            else:
                _rebase_reduction_axis(node, None)
        else:
            # A tile of this tier is over the node's dims as it is. One of another rank
            # (a unit dim that is not a batch dim has no tile) is left alone.
            _rebase_reduction_axis(node, list(main.tmap[edge.src_node].tile_size))


def _iterative_reductions(nodes, rstep_map):
    """The reductions of ``nodes`` that are stepped below their extent."""
    return [
        n
        for n in nodes
        if len(n.raxis) > 0
        and n in rstep_map
        and _get_rstep_for_axis(n, rstep_map[n]) < n.raxis[0].dom.extent
    ]


def reduction_ancestors(nodes):
    """``{reduction: its op ancestors}`` for the reductions of ``nodes``.

    What ``absorbed_reductions`` asks of the graph. It does not depend on the steps, so
    a caller that rates many step maps of one graph (the policy's search walk) builds it
    once and passes it in, instead of walking the graph for every pair of reductions of
    every step map.
    """
    return {n: frozenset(get_op_ancestors(n)) for n in nodes if len(n.raxis) > 0}


def absorbed_reductions(nodes, rstep_map, ancestors=None):
    """The reductions of ``nodes`` that ``apply_tiling`` schedules inside another's loop.

    A reduction that iterates (step < extent) while one of its descendant reductions
    also iterates is absorbed into that descendant's loop (see the merge in
    ``apply_tiling``). The loop has a single counter, which advances the loop's own
    axis; it cannot also advance the absorbed reduction's axis. An absorbed reduction
    is therefore computed over its whole axis in every iteration, whatever its step:
    the tiling is the kernel of the one that steps it at its extent.

    ``ancestors`` is ``reduction_ancestors(nodes)`` when the caller already has it.
    """
    active = _iterative_reductions(nodes, rstep_map)
    if len(active) < 2:
        return []
    if ancestors is None:
        ancestors = reduction_ancestors(active)
    return [
        a
        for a in active
        if any(r is not a and a in ancestors[r] for r in active)
    ]


class _ScheduledAxes(NamedTuple):
    """What the absorbed-reduction pin asks of a flat graph, whatever the candidate."""

    structure: tuple  # ``_structure`` of the flat graph these were read from
    nodes: list  # op nodes of the scheduled clone (split views removed)
    out_tokens: dict  # ``build_output_axis_provenance`` of ``nodes``
    red_tokens: dict  # ``build_reduction_axis_provenance`` of ``nodes``
    ancestors: dict  # ``reduction_ancestors`` of ``nodes``


# id(flat graph) -> (weak reference to it, its _ScheduledAxes). A ``Graph`` is a mutable
# dataclass and so not hashable; the entry goes when the graph does.
_SCHEDULED_AXES: dict = {}


def _structure(flat_graph) -> tuple:
    """The nodes of ``flat_graph`` and how they are wired, by name."""
    return tuple(
        (n.name, tuple((e.src_node.name, e.src_id) for e in n.inputs))
        for n in flat_graph.nodes
    )


def _scheduled_axes(flat_graph) -> _ScheduledAxes:
    """Axis provenance and reduction ancestors of ``flat_graph`` as ``apply_tiling`` sees it.

    They depend on the flat graph only, and every candidate of a search shares one flat
    graph, so they are read once per graph instead of once per candidate (it was two
    thirds of ``apply_tiling``'s time on a candidate). They are read off a clone of
    their own -- the one ``apply_tiling`` schedules is taken apart by it -- and are
    indexed by node: nodes compare by name, so the nodes of any clone find their entry.
    A graph that was rewired since (``_structure``) is read again.
    """
    structure = _structure(flat_graph)
    entry = _SCHEDULED_AXES.get(id(flat_graph))
    if entry is not None and entry[0]() is flat_graph and entry[1].structure == structure:
        return entry[1]

    _, pruned = _extract_split_extent(flat_graph)
    ordered_nodes = [
        n for n in pruned.nodes if not n.is_placeholder() and not n.is_output()
    ]
    out_tokens = build_output_axis_provenance(ordered_nodes)
    axes = _ScheduledAxes(
        structure,
        ordered_nodes,
        out_tokens,
        build_reduction_axis_provenance(ordered_nodes, out_tokens),
        reduction_ancestors(ordered_nodes),
    )

    key = id(flat_graph)

    def forget(ref):
        if _SCHEDULED_AXES.get(key, (None,))[0] is ref:
            del _SCHEDULED_AXES[key]

    _SCHEDULED_AXES[key] = (weakref.ref(flat_graph, forget), axes)
    return axes


def scheduled_reductions(flat_graph):
    """``(op nodes, reduction-axis provenance)`` of ``flat_graph`` as ``apply_tiling`` sees it.

    ``apply_tiling`` schedules a clone of the flat graph with its split views removed, and
    decides what an absorbed reduction holds whole from the axis provenance of that clone.
    Anything that has to agree with it (the policy's verdict on a candidate) reads the
    provenance from here. Nodes compare by name, so a step map keyed by the nodes of
    ``flat_graph`` indexes the returned ones.
    """
    axes = _scheduled_axes(flat_graph)
    return axes.nodes, axes.red_tokens


def _held_axis_tokens(ordered_nodes, rstep_map, red_tokens, ancestors=None):
    """Provenance tokens of the axes that absorbed reductions hold whole.

    Raises ``ScheduleRejected`` for a tiling no loop can serve: one in which another
    reduction steps an axis that an absorbed reduction has to keep whole.
    """

    def axis_tokens(node):
        tokens = red_tokens.get(node)
        return tokens[0] if tokens else set()

    absorbed = absorbed_reductions(ordered_nodes, rstep_map, ancestors)
    # These are refusals, not asserts: without them the tiling would be scheduled into
    # a kernel that reads the wrong slices, and ``python -O`` drops an assert.
    held = set()
    for node in absorbed:
        tokens = axis_tokens(node)
        if not tokens:
            raise ScheduleRejected(
                f"No axis provenance for the reduction axis of {node.name}; cannot tell "
                "which tensor dims it runs over."
            )
        held |= tokens
    for node in _iterative_reductions(ordered_nodes, rstep_map):
        # Keeping a loop's own axis at block extent would stop the loop from advancing.
        if node not in absorbed and held & axis_tokens(node):
            raise ScheduleRejected(
                f"Reduction axis of {node.name} is shared with an absorbed ancestor "
                "reduction; one loop cannot both advance it and hold it whole."
            )
    return held


def absorbed_axis_conflict(ordered_nodes, rstep_map, red_tokens, ancestors=None) -> Optional[str]:
    """Why ``apply_tiling`` refuses ``rstep_map`` over its absorbed reductions, or ``None``.

    ``ordered_nodes`` and ``red_tokens`` are those of ``scheduled_reductions``;
    ``ancestors`` is ``reduction_ancestors(ordered_nodes)`` when the caller has it.
    """
    try:
        _held_axis_tokens(ordered_nodes, rstep_map, red_tokens, ancestors)
    except ScheduleRejected as refusal:
        return str(refusal)
    return None


def _pin_absorbed_reduction_axes(flat_graph, rstep_map, tile_map, itile_map):
    """Iteration tiles with every absorbed reduction axis kept at its block extent.

    Each tensor dim that carries the axis of an absorbed reduction (see
    ``absorbed_reductions``) keeps its block tile, so ``assign_tensor_map`` does not
    slice it with the counter of the loop the reduction was absorbed into.
    ``flat_graph`` is the candidate's; what is asked of it is the same for every
    candidate (``_scheduled_axes``).

    Raises ``ScheduleRejected`` for a tiling in which another reduction steps that same
    axis: its loop would have to advance an axis that has to stay whole.
    """
    axes = _scheduled_axes(flat_graph)
    if not absorbed_reductions(axes.nodes, rstep_map, axes.ancestors):
        return itile_map

    pinned = _held_axis_tokens(axes.nodes, rstep_map, axes.red_tokens, axes.ancestors)

    itile_map = dict(itile_map)
    for (node, out_id), dim_tokens in axes.out_tokens.items():
        if out_id != 0 or node.name not in itile_map:
            continue
        tile, itile = tile_map[node.name], list(itile_map[node.name])
        if len(dim_tokens) != len(itile) or len(tile) != len(itile):
            continue
        for dim, tokens in enumerate(dim_tokens):
            if tokens & pinned:
                itile[dim] = tile[dim]
        itile_map[node.name] = itile
    return itile_map


def _refuse_per_iteration_values_read_outside(main: Graph):
    """Refuse a loop whose per-iteration value is also read by a node outside the loop.

    A reduction that iterates takes its ancestors into its loop. An ancestor that has a
    second reader which did not come along -- ``exp`` under a looped row sum, also read by
    the ``div`` that normalizes it -- then exists once per iteration, and only as that
    iteration's slice: the reader outside has nothing to read. Only an accumulator leaves a
    loop. Checked once every merge is done, because a later reduction may still take the
    reader into the loop.
    """
    for tier, owner in graph_tiers(main):
        if owner is None:
            continue
        inside = {id(n) for n in tier.nodes}
        carried = {id(n) for n in tier.accum_nodes}
        for node in tier.nodes:
            if not isinstance(node, TensorOp) or id(node) in carried:
                continue
            for edge in node.outputs:
                if id(edge.dst_node) not in inside:
                    raise ScheduleRejected(
                        f"{node.name} is computed per iteration of {owner.name} but also "
                        f"read by {edge.dst_node.name} outside it"
                    )


def _refuse_loops_that_miscount_their_tiles(main: Graph):
    """Refuse a loop whose trip count is not the number of tiles it steps through.

    A loop has one counter, and every inner tensor dim that is cut into more than one tile
    is sliced with it (``assign_tensor_map``). The trip count comes from the reduction that
    created the loop; a reduction merged into the loop later with another step leaves dims
    cut into a different number of tiles ("the existing inner graph extent wins"). The
    loop then runs past the end of those dims, or stops short of it: 4096 iterations over
    128 tiles of 32. Such a kernel reads memory it does not own or drops part of the sum.

    A loop that cuts nothing at all is the same miscount at its extreme: every iteration
    reads the whole of every tensor, and the sum is taken once per iteration. It comes
    from a step map that names a reduction axis the node does not have: the trip count
    is read off the map's only value, the tiles off the node's own axis, which then has
    no step.
    """
    for tier, owner in graph_tiers(main):
        if owner is None:
            continue
        cut = False
        for node, it in tier.tmap.items():
            for dim, (extent, step) in enumerate(zip(it.shape, it.tile_size)):
                tiles = int(extent) // int(step)
                cut = cut or tiles > 1
                if tiles > 1 and tiles != tier.extent:
                    raise ScheduleRejected(
                        f"{owner.name} runs {tier.extent} iterations but dim {dim} of "
                        f"{node.name} is cut into {tiles} tiles ({step} of {extent})"
                    )
        if not cut and isinstance(tier.extent, int) and tier.extent > 1:
            raise ScheduleRejected(
                f"{owner.name} runs {tier.extent} iterations but cuts none of its tensors "
                "into tiles: every iteration would read the same slice"
            )


def _pipeline_one_loop(main: Graph):
    """Leave ``T.Pipelined`` to the first loop of a kernel; emit the others as plain loops.

    A loop tier is emitted as ``T.Pipelined``, which lets TileLang turn the copies of the
    slices each iteration loads into asynchronous bulk transfers. The vendored TileLang
    does that for one loop per kernel: the loop variable of a second pipelined loop is
    left behind outside the loop, and the kernel fails at the very end of lowering ("In
    PrimFunc main_kernel variables (i0,) are used, but are not passed in as API
    arguments"). Two sums looped with different steps (a split combine), or two matmuls
    that share no operand, each get a loop of their own; every such kernel died there.

    The second loop computes the same thing as a plain loop, and with it the kernel
    lowers. The first keeps its pipeline: with the first serial and the second pipelined
    TileLang lowers some kernels and not others. ``main.nodes`` is in emission order.
    """
    pipelined = False
    for tier, owner in graph_tiers(main):
        if owner is None or tier.loop_kind != "pipelined":
            continue
        if pipelined:
            tier.loop_kind = "serial"
        pipelined = True


def _refuse_tile_that_misses_dims(node, shape, tile, expanded_tile):
    """Refuse a tiling whose tile has fewer dims than the tensor it tiles.

    The policy gives a tile one extent per axis the op iterates. A dim of extent 1 is not
    iterated, so it has no extent in the tile; ``_expand_tile_to_shape_rank`` puts it back
    when it is a batch dim. A unit dim that is not a batch dim stays missing: the output
    of ``x @ column`` (a ``Matmul`` with N = 1) is ``(M, 1)`` with the tile ``[m]``. The
    grid extent would then be taken from the dims the tile happens to cover, and the
    launch-axis plan and the offsets built for a tensor of lower rank (it used to end in
    ``KeyError: 1`` or an assert on ``iter_repeat_factors``, depending on the loop).

    Next to batch dims the tile has the tensor's rank after all, with its extents on the
    wrong dims: for ``(B, M, 1)`` the policy tiles ``(B, M)`` as ``[b, m]`` and the
    expansion reads that as ``[1, b, m]``, ``m`` rows of the unit dim. Same defect, same
    refusal (it used to end in whichever guard the nonsense tripped first).
    """
    misplaced = len(tile) != len(shape) and any(
        int(extent) == 1 and int(step) > 1 for extent, step in zip(shape, expanded_tile)
    )
    if len(expanded_tile) != len(shape) or misplaced:
        raise ScheduleRejected(
            f"{node.name}: tile {list(tile)} does not cover the {len(shape)} dims of its "
            f"tensor {tuple(int(s) for s in shape)}; a unit dim that is not a batch dim "
            "has no tile"
        )


def _stored_rank(tile) -> int:
    """The rank of the buffer a tile is stored in: its unit extents are squeezed out
    (codegen's ``kept_extents``), and a tile of unit extents only is stored 1-D."""
    return max(1, sum(1 for extent in tile if not _is_unit(extent)))


def _operand_tile(tier: Graph, edge) -> Optional[Iterator]:
    """The tile of the value ``edge`` reads, in the tier that stores it."""
    if isinstance(edge.src_node, GraphNode):
        accum_node = getattr(edge, "accum_node", None)
        it = None if accum_node is None else edge.src_node.graph.tmap.get(accum_node)
    else:
        it = tier.tmap.get(edge.src_node)
    return it if isinstance(it, Iterator) else None


def refuse_unit_extents(main: Graph) -> None:
    """Refuse a scheduled graph whose kernel does not compile because of a unit extent.

    A kernel stores a tile without its unit extents. Three kinds of op are emitted for the
    rank of the tensor, not of the tile, and a tiling that takes one element of a
    non-batch dim leaves them a buffer of lower rank than they index:

    * a reduction that is handed one element of the axis it reduces per step (a loop that
      walks the axis one element at a time): the slice has no such axis, so there is no
      ``dim`` to reduce over (TileLang: "Invalid reduce output shape");
    * an elementwise op, whose macro has one index per non-batch dim (a block of a single
      row: "not enough values to unpack", "Can't deduce copy extents");
    * a matmul, whose operands and result must be stored 2-D (a single row or column:
      "Swizzle layout expects rank >= 2", the gemm 2-D assert).

    Such a candidate used to be built and handed to TileLang, where it died at parse or in
    layout inference, each with whatever message the missing dim produced first. This is
    the default build's check (``runtime.compile.build_candidate``), not ``apply_tiling``'s:
    the paged path schedules step-1 tilings only to read them (``read_dense_combine``) and
    emits its own kernels for them.
    """
    from ..ir.ops import Matmul
    from ..ir.ops.elementwise import BinaryElementwise, UnaryElementwise

    for tier, _ in graph_tiers(main):
        for node in tier.nodes:
            if isinstance(node, ReduceSum) and len(node.inputs) == 1:
                it = _operand_tile(tier, node.inputs[0])
                if it is None:
                    continue
                rank = len(node.dims)
                position = it.position(node.flat_axis, rank)
                step = 1 if position is None else it.tile_size[position]
                if _is_unit(step):
                    raise ScheduleRejected(
                        f"{node.name} is handed one element of the axis it reduces per step "
                        f"({int(node.dims[node.flat_axis])} in all): the stored slice has no "
                        "such axis to reduce over"
                    )
            elif isinstance(node, (BinaryElementwise, UnaryElementwise)):
                it = tier.tmap.get(node)
                if not isinstance(it, Iterator):
                    continue
                indexed = len(node.dims) - node.n_batch_dims
                tile = tuple(int(t) for t in it.tile_size)
                if _stored_rank(tile) != indexed:
                    raise ScheduleRejected(
                        f"{node.name} indexes {indexed} dims but its tile {tile} is stored "
                        f"with {_stored_rank(tile)}: a unit extent that is not a batch dim "
                        "is squeezed out of the tile"
                    )
            elif isinstance(node, Matmul):
                tiles = [("result", tier.tmap.get(node))] + [
                    (f"operand {edge.dst_id}", _operand_tile(tier, edge)) for edge in node.inputs
                ]
                for what, it in tiles:
                    if not isinstance(it, Iterator):
                        continue
                    tile = tuple(int(t) for t in it.tile_size)
                    if _stored_rank(tile) != 2:
                        raise ScheduleRejected(
                            f"{node.name}: its {what} tile {tile} is stored with "
                            f"{_stored_rank(tile)} dim(s); a matmul takes 2-D operands and "
                            "gives a 2-D result, so a single row or column has no kernel"
                        )


def apply_tiling(cand: List) -> List[Graph]:
    output_nodes = [n for n in cand.flat_graph.nodes if n.is_output()]
    output_node = _select_canonical_output_node(output_nodes)

    # this needs to be called before any graph modifications
    # to correctly infer output axis dependencies for the
    # original output node
    output_axis_deps, sibling_disagreements = launch_axis_dependencies(
        output_node, _sibling_launch_axes(output_node, output_nodes)
    )

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
    itile_map = _pin_absorbed_reduction_axes(
        cand.flat_graph, cand.rstep_map, tile_map, itile_map
    )

    # set top level threadblock extent and tmap based on the
    # candidate tile and reduction step choices
    threadblock_extent_map = {}
    for node in graphs["main"].nodes:
        shape_owner = _get_schedule_shape_owner(node)
        tile = tile_map[
            _get_output_binding_node(node).name if node.is_output() else node.name
        ]
        expanded_tile = _expand_tile_to_shape_rank(node, tile)
        _refuse_tile_that_misses_dims(node, shape_owner.get_shape(), tile, expanded_tile)
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
    _refuse_siblings_on_another_axis(sibling_disagreements, launch_axis_plan.axis_coord_exprs)

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
                    # inner graph extent wins. An ancestor of the loop's
                    # reductions that is merged this way does not iterate at
                    # all: _pin_absorbed_reduction_axes kept its axis whole.
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
    _refuse_per_iteration_values_read_outside(graphs["main"])
    _refuse_loops_that_miscount_their_tiles(graphs["main"])
    lock_graph(graphs["main"])  # graph structure is fixed after this point

    graphs["main"].nodes = find_topo_sort(main_output_nodes)
    _pipeline_one_loop(graphs["main"])
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
            stored_axes = tuple(range(len(logical_shape)))
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
                stored_axes = tuple(reorder)

            while len(physical_shape) < len(canonical_output_binding.get_shape()):
                physical_shape.append(1)
                physical_tile_size.append(1)
                physical_coord_exprs.append(None)

            # The stored layout is decided here and nowhere else, so it is recorded here:
            # no consumer has to recover it from the extents of the two shapes. The pad is
            # counted against the rank, not against `logical_shape`: without a reorder
            # that is the very list the unit dims were just appended to.
            graphs["main"].stored_layouts[node.name] = (
                stored_axes,
                len(physical_shape) - len(stored_axes),
            )

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
        if isinstance(node, OutputNode):  # stored as its producer computes it
            graphs["main"].stored_layouts[node.name] = (tuple(range(len(shape))), 0)
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
    _normalize_main_reduction_axes(graphs["main"])

    if any(getattr(n, "stabilize", None) is not None for n in ordered_nodes):
        # A softmax exp the scale analysis marked gets its max here, where the tiers are
        # final. Lazy: the pass is not even imported for a graph without one.
        from .stabilize import stabilize

        return stabilize(graphs["main"])
    return graphs["main"]
