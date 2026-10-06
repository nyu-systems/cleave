from .graph import (
    Graph,
    PlaceHolderNode,
    OutputNode,
    GraphNode,
    ComputeOpNode,
    ViewOpNode,
)
from .util import find_topo_sort
from .ops import Div, Exp, Matmul, Reduction


# TODO: have all levels of the graph with OutputNode (this should make this function smoother)
def annotate_args(graph: Graph, is_main: bool = False) -> Graph:
    for node in find_topo_sort(graph.nodes):
        if is_main:
            if isinstance(node, PlaceHolderNode):
                graph.args.append(node)
            elif isinstance(node, OutputNode):
                graph.args.append(node)
        if isinstance(node, GraphNode):
            in_args = [i.src_node for i in node.inputs]
            # TODO: output nodes
            node.graph.args.extend(in_args)
            annotate_args(node.graph, is_main=False)

    return graph


def annotate_tinfo(graph: Graph, is_main: bool = False) -> Graph:
    all_names = set([node.name for node in graph.nodes])
    assert len(all_names) == len(graph.nodes), "node names should be unique"

    def getname(name: str) -> str:
        name = name.replace(".", "_")
        if name in all_names:
            idx = 1
            while f"{name}_{idx}" in all_names:
                idx += 1
            name = f"{name}_{idx}"
        all_names.add(name)
        return name

    for node in find_topo_sort(graph.nodes):
        if isinstance(node, (PlaceHolderNode, OutputNode)):
            graph.tinfo_map[node] = (
                node.name,
                "g" if is_main else ("r" if len(node.outputs) > 1 else "s"),
            )
        elif isinstance(node, ComputeOpNode):
            graph.tinfo_map[node] = (getname("t_" + node.name), "r")

            # special handling for tilelang matmul that does not support two fragment inputs
            if isinstance(node, Matmul):
                input_tinfos = [
                    (
                        graph.tinfo_map[edge.src_node][1]
                        if not isinstance(edge.src_node, GraphNode)
                        else edge.src_node.graph.tinfo_map[edge.accum_node][1]
                    )
                    for edge in node.inputs
                ]
                assert len(input_tinfos) == 2, "Matmul should have exactly two inputs"
                # put smallest input in shared memory
                if input_tinfos == ["r", "r"]:
                    import numpy as np

                    input_placeholders = [
                        edge.src_node
                        for edge in node.inputs
                        if isinstance(edge.src_node, PlaceHolderNode)
                    ]
                    if len(input_placeholders) == 0:
                        # TODO(david): this is a limitation of tilelang that
                        # forces us to keep at least one operand in smem, might
                        # be solved in a upcoming version.
                        # Both operands are computed fragments (e.g. QKNorm's
                        # QK^T where Q and K are RMSNorm outputs). Hopper WGMMA
                        # requires the B (transpose_B) operand in shared memory
                        # (gemm.cc:436-437). Stage BOTH operands through smem:
                        # operand 1 (B) satisfies WGMMA; operand 0 (A) decouples
                        # the matmul's TC-fragment layout demand from the shared
                        # input fragment (Q), which is also consumed by the
                        # RMSNorm reduction path in a conflicting linear layout.
                        # NOTE(david): annotate_mla_accumulation_layout_conflict resets
                        # stage_fragment_operands_in_shared on every matmul, so if
                        # CompilerPassOptions.mla_decode_pass is ever enabled on a
                        # graph that also has a two-fragment matmul it would clobber
                        # this; today mla_decode_pass defaults False and no graph has
                        # both patterns.
                        node.stage_fragment_operands_in_shared = (0, 1)
                    else:
                        def _tile_volume(p):
                            tile = graph.tmap[p].tile_size
                            if any(not isinstance(t, int) for t in tile):
                                return float("inf")  # whole-buffer operand
                            return int(np.prod(tile))

                        smallest_placeholder = min(input_placeholders, key=_tile_volume)
                        graph.tinfo_map[smallest_placeholder] = (
                            graph.tinfo_map[smallest_placeholder][0],
                            "s",
                        )
        elif isinstance(node, ViewOpNode):
            src_tinfo = graph.tinfo_map[node.inputs[0].src_node]
            graph.tinfo_map[node] = (getname("v_" + node.name), src_tinfo[1])
        elif isinstance(node, GraphNode):
            # ensure that tinfo_map also reflects all args of the subgraph
            node.graph.tinfo_map = {
                k: v for k, v in graph.tinfo_map.items() if k in node.graph.args
            }
            annotate_tinfo(node.graph, is_main=False)

        # if isinstance(node.outputs[0].dst_node, OutputNode):
        #     out_node = node.outputs[0].dst_node # so that it reuses the args buffer # TODO: enable this optimization

    return graph


def annotate_tc_layout(graph: Graph) -> Graph:
    node_to_graph = {}
    graph_to_owner = {}

    def collect_hierarchy(cur_graph: Graph, owner: GraphNode = None) -> None:
        if owner is not None:
            graph_to_owner[id(cur_graph)] = owner

        for cur_node in cur_graph.nodes:
            node_to_graph[id(cur_node)] = cur_graph
            if isinstance(cur_node, GraphNode):
                collect_hierarchy(cur_node.graph, cur_node)

    def annotate_graph(cur_graph: Graph) -> None:
        init_graph_layout(cur_graph)

        for cur_node in cur_graph.nodes:
            if isinstance(cur_node, GraphNode):
                annotate_graph(cur_node.graph)
            elif isinstance(cur_node, Matmul):
                propagate_tc_layout(cur_node, node_to_graph, graph_to_owner)

    collect_hierarchy(graph)
    annotate_graph(graph)
    return graph


def annotate_fragment_layout_conflicts(graph: Graph) -> Graph:
    def reset_graph(cur_graph: Graph) -> None:
        for cur_node in cur_graph.nodes:
            if hasattr(cur_node, "resolve_operand_layout"):
                cur_node.resolve_operand_layout = None
            if isinstance(cur_node, GraphNode):
                reset_graph(cur_node.graph)

    def get_input_scope(cur_graph: Graph, input_edge) -> str | None:
        accum_node = getattr(input_edge, "accum_node", None)
        if accum_node is not None and isinstance(input_edge.src_node, GraphNode):
            return input_edge.src_node.graph.tinfo_map.get(accum_node, (None, None))[1]
        return cur_graph.tinfo_map.get(input_edge.src_node, (None, None))[1]

    def is_independent_reduction_div(cur_node: Div, cur_graph: Graph) -> bool:
        if cur_node.const is not None or cur_node.broadcast != 0:
            return False
        if len(cur_node.inputs) != 2:
            return False

        lhs_edge, rhs_edge = cur_node.inputs
        lhs_accum = getattr(lhs_edge, "accum_node", None)
        rhs_accum = getattr(rhs_edge, "accum_node", None)

        if not isinstance(lhs_accum, Reduction) or not isinstance(rhs_accum, Reduction):
            return False
        if lhs_edge.src_node is not rhs_edge.src_node:
            return False
        if not isinstance(lhs_edge.src_node, GraphNode):
            return False
        if get_input_scope(cur_graph, rhs_edge) != "r":
            return False

        lhs_shape = tuple(lhs_accum.get_shape(0))
        rhs_shape = tuple(rhs_accum.get_shape(0))
        if len(lhs_shape) != len(rhs_shape) + 1:
            return False
        if lhs_shape[:-1] != rhs_shape:
            return False

        if lhs_accum.axis != rhs_accum.axis or rhs_accum.axis != 0:
            return False
        if len(lhs_accum.inputs) != 1 or len(rhs_accum.inputs) != 1:
            return False

        lhs_src = lhs_accum.inputs[0].src_node
        rhs_src = rhs_accum.inputs[0].src_node
        if lhs_src is rhs_src:
            return False

        lhs_input_shape = tuple(
            lhs_accum.inputs[0].src_node.get_shape(lhs_accum.inputs[0].src_id)
        )
        rhs_input_shape = tuple(
            rhs_accum.inputs[0].src_node.get_shape(rhs_accum.inputs[0].src_id)
        )
        if len(lhs_input_shape) != len(rhs_input_shape) + 1:
            return False

        return True

    def annotate_graph(cur_graph: Graph) -> None:
        for cur_node in cur_graph.nodes:
            if isinstance(cur_node, GraphNode):
                annotate_graph(cur_node.graph)
                continue
            if isinstance(cur_node, Div) and is_independent_reduction_div(
                cur_node, cur_graph
            ):
                cur_node.resolve_operand_layout = "rhs"

    reset_graph(graph)
    annotate_graph(graph)
    return graph


def annotate_mla_accumulation_layout_conflict(graph: Graph) -> Graph:
    def reset_graph(cur_graph: Graph) -> None:
        for cur_node in cur_graph.nodes:
            if isinstance(cur_node, Matmul):
                cur_node.stage_fragment_operands_in_shared = ()
            if isinstance(cur_node, GraphNode):
                reset_graph(cur_node.graph)

    def has_reduction_consumer(exp_node: Exp) -> bool:
        return any(isinstance(edge.dst_node, Reduction) for edge in exp_node.outputs)

    def has_upstream_score_matmul(exp_node: Exp, rhs_node: PlaceHolderNode) -> bool:
        worklist = [edge.src_node for edge in exp_node.inputs]
        visited = set()

        while worklist:
            cur_node = worklist.pop()
            cur_node_id = id(cur_node)
            if cur_node_id in visited:
                continue
            visited.add(cur_node_id)

            if isinstance(cur_node, Matmul) and len(cur_node.inputs) == 2:
                if cur_node.transpose_B and cur_node.inputs[1].src_node is rhs_node:
                    return True

            worklist.extend(edge.src_node for edge in getattr(cur_node, "inputs", []))

        return False

    def annotate_graph(cur_graph: Graph) -> None:
        for cur_node in cur_graph.nodes:
            if isinstance(cur_node, GraphNode):
                annotate_graph(cur_node.graph)
                continue

            if not isinstance(cur_node, Matmul) or len(cur_node.inputs) != 2:
                continue

            lhs_node = cur_node.inputs[0].src_node
            rhs_node = cur_node.inputs[1].src_node
            if not isinstance(lhs_node, Exp):
                continue
            if not isinstance(rhs_node, PlaceHolderNode):
                continue
            if not has_reduction_consumer(lhs_node):
                continue
            if not has_upstream_score_matmul(lhs_node, rhs_node):
                continue

            # MLA decode reuses the KV tensor for both score and value paths.
            # Stage the softmax fragment through swizzled shared memory before
            # the value matmul to avoid TileLang fragment layout conflicts.
            cur_node.stage_fragment_operands_in_shared = (0,)

    reset_graph(graph)
    annotate_graph(graph)
    return graph


def init_graph_layout(graph: Graph) -> Graph:
    graph.tlayout_map.clear()
    for node in graph.nodes:
        if isinstance(node, (PlaceHolderNode, OutputNode)):
            graph.tlayout_map[node] = "linear"

    return graph


def propagate_tc_layout(node, node_to_graph, graph_to_owner):
    from collections import deque

    def get_graph(cur_node):
        return node_to_graph.get(id(cur_node))

    # Operands the gemm itself reads. A computed one is a swizzle-able shared buffer just
    # like a staged placeholder -- a gathered KV tile arrives this way -- but only right
    # here: further up its producer chain sit things like the 1-D index tile, which no
    # tensor-core instruction touches.
    gemm_operands = {id(edge.src_node) for edge in node.inputs}

    def mark_tc(cur_node):
        cur_graph = get_graph(cur_node)
        if cur_graph is None:
            return
        if not isinstance(cur_node, (PlaceHolderNode, OutputNode)):
            scope = cur_graph.tinfo_map.get(cur_node, (None, None))[-1]
            if not (
                isinstance(cur_node, ComputeOpNode)
                and scope == "s"
                and id(cur_node) in gemm_operands
            ):
                return
        cur_graph.tlayout_map[cur_node] = "tc"

    def is_reduction_node(cur_node):
        return len(getattr(cur_node, "raxis", [])) > 0

    def move_to_upper_graph_producer(cur_node):
        cur_graph = get_graph(cur_node)
        if cur_graph is None:
            return []

        owner = graph_to_owner.get(id(cur_graph))
        if owner is None:
            return []

        outer_node = cur_graph.binding.get(cur_node.name)
        return [outer_node] if outer_node is not None else []

    def move_to_upper_graph_consumers(cur_node):
        cur_graph = get_graph(cur_node)
        if cur_graph is None:
            return []

        owner = graph_to_owner.get(id(cur_graph))
        if owner is None:
            return []

        next_nodes = []
        bound_node = cur_graph.binding.get(cur_node.name)
        if isinstance(bound_node, OutputNode):
            next_nodes.append(bound_node)

        next_nodes.extend(
            edge.dst_node
            for edge in owner.outputs
            if getattr(edge, "accum_node", None) is cur_node
        )
        return next_nodes

    def get_next_nodes(cur_node, direction):
        if direction == "producers":
            if cur_node.inputs:
                return [edge.src_node for edge in cur_node.inputs]
            return move_to_upper_graph_producer(cur_node)

        if cur_node.outputs:
            return [edge.dst_node for edge in cur_node.outputs]
        return move_to_upper_graph_consumers(cur_node)

    frontier = deque([(node, "producers", True), (node, "consumers", True)])
    visited = {(id(node), "producers"), (id(node), "consumers")}

    while frontier:
        cur_node, direction, is_root = frontier.popleft()
        mark_tc(cur_node)

        if not is_root and (
            isinstance(cur_node, GraphNode) or is_reduction_node(cur_node)
        ):
            continue

        for next_node in get_next_nodes(cur_node, direction):
            state = (id(next_node), direction)
            if state in visited:
                continue
            visited.add(state)
            frontier.append((next_node, direction, False))
