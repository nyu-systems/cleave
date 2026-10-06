from __future__ import annotations

import copy
from typing import List

from tilelang.carver.roller.node import Edge, Node


def find_topo_sort(output_node_list) -> List[Node]:
    def topo_sort_dfs(node, visited, topo_order):
        if node in visited:
            return
        visited.add(node)
        for edge in node.inputs:
            topo_sort_dfs(edge.src_node, visited, topo_order)
        topo_order.append(node)

    visited = set()
    topo_order = []
    for node in output_node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def gather_all_nodes(*nodes, ascending=True, descending=True, change_name=True):
    visited, tensor_names = set(), set()
    result_nodes = []

    def dfs(node: Node):
        if node in visited:
            return
        while change_name and node.name in tensor_names:
            node.name = node.name[:-1] + chr(ord(node.name[-1]) + 1)
        visited.add(node), tensor_names.add(node.name)
        result_nodes.append(node)
        if ascending:
            for edge in node.inputs:
                dfs(edge.src_node)
        if descending:
            for edge in node.outputs:
                dfs(edge.dst_node)

    for node in nodes:
        dfs(node)

    return result_nodes


def get_op_ancestors(node, ordered_nodes=None):
    """
    Get all ancestors of a node from a topologically sorted list of nodes.
    """
    ancestors = gather_all_nodes(
        node, ascending=True, descending=False, change_name=False
    )

    if ordered_nodes is None:
        ordered_nodes = ancestors
    return [n for n in ordered_nodes if n in ancestors and n != node]


def clone_graph(graph_to_clone):
    from ..graph import Graph, GraphNode

    # create a new Graph object with copied mutable attributes
    cloned_graph = Graph(nodes=[])
    cloned_graph.tmap = graph_to_clone.tmap.copy()
    cloned_graph.extent = graph_to_clone.extent
    cloned_graph.tinfo_map = graph_to_clone.tinfo_map.copy()
    cloned_graph.args = list(graph_to_clone.args)
    cloned_graph.accum_nodes = list(graph_to_clone.accum_nodes)
    cloned_graph.binding = graph_to_clone.binding.copy()

    node_map = {}
    # copy nodes
    for node in graph_to_clone.nodes:
        new_node = copy.copy(node)
        if isinstance(new_node, GraphNode):
            new_node.graph = clone_graph(new_node.graph)
        new_node._in_edges = []
        new_node._out_edges = []
        node_map[node] = new_node
        cloned_graph.nodes.append(new_node)

    # reconnect edges
    for old_node, new_node in node_map.items():
        for edge in old_node._out_edges:
            new_dst_node = node_map.get(edge.dst_node)
            if new_dst_node:
                new_edge = Edge(new_node, new_dst_node, edge.src_id, edge.dst_id)
                new_node._out_edges.append(new_edge)
                while len(new_dst_node._in_edges) <= edge.dst_id:
                    new_dst_node._in_edges.append(None)
                new_dst_node._in_edges[edge.dst_id] = new_edge

    # `None` entries indicate an invalid cloned graph and should not be silently kept.
    for node in cloned_graph.nodes:
        if any(edge is None for edge in node._in_edges):
            raise ValueError(
                f"clone_graph failed to preserve all inputs for node {getattr(node, 'name', node)}"
            )

    # Reconnecting from source nodes can leave inputs in producer/topological
    # order; restore the operand order recorded by each edge.
    for new_node in cloned_graph.nodes:
        new_node._in_edges.sort(key=lambda edge: edge.dst_id)

    return cloned_graph


def lock_graph(graph):
    from ..graph import GraphNode

    graph.locked = True
    for node in graph.nodes:
        if isinstance(node, GraphNode):
            lock_graph(node.graph)


def unlock_graph(graph):
    from ..graph import GraphNode

    graph.locked = False
    for node in graph.nodes:
        if isinstance(node, GraphNode):
            unlock_graph(node.graph)
