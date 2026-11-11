from __future__ import annotations

from typing import List

from tilelang.carver.roller.node import Node


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
    return graph_to_clone.clone()


def lock_graph(graph):
    from .graph import GraphNode

    graph.locked = True
    for node in graph.nodes:
        if isinstance(node, GraphNode):
            lock_graph(node.graph)


def unlock_graph(graph):
    from .graph import GraphNode

    graph.locked = False
    for node in graph.nodes:
        if isinstance(node, GraphNode):
            unlock_graph(node.graph)
