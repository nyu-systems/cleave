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


def graph_tiers(graph):
    """``graph`` and every loop tier nested in it, outermost first, as ``(tier, owner)``.

    ``owner`` is the ``GraphNode`` that holds the tier (``None`` for ``graph`` itself).
    """
    from .graph import GraphNode

    tiers = [(graph, None)]
    for tier, _ in tiers:
        tiers.extend((n.graph, n) for n in tier.nodes if isinstance(n, GraphNode))
    return tiers


# Every dict-typed ``Graph`` field is in exactly one of these two lists (a test checks it).
# Node-keyed dicts have to be rebuilt after a structural edit (see ``rekey_graph``);
# name-keyed ones need nothing.
_NODE_KEYED_FIELDS = ("tmap", "tinfo_map", "tlayout_map", "accum_scale")
_NAME_KEYED_FIELDS = ("binding", "smap", "pass_configs")


def rekey_graph(graph):
    """Rebuild every node-keyed dict of every tier from its own items, in place.

    ``GraphNode`` hashes its inner node list and ``OutputNode`` the node it reads, so a
    structural edit of a tier (a node added to a loop, an output rewired) moves the hash
    of the ``GraphNode`` that owns the tier and of every ``GraphNode`` around it. A dict
    filled before the edit then still *holds* those keys but can no longer *find* them.
    Iterating a dict does not hash, so the stale dict can be read out in full and
    refilled under the current hashes: nothing is lost, and no entry has to be looked up
    while it is unfindable.
    """
    for tier, _ in graph_tiers(graph):
        for name in _NODE_KEYED_FIELDS:
            value = getattr(tier, name)
            for mapping in value if isinstance(value, list) else [value]:
                if not isinstance(mapping, dict) or not mapping:
                    continue
                items = list(mapping.items())
                mapping.clear()
                mapping.update(items)
                # One node stored twice (once under a hash it no longer has) collapses,
                # the later entry winning. Two DIFFERENT nodes must never collapse.
                assert len(mapping) == len({id(k) for k, _ in items}), (
                    f"rekey_graph: two distinct `{name}` keys became equal after the edit"
                )


class edit_graph:
    """``with edit_graph(tiled):`` -- the one way to change a graph after ``lock_graph``.

    ``apply_tiling`` returns a locked graph: its node-keyed dicts may hold keys whose
    hash depends on the structure (see :func:`rekey_graph`), and ``GraphNode.__hash__``
    refuses to run on an unlocked graph for that reason. A post-tiling rewrite therefore
    works inside this block: the graph is unlocked on entry, and on exit it is locked
    again and every node-keyed dict is rebuilt, so the caller gets back a graph that
    satisfies the contract ``apply_tiling`` established.

    Inside the block: look nodes up by *name* (``tmap_by_name``), never through a
    node-keyed dict -- a key whose hash has moved is unfindable until the block exits.
    Compute nodes and placeholders hash by name and are always safe to insert.
    """

    def __init__(self, graph):
        self.graph = graph

    def __enter__(self):
        assert self.graph.locked, "edit_graph expects a locked (tiled) graph."
        unlock_graph(self.graph)
        return self.graph

    def __exit__(self, exc_type, exc, tb):
        lock_graph(self.graph)
        if exc_type is None:
            rekey_graph(self.graph)
        return False


def tmap_by_name(graph):
    """``{node.name: Iterator}`` of one tier; safe to read while the graph is being edited."""
    return {node.name: it for node, it in graph.tmap.items()}
