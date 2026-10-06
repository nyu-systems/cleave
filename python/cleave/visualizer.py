# first draft by GPT
import traceback
import graphviz as gv
from typing import List, Tuple, Optional

from .graph import Graph, GraphNode, PlaceHolderNode, OutputNode, Iterator
from tilelang.carver.roller import Edge
from tilelang.carver.roller.node import Node


colors_map = {
    "block": {
        "node": "#70a148",
        "bg": "#e0edd5",
        "edge": "#527536",
        "edge_label": "black",
        "io": "#527536",
    },
    "iterative": {
        "node": "#5a8fcb",
        "bg": "#dbe8f5",
        "edge": "#4273b1",
        "edge_label": "black",
        "io": "#527536",
    },
}

node_font_size = "30"
edge_font_size = "30"
tensor_node_font_size = "23"
graph_label_font_size = "32"


def draw_edge(
    G: gv.Digraph,
    src: str,
    dst: str,
    graph_type: str = "block",
    label: Optional[str] = None,
):
    attrs = {
        "color": colors_map[graph_type]["edge"],
        "penwidth": "3",
        "fontname": "sans-serif",
        "fontsize": edge_font_size,
        "fontcolor": colors_map[graph_type]["edge_label"],
        "label": label,
    }
    G.edge(src, dst, **attrs)


def format_iterator(it: Optional[Iterator]) -> str:
    # Return offset and tile size for IO node labels in block graph.
    if it is None:
        return ""
    parts: List[str] = []
    try:
        # offset can be a callable
        # the # of iterator variables can be 1-3, e.g., (i,), (i,j), (i,j,k). should handle this

        # get num args of callable
        assert callable(it.offset), f"{it.offset=} is not callable"
        num_args = it.offset.__code__.co_argcount
        iter_letters = ["i", "j", "k"]
        off = it.offset(*iter_letters[:num_args])
        off_s = ", ".join(map(str, off)) if off is not None else ""
        if off_s:
            parts.append(f"offset: ({off_s})")
    except Exception:
        traceback.print_exc()
        pass

    sh = it.shape
    assert sh is not None
    parts.append(f"shape: {sh}")
    ts = it.tile_size
    assert ts is not None
    parts.append(f"tile: {ts}")
    return "\n".join(parts)


class Visualizer:
    """Graphviz visualizer for hierarchical tilelang Graph objects.

    Usage:
      from graph_tests.visualizer import Visualizer
      v = Visualizer()
      v.draw_graph(graph, "out/graph")

    The top-level graph passed is considered the "block" graph (parallel tiles).
    Any nested Graph reachable through GraphNode.graph is considered an "iterative" graph.
    """

    def __init__(self):
        self.G = None

    def _node_id(self, node: Node) -> str:
        # Unique id for nodes (human-friendly)
        return f"{node.name}_{id(node)}"

    def _is_graphnode(self, node: Node) -> bool:
        return isinstance(node, GraphNode)

    def _is_placeholder(self, node: Node) -> bool:
        return isinstance(node, PlaceHolderNode)

    def _is_outputnode(self, node: Node) -> bool:
        return isinstance(node, OutputNode)

    def _get_iter(self, graph: Graph, node: Node) -> Optional[Iterator]:
        # return iterator object if exists in graph.tmap for given node
        if isinstance(graph.tmap, dict):
            return graph.tmap.get(node, None)
        return None

    def _label_for_op(self, node: Node) -> str:
        # Prefer using node.name if exists, or class name
        return node.name

    def _collect_edges(self, node: Node) -> List[Edge]:
        # Prefer the public outputs API if available
        return list(node.outputs)

    def _edge_dst(self, edge: Edge) -> Node:
        return edge.dst_node

    def _edge_src(self, edge: Edge) -> Node:
        return edge.src_node

    def _edge_shape_label(
        self, edge: Edge, cur_graph: Optional[Graph] = None
    ) -> Optional[str]:
        # Fallback to src node shape
        src = self._edge_src(edge)
        src_id = edge.src_id  # type: ignore[attr-defined]
        if src._shapes == []:
            return None
        val = str(src.get_shape(src_id))
        return val

    def draw_graph(
        self, graph: Graph, out_path: str, dot: bool = True, png: bool = True
    ):
        # initialize Digraph
        self.G = gv.Digraph(format="png", name="Hierarchical Graph")
        self.G.attr(
            rankdir="LR",
            splines="ortho",
            bgcolor="#ffffff",
            fontname="sans-serif",
            nodesep="1.6",
            ranksep="0.3",
            fontsize="16",
            fontcolor="black",
            compound="true",
        )

        # Draw top-level (block) graph first
        queue: List[Tuple[Graph, str, str]] = []
        nested = self._draw_single_graph(graph, label="Block Graph", graph_type="block")
        queue.extend(nested)
        # Draw nested graphs as separate clusters afterwards
        while queue:
            g, lbl, gtype = queue.pop(0)
            nested_more = self._draw_single_graph(g, label=lbl, graph_type=gtype)
            queue.extend(nested_more)

        if dot:
            self.G.save(out_path + ".dot")
            print(f"Graph saved as {out_path}.dot")
        if png:
            self.G.render(out_path, cleanup=True)
            print(f"Graph saved as {out_path}.png")

    def _draw_single_graph(
        self, graph: Graph, label: str, graph_type: str
    ) -> List[Tuple[Graph, str, str]]:
        # Create a subgraph cluster for this graph. Sanitize name to avoid illegal chars.
        safe_label_hash = abs(hash(label))
        cluster_name = f"cluster_{id(graph)}_{safe_label_hash}"
        nested_graphs: List[Tuple[Graph, str, str]] = []
        with self.G.subgraph(name=cluster_name) as sub:
            # graph-level label: include extent only
            extent_suffix = (
                f" (extent={graph.extent})" if graph.extent is not None else ""
            )
            sub.attr(
                rankdir="LR",
                splines="ortho",
                bgcolor=colors_map[graph_type]["bg"],
                fontname="sans-serif",
                label=label + extent_suffix,
                labelloc="t",
                labeljust="l",
                labeldistance="1.5",
                fontsize=graph_label_font_size,
                fontcolor="black",
                style="filled",
                penwidth="0",
            )

            # Draw nodes per policy
            for node in graph.nodes:
                nid = self._node_id(node)

                if self._is_placeholder(node) or self._is_outputnode(node):
                    # Inputs/Outputs: in block graph show offset+tile size;
                    label_text = node.name
                    it = self._get_iter(graph, node)
                    it_label = format_iterator(it)
                    if it_label:
                        label_text = label_text + "\n" + it_label
                    sub.node(
                        nid,
                        label=label_text,
                        color=colors_map[graph_type]["io"],
                        style="filled",
                        shape="box",
                        penwidth="0",
                        fontsize=tensor_node_font_size,
                        fontcolor="white",
                        fontname="sans-serif",
                    )
                else:
                    # Internal operators (including GraphNode): operator name only
                    label_text = self._label_for_op(node)
                    sub.node(
                        nid,
                        label=label_text,
                        color=colors_map[graph_type]["node"],
                        style="rounded,filled",
                        shape="box",
                        penwidth="0",
                        fontsize=node_font_size,
                        fontcolor="white",
                        fontname="sans-serif",
                        margin="0.4,0.4",
                    )
                    # If it's a GraphNode, enqueue its nested graph as a separate iterative graph cluster
                    if self._is_graphnode(node):
                        nested_graph = node.graph
                        if nested_graph is not None:
                            nested_graphs.append(
                                (nested_graph, f"Iterative: {label_text}", "iterative")
                            )

            # Draw edges; label shape only for block graph
            for node in graph.nodes:
                src_id = self._node_id(node)
                outs = self._collect_edges(node)
                for e in outs:
                    dst = self._edge_dst(e)
                    dst_id = self._node_id(dst)
                    label = self._edge_shape_label(e, cur_graph=graph)
                    draw_edge(sub, src_id, dst_id, graph_type=graph_type, label=label)
        return nested_graphs


if __name__ == "__main__":
    print(
        "This visualizer is meant to be imported and used with tilelang Graph objects."
    )
