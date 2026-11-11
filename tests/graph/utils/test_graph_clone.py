"""Graph.clone: every field copied, no mutable state shared, codegen unchanged."""

from dataclasses import fields

import pytest

pytest.importorskip("tilelang")

from cleave.ir.graph import Graph, GraphNode, OutputNode, PlaceHolderNode
from cleave.ir.ops import Matmul, Mul, wire_input
from cleave.ir.graph_utils import clone_graph, find_topo_sort, gather_all_nodes
from tests.graph.handwritten._cases import CASES
from tests.graph.handwritten._test_utils import gen_handwritten_code


def _all_nodes(graph):
    for node in graph.nodes + graph.else_nodes:
        yield node
        if isinstance(node, GraphNode):
            yield from _all_nodes(node.graph)


def _simple_graph():
    mm = Matmul((16, 16, 16), name="mm")
    scale = Mul((16, 16), const=2.0, name="scale")
    wire_input(mm, scale, 0)
    out = OutputNode(scale, dtype="float32")
    return Graph(nodes=find_topo_sort(gather_all_nodes(out)))


def test_clone_copies_every_field_and_owns_its_state():
    graph = _simple_graph()
    graph.sargs = [("s", "int32")]
    graph.smap = {"s": lambda i: i}
    graph.loop_kind, graph.rasterization = "serial", 1
    graph.args = [n for n in graph.nodes if isinstance(n, (PlaceHolderNode, OutputNode))]
    graph.tinfo_map = {n: (n.name, "g") for n in graph.nodes}
    mm = next(n for n in graph.nodes if isinstance(n, Matmul))
    space = list(mm.get_space_dim())  # memoized before the clone
    clone = clone_graph(graph)

    by_id = {id(n) for n in graph.nodes}
    assert not any(id(n) in by_id for n in clone.nodes)
    for f in fields(Graph):
        assert getattr(clone, f.name) == getattr(graph, f.name), f.name
    assert all(any(c is n for n in clone.nodes) for c in clone.args)
    assert all(any(k is n for n in clone.nodes) for k in clone.tinfo_map)
    assert clone.sargs is not graph.sargs and clone.smap is not graph.smap

    for old, new in zip(graph.nodes, clone.nodes):
        for attr in ("_tag", "_shapes", "_dtypes"):
            assert getattr(new, attr) == getattr(old, attr)
            assert getattr(new, attr) is not getattr(old, attr), (old.name, attr)
        assert all(e.src_node is new or e.src_node in clone.nodes for e in new.inputs)
        assert all(e.dst_node in clone.nodes for e in new.outputs)

    mm_clone = next(n for n in clone.nodes if isinstance(n, Matmul))
    assert mm_clone.get_space_dim() == space
    mm_clone.set_tag("probe", 1)
    mm_clone.get_space_dim().append(0)
    assert mm.get_tag("probe") is None and mm.get_space_dim() == space


@pytest.mark.parametrize("name", sorted(CASES))
def test_clone_generates_the_same_kernel(name):
    graphs = CASES[name].graph()
    for graph in graphs if isinstance(graphs, list) else [graphs]:
        clone = graph.clone()
        old_ids = {id(n) for n in _all_nodes(graph)}
        assert not any(id(n) in old_ids for n in _all_nodes(clone))
        # Codegen mutates its graph, so the clone goes first: the original must not notice.
        assert gen_handwritten_code(clone) == gen_handwritten_code(graph)
