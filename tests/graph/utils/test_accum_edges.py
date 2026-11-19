"""An ``AccumEdge`` names its accumulator by index: ``src_id == accum_nodes.index(...)`` (CPU).

``GraphNode._assign_accum_nodes`` used to number each edge by how many edges the loop already
had (``len(self.outputs)``). That is the accumulator index only while every accumulator has
exactly one reader outside the loop, which every shipped graph happened to satisfy. With a
second reader the numbers run ahead: the second reader of accumulator 0 was told to read
accumulator 1, and the reader of accumulator 1 an output the loop does not have. ``src_id``
is what ``_resolve_output_binding`` (output binding, ``_emit_output``), the reference and
the tile-shape lookups go by.
"""

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.ir.graph import AccumEdge, Graph, GraphNode, OutputNode
from cleave.ir.graph_utils import gather_all_nodes
from cleave.ir.ops import Div, Exp, Matmul, ReduceSum, wire_input

S_Q, S_KV, D = 64, 256, 64


def _loop():
    """``P = exp(S)``; the loop carries ``l = rowsum(P)`` (accumulator 0) and ``O = P @ V``
    (accumulator 1). Outside it, ``l`` is read twice (by ``O / l`` and by an output of its
    own, as an attention kernel that also returns its normaliser would) and ``O`` once."""
    exp = Exp((S_Q, S_KV), name="exp_0")
    rowsum = ReduceSum((S_Q, S_KV), name="rowsum_0")
    value = Matmul((S_Q, D, S_KV), name="mma_1")
    div = Div((S_Q, D), broadcast=0, name="div_0")
    wire_input(exp, rowsum, 0)
    wire_input(exp, value, 0)
    wire_input(value, div, 0)
    wire_input(rowsum, div, 1)
    out_div = OutputNode(div, dtype="float32")
    out_rowsum = OutputNode(rowsum, dtype="float32")

    # As apply_tiling builds a loop: the reductions and their ancestors, readers left outside.
    inner = Graph(nodes=gather_all_nodes(exp, rowsum, value, descending=False))
    loop = GraphNode(name="GraphNode_rowsum_0", graph=inner)
    return loop, rowsum, value, div, out_div, out_rowsum


def test_accumulators_are_the_two_reductions_in_order():
    loop, rowsum, value, *_ = _loop()
    assert [n.name for n in loop.graph.accum_nodes] == ["rowsum_0", "mma_1"]
    assert loop.graph.accum_nodes[0] is rowsum and loop.graph.accum_nodes[1] is value


def test_every_accum_edge_carries_its_accumulator_index():
    loop, rowsum, value, div, out_div, out_rowsum = _loop()
    edges = list(loop.outputs)
    assert all(isinstance(e, AccumEdge) for e in edges)
    readers = sorted((e.accum_node.name, e.dst_node.name, e.dst_id) for e in edges)
    assert readers == [
        ("mma_1", "div_0", 0),
        ("rowsum_0", "OutputNode", 0),
        ("rowsum_0", "div_0", 1),
    ], "the case needs accumulator 0 read twice outside the loop and accumulator 1 once"

    for edge in edges:
        index = next(i for i, n in enumerate(loop.graph.accum_nodes) if n is edge.accum_node)
        where = f"{edge.accum_node.name} -> {edge.dst_node.name}[{edge.dst_id}]"
        assert edge.src_id == index, f"{where}: src_id {edge.src_id}, accumulator {index}"
        bound, shape, _ = loop._resolve_output_binding(edge.src_id)
        assert bound is edge.accum_node, f"{where}: resolves to {bound.name}"
        assert list(shape) == list(edge.accum_node.get_shape(0)), where
        assert list(loop.get_shape(edge.src_id)) == list(edge.accum_node.get_shape(0)), where


def test_an_output_reading_a_shared_accumulator_binds_to_it():
    loop, rowsum, value, div, out_div, out_rowsum = _loop()
    edge = out_rowsum.inputs[0]
    assert isinstance(edge, AccumEdge) and edge.src_node is loop
    # What `_emit_output` and `_get_output_binding_node` do with the edge.
    bound, _, _ = loop._resolve_output_binding(edge.src_id)
    assert bound is rowsum, f"the l output would be written from {bound.name}"
    assert div.inputs[0].src_id == 1 and div.inputs[1].src_id == 0
