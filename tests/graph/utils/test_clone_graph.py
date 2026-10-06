"""Regression test for clone_graph operand-order preservation.

clone_graph used to rebuild each node's in-edges by appending them in source-node
(topological) order, ignoring the wired operand index (dst_id). When an op's
operand 0 is another op (late in topo order) while operand 1 is a placeholder that
is *shared* with an earlier op (and therefore appears early in topo order) -- e.g.
the shared-K matmuls in the backward dQ graph -- this swapped the A and B operands
after cloning, producing a matmul with mismatched contraction dims.
"""

from tilelang.carver.roller.node import Edge

from cleave.graph import OutputNode, Graph
from cleave.ops import Matmul, Mul
from cleave.util import gather_all_nodes, find_topo_sort, clone_graph


def _connect(src, dst, src_id, dst_id):
    edge = Edge(src, dst, src_id, dst_id)
    src._out_edges.append(edge)
    dst.set_inputs(dst_id, edge)
    return edge


def _build_shared_placeholder_graph():
    BH, M, N, K = 1, 16, 16, 16

    # qk = Q @ Kpᵀ ; Kp is the placeholder we will SHARE with the second matmul,
    # so Kp is visited early in topological order (via this first matmul).
    qk = Matmul((BH, M, N, K), name="qk", transpose_B=True)
    Kp = qk.inputs[1].src_node

    # a_op consumes qk, so it is necessarily LATE in topological order.
    a_op = Mul((BH, M, N), const=1.0, name="a_op")
    _connect(qk, a_op, 0, 0)

    # mm = a_op @ Kp : operand 0 is a_op (late), operand 1 is the shared Kp (early).
    mm = Matmul((BH, M, K, N), name="mm")
    mm.inputs[0].src_node._out_edges.clear()
    mm.inputs[1].src_node._out_edges.clear()
    _connect(a_op, mm, 0, 0)
    _connect(Kp, mm, 0, 1)

    OutputNode(mm, dtype="float16")
    nodes = gather_all_nodes(qk, a_op, mm)
    return Graph(nodes=find_topo_sort([n for n in nodes if n.is_output()])), Kp.name


def test_clone_graph_preserves_operand_order():
    graph, shared_placeholder_name = _build_shared_placeholder_graph()

    mm_before = next(n for n in graph.nodes if getattr(n, "name", "") == "mm")
    assert [e.dst_id for e in mm_before.inputs] == [0, 1]
    assert mm_before.inputs[0].src_node.name == "a_op"

    cloned = clone_graph(graph)
    mm_after = next(n for n in cloned.nodes if getattr(n, "name", "") == "mm")

    # Inputs must stay ordered by dst_id (operand index), not producer topo order.
    assert [e.dst_id for e in mm_after.inputs] == [0, 1], (
        f"clone_graph reordered matmul operands: dst_ids={[e.dst_id for e in mm_after.inputs]}"
    )
    assert [edge.src_node.name for edge in mm_after.inputs] == [
        "a_op",
        shared_placeholder_name,
    ]
