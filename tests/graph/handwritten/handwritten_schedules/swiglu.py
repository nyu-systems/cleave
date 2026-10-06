from tilelang.carver.roller import Edge

from cleave import passes
from cleave.graph import Graph, Iterator, OutputNode, GraphNode, AccumEdge
from cleave.util import gather_all_nodes, lock_graph
from cleave.ops import Matmul, Silu, Mul

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True


def get_handwritten_scheduled_graph(dims):
    M, N, K = dims
    PM, PN, PK = 128, 128, 4096
    IM, IN, IK = 128, 128, 32

    inner_node_0 = Matmul((M, N, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_0")
    inner_node_1 = Matmul((M, N, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_1")

    inner_edge_0 = Edge(inner_node_0.inputs[0].src_node, inner_node_1, 0, 0)
    inner_node_1.set_inputs(0, inner_edge_0)
    inner_node_0.inputs[0].src_node._out_edges.append(inner_edge_0)

    inner_nodes = gather_all_nodes(inner_node_0, inner_node_1)
    inner_graph = Graph(nodes=inner_nodes)
    lock_graph(inner_graph)

    node_0 = GraphNode(name="GraphNode_0", graph=inner_graph)
    node_1 = Silu((M, N), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="silu_0")
    node_2 = Mul((M, N), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="mul_0")

    edge_0 = AccumEdge(node_0, node_1, 0, 0, accum_node=inner_node_0)
    node_1.set_inputs(0, edge_0)
    node_0._out_edges.append(edge_0)
    edge_1 = Edge(node_1, node_2, 0, 0)
    node_2.set_inputs(0, edge_1)
    node_1._out_edges.append(edge_1)
    edge_2 = AccumEdge(node_0, node_2, 0, 0, accum_node=inner_node_1)
    node_2.set_inputs(1, edge_2)
    node_0._out_edges.append(edge_2)

    output_nodes = [OutputNode(node_2, dtype=OUT_DTYPE)]

    nodes = gather_all_nodes(node_0, node_1, node_2)
    graph = Graph(nodes=nodes)
    lock_graph(graph)

    # schedule main graph
    graph.extent = (M // PM, N // PN)

    graph.tmap[node_0.inputs[0].src_node] = Iterator(
        lambda i, j: (f"{PM} * {i}", "0"),
        shape=(M, K),
        tile_size=(PM, PK),
    )  # A
    graph.tmap[node_0.inputs[1].src_node] = Iterator(
        lambda i, j: ("0", f"{PN} * {j}"),
        shape=(K, N),
        tile_size=(PK, PN),
    )  # B
    graph.tmap[node_0.inputs[2].src_node] = Iterator(
        lambda i, j: ("0", f"{PN} * {j}"),
        shape=(K, N),
        tile_size=(PK, PN),
    )  # C
    graph.tmap[node_0] = [
        Iterator(
            lambda i, j: (f"{PM} * {i}", f"{PN} * {j}"),
            shape=(M, N),
            tile_size=(PM, PN),
        ),
        Iterator(
            lambda i, j: (f"{PM} * {i}", f"{PN} * {j}"),
            shape=(M, N),
            tile_size=(PM, PN),
        ),
    ]  # GraphNode (mma_0 and mma_1)
    graph.tmap[node_1] = Iterator(
        lambda i, j: (f"{PM} * {i}", f"{PN} * {j}"),
        shape=(M, N),
        tile_size=(PM, PN),
    )  # silu_0
    graph.tmap[node_2] = Iterator(
        lambda i, j: (f"{PM} * {i}", f"{PN} * {j}"),
        shape=(M, N),
        tile_size=(PM, PN),
    )  # mul_0
    graph.tmap[node_2.outputs[0].dst_node] = Iterator(
        lambda i, j: (f"{PM} * {i}", f"{PN} * {j}"),
        shape=(M, N),
        tile_size=(PM, PN),
    )  # Output

    # schedule the inner graph
    inner_graph.extent = PK // IK  # number of iterations for the inner graph

    inner_graph.tmap[inner_node_0.inputs[0].src_node] = Iterator(
        lambda i: ("0", f"{IK} * {i}"),
        shape=(PM, PK),
        tile_size=(IM, IK),
    )  # A
    inner_graph.tmap[inner_node_0.inputs[1].src_node] = Iterator(
        lambda i: (f"{IK} * {i}", "0"),
        shape=(PK, PN),
        tile_size=(IK, IN),
    )  # B
    inner_graph.tmap[inner_node_1.inputs[1].src_node] = Iterator(
        lambda i: (f"{IK} * {i}", "0"),
        shape=(PK, PN),
        tile_size=(IK, IN),
    )  # C
    inner_graph.tmap[inner_node_0] = Iterator(
        lambda i: ("0", "0"),
        shape=(PM, PN),
        tile_size=(IM, IN),
    )  # mma_0
    inner_graph.tmap[inner_node_1] = Iterator(
        lambda i: ("0", "0"),
        shape=(PM, PN),
        tile_size=(IM, IN),
    )  # mma_1

    graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        node_0.inputs[1].src_node: ("PlaceHolder_B", "g"),
        node_0.inputs[2].src_node: ("PlaceHolder_C", "g"),
        node_0: ("t_GraphNode_0", "r"),
        node_1: ("t_silu_0", "r"),
        node_2: ("t_mul_0", "r"),
        node_2.outputs[0].dst_node: ("OutputNode", "g"),
    }

    inner_graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        node_0.inputs[1].src_node: ("PlaceHolder_B", "g"),
        node_0.inputs[2].src_node: ("PlaceHolder_C", "g"),
        inner_node_0.inputs[0].src_node: (
            "PlaceHolder_AA",
            "s",
        ),  # TODO: why "s" is faster than "r"...?
        inner_node_0.inputs[1].src_node: ("PlaceHolder_CC", "s"),
        inner_node_1.inputs[1].src_node: ("PlaceHolder_BB", "s"),
        inner_node_0: ("t_mma_0", "r"),
        inner_node_1: ("t_mma_1", "r"),
    }

    graph = passes.annotate_args(graph, is_main=True)
    graph = passes.annotate_tc_layout(graph)
    graph = passes.annotate_fragment_layout_conflicts(graph)

    for cur_node in gather_all_nodes(*graph.nodes):
        assert getattr(cur_node, "resolve_operand_layout", None) is None, (
            cur_node.name,
            getattr(cur_node, "resolve_operand_layout", None),
        )

    assert graph.tlayout_map[node_0.inputs[0].src_node] == "tc"
    assert graph.tlayout_map[node_0.inputs[1].src_node] == "tc"
    assert graph.tlayout_map[node_0.inputs[2].src_node] == "tc"
    assert graph.tlayout_map[output_nodes[0]] == "tc"

    return graph
