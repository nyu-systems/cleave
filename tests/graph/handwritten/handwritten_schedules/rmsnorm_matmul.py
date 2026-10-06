from tilelang.carver.roller import Edge

from cleave import passes
from cleave.graph import Graph, Iterator, GraphNode, OutputNode, AccumEdge
from cleave.util import gather_all_nodes, lock_graph
from cleave.ops import Matmul, Reduction, Mul, Div, RSqrt

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True


def get_handwritten_scheduled_graph(dims):
    M, N, K = dims
    PM, PN, PK = 128, 128, 4096
    IM, IN, IK = 128, 128, 32
    inner_node_0 = Mul((M, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mul_0")
    inner_node_1 = Reduction(
        (M, K), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    inner_node_2 = Matmul(
        (M, N, K),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        warp_policy="rows",
    )

    # do square
    inner_node_0.set_inputs(1, inner_node_0.inputs[0])

    inner_edge_0 = Edge(inner_node_0, inner_node_1, 0, 0)
    inner_node_0._out_edges.append(inner_edge_0)
    inner_node_1.set_inputs(0, inner_edge_0)

    inner_edge_1 = Edge(inner_node_0.inputs[0].src_node, inner_node_2, 0, 0)
    inner_node_2.set_inputs(0, inner_edge_1)
    inner_node_0.inputs[0].src_node._out_edges.append(inner_edge_1)

    inner_nodes = gather_all_nodes(inner_node_0, inner_node_1, inner_node_2)
    inner_graph = Graph(nodes=inner_nodes)
    lock_graph(inner_graph)

    node_0 = GraphNode(name="GraphNode_0", graph=inner_graph)
    node_1 = Div(
        (M,), const=float(K), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="div_0"
    )
    node_2 = RSqrt((M,), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rsqrt_0")
    node_3 = Mul(
        (M, N), broadcast=0, dtype=ACCUM_DTYPE, out_dtype=OUT_DTYPE, name="mul_1"
    )

    # connect the nodes
    edge_0 = AccumEdge(node_0, node_1, 0, 0, accum_node=inner_node_1)
    node_0._out_edges.append(edge_0)
    node_1.set_inputs(0, edge_0)

    edge_1 = Edge(node_1, node_2, 0, 0)
    node_1._out_edges.append(edge_1)
    node_2.set_inputs(0, edge_1)

    edge_2 = AccumEdge(node_0, node_3, 0, 0, accum_node=inner_node_2)
    node_0._out_edges.append(edge_2)
    node_3.set_inputs(0, edge_2)
    edge_3 = Edge(node_2, node_3, 0, 0)
    node_2._out_edges.append(edge_3)
    node_3.set_inputs(1, edge_3)

    output_nodes = [OutputNode(node_3, dtype=OUT_DTYPE)]
    nodes = gather_all_nodes(node_0, node_1, node_2, node_3)
    graph = Graph(nodes=nodes)
    lock_graph(graph)

    # schedule the main graph
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
    graph.tmap[node_0] = [
        Iterator(
            lambda i, j: (f"{PM} * {i}",),
            shape=(M,),
            tile_size=(PM,),
        ),
        Iterator(
            lambda i, j: (f"{PM} * {i}", f"{PN} * {j}"),
            shape=(M, N),
            tile_size=(PM, PN),
        ),
    ]  # GraphNode
    graph.tmap[node_1] = Iterator(
        lambda i, j: (f"{PM} * {i}",),
        shape=(M,),
        tile_size=(PM,),
    )  # Div
    graph.tmap[node_2] = Iterator(
        lambda i, j: (f"{PM} * {i}",),
        shape=(M,),
        tile_size=(PM,),
    )  # RSqrt
    graph.tmap[node_3] = Iterator(
        lambda i, j: (f"{PM} * {i}", f"{PN} * {j}"),
        shape=(M, N),
        tile_size=(PM, PN),
    )  # Mul
    graph.tmap[node_3.outputs[0].dst_node] = Iterator(
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
    inner_graph.tmap[inner_node_2.inputs[1].src_node] = Iterator(
        lambda i: (f"{IK} * {i}", "0"),
        shape=(PK, PN),
        tile_size=(IK, IN),
    )  # B
    inner_graph.tmap[inner_node_0] = Iterator(
        lambda i: ("0", f"{IK} * {i}"),
        shape=(PM, PK),
        tile_size=(IM, IK),
    )  # mul_0
    inner_graph.tmap[inner_node_1] = Iterator(
        lambda i: ("0",),
        shape=(PM,),
        tile_size=(IM,),
    )  # rowsum_0
    inner_graph.tmap[inner_node_2] = Iterator(
        lambda i: ("0", "0"),
        shape=(PM, PN),
        tile_size=(IM, IN),
    )  # mma_0

    graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        node_0.inputs[1].src_node: ("PlaceHolder_B", "g"),
        node_0: ("t_GraphNode_0", "r"),
        node_1: ("t_div_0", "r"),
        node_2: ("t_rsqrt_0", "r"),
        node_3: ("t_mul_1", "r"),
        node_3.outputs[0].dst_node: ("OutputNode", "g"),
    }

    inner_graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        node_0.inputs[1].src_node: ("PlaceHolder_B", "g"),
        inner_node_0.inputs[0].src_node: ("PlaceHolder_AA", "r"),
        inner_node_2.inputs[1].src_node: ("PlaceHolder_BB", "s"),
        inner_node_0: ("t_mul_0", "r"),
        inner_node_1: ("t_rowsum_0", "r"),
        inner_node_2: ("t_mma_0", "r"),
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
    assert graph.tlayout_map[output_nodes[0]] == "tc"

    return graph
