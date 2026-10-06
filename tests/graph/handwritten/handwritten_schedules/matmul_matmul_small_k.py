from tilelang.carver.roller import Edge

from cleave import passes
from cleave.graph import Graph, Iterator, OutputNode, GraphNode
from cleave.util import gather_all_nodes, lock_graph
from cleave.ops import Matmul

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True


def get_handwritten_scheduled_graph(dims):
    M, N, K = dims
    PM, PN, PK = 64, 4096, 32
    IM, IN, IK = 64, 64, 32
    inner_node_0 = Matmul(
        (M, N, K),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        warp_policy="rows",
    )
    inner_node_1 = Matmul(
        (M, K, N),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
    )

    # connect the two inner nodes
    edge = Edge(inner_node_0, inner_node_1, 0, 0)
    inner_node_0._out_edges.append(edge)
    inner_node_1.set_inputs(0, edge)

    inner_nodes = gather_all_nodes(inner_node_0, inner_node_1)
    inner_graph = Graph(nodes=inner_nodes)
    lock_graph(inner_graph)

    node_0 = GraphNode(name="GraphNode_0", graph=inner_graph)

    output_nodes = [OutputNode(node_0, dtype=OUT_DTYPE)]
    nodes = gather_all_nodes(node_0)
    graph = Graph(nodes=nodes)
    lock_graph(graph)

    # schedule the main graph
    graph.extent = M // PM

    graph.tmap[node_0.inputs[0].src_node] = Iterator(
        lambda i: (f"{PM} * {i}", "0"),
        shape=(M, K),
        tile_size=(PM, PK),
    )  # A
    graph.tmap[node_0.inputs[1].src_node] = Iterator(
        lambda i: ("0", "0"),
        shape=(K, N),
        tile_size=(PK, PN),
    )  # B
    graph.tmap[node_0.inputs[2].src_node] = Iterator(
        lambda i: ("0", "0"),
        shape=(N, K),
        tile_size=(PN, PK),
    )  # C
    graph.tmap[node_0] = Iterator(
        lambda i: (f"{PM} * {i}", "0"),
        shape=(M, K),
        tile_size=(PM, PK),
    )  # GraphNode
    graph.tmap[node_0.outputs[0].dst_node] = Iterator(
        lambda i: (f"{PM} * {i}", "0"),
        shape=(M, K),
        tile_size=(PM, PK),
    )  # Output

    # schedule the inner graph
    inner_graph.extent = PN // IN  # number of iterations for the inner graph

    inner_graph.tmap[inner_node_0.inputs[0].src_node] = Iterator(
        lambda i: ("0", "0"),
        shape=(PM, PK),
        tile_size=(IM, IK),
    )  # A
    inner_graph.tmap[inner_node_0.inputs[1].src_node] = Iterator(
        lambda i: ("0", f"{IN} * {i}"),
        shape=(PK, PN),
        tile_size=(IK, IN),
    )  # B
    inner_graph.tmap[inner_node_1.inputs[1].src_node] = Iterator(
        lambda i: (f"{IN} * {i}", "0"),
        shape=(PN, PK),
        tile_size=(IN, IK),
    )  # C
    inner_graph.tmap[inner_node_0] = Iterator(
        lambda i: ("0", f"{IN} * {i}"),
        shape=(PM, PN),
        tile_size=(IM, IN),
    )  # MMA_0
    inner_graph.tmap[inner_node_1] = Iterator(
        lambda i: ("0", "0"),
        shape=(PM, PK),
        tile_size=(IM, IK),
    )  # MMA_1

    graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        node_0.inputs[1].src_node: ("PlaceHolder_B", "g"),
        node_0.inputs[2].src_node: ("PlaceHolder_C", "g"),
        node_0: ("t_GraphNode_0", "r"),
        node_0.outputs[0].dst_node: ("OutputNode", "g"),
    }

    inner_graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        node_0.inputs[1].src_node: ("PlaceHolder_B", "g"),
        node_0.inputs[2].src_node: ("PlaceHolder_C", "g"),
        inner_node_0.inputs[0].src_node: ("PlaceHolder_AA", "s"),
        inner_node_0.inputs[1].src_node: ("PlaceHolder_BB", "s"),
        inner_node_1.inputs[1].src_node: ("PlaceHolder_CC", "s"),
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
