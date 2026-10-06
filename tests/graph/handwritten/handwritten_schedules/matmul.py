from cleave import passes
from cleave.graph import Graph, Iterator, GraphNode, OutputNode
from cleave.ops import Matmul
from cleave.util import gather_all_nodes, lock_graph

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True


def get_handwritten_scheduled_graph(dims):
    M, N, K = dims
    PM, PN = 64, 64
    IM, IN, IK = 64, 64, 32

    assert M % PM == 0, (M, PM)
    assert N % PN == 0, (N, PN)
    assert K % IK == 0, (K, IK)

    inner_node_0 = Matmul(
        (M, N, K),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        warp_policy="rows",
    )
    inner_nodes = gather_all_nodes(inner_node_0)
    inner_graph = Graph(nodes=inner_nodes)
    lock_graph(inner_graph)

    node_0 = GraphNode(name="GraphNode_0", graph=inner_graph)

    output_nodes = [OutputNode(node_0, dtype=OUT_DTYPE)]
    nodes = gather_all_nodes(node_0)
    graph = Graph(nodes=nodes)
    lock_graph(graph)

    # schedule the main graph (2D grid over M, N tiles)
    graph.extent = (M // PM, N // PN)

    graph.tmap[node_0.inputs[0].src_node] = Iterator(
        lambda i0, i1: (f"{PM} * {i0}", "0"),
        shape=(M, K),
        tile_size=(PM, K),
    )  # A
    graph.tmap[node_0.inputs[1].src_node] = Iterator(
        lambda i0, i1: ("0", f"{PN} * {i1}"),
        shape=(K, N),
        tile_size=(K, PN),
    )  # B
    graph.tmap[node_0] = Iterator(
        lambda i0, i1: (f"{PM} * {i0}", f"{PN} * {i1}"),
        shape=(M, N),
        tile_size=(PM, PN),
    )  # GraphNode
    graph.tmap[node_0.outputs[0].dst_node] = Iterator(
        lambda i0, i1: (f"{PM} * {i0}", f"{PN} * {i1}"),
        shape=(M, N),
        tile_size=(PM, PN),
    )  # Output

    # schedule the inner graph (K reduction tiles)
    inner_graph.extent = K // IK

    inner_graph.tmap[inner_node_0.inputs[0].src_node] = Iterator(
        lambda i: ("0", f"{IK} * {i}"),
        shape=(PM, K),
        tile_size=(IM, IK),
    )  # A
    inner_graph.tmap[inner_node_0.inputs[1].src_node] = Iterator(
        lambda i: (f"{IK} * {i}", "0"),
        shape=(K, PN),
        tile_size=(IK, IN),
    )  # B
    inner_graph.tmap[inner_node_0] = Iterator(
        lambda i: ("0", "0"),
        shape=(PM, PN),
        tile_size=(IM, IN),
    )  # MMA_0

    graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        node_0.inputs[1].src_node: ("PlaceHolder_B", "g"),
        node_0: ("t_GraphNode_0", "r"),
        node_0.outputs[0].dst_node: ("OutputNode", "g"),
    }

    inner_graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        node_0.inputs[1].src_node: ("PlaceHolder_B", "g"),
        inner_node_0.inputs[0].src_node: ("PlaceHolder_AA", "s"),
        inner_node_0.inputs[1].src_node: ("PlaceHolder_BB", "s"),
        inner_node_0: ("t_mma_0", "r"),
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
