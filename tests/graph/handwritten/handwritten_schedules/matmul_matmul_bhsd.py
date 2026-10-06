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
    B, H, S_Q, S_KV, D = dims
    PB, PH, PS_Q, PS_KV, PD = 1, 1, 64, 4096, 64
    IB, IH, IS_Q, IS_KV, ID = 1, 1, 64, 64, 64
    inner_node_0 = Matmul(
        (B * H, S_Q, S_KV, D),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    inner_node_1 = Matmul(
        (B * H, S_Q, D, S_KV),
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
    graph.extent = (
        H // PH,
        S_Q // PS_Q,
    )

    graph.tmap[node_0.inputs[0].src_node] = Iterator(
        lambda i, j: (f"{PH} * {i}", f"{PS_Q} * {j}", "0"),
        shape=(B * H, S_Q, D),
        tile_size=(PB * PH, PS_Q, PD),
    )  # Q
    graph.tmap[node_0.inputs[1].src_node] = Iterator(
        lambda i, j: (f"{PH} * {i}", "0", "0"),
        shape=(B * H, S_KV, D),
        tile_size=(PB * PH, PS_KV, PD),
    )  # K
    graph.tmap[node_0.inputs[2].src_node] = Iterator(
        lambda i, j: (f"{PH} * {i}", "0", "0"),
        shape=(B * H, S_KV, D),
        tile_size=(PB * PH, PS_KV, PD),
    )  # V
    graph.tmap[node_0] = Iterator(
        lambda i, j: (f"{PH} * {i}", f"{PS_Q} * {j}", "0"),
        shape=(B * H, S_Q, D),
        tile_size=(PB * PH, PS_Q, PD),
    )  # GraphNode
    graph.tmap[node_0.outputs[0].dst_node] = Iterator(
        lambda i, j: (f"{PH} * {i}", f"{PS_Q} * {j}", "0"),
        shape=(B * H, S_Q, D),
        tile_size=(PB * PH, PS_Q, PD),
    )  # Output

    # schedule the inner graph
    inner_graph.extent = PS_KV // IS_KV

    inner_graph.tmap[inner_node_0.inputs[0].src_node] = Iterator(
        lambda i: ("0", "0"),
        shape=(PS_Q, PD),
        tile_size=(IS_Q, ID),
    )  # Q'
    inner_graph.tmap[inner_node_0.inputs[1].src_node] = Iterator(
        lambda i: (f"{IS_KV} * {i}", "0"),
        shape=(PS_KV, PD),
        tile_size=(IS_KV, ID),
    )  # K'
    inner_graph.tmap[inner_node_1.inputs[1].src_node] = Iterator(
        lambda i: (f"{IS_KV} * {i}", "0"),
        shape=(PS_KV, PD),
        tile_size=(IS_KV, ID),
    )  # V'
    inner_graph.tmap[inner_node_0] = Iterator(
        lambda i: ("0", f"{IS_KV} * {i}"),
        shape=(PS_Q, PS_KV),
        tile_size=(IS_Q, IS_KV),
    )  # MMA_0
    inner_graph.tmap[inner_node_1] = Iterator(
        lambda i: ("0", "0"),
        shape=(PS_Q, PD),
        tile_size=(IS_Q, ID),
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
