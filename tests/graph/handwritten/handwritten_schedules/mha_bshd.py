from tilelang.carver.roller import Edge

from cleave import passes
from cleave.graph import Graph, Iterator, OutputNode, GraphNode, AccumEdge
from cleave.util import gather_all_nodes, lock_graph
from cleave.ops import Matmul, Mul, Sub, Exp, Reduction, Div

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True


def get_handwritten_scheduled_graph(dims):
    B, S_Q, S_KV, H, D = dims
    PB, PS_Q, PS_KV, PH, PD = 1, 64, 4096, 1, 64
    IB, IS_Q, IS_KV, IH, ID = 1, 64, 64, 1, 64
    inner_node_0 = Matmul(
        (S_Q, S_KV, D),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    inner_node_1 = Mul(
        (S_Q, S_KV),
        const=(1 / (D**0.5)),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mul_0",
    )
    inner_node_2 = Sub(
        (S_Q, S_KV),
        const=(50.0 * (1 / (D**0.5))),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="sub_0",
    )
    inner_node_3 = Exp(
        (S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="exp_0"
    )
    inner_node_4 = Matmul(
        (S_Q, D, S_KV),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
    )
    inner_node_5 = Reduction(
        (S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )

    # connect the two inner nodes
    inner_edge_0 = Edge(inner_node_0, inner_node_1, 0, 0)
    inner_node_0._out_edges.append(inner_edge_0)
    inner_node_1.set_inputs(0, inner_edge_0)
    inner_edge_1 = Edge(inner_node_1, inner_node_2, 0, 0)
    inner_node_1._out_edges.append(inner_edge_1)
    inner_node_2.set_inputs(0, inner_edge_1)
    inner_edge_2 = Edge(inner_node_2, inner_node_3, 0, 0)
    inner_node_2._out_edges.append(inner_edge_2)
    inner_node_3.set_inputs(0, inner_edge_2)
    inner_edge_3 = Edge(inner_node_3, inner_node_4, 0, 0)
    inner_node_3._out_edges.append(inner_edge_3)
    inner_node_4.set_inputs(0, inner_edge_3)
    inner_edge_4 = Edge(inner_node_3, inner_node_5, 0, 0)
    inner_node_3._out_edges.append(inner_edge_4)
    inner_node_5.set_inputs(0, inner_edge_4)

    inner_nodes = gather_all_nodes(
        inner_node_0,
        inner_node_1,
        inner_node_2,
        inner_node_3,
        inner_node_4,
        inner_node_5,
    )
    inner_graph = Graph(nodes=inner_nodes)
    lock_graph(inner_graph)

    node_0 = GraphNode(name="GraphNode_0", graph=inner_graph)
    node_1 = Div(
        (S_Q, D),
        broadcast=0,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )

    edge_0 = AccumEdge(node_0, node_1, 0, 0, accum_node=inner_node_4)
    node_1.set_inputs(0, edge_0)
    node_0._out_edges.append(edge_0)
    edge_1 = AccumEdge(node_0, node_1, 0, 0, accum_node=inner_node_5)
    node_1.set_inputs(1, edge_1)
    node_0._out_edges.append(edge_1)

    output_nodes = [OutputNode(node_1, dtype=OUT_DTYPE)]
    nodes = gather_all_nodes(node_0, node_1)
    graph = Graph(nodes=nodes)
    lock_graph(graph)

    # schedule the main graph
    graph.extent = (
        B // PB,
        S_Q // PS_Q,
        H // PH,
    )

    graph.tmap[node_0.inputs[0].src_node] = Iterator(
        lambda i, j, k: (f"{PB} * {i}", f"{PS_Q} * {j}", f"{PH} * {k}", "0"),
        shape=(B, S_Q, H, D),
        tile_size=(PB, PS_Q, PH, PD),
    )  # Q
    graph.tmap[node_0.inputs[1].src_node] = Iterator(
        lambda i, j, k: (f"{PB} * {i}", "0", f"{PH} * {k}", "0"),
        shape=(B, S_KV, H, D),
        tile_size=(PB, PS_KV, PH, PD),
    )  # K
    graph.tmap[node_0.inputs[2].src_node] = Iterator(
        lambda i, j, k: (f"{PB} * {i}", "0", f"{PH} * {k}", "0"),
        shape=(B, S_KV, H, D),
        tile_size=(PB, PS_KV, PH, PD),
    )  # V
    graph.tmap[node_0] = [
        Iterator(
            lambda i, j, k: (f"{PB} * {i}", f"{PS_Q} * {j}", f"{PH} * {k}", "0"),
            shape=(B, S_Q, H, D),
            tile_size=(PB, PS_Q, PH, PD),
        ),
        Iterator(
            lambda i, j, k: (f"{PB} * {i}", f"{PS_Q} * {j}", f"{PH} * {k}"),
            shape=(B, S_Q, H),
            tile_size=(PB, PS_Q, PH),
        ),
    ]  # GraphNode
    graph.tmap[node_1] = Iterator(
        lambda i, j, k: (f"{PB} * {i}", f"{PS_Q} * {j}", f"{PH} * {k}", "0"),
        shape=(B, S_Q, H, D),
        tile_size=(PB, PS_Q, PH, PD),
    )  # div_0
    graph.tmap[node_1.outputs[0].dst_node] = Iterator(
        lambda i, j, k: (f"{PB} * {i}", f"{PS_Q} * {j}", f"{PH} * {k}", "0"),
        shape=(B, S_Q, H, D),
        tile_size=(PB, PS_Q, PH, PD),
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
    inner_graph.tmap[inner_node_4.inputs[1].src_node] = Iterator(
        lambda i: (f"{IS_KV} * {i}", "0"),
        shape=(PS_KV, PD),
        tile_size=(IS_KV, ID),
    )  # V'
    inner_graph.tmap[inner_node_0] = Iterator(
        lambda i: ("0", f"{IS_KV} * {i}"),
        shape=(PS_Q, PS_KV),
        tile_size=(IS_Q, IS_KV),
    )  # mma_0
    inner_graph.tmap[inner_node_1] = Iterator(
        lambda i: ("0", f"{IS_KV} * {i}"),
        shape=(PS_Q, PS_KV),
        tile_size=(IS_Q, IS_KV),
    )  # mul_0
    inner_graph.tmap[inner_node_2] = Iterator(
        lambda i: ("0", f"{IS_KV} * {i}"),
        shape=(PS_Q, PS_KV),
        tile_size=(IS_Q, IS_KV),
    )  # sub_0
    inner_graph.tmap[inner_node_3] = Iterator(
        lambda i: ("0", f"{IS_KV} * {i}"),
        shape=(PS_Q, PS_KV),
        tile_size=(IS_Q, IS_KV),
    )  # exp_0
    inner_graph.tmap[inner_node_4] = Iterator(
        lambda i: ("0", "0"),
        shape=(PS_Q, PD),
        tile_size=(IS_Q, ID),
    )  # mma_1
    inner_graph.tmap[inner_node_5] = Iterator(
        lambda i: ("0",),
        shape=(PS_Q,),
        tile_size=(IS_Q,),
    )  # rowsum_0

    graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        node_0.inputs[1].src_node: ("PlaceHolder_B", "g"),
        node_0.inputs[2].src_node: ("PlaceHolder_C", "g"),
        node_0: ("t_GraphNode_0", "r"),
        node_1: ("t_div_1", "r"),
        node_1.outputs[0].dst_node: ("OutputNode", "g"),
    }

    inner_graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        node_0.inputs[1].src_node: ("PlaceHolder_B", "g"),
        node_0.inputs[2].src_node: ("PlaceHolder_C", "g"),
        inner_node_0.inputs[0].src_node: ("PlaceHolder_AA", "s"),
        inner_node_0.inputs[1].src_node: ("PlaceHolder_BB", "s"),
        inner_node_4.inputs[1].src_node: ("PlaceHolder_CC", "s"),
        inner_node_0: ("t_mma_0", "r"),
        inner_node_1: ("t_mul_0", "r"),
        inner_node_2: ("t_sub_0", "r"),
        inner_node_3: ("t_exp_0", "r"),
        inner_node_4: ("t_mma_1", "r"),
        inner_node_5: ("t_rowsum_0", "r"),
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
