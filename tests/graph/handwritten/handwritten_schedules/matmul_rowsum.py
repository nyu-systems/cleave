from tilelang.carver.roller import Edge

from cleave import passes
from cleave.graph import Graph, Iterator, GraphNode, OutputNode, AccumEdge
from cleave.util import gather_all_nodes, lock_graph
from cleave.ops import Matmul, Reduction

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True


def get_handwritten_scheduled_graph(dims):
    M, N, K = dims
    PM, PN, PK = 128, 1024, 1024
    IM, IN, IK = 128, 128, 1024
    IIM, IIN, IIK = 128, 128, 32

    inner_inner_node_0 = Matmul(
        (M, N, K),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        warp_policy="rows",
    )
    inner_inner_nodes = gather_all_nodes(inner_inner_node_0)
    inner_inner_graph = Graph(nodes=inner_inner_nodes)
    lock_graph(inner_inner_graph)

    inner_node_0 = GraphNode(name="GraphNode_0", graph=inner_inner_graph)
    inner_node_1 = Reduction(
        (M, N), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )

    # connect the two inner nodes
    edge = AccumEdge(inner_node_0, inner_node_1, 0, 0, accum_node=inner_inner_node_0)
    inner_node_0._out_edges.append(edge)
    inner_node_1.set_inputs(0, edge)

    inner_nodes = gather_all_nodes(inner_node_0, inner_node_1)
    inner_graph = Graph(nodes=inner_nodes)
    lock_graph(inner_graph)

    node_0 = GraphNode(name="GraphNode_1", graph=inner_graph)

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
    graph.tmap[node_0] = Iterator(
        lambda i: (f"{PM} * {i}",),
        shape=(M,),
        tile_size=(PM,),
    )  # GraphNode_1
    graph.tmap[node_0.outputs[0].dst_node] = Iterator(
        lambda i: (f"{PM} * {i}",),
        shape=(M,),
        tile_size=(PM,),
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
    inner_graph.tmap[inner_node_0] = Iterator(
        lambda i: ("0", f"{IN} * {i}"),
        shape=(PM, PN),
        tile_size=(IM, IN),
    )  # GraphNode_0
    inner_graph.tmap[inner_node_1] = Iterator(
        lambda i: ("0",),
        shape=(PM,),
        tile_size=(IM,),
    )  # RowSum_0

    # schedule the inner inner graph
    inner_inner_graph.extent = IK // IIK  # number of iterations for the inner graph

    inner_inner_graph.tmap[inner_inner_node_0.inputs[0].src_node] = Iterator(
        lambda i: ("0", f"{IIK} * {i}"),
        shape=(IM, IK),
        tile_size=(IIM, IIK),
    )  # A
    inner_inner_graph.tmap[inner_inner_node_0.inputs[1].src_node] = Iterator(
        lambda i: (f"{IIK} * {i}", "0"),
        shape=(IK, IN),
        tile_size=(IIK, IIN),
    )  # B
    inner_inner_graph.tmap[inner_inner_node_0] = Iterator(
        lambda i: ("0", "0"),
        shape=(IM, IN),
        tile_size=(IIM, IIN),
    )  # MMA_0

    graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        node_0.inputs[1].src_node: ("PlaceHolder_B", "g"),
        node_0: ("t_GraphNode_1", "r"),
        node_0.outputs[0].dst_node: ("OutputNode", "g"),
    }

    inner_graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        node_0.inputs[1].src_node: ("PlaceHolder_B", "g"),
        inner_node_0.inputs[0].src_node: ("PlaceHolder_AA", "g"),
        inner_node_0.inputs[1].src_node: ("PlaceHolder_BB", "g"),
        inner_node_0: ("t_GraphNode_0", "r"),
        inner_node_1: ("t_rowsum_0", "r"),
    }

    inner_inner_graph.tinfo_map = {
        inner_node_0.inputs[0].src_node: ("PlaceHolder_AA", "g"),
        inner_node_0.inputs[1].src_node: ("PlaceHolder_BB", "g"),
        inner_inner_node_0.inputs[0].src_node: ("PlaceHolder_AAA", "s"),
        inner_inner_node_0.inputs[1].src_node: ("PlaceHolder_BBB", "s"),
        inner_inner_node_0: ("t_mma_0", "r"),
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
    assert graph.tlayout_map[output_nodes[0]] == "linear"

    return graph
