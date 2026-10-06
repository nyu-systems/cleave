from tilelang.carver.roller import Edge

from cleave import passes
from cleave.graph import Graph, Iterator, GraphNode, OutputNode, AccumEdge
from cleave.ops import Div, Mul, Reduction, Sqrt
from cleave.util import gather_all_nodes, lock_graph

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True


def get_handwritten_scheduled_graph(dims):
    split_k, M, N, K = dims
    PK, PM, PN = 4, 128, 128
    IK, IM, IN = 2, 128, 128

    assert split_k == PK, split_k
    assert M == 1024, M
    assert N == 6144, N
    assert K == 4096, K

    combine_rowsum = Reduction(
        (split_k, M),
        axis=0,
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="rowsum_combine_0",
    )
    combine_mma = Reduction(
        (split_k, M, N),
        axis=0,
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_combine_0",
    )
    combine_nodes = gather_all_nodes(combine_rowsum, combine_mma)
    combine_graph = Graph(nodes=combine_nodes)
    lock_graph(combine_graph)

    node_0 = GraphNode(name="GraphNode_0", graph=combine_graph)
    node_1 = Div(
        (M,),
        const=float(K),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )
    node_2 = Sqrt((M,), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="sqrt_0")
    node_3 = Div(
        (M,),
        const=1.0,
        const_on_left=True,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_1",
    )
    node_4 = Mul(
        (M, N),
        broadcast=0,
        dtype=ACCUM_DTYPE,
        out_dtype=OUT_DTYPE,
        name="mul_0",
    )

    edge_0 = AccumEdge(node_0, node_1, 0, 0, accum_node=combine_rowsum)
    node_0._out_edges.append(edge_0)
    node_1.set_inputs(0, edge_0)

    edge_1 = Edge(node_1, node_2, 0, 0)
    node_1._out_edges.append(edge_1)
    node_2.set_inputs(0, edge_1)

    edge_2 = Edge(node_2, node_3, 0, 0)
    node_2._out_edges.append(edge_2)
    node_3.set_inputs(0, edge_2)

    edge_3 = AccumEdge(node_0, node_4, 0, 0, accum_node=combine_mma)
    node_0._out_edges.append(edge_3)
    node_4.set_inputs(0, edge_3)

    edge_4 = Edge(node_3, node_4, 0, 1)
    node_3._out_edges.append(edge_4)
    node_4.set_inputs(1, edge_4)

    output_nodes = [OutputNode(node_4, dtype=OUT_DTYPE)]
    nodes = gather_all_nodes(node_0, node_1, node_2, node_3, node_4)
    graph = Graph(nodes=nodes)
    lock_graph(graph)

    graph.extent = (M // PM, N // PN)

    ph_rowsum = node_0.inputs[0].src_node
    ph_partial = node_0.inputs[1].src_node
    out_node = output_nodes[0]

    graph.tmap[ph_rowsum] = Iterator(
        lambda i, j: ("0", f"{PM} * {i}"),
        shape=(split_k, M),
        tile_size=(PK, PM),
    )
    graph.tmap[ph_partial] = Iterator(
        lambda i, j: ("0", f"{PM} * {i}", f"{PN} * {j}"),
        shape=(split_k, M, N),
        tile_size=(PK, PM, PN),
    )
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
    ]
    graph.tmap[node_1] = Iterator(
        lambda i, j: (f"{PM} * {i}",),
        shape=(M,),
        tile_size=(PM,),
    )
    graph.tmap[node_2] = Iterator(
        lambda i, j: (f"{PM} * {i}",),
        shape=(M,),
        tile_size=(PM,),
    )
    graph.tmap[node_3] = Iterator(
        lambda i, j: (f"{PM} * {i}",),
        shape=(M,),
        tile_size=(PM,),
    )
    graph.tmap[node_4] = Iterator(
        lambda i, j: (f"{PM} * {i}", f"{PN} * {j}"),
        shape=(M, N),
        tile_size=(PM, PN),
    )
    graph.tmap[out_node] = Iterator(
        lambda i, j: (f"{PM} * {i}", f"{PN} * {j}"),
        shape=(M, N),
        tile_size=(PM, PN),
    )

    combine_graph.extent = PK // IK
    combine_graph.tmap[combine_rowsum.inputs[0].src_node] = Iterator(
        lambda ik: (f"{IK} * {ik}", "0"),
        shape=(PK, PM),
        tile_size=(IK, IM),
    )
    combine_graph.tmap[combine_mma.inputs[0].src_node] = Iterator(
        lambda ik: (f"{IK} * {ik}", "0", "0"),
        shape=(PK, PM, PN),
        tile_size=(IK, IM, IN),
    )
    combine_graph.tmap[combine_rowsum] = Iterator(
        lambda ik: ("0",),
        shape=(PM,),
        tile_size=(IM,),
    )
    combine_graph.tmap[combine_mma] = Iterator(
        lambda ik: ("0", "0"),
        shape=(PM, PN),
        tile_size=(IM, IN),
    )

    graph.tinfo_map = {
        ph_rowsum: ("PlaceHolder_A", "g"),
        ph_partial: ("PlaceHolder_B", "g"),
        node_0: ("t_GraphNode_0", "r"),
        node_1: ("t_div_0", "r"),
        node_2: ("t_sqrt_0", "r"),
        node_3: ("t_div_1", "r"),
        node_4: ("t_mul_0", "r"),
        out_node: ("OutputNode", "g"),
    }

    combine_graph.tinfo_map = {
        ph_rowsum: ("PlaceHolder_A", "g"),
        ph_partial: ("PlaceHolder_B", "g"),
        combine_rowsum.inputs[0].src_node: ("PlaceHolder_AA", "s"),
        combine_mma.inputs[0].src_node: ("PlaceHolder_BB", "s"),
        combine_rowsum: ("t_rowsum_combine_0", "r"),
        combine_mma: ("t_mma_combine_0", "r"),
    }

    graph = passes.annotate_args(graph, is_main=True)
    graph = passes.annotate_tc_layout(graph)
    graph = passes.annotate_fragment_layout_conflicts(graph)

    for cur_node in gather_all_nodes(*graph.nodes):
        assert getattr(cur_node, "resolve_operand_layout", None) is None, (
            cur_node.name,
            getattr(cur_node, "resolve_operand_layout", None),
        )

    assert graph.tlayout_map[ph_rowsum] == "linear"
    assert graph.tlayout_map[ph_partial] == "linear"
    assert graph.tlayout_map[out_node] == "linear"

    return graph
