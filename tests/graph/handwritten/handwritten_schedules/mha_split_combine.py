from cleave import passes
from cleave.graph import Graph, Iterator, GraphNode, OutputNode, AccumEdge
from cleave.ops import Div, Reduction
from cleave.util import gather_all_nodes, lock_graph

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True


def get_handwritten_scheduled_graph(dims):
    H_Q, split_k, S_Q, D = dims
    PH, PK, PS_Q, PD = 1, 4, 64, 64
    IK, IS_Q, ID = 2, 64, 64

    assert H_Q == 32, H_Q
    assert split_k == PK, split_k
    assert S_Q == 4096, S_Q
    assert D == PD, D

    combine_rowsum = Reduction(
        (split_k, S_Q),
        axis=0,
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="rowsum_combine_0",
    )
    combine_value = Reduction(
        (split_k, S_Q, D),
        axis=0,
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="value_combine_0",
    )
    combine_nodes = gather_all_nodes(combine_rowsum, combine_value)
    combine_graph = Graph(nodes=combine_nodes)
    lock_graph(combine_graph)

    node_0 = GraphNode(name="GraphNode_0", graph=combine_graph)

    node_1 = Div(
        (S_Q, D),
        broadcast=0,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )

    edge_0 = AccumEdge(node_0, node_1, 0, 0, accum_node=combine_value)
    node_0._out_edges.append(edge_0)
    node_1.set_inputs(0, edge_0)

    edge_1 = AccumEdge(node_0, node_1, 0, 1, accum_node=combine_rowsum)
    node_0._out_edges.append(edge_1)
    node_1.set_inputs(1, edge_1)

    output_nodes = [OutputNode(node_1, dtype=OUT_DTYPE)]
    nodes = gather_all_nodes(node_0, node_1)
    graph = Graph(nodes=nodes)
    lock_graph(graph)

    graph.extent = (H_Q // PH, S_Q // PS_Q)

    ph_rowsum = node_0.inputs[0].src_node
    ph_partial = node_0.inputs[1].src_node
    out_node = output_nodes[0]

    graph.tmap[ph_rowsum] = Iterator(
        lambda ih, iq: (f"{PH} * {ih}", "0", f"{PS_Q} * {iq}"),
        shape=(H_Q, split_k, S_Q),
        tile_size=(PH, PK, PS_Q),
    )
    graph.tmap[ph_partial] = Iterator(
        lambda ih, iq: (f"{PH} * {ih}", "0", f"{PS_Q} * {iq}", "0"),
        shape=(H_Q, split_k, S_Q, D),
        tile_size=(PH, PK, PS_Q, PD),
    )
    graph.tmap[node_0] = [
        Iterator(
            lambda ih, iq: (f"{PH} * {ih}", f"{PS_Q} * {iq}"),
            shape=(H_Q, S_Q),
            tile_size=(PH, PS_Q),
        ),
        Iterator(
            lambda ih, iq: (f"{PH} * {ih}", f"{PS_Q} * {iq}", "0"),
            shape=(H_Q, S_Q, D),
            tile_size=(PH, PS_Q, PD),
        ),
    ]
    graph.tmap[node_1] = Iterator(
        lambda ih, iq: (f"{PH} * {ih}", f"{PS_Q} * {iq}", "0"),
        shape=(H_Q, S_Q, D),
        tile_size=(PH, PS_Q, PD),
    )
    graph.tmap[out_node] = Iterator(
        lambda ih, iq: (f"{PH} * {ih}", f"{PS_Q} * {iq}", "0"),
        shape=(H_Q, S_Q, D),
        tile_size=(PH, PS_Q, PD),
    )

    combine_graph.extent = PK // IK
    combine_graph.tmap[combine_rowsum.inputs[0].src_node] = Iterator(
        lambda ik: (f"{IK} * {ik}", "0"),
        shape=(PK, PS_Q),
        tile_size=(IK, IS_Q),
    )
    combine_graph.tmap[combine_value.inputs[0].src_node] = Iterator(
        lambda ik: (f"{IK} * {ik}", "0", "0"),
        shape=(PK, PS_Q, PD),
        tile_size=(IK, IS_Q, ID),
    )
    combine_graph.tmap[combine_rowsum] = Iterator(
        lambda ik: ("0",),
        shape=(PS_Q,),
        tile_size=(IS_Q,),
    )
    combine_graph.tmap[combine_value] = Iterator(
        lambda ik: ("0", "0"),
        shape=(PS_Q, PD),
        tile_size=(IS_Q, ID),
    )

    graph.tinfo_map = {
        ph_rowsum: ("PlaceHolder_A", "g"),
        ph_partial: ("PlaceHolder_B", "g"),
        node_0: ("t_GraphNode_0", "r"),
        node_1: ("t_div_0", "r"),
        out_node: ("OutputNode", "g"),
    }

    combine_graph.tinfo_map = {
        ph_rowsum: ("PlaceHolder_A", "g"),
        ph_partial: ("PlaceHolder_B", "g"),
        combine_rowsum.inputs[0].src_node: ("PlaceHolder_AA", "s"),
        combine_value.inputs[0].src_node: ("PlaceHolder_BB", "s"),
        combine_rowsum: ("t_rowsum_combine_0", "r"),
        combine_value: ("t_value_combine_0", "r"),
    }

    graph = passes.annotate_args(graph, is_main=True)
    graph = passes.annotate_tc_layout(graph)
    graph = passes.annotate_fragment_layout_conflicts(graph)

    for cur_node in gather_all_nodes(node_0, node_1):
        expected = "rhs" if cur_node is node_1 else None
        assert getattr(cur_node, "resolve_operand_layout", None) == expected, (
            cur_node.name,
            getattr(cur_node, "resolve_operand_layout", None),
            expected,
        )

    assert graph.tlayout_map[ph_rowsum] == "linear"
    assert graph.tlayout_map[ph_partial] == "linear"
    assert graph.tlayout_map[out_node] == "linear"

    return graph
