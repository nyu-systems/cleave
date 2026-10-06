from tilelang.carver.roller import Edge

from cleave import passes
from cleave.graph import Graph, Iterator, GraphNode, OutputNode
from cleave.ops import Add, Div, Exp, Matmul, Reduction
from cleave.util import gather_all_nodes, lock_graph

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True


def get_handwritten_scheduled_graph(dims):
    B, H_Q, H_KV, split_k, S_Q, S_KV_PARTIAL, D = dims
    S_KV = split_k * S_KV_PARTIAL
    GROUP_SIZE = H_Q // H_KV

    PH, PS_Q, PS_KV, PD = 1, 64, S_KV_PARTIAL, 64
    IS_Q, IS_KV, ID = 64, 32, 64

    assert B == 1, B
    assert H_Q == 32, H_Q
    assert H_KV == 8, H_KV
    assert split_k == 4, split_k
    assert S_Q == 4096, S_Q
    assert S_KV_PARTIAL == 1024, S_KV_PARTIAL
    assert D == PD, D
    assert GROUP_SIZE == 4, GROUP_SIZE

    inner_node_0 = Matmul(
        (PH, PS_Q, PS_KV, PD),
        input_batch_dims_A=(PH,),
        input_batch_dims_B=(PH,),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    inner_node_1 = Div(
        (PH, PS_Q, PS_KV),
        const=float(D**0.5),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )
    inner_node_2 = Add(
        (PH, PS_Q, PS_KV),
        const=-(50.0 / (D**0.5)),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="add_0",
    )
    inner_node_3 = Exp(
        (PH, PS_Q, PS_KV),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="exp_0",
    )
    inner_node_4 = Reduction(
        (PH, PS_Q, PS_KV),
        axis=-1,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="rowsum_0",
    )
    inner_node_5 = Matmul(
        (PH, PS_Q, PD, PS_KV),
        input_batch_dims_A=(PH,),
        input_batch_dims_B=(PH,),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
    )

    edge_0 = Edge(inner_node_0, inner_node_1, 0, 0)
    inner_node_0._out_edges.append(edge_0)
    inner_node_1.set_inputs(0, edge_0)

    edge_1 = Edge(inner_node_1, inner_node_2, 0, 0)
    inner_node_1._out_edges.append(edge_1)
    inner_node_2.set_inputs(0, edge_1)

    edge_2 = Edge(inner_node_2, inner_node_3, 0, 0)
    inner_node_2._out_edges.append(edge_2)
    inner_node_3.set_inputs(0, edge_2)

    edge_3 = Edge(inner_node_3, inner_node_4, 0, 0)
    inner_node_3._out_edges.append(edge_3)
    inner_node_4.set_inputs(0, edge_3)

    edge_4 = Edge(inner_node_3, inner_node_5, 0, 0)
    inner_node_3._out_edges.append(edge_4)
    inner_node_5.set_inputs(0, edge_4)

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
    rowsum_out_id = node_0.graph.accum_nodes.index(inner_node_4)
    partial_out_id = node_0.graph.accum_nodes.index(inner_node_5)

    out_rowsum = OutputNode(node_0, dtype=OUT_DTYPE, id=rowsum_out_id)
    out_rowsum.name = "OutputNode_0"
    out_partial = OutputNode(node_0, dtype=OUT_DTYPE, id=partial_out_id)
    out_partial.name = "OutputNode_1"

    nodes = gather_all_nodes(node_0)
    graph = Graph(nodes=nodes)
    lock_graph(graph)

    graph.extent = (split_k, H_Q // PH, S_Q // PS_Q)

    ph_q = node_0.inputs[0].src_node
    ph_k = node_0.inputs[1].src_node
    ph_v = node_0.inputs[2].src_node

    graph.tmap[ph_q] = Iterator(
        lambda isplit, ih, iq: (f"{PH} * {ih}", f"{PS_Q} * {iq}", "0"),
        shape=(B * H_Q, S_Q, D),
        tile_size=(PH, PS_Q, PD),
    )
    graph.tmap[ph_k] = Iterator(
        lambda isplit, ih, iq: (
            f"{PH} * ({ih} // {GROUP_SIZE})",
            f"{PS_KV} * {isplit}",
            "0",
        ),
        shape=(B * H_KV, S_KV, D),
        tile_size=(PH, PS_KV, PD),
    )
    graph.tmap[ph_v] = Iterator(
        lambda isplit, ih, iq: (
            f"{PH} * ({ih} // {GROUP_SIZE})",
            f"{PS_KV} * {isplit}",
            "0",
        ),
        shape=(B * H_KV, S_KV, D),
        tile_size=(PH, PS_KV, PD),
    )
    graph.tmap[node_0] = [
        Iterator(
            lambda isplit, ih, iq: (f"{isplit}", f"{PH} * {ih}", f"{PS_Q} * {iq}", "0"),
            shape=(split_k, B * H_Q, S_Q, 1),
            tile_size=(1, PH, PS_Q, 1),
        ),
        Iterator(
            lambda isplit, ih, iq: (f"{isplit}", f"{PH} * {ih}", f"{PS_Q} * {iq}", "0"),
            shape=(split_k, B * H_Q, S_Q, D),
            tile_size=(1, PH, PS_Q, PD),
        ),
    ]
    graph.tmap[out_rowsum] = Iterator(
        lambda isplit, ih, iq: (f"{isplit}", f"{PH} * {ih}", f"{PS_Q} * {iq}", "0"),
        shape=(split_k, B * H_Q, S_Q, 1),
        tile_size=(1, PH, PS_Q, 1),
    )
    graph.tmap[out_partial] = Iterator(
        lambda isplit, ih, iq: (f"{isplit}", f"{PH} * {ih}", f"{PS_Q} * {iq}", "0"),
        shape=(split_k, B * H_Q, S_Q, D),
        tile_size=(1, PH, PS_Q, PD),
    )

    inner_graph.extent = PS_KV // IS_KV
    inner_graph.tmap[inner_node_0.inputs[0].src_node] = Iterator(
        lambda i: ("0", "0"),
        shape=(PS_Q, PD),
        tile_size=(IS_Q, ID),
    )
    inner_graph.tmap[inner_node_0.inputs[1].src_node] = Iterator(
        lambda i: (f"{IS_KV} * {i}", "0"),
        shape=(PS_KV, PD),
        tile_size=(IS_KV, ID),
    )
    inner_graph.tmap[inner_node_5.inputs[1].src_node] = Iterator(
        lambda i: (f"{IS_KV} * {i}", "0"),
        shape=(PS_KV, PD),
        tile_size=(IS_KV, ID),
    )
    inner_graph.tmap[inner_node_0] = Iterator(
        lambda i: ("0", f"{IS_KV} * {i}"),
        shape=(PS_Q, PS_KV),
        tile_size=(IS_Q, IS_KV),
    )
    inner_graph.tmap[inner_node_1] = Iterator(
        lambda i: ("0", f"{IS_KV} * {i}"),
        shape=(PS_Q, PS_KV),
        tile_size=(IS_Q, IS_KV),
    )
    inner_graph.tmap[inner_node_2] = Iterator(
        lambda i: ("0", f"{IS_KV} * {i}"),
        shape=(PS_Q, PS_KV),
        tile_size=(IS_Q, IS_KV),
    )
    inner_graph.tmap[inner_node_3] = Iterator(
        lambda i: ("0", f"{IS_KV} * {i}"),
        shape=(PS_Q, PS_KV),
        tile_size=(IS_Q, IS_KV),
    )
    inner_graph.tmap[inner_node_4] = Iterator(
        lambda i: ("0",),
        shape=(PS_Q,),
        tile_size=(IS_Q,),
    )
    inner_graph.tmap[inner_node_5] = Iterator(
        lambda i: ("0", "0"),
        shape=(PS_Q, PD),
        tile_size=(IS_Q, ID),
    )

    graph.tinfo_map = {
        ph_q: ("PlaceHolder_A", "g"),
        ph_k: ("PlaceHolder_B", "g"),
        ph_v: ("PlaceHolder_C", "g"),
        node_0: ("t_GraphNode_0", "r"),
        out_rowsum: ("OutputNode_0", "g"),
        out_partial: ("OutputNode_1", "g"),
    }

    inner_graph.tinfo_map = {
        ph_q: ("PlaceHolder_A", "g"),
        ph_k: ("PlaceHolder_B", "g"),
        ph_v: ("PlaceHolder_C", "g"),
        inner_node_0.inputs[0].src_node: ("PlaceHolder_AA", "s"),
        inner_node_0.inputs[1].src_node: ("PlaceHolder_BB", "s"),
        inner_node_5.inputs[1].src_node: ("PlaceHolder_CC", "s"),
        inner_node_0: ("t_mma_0", "r"),
        inner_node_1: ("t_div_0", "r"),
        inner_node_2: ("t_add_0", "r"),
        inner_node_3: ("t_exp_0", "r"),
        inner_node_4: ("t_rowsum_0", "r"),
        inner_node_5: ("t_mma_1", "r"),
    }

    graph = passes.annotate_args(graph, is_main=True)
    graph = passes.annotate_tc_layout(graph)
    graph = passes.annotate_fragment_layout_conflicts(graph)

    for cur_node in gather_all_nodes(*graph.nodes):
        assert getattr(cur_node, "resolve_operand_layout", None) is None, (
            cur_node.name,
            getattr(cur_node, "resolve_operand_layout", None),
        )

    assert graph.tlayout_map[ph_q] == "tc"
    assert graph.tlayout_map[ph_k] == "tc"
    assert graph.tlayout_map[ph_v] == "tc"
    assert graph.tlayout_map[out_rowsum] == "linear"
    assert graph.tlayout_map[out_partial] == "tc"

    return graph
