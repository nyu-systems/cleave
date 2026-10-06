from tilelang.carver.roller import Edge

from cleave import passes
from cleave.graph import Graph, Iterator, OutputNode
from cleave.util import gather_all_nodes, lock_graph
from cleave.ops import Matmul, Reduction

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True


def get_handwritten_scheduled_graph(dims):
    M, N, K = dims
    PM, PN, PK = 128, 128, 128
    node_0 = Matmul(
        dims=(M, N, K),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        warp_policy="rows",
    )
    node_1 = Reduction(
        dims=(M, N), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )

    # connect the two nodes
    edge = Edge(node_0, node_1, 0, 0)
    node_0._out_edges.append(edge)
    node_1.set_inputs(0, edge)

    output_nodes = [OutputNode(node_1, dtype=OUT_DTYPE)]

    nodes = gather_all_nodes(node_0, node_1)
    graph = Graph(nodes=nodes)
    lock_graph(graph)

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
        lambda i: (f"{PM} * {i}", "0"),
        shape=(M, N),
        tile_size=(PM, PN),
    )  # C
    # REVISIT this: this is essentially the same as below. no need to duplicate?
    graph.tmap[node_1] = Iterator(
        lambda i: (f"{PM} * {i}",),
        shape=(M,),
        tile_size=(PM,),
    )  # D
    graph.tmap[node_1.outputs[0].dst_node] = Iterator(
        lambda i: (f"{PM} * {i}",),
        shape=(M,),
        tile_size=(PM,),
    )  # D

    graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        node_0.inputs[1].src_node: ("PlaceHolder_B", "g"),
        node_0: ("t_mma_0", "r"),
        node_1: ("t_rowsum_0", "r"),
        node_1.outputs[0].dst_node: ("OutputNode", "g"),
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
