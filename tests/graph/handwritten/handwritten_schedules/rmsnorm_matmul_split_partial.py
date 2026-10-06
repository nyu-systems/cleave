from tilelang.carver.roller import Edge

from cleave import passes
from cleave.graph import Graph, Iterator, GraphNode, OutputNode, AccumEdge
from cleave.ops import Matmul, Mul, Reduction
from cleave.util import gather_all_nodes, lock_graph

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True


def get_handwritten_scheduled_graph(dims):
    split_k, M, N, k_partial = dims
    K = split_k * k_partial
    PM, PN, PK = 128, 128, k_partial
    IM, IN, IK = 128, 128, 32

    assert split_k == 4, split_k
    assert M == 1024, M
    assert N == 6144, N
    assert k_partial == 1024, k_partial

    inner_node_0 = Mul((M, k_partial), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mul_0")
    inner_node_1 = Reduction(
        (M, k_partial),
        axis=-1,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="rowsum_0",
    )
    inner_node_2 = Matmul(
        (M, N, k_partial),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        warp_policy="rows",
    )

    inner_node_0.set_inputs(1, inner_node_0.inputs[0])

    inner_edge_0 = Edge(inner_node_0, inner_node_1, 0, 0)
    inner_node_0._out_edges.append(inner_edge_0)
    inner_node_1.set_inputs(0, inner_edge_0)

    inner_edge_1 = Edge(inner_node_0.inputs[0].src_node, inner_node_2, 0, 0)
    inner_node_0.inputs[0].src_node._out_edges.append(inner_edge_1)
    inner_node_2.set_inputs(0, inner_edge_1)

    inner_nodes = gather_all_nodes(inner_node_0, inner_node_1, inner_node_2)
    inner_graph = Graph(nodes=inner_nodes)
    lock_graph(inner_graph)

    node_0 = GraphNode(name="GraphNode_0", graph=inner_graph)
    rowsum_out_id = node_0.graph.accum_nodes.index(inner_node_1)
    mma_out_id = node_0.graph.accum_nodes.index(inner_node_2)

    out_rowsum = OutputNode(node_0, dtype=OUT_DTYPE, id=rowsum_out_id)
    out_rowsum.name = "OutputNode_0"
    out_mma = OutputNode(node_0, dtype=OUT_DTYPE, id=mma_out_id)
    out_mma.name = "OutputNode_1"

    nodes = gather_all_nodes(node_0)
    graph = Graph(nodes=nodes)
    lock_graph(graph)

    graph.extent = (split_k, M // PM, N // PN)

    ph_a = node_0.inputs[0].src_node
    ph_b = node_0.inputs[1].src_node

    graph.tmap[ph_a] = Iterator(
        lambda isplit, im, in_: (f"{PM} * {im}", f"{PK} * {isplit}"),
        shape=(M, K),
        tile_size=(PM, PK),
    )
    graph.tmap[ph_b] = Iterator(
        lambda isplit, im, in_: (f"{PK} * {isplit}", f"{PN} * {in_}"),
        shape=(K, N),
        tile_size=(PK, PN),
    )
    graph.tmap[node_0] = [
        Iterator(
            lambda isplit, im, in_: (f"{isplit}", f"{PM} * {im}", "0"),
            shape=(split_k, M, 1),
            tile_size=(1, PM, 1),
        ),
        Iterator(
            lambda isplit, im, in_: (f"{isplit}", f"{PM} * {im}", f"{PN} * {in_}"),
            shape=(split_k, M, N),
            tile_size=(1, PM, PN),
        ),
    ]
    graph.tmap[out_rowsum] = Iterator(
        lambda isplit, im, in_: (f"{isplit}", f"{PM} * {im}", "0"),
        shape=(split_k, M, 1),
        tile_size=(1, PM, 1),
    )
    graph.tmap[out_mma] = Iterator(
        lambda isplit, im, in_: (f"{isplit}", f"{PM} * {im}", f"{PN} * {in_}"),
        shape=(split_k, M, N),
        tile_size=(1, PM, PN),
    )

    inner_graph.extent = PK // IK
    inner_graph.tmap[inner_node_0.inputs[0].src_node] = Iterator(
        lambda i: ("0", f"{IK} * {i}"),
        shape=(PM, PK),
        tile_size=(IM, IK),
    )
    inner_graph.tmap[inner_node_2.inputs[1].src_node] = Iterator(
        lambda i: (f"{IK} * {i}", "0"),
        shape=(PK, PN),
        tile_size=(IK, IN),
    )
    inner_graph.tmap[inner_node_0] = Iterator(
        lambda i: ("0", f"{IK} * {i}"),
        shape=(PM, PK),
        tile_size=(IM, IK),
    )
    inner_graph.tmap[inner_node_1] = Iterator(
        lambda i: ("0",),
        shape=(PM,),
        tile_size=(IM,),
    )
    inner_graph.tmap[inner_node_2] = Iterator(
        lambda i: ("0", "0"),
        shape=(PM, PN),
        tile_size=(IM, IN),
    )

    graph.tinfo_map = {
        ph_a: ("PlaceHolder_A", "g"),
        ph_b: ("PlaceHolder_B", "g"),
        node_0: ("t_GraphNode_0", "r"),
        out_rowsum: ("OutputNode_0", "g"),
        out_mma: ("OutputNode_1", "g"),
    }

    inner_graph.tinfo_map = {
        ph_a: ("PlaceHolder_A", "g"),
        ph_b: ("PlaceHolder_B", "g"),
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

    assert graph.tlayout_map[ph_a] == "tc"
    assert graph.tlayout_map[ph_b] == "tc"
    assert graph.tlayout_map[out_rowsum] == "linear"
    assert graph.tlayout_map[out_mma] == "tc"

    return graph
