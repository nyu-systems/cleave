from cleave import passes
from cleave.graph import Graph, Iterator, GraphNode, OutputNode
from cleave.ops import Matmul
from cleave.util import gather_all_nodes, lock_graph

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True


def get_handwritten_scheduled_graph(dims):
    B, split_k, M, N, k_partial = dims
    K = split_k * k_partial
    PB, PM, PN, PK = 1, 128, 128, k_partial
    IM, IN, IK = 128, 128, 32

    assert B % PB == 0, (B, PB)
    assert M % PM == 0, (M, PM)
    assert N % PN == 0, (N, PN)
    assert PK % IK == 0, (k_partial, IK)

    inner_node_0 = Matmul(
        (B, split_k, M, N, k_partial),
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

    graph.extent = ((B // PB) * split_k, M // PM, N // PN)

    ph_a = node_0.inputs[0].src_node
    ph_b = node_0.inputs[1].src_node
    out_node = output_nodes[0]

    graph.tmap[ph_a] = Iterator(
        lambda i0, im, in_: (
            f"{PB} * ({i0} // {split_k})",
            f"{PM} * {im}",
            f"{PK} * ({i0} % {split_k})",
        ),
        shape=(B, M, K),
        tile_size=(PB, PM, PK),
    )  # A: (B, M, K)
    graph.tmap[ph_b] = Iterator(
        lambda i0, im, in_: (
            f"{PB} * ({i0} // {split_k})",
            f"{PK} * ({i0} % {split_k})",
            f"{PN} * {in_}",
        ),
        shape=(B, K, N),
        tile_size=(PB, PK, PN),
    )  # B: (B, K, N)
    graph.tmap[node_0] = Iterator(
        lambda i0, im, in_: (
            f"{PB} * ({i0} // {split_k})",
            f"({i0} % {split_k})",
            f"{PM} * {im}",
            f"{PN} * {in_}",
        ),
        shape=(B, split_k, M, N),
        tile_size=(PB, 1, PM, PN),
    )  # partial output (split_k, B, M, N)
    graph.tmap[out_node] = Iterator(
        lambda i0, im, in_: (
            f"{PB} * ({i0} // {split_k})",
            f"({i0} % {split_k})",
            f"{PM} * {im}",
            f"{PN} * {in_}",
        ),
        shape=(B, split_k, M, N),
        tile_size=(PB, 1, PM, PN),
    )  # Output

    # schedule the inner graph (k_partial reduction tiles)
    inner_graph.extent = k_partial // IK

    inner_graph.tmap[inner_node_0.inputs[0].src_node] = Iterator(
        lambda i: ("0", f"{IK} * {i}"),
        shape=(PM, k_partial),
        tile_size=(IM, IK),
    )  # A
    inner_graph.tmap[inner_node_0.inputs[1].src_node] = Iterator(
        lambda i: (f"{IK} * {i}", "0"),
        shape=(k_partial, PN),
        tile_size=(IK, IN),
    )  # B
    inner_graph.tmap[inner_node_0] = Iterator(
        lambda i: ("0", "0"),
        shape=(PM, PN),
        tile_size=(IM, IN),
    )  # MMA_0

    graph.tinfo_map = {
        ph_a: ("PlaceHolder_A", "g"),
        ph_b: ("PlaceHolder_B", "g"),
        node_0: ("t_GraphNode_0", "r"),
        out_node: ("OutputNode", "g"),
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

    assert graph.tlayout_map[ph_a] == "tc"
    assert graph.tlayout_map[ph_b] == "tc"
    assert graph.tlayout_map[out_node] == "tc"

    return graph
