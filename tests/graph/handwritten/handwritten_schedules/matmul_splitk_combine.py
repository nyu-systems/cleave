from cleave import passes
from cleave.graph import Graph, Iterator, GraphNode, OutputNode
from cleave.ops import Reduction
from cleave.util import gather_all_nodes, lock_graph

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True


def get_handwritten_scheduled_graph(dims):
    split_k, M, N = dims
    PK, PM, PN = 2, 128, 128

    assert split_k == 4, split_k
    assert M == PM, (M, PM)
    assert N == PN, (N, PN)

    inner_node_0 = Reduction(
        (split_k, M, N),
        axis=0,
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="reduction_0",
    )
    inner_nodes = gather_all_nodes(inner_node_0)
    inner_graph = Graph(nodes=inner_nodes)
    lock_graph(inner_graph)

    node_0 = GraphNode(name="GraphNode_0", graph=inner_graph)

    output_nodes = [OutputNode(node_0, dtype=OUT_DTYPE)]
    nodes = gather_all_nodes(node_0)
    graph = Graph(nodes=nodes)
    lock_graph(graph)

    # Full output tile (128, 128) => single kernel instance.
    graph.extent = (1,)

    graph.tmap[node_0.inputs[0].src_node] = Iterator(
        lambda i: ("0", "0", "0"),
        shape=(split_k, M, N),
        tile_size=(split_k, M, N),
    )  # split-k partial tensor
    graph.tmap[node_0] = Iterator(
        lambda i: ("0", "0"),
        shape=(M, N),
        tile_size=(PM, PN),
    )  # GraphNode
    graph.tmap[node_0.outputs[0].dst_node] = Iterator(
        lambda i: ("0", "0"),
        shape=(M, N),
        tile_size=(PM, PN),
    )  # Output

    # Inner graph loops over the split-k axis (combine = sum over axis 0).
    inner_graph.extent = split_k // PK

    inner_graph.tmap[inner_node_0.inputs[0].src_node] = Iterator(
        lambda i: (f"{PK} * {i}", "0", "0"),
        shape=(split_k, M, N),
        tile_size=(PK, PM, PN),
    )  # A slice from split-k partial tensor
    inner_graph.tmap[inner_node_0] = Iterator(
        lambda i: ("0", "0"),
        shape=(M, N),
        tile_size=(PM, PN),
    )  # reduction_0 accumulator/output tile

    graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        node_0: ("t_GraphNode_0", "r"),
        node_0.outputs[0].dst_node: ("OutputNode", "g"),
    }

    inner_graph.tinfo_map = {
        node_0.inputs[0].src_node: ("PlaceHolder_A", "g"),
        inner_node_0.inputs[0].src_node: ("PlaceHolder_AA", "s"),
        inner_node_0: ("t_reduction_0", "r"),
    }

    graph = passes.annotate_args(graph, is_main=True)
    graph = passes.annotate_tc_layout(graph)
    graph = passes.annotate_fragment_layout_conflicts(graph)

    for cur_node in gather_all_nodes(*graph.nodes):
        assert getattr(cur_node, "resolve_operand_layout", None) is None, (
            cur_node.name,
            getattr(cur_node, "resolve_operand_layout", None),
        )

    assert graph.tlayout_map[node_0.inputs[0].src_node] == "linear"
    assert graph.tlayout_map[output_nodes[0]] == "linear"

    return graph
