from cleave import passes
from cleave.graph import Graph, Iterator, GraphNode, OutputNode
from cleave.ops import Reduction
from cleave.util import gather_all_nodes, lock_graph

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True


def get_handwritten_scheduled_graph(dims):
    B, split_k, M, N = dims
    PB, PK, PM, PN = 1, split_k, M, N
    IK, IM, IN = 2, PM, PN

    inner_node_0 = Reduction(
        (B, split_k, M, N),
        axis=-3,
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

    # Full (M, N) output tile, batched only along B.
    graph.extent = (B // PB,)

    ph_a = node_0.inputs[0].src_node
    out_node = output_nodes[0]

    graph.tmap[ph_a] = Iterator(
        lambda ib: (f"{PB} * {ib}", "0", "0", "0"),
        shape=(B, split_k, M, N),
        tile_size=(PB, split_k, M, N),
    )  # split-k partial tensor, layout (B, split_k, M, N)
    graph.tmap[node_0] = Iterator(
        lambda ib: (f"{PB} * {ib}", "0", "0"),
        shape=(B, M, N),
        tile_size=(PB, PM, PN),
    )  # GraphNode output tile
    graph.tmap[out_node] = Iterator(
        lambda ib: (f"{PB} * {ib}", "0", "0"),
        shape=(B, M, N),
        tile_size=(PB, PM, PN),
    )  # Output

    # Inner graph combines along split-k in chunks.
    inner_graph.extent = PK // IK

    inner_graph.tmap[inner_node_0.inputs[0].src_node] = Iterator(
        lambda ik: (f"{IK} * {ik}", "0", "0"),
        shape=(PK, PM, PN),
        tile_size=(IK, IM, IN),
    )  # A slice from split-k partial tensor (batch dim fixed by outer graph)
    inner_graph.tmap[inner_node_0] = Iterator(
        lambda ik: ("0", "0"),
        shape=(PM, PN),
        tile_size=(IM, IN),
    )  # reduction accumulator/output tile

    graph.tinfo_map = {
        ph_a: ("PlaceHolder_A", "g"),
        node_0: ("t_GraphNode_0", "r"),
        out_node: ("OutputNode", "g"),
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

    assert graph.tlayout_map[ph_a] == "linear"
    assert graph.tlayout_map[out_node] == "linear"

    return graph
