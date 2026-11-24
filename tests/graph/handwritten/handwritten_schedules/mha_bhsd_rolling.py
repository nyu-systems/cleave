"""``mha_bhsd`` with a rolling (online) softmax instead of the fixed shift.

Same tiling as ``mha_bhsd`` (64 query rows per block, the KV axis looped 64 keys at a
time). The constant ``sub_0`` is gone; the loop carries a running row max ``m`` and
declares the rolling update through ``Graph.lag_nodes`` / ``Graph.accum_scale``:

    m_prev = m                         lag node, emitted before the max is updated
    m      = max(m, max_kv s)          carried, starts at NEG_FLT_MAX
    P      = exp(s - m)
    alpha  = exp(m_prev - m)
    l      = l * alpha + sum_kv P      rowsum_0
    O      = O * alpha[row] + P @ V    mma_1
    out    = O / l[row]                after the loop
"""

from cleave.codegen import passes
from cleave.ir.graph import Graph, GraphNode, Iterator, OutputNode
from cleave.ir.graph_utils import gather_all_nodes, lock_graph
from cleave.ir.ops import NEG_FLT_MAX, Div, Exp, Matmul, Mul, ReduceMax, ReduceSum, Sub, wire_input
from cleave.ir.rolling import rolling_nodes

DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"


def get_handwritten_scheduled_graph(dims):
    B, H, S_Q, S_KV, D = dims
    PB, PH, PS_Q, PS_KV, PD = 1, 1, 64, 4096, 64
    IB, IH, IS_Q, IS_KV, ID = 1, 1, 64, 64, 64
    F32 = dict(dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE)
    scores = (B * H, S_Q, S_KV)

    mma_0 = Matmul(
        (B * H, S_Q, S_KV, D), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_0",
        transpose_B=True, warp_policy="rows",
    )
    mul_0 = Mul(scores, const=(1 / (D**0.5)), name="mul_0", **F32)
    # The statistics are float32: NEG_FLT_MAX is not a float16.
    row_max = ReduceMax(scores, axis=-1, name="exp_0_max", accum_init=NEG_FLT_MAX, **F32)
    shifted = Sub(scores, broadcast=0, name="exp_0_shifted", **F32)
    exp_0 = Exp(scores, name="exp_0", **F32)
    max_prev, max_step, alpha = rolling_nodes(
        row_max, stem="exp_0", rows=(B * H, S_Q), n_batch_dims=1, exp_cls=Exp
    )
    mma_1 = Matmul(
        (B * H, S_Q, D, S_KV), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_1",
        warp_policy="rows",
    )
    rowsum_0 = ReduceSum(scores, name="rowsum_0", **F32)
    div_0 = Div((B * H, S_Q, D), broadcast=0, name="div_0", **F32)

    wire_input(mma_0, mul_0, 0)
    wire_input(mul_0, row_max, 0)
    wire_input(mul_0, shifted, 0)
    wire_input(row_max, shifted, 1)
    wire_input(shifted, exp_0, 0)
    wire_input(exp_0, mma_1, 0)
    wire_input(exp_0, rowsum_0, 0)
    wire_input(mma_1, div_0, 0)
    wire_input(rowsum_0, div_0, 1)
    output_nodes = [OutputNode(div_0, dtype=OUT_DTYPE)]

    # The loop: the two reductions and their ancestors, as apply_tiling gathers them.
    inner_ops = [
        mma_0, mul_0, row_max, shifted, exp_0, max_prev, max_step, alpha, mma_1, rowsum_0,
    ]
    inner_graph = Graph(nodes=gather_all_nodes(*inner_ops, descending=False))
    # Declared before the GraphNode exists: alpha has no dataflow consumer, and must not
    # be taken for an accumulator.
    inner_graph.lag_nodes = [max_prev]
    inner_graph.accum_scale = {mma_1: alpha, rowsum_0: alpha}
    node_0 = GraphNode(name="GraphNode_0", graph=inner_graph)
    assert inner_graph.accum_nodes == [mma_1, rowsum_0]
    # The max is read inside the loop, so it is carried by declaration; appended last.
    assert node_0.carry(row_max) == 2
    lock_graph(inner_graph)

    nodes = gather_all_nodes(node_0, div_0)
    graph = Graph(nodes=nodes)
    lock_graph(graph)

    # schedule the main graph
    graph.extent = (
        H // PH,
        S_Q // PS_Q,
    )

    q, k, v = (edge.src_node for edge in node_0.inputs)
    graph.tmap[q] = Iterator(
        lambda i, j: (f"{PH} * {i}", f"{PS_Q} * {j}", "0"),
        shape=(B * H, S_Q, D),
        tile_size=(PB * PH, PS_Q, PD),
    )  # Q
    graph.tmap[k] = Iterator(
        lambda i, j: (f"{PH} * {i}", "0", "0"),
        shape=(B * H, S_KV, D),
        tile_size=(PB * PH, PS_KV, PD),
    )  # K
    graph.tmap[v] = Iterator(
        lambda i, j: (f"{PH} * {i}", "0", "0"),
        shape=(B * H, S_KV, D),
        tile_size=(PB * PH, PS_KV, PD),
    )  # V
    graph.tmap[node_0] = [
        Iterator(
            lambda i, j: (f"{PH} * {i}", f"{PS_Q} * {j}", "0"),
            shape=(B * H, S_Q, D),
            tile_size=(PB * PH, PS_Q, PD),
        ),  # Matmul
        Iterator(
            lambda i, j: (f"{PH} * {i}", f"{PS_Q} * {j}"),
            shape=(B * H, S_Q),
            tile_size=(PB * PH, PS_Q),
        ),  # ReduceSum
        Iterator(
            lambda i, j: (f"{PH} * {i}", f"{PS_Q} * {j}"),
            shape=(B * H, S_Q),
            tile_size=(PB * PH, PS_Q),
        ),  # ReduceMax
    ]  # GraphNode
    graph.tmap[div_0] = Iterator(
        lambda i, j: (f"{PH} * {i}", f"{PS_Q} * {j}", "0"),
        shape=(B * H, S_Q, D),
        tile_size=(PB * PH, PS_Q, PD),
    )  # div_0
    graph.tmap[output_nodes[0]] = Iterator(
        lambda i, j: (f"{PH} * {i}", f"{PS_Q} * {j}", "0"),
        shape=(B * H, S_Q, D),
        tile_size=(PB * PH, PS_Q, PD),
    )  # Output

    # schedule the inner graph
    inner_graph.extent = PS_KV // IS_KV

    inner_q, inner_k = (edge.src_node for edge in mma_0.inputs)
    inner_v = mma_1.inputs[1].src_node
    inner_graph.tmap[inner_q] = Iterator(
        lambda i: ("0", "0"),
        shape=(PS_Q, PD),
        tile_size=(IS_Q, ID),
    )  # Q'
    inner_graph.tmap[inner_k] = Iterator(
        lambda i: (f"{IS_KV} * {i}", "0"),
        shape=(PS_KV, PD),
        tile_size=(IS_KV, ID),
    )  # K'
    inner_graph.tmap[inner_v] = Iterator(
        lambda i: (f"{IS_KV} * {i}", "0"),
        shape=(PS_KV, PD),
        tile_size=(IS_KV, ID),
    )  # V'
    for node in (mma_0, mul_0, shifted, exp_0):  # the score tile
        inner_graph.tmap[node] = Iterator(
            lambda i: ("0", f"{IS_KV} * {i}"),
            shape=(PS_Q, PS_KV),
            tile_size=(IS_Q, IS_KV),
        )
    inner_graph.tmap[mma_1] = Iterator(
        lambda i: ("0", "0"),
        shape=(PS_Q, PD),
        tile_size=(IS_Q, ID),
    )  # mma_1
    for node in (rowsum_0, row_max, max_prev, max_step, alpha):  # one value per query row
        inner_graph.tmap[node] = Iterator(
            lambda i: ("0",),
            shape=(PS_Q,),
            tile_size=(IS_Q,),
        )

    graph.tinfo_map = {
        q: ("PlaceHolder_A", "g"),
        k: ("PlaceHolder_B", "g"),
        v: ("PlaceHolder_C", "g"),
        node_0: ("t_GraphNode_0", "r"),
        div_0: ("t_div_1", "r"),
        output_nodes[0]: ("OutputNode", "g"),
    }

    inner_graph.tinfo_map = {
        q: ("PlaceHolder_A", "g"),
        k: ("PlaceHolder_B", "g"),
        v: ("PlaceHolder_C", "g"),
        inner_q: ("PlaceHolder_AA", "s"),
        inner_k: ("PlaceHolder_BB", "s"),
        inner_v: ("PlaceHolder_CC", "s"),
        mma_0: ("t_mma_0", "r"),
        mul_0: ("t_mul_0", "r"),
        row_max: ("t_exp_0_max", "r"),
        max_prev: ("t_exp_0_max_prev", "r"),
        max_step: ("t_exp_0_max_step", "r"),
        alpha: ("t_exp_0_alpha", "r"),
        shifted: ("t_exp_0_shifted", "r"),
        exp_0: ("t_exp_0", "r"),
        mma_1: ("t_mma_1", "r"),
        rowsum_0: ("t_rowsum_0", "r"),
    }

    graph = passes.annotate_args(graph, is_main=True)
    graph = passes.annotate_tc_layout(graph)
    graph = passes.annotate_fragment_layout_conflicts(graph)

    for cur_node in gather_all_nodes(*graph.nodes):
        assert getattr(cur_node, "resolve_operand_layout", None) is None, (
            cur_node.name,
            getattr(cur_node, "resolve_operand_layout", None),
        )

    assert graph.tlayout_map[q] == "tc"
    assert graph.tlayout_map[k] == "tc"
    assert graph.tlayout_map[v] == "tc"
    assert graph.tlayout_map[output_nodes[0]] == "tc"

    return graph
