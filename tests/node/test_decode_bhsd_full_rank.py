import pytest

from tilelang.carver.roller import Edge  # noqa: F401 (ensures tilelang/tvm path setup)
from cleave.arch import get_arch
from cleave.graph import Graph, OutputNode
from cleave.ops import Div, Matmul, Reduction
from cleave.policy.default import TCPolicy
from cleave.util import find_topo_sort, gather_all_nodes


def _build_policy_and_decode_nodes(b: int) -> tuple[TCPolicy, dict[str, object]]:
    dtype = "float16"
    accum_dtype = "float32"
    out_dtype = "float16"

    h, s_kv, hd = 128, 4096, 128
    node_4 = Matmul(
        (b, h, hd, s_kv),
        input_batch_dims_A=(b,),
        input_batch_dims_B=(b,),
        dtype=dtype,
        out_dtype=accum_dtype,
        name="mma_1",
        warp_policy="rows",
    )
    node_5 = Reduction(
        (b, h, s_kv),
        dtype=accum_dtype,
        out_dtype=accum_dtype,
        name="rowsum_0",
    )
    node_6 = Div(
        (b, h, hd),
        broadcast=0,
        dtype=accum_dtype,
        out_dtype=accum_dtype,
        name="div_0",
    )

    for src, dst, src_id, dst_id in [
        (node_4, node_6, 0, 0),
        (node_5, node_6, 0, 1),
    ]:
        edge = Edge(src, dst, src_id, dst_id)
        src._out_edges.append(edge)
        dst.set_inputs(edge.dst_id, edge)

    output_node = OutputNode(node_6, dtype=out_dtype)
    nodes = gather_all_nodes(node_4, node_5, node_6)
    main_graph = Graph(nodes=find_topo_sort([n for n in nodes if n.is_output()]))

    for node in main_graph.nodes:
        if isinstance(node, Matmul):
            node.set_tag("tensorcore_config", True)

    policy = TCPolicy([output_node], get_arch("cuda"), flat_graph=main_graph)
    return policy, {"mma_1": node_4, "rowsum_0": node_5, "div_0": node_6}


@pytest.mark.parametrize("b", [1, 2])
def test_decode_bhsd_full_rank(b):
    policy, nodes = _build_policy_and_decode_nodes(b)
    node_div = nodes["div_0"]
    node_mma = nodes["mma_1"]
    node_rowsum = nodes["rowsum_0"]

    # div -> (mma_1, rowsum_0)
    div_inputs = policy._propagate_inputs_full_rank(node_div, [1, 64, 32])
    mma_tile = div_inputs[0]
    rowsum_tile = div_inputs[1]

    assert len(mma_tile) == len(node_mma.get_shape()), (
        b,
        "mma_1",
        mma_tile,
        list(node_mma.get_shape()),
    )
    assert len(rowsum_tile) == len(node_rowsum.get_shape()), (
        b,
        "rowsum_0",
        rowsum_tile,
        list(node_rowsum.get_shape()),
    )

    assert mma_tile == [1, 64, 32]
    assert rowsum_tile == [1, 64]
