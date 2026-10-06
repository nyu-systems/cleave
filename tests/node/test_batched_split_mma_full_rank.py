import pytest

from cleave.arch import get_arch
from cleave.graph import Graph, OutputNode
from cleave.ops import Matmul, SplitView, wire_input
from cleave.policy.default import TCPolicy
from cleave.util import find_topo_sort, gather_all_nodes


def _build_policy_and_split_mma_nodes(b: int) -> tuple[TCPolicy, dict[str, object]]:
    SPLIT_K, M, N, K_PARTIAL = 4, 128, 128, 1024
    K = SPLIT_K * K_PARTIAL
    split_a = SplitView(
        (b, M, K),
        axis=2,
        factor=SPLIT_K,
        dtype="float16",
        name="split_a",
    )
    split_b = SplitView(
        (b, K, N),
        axis=1,
        factor=SPLIT_K,
        dtype="float16",
        name="split_b",
    )
    node_0 = Matmul(
        (b, SPLIT_K, M, N, K_PARTIAL),
        dtype="float16",
        out_dtype="float32",
        name="mma_0",
        warp_policy="rows",
    )
    node_0.set_tag("tensorcore_config", True)
    wire_input(split_a, node_0, 0)
    wire_input(split_b, node_0, 1)
    output_node = OutputNode(node_0, dtype="float16")
    nodes = gather_all_nodes(node_0)
    main_graph = Graph(nodes=find_topo_sort([n for n in nodes if n.is_output()]))

    policy = TCPolicy([output_node], get_arch("cuda"), flat_graph=main_graph)
    return policy, {"split_a": split_a, "split_b": split_b, "mma_0": node_0}


@pytest.mark.parametrize("b", [1, 2])
def test_batched_split_mma_full_rank(b):
    policy, nodes = _build_policy_and_split_mma_nodes(b)
    node_split_a = nodes["split_a"]
    node_split_b = nodes["split_b"]
    node_mma = nodes["mma_0"]

    # mma -> (split_a, split_b)
    split_a_tile, split_b_tile = policy._propagate_inputs_full_rank(node_mma, [1, 1, 64, 32])

    assert len(split_a_tile) == len(node_split_a.get_shape()), (
        b,
        "mma_1",
        split_a_tile,
        list(node_mma.get_shape()),
    )
    assert len(split_b_tile) == len(node_split_b.get_shape()), (
        b,
        "split_b",
        split_b_tile,
        list(node_split_b.get_shape()),
    )

    assert split_a_tile == [1, 1, 64, 1024]
    assert split_b_tile == [1, 1, 1024, 32]
