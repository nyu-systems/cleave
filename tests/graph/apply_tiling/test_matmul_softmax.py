import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.graph import GraphNode
from cleave.schedule import apply_tiling

from tests._graphs import matmul_softmax_graph
from tests.graph.apply_tiling._test_utils import (
    assert_candidate_compiles_and_runs,
    build_tiled_candidate,
)

@pytest.mark.parametrize(
    ("target_output_tile", "target_rstep", "expected_main_extent", "expected_offsets"),
    [
        (
            [1, 128, 128],
            {
                "mma_0": {"v3": 64},
                "mul_0": {},
                "sub_0": {},
                "exp_0": {},
                "rowsum_0": {"v2": 32},
                "div_0": {},
            },
            (32, 32, 32),
            {
                "PlaceHolder_A": ("1 * i0", "128 * i1", "0"),
                "PlaceHolder_B": ("1 * i0", "0", "0"),
                "div_0": ("1 * i0", "128 * i1", "128 * i2"),
                "OutputNode": ("1 * i0", "128 * i1", "128 * i2"),
            },
        ),
    ],
)
def test_apply_tiling_matmul_softmax(
    target_output_tile, target_rstep, expected_main_extent, expected_offsets
):
    main_graph = matmul_softmax_graph(1, 32, 4096, 4096, 64)

    cand = build_tiled_candidate(main_graph, target_output_tile, target_rstep)
    scheduled_graph = apply_tiling(cand)

    graph_nodes = [n for n in scheduled_graph.nodes if isinstance(n, GraphNode)]
    assert len(graph_nodes) == 1
    assert graph_nodes[0].name == "GraphNode_rowsum_0"

    assert scheduled_graph.extent == expected_main_extent

    main_node_names = {getattr(n, "name", "") for n in scheduled_graph.nodes}
    assert "GraphNode_rowsum_0" in main_node_names
    assert "div_0" in main_node_names
    assert "rowsum_0" not in main_node_names

    num_args = (
        max(1, len(scheduled_graph.extent))
        if isinstance(scheduled_graph.extent, tuple)
        else 1
    )
    vars_ = [f"i{i}" for i in range(num_args)]

    for node, iterator in scheduled_graph.tmap.items():
        if node.name not in expected_offsets:
            continue
        assert iterator.offset(*vars_) == expected_offsets[node.name]

    assert_candidate_compiles_and_runs(cand)
