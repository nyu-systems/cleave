import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.graph import GraphNode
from cleave.schedule import apply_tiling

from tests._graphs import rmsnorm_matmul_graph
from tests.graph.apply_tiling._test_utils import (
    assert_candidate_compiles_and_runs,
    build_tiled_candidate,
)


@pytest.mark.parametrize(
    ("target_output_tile", "target_rstep", "expected_main_extent", "expected_offsets"),
    [
        (
            [128, 128],
            {
                "mul_0": {},
                "rowsum_0": {"v1": 32},
                "mma_0": {"v2": 32},
                "div_0": {},
                "rsqrt_0": {},
                "mul_1": {},
            },
            (8, 48),
            {
                "PlaceHolder_A": ("128 * i0", "0"),
                "PlaceHolder_B": ("0", "128 * i1"),
                "div_0": ("128 * i0",),
                "rsqrt_0": ("128 * i0",),
                "mul_1": ("128 * i0", "128 * i1"),
                "OutputNode": ("128 * i0", "128 * i1"),
            },
        ),
    ],
)
def test_apply_tiling_rmsnorm_matmul(
    target_output_tile, target_rstep, expected_main_extent, expected_offsets
):
    main_graph = rmsnorm_matmul_graph(1024, 6144, 4096)

    cand = build_tiled_candidate(main_graph, target_output_tile, target_rstep)
    scheduled_graph = apply_tiling(cand)

    graph_nodes = [n for n in scheduled_graph.nodes if isinstance(n, GraphNode)]
    assert len(graph_nodes) == 1
    assert graph_nodes[0].name == "GraphNode_rowsum_0"

    inner_accum_names = {n.name for n in graph_nodes[0].graph.accum_nodes}
    assert inner_accum_names == {"mma_0", "rowsum_0"}
    assert graph_nodes[0].graph.extent == 4096 // target_rstep["rowsum_0"]["v1"]

    assert scheduled_graph.extent == expected_main_extent

    main_node_names = {getattr(n, "name", "") for n in scheduled_graph.nodes}
    assert {"div_0", "rsqrt_0", "mul_1", "GraphNode_rowsum_0"}.issubset(main_node_names)
    assert {"mul_0", "rowsum_0", "mma_0"}.isdisjoint(main_node_names)

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
