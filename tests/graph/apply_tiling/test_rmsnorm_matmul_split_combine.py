import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.graph import GraphNode
from cleave.schedule import apply_tiling

from tests._graphs import rmsnorm_matmul_split_combine_graph
from tests.graph.apply_tiling._test_utils import (
    assert_candidate_compiles_and_runs,
    build_tiled_candidate,
)

@pytest.mark.parametrize(
    ("target_output_tile", "target_rstep", "expected_main_extent", "expected_offsets"),
    [
        (
            [2, 6144],
            {
                "mma_combine_0": {"v2": 2},
                "rowsum_combine_0": {"v1": 2},
                "div_0": {},
                "sqrt_0": {},
                "div_1": {},
                "mul_0": {},
            },
            (512,),
            {
                "PlaceHolder_A": ("0", "2 * i0"),
                "PlaceHolder_B": ("0", "2 * i0", "0"),
                "div_0": ("2 * i0",),
                "sqrt_0": ("2 * i0",),
                "div_1": ("2 * i0",),
                "mul_0": ("2 * i0", "0"),
                "OutputNode": ("2 * i0", "0"),
            },
        ),
    ],
)
def test_apply_tiling_rmsnorm_matmul_split_combine(
    target_output_tile, target_rstep, expected_main_extent, expected_offsets
):
    main_graph = rmsnorm_matmul_split_combine_graph(4, 1024, 6144, 4096)

    cand = build_tiled_candidate(main_graph, target_output_tile, target_rstep)
    scheduled_graph = apply_tiling(cand)

    graph_nodes = [n for n in scheduled_graph.nodes if isinstance(n, GraphNode)]
    assert len(graph_nodes) == 1
    assert graph_nodes[0].name == "GraphNode_rowsum_combine_0"

    inner_accum_names = {n.name for n in graph_nodes[0].graph.accum_nodes}
    assert inner_accum_names == {"rowsum_combine_0", "mma_combine_0"}
    assert graph_nodes[0].graph.extent == 4 // target_rstep["rowsum_combine_0"]["v1"]

    assert scheduled_graph.extent == expected_main_extent

    main_node_names = {getattr(n, "name", "") for n in scheduled_graph.nodes}
    assert {
        "div_0",
        "sqrt_0",
        "div_1",
        "mul_0",
        "GraphNode_rowsum_combine_0",
    }.issubset(main_node_names)
    assert {"rowsum_combine_0", "mma_combine_0"}.isdisjoint(main_node_names)

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
