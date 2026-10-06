import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.graph import GraphNode
from cleave.schedule import apply_tiling

from tests._graphs import matmul_splitk_combine_graph
from tests.graph.apply_tiling._test_utils import (
    assert_candidate_compiles_and_runs,
    build_tiled_candidate,
)

@pytest.mark.parametrize(
    ("target_output_tile", "target_rstep", "expected_main_extent", "expected_offsets"),
    [
        (
            [4, 128],
            {
                "reduction_0": {"v2": 2},
            },
            (32,),
            {
                "PlaceHolder_A": ("0", "4 * i0", "0"),
                "OutputNode": ("4 * i0", "0"),
            },
        ),
        (
            [128, 4],
            {
                "reduction_0": {"v2": 4},
            },
            (32,),
            {
                "PlaceHolder_A": ("0", "0", "4 * i0"),
                "OutputNode": ("0", "4 * i0"),
            },
        ),
        (
            [128, 128],
            {
                "reduction_0": {"v2": 2},
            },
            (1,),
            {
                "PlaceHolder_A": ("0", "0", "0"),
                "OutputNode": ("0", "0"),
            },
        ),
        (
            [128, 128],
            {
                "reduction_0": {"v2": 4},
            },
            (1,),
            {
                "PlaceHolder_A": ("0", "0", "0"),
                "OutputNode": ("0", "0"),
            },
        ),
        (
            [64, 64],
            {
                "reduction_0": {"v2": 2},
            },
            (2, 2),
            {
                "PlaceHolder_A": ("0", "64 * i0", "64 * i1"),
                "OutputNode": ("64 * i0", "64 * i1"),
            },
        ),
        (
            [64, 64],
            {
                "reduction_0": {"v2": 4},
            },
            (2, 2),
            {
                "PlaceHolder_A": ("0", "64 * i0", "64 * i1"),
                "OutputNode": ("64 * i0", "64 * i1"),
            },
        ),
    ],
)
def test_apply_tiling_matmul_splitk_combine(
    target_output_tile, target_rstep, expected_main_extent, expected_offsets
):
    main_graph = matmul_splitk_combine_graph(4, 128, 128)

    cand = build_tiled_candidate(main_graph, target_output_tile, target_rstep)
    scheduled_graph = apply_tiling(cand)

    graph_nodes = [n for n in scheduled_graph.nodes if isinstance(n, GraphNode)]
    if target_rstep["reduction_0"]["v2"] != 4:
        assert len(graph_nodes) == 1
        assert graph_nodes[0].name == "GraphNode_reduction_0"

        inner_accum_names = {n.name for n in graph_nodes[0].graph.accum_nodes}
        assert inner_accum_names == {"reduction_0"}
        assert graph_nodes[0].graph.extent == 4 // target_rstep["reduction_0"]["v2"]

        main_node_names = {getattr(n, "name", "") for n in scheduled_graph.nodes}
        assert "GraphNode_reduction_0" in main_node_names
        assert "reduction_0" not in main_node_names

    assert scheduled_graph.extent == expected_main_extent

    num_args = (
        max(1, len(scheduled_graph.extent))
        if isinstance(scheduled_graph.extent, tuple)
        else 1
    )
    vars_ = [f"i{i}" for i in range(num_args)]

    for i in range(len(expected_offsets)):
        node = list(scheduled_graph.tmap.keys())[i]
        if node.name not in expected_offsets:
            continue
        expected_offset = expected_offsets[node.name]
        actual_offset = scheduled_graph.tmap[node].offset(*vars_)
        assert actual_offset == expected_offset

    assert_candidate_compiles_and_runs(cand)
