import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.graph import GraphNode
from cleave.schedule import apply_tiling

from tests._graphs import mha_split_partial_graph
from tests.graph.apply_tiling._test_utils import (
    assert_candidate_compiles_and_runs,
    build_tiled_candidate,
)

@pytest.mark.parametrize(
    (
        "target_output_tile",
        "target_rstep",
        "expected_main_extent",
        "expected_offsets",
        "expected_output_shapes",
        "expected_output_tiles",
    ),
    [
        (
            [1, 64, 64],
                {
                    "split_k": {},
                    "split_v": {},
                    "mma_0": {"v4": 64},
                    "div_0": {},
                    "add_0": {},
                    "exp_0": {},
                    "rowsum_0": {"v3": 32},
                    "mma_1": {"v3": 32},
                },
            (128, 64),
            {
                "PlaceHolder_A": ("1 * (i0 // 4)", "64 * i1", "0"),
                "PlaceHolder_B": ("1 * (i0 // 4)", "1024 * (i0 % 4)", "0"),
                "PlaceHolder_C": ("1 * (i0 // 4)", "1024 * (i0 % 4)", "0"),
                "OutputNode_0": ("(i0 % 4)", "1 * (i0 // 4)", "64 * i1", "0"),
                "OutputNode_1": ("(i0 % 4)", "1 * (i0 // 4)", "64 * i1", "0"),
            },
            {
                "OutputNode_0": (4, 32, 4096, 1),
                "OutputNode_1": (4, 32, 4096, 64),
            },
            {
                "OutputNode_0": (1, 1, 64, 1),
                "OutputNode_1": (1, 1, 64, 64),
            },
        ),
    ],
)
def test_apply_tiling_mha_split_partial(
    target_output_tile,
    target_rstep,
    expected_main_extent,
    expected_offsets,
    expected_output_shapes,
    expected_output_tiles,
):
    main_graph = mha_split_partial_graph(1, 32, 4, 4096, 1024, 64)

    cand = build_tiled_candidate(main_graph, target_output_tile, target_rstep)
    scheduled_graph = apply_tiling(cand)

    graph_nodes = [n for n in scheduled_graph.nodes if isinstance(n, GraphNode)]
    assert len(graph_nodes) == 1
    assert graph_nodes[0].name == "GraphNode_mma_1"

    inner_accum_names = {n.name for n in graph_nodes[0].graph.accum_nodes}
    assert inner_accum_names == {"mma_1", "rowsum_0"}
    assert graph_nodes[0].graph.extent == 1024 // next(iter(target_rstep["mma_1"].values()))

    main_node_names = {getattr(n, "name", "") for n in scheduled_graph.nodes}
    assert "GraphNode_mma_1" in main_node_names
    assert {
        "split_k",
        "split_v",
        "mma_0",
        "div_0",
        "add_0",
        "exp_0",
        "mma_1",
        "rowsum_0",
    }.isdisjoint(main_node_names)

    assert scheduled_graph.extent == expected_main_extent

    num_args = (
        max(1, len(scheduled_graph.extent))
        if isinstance(scheduled_graph.extent, tuple)
        else 1
    )
    vars_ = [f"i{i}" for i in range(num_args)]

    for node, iterator in scheduled_graph.tmap.items():
        if node.name in expected_offsets:
            assert iterator.offset(*vars_) == expected_offsets[node.name]
        if node.name in expected_output_shapes:
            assert iterator.shape == expected_output_shapes[node.name]
            assert iterator.tile_size == expected_output_tiles[node.name]

    assert_candidate_compiles_and_runs(cand)
