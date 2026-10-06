import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.graph import GraphNode
from cleave.schedule import apply_tiling

from tests._graphs import rmsnorm_matmul_split_partial_graph
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
            [1, 128, 128],
            {
                "split_a": {},
                "split_b": {},
                "mul_0": {},
                "rowsum_0": {"v2": 32},
                "mma_0": {"v3": 32},
            },
            (4, 8, 48),
            {
                "PlaceHolder_A": ("128 * i1", "1024 * i0"),
                "PlaceHolder_B": ("1024 * i0", "128 * i2"),
                "OutputNode_0": ("1 * i0", "128 * i1", "0"),
                "OutputNode_1": ("1 * i0", "128 * i1", "128 * i2"),
            },
            {
                "OutputNode_0": (4, 1024, 1),
                "OutputNode_1": (4, 1024, 6144),
            },
            {
                "OutputNode_0": (1, 128, 1),
                "OutputNode_1": (1, 128, 128),
            },
        ),
    ],
)
def test_apply_tiling_rmsnorm_matmul_split_partial(
    target_output_tile,
    target_rstep,
    expected_main_extent,
    expected_offsets,
    expected_output_shapes,
    expected_output_tiles,
):
    main_graph = rmsnorm_matmul_split_partial_graph(4, 1024, 6144, 1024)

    cand = build_tiled_candidate(main_graph, target_output_tile, target_rstep)
    scheduled_graph = apply_tiling(cand)

    graph_nodes = [n for n in scheduled_graph.nodes if isinstance(n, GraphNode)]
    assert len(graph_nodes) == 1
    assert graph_nodes[0].name == "GraphNode_mma_0"

    inner_accum_names = {n.name for n in graph_nodes[0].graph.accum_nodes}
    assert inner_accum_names == {"mma_0", "rowsum_0"}
    assert graph_nodes[0].graph.extent == 1024 // next(
        iter(target_rstep["mma_0"].values())
    )
    inner_graph_nodes = set(graph_nodes[0].graph.nodes)
    for inner_node in graph_nodes[0].graph.nodes:
        for edge in inner_node.inputs:
            assert edge.src_node in inner_graph_nodes, (
                f"Inner node {inner_node.name} depends on outer node "
                f"{edge.src_node.name} after tiling merge."
            )

    main_node_names = {getattr(n, "name", "") for n in scheduled_graph.nodes}
    assert "GraphNode_mma_0" in main_node_names
    assert {"split_a", "split_b", "mul_0", "rowsum_0", "mma_0"}.isdisjoint(
        main_node_names
    )

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
