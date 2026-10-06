import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.graph import GraphNode
from cleave.schedule import apply_tiling

from tests._graphs import mha_bhsd_graph
from tests.graph.apply_tiling._test_utils import (
    assert_candidate_compiles_and_runs,
    build_tiled_candidate,
)

@pytest.mark.parametrize(
    ("target_output_tile", "target_rstep", "expected_main_extent", "expected_offsets"),
    [
        (
            [1, 64, 64],
            {
                "mma_0": {"v3": 64},
                "mul_0": {},
                "sub_0": {},
                "exp_0": {},
                "mma_1": {"v3": 32},
                "rowsum_0": {"v2": 32},
                "div_0": {},
            },
            (32, 64),
            {
                "PlaceHolder_A": ("1 * i0", "64 * i1", "0"),
                "PlaceHolder_B": ("1 * i0", "0", "0"),
                "PlaceHolder_C": ("1 * i0", "0", "0"),
                "div_0": ("1 * i0", "64 * i1", "0"),
                "OutputNode": ("1 * i0", "64 * i1", "0"),
            },
        ),
        (
            [1, 64, 64],
            {
                "mma_0": {"v3": 64},
                "mul_0": {},
                "sub_0": {},
                "exp_0": {},
                "mma_1": {"v3": 64},
                "rowsum_0": {"v2": 64},
                "div_0": {},
            },
            (32, 64),
            {
                "PlaceHolder_A": ("1 * i0", "64 * i1", "0"),
                "PlaceHolder_B": ("1 * i0", "0", "0"),
                "PlaceHolder_C": ("1 * i0", "0", "0"),
                "div_0": ("1 * i0", "64 * i1", "0"),
                "OutputNode": ("1 * i0", "64 * i1", "0"),
            },
        ),
        (
            [1, 128, 32],
            {
                "mma_0": {"v3": 64},
                "mul_0": {},
                "sub_0": {},
                "exp_0": {},
                "mma_1": {"v3": 64},
                "rowsum_0": {"v2": 64},
                "div_0": {},
            },
            (32, 32, 2),
            {
                "PlaceHolder_A": ("1 * i0", "128 * i1", "0"),
                "PlaceHolder_B": ("1 * i0", "0", "0"),
                "PlaceHolder_C": ("1 * i0", "0", "32 * i2"),
                "div_0": ("1 * i0", "128 * i1", "32 * i2"),
                "OutputNode": ("1 * i0", "128 * i1", "32 * i2"),
            },
        ),
        (
            [1, 256, 32],
            {
                "mma_0": {"v3": 64},
                "mul_0": {},
                "sub_0": {},
                "exp_0": {},
                "mma_1": {"v3": 32},
                "rowsum_0": {"v2": 32},
                "div_0": {},
            },
            (32, 16, 2),
            {
                "PlaceHolder_A": ("1 * i0", "256 * i1", "0"),
                "PlaceHolder_B": ("1 * i0", "0", "0"),
                "PlaceHolder_C": ("1 * i0", "0", "32 * i2"),
                "div_0": ("1 * i0", "256 * i1", "32 * i2"),
                "OutputNode": ("1 * i0", "256 * i1", "32 * i2"),
            },
        ),
        (
            [1, 64, 32],
            {
                "mma_0": {"v3": 64},
                "mul_0": {},
                "sub_0": {},
                "exp_0": {},
                "mma_1": {"v3": 128},
                "rowsum_0": {"v2": 128},
                "div_0": {},
            },
            (32, 64, 2),
            {
                "PlaceHolder_A": ("1 * i0", "64 * i1", "0"),
                "PlaceHolder_B": ("1 * i0", "0", "0"),
                "PlaceHolder_C": ("1 * i0", "0", "32 * i2"),
                "div_0": ("1 * i0", "64 * i1", "32 * i2"),
                "OutputNode": ("1 * i0", "64 * i1", "32 * i2"),
            },
        ),
        (
            [1, 64, 32],
            {
                "mma_0": {"v3": 64},
                "mul_0": {},
                "sub_0": {},
                "exp_0": {},
                "mma_1": {"v3": 32},
                "rowsum_0": {"v2": 32},
                "div_0": {},
            },
            (32, 64, 2),
            {
                "PlaceHolder_A": ("1 * i0", "64 * i1", "0"),
                "PlaceHolder_B": ("1 * i0", "0", "0"),
                "PlaceHolder_C": ("1 * i0", "0", "32 * i2"),
                "div_0": ("1 * i0", "64 * i1", "32 * i2"),
                "OutputNode": ("1 * i0", "64 * i1", "32 * i2"),
            },
        ),
        (
            [1, 256, 32],
            {
                "mma_0": {"v3": 64},
                "mul_0": {},
                "sub_0": {},
                "exp_0": {},
                "mma_1": {"v3": 64},
                "rowsum_0": {"v2": 64},
                "div_0": {},
            },
            (32, 16, 2),
            {
                "PlaceHolder_A": ("1 * i0", "256 * i1", "0"),
                "PlaceHolder_B": ("1 * i0", "0", "0"),
                "PlaceHolder_C": ("1 * i0", "0", "32 * i2"),
                "div_0": ("1 * i0", "256 * i1", "32 * i2"),
                "OutputNode": ("1 * i0", "256 * i1", "32 * i2"),
            },
        ),
    ],
)
def test_apply_tiling_mha_bhsd(
    target_output_tile, target_rstep, expected_main_extent, expected_offsets
):
    main_graph = mha_bhsd_graph(1, 32, 4096, 4096, 64)

    cand = build_tiled_candidate(main_graph, target_output_tile, target_rstep)

    scheduled_graph = apply_tiling(cand)

    graph_nodes = [n for n in scheduled_graph.nodes if isinstance(n, GraphNode)]
    assert len(graph_nodes) == 1
    assert graph_nodes[0].name == "GraphNode_rowsum_0"

    inner_accum_names = {n.name for n in graph_nodes[0].graph.accum_nodes}
    assert inner_accum_names == {"mma_1", "rowsum_0"}
    assert graph_nodes[0].graph.extent == 4096 // target_rstep["rowsum_0"]["v2"]

    assert scheduled_graph.extent == expected_main_extent

    main_node_names = {getattr(n, "name", "") for n in scheduled_graph.nodes}
    assert "div_0" in main_node_names
    assert "GraphNode_rowsum_0" in main_node_names
    assert {"mma_0", "mul_0", "sub_0", "exp_0", "mma_1", "rowsum_0"}.isdisjoint(
        main_node_names
    )

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
