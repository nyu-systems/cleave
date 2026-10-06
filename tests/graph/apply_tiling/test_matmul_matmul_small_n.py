import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.schedule import apply_tiling

from tests._graphs import matmul_matmul_graph
from tests.graph.apply_tiling._test_utils import (
    assert_candidate_compiles_and_runs,
    build_tiled_candidate,
)

@pytest.mark.parametrize(
    ("target_output_tile", "target_rstep", "expected_main_extent", "expected_offsets"),
    [
        (
            [64, 64],
            {
                "mma_0": {"v2": 64},
                "mma_1": {"v2": 64},
            },
            (64,),
            {
                "PlaceHolder_A": ("64 * i0", "0"),
                "PlaceHolder_B": ("0", "0"),
                "PlaceHolder_C": ("0", "0"),
                "mma_0": ("64 * i0", "0"),
                "mma_1": ("64 * i0", "0"),
                "OutputNode": ("64 * i0", "0"),
            },
        ),
        (
            [64, 64],
            {
                "mma_0": {"v2": 32},
                "mma_1": {"v2": 64},
            },
            (64,),
            {
                "PlaceHolder_A": ("64 * i0", "0"),
                "PlaceHolder_B": ("0", "0"),
                "PlaceHolder_C": ("0", "0"),
                "mma_1": ("64 * i0", "0"),
                "OutputNode": ("64 * i0", "0"),
            },
        ),
    ],
)
def test_apply_tiling_matmul_matmul_small_n(
    target_output_tile, target_rstep, expected_main_extent, expected_offsets
):
    main_graph = matmul_matmul_graph(4096, 64, 64)

    cand = build_tiled_candidate(main_graph, target_output_tile, target_rstep)
    scheduled_graph = apply_tiling(cand)

    assert scheduled_graph.extent == expected_main_extent

    num_args = (
        max(1, len(scheduled_graph.extent))
        if isinstance(scheduled_graph.extent, tuple)
        else 1
    )
    vars_ = [f"i{i}" for i in range(num_args)]

    tmap_node_names = {getattr(node, "name", "") for node in scheduled_graph.tmap}
    assert set(expected_offsets) <= tmap_node_names

    for node, iterator in scheduled_graph.tmap.items():
        if node.name not in expected_offsets:
            continue
        assert iterator.offset(*vars_) == expected_offsets[node.name]

    assert_candidate_compiles_and_runs(cand)
