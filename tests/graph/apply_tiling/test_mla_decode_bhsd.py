import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave import passes
from cleave.codegen import TilelangCodeGen
from cleave.graph import GraphNode
from cleave.schedule import apply_tiling
from cleave.util import CompilerPassOptions
from cleave.util.runtime import Tolerance, compile_and_run

from tests._graphs import mla_decode_bhsd_graph
from tests.graph.apply_tiling._test_utils import (
    DTYPE,
    assert_candidate_compiles_and_runs,
    build_tiled_candidate,
)

STRICT_TOLERANCE = Tolerance(rtol=1e-2, atol=1e-2, equal_nan=False)

_BASE_RSTEP = {
    "mma_0": {"v3": 128},
    "mma_pe": {"v3": 64},
    "add_0": {},
    "mul_0": {},
    "sub_0": {},
    "exp_0": {},
    "div_0": {},
}


def _rstep(kv_step):
    return {**_BASE_RSTEP, "mma_1": {"v3": kv_step}, "rowsum_0": {"v2": kv_step}}


_OFFSETS_HD_WHOLE = {
    "PlaceHolder_A": ("1 * i0", "64 * i1", "0"),
    "PlaceHolder_B": ("1 * i0", "0", "0"),
    "PlaceHolder_C": ("1 * i0", "64 * i1", "0"),
    "PlaceHolder_D": ("1 * i0", "0", "0"),
    "div_0": ("1 * i0", "64 * i1", "0"),
    "OutputNode": ("1 * i0", "64 * i1", "0"),
}


@pytest.mark.parametrize(
    ("target_output_tile", "target_rstep", "expected_main_extent", "expected_offsets"),
    [
        ([1, 64, 128], _rstep(32), (2, 2), _OFFSETS_HD_WHOLE),
        ([1, 64, 128], _rstep(64), (2, 2), _OFFSETS_HD_WHOLE),
    ],
)
def test_apply_tiling_mla_decode_bhsd(
    target_output_tile, target_rstep, expected_main_extent, expected_offsets
):
    main_graph = mla_decode_bhsd_graph(2, 128, 4096, 128, 64)

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
    assert {
        "mma_0",
        "mma_pe",
        "add_0",
        "mul_0",
        "sub_0",
        "exp_0",
        "mma_1",
        "rowsum_0",
    }.isdisjoint(main_node_names)

    num_args = (
        max(1, len(scheduled_graph.extent))
        if isinstance(scheduled_graph.extent, tuple)
        else 1
    )
    vars_ = [f"i{i}" for i in range(num_args)]

    assert target_output_tile[-1] == 128, target_output_tile
    assert expected_offsets["OutputNode"][-1] == expected_offsets["PlaceHolder_B"][-1]

    for node in scheduled_graph.tmap:
        if node.name not in expected_offsets:
            continue
        expected_offset = expected_offsets[node.name]
        actual_offset = scheduled_graph.tmap[node].offset(*vars_)
        assert actual_offset == expected_offset

    compiler_pass_options = CompilerPassOptions(mla_decode_pass=True)
    assert_candidate_compiles_and_runs(
        cand, compiler_pass_options=compiler_pass_options
    )

    graph = apply_tiling(cand)
    graph = passes.annotate_args(graph, is_main=True)
    graph = passes.annotate_tinfo(graph, is_main=True)
    graph = passes.annotate_fragment_layout_conflicts(graph)
    graph = passes.annotate_mla_accumulation_layout_conflict(graph)
    compile_and_run(
        TilelangCodeGen().gen(graph),
        graph=mla_decode_bhsd_graph(2, 128, 4096, 128, 64),
        dtype=DTYPE,
        tolerance=STRICT_TOLERANCE,
    )
