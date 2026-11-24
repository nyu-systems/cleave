import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.codegen import passes
from cleave.core.errors import ScheduleRejected
from cleave.ir.analysis import infer_output_axis_dependencies
from cleave.ir.graph import GraphNode, OutputNode
from cleave.schedule import apply_tiling
from cleave.schedule import tiling

from tests._graphs import mha_split_partial_graph
from tests._tolerances import tolerance_for
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

    assert_candidate_compiles_and_runs(cand, tolerance_for(mha_split_partial_graph))


# A partial whose two sums take the whole per-split KV extent in one step: neither is looped,
# so the row sum stays in the main tier next to the value matmul. It is not an ancestor of the
# canonical (value) output, and until the outputs next to the canonical one were given launch
# axes too, ``apply_tiling`` refused every such tiling ("Expected exactly one launch-axis
# mapping for iter dim 0, got []"): 40 of 160 partial plans of a split attention at factor 4.
WHOLE_SPLIT_DIMS = (1, 4, 4, 64, 256, 64)  # B, H, SPLIT_K, S_Q, S_KV_PARTIAL, D
WHOLE_SPLIT_RSTEP = {
    "split_k": {},
    "split_v": {},
    "mma_0": {"v3": 32},
    "div_0": {},
    "add_0": {},
    "exp_0": {},
    "rowsum_0": {"v3": 256},
    "mma_1": {"v3": 256},
}


def test_nodes_that_feed_only_a_sibling_output_depend_on_the_launch_axes():
    graph = mha_split_partial_graph(*WHOLE_SPLIT_DIMS)
    outputs = {n.name: n for n in graph.nodes if isinstance(n, OutputNode)}
    rowsum, value = outputs["OutputNode_0"], outputs["OutputNode_1"]
    assert tiling._select_canonical_output_node(list(outputs.values())) is value

    alone = infer_output_axis_dependencies(value)
    # The defect: nothing says which launch axis walks the row sum, or its output.
    assert "rowsum_0" not in alone and "OutputNode_0" not in alone

    siblings = tiling._sibling_launch_axes(value, list(outputs.values()))
    assert [(out.name, axes) for out, axes in siblings] == [("OutputNode_0", [0, 1, 2])]
    deps = infer_output_axis_dependencies(value, siblings)
    assert deps["rowsum_0"] == deps["OutputNode_0"] == [{0}, {1}, {2}]
    # What the canonical output says about a node is what it said before.
    assert {name: deps[name] for name in alone} == alone


def test_apply_tiling_mha_split_partial_whole_split_in_one_step():
    main_graph = mha_split_partial_graph(*WHOLE_SPLIT_DIMS)
    cand = build_tiled_candidate(main_graph, [1, 64, 32], WHOLE_SPLIT_RSTEP)
    scheduled_graph = apply_tiling(cand)

    # Only Q K^T is looped (over D); both sums are computed in the main tier.
    (loop,) = [n for n in scheduled_graph.nodes if isinstance(n, GraphNode)]
    assert loop.name == "GraphNode_mma_0" and loop.graph.extent == 2
    inner = {(tuple(it.shape), tuple(it.tile_size)) for node, it in loop.graph.tmap.items()
             if node.is_placeholder()}
    assert inner == {((64, 64), (64, 32)), ((256, 64), (256, 32))}  # Q and K: D in two steps
    # The kernel takes its inputs first: its outputs are its trailing arguments, although
    # the row sum is finished before V is reached.
    args = passes.annotate_args(apply_tiling(cand), is_main=True).args
    assert [a.name for a in args] == [
        "PlaceHolder_A", "PlaceHolder_B", "PlaceHolder_C", "OutputNode_0", "OutputNode_1"]
    assert {"rowsum_0", "mma_1"} <= {n.name for n in scheduled_graph.nodes}
    assert scheduled_graph.extent == (16, 2)

    expected = {  # name: (offsets, shape, tile)
        "PlaceHolder_A": (("1 * (i0 // 4)", "0", "0"), (4, 64, 64), (1, 64, 64)),
        "PlaceHolder_B": (("1 * (i0 // 4)", "256 * (i0 % 4)", "0"), (4, 1024, 64), (1, 256, 64)),
        "PlaceHolder_C": (("1 * (i0 // 4)", "256 * (i0 % 4)", "32 * i1"), (4, 1024, 64),
                          (1, 256, 32)),
        # the head and the split index, as for the value: every block of a row writes it
        "rowsum_0": (("1 * (i0 // 4)", "(i0 % 4)", "0"), (4, 4, 64), (1, 1, 64)),
        "OutputNode_0": (("1 * (i0 // 4)", "(i0 % 4)", "0"), (4, 4, 64), (1, 1, 64)),
        "mma_1": (("1 * (i0 // 4)", "(i0 % 4)", "0", "32 * i1"), (4, 4, 64, 64), (1, 1, 64, 32)),
        "OutputNode_1": (("1 * (i0 // 4)", "(i0 % 4)", "0", "32 * i1"), (4, 4, 64, 64),
                         (1, 1, 64, 32)),
    }
    seen = {}
    for node, iterator in scheduled_graph.tmap.items():
        seen[node.name] = (iterator.offset("i0", "i1"), tuple(iterator.shape),
                           tuple(iterator.tile_size))
    assert {name: seen[name] for name in expected} == expected
    # Neither output comes out of a loop: both are stored as their producers compute them.
    assert scheduled_graph.stored_layouts == {
        "OutputNode_0": ((0, 1, 2), 0),
        "OutputNode_1": ((0, 1, 2, 3), 0),
    }

    assert_candidate_compiles_and_runs(cand, tolerance_for(mha_split_partial_graph))


def test_a_step_for_an_axis_the_node_does_not_have_is_refused():
    """``mma_0`` of this graph reduces ``v3``. Keyed ``v4``, its step is still what the
    loop's trip count is read off (the map has one value), but not what cuts its tiles:
    the loop ran twice over the whole of Q and K, and the kernel returned exp(2 Q K^T)
    summed -- built without complaint, 0.1 off a row sum of 0.5 on the GPU."""
    main_graph = mha_split_partial_graph(*WHOLE_SPLIT_DIMS)
    (axis,) = [n.raxis[0].var.name for n in main_graph.nodes if n.name == "mma_0"]
    assert axis == "v3"
    cand = build_tiled_candidate(main_graph, [1, 64, 32],
                                 {**WHOLE_SPLIT_RSTEP, "mma_0": {"v4": 32}})
    with pytest.raises(ScheduleRejected, match=(
            r"^GraphNode_mma_0 runs 2 iterations but cuts none of its tensors into tiles: "
            r"every iteration would read the same slice$")):
        apply_tiling(cand)
