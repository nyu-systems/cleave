import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.ir.graph import GraphNode
from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops.elementwise import BinaryElementwise
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling

from tests._graphs import rmsnorm_matmul_graph, rmsnorm_matmul_split_combine_graph
from tests._lowering import lower_to_cuda
from tests._tolerances import tolerance_for
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

    assert_candidate_compiles_and_runs(cand, tolerance_for(rmsnorm_matmul_split_combine_graph))


# (output tile, step of the matmul-partials sum, step of the row-sum-partials sum): the
# tilings of this combine whose tiles have no unit extent and whose sums take two or more
# partials per step. TileLang stopped at every one of them with "no available layout
# found": the multiply reads the summed matmul partials, a fragment reduced from a rank-3
# tensor, and ``1 / sqrt(rowsum / K)``, three ops on a fragment reduced from a rank-2 one.
# The sixth, ([2, 6144], 4, 2), has only the row sum looped, two rows at a time: its loop
# loads 4 bytes a row, and TileLang lost the loop variable until such a loop was emitted
# as a plain one (tests/graph/apply_tiling/test_split_combine_placement.py).
STAGED = [([2, 6144], 2, 2), ([2, 6144], 2, 4), ([2, 6144], 4, 4), ([4, 6144], 2, 2),
          ([4, 6144], 2, 4), ([2, 6144], 4, 2)]


def _combine_candidate(output_tile, mma_step, rowsum_step):
    graph = rmsnorm_matmul_split_combine_graph(4, 1024, 6144, 4096)
    steps = {"mma_combine_0": mma_step, "rowsum_combine_0": rowsum_step}
    rstep = {
        node.name: {node.raxis[0].var.name: steps[node.name]} if node.name in steps else {}
        for node in graph.nodes
        if not node.is_placeholder() and not node.is_output()
    }
    return build_tiled_candidate(graph, output_tile, rstep)


@pytest.mark.parametrize("output_tile, mma_step, rowsum_step", STAGED)
def test_the_factor_computed_from_the_row_sum_is_staged(output_tile, mma_step, rowsum_step):
    """The divisor of the attention combine is staged through shared memory wherever its
    sums are placed; the same conflict behind a chain of elementwise ops, and in front of
    a multiply, was not recognised."""
    code = build_candidate_code(_combine_candidate(output_tile, mma_step, rowsum_step))
    assert "ukernel_mul_broadcast_0_resolve_rhs_layout_2dims_rrr(" in code
    assert "ukernel_mul_broadcast_0_2dims_rrr(" not in code
    # CPU: TileLang finds a layout.
    assert "main_kernel" in lower_to_cuda(code)


def test_only_a_factor_from_a_lower_rank_sum_is_staged():
    """The un-split RMSNorm matmul multiplies a *matmul* by the same chain: its first
    operand is no sum over a split axis, and nothing is staged (its kernels are as they
    were)."""
    from cleave.codegen import passes
    from cleave.schedule import search_tiling

    graph = rmsnorm_matmul_graph(4096, 6144, 4096)
    scheduled = apply_tiling(search_tiling(graph, topk=1)[0])
    scheduled = passes.annotate_args(scheduled, is_main=True)
    scheduled = passes.annotate_tinfo(scheduled, is_main=True)
    scheduled = passes.annotate_fragment_layout_conflicts(scheduled)
    binary = [n for tier, _ in graph_tiers(scheduled) for n in tier.nodes
              if isinstance(n, BinaryElementwise)]
    assert binary and all(n.resolve_operand_layout is None for n in binary)
