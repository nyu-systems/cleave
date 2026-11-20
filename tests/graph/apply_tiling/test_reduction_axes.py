"""A reduction is emitted with the ``dim`` its axis has in the tile the kernel stores.

A kernel stores a tile without its unit extents (a batch dim is one in every tile), so the
axis a reduction was built with is not the ``dim`` its macro has to name: ``apply_tiling``
rebases it. Two defects of that rebase are pinned here.

* It only ran for loop tiers. A reduction that is not looped stays in the main tier, where
  it was emitted with its flat ``dim``: the split-KV combine's sums over the split axis
  (axis 1 of ``(H, split, rows[, D])``) named ``dim=1`` on a ``(split, rows[, D])`` tile.
  TileLang refused the shape, except where the row tile happened to equal the split count:
  that one candidate per split factor lowered, and summed over the rows.
* It ran again, over every node of a loop, each time another reduction was merged into
  that loop, and took the previous answer for the flat axis. An axis whose rebased value
  was itself a stored dim was moved a second time.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.ir.graph import Graph, GraphNode, Iterator, OutputNode
from cleave.ir.graph_utils import find_topo_sort, gather_all_nodes
from cleave.ir.ops import Matmul, ReduceMax, ReduceSum, wire_input
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling
from cleave.schedule.tiling import _normalize_inner_reduction_axes, rebase_reduction_axis

from tests._graphs import (
    batched_matmul_splitk_combine_graph,
    matmul_rowsum_graph,
    mha_split_combine_graph,
)
from tests._tolerances import tolerance_for
from tests.graph.apply_tiling._test_utils import (
    assert_candidate_compiles_and_runs,
    build_tiled_candidate,
)

DTYPE, ACCUM_DTYPE = "float16", "float32"


def _steps(graph, **steps):
    """An rstep map by node name: ``steps`` for the reductions named, ``{}`` for the rest."""
    rstep = {}
    for node in graph.nodes:
        if node.is_placeholder() or node.is_output():
            continue
        step = steps.get(node.name)
        rstep[node.name] = {} if step is None else {node.raxis[0].var.name: step}
    assert set(steps) <= set(rstep), f"no such reduction: {set(steps) - set(rstep)}"
    return rstep


def _reduce_dims(code):
    """The ``dim`` of every reduce macro in a generated kernel, by macro name."""
    return dict(
        re.findall(
            r"def (ukernel_\w+)\(A, B\):.*?T\.reduce_(?:sum|max)\(A_ptr, B_ptr, dim=(-?\d+)",
            code,
            flags=re.S,
        )
    )


def _nodes(scheduled):
    """Every op node of a scheduled graph by name, with the tier it sits in (None: main)."""
    found = {}
    for node in scheduled.nodes:
        if isinstance(node, GraphNode):
            found.update({n.name: (n, node) for n in node.graph.nodes})
        else:
            found[node.name] = (node, None)
    return found


# ---------------------------------------------------------------------- the loop tier
@pytest.mark.parametrize("cls", [ReduceSum, ReduceMax])
def test_rebasing_a_loops_reduction_axes_again_gives_the_same_axis(cls):
    """Axis 2 of ``(2, 4, 64, 32)`` at an iteration tile ``(1, 2, 64, 32)``.

    The first batch dim is a unit extent and is not stored; the second one is. The reduced
    axis is therefore dim 1 of the stored ``(2, 64, 32)`` tile -- and stays dim 1 however
    often the loop is visited (each merge of another reduction into it is one visit). It
    used to become dim 0 on the second visit: 1 was taken for the flat axis, the stored
    second batch dim.
    """
    red = cls((2, 4, 64, 32), axis=2, dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="red")
    OutputNode(red, dtype=DTYPE)
    outer = red.inputs[0].src_node
    loop = GraphNode(name="GraphNode_red", graph=Graph(nodes=gather_all_nodes(red, descending=False)))
    itile_map = {outer.name: [1, 2, 64, 32]}

    for visit in range(3):
        _normalize_inner_reduction_axes(loop, itile_map)
        assert (red.axis, red.axis_arg) == (1, 1), f"visit {visit}"
    assert (red.flat_axis, red.flat_axis_arg) == (2, 2)


def test_an_axis_given_from_the_end_is_rebased_from_the_end():
    """``axis=-1`` stays ``-1`` (the generated text of every shipped kernel), and an axis
    with a unit extent stored behind it moves up by one."""
    red = ReduceSum((2, 4, 64, 32), axis=-3, dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="red")
    OutputNode(red, dtype=DTYPE)
    outer = red.inputs[0].src_node
    loop = GraphNode(name="GraphNode_red", graph=Graph(nodes=gather_all_nodes(red, descending=False)))

    _normalize_inner_reduction_axes(loop, {outer.name: [1, 2, 64, 32]})
    assert (red.axis, red.axis_arg) == (1, -3)  # the flat axis names the macro, as before
    _normalize_inner_reduction_axes(loop, {outer.name: [1, 2, 1, 32]})
    assert (red.axis, red.axis_arg) == (1, -2)
    _normalize_inner_reduction_axes(loop, {outer.name: [1, 2, 64, 32]})
    assert (red.axis, red.axis_arg) == (1, -3)


# ---------------------------------------------------------------------- the main tier
@pytest.mark.parametrize(
    ("output_tile", "note"),
    [
        pytest.param([1, 128, 64], "TileLang refused the shape", id="rows128"),
        pytest.param(
            [1, 4, 64], "row tile == split count: it lowered and summed over the rows",
            id="rows4_equal_split",
        ),
    ],
)
def test_unlooped_combine_sums_reduce_the_split_axis(output_tile, note):
    """Split-KV combine, both sums at the whole split (4): neither is looped."""
    graph = mha_split_combine_graph(32, 4, 4096, 64)
    cand = build_tiled_candidate(
        graph, output_tile, _steps(graph, value_combine_0=4, rowsum_combine_0=4)
    )
    scheduled = apply_tiling(cand)

    nodes = _nodes(scheduled)
    for name, stored in (("value_combine_0", 3), ("rowsum_combine_0", 2)):
        node, tier = nodes[name]
        assert tier is None, f"{name} is looped; the case tests nothing"
        tile = scheduled.tmap[node.inputs[0].src_node].tile_size
        # (H, split, rows[, D]) at one head per block: the split axis leads the stored tile.
        assert [int(t) for t in tile][:2] == [1, 4] and len(tile) == stored + 1
        assert (node.axis, node.axis_arg) == (0, 0), note
        assert (node.flat_axis, node.flat_axis_arg) == (1, 1)

    dims = _reduce_dims(build_candidate_code(cand))
    assert dims and set(dims.values()) == {"0"}, (dims, note)


def test_a_looped_and_an_unlooped_sum_each_reduce_the_split_axis():
    """One sum looped (step 2 of 4), the other not: one rule for both tiers."""
    graph = mha_split_combine_graph(32, 4, 4096, 64)
    cand = build_tiled_candidate(
        graph, [1, 128, 64], _steps(graph, value_combine_0=4, rowsum_combine_0=2)
    )
    nodes = _nodes(apply_tiling(cand))
    (value, value_tier), (rowsum, rowsum_tier) = nodes["value_combine_0"], nodes["rowsum_combine_0"]
    assert value_tier is None and rowsum_tier is not None
    assert (value.axis, value.axis_arg) == (0, 0)
    assert (rowsum.axis, rowsum.axis_arg) == (0, 0)


def _batched_combine_positive_axis_graph(B, SPLIT_K, M, N):
    """``batched_matmul_splitk_combine_graph`` with the split axis named from the front.

    The same operator -- ``(B, SPLIT_K, M, N) -> (B, M, N)``, a sum over the split axis --
    but ``axis=1`` where the fixture says ``axis=-3``. Named from the end the axis survives
    a dropped batch dim by itself; named from the front it has to be rebased.
    """
    reduction = ReduceSum(
        (B, SPLIT_K, M, N), axis=1, dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="reduction_0"
    )
    OutputNode(reduction, dtype=DTYPE)
    nodes = gather_all_nodes(reduction)
    return Graph(nodes=find_topo_sort([n for n in nodes if n.is_output()]))


@pytest.mark.parametrize(
    ("output_tile", "note"),
    [
        pytest.param([1, 128, 128], "TileLang refused the shape", id="rows128"),
        pytest.param(
            [1, 4, 128], "row tile == split count: it compiled and summed over the rows",
            id="rows4_equal_split",
        ),
    ],
)
def test_unlooped_sum_over_a_front_named_axis_validates(output_tile, note):
    """GPU: the kernel of an un-looped sum over a non-last, front-named axis is correct."""
    graph = _batched_combine_positive_axis_graph(2, 4, 128, 128)
    cand = build_tiled_candidate(graph, output_tile, _steps(graph, reduction_0=4))
    scheduled = apply_tiling(cand)

    node, tier = _nodes(scheduled)["reduction_0"]
    assert tier is None, "reduction_0 is looped; the case tests nothing"
    assert (node.axis, node.axis_arg) == (0, 0), note
    assert set(_reduce_dims(build_candidate_code(cand)).values()) == {"0"}, note

    # The same sums, the same shapes and the same bound as the fixture named from the end.
    assert_candidate_compiles_and_runs(cand, tolerance_for(batched_matmul_splitk_combine_graph))


# ------------------------------------------------- the main tier, reading a loop's accumulator
def _matmul_column_sum_graph(B, M, N, K):
    """``(A @ B).sum(dim=1) -> (B, N)``: a batched matmul summed over its rows.

    ``matmul_rowsum_graph`` with the other axis summed, and named from the front. With the
    matmul looped over ``K`` and the sum not looped, the sum reads the loop's accumulator.
    """
    mma = Matmul((B, M, N, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_0", warp_policy="rows")
    colsum = ReduceSum(
        (B, M, N), axis=1, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="colsum_0"
    )
    wire_input(mma, colsum, 0)
    OutputNode(colsum, dtype=DTYPE)
    nodes = gather_all_nodes(mma, colsum)
    return Graph(nodes=find_topo_sort([n for n in nodes if n.is_output()]))


ACCUMULATOR_TILES = [
    pytest.param([1, 64], "columns == rows: a wrong dim compiles and sums over the columns",
                 id="cols64_equal_rows"),
    pytest.param([1, 128], "TileLang refuses the shape a wrong dim gives", id="cols128"),
]


def _column_sum_candidate(output_tile):
    graph = _matmul_column_sum_graph(2, 64, 128, 4096)
    return build_tiled_candidate(graph, output_tile, _steps(graph, mma_0=64, colsum_0=64))


@pytest.mark.parametrize(("output_tile", "note"), ACCUMULATOR_TILES)
def test_an_unlooped_sum_over_a_loops_accumulator_reduces_its_own_axis(output_tile, note):
    """The matmul is looped, the sum over its rows is not: the sum's input is the loop's
    accumulator, stored ``(rows, columns)`` -- the batch dim is a unit extent of the block."""
    cand = _column_sum_candidate(output_tile)
    scheduled = apply_tiling(cand)

    node, tier = _nodes(scheduled)["colsum_0"]
    assert tier is None, "colsum_0 is looped; the case tests nothing"
    edge = node.inputs[0]
    assert isinstance(edge.src_node, GraphNode) and edge.accum_node.name == "mma_0"
    stored = edge.src_node.graph.tmap[edge.accum_node].tile_size
    assert [int(t) for t in stored] == [64, output_tile[-1]]
    assert (node.axis, node.axis_arg) == (0, 0), note
    assert (node.flat_axis, node.flat_axis_arg) == (1, 1)
    assert set(_reduce_dims(build_candidate_code(cand)).values()) == {"0"}, note


@pytest.mark.parametrize(("output_tile", "note"), ACCUMULATOR_TILES)
def test_unlooped_sum_over_a_loops_accumulator_validates(output_tile, note):
    """GPU: the kernel of an un-looped sum over a loop's accumulator is correct."""
    cand = _column_sum_candidate(output_tile)
    # The operator of matmul_rowsum_graph -- an fp16 matmul accumulated in fp32, summed
    # over one axis, stored fp16 -- with the other axis summed: its bound.
    assert_candidate_compiles_and_runs(cand, tolerance_for(matmul_rowsum_graph))


# ------------------------------------------------- a reduction added after apply_tiling
@pytest.mark.parametrize(
    ("dims", "axis", "it", "expected"),
    [
        # A tier of apply_tiling: the leading batch dim is not carried.
        pytest.param((2, 4, 64, 32), 2,
                     Iterator(None, shape=(4, 64, 32), tile_size=(2, 64, 32), axes=(1, 2, 3)),
                     (1, 1), id="front_named"),
        pytest.param((2, 4, 64, 32), -3,
                     Iterator(None, shape=(4, 64, 32), tile_size=(2, 64, 32), axes=(1, 2, 3)),
                     (1, -3), id="named_from_the_end"),
        # A unit extent the tier carries is not stored: one row of a block of 16.
        pytest.param((16, 256), 1,
                     Iterator(None, shape=(16, 256), tile_size=(1, 64), axes=(0, 1)),
                     (0, 0), id="unit_extent_in_front"),
        pytest.param((2, 4, 64, 32), -3,
                     Iterator(None, shape=(4, 64, 32), tile_size=(2, 1, 32), axes=(1, 2, 3)),
                     (1, -2), id="unit_extent_behind"),
        # A hand-built tier (no record of the dims it carries): its tile is over the
        # node's dims, here with the single row kept as a unit extent.
        pytest.param((16, 256), 1, Iterator(None, shape=(1, 256), tile_size=(1, 64)),
                     (0, 0), id="hand_built_single_row"),
        pytest.param((2, 16, 256), 2, Iterator(None, shape=(16, 256), tile_size=(16, 64)),
                     (1, 1), id="hand_built_leading_dim_missing"),
        # The reduced axis is itself a unit extent: no dim to name, the built axis stays.
        pytest.param((16, 256), 1, Iterator(None, shape=(16, 256), tile_size=(16, 1), axes=(0, 1)),
                     (1, 1), id="reduced_axis_not_stored"),
    ],
)
def test_a_reduction_is_rebased_from_the_tile_of_its_input(dims, axis, it, expected):
    """``rebase_reduction_axis``: what ``apply_tiling`` does for the reductions it schedules,
    for a reduction that is added to a scheduled graph afterwards."""
    red = ReduceMax(dims, axis=axis, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="red")
    for _ in range(2):
        rebase_reduction_axis(red, it)
        assert (red.axis, red.axis_arg) == expected
    assert (red.flat_axis, red.flat_axis_arg) == (axis % len(dims), axis)


def test_a_tile_knows_where_a_dim_of_its_node_sits():
    carried = Iterator(None, shape=(4, 32), tile_size=(2, 32), axes=(0, 2))
    assert [carried.position(dim, 3) for dim in range(3)] == [0, None, 1]
    # Not recorded: aligned from the end, the missing dims missing from the front.
    plain = Iterator(None, shape=(64, 32), tile_size=(64, 32))
    assert [plain.position(dim, 4) for dim in range(4)] == [None, None, 0, 1]
    assert [plain.position(dim, 2) for dim in range(2)] == [0, 1]
