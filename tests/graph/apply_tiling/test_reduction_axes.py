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

from cleave.ir.graph import Graph, GraphNode, OutputNode
from cleave.ir.graph_utils import find_topo_sort, gather_all_nodes
from cleave.ir.ops import ReduceMax, ReduceSum
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling
from cleave.schedule.tiling import _normalize_inner_reduction_axes

from tests._graphs import batched_matmul_splitk_combine_graph, mha_split_combine_graph
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
