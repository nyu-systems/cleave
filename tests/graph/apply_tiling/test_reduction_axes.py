"""A reduction is emitted with the ``dim`` its axis has in the tile the kernel stores.

A kernel stores a tile without its unit extents (a batch dim is one in every tile), so the
axis a reduction was built with is not the ``dim`` its macro has to name: ``apply_tiling``
rebases it onto the stored tile.

The rebase runs over every node of a loop each time another reduction is merged into that
loop, and it used to take its own previous answer for the flat axis: an axis whose rebased
value was itself a stored dim was moved a second time.
"""

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.ir.graph import Graph, GraphNode, OutputNode
from cleave.ir.graph_utils import gather_all_nodes
from cleave.ir.ops import ReduceMax, ReduceSum
from cleave.schedule.tiling import _normalize_inner_reduction_axes

DTYPE, ACCUM_DTYPE = "float16", "float32"


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
