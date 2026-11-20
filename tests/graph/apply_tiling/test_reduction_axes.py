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

A third case has no ``dim`` to give: a loop that walks the reduced axis one element at a
time (step 1). The iteration tile then has a unit extent on that axis, the stored tile has
no such axis, and the reduction is the tile folded into the output element by element. It
used to be emitted as a ``T.reduce_*`` over an axis the tile does not have; and where the
axis is a batch dim (the split axis of the value partials) the loop tier dropped it and
handed every iteration the whole block.
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

from tests._graphs import (
    batched_matmul_splitk_combine_graph,
    matmul_splitk_combine_graph,
    mha_split_combine_graph,
)
from tests._lowering import lower_to_cuda
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


# ------------------------------------------------- a unit extent on the reduced axis
@pytest.mark.parametrize(
    ("cls", "fold"),
    [
        pytest.param(ReduceSum, "B[i0, i1] += A[i0, i1]", id="ReduceSum"),
        pytest.param(ReduceMax, "B[i0, i1] = T.max(B[i0, i1], A[i0, i1])", id="ReduceMax"),
    ],
)
def test_a_reduction_over_a_unit_extent_folds_the_tile(cls, fold):
    """Axis 1 of ``(2, 4, 64, 32)`` at an iteration tile ``(1, 1, 64, 32)``: stored ``(64, 32)``."""
    red = cls((2, 4, 64, 32), axis=1, dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="red")
    OutputNode(red, dtype=DTYPE)
    outer = red.inputs[0].src_node
    loop = GraphNode(name="GraphNode_red", graph=Graph(nodes=gather_all_nodes(red, descending=False)))
    plain = red.gen_macro(mem_scope="sr", clear=False)
    assert f"{cls._tl_reduce}(A_ptr, B_ptr, dim=1, clear=False)" in plain

    _normalize_inner_reduction_axes(loop, {outer.name: [1, 1, 64, 32]})
    assert red.fold_rank == 2 and (red.axis, red.axis_arg) == (1, 1)
    name = red.get_macro_name(mem_scope="sr", clear=False)
    assert name == f"ukernel_{cls.kind}_fold_2dims_sr"
    macro = red.gen_macro(mem_scope="sr", clear=False)
    assert "T.reduce" not in macro and fold in macro
    assert "for i0, i1 in T.Parallel(*B.shape):" in macro
    # Not an accumulator (the block holds one element of the axis): the tile, as it is.
    assert "B[i0, i1] = A[i0, i1]" in red.gen_macro(mem_scope="sr", clear=True)

    # Derived from the tile every time, like the axis: a wider tile reduces again.
    _normalize_inner_reduction_axes(loop, {outer.name: [1, 2, 64, 32]})
    assert red.fold_rank is None and (red.axis, red.axis_arg) == (0, 0)
    assert red.gen_macro(mem_scope="sr", clear=False) == plain.replace("dim=1", "dim=0").replace(
        "axis1", "axis0"
    )


def test_a_loop_walks_a_split_axis_one_partial_at_a_time():
    """Split-KV combine, both sums at step 1 of 4: one loop of four iterations.

    The split axis is a batch dim of the value partials ``(H, split, rows, D)`` and a
    non-batch dim of the row-sum partials ``(H, split, rows)``. Both are stored by the loop
    tier (the block holds four of them) and both are sliced by its counter.
    """
    graph = mha_split_combine_graph(32, 4, 4096, 64)
    cand = build_tiled_candidate(
        graph, [1, 128, 64], _steps(graph, value_combine_0=1, rowsum_combine_0=1)
    )
    scheduled = apply_tiling(cand)

    nodes = _nodes(scheduled)
    (value, tier), (rowsum, rowsum_tier) = nodes["value_combine_0"], nodes["rowsum_combine_0"]
    assert tier is not None and tier is rowsum_tier and tier.graph.extent == 4
    for node, block, tile in ((value, [4, 128, 64], [1, 128, 64]), (rowsum, [4, 128], [1, 128])):
        it = tier.graph.tmap[node.inputs[0].src_node]
        assert ([int(x) for x in it.shape], [int(x) for x in it.tile_size]) == (block, tile)
        assert node.fold_rank == len(block) - 1

    code = build_candidate_code(cand)
    assert "T.reduce_sum" not in code
    assert "[1 * i0 : 1 * i0 + 1, 0 : 0 + 128, 0 : 0 + 64]" in code  # value partials
    assert "[1 * i0 : 1 * i0 + 1, 0 : 0 + 128]" in code  # row-sum partials
    lower_to_cuda(code)


def test_a_max_over_a_split_axis_at_step_one_lowers():
    """The fold of a ``ReduceMax``: a running max, one partial per iteration."""
    reduction = ReduceMax((4, 128, 128), axis=0, dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="reduction_0")
    OutputNode(reduction, dtype=DTYPE)
    graph = Graph(nodes=find_topo_sort([n for n in gather_all_nodes(reduction) if n.is_output()]))
    cand = build_tiled_candidate(graph, [64, 64], _steps(graph, reduction_0=1))
    code = build_candidate_code(cand)
    assert "T.reduce_max" not in code and "= T.max(B[i0, i1], A[i0, i1])" in code
    assert 'T.fill(t_reduction_0, -T.infinity("float32"))' in code
    lower_to_cuda(code)


@pytest.mark.parametrize(
    ("build", "dims", "output_tile", "steps"),
    [
        pytest.param(
            mha_split_combine_graph, (32, 4, 4096, 64), [1, 128, 64],
            {"value_combine_0": 1, "rowsum_combine_0": 1}, id="mha_split_combine",
        ),
        pytest.param(
            matmul_splitk_combine_graph, (4, 128, 128), [64, 64], {"reduction_0": 1},
            id="matmul_splitk_combine",
        ),
        pytest.param(
            batched_matmul_splitk_combine_graph, (2, 4, 128, 128), [1, 64, 64], {"reduction_0": 1},
            id="batched_matmul_splitk_combine",
        ),
    ],
)
def test_a_combine_at_step_one_validates(build, dims, output_tile, steps):
    """GPU: a split combine that takes one partial per iteration is a correct kernel."""
    graph = build(*dims)
    cand = build_tiled_candidate(graph, output_tile, _steps(graph, **steps))
    assert "T.reduce_sum" not in build_candidate_code(cand)
    assert_candidate_compiles_and_runs(cand, tolerance_for(build))
