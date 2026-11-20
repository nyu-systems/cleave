"""A loop tier is handed the tiles its parent stores.

A kernel stores a tile without its unit extents: the block's tile of a tensor is loaded,
or allocated, at the dims the block holds more than one element of. A loop is a macro
called with those buffers, and its own tiles (``Graph.tmap`` of the tier) say how it slices
them, so a tier tile has to have exactly the dims of the stored block tile.

``apply_tiling`` used to decide that by another rule -- keep a dim unless it is a batch dim
the *iteration* holds one element of -- which is the stored rank only when no other unit
extent is around. Two kinds of tiling fell outside it:

* a block one row (or one column) wide. The row dim is not a batch dim, so the tier kept
  it, and sliced a buffer that does not have it: ``Buffer is 1-dimensional (shape=(4096,)),
  but 2 index(es) were provided``. Every looped single-row candidate failed there;
* a loop that walks a batch dim one element at a time (a split axis at step 1). The tier
  dropped the dim although the block stores all of it, so every iteration was handed the
  whole block, and the paged recognizer read the row tile for the split step.

The first kind now builds and validates. The second is handed the right slices but still
has no kernel: the slice has no reduced axis left for ``T.reduce_*`` to name.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.codegen.operands import kept_extents
from cleave.dynamic.paging import read_dense_combine
from cleave.ir.graph import Graph, GraphNode, OutputNode, PlaceHolderNode
from cleave.ir.graph_utils import find_topo_sort, gather_all_nodes
from cleave.ir.ops import Div, ReduceSum, wire_input
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling

from tests._graphs import (
    batched_matmul_splitk_combine_graph,
    matmul_splitk_combine_graph,
    rowsum_graph,
)
from tests._lowering import lower_to_cuda
from tests._tolerances import tolerance_for
from tests.graph.apply_tiling._test_utils import (
    assert_candidate_compiles_and_runs,
    build_tiled_candidate,
)


def _candidate(graph, output_tile, **steps):
    rstep = {
        node.name: {} if node.name not in steps else {node.raxis[0].var.name: steps[node.name]}
        for node in graph.nodes
        if not node.is_placeholder() and not node.is_output()
    }
    assert set(steps) <= set(rstep), f"no such reduction: {set(steps) - set(rstep)}"
    return build_tiled_candidate(graph, output_tile, rstep)


def _loop(scheduled):
    (loop,) = [n for n in scheduled.nodes if isinstance(n, GraphNode)]
    return loop.graph


def _ints(extents):
    return tuple(int(e) for e in extents)


# name -> (graph builder, dims, output tile, steps, the operand's tier tile: (tile, of, axes))
SINGLE_ROW = {
    "rowsum_one_row": (rowsum_graph, (4096, 4096), [1], dict(rowsum_0=64),
                       ((64,), (4096,), (1,))),
    "combine_one_row": (matmul_splitk_combine_graph, (4, 128, 128), [1, 32], dict(reduction_0=2),
                        ((2, 32), (4, 32), (0, 2))),
    "combine_one_column": (matmul_splitk_combine_graph, (4, 128, 128), [32, 1], dict(reduction_0=2),
                           ((2, 32), (4, 32), (0, 1))),
    "batched_combine_one_row": (batched_matmul_splitk_combine_graph, (2, 4, 128, 128), [1, 1, 64],
                                dict(reduction_0=2), ((2, 64), (4, 64), (1, 3))),
    # every extent of the block is 1: the tile is stored 1-D, with its last dim
    "combine_one_element": (matmul_splitk_combine_graph, (4, 128, 128), [1, 1], dict(reduction_0=2),
                            ((2,), (4,), (0,))),
}
STEP_ONE = {
    "combine_step_one": (matmul_splitk_combine_graph, (4, 128, 128), [2, 16], dict(reduction_0=1),
                         ((1, 2, 16), (4, 2, 16), (0, 1, 2))),
    "batched_combine_step_one": (batched_matmul_splitk_combine_graph, (2, 4, 128, 128), [1, 2, 32],
                                 dict(reduction_0=1), ((1, 2, 32), (4, 2, 32), (1, 2, 3))),
}
# Tilings the old rule already got right: nothing about them may move.
CONTROLS = {
    "rowsum_eight_rows": (rowsum_graph, (4096, 4096), [8], dict(rowsum_0=64),
                          ((8, 64), (8, 4096), (0, 1))),
    "combine_step_two": (matmul_splitk_combine_graph, (4, 128, 128), [2, 16], dict(reduction_0=2),
                         ((2, 2, 16), (4, 2, 16), (0, 1, 2))),
    "batched_combine_step_two": (batched_matmul_splitk_combine_graph, (2, 4, 128, 128), [1, 2, 32],
                                 dict(reduction_0=2), ((2, 2, 32), (4, 2, 32), (1, 2, 3))),
}
TILINGS = {**SINGLE_ROW, **STEP_ONE, **CONTROLS}


@pytest.mark.parametrize("name", sorted(TILINGS))
def test_a_loop_tier_carries_the_dims_its_block_stores(name):
    build, dims, output_tile, steps, (tile, whole, axes) = TILINGS[name]
    scheduled = apply_tiling(_candidate(build(*dims), output_tile, **steps))
    tier = _loop(scheduled)

    (operand,) = [n for n in tier.nodes if isinstance(n, PlaceHolderNode)]
    it = tier.tmap[operand]
    assert (_ints(it.tile_size), _ints(it.shape), it.axes) == (tile, whole, axes)
    # ... which is the buffer the main tier stores for it and calls the loop with
    block_tile = scheduled.tmap[tier.binding[operand.name]].tile_size
    assert list(_ints(it.shape)) == (kept_extents(_ints(block_tile)) or [1])
    # ... and the dims of the node the tile stands for
    assert all(int(block_tile[axis]) == extent for axis, extent in zip(it.axes, it.shape))

    # An accumulator is not sliced by its loop, so its tile is its stored buffer as it is:
    # the tier allocates it at the tile and the reader indexes it at the stored rank.
    for node in tier.accum_nodes:
        acc = _ints(tier.tmap[node].tile_size)
        assert list(acc) == (kept_extents(acc) or [1]), (node.name, acc)


@pytest.mark.parametrize("name", sorted(SINGLE_ROW))
def test_a_loop_over_a_single_row_block_slices_the_stored_tile(name):
    """In the generated text: the loop declares its operand as the buffer it is called
    with, and indexes it with one slice per dim of that buffer."""
    build, dims, output_tile, steps, (tile, whole, _) = SINGLE_ROW[name]
    code = build_candidate_code(_candidate(build(*dims), output_tile, **steps))

    declared = [
        tuple(int(e) for e in shape.split(",") if e.strip())
        for shape in re.findall(r"PlaceHolder_A: T\.Tensor\(\(([^)]*)\)", code)
    ]
    # one iteration's tile, the loop's operand, the block's stored tile, the whole tensor
    assert declared == [tile, whole, whole, dims], declared
    # the loop's own slice of it comes first in the text, the grid's slice of the tensor last
    in_loop, in_grid = re.findall(
        r"PlaceHolder_A=_S\(\s*PlaceHolder_A\[(.*?)\],\s*squeeze_dim", code, flags=re.S
    )
    assert (in_loop.count(":"), in_grid.count(":")) == (len(whole), len(dims)), (in_loop, in_grid)


@pytest.mark.parametrize("name", ["rowsum_one_row", "combine_one_row", "batched_combine_one_row"])
def test_a_loop_over_a_single_row_block_lowers(name):
    """CPU: TileLang takes the kernel (it used to stop at the first slice of the loop)."""
    build, dims, output_tile, steps, _ = SINGLE_ROW[name]
    code = build_candidate_code(_candidate(build(*dims), output_tile, **steps))
    assert "main_kernel" in lower_to_cuda(code)


@pytest.mark.parametrize("name", ["rowsum_one_row", "combine_one_row", "batched_combine_one_row"])
def test_a_loop_over_a_single_row_block_validates(name):
    """GPU: the kernel of a looped single-row tiling is correct."""
    build, dims, output_tile, steps, _ = SINGLE_ROW[name]
    cand = _candidate(build(*dims), output_tile, **steps)
    assert_candidate_compiles_and_runs(cand, tolerance_for(build))


# ------------------------------------------------- a split axis walked one partial at a time
def _dense_combine_graph(split, rows, cols, from_end=False):
    """The combine the paged path reads: ``(split, rows)`` and ``(split, rows, cols)``
    partials summed over the split axis, then divided. ``from_end`` names that axis from
    the end (``-2`` and ``-3``)."""
    rowsum = ReduceSum((split, rows), axis=-2 if from_end else 0, dtype="float32",
                       out_dtype="float32", name="rowsum_combine_0")
    value = ReduceSum((split, rows, cols), axis=-3 if from_end else 0, dtype="float32",
                      out_dtype="float32", name="value_combine_0")
    div = Div((rows, cols), broadcast=0, dtype="float32", out_dtype="float32", name="div_0")
    wire_input(value, div, 0)
    wire_input(rowsum, div, 1)
    OutputNode(div, dtype="float16")
    nodes = gather_all_nodes(rowsum, value, div)
    return Graph(nodes=find_topo_sort([n for n in nodes if n.is_output()]))


@pytest.mark.parametrize("from_end", [False, True], ids=["from_front", "from_end"])
@pytest.mark.parametrize("rows", [1, 2, 4, 16])
@pytest.mark.parametrize("step", [1, 2, 4])
def test_the_recognizer_reads_the_split_step_off_the_split_axis(rows, step, from_end):
    """``read_dense_combine`` takes the value sum's step from the leading extent of its
    operand's tier tile. At step 1 the tier used to drop the split dim, so that extent was
    the row tile: 2, 4 and 4 for row tiles 2, 4 and 16 (capped at the split count)."""
    graph = _dense_combine_graph(4, 16, 64, from_end=from_end)
    cand = _candidate(graph, [rows, 64], value_combine_0=step, rowsum_combine_0=step)
    comb = read_dense_combine(apply_tiling(cand))
    assert (comb.split, comb.split_step) == (4, step)


def test_the_recognizer_asks_which_axis_a_sum_was_built_to_reduce():
    """Sums over the *rows* of each partial, not over the split axis: per-split
    normalisation, not a split combine. With one partial per block the split dim is a unit
    extent of every tile, so the emitted ``dim`` of both sums is 0 -- the first axis of
    the stored tile -- and the recognizer, which tested the emitted axis, took the graph
    for a combine over the split axis."""
    from cleave.dynamic.paging.recognize import NotDenseAttention

    split, rows, cols = 4, 16, 64
    rowsum = ReduceSum((split, rows), axis=1, dtype="float32", out_dtype="float32",
                       name="rowsum_combine_0")
    value = ReduceSum((split, rows, cols), axis=1, dtype="float32", out_dtype="float32",
                      name="value_combine_0")
    div = Div((split, cols), broadcast=0, dtype="float32", out_dtype="float32", name="div_0")
    wire_input(value, div, 0)
    wire_input(rowsum, div, 1)
    OutputNode(div, dtype="float16")
    nodes = gather_all_nodes(rowsum, value, div)
    graph = Graph(nodes=find_topo_sort([n for n in nodes if n.is_output()]))

    tiled = apply_tiling(_candidate(graph, [1, 64], value_combine_0=rows, rowsum_combine_0=rows))
    sums = {n.name: n for n in tiled.nodes if isinstance(n, ReduceSum)}
    assert {(n.flat_axis, n.axis, n.stored_axis) for n in sums.values()} == {(1, 0, 0)}
    with pytest.raises(NotDenseAttention, match=r"expected two reductions over the leading "
                                                r"\(split\) axis, got .*'rowsum_combine_0', 1"):
        read_dense_combine(tiled)


# ------------------------------------------- what a unit extent still leaves without a kernel
#
# A kernel stores a tile without its unit extents, and three kinds of op are emitted for
# the rank of the tensor, not of the tile. A tiling that hands one of them a buffer of
# lower rank used to be built and handed to TileLang, where it died with whatever the
# missing dim produced first ("Invalid reduce output shape", "not enough values to unpack",
# "Can't deduce copy extents", "Swizzle layout expects rank >= 2"). The default build
# refuses it, typed and with the reason.


def _build(cand):
    from cleave.core.errors import ScheduleRejected

    try:
        return build_candidate_code(cand), None
    except ScheduleRejected as refusal:
        return None, str(refusal)


@pytest.mark.parametrize("name", sorted(STEP_ONE))
def test_a_reduction_stepped_one_element_at_a_time_is_a_typed_refusal(name):
    build, dims, output_tile, steps, _ = STEP_ONE[name]
    cand = _candidate(build(*dims), output_tile, **steps)
    # apply_tiling schedules it: the paged path reads such a tiling (read_dense_combine)
    # and emits its own kernel for it.
    assert _loop(apply_tiling(cand)).extent == 4
    code, refusal = _build(cand)
    assert code is None and refusal == (
        "reduction_0 is handed one element of the axis it reduces per step (4 in all): the "
        "stored slice has no such axis to reduce over")


def test_an_elementwise_op_on_a_single_row_tile_is_a_typed_refusal():
    from tests._graphs import mha_split_combine_graph

    cand = _candidate(mha_split_combine_graph(32, 4, 4096, 64), [1, 1, 64],
                      value_combine_0=2, rowsum_combine_0=2)
    code, refusal = _build(cand)
    assert code is None and refusal == (
        "div_0 indexes 2 dims but its tile (1, 1, 64) is stored with 1: a unit extent that is "
        "not a batch dim is squeezed out of the tile")


def test_a_single_row_matmul_is_a_typed_refusal():
    from tests._graphs import matmul_graph

    graph = matmul_graph(128, 128, 4096)
    (mma,) = [n for n in graph.nodes if n.name == "mma_0"]
    cand = _candidate(graph, [1, 32], mma_0=32)
    code, refusal = _build(cand)
    assert code is None and refusal == (
        "mma_0: its result tile (32,) is stored with 1 dim(s); a matmul takes 2-D operands "
        "and gives a 2-D result, so a single row or column has no kernel")
    # the same matmul on a block of rows is built as ever
    code, refusal = _build(_candidate(matmul_graph(128, 128, 4096), [64, 32], mma_0=32))
    assert refusal is None and "T.gemm" in code


@pytest.mark.parametrize("name", sorted({**SINGLE_ROW, **CONTROLS}))
def test_reductions_on_single_row_tiles_are_not_refused(name):
    """What the refusals leave alone: a reduction is emitted for the stored tile (its
    ``dim`` is rebased), so a single row, column or element still has a kernel."""
    build, dims, output_tile, steps, _ = {**SINGLE_ROW, **CONTROLS}[name]
    code, refusal = _build(_candidate(build(*dims), output_tile, **steps))
    assert refusal is None and "main_kernel" in code


# fixture -> (candidates, kernels built, refusals by reason)
STEP = "is handed one element of the axis it reduces per step"
ROW = "a unit extent that is not a batch dim is squeezed out of the tile"
OUTCOMES = {
    "mha_split_combine": (lambda: __import__("tests._graphs", fromlist=["g"]).mha_split_combine_graph(
        32, 4, 4096, 64), 87, 34, {STEP: 44, ROW: 9}),
    "rmsnorm_split_combine": (
        lambda: __import__("tests._graphs", fromlist=["g"]).rmsnorm_matmul_split_combine_graph(
            4, 1024, 6144, 4096), 24, 6, {STEP: 9, ROW: 9}),
}


@pytest.mark.parametrize("name", sorted(OUTCOMES))
def test_every_candidate_is_built_or_refused_with_a_reason(name):
    """Every tiling the policy proposes for the two split combines: a kernel, or a typed
    refusal -- nothing else escapes the build (anything but ``ScheduleRejected`` propagates
    out of ``_build``)."""
    from collections import Counter

    from cleave.schedule import search_tiling

    build, n_candidates, n_built, refusals = OUTCOMES[name]
    cands = search_tiling(build(), topk=1000)
    assert len(cands) == n_candidates
    built, reasons = 0, Counter()
    for cand in cands:
        code, refusal = _build(cand)
        if code is not None:
            built += 1
            continue
        (reason,) = [r for r in (STEP, ROW) if r in refusal]
        reasons[reason] += 1
    assert (built, dict(reasons)) == (n_built, refusals)
