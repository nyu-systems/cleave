"""A reduction that reads an input through a split view (CPU).

``apply_tiling`` takes the split views out of a graph, so a reduction that read
``SplitView(x)`` reads the input ``x`` itself: a tensor of one dim less than the reduction's
own ``dims``, which has no split dim (the grid walks the splits; a block holds one of them).
Every tile of that input -- the candidate's, the main tier's, a loop tier's ``Iterator`` --
numbers the input's dims. A reduction of ``(f, R, C / f)`` over its last dim reduces dim 1 of
``(R, C)``.

The tiler used to ask those tiles for the reduction's own dim number. Two things followed,
for a ``ReduceSum`` as for the ``ReduceMax`` a split partial exports as its shift:

* in a loop tier the answer was "this tile does not carry that dim", which reads as a loop
  that walks the reduced axis one element at a time: every looped tiling was refused as
  "handed one element of the axis it reduces per step", whatever its step (64, 128, ...).
  The partial of ``cl.softmax(x) @ v`` split over the softmax axis lost every tiling that
  loops over its keys that way;
* a reduction built with its axis counted from the front kept that number as the ``dim``
  of ``T.reduce_*``, an index into the stored slice, which has no split dim: TileLang
  refused the kernel ("Invalid reduce output shape"), or, where the number happened to
  name a dim of the slice, reduced another axis.

``ReduceSum.operand_dims`` says which dim of the value it reads each of a reduction's dims
is, and everything that asks a tile about a reduction goes through it.
"""

from collections import Counter

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.core.errors import ScheduleRejected
from cleave.ir.graph import GraphNode, OutputNode, PlaceHolderNode
from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops import ReduceMax, ReduceSum, SplitView
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import search_tiling
from cleave.schedule.tiling import schedule_kernel

from tests import _graphs as g
from tests._lowering import lower_to_cuda

FACTOR = 4
# name -> (input shape, the input's axis that is split, the reduction's axis as it is built,
#          the dim of the reduction's tensor that is, the input's dim that is)
CASES = {
    "rows": ((64, 1024), 1, -1, 2, 1),
    "rows_from_the_front": ((64, 1024), 1, 2, 2, 1),
    "batched_rows": ((4, 64, 1024), 2, -1, 3, 2),
    "batched_rows_from_the_front": ((4, 64, 1024), 2, 3, 3, 2),
    "columns": ((1024, 64), 0, 1, 1, 0),
}
REDUCERS = {"sum": ReduceSum, "max": ReduceMax}


def _graph(shape, split_axis, axis, cls):
    """``cls(SplitView(x), axis)``: the smallest graph in which a reduction reads an input
    through a split."""
    view = SplitView(shape, axis=split_axis, factor=FACTOR, dtype="float16", name="split_x")
    dims = tuple(int(s) for s in view.output_shape)
    reduction = cls(dims, axis=axis, name="red_0", dtype="float16", out_dtype="float32")
    g._connect(view, reduction, 0, 0)
    OutputNode(reduction, dtype="float32")
    return g._finalize(reduction), view


def _step(cand, name="red_0"):
    (steps,) = [v for k, v in cand.rstep_map.items() if k.name == name]
    (step,) = steps.values()
    return int(step)


def _scheduled(cand):
    """``(tier, the reduction, the tile of the input it reads)`` of the scheduled kernel."""
    tiled = schedule_kernel(cand)
    for tier, _ in graph_tiers(tiled):
        for node in tier.nodes:
            if node.name == "red_0":
                (edge,) = node.inputs
                assert isinstance(edge.src_node, PlaceHolderNode)
                return tier, node, tier.tmap[edge.src_node]
    raise AssertionError("no reduction in the scheduled graph")


def _source(cand):
    built = build_candidate_code(cand)
    return built[0] if isinstance(built, tuple) else built


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_dims_of_the_input_are_numbered_without_the_split_dim(name):
    """On the scheduled graph the reduction reads the input, and says which of its dims is
    which of the input's: the split dim is none of them, the dims in front of it keep
    their number and the ones behind it move up by one. The split dim is where
    ``SplitView`` put it."""
    shape, split_axis, axis, reduced, own = CASES[name]
    ir, view = _graph(shape, split_axis, axis, ReduceSum)
    cand = next(c for c in search_tiling(ir, topk=1000) if _step(c) > 1)
    _, node, _ = _scheduled(cand)
    dims, rank = node.operand_dims()
    assert rank == len(shape) == len(node.dims) - 1
    assert dims.index(None) == view.split_pos and dims.count(None) == 1
    assert [d for d in dims if d is not None] == list(range(len(shape)))
    assert node.flat_axis == reduced and dims[reduced] == own
    # every dim but the split dim has the extent of the input's dim, the split one aside
    for dim, own_dim in enumerate(dims):
        if own_dim is not None and own_dim != split_axis:
            assert int(node.dims[dim]) == shape[own_dim]
    assert int(node.dims[dims.index(split_axis)]) * FACTOR == shape[split_axis]


def test_a_reduction_that_reads_a_value_of_its_own_rank_numbers_the_dims_alike():
    """Nothing changes for a reduction that reads what it was built for (every reduction
    of a graph without a split, and every one that reads a computed value)."""
    ir = g.rowsum_graph(256, 4096)
    (node,) = [n for n in ir.nodes if isinstance(n, ReduceSum)]
    assert node.operand_dims() == (list(range(len(node.dims))), len(node.dims))
    cand = next(c for c in search_tiling(ir, topk=1000) if 1 < _step(c, node.name) < 4096)
    tiled = schedule_kernel(cand)
    (loop,) = [n for n in tiled.nodes if isinstance(n, GraphNode)]
    (looped,) = [n for n in loop.graph.nodes if isinstance(n, ReduceSum)]
    it = loop.graph.tmap[looped.inputs[0].src_node]
    assert looped.position_in(it) == it.position(looped.flat_axis, len(looped.dims))


@pytest.mark.parametrize("reducer", sorted(REDUCERS))
@pytest.mark.parametrize("name", sorted(CASES))
def test_a_looped_tiling_is_not_refused_as_a_step_of_one(name, reducer):
    """Every tiling that loops over the reduced axis in steps of two or more is scheduled;
    it used to be refused with the step-1 message, at any step. A loop that really takes
    one element per iteration is still refused, and says so."""
    shape, split_axis, axis, reduced, own = CASES[name]
    ir, _ = _graph(shape, split_axis, axis, REDUCERS[reducer])
    extent = shape[split_axis] // FACTOR
    verdicts = Counter()
    for cand in search_tiling(ir, topk=1000):
        step = _step(cand)
        try:
            tier, node, it = _scheduled(cand)
        except ScheduleRejected as refusal:
            assert step == 1, (step, str(refusal))
            assert "is handed one element of the axis it reduces per step" in str(refusal)
            verdicts["refused, step 1"] += 1
            continue
        assert step > 1
        position = node.position_in(it)
        assert position is not None and int(it.tile_size[position]) == min(step, extent)
        # the tile is the input's: it is asked for the input's dim, not the reduction's
        assert it.position(own, len(shape)) == position
        verdicts["looped" if step < extent else "whole axis"] += 1
    assert verdicts["looped"] >= 2 and verdicts["whole axis"] >= 1, verdicts
    if name != "columns":  # the row tilings start at a coalesced read, never at one element
        assert not verdicts["refused, step 1"], verdicts


@pytest.mark.parametrize("reducer", sorted(REDUCERS))
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_emitted_dim_indexes_the_stored_slice(name, reducer):
    """``dim`` of the emitted ``T.reduce_*`` is the place of the reduced axis in the slice
    the kernel stores: counted from the front for an axis built from the front (it used to
    stay the number among the reduction's own dims, split dim included), from the end for
    one built from the end. A sample of the kernels goes through TileLang."""
    shape, split_axis, axis, reduced, own = CASES[name]
    cls = REDUCERS[reducer]
    ir, _ = _graph(shape, split_axis, axis, cls)
    lowered = Counter()
    for cand in search_tiling(ir, topk=1000):
        if _step(cand) == 1:
            continue
        tier, node, it = _scheduled(cand)
        position = node.position_in(it)
        stored = [int(t) for t in it.tile_size if int(t) > 1] or [1]
        in_front = sum(1 for t in list(it.tile_size)[:position] if int(t) > 1)
        assert node.stored_axis == in_front
        if axis >= 0:
            assert node.axis == node.axis_arg == in_front
        else:
            assert node.axis == reduced and node.axis_arg == in_front - len(stored)
        looped = _step(cand) < shape[split_axis] // FACTOR
        key = (looped, len(stored))
        if lowered[key] < 2:  # two kernels of each kind: looped or not, by stored rank
            lowered[key] += 1
            code = _source(cand)
            macro = f"ukernel_{cls.kind}_axis{node.axis}_"
            assert macro in code and f"dim={node.axis_arg}," in code
            lower_to_cuda(code)
    assert {looped for looped, _ in lowered} == {True, False}, lowered
