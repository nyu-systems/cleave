"""A split-KV combine builds wherever the scheduler places its two sums.

The combine sums the value partials and the row-sum partials over the split axis and
divides one by the other. Each sum has its own step, so each is either a loop over the
split axis or, at the whole split count, a plain reduction next to the divide. Only one
placement used to give a kernel -- both sums in one loop -- for two reasons:

* the divide reads two fragments that TileLang lays out independently, which it can only
  do with the divisor staged through shared memory, and the pass that asks for the staging
  recognised the divide only when both operands were accumulators of one loop. Every other
  placement stopped at ``no available layout found``;
* with the staging, the placement in which only the row sum is looped lowered -- and hung
  on the GPU. Its value sum was emitted in front of the loop, where TileLang runs it on
  both of the warp groups it splits the block into at a pipelined loop.

Both are fixed in codegen: the staging is asked for whatever the placement, and work that
no loop reads is emitted behind the loops.

The pass that asks for the staging also recognised the two sums by ``ReduceSum.axis == 0``,
which is the place of the split axis in the stored tile only for an axis named from the
front: named from the end (``axis=-2`` / ``-3``, the same axis) it stays the built axis, 1
behind the head dim, and no placement of such a combine got the staging.

One placement was left to TileLang's own limits: each sum in a loop of its own (two steps
below the split count). The vendored TileLang pipelines one loop per kernel -- the loop
variable of a second ``T.Pipelined`` loop, and of one that loads nothing wide enough to
transfer, is left behind outside the loop -- so the tiler emits those loops as plain ones.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.codegen import passes
from cleave.codegen.kernel import emission_order
from cleave.ir.graph import GraphNode
from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops import ReduceSum
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling

from tests._graphs import mha_split_combine_graph
from tests._lowering import lower_to_cuda
from tests._tolerances import tolerance_for
from tests.graph.apply_tiling._test_utils import (
    assert_candidate_compiles_and_runs,
    build_tiled_candidate,
)

SPLIT = 4
# (value step, row-sum step) -> the loops of the kernel, in the order they are emitted
PLACEMENTS = {
    "neither_looped": ((SPLIT, SPLIT), []),
    "value_looped": ((2, SPLIT), ["GraphNode_value_combine_0"]),
    "rowsum_looped": ((SPLIT, 2), ["GraphNode_rowsum_combine_0"]),
    "one_loop_for_both": ((2, 2), ["GraphNode_rowsum_combine_0"]),  # what always built
}


def _candidate(value_step, rowsum_step, split=SPLIT, from_end=False, rows=128):
    graph = mha_split_combine_graph(32, split, 4096, 64, from_end=from_end)
    steps = {"value_combine_0": value_step, "rowsum_combine_0": rowsum_step}
    rstep = {
        node.name: {node.raxis[0].var.name: steps[node.name]} if node.name in steps else {}
        for node in graph.nodes
        if not node.is_placeholder() and not node.is_output()
    }
    return build_tiled_candidate(graph, [1, rows, 64], rstep)


def _loops(code):
    """``(loop, kind)`` of every loop of the kernel, in the order they are emitted."""
    return re.findall(r"def (GraphNode_\w+)\((?:(?!\ndef ).)*?for i0 in T\.(\w+)\(", code,
                      flags=re.S)


def _main_tier_calls(code):
    """The macro calls of the block's own tier, in the order they are emitted."""
    body = re.search(r"def main_kernel_iter\(.*?\n\n\n", code, flags=re.S).group(0)
    return re.findall(r"^    (ukernel_\w+|GraphNode_\w+)\(", body, flags=re.M)


@pytest.mark.parametrize("from_end", [False, True], ids=["from_front", "from_end"])
@pytest.mark.parametrize("placement", sorted(PLACEMENTS))
def test_the_divisor_is_staged_wherever_the_sums_are_placed(placement, from_end):
    steps, loops = PLACEMENTS[placement]
    cand = _candidate(*steps, from_end=from_end)
    code = build_candidate_code(cand)
    calls = _main_tier_calls(code)

    # Both sums run over the leading axis of the tensor they are stored in, whichever end
    # their axis was named from; only what is emitted follows the naming.
    sums = [n for tier, _ in graph_tiers(apply_tiling(cand)) for n in tier.nodes
            if isinstance(n, ReduceSum)]
    assert sorted((n.name, n.flat_axis, n.stored_axis) for n in sums) == [
        ("rowsum_combine_0", 1, 0), ("value_combine_0", 1, 0)]
    assert sorted((n.name, n.axis, n.axis_arg) for n in sums) == (
        [("rowsum_combine_0", 1, -2), ("value_combine_0", 1, -3)] if from_end
        else [("rowsum_combine_0", 0, 0), ("value_combine_0", 0, 0)])

    assert [c for c in calls if c.startswith("GraphNode_")] == loops
    (divide,) = [c for c in calls if c.startswith("ukernel_div_")]
    assert divide == "ukernel_div_broadcast_0_resolve_rhs_layout_2dims_rrr", placement
    # CPU: TileLang finds a layout (it used to for one_loop_for_both only).
    assert "main_kernel" in lower_to_cuda(code)


@pytest.mark.parametrize("placement", sorted(PLACEMENTS))
def test_work_no_loop_reads_is_emitted_behind_the_loops(placement):
    """A sum that is not looped neither feeds the other sum's loop nor reads it, so dataflow
    order may put it on either side. It goes behind: with only the row sum looped, the
    value sum used to be the first statement of the tier, in front of the loop."""
    steps, loops = PLACEMENTS[placement]
    scheduled = apply_tiling(_candidate(*steps))
    scheduled = passes.annotate_args(scheduled, is_main=True)
    scheduled = passes.annotate_tinfo(scheduled, is_main=True)

    order = [n for n in emission_order(scheduled) if not n.is_placeholder()]
    kinds = ["loop" if isinstance(n, GraphNode) else n.name for n in order]
    n_loops = len(loops)
    assert kinds[:n_loops] == ["loop"] * n_loops, kinds
    assert kinds[-2:] == ["div_0", "OutputNode"], kinds
    # ... and in the text: no macro call stands in front of a loop
    calls = _main_tier_calls(build_candidate_code(_candidate(*steps)))
    assert [c.startswith("GraphNode_") for c in calls[:n_loops]] == [True] * n_loops, calls


@pytest.mark.parametrize("steps", [(2, 4), (4, 2)], ids=["value_2_rowsum_4", "value_4_rowsum_2"])
def test_each_sum_in_a_loop_of_its_own_gets_the_staging_too(steps):
    """Split 8 with steps 2 and 4: two loops over the split axis with different trip
    counts. The divide is the one of every other placement. TileLang 0.1.13 does not
    lower two pipelined loops in one kernel ("variables (i0,) are used, but are not
    passed in as API arguments": 13 of this combine's tilings at split 8), so the second
    loop is emitted as a plain one, and the kernel lowers."""
    code = build_candidate_code(_candidate(*steps, split=8))
    calls = _main_tier_calls(code)
    assert [c.startswith("GraphNode_") for c in calls] == [True, True, False], calls
    assert calls[-1] == "ukernel_div_broadcast_0_resolve_rhs_layout_2dims_rrr"
    assert [kind for _, kind in _loops(code)] == ["Pipelined", "serial"]
    assert [name for name, _ in _loops(code)] == [c for c in calls if c.startswith("GraphNode_")]
    assert "main_kernel" in lower_to_cuda(code)


@pytest.mark.parametrize("placement", sorted(PLACEMENTS))
def test_a_kernel_with_one_loop_keeps_its_pipeline(placement):
    """What that leaves alone: a kernel with one loop, whichever sum is in it."""
    steps, loops = PLACEMENTS[placement]
    code = build_candidate_code(_candidate(*steps))
    assert _loops(code) == [(name, "Pipelined") for name in loops]


@pytest.mark.parametrize("rows, kind", [(2, "serial"), (4, "serial"), (8, "Pipelined"),
                                        (128, "Pipelined")])
def test_a_loop_with_nothing_to_transfer_is_not_pipelined(rows, kind):
    """Only the row sum looped, on blocks of a few rows: each iteration loads two partials
    of ``rows`` float16 row sums. Under 16 bytes a row, TileLang has nothing to transfer
    in a pipeline and loses the loop variable (2 of this combine's tilings at split 4, 1 of
    the RMSNorm combine's, 24 of each matmul split-K combine's); such a loop is emitted as
    a plain one. From 16 bytes on it is pipelined as ever."""
    steps, _ = PLACEMENTS["rowsum_looped"]
    code = build_candidate_code(_candidate(*steps, rows=rows))
    assert _loops(code) == [("GraphNode_rowsum_combine_0", kind)]
    assert "main_kernel" in lower_to_cuda(code)


def test_one_loop_stays_pipelined_whatever_else_it_loads():
    """What that leaves alone: both sums in one loop load the row sums next to the value
    partials, which are wide enough, so the loop keeps its pipeline even on four rows."""
    steps, _ = PLACEMENTS["one_loop_for_both"]
    code = build_candidate_code(_candidate(*steps, rows=4))
    assert _loops(code) == [("GraphNode_rowsum_combine_0", "Pipelined")]
    assert "main_kernel" in lower_to_cuda(code)


@pytest.mark.parametrize("case", ["two_loops", "two_loops_swapped", "four_rows", "two_rows"])
def test_a_split_combine_with_a_serial_loop_validates(case):
    """GPU: the kernels that exist because a loop is emitted serially."""
    cand = {
        "two_loops": lambda: _candidate(2, 4, split=8),
        "two_loops_swapped": lambda: _candidate(4, 2, split=8),
        "four_rows": lambda: _candidate(*PLACEMENTS["rowsum_looped"][0], rows=4),
        "two_rows": lambda: _candidate(*PLACEMENTS["rowsum_looped"][0], rows=2),
    }[case]()
    assert_candidate_compiles_and_runs(cand, tolerance_for(mha_split_combine_graph))


@pytest.mark.parametrize("placement", ["neither_looped", "value_looped", "rowsum_looped"])
def test_a_split_combine_validates_wherever_its_sums_are_placed(placement):
    """GPU: the three placements that had no kernel. ``rowsum_looped`` did not return
    with its value sum in front of the loop."""
    steps, _ = PLACEMENTS[placement]
    assert_candidate_compiles_and_runs(_candidate(*steps), tolerance_for(mha_split_combine_graph))


@pytest.mark.parametrize("placement", sorted(PLACEMENTS))
def test_a_split_combine_named_from_the_end_validates(placement):
    """GPU: the same combine with its split axis named from the end. None of its
    placements had a kernel ("no available layout found")."""
    steps, _ = PLACEMENTS[placement]
    assert_candidate_compiles_and_runs(_candidate(*steps, from_end=True),
                                       tolerance_for(mha_split_combine_graph))
