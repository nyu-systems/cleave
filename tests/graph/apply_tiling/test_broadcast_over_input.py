"""A loop's accumulator broadcast over an operand in global memory is read from shared
memory.

``x / sum(x, -1)``: the quotient's tile is the whole row, so the kernel's input is read
where it lies, in global memory, and the row sum is a fragment with one entry per row. The
quotient is then the only other fragment of the op, and TileLang lays it out after the
sum. A sum that a loop accumulates has the layout of the slices it was reduced from, and
slices of 512 columns or fewer put each row on a few of the block's 128 threads only: one
row per warp for a slice of 64 columns, held by each of the warp's 32 threads. The
quotient then had whole rows per thread. On 4096 columns the kernels were right, and

* did up to 32 times the work: 4.6 ms a launch where it is 0.13 ms now (tile of 4 rows);
* kept 4096 to 32768 elements per thread, a frame of 8 to 64 KiB, for which the driver
  allocates 3.7 GiB and more of thread stacks at launch
  (``tests/runtime/test_thread_stacks.py``);
* took the device compiler 100 to 158 s each, for loops unrolled 512-fold and more -- past
  the 300 s a candidate has when a whole case compiles at once. One tiling of ``sum(x, -1)``
  returned next to ``x / sum(x, -1)`` had not compiled after 36 minutes.

With the sum read from shared memory nothing ties the quotient to it, and it is laid out
across the threads of the block: 128 elements per thread for a tile of 4 rows.

A sum taken in one step is no accumulator: it is reduced from the whole tile in the tier of
the quotient, every thread holds every row's sum, and the quotient was laid out across the
threads all along. The same holds for the max of an in-tile softmax. Those are left as
they are.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.lang as cl
from cleave.codegen import passes
from cleave.core.errors import ScheduleRejected
from cleave.frontend import trace
from cleave.ir.graph import GraphNode
from cleave.ir.ops import Add, Div, Mul, Sub
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import search_tiling
from cleave.schedule.tiling import schedule_kernel

from tests._graphs import mha_bhsd_graph, rowsum_div_graph
from tests._lowering import lower_to_cuda
from tests._tolerances import tolerance_for
from tests.graph.apply_tiling._test_utils import assert_candidate_compiles_and_runs

DIMS = (4096, 4096)  # of the end2end case


def _steps(cand):
    return [int(step) for steps in cand.rstep_map.values() for step in steps.values()]


def _candidate(tile, step, dims=DIMS):
    (cand,) = [c for c in search_tiling(rowsum_div_graph(*dims), topk=1000)
               if list(c.output_tile) == tile and _steps(c) == [step]]
    return cand


def _annotated(cand):
    """The scheduled graph after the passes the build runs before it emits a kernel."""
    graph = schedule_kernel(cand)
    graph = passes.annotate_args(graph, is_main=True)
    graph = passes.annotate_tinfo(graph, is_main=True)
    graph = passes.annotate_tc_layout(graph)
    return passes.annotate_fragment_layout_conflicts(graph)


def _scope(graph, edge):
    """Where an operand lies: ``g``lobal, ``s``hared or a fragment (``r``)."""
    if isinstance(edge.src_node, GraphNode):  # the accumulator of a loop
        return edge.src_node.graph.tinfo_map[edge.accum_node][1]
    return graph.tinfo_map[edge.src_node][1]


def _broadcasts(graph):
    """``(node, scope of its first operand, scope of its second)`` for every elementwise
    node of the kernel whose second operand lacks an axis of the first."""
    found = []
    for node in graph.nodes:
        if isinstance(node, GraphNode):
            found += _broadcasts(node.graph)
        elif isinstance(node, (Add, Sub, Mul, Div)) and node.const is None and node.broadcast != -1:
            found.append((node, *(_scope(graph, edge) for edge in node.inputs)))
    return found


def _frame(code):
    """``{array: elements}`` a thread keeps, and the longest unrolled loop, of the lowered
    kernel."""
    src = lower_to_cuda(code)
    arrays = {name: int(n) for name, n in
              re.findall(r"^\s+(?:float|half_t|uint\d)\s+(\w+)\[(\d+)\];", src, re.M)}
    loops = [int(n) for n in re.findall(r"#pragma unroll\s+for \(int \w+ = 0; \w+ < (\d+);", src)]
    return arrays, max(loops)


def test_a_looped_row_sum_is_read_from_shared_memory_by_the_quotient():
    """Every tiling of the end2end case: the input in global memory, the sum a fragment;
    24 of the 28 loop the sum."""
    staged = 0
    for cand in search_tiling(rowsum_div_graph(*DIMS), topk=1000):
        try:
            graph = _annotated(cand)
        except ScheduleRejected:
            continue
        ((div, lhs, rhs),) = _broadcasts(graph)
        looped = _steps(cand) != [4096]
        assert (type(div), lhs, rhs) == (Div, "g", "r")
        assert div.resolve_operand_layout == ("rhs" if looped else None), cand.output_tile
        staged += looped
    assert staged == 24
    code = build_candidate_code(_candidate([4, 4096], 64))
    assert "def ukernel_div_broadcast_0_resolve_rhs_layout_2dims_grr(A, B, C):" in code
    assert "B_ptr = T.alloc_shared(B.shape, 'float16')" in code.replace('"', "'")
    assert "C[i0, i1] = A[i0, i1] / B_ptr[i0]" in code
    code = build_candidate_code(_candidate([4, 4096], 4096))
    assert "def ukernel_div_broadcast_0_2dims_grr(A, B, C):" in code


@pytest.mark.parametrize("tile, step, rows", [
    ([4, 4096], 64, 4), ([4, 4096], 512, 4), ([8, 4096], 256, 8), ([16, 4096], 128, 16),
    ([16, 4096], 512, 16), ([4, 4096], 2048, 4), ([4, 4096], 4096, 4)],
    ids=["4_rows", "4_rows_step_512", "8_rows", "16_rows", "16_rows_step_512",
         "4_rows_two_steps", "4_rows_one_step"])
def test_the_quotient_is_laid_out_across_the_threads(tile, step, rows):
    """A thread keeps its share of the tile, ``rows * 4096 / 128`` elements, where it kept
    one to eight whole rows of 4096 (steps up to 512); the loops that fill and store it
    shrink with it. The last two tilings were laid out so before."""
    arrays, longest = _frame(build_candidate_code(_candidate(tile, step)))
    assert arrays["t_div_0"] == rows * 4096 // 128
    assert longest <= 64  # it was 512 and more


def test_the_quotient_took_the_layout_of_the_sum():
    """What the rule changes, on the kernel as it was built: with the divisor left in its
    fragment every thread holds a row."""
    from cleave.codegen import TilelangCodeGen

    for tile, step, kept, unrolled in (([4, 4096], 64, 4096, 512),
                                       ([16, 4096], 512, 32768, 4096)):
        graph = _annotated(_candidate(tile, step))
        ((div, _, _),) = _broadcasts(graph)
        div.resolve_operand_layout = None
        arrays, longest = _frame(TilelangCodeGen().gen(graph))
        assert (arrays["t_div_0"], longest) == (kept, unrolled)


def quotient(x: cl.Tensor["R", "C"]):
    return x / cl.sum(x, -1, keepdim=True)


def product(x: cl.Tensor["R", "C"]):
    return x * cl.sum(x, -1, keepdim=True)


def total(x: cl.Tensor["R", "C"]):
    return x + cl.sum(x, -1, keepdim=True)


def difference(x: cl.Tensor["R", "C"]):
    return x - cl.sum(x, -1, keepdim=True)


@pytest.mark.parametrize("fn, op", [(quotient, Div), (product, Mul), (total, Add),
                                    (difference, Sub)], ids=lambda v: v.__name__)
def test_any_operator_that_broadcasts_a_looped_sum_over_the_input_stages_it(fn, op):
    """The layout is TileLang's answer to the operands, not to the operator."""
    (graph,) = list(GraphSet.identity(trace(fn, dict(R=64, C=1024), cl.f16)))
    (ir,) = lower(graph)
    staged, lowered = 0, set()
    for cand in search_tiling(ir, topk=1000):
        try:
            tiled = _annotated(cand)
        except ScheduleRejected:
            continue
        ((node, lhs, rhs),) = _broadcasts(tiled)
        looped = _steps(cand) != [1024]
        assert (type(node), lhs, rhs) == (op, "g", "r")
        assert node.resolve_operand_layout == ("rhs" if looped else None)
        staged += looped
        rows = int(cand.output_tile[0])
        if rows not in lowered:  # one tiling per tile: a thread keeps its share of it
            lowered.add(rows)
            arrays, _ = _frame(build_candidate_code(cand))
            assert max(arrays.values()) == rows * 1024 // cand.n_threads, arrays
    assert staged >= 20 and len(lowered) >= 5


def test_a_fragment_broadcast_over_a_fragment_is_left_where_it_is():
    """The rule is for an operand with no layout of its own. ``P / l`` of an attention
    divides a fragment by the row sum: the quotient takes the layout of ``P``."""
    seen = 0
    for cand in search_tiling(mha_bhsd_graph(1, 32, 4096, 4096, 64), topk=8):
        for node, lhs, rhs in _broadcasts(_annotated(cand)):
            assert (lhs, rhs, node.resolve_operand_layout) == ("r", "r", None)
            seen += 1
    assert seen == 8


@pytest.mark.parametrize("tile, step", [([4, 4096], 64), ([8, 4096], 64), ([16, 4096], 4096)],
                         ids=["4_rows", "8_rows", "16_rows_one_step"])
def test_gpu_the_quotient_validates(tile, step):
    assert_candidate_compiles_and_runs(_candidate(tile, step), tolerance_for(rowsum_div_graph))
