"""Reductions that share nothing are put in one loop only where one loop steps them both.

``apply_tiling`` gives every reduction that is stepped below its extent a loop, and puts a
second reduction into an existing loop when the two share a node or an input -- or simply
when their steps are equal. Equal steps are one loop only over axes of equal length: the
loop has one counter and one trip count, taken from the reduction that created it.
``x @ w`` over ``K = 256`` and ``y @ z`` over ``K2 = 512``, both stepped by 32, were put
in one loop of 8 iterations that cut ``z`` into 16 tiles, and the loop-extent guard then
refused the tiling: 36 of the 216 tilings of that program (and of ``x @ w + y @ z``), every
one in which the two steps agree.

Nothing needs that merge. The two matmuls read different tensors, so a loop of its own
costs the second one nothing but the pipeline TileLang gives one loop per kernel -- the
form every tiling with two different steps already has. A reduction now joins a loop it
shares nothing with only where the loop steps as it does: the same step, the same number
of times.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.lang as cl
from cleave.frontend import trace
from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops import Matmul
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import search_tiling
from cleave.schedule.tiling import schedule_kernel

from tests._lowering import lower_to_cuda
from tests._tolerances import tolerance_for


def two_matmuls(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"], y: cl.Tensor["M", "K2"],
                z: cl.Tensor["K2", "N"]):
    """Two outputs, two matmuls, no tensor in common."""
    return x @ w, y @ z


def sum_of_matmuls(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"], y: cl.Tensor["M", "K2"],
                   z: cl.Tensor["K2", "N"]):
    """One output; the matmuls still share no operand."""
    return x @ w + y @ z


LONGER = dict(M=128, K=256, N=128, K2=512)
EQUAL = dict(M=128, K=256, N=128, K2=256)


def _ir(fn, sizes):
    (graph,) = list(GraphSet.identity(trace(fn, sizes, cl.f16)))
    (ir,) = lower(graph)
    return ir


def _steps(cand):
    """``{reduced extent: step}`` of the two matmuls of ``cand``."""
    return {int(node.raxis[0].dom.extent): int(next(iter(step.values())))
            for node, step in cand.rstep_map.items() if isinstance(node, Matmul)}


def _loops(tiled):
    """``(trip count, matmuls inside)`` of every loop of a scheduled kernel, in order."""
    return [(tier.extent, sorted(n.name for n in tier.nodes if isinstance(n, Matmul)))
            for tier, owner in graph_tiers(tiled) if owner is not None]


def _candidate(fn, sizes, tile, steps, n_threads=128):
    (cand,) = [c for c in search_tiling(_ir(fn, sizes), topk=1000)
               if list(c.output_tile) == tile and _steps(c) == steps
               and c.n_threads == n_threads]
    return cand


@pytest.mark.parametrize("fn", [two_matmuls, sum_of_matmuls], ids=lambda fn: fn.__name__)
def test_equal_steps_over_axes_of_different_length_get_a_loop_each(fn):
    """Every tiling of the program is scheduled; it was all but the 36 with equal steps."""
    cands = search_tiling(_ir(fn, LONGER), topk=1000)
    equal = [c for c in cands if _steps(c)[256] == _steps(c)[512] < 256]
    assert len(equal) == 36
    for cand in cands:
        tiled = schedule_kernel(cand)  # no ScheduleRejected
        steps = _steps(cand)
        # one loop per matmul that is stepped below its extent, each with its own trips
        assert sorted(trips for trips, _ in _loops(tiled)) == sorted(
            extent // step for extent, step in steps.items() if step < extent)
        assert all(len(inside) == 1 for _, inside in _loops(tiled))


@pytest.mark.parametrize("fn", [two_matmuls, sum_of_matmuls], ids=lambda fn: fn.__name__)
def test_equal_steps_over_axes_of_equal_length_stay_in_one_loop(fn):
    """What the rule leaves alone: with ``K2 == K`` equal steps are one loop for both
    matmuls, as they were, and different steps a loop each."""
    cands = search_tiling(_ir(fn, EQUAL), topk=1000)
    merged = 0
    for cand in cands:
        loops = _loops(schedule_kernel(cand))
        first, second = (step for _, step in sorted(_steps_by_name(cand).items()))
        if first == second < 256:
            assert [len(inside) for _, inside in loops] == [2]
            merged += 1
        else:
            assert all(len(inside) == 1 for _, inside in loops)
    assert merged


def _steps_by_name(cand):
    return {node.name: int(next(iter(step.values())))
            for node, step in cand.rstep_map.items() if isinstance(node, Matmul)}


def test_each_loop_has_the_trip_count_of_its_own_reduction():
    """The tiling that was one loop of 8 iterations over 16 tiles of ``z``."""
    tiled = schedule_kernel(_candidate(two_matmuls, LONGER, [64, 64], {256: 32, 512: 32}))
    assert sorted(trips for trips, _ in _loops(tiled)) == [8, 16]


def test_the_second_loop_is_a_plain_one_and_the_kernel_lowers():
    """Two loops in one kernel: TileLang pipelines the first only (``_pipeline_one_loop``)."""
    code = build_candidate_code(_candidate(two_matmuls, LONGER, [64, 64], {256: 32, 512: 32}))
    assert re.findall(r"for i0 in T\.(\w+)\((\d+)", code) == [("Pipelined", "8"), ("serial", "16")]
    assert "main_kernel" in lower_to_cuda(code)


def _validate_equal_steps(fn, tolerance):
    """GPU: every tiling of ``fn`` whose two steps agree -- the 36 that had no kernel."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to compile and run candidates.")
    from cleave.runtime.evaluate import run_candidates

    cands = [c for c in search_tiling(_ir(fn, LONGER), topk=1000)
             if _steps(c)[256] == _steps(c)[512] < 256]
    results = run_candidates(cands, "float16", max_parallel=8, tolerance=tolerance)
    assert len(results) == 36
    assert all(r.ok for r in results), [str(r.error)[:300] for r in results if not r.ok]


def test_gpu_a_loop_each_validates():
    """Two outputs, a plain matmul each, held to the bound measured on this program."""
    _validate_equal_steps(two_matmuls, tolerance_for("two_matmuls"))


def test_gpu_a_loop_each_validates_where_the_products_are_added():
    """One output: the sum of two products rounds once more than either."""
    _validate_equal_steps(sum_of_matmuls, tolerance_for("sum_of_matmuls"))
