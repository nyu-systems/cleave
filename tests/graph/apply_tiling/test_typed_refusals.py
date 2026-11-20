"""``apply_tiling`` refuses a tiling it cannot build with ``ScheduleRejected`` (CPU).

Four tilings the policy proposes and the tiler used to mishandle or mis-type:

* a value computed inside a loop and also read outside it. The loop's inner placeholders
  then leaked into the main graph and the candidate died with ``KeyError:
  'PlaceHolder_...'`` at the main ``tmap`` fill. It is the traced-order softmax,
  ``exp / sum(exp)`` followed by the value matmul, whenever the row sum is looped: ``exp``
  goes into the loop with the sum and is still read by the ``div`` outside it.
* reductions merged into one loop with different steps. The loop keeps the trip count of
  the reduction that created it while the tiles are cut by the other one, so the kernel was
  *built*, with a loop that ran past the end of its tiles: ``for i0 in T.Pipelined(1024)``
  over 32 tiles of 128.

* a grid dim that no launch axis indexes. The tiler always refused it on purpose, but
  with ``ValueError``.
* a tile with fewer dims than its tensor. The output of ``x @ column`` is ``(M, 1)`` and
  its tile ``[m]``: the unit dim is not a batch dim, so nothing puts it back. The grid was
  then planned for a rank-1 tensor and every candidate of such a graph died, with
  ``KeyError: 1`` when the matmul was looped and on an assert when it was not.

All four are per-candidate refusals now: callers that drop a candidate on
``ScheduleRejected`` (the pipeline's schedule stage, the softmax accept filter) can tell
them from a bug.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.lang as cl
from cleave.core.errors import ScheduleRejected
from cleave.frontend import trace
from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops import Div, Exp, Matmul, ReduceSum
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet
from cleave.schedule import apply_tiling, search_tiling
from cleave.schedule import tiling


def traced_order_attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                           v: cl.Tensor["H", "SKV", "D"]):
    """softmax(QK^T) @ V with the softmax written out: the exp feeds the sum AND the div."""
    e = cl.exp((q @ k.mT) / q.shape[-1] ** 0.5)
    return (e / cl.sum(e, -1, keepdim=True)) @ v


def column_weighted(v: cl.Tensor["M", "K"], x: cl.Tensor["K", "N"]):
    """(v @ e) / sum(e, 0): the matmul and the column sum both reduce K, and share ``e``."""
    e = cl.exp(x)
    return (v @ e) / cl.sum(e, 0, keepdim=True)


def matmul_of_rowsum(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    """x @ rowsum(w), the form the search finds for rowsum(x @ w): a matmul with a column."""
    return x @ w.sum(-1, keepdim=True)


def flash_rows(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """(e @ v) / sum(e): the value matmul and the row sum both reduce C, and share ``e``."""
    e = cl.exp(x)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def _identity_graph(fn, shapes, dtype=cl.f16):
    (graph,) = list(GraphSet.identity(trace(fn, shapes, dtype)))
    (ir,) = lower(graph)
    return ir


def _one(ir, cls, where=lambda n: True):
    (node,) = [n for n in ir.nodes if type(n) is cls and where(n)]
    return node


def _step(cand, node):
    """The reduction step ``cand`` gives the node named like ``node``."""
    (rstep,) = [v for n, v in cand.rstep_map.items() if n.name == node.name]
    (step,) = rstep.values()
    return int(step)


def _outcomes(cands):
    built, refused = [], []
    for cand in cands:
        try:
            built.append((cand, apply_tiling(cand)))
        except ScheduleRejected as e:
            refused.append((cand, str(e)))
    return built, refused


def test_value_read_inside_and_outside_a_loop_is_a_typed_refusal():
    ir = _identity_graph(traced_order_attention, dict(H=32, SQ=64, SKV=256, D=64))
    exp, rowsum = _one(ir, Exp), _one(ir, ReduceSum)
    div = _one(ir, Div, lambda n: n.const is None)
    cands = search_tiling(ir, topk=1000)

    # Anything but ScheduleRejected propagates out of here: KeyError was the defect.
    built, refused = _outcomes(cands)

    looped = [c for c in cands if _step(c, rowsum) < 256]
    assert looped and len(looped) < len(cands), "the case needs both kinds of tiling"
    # Exactly the tilings that loop the row sum are refused, and for this reason.
    assert [c for c, _ in refused] == looped
    assert {text for _, text in refused} == {
        f"{exp.name} is computed per iteration of GraphNode_{rowsum.name} but also read "
        f"by {div.name} outside it"
    }
    # The others hold the whole row in one tile: no loop contains the exp.
    assert len(built) == len(cands) - len(looped)
    for _, tiled in built:
        for tier, owner in graph_tiers(tiled):
            assert owner is None or exp.name not in {n.name for n in tier.nodes}


def test_merged_reductions_with_different_steps_are_a_typed_refusal(monkeypatch):
    ir = _identity_graph(column_weighted, dict(M=128, K=4096, N=256))
    matmul, colsum = _one(ir, Matmul), _one(ir, ReduceSum)
    cands = search_tiling(ir, topk=1000)
    assert cands

    # The defect, with the guard taken out: every candidate builds a loop whose trip count
    # is not the number of tiles its inner tensors are cut into.
    with monkeypatch.context() as patch:
        patch.setattr(
            tiling, "_refuse_loops_that_miscount_their_tiles", lambda main: None, raising=False
        )
        for cand in cands:
            assert _step(cand, matmul) != _step(cand, colsum)
            (loop,) = [tier for tier, owner in graph_tiers(apply_tiling(cand)) if owner]
            assert loop.extent == 4096 // _step(cand, colsum)
            cut = {
                it.shape[d] // it.tile_size[d]
                for it in loop.tmap.values()
                for d in range(len(it.shape))
                if it.shape[d] // it.tile_size[d] > 1
            }
            assert cut == {4096 // _step(cand, matmul)}

    # With it: a typed refusal that names the loop, both counts and the tensor.
    built, refused = _outcomes(cands)
    assert not built and len(refused) == len(cands)
    for cand, text in refused:
        trips, tiles = 4096 // _step(cand, colsum), 4096 // _step(cand, matmul)
        assert re.fullmatch(
            rf"GraphNode_{colsum.name} runs {trips} iterations but dim \d of \S+ is cut "
            rf"into {tiles} tiles \({_step(cand, matmul)} of 4096\)",
            text,
        ), text


def test_grid_dim_without_a_launch_axis_is_a_typed_refusal():
    """A multi-output graph with every reduction un-looped: the outputs next to the
    canonical one are not its ancestors, so no launch axis is derived for their dims. The
    tiler has always refused such a tiling, with ``ValueError``."""
    from tests._softmax import exps, raw_statistics_graph
    from tests.graph.apply_tiling._test_utils import build_tiled_candidate

    ir = raw_statistics_graph(32, 64, 256, 64)
    for exp in exps(ir):
        exp.stabilize = None  # the refusal is the tiler's own, softmax or not
    rstep = {"mma_0": {"v3": 64}, "mul_0": {}, "exp_0": {}, "rowsum_0": {"v2": 256},
             "mma_1": {"v3": 256}, "reduce_max_2_exp_0": {"v2": 256}}
    cand = build_tiled_candidate(ir, [1, 32, 32], rstep)
    with pytest.raises(
        ScheduleRejected,
        match=r"^Expected exactly one launch-axis mapping for iter dim 0, got \[\]\.$",
    ):
        apply_tiling(cand)


def test_tile_that_misses_a_unit_dim_is_a_typed_refusal():
    ir = _identity_graph(matmul_of_rowsum, dict(M=4096, K=64, N=4096))
    matmul = _one(ir, Matmul)
    assert tuple(int(s) for s in matmul.get_shape()) == (4096, 1)
    cands = search_tiling(ir, topk=1000)
    # Both spellings of the old failure are among them: the matmul looped over K
    # (KeyError: 1 at the output's offsets) and not (the assert on iter_repeat_factors).
    assert {_step(c, matmul) < 64 for c in cands} == {True, False}

    # Anything but ScheduleRejected propagates out of here.
    built, refused = _outcomes(cands)
    assert not built and len(refused) == len(cands)
    for cand, text in refused:
        assert len(cand.output_tile) == 1  # one extent for a tensor of two dims
        assert text == (
            f"{matmul.name}: tile {list(cand.output_tile)} does not cover the 2 dims of its "
            "tensor (4096, 1); a unit dim that is not a batch dim has no tile"
        )


def test_float32_siblings_are_refused_rather_than_miscounted():
    """A limit that holds today, pinned so that its fix has a case that flips.

    The policy picks the smallest step of each reduction on its own (``TCPolicy.
    _assign_reduce_step``: a tensor-core matmul by the byte width of its operand, any other
    reduction by its read pattern) and then enlarges the steps of a group in lockstep. Two
    reductions of one operand that start apart stay apart: with float32 operands the value
    matmul steps half of what the row sum steps, for every candidate. The tiler merges
    both into one loop, which used to run the row sum's trip count over the matmul's
    tiles and so summed half the row. The loop-extent guard refuses all of them: a
    float32 flash-form softmax has no tiling, where float16 has all of them.
    """
    shapes = dict(R=64, C=4096, D=64)

    ir = _identity_graph(flash_rows, shapes, cl.f32)
    matmul, rowsum = _one(ir, Matmul), _one(ir, ReduceSum)
    cands = search_tiling(ir, topk=1000)
    assert cands and all(2 * _step(c, matmul) == _step(c, rowsum) for c in cands)
    built, refused = _outcomes(cands)
    for cand, text in refused:
        assert re.fullmatch(
            rf"GraphNode_\S+ runs {4096 // _step(cand, rowsum)} iterations but dim \d of "
            rf"\S+ is cut into {4096 // _step(cand, matmul)} tiles "
            rf"\({_step(cand, matmul)} of 4096\)",
            text,
        ), text
    # THE LIMIT. Once the policy steps the two together, these build.
    assert not built and len(refused) == len(cands)

    ir = _identity_graph(flash_rows, shapes, cl.f16)
    matmul, rowsum = _one(ir, Matmul), _one(ir, ReduceSum)
    cands = search_tiling(ir, topk=1000)
    assert cands and all(_step(c, matmul) == _step(c, rowsum) for c in cands)
    built, refused = _outcomes(cands)
    assert len(built) == len(cands) and not refused


def test_a_consistent_loop_is_not_refused():
    """The guard looks at the trip count only: the flash-form loop (row sum and value
    matmul stepped together over the keys) passes it untouched."""
    from tests._graphs import mha_bhsd_graph
    from tests.graph.apply_tiling._test_utils import build_tiled_candidate

    rstep = {"mma_0": {"v3": 64}, "mul_0": {}, "sub_0": {}, "exp_0": {}, "div_0": {},
             "mma_1": {"v3": 32}, "rowsum_0": {"v2": 32}}
    tiled = apply_tiling(build_tiled_candidate(mha_bhsd_graph(1, 32, 4096, 4096, 64),
                                               [1, 64, 64], rstep))
    (loop,) = [tier for tier, owner in graph_tiers(tiled) if owner is not None]
    assert loop.extent == 4096 // 32 and len(loop.accum_nodes) == 2
