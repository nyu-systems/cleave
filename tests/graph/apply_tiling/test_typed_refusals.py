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
  over 32 tiles of 128. The policy proposed nothing else for two reductions of one operand
  whose first steps differ; it starts them together now, and the refusal is for step maps
  that come from elsewhere.

* a grid dim that no single launch axis indexes. The tiler always refused it on purpose,
  but with ``ValueError``.
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
from cleave.arch import get_arch
from cleave.core.errors import ScheduleRejected
from cleave.frontend import trace
from cleave.ir.graph import OutputNode
from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops import Div, Exp, Matmul, ReduceSum
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet
from cleave.schedule import apply_tiling, search_tiling
from cleave.schedule import tiling
from cleave.schedule.policy.default import TCPolicy


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


def _stepping_half(ir, cand, node):
    """``cand`` with ``node`` stepping half as far as it does: what the policy proposed
    for a pair of sibling reductions while it picked each one's first step on its own."""
    policy = TCPolicy([n for n in ir.nodes if isinstance(n, OutputNode)], get_arch("cuda"),
                      flat_graph=ir)
    rstep = {n: {k: int(step) // 2 for k, step in v.items()} if n.name == node.name else v
             for n, v in cand.rstep_map.items()}
    apart = policy.compute_tile_dict(list(cand.output_tile), rstep)
    assert apart.valid
    apart.n_threads = cand.n_threads
    return apart


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
    together = search_tiling(ir, topk=1000)
    assert together and all(_step(c, matmul) == _step(c, colsum) for c in together)
    # The step maps the policy used to propose for this graph: the matmul at half the
    # step of the column sum, in every candidate.
    cands = [_stepping_half(ir, cand, matmul) for cand in together]

    # The defect, with the guard taken out: every candidate builds a loop whose trip count
    # is not the number of tiles its inner tensors are cut into.
    with monkeypatch.context() as patch:
        patch.setattr(
            tiling, "_refuse_loops_that_miscount_their_tiles", lambda main: None, raising=False
        )
        for cand in cands:
            assert 2 * _step(cand, matmul) == _step(cand, colsum)
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

    # The policy's own candidates step the two together: one loop that counts its tiles.
    built, refused = _outcomes(together)
    assert len(built) == len(together) and not refused
    for cand, tiled in built:
        (loop,) = [tier for tier, owner in graph_tiers(tiled) if owner]
        assert loop.extent == 4096 // _step(cand, matmul)
        assert {n.name for n in loop.accum_nodes} == {matmul.name, colsum.name}


def gram(x: cl.Tensor["M", "K"]):
    """x @ x.mT: one input read as both operands, its rows on both axes of the output."""
    return x @ x.mT


def test_grid_dim_without_one_launch_axis_is_a_typed_refusal():
    """A dim that two launch axes index: the rows of ``x`` are the rows of ``x @ x.mT`` and
    its columns, and a kernel has one tile of each input. The tiler has always refused a
    tiling that walks such a dim, but with ``ValueError``. One that holds the dim whole
    while the grid walks one of the two axes was built -- with the tile of whichever
    operand was propagated last -- and died in TileLang ("T.gemm M shape check failed:
    M_A = 128, M_C = 64")."""
    ir = _identity_graph(gram, dict(M=128, K=256))
    cands = search_tiling(ir, topk=1000)
    built, refused = _outcomes(cands)
    # Only a block that holds every row and every column serves both readers.
    assert built and all(list(cand.output_tile) == [128, 128] for cand, _ in built)
    reasons = {}
    for cand, text in refused:
        reasons.setdefault(text, set()).add(tuple(cand.output_tile))
    assert reasons == {
        "Expected exactly one launch-axis mapping for iter dim 0, got [0, 1].": {
            (64, 32), (64, 64), (128, 32), (128, 64)},
        "Dim 0 is held whole but launch axes [0, 1] index it and the grid walks [0]: one "
        "tile cannot be a slice and the whole.": {(64, 128)},
    }


def test_outputs_next_to_the_canonical_one_have_launch_axes():
    """A multi-output graph with every reduction un-looped. The outputs next to the
    canonical one are not its ancestors, so no launch axis used to be derived for their
    dims and such a tiling was refused ("... for iter dim 0, got []"). They are launched
    on the canonical output's axes, as the outputs of a loop always were."""
    from tests._softmax import exps, raw_statistics_graph
    from tests.graph.apply_tiling._test_utils import build_tiled_candidate

    ir = raw_statistics_graph(32, 64, 256, 64)
    for exp in exps(ir):
        exp.stabilize = None  # the tiler's own business, softmax or not
    rstep = {"mma_0": {"v3": 64}, "mul_0": {}, "exp_0": {}, "rowsum_0": {"v2": 256},
             "mma_1": {"v3": 256}, "reduce_max_2_exp_0": {"v2": 256}}
    tiled = apply_tiling(build_tiled_candidate(ir, [1, 32, 32], rstep))
    assert not [owner for _, owner in graph_tiers(tiled) if owner is not None]
    assert tiled.extent == (32, 2, 2)  # heads, rows, value columns
    offsets = {node.name: it.offset("i0", "i1", "i2") for node, it in tiled.tmap.items()}
    for name in ("rowsum_0", "OutputNode_0", "reduce_max_2_exp_0", "OutputNode_2"):
        assert offsets[name] == ("1 * i0", "32 * i1"), name  # the head and the row block
    assert offsets["OutputNode_1"] == ("1 * i0", "32 * i1", "32 * i2")
    assert tiled.stored_layouts == {
        "OutputNode_0": ((0, 1), 0), "OutputNode_1": ((0, 1, 2), 0), "OutputNode_2": ((0, 1), 0)}


def test_tile_that_misses_a_unit_dim_is_a_typed_refusal():
    ir = _identity_graph(matmul_of_rowsum, dict(M=4096, K=64, N=4096))
    matmul = _one(ir, Matmul)
    assert tuple(int(s) for s in matmul.get_shape()) == (4096, 1)
    cands = search_tiling(ir, topk=1000)
    # The old failure had two spellings: the assert on iter_repeat_factors for a matmul
    # that holds K whole, and KeyError: 1 at the output's offsets for one looped over K
    # (which the policy may or may not propose).
    assert any(_step(c, matmul) == 64 for c in cands)

    # Anything but ScheduleRejected propagates out of here.
    built, refused = _outcomes(cands)
    assert not built and len(refused) == len(cands)
    for cand, text in refused:
        assert len(cand.output_tile) == 1  # one extent for a tensor of two dims
        assert text == (
            f"{matmul.name}: tile {list(cand.output_tile)} does not cover the 2 dims of its "
            "tensor (4096, 1); a unit dim that is not a batch dim has no tile"
        )


def batched_matmul_of_column(x: cl.Tensor["B", "M", "K"], v: cl.Tensor["ONE", "K", "ONE2"]):
    """x @ column for every batch entry: the output is (B, M, 1)."""
    return x @ v


def test_tile_that_misses_a_unit_dim_next_to_batch_dims_is_a_typed_refusal():
    """The same missing extent with a batch dim in front: the policy tiles ``(B, M, 1)`` as
    ``[b, m]``, and with the batch dim put back the tile has the tensor's rank -- ``[1, b,
    m]``, the extents on the wrong dims. Every candidate was refused, by the launch-axis
    plan and for a reason that says nothing ("for iter dim 1, got []")."""
    ir = _identity_graph(batched_matmul_of_column, dict(B=4, M=64, K=128, ONE=1, ONE2=1), cl.f32)
    matmul = _one(ir, Matmul)
    assert tuple(int(s) for s in matmul.get_shape()) == (4, 64, 1)
    cands = search_tiling(ir, topk=1000)
    assert cands

    built, refused = _outcomes(cands)
    assert not built and len(refused) == len(cands)
    for cand, text in refused:
        assert len(cand.output_tile) == 2  # two extents for a tensor of three dims
        assert text == (
            f"{matmul.name}: tile {list(cand.output_tile)} does not cover the 3 dims of its "
            "tensor (4, 64, 1); a unit dim that is not a batch dim has no tile"
        )


def test_matmul_operand_with_a_unit_batch_dim_is_tiled():
    """The neighbour that works, so that the refusals do not grow over it: an operand
    broadcast over the batch through a unit batch dim builds for every candidate."""
    def broadcast_b(x: cl.Tensor["B", "M", "K"], w: cl.Tensor["ONE", "K", "N"]):
        return x @ cl.exp(w)

    ir = _identity_graph(broadcast_b, dict(B=4, M=64, K=128, N=32, ONE=1), cl.f32)
    cands = search_tiling(ir, topk=1000)
    built, refused = _outcomes(cands)
    assert built and not refused


@pytest.mark.parametrize("dtype, first", [(cl.f32, 32), (cl.f16, 32)], ids=["f32", "f16"])
def test_siblings_over_one_axis_are_stepped_together(dtype, first):
    """The flash-form softmax: the row sum and the value matmul reduce the same axis of
    the same ``exp`` and end up in one loop.

    The policy picks the smallest step of each reduction on its own (``TCPolicy.
    _assign_reduce_step``: a tensor-core matmul by the byte width of its operand, any other
    reduction by its read pattern) and then enlarges the steps of a group in lockstep, so
    two that start apart stay apart. With float32 operands the value matmul started at 16
    and the row sum at 32, for every candidate; the loop-extent guard refused all of them
    (the kernels they used to build summed half the row), and a float32 flash-form softmax
    had no tiling where float16 had all of them. Siblings over one axis start together.
    """
    shapes = dict(R=64, C=4096, D=64)

    ir = _identity_graph(flash_rows, shapes, dtype)
    matmul, rowsum = _one(ir, Matmul), _one(ir, ReduceSum)
    policy = TCPolicy([n for n in ir.nodes if isinstance(n, OutputNode)], get_arch("cuda"),
                      flat_graph=ir)
    by_name = {n.name: n for n in policy.ordered_nodes}
    alone = {n.name: policy._assign_reduce_step(n) for n in (matmul, rowsum)}
    starts = policy._start_siblings_together(
        {n: policy._assign_reduce_step(n) for n in policy.ordered_nodes})
    own = {name: int(next(iter(step.values()))) for name, step in alone.items()}
    # float32: the matmul's own first step is half the row sum's; float16: they agree
    assert own == {matmul.name: 16 if dtype is cl.f32 else 32, rowsum.name: 32}
    # neither starts below its own first step, and both start at the larger one
    assert {int(next(iter(starts[by_name[n.name]].values()))) for n in (matmul, rowsum)} == {first}

    cands = search_tiling(ir, topk=1000)
    assert cands and all(_step(c, matmul) == _step(c, rowsum) for c in cands)
    assert min(_step(c, matmul) for c in cands) == first
    built, refused = _outcomes(cands)
    assert len(built) == len(cands) and not refused
    for cand, tiled in built:
        (loop,) = [tier for tier, owner in graph_tiers(tiled) if owner]
        assert loop.extent == 4096 // _step(cand, matmul)
        assert {n.name for n in loop.accum_nodes} == {matmul.name, rowsum.name}


def test_reductions_that_share_no_axis_keep_their_own_first_step():
    """What starting together must leave alone: LoRA's ``X @ W`` and ``X @ A`` read one
    input but are not siblings of one group, and ``(X @ A) @ B`` reduces another axis."""
    from tests._graphs import lora_graph, matmul_matmul_graph

    for graph in (lora_graph(16, 4096, 4096, 16), matmul_matmul_graph(4096, 64, 64)):
        policy = TCPolicy([n for n in graph.nodes if isinstance(n, OutputNode)],
                          get_arch("cuda"), flat_graph=graph)
        alone = {n: policy._assign_reduce_step(n) for n in policy.ordered_nodes}
        assert policy._start_siblings_together(alone) == alone


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
