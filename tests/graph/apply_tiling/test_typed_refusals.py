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

One more tiling was *built*, and is refused now:

* an output next to the canonical one whose axis is not the canonical axis it is launched
  on. Sibling outputs are paired with the canonical output's axes by position and extent;
  with equal extents the column sums of ``w`` were stored by the row block of ``x @ w``,
  and ``exp(x) * 2`` was cut along the columns of ``exp(x) @ b`` while ``exp(x)`` was held
  whole. The kernels lowered and wrote wrong values.

And one graph had no tiling and no reason:

* outputs whose axes do not pair with the canonical output's at all (the extents differ).
  The policy raised ``NotImplementedError`` while it rated its first tile, which ended the
  search of the whole graph; only the pipeline's schedule stage turned that into a
  refusal, with the name of the exception for a reason. The policy refuses the graph
  itself now, and says which output and which axis.
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
    # (A tile that is not square reads ``x`` with one tile of rows for one operand and
    # another for the other: the policy does not propose it, DefaultPolicy.
    # _reads_placeholder_unevenly. Those were refused here as (64, 32), (128, 32), (128, 64)
    # for the launch axis and (64, 128) for the dim held whole.)
    assert reasons == {
        "Expected exactly one launch-axis mapping for iter dim 0, got [0, 1].": {(64, 64)},
    }


def _raw_statistics_graph(BH, S_Q, S_KV, D):
    """Three outputs of one launch: the unnormalized statistics of a softmax and a max.

        scores = Q K^T / sqrt(D)
        exp    = Exp(scores)
        l      = ReduceSum(exp)       ->  OutputNode_0  (BH, S_Q)      float16
        o      = exp @ V              ->  OutputNode_1  (BH, S_Q, D)   float16
        m      = ReduceMax(scores)    ->  OutputNode_2  (BH, S_Q)      float32
    """
    from cleave.ir.ops import Mul, ReduceMax
    from tests._graphs import ACCUM_DTYPE, DTYPE, OUT_DTYPE, _connect, _finalize

    f32 = dict(dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE)
    scores = (BH, S_Q, S_KV)
    mma_0 = Matmul((BH, S_Q, S_KV, D), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_0",
                   transpose_B=True, warp_policy="rows")
    mul_0 = Mul(scores, const=1.0 / (D**0.5), name="mul_0", **f32)
    exp_0 = Exp(scores, name="exp_0", **f32)
    rowsum_0 = ReduceSum(scores, name="rowsum_0", **f32)
    mma_1 = Matmul((BH, S_Q, D, S_KV), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_1",
                   warp_policy="rows")
    row_max = ReduceMax(scores, axis=-1, name="reduce_max_2_exp_0", **f32)
    _connect(mma_0, mul_0, 0, 0)
    _connect(mul_0, exp_0, 0, 0)
    _connect(exp_0, rowsum_0, 0, 0)
    _connect(exp_0, mma_1, 0, 0)
    _connect(mul_0, row_max, 0, 0)
    for i, (node, dtype) in enumerate(
        ((rowsum_0, OUT_DTYPE), (mma_1, OUT_DTYPE), (row_max, "float32"))
    ):
        OutputNode(node, dtype=dtype).name = f"OutputNode_{i}"
    return _finalize(mma_0, mul_0, exp_0, rowsum_0, mma_1, row_max)


def test_outputs_next_to_the_canonical_one_have_launch_axes():
    """A multi-output graph with every reduction un-looped. The outputs next to the
    canonical one are not its ancestors, so no launch axis used to be derived for their
    dims and such a tiling was refused ("... for iter dim 0, got []"). They are launched
    on the canonical output's axes, as the outputs of a loop always were."""
    from tests.graph.apply_tiling._test_utils import build_tiled_candidate

    ir = _raw_statistics_graph(32, 64, 256, 64)  # the tiler's own business, softmax or not
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


def reduced_operands(x: cl.Tensor["B", "M", "K"], y: cl.Tensor["B", "K", "N"]):
    """sum(x, 0) @ sum(y, 0): both operands lost the batch axis, the output has one entry."""
    return cl.sum(x, 0, keepdim=True) @ cl.sum(y, 0, keepdim=True)


def reduced_operand_and_input(x: cl.Tensor["B", "M", "K"], w: cl.Tensor["ONE", "K", "N"]):
    """sum(x, 0) @ w: the reduced operand next to an input."""
    return cl.sum(x, 0, keepdim=True) @ w


def input_and_reduced_operand(x: cl.Tensor["ONE", "M", "K"], y: cl.Tensor["B", "K", "N"]):
    return x @ cl.sum(y, 0, keepdim=True)


def chain_from_reduced_operand_and_input(x: cl.Tensor["B", "M", "K"],
                                         w: cl.Tensor["ONE", "K", "N"]):
    return cl.exp(cl.sum(x, 0, keepdim=True)) @ w


REDUCED_OPERANDS = [reduced_operands, reduced_operand_and_input, input_and_reduced_operand,
                    chain_from_reduced_operand_and_input]


@pytest.mark.parametrize("fn", REDUCED_OPERANDS, ids=[fn.__name__ for fn in REDUCED_OPERANDS])
def test_matmul_operand_that_lost_a_unit_batch_axis_is_tiled(fn):
    """A matmul that reads a sum over a batch axis as an operand. Every tiling builds
    (but the ones that walk the batch axis one entry at a time, which have no kernel).
    CPU: they lower. Next to an *input* they did not: the sum is a fragment laid out from
    the tensor it reduces, the gemm wants its operand in the tensor-core layout, and
    TileLang found "no available layout" for 22 of 24 tilings. The operand is staged
    through shared memory, as both are when both are computed."""
    from cleave.runtime.compile import build_candidate_code
    from tests._lowering import lower_to_cuda

    ir = _identity_graph(fn, dict(B=4, M=128, K=256, N=128, ONE=1)
                         if fn is not reduced_operands else dict(B=4, M=128, K=256, N=128))
    cands = search_tiling(ir, topk=1000)
    built, refused = _outcomes(cands)
    assert built and not refused
    lowered = 0
    for cand, _ in built:
        try:
            code = build_candidate_code(cand)
        except ScheduleRejected as refusal:
            assert "is handed one element of the axis it reduces per step (4 in all)" in str(refusal)
            continue
        # a reduced operand never reaches the gemm as the fragment the sum left it in:
        # neither operand is read from a fragment (the scopes in the macro's name)
        (scopes,) = re.findall(r"^    ukernel_mma_([a-z]{3})\w*\(", code, flags=re.M)
        assert "r" not in scopes[:2] and scopes[2] == "r", scopes
        if lowered < 3 and list(cand.output_tile)[-2:] == [64, 64] and cand.n_threads == 128:
            assert "main_kernel" in lower_to_cuda(code)
            lowered += 1
    assert lowered


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

    CPU only, and on purpose: what is pinned is that the tilings exist and that their loop
    counts its tiles. On a GPU the float32 kernels run (12 of 14; 2 exceed shared memory)
    at 1.04e-3 to 1.12e-3 of the row maximum, which is what every float32 tensor-core
    matmul measures here (a plain float32 matmul: 1.02e-3, on the tree without this
    change too) and 50 times the 2e-5 ceiling of a float32 output: no float32 matmul
    program can be validated under the current rule, so there is no tolerance to declare
    for these and no GPU test.
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


# ------------------------------------------------- outputs next to the canonical one


def scaled_next_to_product(x: cl.Tensor["M", "K"], b: cl.Tensor["K", "N"]):
    """exp(x) @ b and exp(x) * 2: the second output keeps the axis the first contracts."""
    e = cl.exp(x)
    return e @ b, e * 2.0


def column_sums_next_to_product(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    """x @ w and the column sums of w: the second output runs along the columns."""
    return x @ w, cl.sum(w, 0, keepdim=True)


def row_sums_next_to_product(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    """x @ w and the row sums of x: the second output runs along the rows, as launched."""
    return x @ w, cl.sum(x, -1, keepdim=True)


SIBLING = re.compile(
    r"(OutputNod\w+) is launched on the axes of the canonical output by extent, but dim (\d) "
    r"of (\w+) is indexed by launch axes (\[[\d, ]*\]) for the canonical output and "
    r"(\[[\d, ]*\]) for \1: the grid walks (\[[\d, ]*\]), where the two are not the same "
    r"axis\."
)


def _grid(cand, shape):
    """Blocks per axis of the canonical output for ``cand``."""
    return tuple(extent // tile for extent, tile in zip(shape, cand.output_tile))


def test_a_sibling_output_on_another_axis_is_refused_where_the_grid_walks_it():
    """``exp(x) * 2`` next to ``exp(x) @ b`` with as many columns in ``b`` as in ``x``. The
    second output is ``(M, K)``, and its ``K`` was paired with the ``N`` of the product
    because the extents agree. ``exp(x)`` is the product's operand and holds ``K`` whole;
    a grid that walks the columns of the product cut the second output into column blocks
    and filled each from the *first* columns of ``exp(x)`` (0.7 of the row max off, in a
    kernel that lowered). Such a tiling was refused before siblings were launched at all
    ("... for iter dim 1, got []"); it is refused again, for the reason. While the grid
    has one block of columns the two axes cannot be told apart and the kernel is right."""
    ir = _identity_graph(scaled_next_to_product, dict(M=128, K=64, N=64))
    matmul, exp = _one(ir, Matmul), _one(ir, Exp)
    cands = [c for c in search_tiling(ir, topk=1000) if _step(c, matmul) == 64]  # not looped
    assert {tuple(c.output_tile) for c in cands} == {(64, 32), (64, 64), (128, 32), (128, 64)}
    built, refused = _outcomes(cands)
    assert {tuple(c.output_tile) for c, _ in built} == {(64, 64), (128, 64)}
    assert {tuple(c.output_tile) for c, _ in refused} == {(64, 32), (128, 32)}
    for cand, text in refused:
        sibling, dim, node, canonical, own, walked = SIBLING.fullmatch(text).groups()
        assert (dim, node, canonical, own, walked) == ("1", exp.name, "[]", "[1]", "[1]"), text
    for cand, tiled in built:
        # one tile of exp(x), read whole by both outputs
        tiles = {n.name: tuple(int(t) for t in it.tile_size) for n, it in tiled.tmap.items()}
        (scaled,) = [n for n in ir.nodes if n.name.startswith("mul_")]
        assert tiles[exp.name] == tiles[scaled.name] == (cand.output_tile[0], 64)


def test_a_sibling_output_of_another_axis_is_built_in_one_block_only():
    """The column sums of ``w`` next to ``x @ w``, square: ``(N,)`` was paired with the
    rows of the product. ``w`` is sliced by the column block, so the sums of column block
    ``i1`` were stored at row block ``i0`` -- several blocks wrote one slice, each its own
    values. Looped or not; the un-looped ones were refused until siblings were launched,
    the looped ones were always built. Only the single block is right."""
    ir = _identity_graph(column_sums_next_to_product, dict(M=128, K=256, N=128))
    (w,) = [n for n in ir.nodes if n.is_placeholder() and tuple(n.get_shape(0)) == (256, 128)]
    cands = search_tiling(ir, topk=1000)
    # (The grids of one block of rows or one of columns read ``w`` with two different
    # tiles, one per reader: the policy does not propose them,
    # DefaultPolicy._reads_placeholder_unevenly. They were refused here as well.)
    assert {_grid(c, (128, 128)) for c in cands} == {(1, 1), (2, 2)}
    built, refused = _outcomes(cands)
    assert built and {_grid(c, (128, 128)) for c, _ in built} == {(1, 1)}
    assert {_grid(c, (128, 128)) for c, _ in refused} == {
        _grid(c, (128, 128)) for c in cands} - {(1, 1)}
    for cand, text in refused:
        sibling, dim, node, canonical, own, walked = SIBLING.fullmatch(text).groups()
        assert (dim, node, canonical, own) == ("1", w.name, "[1]", "[0]"), text
        rows, cols = _grid(cand, (128, 128))
        assert walked == str([axis for axis, blocks in enumerate((rows, cols)) if blocks > 1])


def test_a_sibling_output_on_its_own_axis_is_launched_as_before():
    """What the refusal leaves alone: the row sums of ``x`` next to ``x @ w`` run along the
    rows, the axis they are launched on. Every tiling is built, square or not."""
    for sizes in (dict(M=128, K=256, N=128), dict(M=128, K=256, N=64)):
        ir = _identity_graph(row_sums_next_to_product, sizes)
        built, refused = _outcomes(search_tiling(ir, topk=1000))
        assert built and not refused


UNPAIRED = [
    # the column sums of w run along N, and would be launched on the rows of x @ w
    (column_sums_next_to_product, dict(M=128, K=256, N=64),
     "reduction_0_3 (64,) is an output next to mma_2 (128, 64), whose launch grid serves "
     "every output of the kernel: its axis 0 would be launched on axis 0 of the grid, "
     "which has 128 elements where it has 64"),
    (column_sums_next_to_product, dict(M=64, K=256, N=128),
     "reduction_0_3 (128,) is an output next to mma_2 (64, 128), whose launch grid serves "
     "every output of the kernel: its axis 0 would be launched on axis 0 of the grid, "
     "which has 64 elements where it has 128"),
    # exp(x) * 2 keeps the K columns the product contracts
    (scaled_next_to_product, dict(M=128, K=64, N=32),
     "mul_4 (128, 64) is an output next to mma_3 (128, 32), whose launch grid serves "
     "every output of the kernel: its axis 1 would be launched on axis 1 of the grid, "
     "which has 32 elements where it has 64"),
]


@pytest.mark.parametrize("fn, sizes, reason", UNPAIRED,
                         ids=["columns_shorter", "columns_longer", "contracted_axis_kept"])
def test_an_output_the_launch_grid_cannot_serve_is_a_typed_refusal(fn, sizes, reason):
    """``x @ w`` next to ``sum(w, 0)`` with ``M != N``, and ``exp(x) @ b`` next to
    ``exp(x) * 2`` with ``K != N``. Every output of a kernel is tiled on the launch grid of
    the canonical one, its axes paired with that output's by position and extent. Where
    the extents differ there is no pairing, and the policy raised ``NotImplementedError``
    ("Same-launch multi-output currently requires sibling outputs to drop only trailing
    non-batch axes.") from the rating of its first tile: the search of the whole graph
    ended with an exception no caller of ``search_tiling`` expects. It is the policy's
    own refusal of the graph now, and names the output and the axis."""
    ir = _identity_graph(fn, sizes)
    with pytest.raises(ScheduleRejected) as refusal:
        search_tiling(ir, topk=1000)
    assert str(refusal.value) == reason


@pytest.mark.parametrize("fn, sizes, reason", UNPAIRED,
                         ids=["columns_shorter", "columns_longer", "contracted_axis_kept"])
def test_the_pipeline_reports_an_unserved_output_with_its_reason(fn, sizes, reason):
    """The schedule stage records the policy's refusal as it is. It used to be the one
    place that caught the exception, and the reason it gave was the exception's name:
    "ScheduleRejected: NotImplementedError: Same-launch multi-output currently ..."."""
    from cleave import pipeline as P

    plans = P.schedule(GraphSet.identity(trace(fn, sizes, cl.f16)), P.ScheduleConfig(topk=8))
    assert not list(plans)
    # refused once as it is and once more under forced fusion, for the same reason
    assert plans.refused and {str(r.error) for r in plans.refused} == {reason}
    assert all(type(r.error) is ScheduleRejected for r in plans.refused)


def test_outputs_that_pair_with_the_grid_are_searched_as_before():
    """What the refusal leaves alone: the same programs where the extents agree (the
    pairing is then checked per tiling, above), and outputs on the axes they are launched
    on at any extents."""
    for fn, sizes in ((column_sums_next_to_product, dict(M=128, K=256, N=128)),
                      (scaled_next_to_product, dict(M=128, K=64, N=64)),
                      (row_sums_next_to_product, dict(M=128, K=256, N=64)),
                      (row_sums_next_to_product, dict(M=64, K=256, N=128))):
        assert search_tiling(_identity_graph(fn, sizes), topk=1000)


def test_the_canonical_view_of_a_shared_ancestor_is_kept_and_the_other_reported():
    """``launch_axis_dependencies``: what the canonical output says of a node stands, and
    each dim a sibling would have indexed otherwise is reported with both views."""
    from cleave.ir.analysis import infer_output_axis_dependencies, launch_axis_dependencies

    ir = _identity_graph(scaled_next_to_product, dict(M=128, K=64, N=64))
    exp = _one(ir, Exp)
    outputs = [n for n in ir.nodes if isinstance(n, OutputNode)]
    canonical = tiling._select_canonical_output_node(outputs)
    siblings = tiling._sibling_launch_axes(canonical, outputs)
    ((sibling, axis_map),) = siblings
    assert axis_map == [0, 1]  # paired by position and extent
    alone = infer_output_axis_dependencies(canonical)
    deps, disagreements = launch_axis_dependencies(canonical, siblings)
    assert alone[exp.name] == deps[exp.name] == [{0}, set()]
    assert infer_output_axis_dependencies(canonical, siblings) == deps
    (x,) = [n for n in ir.nodes if n.is_placeholder() and tuple(n.get_shape(0)) == (128, 64)]
    # the contracted axis, of exp(x) and of x under it: no launch axis for the product,
    # the columns for the second output
    assert sorted(tuple(d) for d in disagreements) == sorted(
        (sibling.name, name, 1, frozenset(), frozenset({1})) for name in (exp.name, x.name))
    # the sibling's own ops are seen from the sibling only
    (scaled,) = [n for n in ir.nodes if n.name.startswith("mul_")]
    assert scaled.name not in alone and deps[scaled.name] == [{0}, {1}]

    # a sibling on its own axis disagrees with nothing
    ir = _identity_graph(row_sums_next_to_product, dict(M=128, K=256, N=128))
    outputs = [n for n in ir.nodes if isinstance(n, OutputNode)]
    canonical = tiling._select_canonical_output_node(outputs)
    assert launch_axis_dependencies(
        canonical, tiling._sibling_launch_axes(canonical, outputs))[1] == []


# --------------------------------------------- tensor-core matmuls over a short axis


def plain_matmul(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    return x @ w


@pytest.mark.parametrize("K", [8, 24, 40])
def test_a_tensor_core_matmul_over_an_axis_no_multiple_of_16_divides_is_a_typed_refusal(K):
    """``TCPolicy._assign_reduce_step`` started such an axis at 16 (or at the transaction
    size) "for the case of padding": a step that does not divide the extent and is none of
    the axis's step candidates. The walk over the candidates then died on it, ``ValueError:
    16 is not in list``, for the whole graph. With 16-bit operands there is no kernel to
    find: TileLang's gemm asserts on a K block of 24 or 40, and the kernel it emits for a
    block of 8 is wrong (measured on a GPU: every entry off, by the row maximum). The
    graph is refused, with the reason."""
    from tests._graphs import matmul_graph, mha_bhsd_graph

    with pytest.raises(ScheduleRejected) as refusal:
        search_tiling(matmul_graph(128, 128, K), topk=1000)
    assert str(refusal.value) == (
        f"mma_0 contracts {K} float16 elements: a tensor-core matmul of 16-bit operands "
        "steps its reduction axis in multiples of 16, and none divides that extent")
    # attention over that many keys: the value matmul contracts them
    with pytest.raises(ScheduleRejected, match=rf"mma_1 contracts {K} float16 elements"):
        search_tiling(mha_bhsd_graph(1, 32, 64, K, 64), topk=1000)
    # the neighbours: 48 and 80 keys are stepped as ever
    for whole in (48, 80):
        cands = search_tiling(matmul_graph(128, 128, whole), topk=1000)
        assert cands and {_step(c, _one(cands[0].flat_graph, Matmul)) for c in cands} == {whole}


def test_the_pipeline_reports_the_policys_refusal_as_it_is():
    """``pipeline.plans._search`` names whatever the search raises in the refusal of the
    graph ("ValueError: 16 is not in list"). A refusal the policy types itself is passed
    through, with its reason and without the name of an exception in front."""
    from cleave import pipeline as P

    prog = trace(plain_matmul, dict(M=128, K=8, N=128), cl.f16)
    plans = P.schedule(GraphSet.identity(prog), P.ScheduleConfig(topk=8))
    assert not list(plans) and plans.refused
    for refusal in plans.refused:
        assert str(refusal.reason) == (
            "ScheduleRejected: mma_2 contracts 8 float16 elements: a tensor-core matmul of "
            "16-bit operands steps its reduction axis in multiples of 16, and none divides "
            "that extent")


@pytest.mark.parametrize("K", [8, 24, 40])
def test_a_float32_matmul_over_such_an_axis_starts_at_a_step_it_has(K):
    """float32 operands are contracted in K blocks of 8, so these extents have kernels;
    they had no tiling for the same ``ValueError``. The first step is one of the axis's
    own candidates, picked like any other reduction's: a multiple of 8, which TileLang
    lowers."""
    from cleave.runtime.compile import build_candidate_code
    from tests._lowering import lower_to_cuda

    ir = _identity_graph(plain_matmul, dict(M=128, K=K, N=128), cl.f32)
    matmul = _one(ir, Matmul)
    policy = TCPolicy([n for n in ir.nodes if isinstance(n, OutputNode)], get_arch("cuda"),
                      flat_graph=ir)
    (node,) = [n for n in policy.ordered_nodes if n.name == matmul.name]
    (name,) = [it.var.name for it in node.raxis]
    assert policy._assign_reduce_step(node)[name] in policy.get_node_reduce_step_candidates(node)[name]
    cands = search_tiling(ir, topk=1000)
    assert cands and all(K % _step(c, matmul) == 0 for c in cands)
    assert all(_step(c, matmul) % 8 == 0 for c in cands)
    built, refused = _outcomes(cands)
    assert len(built) == len(cands) and not refused
    assert "main_kernel" in lower_to_cuda(build_candidate_code(cands[0]))


@pytest.mark.parametrize("K", [4, 12, 20, 100, 257, 1004])
def test_a_float32_matmul_over_an_axis_no_multiple_of_8_divides_is_a_typed_refusal(K):
    """The steps of such an axis are 4 (where it divides) and the extent itself, and
    TileLang lowers neither: a K block that is no multiple of 8 fails its layout check
    (``stride == 4 || stride % 8 == 0``), and a block of 4 is taken for a float64 matmul.
    These graphs were given 12 tilings, each built and each refused by TileLang one by
    one, with no reason on the graph; the float16 sibling was already refused here."""
    from cleave import pipeline as P

    reason = (f"mma_2 contracts {K} float32 elements: a tensor-core matmul of float32 "
              "operands steps its reduction axis in multiples of 8, and none divides that "
              "extent")
    ir = _identity_graph(plain_matmul, dict(M=128, K=K, N=128), cl.f32)
    with pytest.raises(ScheduleRejected) as refusal:
        search_tiling(ir, topk=1000)
    assert str(refusal.value) == reason
    prog = trace(plain_matmul, dict(M=128, K=K, N=128), cl.f32)
    plans = P.schedule(GraphSet.identity(prog), P.ScheduleConfig(topk=8))
    assert not list(plans) and plans.refused
    assert {str(r.reason) for r in plans.refused} == {"ScheduleRejected: " + reason}


# ------------------------------------------ what starting siblings together leaves alone


def _policy_of(ir):
    return TCPolicy([n for n in ir.nodes if isinstance(n, OutputNode)], get_arch("cuda"),
                    flat_graph=ir)


def _first_steps(policy):
    alone = {n: policy._assign_reduce_step(n) for n in policy.ordered_nodes}
    return alone, policy._start_siblings_together(alone)


def sums_over_two_axes(v: cl.Tensor["R", "R"], x: cl.Tensor["R", "C"]):
    """(v @ exp(x)) / sum(exp(x), -1): the matmul reduces the rows of exp(x), the row sum
    its columns -- one group, two axes."""
    e = cl.exp(x)
    return (v @ e) / cl.sum(e, -1, keepdim=True)


def test_group_members_on_different_axes_are_not_started_together():
    """The shared-axis condition. The matmul and the row sum read one ``exp`` and feed one
    divide, so they are one compatible group, and the larger of their first steps is a
    step the other could take. They reduce different axes of it -- each gets a loop of
    its own -- and keep their own first steps."""
    ir = _identity_graph(sums_over_two_axes, dict(R=1024, C=4096), cl.f32)
    matmul, rowsum = _one(ir, Matmul), _one(ir, ReduceSum)
    policy = _policy_of(ir)
    by_name = {n.name: n for n in policy.ordered_nodes}
    assert any({by_name[matmul.name], by_name[rowsum.name]} <= set(group)
               for group in policy.compatible_reduction_groups)
    tokens = policy.reduction_axis_provenance
    assert tokens[by_name[matmul.name]][0] and tokens[by_name[rowsum.name]][0]
    assert not tokens[by_name[matmul.name]][0] & tokens[by_name[rowsum.name]][0]
    alone, starts = _first_steps(policy)
    own = {n.name: int(next(iter(alone[by_name[n.name]].values()))) for n in (matmul, rowsum)}
    assert own == {matmul.name: 16, rowsum.name: 32}  # apart, as in the flash form
    (name,) = [it.var.name for it in matmul.raxis]
    assert 32 in policy.get_node_reduce_step_candidates(by_name[matmul.name])[name]
    assert starts == alone


def normalized_and_plain(x: cl.Tensor["R", "C"], w: cl.Tensor["C", "D"]):
    """softmax(x) @ w + exp(x) @ w: both matmuls contract the columns of exp(x); the first
    one behind the row sum that normalizes them."""
    e = cl.exp(x)
    return (e / cl.sum(e, -1, keepdim=True)) @ w + e @ w


def test_a_sibling_pinned_to_its_whole_axis_is_not_moved_and_moves_nobody():
    """The full-extent exclusion. ``softmax(x) @ w`` contracts the axis its own ancestor,
    the row sum, reduces, so the policy pins it to the whole axis
    (``_full_extent_reduction_axis_ids``). ``exp(x) @ w`` next to it contracts the same
    axis of the same ``exp``, is in its group, and is not behind the sum. The pinned one
    stays at its extent; the other is not pulled up to it and keeps its own first step."""
    C = 4096
    ir = _identity_graph(normalized_and_plain, dict(R=64, C=C, D=64))
    policy = _policy_of(ir)
    matmuls = [n for n in policy.ordered_nodes if isinstance(n, Matmul)]
    (pinned,) = [n for n in matmuls if 0 in policy._full_extent_reduction_axis_ids.get(n, set())]
    (plain,) = [n for n in matmuls if n is not pinned]
    assert any({pinned, plain} <= set(group) for group in policy.compatible_reduction_groups)
    tokens = policy.reduction_axis_provenance
    assert tokens[pinned][0] & tokens[plain][0]  # one axis of one tensor
    alone, starts = _first_steps(policy)
    assert [int(v) for v in alone[pinned].values()] == [C]
    assert [int(v) for v in alone[plain].values()] == [32]
    assert starts == alone


def test_siblings_are_only_started_at_steps_they_can_take():
    """The step-candidate check: whatever the row length, a member starts at one of its
    own step candidates. (Members of one cluster reduce one dim of one tensor, so they
    have one extent and, the pinned ones aside, one list of candidates: the check cannot
    be made to fail from a program, and removing it changes no candidate list.)"""
    for C in (48, 80, 4096):
        ir = _identity_graph(flash_rows, dict(R=64, C=C, D=64), cl.f32)
        policy = _policy_of(ir)
        _, starts = _first_steps(policy)
        for node, step in starts.items():
            for name, value in step.items():
                assert int(value) in policy.get_node_reduce_step_candidates(node)[name], (C, node.name)

