"""A reduction or an output axis whose extent the policy's step does not divide (CPU).

The tensor-core policy used to start a matmul's reduction at 16 or at the tensor-core step
whatever its extent was (``TCPolicy._assign_reduce_step``, "for the case of padding"), and
the walk over the iterative tilings only knows the steps that divide the extent. Nothing
pads the last tile of a reduction, so such a graph has no tiling. The policy used to find
that out with ``ValueError: 32 is not in list`` from a list lookup.

It is a refusal with its reason now, in two places:

* ``TCPolicy._assign_reduce_step`` no longer starts outside the axis's own steps. A matmul
  of 16-bit operands over an extent that no multiple of 16 divides is refused there
  (TileLang's gemm takes such operands in K blocks of a multiple of 16; see
  ``tests/graph/apply_tiling/test_typed_refusals.py``), and a wider operand starts at a
  step it has;
* the walk itself (``expand_iterative_tiles``) refuses a start that is none of the steps it
  knows, instead of dying on the lookup, whoever hands it one.

The attention at S_KV = 1000 is the case that showed it: the value matmul contracts over the
keys, in float16, and 1000 = 8 x 125. It is the same for the exact softmax and for a plain
``exp`` behind a constant shift written by hand, so the exact softmax still has no plan at
such a length. That needs a padded last tile, which is not built here.

Two more of the same kind:

* any reduction over an odd extent above 64 (65, 257, 4099 columns under a row sum or a
  ``cl.softmax``) was offered the powers of two below it, none of which divides it, and
  never the extent itself. Every candidate then died on a bare ``assert`` in
  ``apply_tiling`` -- ``ScheduleRejected: AssertionError:`` with no text -- and under
  ``python -O`` a kernel was built whose loop never reads the tail of the axis. The steps
  are the ones that divide the extent now (1 and the whole axis), and a step that does not
  is a refusal with its reason, not an assert. (That does not give these shapes a kernel:
  the whole row in one step is generated, and TileLang has no layout for a fragment of
  such a row -- "no available layout found" on a GPU at 61, 65, 96, 130, 192 and 257
  columns alike, with or without this change. A row the tile has to hold whole compiles
  where its length is a power of two; any other needs a padded last tile, which is not
  built here.);
* the output tile of a tensor-core matmul started at 64 rows and 32 columns whatever the
  extents were, and the walk over the output tiles died on that start for 96 or 100 rows or
  48 columns (``ValueError: 64 is not in list``). It starts at the largest step of the axis
  at or below that tile now (32 of 96 rows, 16 of 48 columns), and only an axis with no
  step a tensor-core tile takes (100 rows: 1, 2, 4, 100) is refused, with that reason. The
  tilings TileLang cannot partition among the warps or lay out are not proposed
  (``GEMM_TILE_CONFLICT``): attention over 96 queries has none left, and says so.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import subprocess
import sys
import textwrap

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import ScheduleRejected
from cleave.frontend import trace
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling
from cleave.schedule.policy.default import TCPolicy

from tests import _explicit_softmax as X

NO_MULTIPLE_OF_16 = (
    r"mma_\d+ contracts 1000 float16 elements: a tensor-core matmul of 16-bit operands "
    r"steps its reduction axis in multiples of 16, and none divides that extent"
)
NOT_PADDED = "the last tile of a reduction is not padded"


def flash_fixed_shift(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                      v: cl.Tensor["H", "SKV", "D"]):
    """The flash form with a constant shift written by hand: no max, no record."""
    e = cl.exp((q @ k.mT) / q.shape[-1] ** 0.5 + -6.25)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def _shapes(skv):
    return dict(H=32, SQ=64, SKV=skv, D=64)


def _lowered(fn, shapes, dtype=cl.f16):
    """The identity (un-searched) graph of ``fn``, lowered as the pipeline lowers it (the
    update rule of a softmax's max derived)."""
    (graph,) = list(GraphSet.identity(trace(fn, shapes, dtype)))
    (ir,) = lower(graph)
    return ir


@pytest.mark.parametrize("fn", [X.flash_attention, flash_fixed_shift],
                         ids=["exact", "fixed_shift"])
def test_an_extent_the_tensor_core_step_does_not_divide_is_refused_with_its_reason(fn):
    plans = P.schedule(GraphSet.identity(trace(fn, _shapes(1000), cl.f16)),
                       P.ScheduleConfig(topk=16))
    assert not plans and plans.refused
    reasons = {str(r.error) for r in plans.refused}
    assert len(reasons) == 1, reasons
    (reason,) = reasons
    # the value matmul, over the keys: the refusal names the node, the extent and the cause
    assert re.fullmatch(NO_MULTIPLE_OF_16, reason), reason
    assert "ValueError" not in reason and "not in list" not in reason
    assert reason in plans.summary()


def outer(x: cl.Tensor[1, "K"], s: cl.Tensor[1, "N"]):
    """A matmul that contracts one element: the rank-1 product an RMSNorm at M = 1 becomes
    (``x^T * rsqrt(...)``)."""
    return x.mT @ s


@pytest.mark.parametrize("n", [1, 64])
def test_a_matmul_over_one_element_is_refused_with_its_reason(n):
    """The carver simplifies the unit reduction axis away, so the matmul's block has no
    reduction; reading its input buffers through it ended the tiling search with ``TypeError:
    In "Schedule.get", "rand_var_or_sref" has wrong type`` (RMSNorm(x) @ W at M = 1)."""
    plans = P.schedule(GraphSet.identity(trace(outer, dict(K=256, N=n), cl.f16)),
                       P.ScheduleConfig(topk=8))
    assert not plans and plans.refused
    for refusal in plans.refused:
        text = str(refusal.error)
        assert re.fullmatch(r"mma_\d+ contracts 1 element \(its unit reduction axis is "
                            r"simplified away\): a tensor-core matmul steps its reduction "
                            r"axis in multiples of 16 \(8 for float32 operands\), and none "
                            r"divides that extent", text), text
        assert "TypeError" not in text and "rand_var_or_sref" not in text


def test_the_policy_itself_raises_the_refusal():
    ir = _lowered(X.flash_attention, _shapes(1000))
    for force in (False, True):
        with pytest.raises(ScheduleRejected, match=NO_MULTIPLE_OF_16):
            list(search_tiling(ir, topk=16, force_fusion=force))


@pytest.mark.parametrize("skv", [96, 1024, 1536])
def test_extents_the_step_divides_schedule_as_before(skv):
    plans = P.schedule(GraphSet.identity(trace(X.flash_attention, _shapes(skv), cl.f16)),
                       P.ScheduleConfig(topk=16))
    assert len(plans) >= 1
    assert not any(NOT_PADDED in str(r.error) or "none divides that extent" in str(r.error)
                   for r in plans.refused)


def test_the_walk_refuses_a_start_that_is_none_of_its_steps(monkeypatch):
    """``expand_iterative_tiles`` walks the steps that divide a reduction's extent, from the
    step ``_assign_reduce_step`` starts at. The tensor-core policy starts at one of them
    today, so a program no longer reaches this; a start from anywhere else is still a
    refusal that names the reduction, its extent and the step, not a ``ValueError`` from the
    lookup. Here a matmul over 1024 elements is started at 24."""
    from tests._graphs import matmul_graph

    real = TCPolicy._assign_reduce_step
    started = []

    def start_outside_the_steps(self, node):
        steps = real(self, node)
        if node.get_tag("tensorcore_config"):
            started.append(node.name)
            return {name: 24 for name in steps}
        return steps

    monkeypatch.setattr(TCPolicy, "_assign_reduce_step", start_outside_the_steps)
    with pytest.raises(ScheduleRejected) as refusal:
        list(search_tiling(matmul_graph(128, 128, 1024), topk=16))
    assert started == ["mma_0"]
    assert str(refusal.value) == (
        "mma_0: its reduction over 1024 elements starts at step 24, which does not "
        f"divide it; {NOT_PADDED}")


# ------------------------------------------------ a reduction over an odd extent above 64


def row_sum(x: cl.Tensor["R", "C"]):
    return cl.sum(x, -1, keepdim=True)


def softmax_rows(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1)


def _reduction_steps(tile):
    return sorted({int(step) for steps in tile.rstep_map.values() for step in (steps or {}).values()})


@pytest.mark.parametrize("columns", [65, 255, 257])
@pytest.mark.parametrize("fn", [row_sum, softmax_rows], ids=["row_sum", "softmax"])
def test_a_reduction_over_an_odd_extent_is_stepped_by_the_whole_axis(fn, columns):
    """65, 255 or 257 columns: the steps that divide such an extent are 1 and the whole
    axis. Neither was what the policy proposed (64, 128, ...), so the row sum had
    candidates that all died on the assert in ``apply_tiling`` and ``cl.softmax`` had 0
    plans, refused with ``ScheduleRejected: AssertionError:``. Every plan now steps its
    reduction by something that divides it and is generated or refused with a reason."""
    plans = P.schedule(GraphSet.identity(trace(fn, dict(R=64, C=columns), cl.f16)),
                       P.ScheduleConfig(topk=1000))
    assert len(plans) >= 1, plans.summary()
    assert not any("AssertionError" in r.reason for r in plans.refused), plans.summary()
    built = 0
    for plan in plans:
        steps = _reduction_steps(plan.tile)
        assert steps and all(columns % step == 0 for step in steps), plan.label
        try:
            build_candidate_code(plan.tile)
        except ScheduleRejected:
            continue  # a refusal with its reason (a single-row tile); anything else fails
        built += 1
        assert columns in steps, plan.label
    assert built >= 1


def test_a_long_odd_row_is_summed_in_one_step_or_refused_with_the_reason():
    """4099 columns: the whole axis is still a step, and the tilings that hold it build."""
    plans = P.schedule(GraphSet.identity(trace(row_sum, dict(R=64, C=4099), cl.f16)),
                       P.ScheduleConfig(topk=1000))
    assert len(plans) >= 1 and not any("AssertionError" in r.reason for r in plans.refused)
    outcomes = []
    for plan in plans:
        assert all(4099 % step == 0 for step in _reduction_steps(plan.tile))
        try:
            build_candidate_code(plan.tile)
            outcomes.append("built")
        except ScheduleRejected as refusal:
            assert str(refusal) and "AssertionError" not in str(refusal)
            outcomes.append("refused")
    assert "built" in outcomes


def _stepped_by(columns, step):
    """A tiling of the row sum over ``columns`` whose reduction is stepped by ``step``: one
    the search proposes, with its step replaced."""
    ir = _lowered(row_sum, dict(R=64, C=columns))
    cand = next(c for c in search_tiling(ir, topk=1000) if _reduction_steps(c) == [columns])
    for steps in cand.rstep_map.values():
        for name in steps:
            steps[name] = step
    return cand


UNDIVIDED = (r"reduction_1_\d+: its reduction over 257 elements is stepped by 64, which does "
             "not divide it; the last tile of a reduction is not padded")


def test_a_step_that_does_not_divide_the_extent_is_a_refusal_not_an_assert():
    with pytest.raises(ScheduleRejected, match=UNDIVIDED):
        apply_tiling(_stepped_by(257, 64))
    with pytest.raises(ScheduleRejected, match=UNDIVIDED):
        build_candidate_code(_stepped_by(257, 64))


def test_the_refusal_holds_without_asserts():
    """``python -O`` drops an assert. The loop built then ran 4 steps of 64 over 257
    columns and never read the last one."""
    script = textwrap.dedent("""
        import sys
        assert False, "asserts are off"
        from cleave.core.errors import ScheduleRejected
        from cleave.runtime.compile import build_candidate_code
        from tests.schedule.search_tiling.test_undivided_extent import _stepped_by
        try:
            build_candidate_code(_stepped_by(257, 64))
        except ScheduleRejected as refusal:
            print("REFUSED", refusal)
        else:
            print("BUILT")
    """)
    done = subprocess.run([sys.executable, "-O", "-c", script], capture_output=True, text=True,
                          env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    assert done.returncode == 0, done.stderr[-2000:]
    assert re.search("REFUSED " + UNDIVIDED, done.stdout), done.stdout[-500:]


def two_row_sums(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "D"]):
    return cl.sum(x, -1, keepdim=True) + cl.sum(y, -1, keepdim=True)


def _both_stepped_by(shapes, step):
    """A tiling of ``two_row_sums`` the search proposes -- the sum over 256 columns stepped
    by ``step``, the one over 257 by the whole axis -- with the sum over 257 stepped by
    ``step`` as well: its entry of the step map and the inner tile of the input it reads."""
    ir = _lowered(two_row_sums, shapes)

    def steps(cand, extent):
        (node,) = [n for n in cand.rstep_map if n.raxis and int(n.raxis[0].dom.extent) == extent]
        return node, sorted(cand.rstep_map[node].values())

    cand = next(c for c in search_tiling(ir, topk=1000)
                if steps(c, 256)[1] == [step] and steps(c, 257)[1] == [257]
                and int(c.tile_map[steps(c, 257)[0]][0]) == 16)
    odd, _ = steps(cand, 257)
    cand.rstep_map[odd] = {name: step for name in cand.rstep_map[odd]}
    source = odd.inputs[0].src_node
    assert [int(t) for t in cand.itile_map[source]] == [16, 257]
    cand.itile_map[source] = [16, step]
    return cand


JOINED = (r"GraphNode_reduction_1_\d+ steps dim 1 of PlaceHolder_input_\d+ by 64, which does "
          "not divide its 257 elements: the tail would never be read")


def test_a_sum_that_joins_another_sums_loop_is_held_to_a_dividing_step_too():
    """``apply_tiling`` checks the step of the reduction that opens a loop. A second
    reduction with the same step and the same number of whole steps joins that loop
    without passing the check: 256 columns and 257 columns, both stepped by 64, are 4
    iterations each, rounded down. The search proposed exactly that, and the kernel summed
    ``x[64 * i : 64 * i + 64]`` for ``i`` below 4: the last column of the 257 was never
    read. The policy's steps divide the extent now, so the search no longer proposes it;
    stepped by hand, the loop is refused once every merge is done, for the dim that the
    step does not divide."""
    # x has the 257 columns: its sum comes first in the graph, the sum over y opens the
    # loop and the sum over x joins it
    with pytest.raises(ScheduleRejected, match=JOINED):
        apply_tiling(_both_stepped_by(dict(R=64, C=257, D=256), 64))
    with pytest.raises(ScheduleRejected, match=JOINED):
        build_candidate_code(_both_stepped_by(dict(R=64, C=257, D=256), 64))
    # y has them: its sum opens the loop, and that is the check in apply_tiling itself
    with pytest.raises(ScheduleRejected, match=UNDIVIDED):
        apply_tiling(_both_stepped_by(dict(R=64, C=256, D=257), 64))


def test_two_row_sums_are_stepped_by_what_divides_each():
    """What the search proposes for the same program: every step divides its own axis."""
    for shapes in (dict(R=64, C=257, D=256), dict(R=64, C=256, D=257)):
        cands = list(search_tiling(_lowered(two_row_sums, shapes), topk=1000))
        assert cands
        for cand in cands:
            for node, steps in cand.rstep_map.items():
                for step in (steps or {}).values():
                    assert int(node.raxis[0].dom.extent) % int(step) == 0, (node.name, step)
            apply_tiling(cand)


# ------------------------------------------- an output axis the tensor-core tile does not divide


def plain_matmul(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    return x @ w


def _no_tile(axis, extent, steps):
    return (rf"\w+: axis {axis} of the output has {extent} elements and no step of 16 to 64 "
            rf"\(its steps: \[{steps}\]\); a tensor-core tile is not padded")


@pytest.mark.parametrize("shapes", [
    dict(M=96, K=4096, N=128), dict(M=80, K=256, N=128), dict(M=192, K=256, N=48),
    dict(M=128, K=256, N=80),
], ids=["96_rows", "80_rows", "48_columns", "80_columns"])
def test_an_output_axis_the_tensor_core_tile_does_not_divide_starts_at_a_step_it_has(shapes):
    """The tensor-core tile starts at 64 rows and 32 columns; for 96 or 80 rows, or 48 or 80
    columns, that is none of the axis's steps, and the walk refused every such matmul ("an
    output tile is not padded", and before that ``ValueError: 64 is not in list``). It
    starts at the largest step at or below it now (32 of 96 rows, 16 of 80, 16 of 48 or 80
    columns): nothing needs padding. Every proposed tiling lowers, with the device compile;
    those TileLang cannot partition or lay out are not proposed (``GEMM_TILE_CONFLICT``)."""
    from tests._lowering import lower_to_cuda

    ir = _lowered(plain_matmul, shapes)
    cands = list(search_tiling(ir, topk=100000))
    assert cands
    seen = {}
    for c in cands:
        seen.setdefault((tuple(int(x) for x in c.output_tile), c.n_threads), c)
    for (tile, threads), c in sorted(seen.items()):
        assert shapes["M"] % tile[0] == 0 and shapes["N"] % tile[1] == 0, tile
        lower_to_cuda(build_candidate_code(c), device_compile=True)
    plans = P.schedule(GraphSet.identity(trace(plain_matmul, shapes, cl.f16)),
                       P.ScheduleConfig(topk=16))
    assert plans and not plans.refused


def test_a_row_count_with_no_step_a_tensor_core_tile_takes_is_refused_with_its_reason():
    """100 rows: the steps that divide 100 are 1, 2, 4 and 100, none from the 16 rows a
    tensor-core matmul tile needs to the 64 it starts at. That is the reason: the last tile
    of the output is not padded."""
    shapes = dict(M=100, K=256, N=128)
    reason = _no_tile(0, 100, "1, 2, 4, 100")
    ir = _lowered(plain_matmul, shapes)
    for force in (False, True):
        with pytest.raises(ScheduleRejected, match=reason):
            list(search_tiling(ir, topk=16, force_fusion=force))
    plans = P.schedule(GraphSet.identity(trace(plain_matmul, shapes, cl.f16)),
                       P.ScheduleConfig(topk=16))
    assert not plans and plans.refused
    reasons = {str(r.error) for r in plans.refused}
    assert len(reasons) == 1, reasons
    (text,) = reasons
    assert re.fullmatch(reason, text), text
    assert "ValueError" not in text and "not in list" not in text and text in plans.summary()


def test_attention_over_96_queries_is_refused_with_the_partition_reason():
    """The rows start at 32 of 96 now, and the walk proposes 96-row tiles, which the
    warp-group MMA cannot split, and 32-row ones, which leave a warp fewer than 16 rows of
    the exp's registers to read. No tiling: the reason is what TileLang cannot do."""
    from cleave.schedule.policy.default import GEMM_TILE_CONFLICT

    shapes = dict(H=12, SQ=96, SKV=256, D=64)
    assert list(search_tiling(_lowered(X.flash_attention, shapes), topk=16)) == []
    plans = P.schedule(GraphSet.identity(trace(X.flash_attention, shapes, cl.f16)),
                       P.ScheduleConfig(topk=16))
    assert not plans and plans.refused
    for refusal in plans.refused:
        text = str(refusal.error)
        assert text.startswith("no tiling candidates") and f"x {GEMM_TILE_CONFLICT}" in text
        assert "not padded" not in text


def test_the_gemm_tile_rule_is_what_tilelang_cannot_build():
    """Each case of ``TCPolicy._gemm_tile_conflict`` on sm_90, and none elsewhere."""
    import copy

    ir = _lowered(plain_matmul, dict(M=256, K=256, N=256))
    policy = TCPolicy([n for n in ir.nodes if n.is_output()],
                      __import__("cleave.core.target", fromlist=["x"]).current_target().arch,
                      flat_graph=ir)
    (mma,) = [n for n in ir.nodes if n.get_tag("tensorcore_config")]
    cand = copy.copy(next(iter(search_tiling(ir, topk=1))))

    def conflict(rows, cols, threads):
        cand.get_itile = lambda node: [rows, cols]
        return policy._gemm_tile_conflict(cand, threads)

    assert not conflict(64, 32, 128) and not conflict(64, 32, 256)
    assert not conflict(32, 32, 256) and not conflict(128, 16, 256) and not conflict(64, 48, 128)
    assert conflict(16, 32, 256)  # room for 4 warps only
    assert conflict(96, 64, 128) and conflict(80, 128, 256)  # warp-group MMA rows
    assert conflict(64, 16, 256) and conflict(192, 48, 256)  # two warp groups, 16 / 48 columns
    assert mma is not None
    policy.arch = copy.copy(policy.arch)
    policy.arch.sm_version = 80
    assert not conflict(96, 64, 128)


@pytest.mark.parametrize("shapes", [dict(M=128, K=256, N=128), dict(M=64, K=256, N=32),
                                    dict(M=48, K=256, N=24)],
                         ids=["128x128", "64x32", "48x24"])
def test_output_axes_the_tile_divides_or_holds_whole_schedule_as_before(shapes):
    assert len(list(search_tiling(_lowered(plain_matmul, shapes), topk=16))) >= 1
