"""A rolling loop that multiplies a slice it loads into the softmax's output is a plain loop (CPU).

A pipelined loop keeps the slices of several iterations in one buffer and loads the next
ones while its body runs. The body tells the loader when it is done with a slice, and that
slot is loaded again. What tells it is not something the device compiler treats as touching
memory, so a slice the body copied into registers may in fact be read later, where its
values are used -- after the slot was handed back. A matmul reads its slice in place and is
not exposed; a slice that is multiplied into the softmax's output is used last by the sums
over the softmax axis, behind the running max and the rescale, and is.

Measured on the combine of an exact split attention (8 partials, blocks of 64 rows, 2
partials an iteration, 64 value columns a thread): columns 24 and up of every row came
from a later iteration's partials, a different count of them on every launch, where the
same loop emitted as a plain one is right on every input case. ``tests/pipeline_api/
test_exact_split_attention.py`` runs that kernel on a GPU.

Pinned here: which loops are plain ones, and that the attention kernels keep their pipeline.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.schedule.stabilize as stab
from cleave.core.errors import ScheduleRejected
from cleave.ir.graph import Iterator
from cleave.ir.graph_utils import edit_graph, graph_tiers, tmap_by_name
from cleave.ir.ops import Matmul, Mul, ReduceSum
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling

from tests import _softmax as S
from tests._lowering import lower_to_cuda
from tests.graph.apply_tiling.test_stabilize import _find, _pick, _pristine

COLUMN = dict(H=4, K=4096, N=256)


def _loop_kinds(code):
    """How each loop tier of the kernel is emitted, in order."""
    return re.findall(r"for i0 in T\.(Pipelined|serial)\(", code)


def _rolling(ir, tile, **steps):
    """``(candidate, loop tier, exp, rescaled accumulators)`` of one rolling tiling."""
    cand = _pick(ir, tile, **steps)
    tiled = apply_tiling(cand)
    (exp,) = [n for e in S.exps(ir) for n in [_find(tiled, e.name)[2]]]
    loop = _find(tiled, exp.name)[0]
    site = exp.stabilized
    assert site.case == "rolling"
    accs = [a for a in loop.accum_nodes if a.name in site.scaled]
    assert len(accs) == len(site.scaled) == 2
    return cand, loop, exp, accs


def test_a_slice_multiplied_into_the_softmax_makes_its_loop_a_plain_one():
    """``sum_k(softmax_k(x) * y)``: every iteration loads a slice of ``y`` and multiplies
    it by the exp of the slice of ``x``."""
    ir = S.softmax_ir(S.column_average, COLUMN, dict(axis=-2))
    sums = [n for n in ir.nodes if type(n) is ReduceSum]
    cand, loop, exp, accs = _rolling(ir, [1, 64], **{n.name: 32 for n in sums})
    (product,) = [n for n in loop.nodes if type(n) is Mul]
    (y,) = [e.src_node for e in product.inputs if e.src_node is not exp]
    assert y.is_placeholder()
    # y is the one: the logits pass through the exp, in front of the statistics
    assert stab._weighted_slices(loop, exp, accs) == [y]
    assert loop.loop_kind == "serial"

    code = build_candidate_code(cand)
    assert _loop_kinds(code) == ["serial"]
    cuda = lower_to_cuda(code)
    # nothing is loaded behind the body's back: no transfer, no barrier to wait on
    assert "tma_load" not in cuda and "mbarrier" not in cuda and "cp_async" not in cuda
    assert "tl::MaxOp" in cuda and cuda.count("* t_" + exp.name + "_alpha[") == 2


@pytest.mark.parametrize("fn, shapes, tile", [
    (S.flash_attention, dict(H=32, SQ=4096, SKV=4096, D=64), [1, 128, 64]),
    (S.flash_attention, dict(H=32, SQ=64, SKV=256, D=64), [1, 64, 32]),
], ids=["prefill", "decode"])
def test_attention_keeps_its_pipeline(fn, shapes, tile):
    """The keys reach the accumulators through the exp, and the values are read in place by
    their matmul: nothing is weighted in registers."""
    ir = S.softmax_ir(fn, shapes)
    matmuls = [n for n in ir.nodes if type(n) is Matmul]
    rowsum = next(n for n in ir.nodes if type(n) is ReduceSum)
    steps = {matmuls[0].name: 64, rowsum.name: 32, matmuls[-1].name: 32}
    cand, loop, exp, accs = _rolling(ir, tile, **steps)
    assert stab._weighted_slices(loop, exp, accs) == []
    assert loop.loop_kind == "pipelined"
    assert _loop_kinds(build_candidate_code(cand)) == ["Pipelined"]


def test_logits_that_are_an_input_keep_the_pipeline():
    """``(softmax(x) @ v)``: the loop loads slices of the logits and of the values. The
    logits are consumed by the exp, the values by the matmul."""
    ir = S.softmax_ir(S.flash_rows, dict(R=256, C=4096, D=64))
    matmul = next(n for n in ir.nodes if type(n) is Matmul)
    rowsum = next(n for n in ir.nodes if type(n) is ReduceSum)
    cand, loop, exp, accs = _rolling(ir, [128, 64], **{matmul.name: 32, rowsum.name: 32})
    assert stab._weighted_slices(loop, exp, accs) == []
    assert _loop_kinds(build_candidate_code(cand)) == ["Pipelined"]


def test_a_tile_the_loop_does_not_cut_is_not_a_loaded_slice():
    """What the parent hands down whole is loaded once, in front of the loop: nothing can
    overwrite it. Only a tile the loop cuts per iteration counts."""
    ir = S.softmax_ir(S.column_average, COLUMN, dict(axis=-2))
    sums = [n for n in ir.nodes if type(n) is ReduceSum]
    cand = _pick(ir, [1, 64], **{n.name: 32 for n in sums})
    tiled = _pristine(ir, cand)
    loop, _, exp = _find(tiled, S.exps(ir)[0].name)
    (product,) = [n for n in loop.nodes if type(n) is Mul]
    (y,) = [e.src_node for e in product.inputs if e.src_node is not exp]
    with edit_graph(tiled):
        it = tmap_by_name(loop)[y.name]
        loop.tmap[y] = Iterator(it.offset, shape=it.tile_size, tile_size=it.tile_size,
                                axes=it.axes)
    accs = [a for a in loop.accum_nodes if a.name in dict(exp.stabilize.scaled)]
    assert len(accs) == 2 and stab._weighted_slices(loop, exp, accs) == []


def test_only_the_rolling_form_changes_the_loop():
    """A softmax whose axis is whole in its tile keeps the loop it sits in as it is: the
    rule is about what a running max and its rescale put between a load and its use."""
    ir = S.softmax_ir(S.column_average_2d, dict(K=256, N=1024), dict(axis=-2))
    kinds = {}
    for cand in search_tiling(ir, topk=1000):
        try:
            tiled = apply_tiling(cand)
        except ScheduleRejected:  # a refused tiling has no kernel
            continue
        for site in stab.sites(tiled):
            if site.tier == "main":
                continue
            loop = next(t for t, o in graph_tiers(tiled) if o is not None
                        and o.name == site.tier)
            kinds.setdefault(site.case, set()).add(loop.loop_kind)
    assert kinds.get("rolling") == {"serial"}
    assert kinds.get("in-tile", {"pipelined"}) == {"pipelined"}
