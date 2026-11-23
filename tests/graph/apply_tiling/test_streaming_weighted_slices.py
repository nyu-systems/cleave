"""A loop that carries a max and multiplies a slice it loads into the softmax's output is a
plain loop (CPU).

A pipelined loop keeps the slices of several iterations in one buffer and loads the next
ones while its body runs. The body tells the loader when it is done with a slice, and that
slot is loaded again. What tells it is not something the device compiler treats as touching
memory, so a slice the body copied into registers may in fact be read later, where its
values are used -- after the slot was handed back. A matmul reads its slice in place and is
not exposed; a slice that is multiplied into the softmax's output is used last by the sums
over the softmax axis, behind the running max and the repair, and is (measured on the
hand-derived pass's split combine: columns 24 and up of every row came from a later
iteration's partials). The assembly makes such a loop a plain one
(``streaming._weighted_slices``), as the hand-derived pass did.

Pinned here: which loops are plain ones, and that the attention kernels keep their pipeline.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.schedule.streaming as stream
from cleave.core.errors import ScheduleRejected
from cleave.ir.graph import Iterator
from cleave.ir.graph_utils import edit_graph, graph_tiers, tmap_by_name
from cleave.ir.ops import Matmul, Mul, ReduceSum
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling

from tests import _explicit_softmax as X
from tests._lowering import lower_to_cuda
from tests.graph.apply_tiling.test_streaming import _find, _pick, _unassembled

COLUMN = dict(H=4, K=4096, N=256)


def _loop_kinds(code):
    """How each loop tier of the kernel is emitted, in order."""
    return re.findall(r"for i0 in T\.(Pipelined|serial)\(", code)


def _carried(ir, tile, **steps):
    """``(candidate, loop tier, exp, repaired accumulators)`` of one looped tiling."""
    cand = _pick(ir, tile, **steps)
    tiled = apply_tiling(cand)
    (site,) = stream.sites(tiled)
    (exp,) = [_find(tiled, e.name)[2] for e in X.exps(ir)]
    loop = _find(tiled, exp.name)[0]
    accs = [a for a in loop.accum_nodes if a.name in site.repaired]
    assert len(accs) == len(site.repaired) == 2
    return cand, loop, exp, accs


def _column_steps(ir, step):
    (m,) = X.maxes(ir)
    return {m.name: step, **{n.name: step for n in ir.nodes if type(n) is ReduceSum}}


def test_a_slice_multiplied_into_the_softmax_makes_its_loop_a_plain_one():
    """``sum_k(softmax_k(x) * y)``: every iteration loads a slice of ``y`` and multiplies
    it by the exp of the slice of ``x``."""
    ir = X.streamed_ir(X.column_average, COLUMN)
    cand, loop, exp, accs = _carried(ir, [1, 64], **_column_steps(ir, 32))
    (product,) = [n for n in loop.nodes if type(n) is Mul]
    (y,) = [e.src_node for e in product.inputs if e.src_node is not exp]
    assert y.is_placeholder()
    # y is the one: the logits pass through the exp, in front of the statistics
    assert stream._weighted_slices(loop, exp, accs) == [y]
    assert loop.loop_kind == "serial"

    code = build_candidate_code(cand)
    assert _loop_kinds(code) == ["serial"]
    cuda = lower_to_cuda(code)
    # nothing is loaded behind the body's back: no transfer, no barrier to wait on
    assert "tma_load" not in cuda and "mbarrier" not in cuda and "cp_async" not in cuda
    m = X.maxes(ir)[0].name
    assert "tl::MaxOp" in cuda and cuda.count(f"* t_{m}_upd1[") == 2


@pytest.mark.parametrize("shapes, tile", [
    (dict(H=32, SQ=4096, SKV=4096, D=64), [1, 128, 64]),
    (dict(H=32, SQ=64, SKV=256, D=64), [1, 64, 32]),
], ids=["prefill", "decode"])
def test_attention_keeps_its_pipeline(shapes, tile):
    """The keys reach the accumulators through the exp, and the values are read in place by
    their matmul: nothing is weighted in registers."""
    ir = X.streamed_ir(X.flash_attention, shapes)
    matmuls = [n for n in ir.nodes if type(n) is Matmul]
    rowsum = next(n for n in ir.nodes if type(n) is ReduceSum)
    (m,) = X.maxes(ir)
    steps = {matmuls[0].name: 64, m.name: 32, rowsum.name: 32, matmuls[-1].name: 32}
    cand, loop, exp, accs = _carried(ir, tile, **steps)
    assert stream._weighted_slices(loop, exp, accs) == []
    assert loop.loop_kind == "pipelined"
    assert _loop_kinds(build_candidate_code(cand)) == ["Pipelined"]


def test_logits_that_are_an_input_keep_the_pipeline():
    """``(softmax(x) @ v)``: the loop loads slices of the logits and of the values. The
    logits are consumed by the max and the shift, the values by the matmul."""
    ir = X.streamed_ir(X.flash_rows, dict(R=256, C=4096, D=64))
    matmul = next(n for n in ir.nodes if type(n) is Matmul)
    rowsum = next(n for n in ir.nodes if type(n) is ReduceSum)
    (m,) = X.maxes(ir)
    cand, loop, exp, accs = _carried(ir, [128, 64], **{m.name: 64, matmul.name: 64,
                                                       rowsum.name: 64})
    assert stream._weighted_slices(loop, exp, accs) == []
    assert _loop_kinds(build_candidate_code(cand)) == ["Pipelined"]


def test_a_tile_the_loop_does_not_cut_is_not_a_loaded_slice(monkeypatch):
    """What the parent hands down whole is loaded once, in front of the loop: nothing can
    overwrite it. Only a tile the loop cuts per iteration counts."""
    ir = X.streamed_ir(X.column_average, COLUMN)
    cand = _pick(ir, [1, 64], **_column_steps(ir, 32))
    tiled = _unassembled(cand, monkeypatch)
    loop, _, exp = _find(tiled, X.exps(ir)[0].name)
    (product,) = [n for n in loop.nodes if type(n) is Mul]
    (y,) = [e.src_node for e in product.inputs if e.src_node is not exp]
    sums = [a for a in loop.accum_nodes if type(a) is ReduceSum]
    assert len(sums) == 2 and stream._weighted_slices(loop, exp, sums) == [y]
    with edit_graph(tiled):
        it = tmap_by_name(loop)[y.name]
        loop.tmap[y] = Iterator(it.offset, shape=it.tile_size, tile_size=it.tile_size,
                                axes=it.axes)
    assert stream._weighted_slices(loop, exp, sums) == []


def test_only_the_carried_form_changes_the_loop():
    """A max whose row is whole in its tile keeps the loop it sits in as it is: the rule is
    about what a running max and its repair put between a load and its use."""
    ir = X.streamed_ir(X.column_average_2d, dict(K=4096, N=1024))
    kinds = {}
    for cand in search_tiling(ir, topk=1000):
        try:
            tiled = apply_tiling(cand)
        except ScheduleRejected:  # a refused tiling has no kernel
            continue
        carried = {s.tier for s in stream.sites(tiled)}
        for tier, owner in graph_tiers(tiled):
            if owner is None:
                continue
            kinds.setdefault("carried" if owner.name in carried else "not carried",
                             set()).add(tier.loop_kind)
    assert kinds.get("carried") == {"serial"}
    assert kinds.get("not carried", {"pipelined"}) == {"pipelined"}
