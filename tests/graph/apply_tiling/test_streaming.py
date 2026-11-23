"""The assembly of a carried max on tiled graphs: ``cleave.schedule.streaming`` (CPU).

The flat graph keeps the stable softmax as it is written, ``exp(x - max(x)) / sum``. Where
a reduction loops under that max, ``apply_tiling`` builds the loop it always built and ends
with :func:`~cleave.schedule.streaming.assemble`: the max becomes carried state of the loop
(started at ``-FLT_MAX``, snapshot before it folds in the tile), and every accumulator under
it is multiplied by the factor of its proven update rule right before its own macro. Where
the row is whole in one tile, the flat graph's own max is right as it stands and nothing is
assembled. The records are put on the max by ``tests/_explicit_softmax.py``, keyed by the
graph itself, as lowering writes them after the search.

Pinned here: the hook and what it leaves alone, the carried tier and its text, the returned
max, the one-tile tilings, the tiers the tiler builds, the properties (idempotent, clones,
hash seed), lowering without a GPU, and every refusal a tiling can be refused for (L1-L14).
"""

import dataclasses
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from tilelang.carver.roller.node import Edge

import cleave.schedule.streaming as stream
from cleave.codegen import TilelangCodeGen, passes
from cleave.core.errors import ScheduleRejected
from cleave.ir.graph import AccumEdge, GraphNode, Iterator
from cleave.ir.graph_utils import graph_tiers, tmap_by_name
from cleave.ir.ops import NEG_FLT_MAX, Copy, Exp, Matmul, ReduceMax, ReduceSum, Sub
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling

from tests import _explicit_softmax as X
from tests._lowering import lower_to_cuda

ROOT = Path(__file__).resolve().parents[3]
PREFILL = dict(H=32, SQ=4096, SKV=4096, D=64)
SMALL = dict(H=32, SQ=64, SKV=256, D=64)
COLUMN = dict(H=4, K=4096, N=256)
RETURNED = "is also read by"  # T4: a value computed from the max is read next to its sum
TWO_PASSES = "a finished reduction over the dim"  # T3: the traced order


# ------------------------------------------------------------------------------- helpers


def _steps(cand) -> dict:
    return {n.name: int(next(iter(v.values()))) for n, v in cand.rstep_map.items() if v}


def _pick(ir, tile, threads=128, **steps):
    """The candidate with this output tile, thread count and reduction steps (by name)."""
    for cand in search_tiling(ir, topk=1000):
        if list(cand.output_tile) == tile and cand.n_threads == threads and _steps(cand) == steps:
            return cand
    raise AssertionError(f"no candidate tile={tile} threads={threads} steps={steps}")


def _attention(fn, shapes, *refused):
    """A lowered attention graph with its record, and its nodes by role."""
    ir = X.streamed_ir(fn, shapes, *refused)
    matmuls = [n for n in ir.nodes if type(n) is Matmul]
    return SimpleNamespace(
        ir=ir,
        max=X.maxes(ir)[0],
        exp=X.exps(ir)[0],
        rowsum=next(n for n in ir.nodes if type(n) is ReduceSum),
        score=matmuls[0],
        value=matmuls[-1],
    )


def _find(tiled, name):
    """``(tier, owner, node)`` of the node called ``name``."""
    for tier, owner in graph_tiers(tiled):
        for node in tier.nodes:
            if node.name == name:
                return tier, owner, node
    raise AssertionError(f"no node {name!r}")


def _loops(tiled):
    return [(tier, owner) for tier, owner in graph_tiers(tiled) if owner is not None]


def _names(tiled):
    return [(None if owner is None else owner.name, [n.name for n in tier.nodes])
            for tier, owner in graph_tiers(tiled)]


def _unassembled(cand, monkeypatch):
    """The tiled graph as the tiler alone builds it, the record still on the tiled max:
    what ``assemble`` is handed by ``apply_tiling``, open to doctoring."""
    with monkeypatch.context() as patch:
        patch.setattr(stream, "assemble", lambda tiled, flat_graph=None: tiled)
        return apply_tiling(cand)


def _code(tiled) -> str:
    """The passes and codegen ``build_candidate_code`` runs after ``apply_tiling``."""
    g = passes.annotate_args(tiled, is_main=True)
    g = passes.annotate_tinfo(g, is_main=True)
    g = passes.annotate_tc_layout(g)
    g = passes.annotate_fragment_layout_conflicts(g)
    return TilelangCodeGen().gen(g)


def _calls(code, macro):
    """The statements of one macro body, one per line, allocations folded onto one line."""
    start = code.index(f"def {macro}(")
    body = code[code.index("):\n", start) + 3:]
    body = body[: body.index("\n\n\n")] if "\n\n\n" in body else body
    return [re.sub(r"\s+", " ", s).strip() for s in re.split(r"\n(?=    \S)", body) if s.strip()]


def _macro_calls(code, macro):
    return [s for s in _calls(code, macro) if s.startswith("ukernel_")]


def _retile(tier, node, *, shape=None, tile=None):
    it = tmap_by_name(tier)[node.name]
    tier.tmap[node] = Iterator(
        it.offset, shape=it.shape if shape is None else shape,
        tile_size=it.tile_size if tile is None else tile, axes=it.axes,
    )


@pytest.fixture(scope="module")
def prefill():
    a = _attention(X.flash_attention, PREFILL)
    a.cand = _pick(a.ir, [1, 128, 64], **{a.score.name: 64, a.max.name: 32,
                                         a.rowsum.name: 32, a.value.name: 32})
    return a


@pytest.fixture(scope="module")
def small():
    a = _attention(X.flash_attention, SMALL)
    looped = {a.score.name: 64, a.max.name: 32, a.rowsum.name: 32, a.value.name: 32}
    whole = {a.score.name: 64, a.max.name: 256, a.rowsum.name: 256, a.value.name: 256}
    a.looped = _pick(a.ir, [1, 64, 32], **looped)
    a.one_tile = _pick(a.ir, [1, 64, 32], **whole)
    return a


@pytest.fixture(scope="module")
def returned():
    """The flash form that also returns its max."""
    a = _attention(X.flash_and_its_max, SMALL)
    a.cand = _pick(a.ir, [1, 64, 32], **{a.score.name: 64, a.max.name: 32,
                                        a.rowsum.name: 32, a.value.name: 32})
    return a


# ---------------------------------------------------------------------- the hook, and off


def test_a_graph_without_a_record_never_reaches_the_assembly(monkeypatch):
    ir = X.lowered(X.flash_attention, SMALL)
    assert not stream.recorded(ir)

    def tripwire(*args, **kwargs):
        raise AssertionError("the assembly ran on a graph without a record")

    monkeypatch.setattr(stream, "assemble", tripwire)
    monkeypatch.setattr(stream, "carry_early", tripwire)
    cands = search_tiling(ir, topk=1000)
    assert cands
    for cand in cands:
        try:
            tiled = apply_tiling(cand)
        except ScheduleRejected:
            continue  # the pin is kept: what the tiler refuses, it refuses as before
        for tier, _ in graph_tiers(tiled):
            assert not tier.lag_nodes and not tier.accum_scale
            assert not [n for n in tier.nodes if n.name.endswith(("_prev", "_upd0", "_upd1"))]


def test_the_engine_is_not_even_imported_for_a_graph_without_a_record(tmp_path):
    """Tiling a graph is the scheduler's business: neither the update search nor its solver
    is loaded for it, with or without a max in the graph."""
    script = (
        "import sys\n"
        "from cleave.schedule import apply_tiling, search_tiling\n"
        "from tests._graphs import mha_bhsd_graph\n"
        "from tests import _explicit_softmax as X\n"
        "for ir in (mha_bhsd_graph(1, 32, 4096, 4096, 64),\n"
        "           X.lowered(X.flash_attention, dict(H=32, SQ=64, SKV=256, D=64))):\n"
        "    cands = search_tiling(ir, topk=4)\n"
        "    assert cands\n"
        "    for c in cands:\n"
        "        try:\n"
        "            apply_tiling(c)\n"
        "        except Exception as e:\n"
        "            assert type(e).__name__ == 'ScheduleRejected', e\n"
        "loaded = [m for m in ('cleave.pipeline.update', 'z3') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CLEAVE_CACHE_ROOT=str(tmp_path))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "python"), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_assemble_returns_a_graph_without_a_record_as_it_came(small, monkeypatch):
    tiled = _unassembled(small.looped, monkeypatch)
    _find(tiled, small.max.name)[2].streaming = None
    before = _names(tiled)

    def tripwire(graph):
        raise AssertionError("a graph without a record was opened for editing")

    monkeypatch.setattr(stream, "edit_graph", tripwire)
    assert stream.assemble(tiled, small.ir) is tiled
    assert _names(tiled) == before and tiled.locked


# ------------------------------------------------------------------------------- looped


def test_carried_tier(prefill):
    m, rowsum, value = prefill.max, prefill.rowsum, prefill.value
    tiled = apply_tiling(prefill.cand)
    (loop, owner), = _loops(tiled)
    carried = _find(tiled, m.name)[2]
    m_prev, step, alpha = (_find(tiled, f"{m.name}_{s}")[2] for s in ("prev", "upd0", "upd1"))

    # the carried max is appended: the reductions keep their accumulator positions
    assert [n.name for n in loop.accum_nodes] == [rowsum.name, value.name, m.name]
    assert loop.accum_nodes[2] is carried and [n.name for n in loop.lag_nodes] == [m_prev.name]
    assert {acc.name: f.name for acc, f in loop.accum_scale.items()} == {
        rowsum.name: alpha.name, value.name: alpha.name}
    assert all(f is alpha for f in loop.accum_scale.values())
    assert owner.get_shape(2) == carried.get_shape(0)

    # wiring: the max and the shifted scores are the flat graph's; the repair is the rule's
    scores = carried.inputs[0].src_node
    assert scores.name == "div_4" and scores.out_dtype == "float32"
    shifted = _find(tiled, "sub_6")[2]
    assert [e.src_node for e in shifted.inputs] == [scores, carried]
    assert _find(tiled, prefill.exp.name)[2].inputs[0].src_node is shifted
    assert type(m_prev) is Copy and m_prev.inputs[0].src_node is carried
    assert type(step) is Sub and [e.src_node for e in step.inputs] == [m_prev, carried]
    assert type(alpha) is Exp and alpha.inputs[0].src_node is step
    for node in (carried, m_prev, step, alpha, shifted):
        assert node.out_dtype == "float32", node.name

    # the sentinel is this instance's start value; the class default is still the identity
    assert carried.accum_init == NEG_FLT_MAX and ReduceMax.accum_init == -float("inf")
    assert ReduceMax.carry_init == NEG_FLT_MAX
    assert all(getattr(acc, "accum_init", None) is None for acc in loop.accum_nodes[:2])

    # the record is consumed on the tiled node; the flat graph keeps it and gains nothing
    site = carried.streamed
    assert carried.streaming is None and m.streaming is not None
    assert site == stream.Site(m.name, owner.name, axis=1, extent=4096, block=4096, step=32,
                               repaired=(rowsum.name, value.name))
    assert stream.sites(tiled) == [site]
    assert not [n for n in prefill.ir.nodes if n.name.startswith(f"{m.name}_")]
    assert m.accum_init == -float("inf") and "accum_init" not in vars(m)


def test_carried_kernel_text(prefill):
    m, l, o = prefill.max.name, prefill.rowsum.name, prefill.value.name
    code = build_candidate_code(prefill.cand)
    owner = _loops(apply_tiling(prefill.cand))[0][1].name
    # per iteration: snapshot, fold the tile's max in, the flat graph's shift and exp, the
    # rule's factor, and each accumulator repaired right before it accumulates
    calls = [re.sub(r"PlaceHolder_\w+", "P", c) for c in _macro_calls(code, f"{owner}_iter")]
    assert calls[2:] == [
        f"ukernel_copy_1dims_rr(t_{m}, t_{m}_prev)",
        f"ukernel_reduce_max_axis2_rr(t_div_4, t_{m})",
        f"ukernel_sub_broadcast_0_2dims_rrr(t_div_4, t_{m}, t_sub_6)",
        "ukernel_exp_2dims_rr(t_sub_6, t_exp_7)",
        f"ukernel_sub_1dims_rrr(t_{m}_prev, t_{m}, t_{m}_upd0)",
        f"ukernel_exp_1dims_rr(t_{m}_upd0, t_{m}_upd1)",
        f"ukernel_rescale_1dims_rr(t_{m}_upd1, t_{l})",
        f"ukernel_reduction_axis2_rr(t_exp_7, t_{l})",
        f"ukernel_rescale_lead1_2dims_rr(t_{m}_upd1, t_{o})",
        f"ukernel_mma_rsr(t_exp_7_cast, P, t_{o})",
    ]
    # the parent tier fills the carried max with the sentinel and clears the sums
    main = _calls(code, "main_kernel_iter")
    assert f"T.fill(t_{m}, -3.4028234663852886e38)" in main
    assert f"T.clear(t_{l})" in main and f"T.clear(t_{o})" in main
    assert "infinity" not in code
    assert f"t_{m}: T.Tensor((128), \"float32\")" in code


def test_carried_over_the_first_non_batch_axis():
    ir = X.streamed_ir(X.column_average, COLUMN)
    (m,) = X.maxes(ir)
    sums = [n for n in ir.nodes if type(n) is ReduceSum]
    cand = _pick(ir, [1, 64], **{m.name: 32, **{n.name: 32 for n in sums}})
    tiled = apply_tiling(cand)
    loop, owner, carried = _find(tiled, m.name)
    shifted = _find(tiled, "sub_3")[2]
    (site,) = stream.sites(tiled)

    # flat axis 1 of (H, K, N) is axis 0 of the loop's (K-slice, N) tile
    assert site.axis == 0 and (site.extent, site.block, site.step) == (4096, 4096, 32)
    assert (carried.axis, carried.axis_arg) == (0, 0) and shifted.broadcast == 1
    assert tuple(tmap_by_name(loop)[m.name].tile_size) == (64,)
    code = build_candidate_code(cand)
    calls = _macro_calls(code, f"{owner.name}_iter")
    assert any(c.startswith("ukernel_reduce_max_axis0_rr(") for c in calls)
    assert any(c.startswith("ukernel_sub_broadcast_1_2dims_rrr(") for c in calls)
    assert sum(c.startswith("ukernel_rescale_1dims_rr(") for c in calls) == 2


def test_a_returned_max_is_carried_not_duplicated(returned):
    """The max a split partial hands on next to its sums: the loop's own carried max (a
    float32 statistic) is what leaves the kernel, stored in the dtype the program returns
    it in; no second max."""
    m = returned.max.name
    tiled = apply_tiling(returned.cand)
    (loop, owner), = _loops(tiled)
    carried = _find(tiled, m)[2]
    assert sum(type(n) is ReduceMax for t, _ in graph_tiers(tiled) for n in t.nodes) == 1
    assert [n.name for n in loop.accum_nodes] == [returned.rowsum.name, returned.value.name, m]
    assert carried.accum_init == NEG_FLT_MAX and carried.inputs[0].src_node.name == "div_4"
    assert _find(tiled, f"{m}_prev")[2].inputs[0].src_node is carried
    assert {acc.name for acc in loop.accum_scale} == {returned.value.name, returned.rowsum.name}
    outputs = [n for n in tiled.nodes if n.is_output()]
    assert [str(n.get_dtype(0)) for n in outputs] == ["float16", "float16"]
    assert carried.out_dtype == "float32" and outputs[1].inputs[0].src_node.name == owner.name
    code = build_candidate_code(returned.cand)
    assert f"T.fill(t_{m}, -3.4028234663852886e38)" in _calls(code, "main_kernel_iter")
    assert f"ukernel_copy_1dims_rr(t_{m}, t_{m}_prev)" in _macro_calls(code, f"{owner.name}_iter")


# ------------------------------------------------------------------------------ one tile


def test_one_tile(small):
    m = small.max
    tiled = apply_tiling(small.one_tile)
    tier, owner, node = _find(tiled, m.name)
    assert tier is tiled and owner is None  # the whole row is in the block's tile
    assert stream.sites(tiled) == []
    # nothing assembled: the record stays on the tiled max, which starts at the identity
    assert node.streaming is not None and "accum_init" not in vars(node)
    for t, _ in graph_tiers(tiled):
        assert not t.lag_nodes and not t.accum_scale
        assert not [n for n in t.nodes if n.name.endswith(("_prev", "_upd0", "_upd1"))]

    main = _macro_calls(build_candidate_code(small.one_tile), "main_kernel_iter")
    at = main.index(f"ukernel_reduce_max_axis2_rr_clear(t_div_4, t_{m.name})")
    assert main[at + 1: at + 3] == [
        f"ukernel_sub_broadcast_0_2dims_rrr(t_div_4, t_{m.name}, t_sub_6)",
        "ukernel_exp_2dims_rr(t_sub_6, t_exp_7)",
    ]
    assert "T.fill" not in "".join(main) and "rescale" not in "".join(main)


def test_one_tile_reads_a_loop_accumulator_through_an_accum_edge():
    """``softmax(q @ k.mT) @ v`` in traced order: with QK^T looped over D the max and the
    shifted scores read an accumulator of that loop, both through an ``AccumEdge``. (The
    traced order keeps its pin: only the tilings that hold the row whole build.)"""
    a = _attention(X.unscaled_attention, SMALL, TWO_PASSES)
    cands = search_tiling(a.ir, topk=1000)
    built = []
    for cand in cands:
        try:
            built.append((cand, apply_tiling(cand)))
        except ScheduleRejected:
            pass
    assert len(built) == 8 and len(cands) == 32
    through_loop = 0
    for cand, tiled in built:
        m = _find(tiled, a.max.name)[2]
        shifted = _find(tiled, next(n.name for n in a.ir.nodes if type(n) is Sub))[2]
        assert stream.sites(tiled) == []
        edges = [m.inputs[0], shifted.inputs[0]]
        if isinstance(edges[0].src_node, GraphNode):
            through_loop += 1
            for edge in edges:
                assert isinstance(edge, AccumEdge) and edge.accum_node.name == a.score.name
                assert any(edge is e for e in edge.src_node.outputs)
        assert "ukernel_reduce_max_axis2_rr_clear(" in build_candidate_code(cand)
    assert through_loop == 4


def test_one_tile_over_the_first_axis():
    ir = X.streamed_ir(X.column_average_2d, dict(K=256, N=1024))
    (m,) = X.maxes(ir)
    sums = [n for n in ir.nodes if type(n) is ReduceSum]
    cand = _pick(ir, [32], **{m.name: 256, **{n.name: 256 for n in sums}})
    tiled = apply_tiling(cand)
    tier, owner, node = _find(tiled, m.name)
    assert owner is None and (node.axis, node.axis_arg) == (0, 0)
    assert next(n for n in tier.nodes if type(n) is Sub).broadcast == 1
    assert stream.sites(tiled) == []
    main = _macro_calls(build_candidate_code(cand), "main_kernel_iter")
    assert main[0].startswith("ukernel_reduce_max_axis0_gr_clear(")
    assert main[1].startswith("ukernel_sub_broadcast_1_2dims_rrr(")


# ------------------------------------------------------------ the tiers the tiler builds


@pytest.mark.parametrize(
    ("fn", "shapes", "kinds"),
    [
        (X.flash_attention, SMALL, {"looped", "one tile"}),
        (X.column_average, COLUMN, {"looped", "one tile"}),
        (X.column_average_2d, dict(K=4096, N=1024), {"looped", "one tile"}),
    ],
    ids=["last_axis", "first_axis_behind_a_batch_dim", "first_axis"],
)
def test_the_max_is_emitted_with_the_dim_of_the_sums_next_to_it(fn, shapes, kinds):
    """The max is a reduction over the softmax axis like the sums in its tier and gets its
    emitted ``dim`` the way they get theirs (``rebase_reduction_axis``, on the tile its
    operand is stored in): in every tiling that builds, looped or not, the two agree."""
    ir = X.streamed_ir(fn, shapes)
    (m,) = X.maxes(ir)
    seen = set()
    for cand in search_tiling(ir, topk=1000):
        try:
            tiled = apply_tiling(cand)
        except ScheduleRejected:
            continue
        tier, _, node = _find(tiled, m.name)
        sums = [n for n in tier.nodes if type(n) is ReduceSum
                and (n.flat_axis, n.flat_axis_arg) == (node.flat_axis, node.flat_axis_arg)]
        assert sums, _names(tiled)
        for s in sums:
            assert (node.axis, node.axis_arg) == (s.axis, s.axis_arg), (cand.output_tile, s.name)
        seen.add("looped" if stream.sites(tiled) else "one tile")
    assert seen == kinds


def test_a_loop_tile_of_one_column_is_refused_as_a_single_row():
    """A loop tier carries the dims its block stores. Where a block holds one column of a
    column softmax, the tile of the max is empty of rows: the axis is found where the tile
    says it is (``Iterator.position``), and the refusal says what is wrong with the tiling."""
    ir = X.streamed_ir(X.column_average, COLUMN)
    (m,) = X.maxes(ir)
    reasons, single = Counter(), None
    for cand in search_tiling(ir, topk=1000):
        try:
            apply_tiling(cand)
        except ScheduleRejected as e:
            reasons[stream._reason_key(e)] += 1
            if "holds a single row" in str(e):
                single = single or cand
    assert reasons == {
        f"{m.name}: its tile (N,) holds a single row": 11,
        f"{m.name}: one element of the reduced dim per iteration; the tile would lose that "
        "axis": 9,
    }
    assert single is not None


def test_added_nodes_say_which_dims_their_tiles_carry(small, monkeypatch):
    """A loop tier's ``Iterator`` records the dims it carries (``axes``). The nodes the
    assembly adds are row vectors like the max and carry what it carries."""
    column = X.streamed_ir(X.column_average, COLUMN)
    (cm,) = X.maxes(column)
    sums = [n for n in column.nodes if type(n) is ReduceSum]
    looped = [
        (small.max.name, small.looped),  # (H, SQ, SKV): the softmax axis is the last one
        (cm.name, _pick(column, [1, 64], **{cm.name: 32, **{n.name: 32 for n in sums}})),
    ]
    for name, cand in looped:
        tiled = apply_tiling(cand)
        tier, owner, _ = _find(tiled, name)
        its = tmap_by_name(tier)
        assert owner is not None and its[name].axes == (1,)
        for added in ("prev", "upd0", "upd1"):
            it = its[f"{name}_{added}"]
            assert it.axes == its[name].axes and tuple(it.tile_size) == tuple(its[name].tile_size)
            assert tuple(it.shape) == tuple(its[name].shape)
    # the main tier's tiles are over every dim and record nothing
    tiled = apply_tiling(small.one_tile)
    tier, owner, _ = _find(tiled, small.max.name)
    assert owner is None and tmap_by_name(tier)[small.max.name].axes is None


def test_a_unit_batch_dim_leads_like_any_other():
    """The accumulators must lead with the max's own dims (``Rescale(lead)`` multiplies
    element ``[i, ..]`` by factor element ``[i]``), which is read off where each output dim
    comes from. A unit dim comes from nowhere -- one head, a batch of one (MLA decode) --
    on both sides, and is the same dim: every looped tiling is assembled."""
    ir = X.streamed_ir(X.flash_attention, dict(H=1, SQ=64, SKV=256, D=64))
    (m,) = X.maxes(ir)
    assert int(m.get_shape()[0]) == 1
    built = Counter()
    for cand in search_tiling(ir, topk=1000):
        try:
            tiled = apply_tiling(cand)
        except ScheduleRejected as e:
            raise AssertionError(f"refused: {e}") from e
        built["looped" if stream.sites(tiled) else "one tile"] += 1
    assert built["looped"] > 0, built


def test_a_max_of_logits_in_storage_reads_them_as_stored():
    """``softmax(x) @ v`` with ``x`` an fp16 input: the max reads storage and is float32;
    the subtraction and the exp are float32 compute nodes of the flat graph, so nothing has
    to be rebuilt for the carried form."""
    ir = X.streamed_ir(X.flash_rows, dict(R=256, C=4096, D=64))
    (m,) = X.maxes(ir)
    (exp,) = X.exps(ir)
    assert (m.dtype, m.out_dtype) == ("float16", "float32")
    assert exp.dtype == "float32" and exp.inputs[0].src_node.out_dtype == "float32"
    reducers = [n for n in ir.nodes if type(n) in (ReduceSum, Matmul)]
    # (the policy starts a max of 16-bit logits where its read fills a transaction, 64
    # keys, and what loops with it starts there too)
    cand = _pick(ir, [128, 64], **{m.name: 64, **{n.name: 64 for n in reducers}})
    tiled = apply_tiling(cand)
    (site,) = stream.sites(tiled)
    tier, _, node = _find(tiled, m.name)
    assert node.inputs[0].src_node.is_placeholder() and _find(tiled, exp.name)[2].dtype == "float32"
    assert set(site.repaired) == {n.name for n in reducers}
    assert "ukernel_rescale" in build_candidate_code(cand)


# ---------------------------------------------------------------------------- properties


@pytest.mark.parametrize("which", ["looped", "one_tile"])
def test_idempotent(small, which):
    cand = getattr(small, which)
    tiled = apply_tiling(cand)
    before = _names(tiled)
    assert stream.assemble(tiled, small.ir) is tiled and _names(tiled) == before
    assert _code(tiled) == build_candidate_code(cand)


@pytest.mark.parametrize("which", ["looped", "one_tile"])
def test_clone_generates_the_same_text(small, which):
    tiled = apply_tiling(getattr(small, which))
    clone = tiled.clone()
    assert _code(clone) == _code(tiled)


_SEED_SCRIPT = """
import hashlib, json
from cleave.core.errors import ScheduleRejected
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import search_tiling
from tests import _explicit_softmax as X

out = {}
for name, fn, shapes, refused in (
    ("attention", X.flash_attention, dict(H=32, SQ=64, SKV=256, D=64), ()),
    ("rows", X.flash_rows, dict(R=256, C=4096, D=64), ()),
    ("softmax", X.softmax_rows, dict(R=64, C=1024), ("is also read by",)),
):
    for i, cand in enumerate(search_tiling(X.streamed_ir(fn, shapes, *refused), topk=1000)):
        try:
            code = build_candidate_code(cand)
        except ScheduleRejected as e:
            out[f"{name} {i}"] = f"refused: {e}"
            continue
        kind = "looped" if "ukernel_rescale" in code else "one-tile"
        out[f"{name} {i}"] = kind + " " + hashlib.sha256(code.encode()).hexdigest()
print(json.dumps(out))
"""


def test_text_does_not_depend_on_the_hash_seed(tmp_path):
    """The assembly edits a locked graph, whose node-keyed dicts hash placeholders and loop
    nodes by structure: an edit that leaves one of them keyed by a stale hash loses the
    entry, or not, depending on the process hash seed. Every candidate of three graphs --
    an attention whose max reads a float32 tile, one whose max reads fp16 storage, and a
    returned softmax that keeps its pin -- generates the same text, or is refused alike,
    under three seeds."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CLEAVE_CACHE_ROOT=str(tmp_path))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "python"), env.get("PYTHONPATH", "")]
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", _SEED_SCRIPT], cwd=ROOT,
                         env=dict(env, PYTHONHASHSEED=str(seed)),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for seed in (0, 1, 2)
    ]
    results = []
    for proc in procs:
        out, err = proc.communicate()
        assert proc.returncode == 0, err[-2000:]
        results.append(json.loads(out.strip().splitlines()[-1]))

    first = results[0]
    assert all(other == first for other in results[1:])
    served = {"attention": {"looped", "one-tile"}, "rows": {"looped"}, "softmax": {"one-tile"}}
    for graph, kinds in served.items():
        got = {v.split()[0] for k, v in first.items() if k.startswith(graph)}
        assert got - {"refused:"} == kinds, (graph, got)


def test_every_assembly_builds_its_own_nodes(small):
    """Two tilings of one flat graph get distinct carried state; the flat graph and its
    record are left as they were."""
    first, second = apply_tiling(small.looped), apply_tiling(small.looped)
    names = [f"{small.max.name}_{s}" for s in ("prev", "upd0", "upd1")]
    for name in names:
        assert _find(first, name)[2] is not _find(second, name)[2]
    assert _code(first) == _code(second)
    assert not [n for n in small.ir.nodes if n.name in names]
    assert small.max.streaming is not None and "streamed" not in vars(small.max)


# ------------------------------------------------------------------------------ lowering


def test_carried_kernel_lowers_without_a_gpu(prefill):
    m = prefill.max.name
    cuda = lower_to_cuda(build_candidate_code(prefill.cand))
    # the carried max starts at the most negative finite float32, not at -inf
    assert "-0x1.fffffep+127f" in cuda
    # the snapshot is taken before the max of this tile is folded in ...
    snapshot = cuda.index(f"t_{m}_prev[")
    snapshot = cuda.index(f"t_{m}_prev[", snapshot + 1)  # past the declaration
    fold = cuda.index("tl::MaxOp")
    assert snapshot < fold
    # ... and both accumulators are multiplied by the factor before this tile is added
    alpha = cuda.index(f"t_{m}_upd1[", cuda.index(f"expf(t_{m}_upd0["))
    assert fold < alpha < cuda.index("tl::SumOp")
    assert len(re.findall(rf"\* t_{m}_upd1\[", cuda)) == 2


def test_one_tile_kernel_lowers_without_a_gpu(small):
    cuda = lower_to_cuda(build_candidate_code(small.one_tile))
    assert cuda.index("tl::MaxOp") < cuda.index("expf(") < cuda.index("tl::SumOp")
    assert "0x1.fffffep+127f" not in cuda


def test_returned_max_kernel_lowers_without_a_gpu(returned):
    cuda = lower_to_cuda(build_candidate_code(returned.cand))
    assert "-0x1.fffffep+127f" in cuda
    assert len(re.findall(rf"\* t_{returned.max.name}_upd1\[", cuda)) == 2


# ------------------------------------------------------------------------------ refusals
#
# Each doctor edits the unassembled tiled graph of the looped tiling of ``small`` (or the
# record on its max) into the state one refusal is for; ``assemble`` must refuse it with
# its text and edit nothing.


def _record(t, **changes):
    t.max.streaming = dataclasses.replace(t.max.streaming, **changes)


def _axis_outside_the_tile(t):  # L1: the scores' tile carries the queries only
    scores = t.max.inputs[0].src_node
    it = tmap_by_name(t.tier)[scores.name]
    t.tier.tmap[scores] = Iterator(it.offset, shape=(64,), tile_size=(64,), axes=(1,))


def _block_holds_half_the_row(t):  # L2
    _retile(t.tier, t.max.inputs[0].src_node, shape=(64, 128))


def _one_element_steps(t):  # L3
    _retile(t.tier, t.max.inputs[0].src_node, tile=(64, 1))


def _wrong_trip_count(t):  # L4
    t.tier.extent = 4


def _not_float32(t):  # L5
    t.max.out_dtype = "float16"


def _single_row(t):  # L6
    _retile(t.tier, t.max, tile=(1,))


def _reduction_not_in_the_loop(t):  # L7
    _record(t, rules=t.max.streaming.rules + (("nope", X.FACTOR),))


def _accumulator_without_a_rule(t):  # L8
    _record(t, rules=tuple(r for r in t.max.streaming.rules if r[0] != t.value.name))


def _accumulator_starts_elsewhere(t):  # L9
    t.rowsum.accum_init = 1.0


def _already_repaired(t):  # L10
    t.tier.accum_scale[t.rowsum] = next(n for n in t.tier.nodes if n.name.startswith("div_"))


def _rows_do_not_lead(t):  # L11
    _retile(t.tier, t.value, tile=(32, 32))


def _unit_extent_accumulator(t):  # L12
    _retile(t.tier, t.value, tile=(64, 1))


def _read_outside(t):  # L13
    reader = next(n for n in t.tiled.nodes if n.name.startswith("div_"))
    t.exp._out_edges.append(Edge(t.exp, reader, 0, 0))


def _no_tile_op(t):  # L14
    _record(t, rules=tuple((a, ("max", "m", "m'")) for a, _ in t.max.streaming.rules))


REFUSALS = [
    (_axis_outside_the_tile, r"the dim it reduces is not in the tile \(64,\) of its operand"),
    (_block_holds_half_the_row,
     r"the dim it reduces is spread over grid blocks \(128 of 256 per block\); a value "
     r"cannot be carried across blocks"),
    (_one_element_steps,
     r"one element of the reduced dim per iteration; the tile would lose that axis"),
    (_wrong_trip_count,
     r"GraphNode_\w+ iterates 4 times but the reduced dim is stepped 32 of 256"),
    (_not_float32, r"it is float16; a carried value starts at a float32 number"),
    (_single_row, r"its tile \(1,\) holds a single row"),
    (_reduction_not_in_the_loop,
     r"\['nope'\] read\(s\) it but not as accumulator\(s\) of the loop GraphNode_\w+ that "
     r"steps its dim"),
    (_accumulator_without_a_rule,
     r"accumulator mma_\d+ of GraphNode_\w+ is computed from it but has no proven update rule"),
    (_accumulator_starts_elsewhere,
     r"accumulator reduction_\w+ starts at 1.0; a repaired accumulator must start at zero"),
    (_already_repaired,
     r"accumulator reduction_\w+ is already repaired; one repair per accumulator"),
    (_rows_do_not_lead,
     r"accumulator mma_\d+ tile \(32, 32\) does not lead with the row tile \(64,\)"),
    (_unit_extent_accumulator, r"accumulator mma_\d+ has a unit extent in its tile \(64, 1\)"),
    (_read_outside,
     r"exp_\d+, computed from it per iteration, is read by \['div_\d+'\] outside GraphNode_\w+"),
    (_no_tile_op, r"the repair uses 'max', which has no tile op"),
]


def _doctored(a, cand, doctor, monkeypatch):
    tiled = _unassembled(cand, monkeypatch)
    tier, owner, m = _find(tiled, a.max.name)
    t = SimpleNamespace(tiled=tiled, tier=tier, owner=owner, max=m,
                        exp=_find(tiled, a.exp.name)[2])
    for role in ("rowsum", "value"):
        setattr(t, role, _find(tiled, getattr(a, role).name)[2])
    doctor(t)
    return t


def _assert_refused(t, flat_graph, text, name):
    before = _names(t.tiled)
    with pytest.raises(stream.StreamingRefused) as raised:
        stream.assemble(t.tiled, flat_graph)
    assert re.fullmatch(rf"{name}: {text}", str(raised.value)), str(raised.value)
    assert isinstance(raised.value, ScheduleRejected)
    # nothing was edited: same nodes, still locked, the record still there
    assert _names(t.tiled) == before and t.tiled.locked
    tier = _find(t.tiled, name)[0]
    assert not tier.lag_nodes and _find(t.tiled, name)[2].streaming is not None


@pytest.mark.parametrize(("doctor", "text"), REFUSALS,
                         ids=[d.__name__.lstrip("_") for d, _ in REFUSALS])
def test_looped_tiling_refused(small, doctor, text, monkeypatch):
    t = _doctored(small, small.looped, doctor, monkeypatch)
    assert t.owner is not None
    _assert_refused(t, small.ir, text, small.max.name)


def test_every_refusal_text_is_pinned():
    """L1-L14 of the design, each exactly once above."""
    assert len(REFUSALS) == 14 and len({d for d, _ in REFUSALS}) == 14


def test_a_doctored_tiling_is_assembled_once_put_right(small, monkeypatch):
    """The refusals are about the tiling: the same unassembled graph, undoctored, is."""
    t = _doctored(small, small.looped, lambda t: None, monkeypatch)
    assert stream.assemble(t.tiled, small.ir) is t.tiled
    assert [s.full for s in stream.sites(t.tiled)] == [small.max.name]


def test_one_accumulator_repaired_for_two_maxes_is_refused():
    """``sum(pa * pb)`` sits under the maxes of two softmaxes. With a record on both, the
    assembly still keeps one factor per accumulator: the sum would be repaired for one of
    them only (``Graph.accum_scale`` holds one factor), so the tiling is refused."""
    ir = X.streamed_ir(X.two_softmaxes, dict(R=256, C=4096), None, None)
    first, second = X.obstacles_of(ir)
    (shared,) = set(first.loops) & set(second.loops)
    texts, built = Counter(), Counter()
    for cand in search_tiling(ir, topk=1000):
        try:
            tiled = apply_tiling(cand)
        except stream.StreamingRefused as e:
            texts[stream._reason_key(e)] += 1
            continue
        built["looped" if stream.sites(tiled) else "one tile"] += 1
    wanted = {f"{x}: accumulator {shared} is already repaired; one repair per accumulator"
              for x in (first.full, second.full)}
    assert set(texts) & wanted and built == {"one tile": 2}, (texts, built)


def test_a_taken_node_name_is_a_bug_not_a_refusal(small, monkeypatch):
    tiled = _unassembled(small.looped, monkeypatch)
    next(n for n in tiled.nodes if n.name.startswith("div_")).name = f"{small.max.name}_prev"
    with pytest.raises(AssertionError, match="already taken in this graph"):
        stream.assemble(tiled, small.ir)


def test_every_codegen_caller_gets_the_assembly(prefill):
    """The assembly is the tail of ``apply_tiling``, so a caller that runs the tiler and the
    codegen passes itself -- the snapshot mirror does -- emits the same kernel as the
    default build, not a max of one slice."""
    from tests.pipeline._pipeline import _build_code

    code = _build_code(prefill.cand, False)
    assert code == build_candidate_code(prefill.cand)
    assert f"ukernel_rescale_1dims_rr(t_{prefill.max.name}_upd1" in code
