"""The stabilization rewrite on tiled graphs: ``cleave.schedule.stabilize`` (CPU).

``apply_tiling`` ends by giving every ``Exp`` that carries a ``Stabilize`` record the max its
tiling needs: in place when the softmax axis is whole in the Exp's tile, carried across
iterations (with the accumulators rescaled) when the tier's loop tiles it. The records are
the scale analysis's own, written at lowering and checked by ``tests/_softmax.py`` against
what each test expects.

Pinned here: the hook and what it leaves alone, both forms and their text, the exported max
of a split-partial-shaped graph, the Exp that read storage, and every refusal.
"""

import dataclasses
import json
from collections import Counter
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from tilelang.carver.roller.node import Edge

import cleave.schedule.stabilize as stab
from cleave.codegen import TilelangCodeGen, passes
from cleave.core.errors import ScheduleRejected
from cleave.core.target import current_target
from cleave.ir.graph import AccumEdge, GraphNode, Iterator
from cleave.ir.graph_utils import edit_graph, graph_tiers, tmap_by_name
from cleave.ir.ops import NEG_FLT_MAX, Exp, Matmul, ReduceMax, ReduceSum, Sub
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling

from tests import _softmax as S
from tests._lowering import lower_to_cuda

ROOT = Path(__file__).resolve().parents[3]
PREFILL = dict(H=32, SQ=4096, SKV=4096, D=64)
SMALL = dict(H=32, SQ=64, SKV=256, D=64)
NEXT_TO_SUM = "div reads the exp next to the finished reduction"


# ------------------------------------------------------------------------------- helpers


def _steps(cand) -> dict:
    return {n.name: int(next(iter(v.values()))) for n, v in cand.rstep_map.items() if v}


def _pick(ir, tile, threads=128, **steps):
    """The candidate with this output tile, thread count and reduction steps (by name)."""
    for cand in search_tiling(ir, topk=1000):
        if list(cand.output_tile) == tile and cand.n_threads == threads and _steps(cand) == steps:
            return cand
    raise AssertionError(f"no candidate tile={tile} threads={threads} steps={steps}")


def _attention(fn, shapes, *softmax):
    """A lowered attention graph and its nodes by role."""
    ir = S.softmax_ir(fn, shapes, *softmax)
    matmuls = [n for n in ir.nodes if type(n) is Matmul]
    return SimpleNamespace(
        ir=ir,
        exp=S.exps(ir)[0],
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


def _pristine(ir, cand):
    """The tiled graph as the tiler alone builds it, with each record put back on the tiled
    Exp: what ``stabilize`` is handed by ``apply_tiling``, open to doctoring."""
    marks = [(e, e.stabilize) for e in S.exps(ir)]
    for e, _ in marks:
        e.stabilize = None
    try:
        tiled = apply_tiling(cand)
    finally:
        for e, record in marks:
            e.stabilize = record
    for e, record in marks:
        _find(tiled, e.name)[2].stabilize = record
    return tiled


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
        tile_size=it.tile_size if tile is None else tile,
    )


@pytest.fixture(scope="module")
def prefill():
    a = _attention(S.flash_attention, PREFILL)
    a.cand = _pick(a.ir, [1, 128, 64],
                   **{a.score.name: 64, a.rowsum.name: 32, a.value.name: 32})
    return a


@pytest.fixture(scope="module")
def small():
    a = _attention(S.flash_attention, SMALL)
    looped = {a.score.name: 64, a.rowsum.name: 32, a.value.name: 32}
    whole = {a.score.name: 64, a.rowsum.name: 256, a.value.name: 256}
    a.rolling = _pick(a.ir, [1, 64, 32], **looped)
    a.in_tile = _pick(a.ir, [1, 64, 32], **whole)
    return a


@pytest.fixture(scope="module")
def exported():
    ir = S.raw_statistics_graph(32, 64, 256, 64)
    exp = S.exps(ir)[0]
    cand = _pick(ir, [1, 64, 32], mma_0=64, mma_1=32, rowsum_0=32, reduce_max_2_exp_0=32)
    return SimpleNamespace(ir=ir, exp=exp, cand=cand)


# ---------------------------------------------------------------------- the hook, and off


def test_a_graph_without_a_record_never_reaches_the_pass(monkeypatch):
    ir = S.lowered(S.flash_attention, SMALL)
    for exp in S.exps(ir):
        exp.stabilize = None

    def tripwire(*args, **kwargs):
        raise AssertionError("the pass ran on a graph without a record")

    monkeypatch.setattr(stab, "stabilize", tripwire)
    cands = search_tiling(ir, topk=1000)
    assert cands
    for cand in cands:
        tiled = apply_tiling(cand)
        for tier, _ in graph_tiers(tiled):
            assert not tier.lag_nodes and not tier.accum_scale
            assert not [n for n in tier.nodes if type(n) in (ReduceMax, Sub)]


def test_the_pass_is_not_even_imported_for_a_graph_without_a_record(tmp_path):
    script = (
        "import sys\n"
        "from cleave.schedule import apply_tiling, search_tiling\n"
        "from tests._graphs import mha_bhsd_graph\n"
        "cands = search_tiling(mha_bhsd_graph(1, 32, 4096, 4096, 64), topk=4)\n"
        "assert cands and all(apply_tiling(c) is not None for c in cands)\n"
        "assert 'cleave.schedule.stabilize' not in sys.modules, 'the pass was imported'\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CLEAVE_CACHE_ROOT=str(tmp_path))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "python"), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_stabilize_returns_an_unmarked_graph_as_it_came(small, monkeypatch):
    tiled = _pristine(small.ir, small.rolling)
    _find(tiled, small.exp.name)[2].stabilize = None
    before = _names(tiled)

    def tripwire(graph):
        raise AssertionError("an unmarked graph was opened for editing")

    monkeypatch.setattr(stab, "edit_graph", tripwire)
    assert stab.stabilize(tiled) is tiled
    assert _names(tiled) == before and tiled.locked


# ------------------------------------------------------------------------------- rolling


def test_rolling_tier(prefill):
    exp, rowsum, value = prefill.exp, prefill.rowsum, prefill.value
    tiled = apply_tiling(prefill.cand)
    (loop, owner), = _loops(tiled)
    m, m_prev, step, alpha, shifted = (
        _find(tiled, f"{exp.name}_{suffix}")[2]
        for suffix in ("max", "max_prev", "max_step", "alpha", "shifted")
    )
    tiled_exp = _find(tiled, exp.name)[2]

    # the carried max is appended: the reductions keep their accumulator positions
    assert [n.name for n in loop.accum_nodes] == [rowsum.name, value.name, m.name]
    assert loop.accum_nodes[2] is m and [n.name for n in loop.lag_nodes] == [m_prev.name]
    assert {acc.name: f.name for acc, f in loop.accum_scale.items()} == {
        rowsum.name: alpha.name, value.name: alpha.name}
    assert all(f is alpha for f in loop.accum_scale.values())
    assert owner.get_shape(2) == m.get_shape(0)

    # wiring: m and the shifted scores read what the exp read; the exp reads the difference
    scores = m.inputs[0].src_node
    assert scores.name == "div_4" and scores.out_dtype == "float32"
    assert type(m) is ReduceMax and type(shifted) is Sub and shifted.broadcast == 0
    assert [e.src_node for e in shifted.inputs] == [scores, m]
    assert tiled_exp.inputs[0].src_node is shifted
    assert [e.src_node for e in step.inputs] == [m_prev, m] and m_prev.inputs[0].src_node is m
    assert alpha.inputs[0].src_node is step and type(alpha) is Exp
    for node in (m, m_prev, step, alpha, shifted):
        assert node.out_dtype == "float32", node.name

    # the sentinel is this instance's start value; the class default is still the identity
    assert m.accum_init == NEG_FLT_MAX and ReduceMax.accum_init == -float("inf")
    assert all(getattr(acc, "accum_init", None) is None for acc in loop.accum_nodes[:2])

    # the request is consumed on the tiled node; the flat graph keeps it and gains nothing
    site = tiled_exp.stabilized
    assert tiled_exp.stabilize is None and exp.stabilize is not None
    assert site == stab.Site(exp.name, owner.name, "rolling", axis=1, full=4096, block=4096,
                             step=32, scaled=(rowsum.name, value.name))
    assert stab.sites(tiled) == [site]
    assert not [n for n in prefill.ir.nodes if n.name.startswith(f"{exp.name}_")]
    assert stab.overhead_bytes(tiled) == 4 * 128 * 4


def test_rolling_kernel_text(prefill):
    e, l, o = prefill.exp.name, prefill.rowsum.name, prefill.value.name
    code = build_candidate_code(prefill.cand)
    # per iteration: snapshot, fold the tile's max in, shift, and rescale each accumulator
    # right before it accumulates
    calls = [re.sub(r"PlaceHolder_\w+", "P", c) for c in _macro_calls(code, f"GraphNode_{l}_iter")]
    assert calls[2:] == [
        f"ukernel_copy_1dims_rr(t_{e}_max, t_{e}_max_prev)",
        f"ukernel_reduce_max_axis2_rr(t_div_4, t_{e}_max)",
        f"ukernel_sub_broadcast_0_2dims_rrr(t_div_4, t_{e}_max, t_{e}_shifted)",
        f"ukernel_exp_2dims_rr(t_{e}_shifted, t_{e})",
        f"ukernel_sub_1dims_rrr(t_{e}_max_prev, t_{e}_max, t_{e}_max_step)",
        f"ukernel_exp_1dims_rr(t_{e}_max_step, t_{e}_alpha)",
        f"ukernel_rescale_1dims_rr(t_{e}_alpha, t_{l})",
        f"ukernel_reduction_axis2_rr(t_{e}, t_{l})",
        f"ukernel_rescale_lead1_2dims_rr(t_{e}_alpha, t_{o})",
        f"ukernel_mma_rsr(t_{e}_cast, P, t_{o})",
    ]
    # the parent tier fills the carried max with the sentinel and clears the sums
    main = _calls(code, "main_kernel_iter")
    assert f"T.fill(t_{e}_max, -3.4028234663852886e38)" in main
    assert f"T.clear(t_{l})" in main and f"T.clear(t_{o})" in main
    assert "infinity" not in code
    assert f"t_{e}_max: T.Tensor((128), \"float32\")" in code


def test_rolling_over_the_first_non_batch_axis():
    ir = S.softmax_ir(S.column_average, dict(H=4, K=4096, N=256), dict(axis=-2))
    (exp,) = S.exps(ir)
    sums = [n for n in ir.nodes if type(n) is ReduceSum]
    cand = _pick(ir, [1, 64], **{n.name: 32 for n in sums})
    tiled = apply_tiling(cand)
    loop, owner, m = _find(tiled, f"{exp.name}_max")
    shifted = _find(tiled, f"{exp.name}_shifted")[2]
    site = _find(tiled, exp.name)[2].stabilized

    # flat axis 1 of (H, K, N) is axis 0 of the loop's (K-slice, N) tile
    assert (site.case, site.axis) == ("rolling", 0)
    assert (site.full, site.block, site.step) == (4096, 4096, 32)
    assert (m.axis, m.axis_arg) == (0, 0) and shifted.broadcast == 1
    assert tuple(tmap_by_name(loop)[m.name].tile_size) == (64,)
    code = build_candidate_code(cand)
    calls = _macro_calls(code, f"{owner.name}_iter")
    assert any(c.startswith("ukernel_reduce_max_axis0_rr(") for c in calls)
    assert any(c.startswith("ukernel_sub_broadcast_1_2dims_rrr(") for c in calls)
    assert sum(c.startswith("ukernel_rescale_1dims_rr(") for c in calls) == 2


def test_exported_max_is_carried_not_duplicated(exported):
    tiled = apply_tiling(exported.cand)
    (loop, owner), = _loops(tiled)
    name = exported.exp.stabilize.export
    m = _find(tiled, name)[2]

    assert not [n for t, _ in graph_tiers(tiled) for n in t.nodes if n.name == "exp_0_max"]
    assert [n.name for n in loop.accum_nodes] == [name, "mma_1", "rowsum_0"]
    assert m.accum_init == NEG_FLT_MAX and m.inputs[0].src_node.name == "mul_0"
    assert [e.src_node for e in _find(tiled, "exp_0_shifted")[2].inputs][1] is m
    assert _find(tiled, "exp_0_max_prev")[2].inputs[0].src_node is m
    assert {acc.name for acc in loop.accum_scale} == {"mma_1", "rowsum_0"}
    # all three statistics still leave the kernel, the shift in float32
    outputs = [n for n in tiled.nodes if n.is_output()]
    assert [str(n.get_dtype(0)) for n in outputs] == ["float16", "float16", "float32"]
    code = build_candidate_code(exported.cand)
    main = _calls(code, "main_kernel_iter")
    assert f"T.fill(t_{name}, -3.4028234663852886e38)" in main
    assert f"ukernel_copy_1dims_rr(t_{name}, t_exp_0_max_prev)" in _macro_calls(
        code, f"{owner.name}_iter")


# ------------------------------------------------------------------------------- in-tile


def test_in_tile(small):
    exp = small.exp
    tiled = apply_tiling(small.in_tile)
    tier, owner, m = _find(tiled, f"{exp.name}_max")
    shifted = _find(tiled, f"{exp.name}_shifted")[2]
    site = _find(tiled, exp.name)[2].stabilized

    assert tier is tiled and owner is None  # the whole row is in the block's tile
    assert site == stab.Site(exp.name, "main", "in-tile", axis=2, full=256, block=256, step=256)
    assert getattr(m, "accum_init") == -float("inf") and "accum_init" not in vars(m)
    assert shifted.broadcast == 0 and [e.src_node for e in shifted.inputs][1] is m
    for t, _ in graph_tiers(tiled):
        assert not t.lag_nodes and not t.accum_scale
        assert not [n for n in t.nodes if n.name.endswith(("_max_prev", "_alpha"))]
    assert stab.overhead_bytes(tiled) == 4 * 64

    main = _macro_calls(build_candidate_code(small.in_tile), "main_kernel_iter")
    e = exp.name
    at = main.index(f"ukernel_reduce_max_axis2_rr_clear(t_div_4, t_{e}_max)")
    assert main[at + 1: at + 3] == [
        f"ukernel_sub_broadcast_0_2dims_rrr(t_div_4, t_{e}_max, t_{e}_shifted)",
        f"ukernel_exp_2dims_rr(t_{e}_shifted, t_{e})",
    ]
    assert "T.fill" not in "".join(main) and "rescale" not in "".join(main)


def test_in_tile_reads_a_loop_accumulator_through_an_accum_edge():
    """``softmax(q @ k.mT) @ v``: with QK^T looped over D the Exp reads an accumulator of
    that loop. The max and the shifted scores must read it the same way."""
    a = _attention(S.unscaled_attention, SMALL, dict(rolling=NEXT_TO_SUM))
    cands = search_tiling(a.ir, topk=1000)
    built = []
    for cand in cands:
        try:
            built.append((cand, apply_tiling(cand)))
        except ScheduleRejected:
            pass
    # (64 query rows over 256 threads are no candidates: the value matmul would read 8 rows
    # a warp from the exp's registers, TCPolicy._register_operand_layout_conflict)
    assert len(built) == 4 and len(cands) == 16
    through_loop = 0
    for cand, tiled in built:
        m = _find(tiled, f"{a.exp.name}_max")[2]
        shifted = _find(tiled, f"{a.exp.name}_shifted")[2]
        assert _find(tiled, a.exp.name)[2].stabilized.case == "in-tile"
        edges = [m.inputs[0], shifted.inputs[0]]
        if isinstance(edges[0].src_node, GraphNode):
            through_loop += 1
            for edge in edges:
                assert isinstance(edge, AccumEdge) and edge.accum_node.name == a.score.name
                assert any(edge is e for e in edge.src_node.outputs)
        assert "ukernel_reduce_max_axis2_rr_clear(" in build_candidate_code(cand)
    assert through_loop == 2


def test_in_tile_over_the_first_axis():
    ir = S.softmax_ir(S.column_average_2d, dict(K=256, N=1024), dict(axis=-2))
    (exp,) = S.exps(ir)
    sums = [n for n in ir.nodes if type(n) is ReduceSum]
    cand = _pick(ir, [32], **{n.name: 256 for n in sums})
    tiled = apply_tiling(cand)
    tier, owner, m = _find(tiled, f"{exp.name}_max")
    assert owner is None and (m.axis, m.axis_arg) == (0, 0)
    assert _find(tiled, f"{exp.name}_shifted")[2].broadcast == 1
    assert _find(tiled, exp.name)[2].stabilized.case == "in-tile"
    main = _macro_calls(build_candidate_code(cand), "main_kernel_iter")
    assert main[0].startswith("ukernel_reduce_max_axis0_gr_clear(")
    assert main[1].startswith("ukernel_sub_broadcast_1_2dims_rrr(")


def test_exported_max_is_reused_in_tile(small):
    """A max the record names, in the tier of an Exp whose row is whole in its tile, is
    taken as the shift: found by name, no second max. (The tiler builds no such tier for a
    multi-output graph today, so the max is put into a single-output one by hand.)"""
    tiled = _pristine(small.ir, small.in_tile)
    tier, _, exp = _find(tiled, small.exp.name)
    with edit_graph(tiled):
        m = ReduceMax(tuple(exp.get_shape()), axis=-1, name="shift_out", dtype="float32",
                      out_dtype="float32")
        stab._read(exp.inputs[0], m, 0)
        tier.tmap[m] = stab._drop_axis(tmap_by_name(tier)[exp.name], 2)
        tier.nodes.append(m)
    exp.stabilize = dataclasses.replace(exp.stabilize, export="shift_out")

    stab.stabilize(tiled)
    assert exp.stabilized.case == "in-tile" and "accum_init" not in vars(m)
    assert not [n for n in tiled.nodes if n.name == f"{exp.name}_max"]
    assert [e.src_node for e in _find(tiled, f"{exp.name}_shifted")[2].inputs][1] is m
    main = _macro_calls(_code(tiled), "main_kernel_iter")
    e = exp.name
    at = main.index("ukernel_reduce_max_axis2_rr_clear(t_div_4, t_shift_out)")
    assert main[at + 1] == f"ukernel_sub_broadcast_0_2dims_rrr(t_div_4, t_shift_out, t_{e}_shifted)"
    assert sum("reduce_max" in c.split("(")[0] for c in main) == 1


# ------------------------------------------------------------ the tiers the tiler builds

COLUMN = dict(H=4, K=4096, N=256)


@pytest.mark.parametrize(
    ("fn", "shapes", "softmax", "cases"),
    [
        (S.flash_attention, SMALL, (), {"rolling", "in-tile"}),
        (S.column_average, COLUMN, (dict(axis=-2),), {"rolling", "in-tile"}),
        (S.column_average_2d, dict(K=4096, N=1024), (dict(axis=-2),), {"rolling", "in-tile"}),
    ],
    ids=["last_axis", "first_axis_behind_a_batch_dim", "first_axis"],
)
def test_the_max_is_emitted_with_the_dim_of_the_sums_next_to_it(fn, shapes, softmax, cases):
    """The max the pass adds is a reduction over the softmax axis like the sums in its
    tier, and gets its emitted ``dim`` the way the tiler gives them theirs
    (``rebase_reduction_axis``, on the tile the scores are stored in): in every tiling that
    builds, rolling or in place, the two are the same."""
    ir = S.softmax_ir(fn, shapes, *softmax)
    (exp,) = S.exps(ir)
    seen = set()
    for cand in search_tiling(ir, topk=1000):
        try:
            tiled = apply_tiling(cand)
        except ScheduleRejected:
            continue
        tier, _, m = _find(tiled, f"{exp.name}_max")
        sums = [n for n in tier.nodes if type(n) is ReduceSum
                and (n.flat_axis, n.flat_axis_arg) == (m.flat_axis, m.flat_axis_arg)]
        assert sums, _names(tiled)
        for s in sums:
            assert (m.axis, m.axis_arg) == (s.axis, s.axis_arg), (cand.output_tile, s.name)
        seen.add(_find(tiled, exp.name)[2].stabilized.case)
    assert seen == cases


def test_a_loop_tile_of_one_column_is_refused_as_a_single_row():
    """A loop tier carries the dims its block stores. Where a block holds one column of a
    column softmax, the tile of the exp is its softmax axis alone. The axis is found where
    the tile says it is (``Iterator.position``), not counted from the end of a tile that
    has no trailing dim, so the refusal says what is wrong with the tiling."""
    ir = S.softmax_ir(S.column_average, COLUMN, dict(axis=-2))
    (exp,) = S.exps(ir)
    reasons, single = Counter(), None
    for cand in search_tiling(ir, topk=1000):
        try:
            apply_tiling(cand)
        except ScheduleRejected as e:
            reasons[stab._reason_key(e)] += 1
            if "(N,) holds a single row" in stab._reason_key(e):
                single = single or cand
    e = exp.name
    assert reasons == {
        f"{e}: its tile (N,) holds a single row": 12,
        f"{e}: its tile (N, N, N) holds a single row": 1,
        f"{e}: one element of the softmax axis per iteration; the tile would lose that axis": 8,
    }
    tier, owner, node = _find(_pristine(ir, single), e)
    it = tmap_by_name(tier)[e]
    assert owner is not None and it.axes == (1,) and len(it.tile_size) == 1
    assert it.position(1, 3) == 0 and it.position(2, 3) is None
    with pytest.raises(stab.StabilizeRefused, match=r"its tile \(\d+,\) holds a single row"):
        stab._locate(tier, owner, node)


def test_added_nodes_say_which_dims_their_tiles_carry(small):
    """A loop tier's ``Iterator`` records the dims it carries (``axes``). The nodes the
    pass adds keep that record: the shifted scores carry what the exp carries, the row
    vectors the same dims without the softmax axis, renumbered past it."""
    column = S.softmax_ir(S.column_average, COLUMN, dict(axis=-2))
    sums = [n for n in column.nodes if type(n) is ReduceSum]
    rolling = [
        (small.ir, small.rolling),  # (H, SQ, SKV): the softmax axis is the last one
        (column, _pick(column, [1, 64], **{n.name: 32 for n in sums})),  # (H, K, N): K
    ]
    for ir, cand in rolling:
        (exp,) = S.exps(ir)
        tiled = apply_tiling(cand)
        tier, owner, _ = _find(tiled, exp.name)
        its = tmap_by_name(tier)
        assert owner is not None and its[exp.name].axes == (1, 2)
        assert its[f"{exp.name}_shifted"].axes == (1, 2)
        for stat in ("max", "max_prev", "max_step", "alpha"):
            it = its[f"{exp.name}_{stat}"]
            # the remaining non-batch dim is dim 1 of the row vector either way
            assert it.axes == (1,) and it.position(1, 2) == 0 and it.position(0, 2) is None
    # the main tier's tiles are over every dim and record nothing
    tiled = apply_tiling(small.in_tile)
    tier, owner, _ = _find(tiled, small.exp.name)
    assert owner is None
    assert all(it.axes is None for name, it in tmap_by_name(tier).items()
               if name.startswith(small.exp.name))


# ---------------------------------------------------------- the Exp that read fp16 storage


@pytest.mark.parametrize(
    ("fn", "shapes", "softmax", "tile", "step"),
    [
        (S.softmax_rows, dict(R=64, C=1024), (dict(rolling=NEXT_TO_SUM),), [2, 1024], 1024),
        (S.flash_rows, dict(R=256, C=4096, D=64), (), [128, 64], 32),
    ],
    ids=["in-tile", "rolling"],
)
def test_exp_that_read_storage_is_replaced(fn, shapes, softmax, tile, step, monkeypatch):
    ir = S.softmax_ir(fn, shapes, *softmax)
    (exp,) = S.exps(ir)
    assert exp.dtype == "float16"
    reducers = [n for n in ir.nodes if type(n) in (ReduceSum, Matmul)]
    cand = _pick(ir, tile, **{n.name: step for n in reducers})

    tiled = _pristine(ir, cand)
    old = _find(tiled, exp.name)[2]
    stab.stabilize(tiled)
    tier, _, new = _find(tiled, exp.name)
    # a new node built for float32 input, same identity; nothing still points at the old one
    assert new is not old and type(new) is Exp and new == old
    assert new.dtype == "float32" and str(new.input_buffers[0].dtype) == "float32"
    assert new.out_dtype == exp.out_dtype and new.stabilized.exp == exp.name
    assert old.dtype == "float16" and not old.inputs and not old.outputs
    assert sum(n is old for t, _ in graph_tiers(tiled) for n in t.nodes) == 0
    assert all(k is not old for k in tier.tmap) and tier.tmap[new] is not None
    assert all(e.src_node is new for e in new.outputs)
    assert new.inputs[0].src_node.name == f"{exp.name}_shifted"
    assert exp.dtype == "float16"  # the flat graph's node is not this one
    replaced = _code(tiled)

    # the variant every prototype used: dtype set in place, the node kept
    rewrite = stab._rewrite

    def in_place(tier, owner, node, site, taken, cache=True):
        node.dtype = "float32"
        return rewrite(tier, owner, node, site, taken, cache)

    monkeypatch.setattr(stab, "_rewrite", in_place)
    tiled = _pristine(ir, cand)
    old = _find(tiled, exp.name)[2]
    stab.stabilize(tiled)
    assert _find(tiled, exp.name)[2] is old
    assert _code(tiled) == replaced


# ---------------------------------------------------------------------------- properties


@pytest.mark.parametrize("which", ["rolling", "in_tile"])
def test_idempotent(small, which):
    cand = getattr(small, which)
    tiled = apply_tiling(cand)
    before = _names(tiled)
    assert stab.stabilize(tiled) is tiled and _names(tiled) == before
    assert _code(tiled) == build_candidate_code(cand)


@pytest.mark.parametrize("which", ["rolling", "in_tile"])
def test_clone_generates_the_same_text(small, which):
    tiled = apply_tiling(getattr(small, which))
    clone = tiled.clone()
    assert _code(clone) == _code(tiled)


_SEED_SCRIPT = """
import hashlib, json
from cleave.core.errors import ScheduleRejected
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import search_tiling
from tests import _softmax as S

out = {}
for name, fn, shapes, softmax in (
    ("attention", S.flash_attention, dict(H=32, SQ=64, SKV=256, D=64), ()),
    ("rows", S.flash_rows, dict(R=256, C=4096, D=64), ()),
    ("softmax", S.softmax_rows, dict(R=64, C=1024),
     (dict(rolling="div reads the exp next to the finished reduction"),)),
):
    for i, cand in enumerate(search_tiling(S.softmax_ir(fn, shapes, *softmax), topk=1000)):
        try:
            code = build_candidate_code(cand)
        except ScheduleRejected as e:
            out[f"{name} {i}"] = f"refused: {e}"
            continue
        kind = "rolling" if "ukernel_rescale" in code else "in-tile"
        out[f"{name} {i}"] = kind + " " + hashlib.sha256(code.encode()).hexdigest()
print(json.dumps(out))
"""


def test_text_does_not_depend_on_the_hash_seed(tmp_path):
    """The pass edits a locked graph, whose node-keyed dicts hash placeholders and loop
    nodes by structure: an edit that leaves one of them keyed by a stale hash loses the
    entry, or not, depending on the process hash seed. Every candidate of three graphs --
    an attention whose Exp reads a float32 tile, and two whose Exp read storage and is
    replaced, one looped and one not -- generates the same text, or is refused alike,
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
    served = {"attention": {"rolling", "in-tile"}, "rows": {"rolling"}, "softmax": {"in-tile"}}
    for graph, kinds in served.items():
        got = {v.split()[0] for k, v in first.items() if k.startswith(graph)}
        assert got - {"refused:"} == kinds, (graph, got)


def test_parts_are_cloned_from_a_template_that_stays_untouched():
    ir = S.softmax_ir(S.column_average, dict(H=4, K=4096, N=256), dict(axis=-2))
    (exp,) = S.exps(ir)
    sums = [n for n in ir.nodes if type(n) is ReduceSum]
    cand = _pick(ir, [1, 64], **{n.name: 32 for n in sums})

    stab._part_templates.cache_clear()
    first, second = apply_tiling(cand), apply_tiling(cand)
    info = stab._part_templates.cache_info()
    assert (info.misses, info.hits) == (1, 1)
    key = (current_target(), Exp, exp.name, (4, 4096, 256), 1, -2, "float16", "float32",
           True, True)
    template = stab._part_templates(*key)
    assert stab._part_templates.cache_info().misses == 1  # that was the cached graph
    t_max = next(n for n in template.nodes if n.name == f"{exp.name}_max")
    # each rewrite got its own nodes; what it changed on them never reached the template
    a, b = (_find(t, f"{exp.name}_max")[2] for t in (first, second))
    assert a is not b and a is not t_max and b is not t_max
    assert (a.axis_arg, t_max.axis_arg) == (0, 1) and "accum_init" not in vars(t_max)
    assert a.accum_init == NEG_FLT_MAX
    real = {id(n) for t in (first, second) for tier, _ in graph_tiers(t) for n in tier.nodes}
    for node in template.nodes:
        assert id(node) not in real
        assert all(id(e.dst_node) not in real and id(e.src_node) not in real
                   for e in list(node.inputs) + list(node.outputs))

    # and the text is what building the ops afresh gives
    cached = _code(first)
    fresh = _pristine(ir, cand)
    stab.stabilize(fresh, cache=False)
    assert stab._part_templates.cache_info().misses == 1
    assert _code(fresh) == cached


# ------------------------------------------------------------------------------ lowering


def test_rolling_kernel_lowers_without_a_gpu(prefill):
    e = prefill.exp.name
    cuda = lower_to_cuda(build_candidate_code(prefill.cand))
    # the carried max starts at the most negative finite float32, not at -inf
    assert "-0x1.fffffep+127f" in cuda
    # the snapshot is taken before the max of this tile is folded in ...
    snapshot = cuda.index(f"t_{e}_max_prev[")
    snapshot = cuda.index(f"t_{e}_max_prev[", snapshot + 1)  # past the declaration
    fold = cuda.index("tl::MaxOp")
    assert snapshot < fold
    # ... and both accumulators are multiplied by alpha before this tile is added to them
    alpha = cuda.index(f"t_{e}_alpha[", cuda.index(f"expf(t_{e}_max_step["))
    assert fold < alpha < cuda.index("tl::SumOp")
    assert len(re.findall(rf"\* t_{e}_alpha\[", cuda)) == 2


def test_in_tile_kernel_lowers_without_a_gpu(small):
    cuda = lower_to_cuda(build_candidate_code(small.in_tile))
    assert cuda.index("tl::MaxOp") < cuda.index("expf(") < cuda.index("tl::SumOp")
    assert "0x1.fffffep+127f" not in cuda


def test_exported_max_kernel_lowers_without_a_gpu(exported):
    cuda = lower_to_cuda(build_candidate_code(exported.cand))
    assert "-0x1.fffffep+127f" in cuda
    assert len(re.findall(r"\* t_exp_0_alpha\[", cuda)) == 2


# ------------------------------------------------------------------------------ refusals


def _record(t, **changes):
    t.exp.stabilize = dataclasses.replace(t.exp.stabilize, **changes)


def _too_many_axes(t):
    _record(t, axes=(-1, -2))


def _batch_axis(t):
    _record(t, axes=(-3,))


def _three_non_batch_dims(t):
    t.exp.n_batch_dims = 0


def _exp_accumulates(t):
    t.tier.accum_nodes.append(t.exp)


def _axis_outside_the_tile(t):
    _record(t, axes=(-2,))
    t.tier.tmap[t.exp] = Iterator(lambda i: ("0",), shape=(256,), tile_size=(32,))


def _block_holds_half_the_row(t):
    _retile(t.tier, t.exp, shape=(64, 128))


def _single_row(t):
    _retile(t.tier, t.exp, tile=(1, 32))


def _export_in_a_loop_that_holds_the_row(t):
    _retile(t.tier, t.exp, tile=(64, 256))
    _record(t, export="m")


def _one_element_steps(t):
    _retile(t.tier, t.exp, tile=(64, 1))


def _no_rolling_form(t):
    _record(t, rolling="x reads the exp next to the finished reduction y")


def _wrong_trip_count(t):
    t.tier.extent = 4


def _read_outside(t):
    reader = next(n for n in t.tiled.nodes if n.name.startswith("div_"))
    t.exp._out_edges.append(Edge(t.exp, reader, 0, 0))


def _reduction_not_in_the_loop(t):
    _record(t, scaled=t.exp.stabilize.scaled + (("nope", 1),))


def _accumulator_not_analysed(t):
    _record(t, scaled=((t.rowsum.name, 1),))


def _degree_two(t):
    _record(t, scaled=((t.rowsum.name, 2), (t.value.name, 1)))


def _accumulator_starts_elsewhere(t):
    t.rowsum.accum_init = 1.0


def _already_rescaled(t):
    t.tier.accum_scale[t.rowsum] = next(n for n in t.tier.nodes if n.name.startswith("div_"))


def _rows_do_not_lead(t):
    _retile(t.tier, t.value, tile=(32, 32))


def _unit_extent_accumulator(t):
    _retile(t.tier, t.value, tile=(64, 1))


ROLLING_REFUSALS = [
    (_too_many_axes, r"softmax over 2 axes; one softmax axis is supported"),
    (_batch_axis, r"softmax axis -3 is not a non-batch dim of the exp \(32, 64, 256\)"),
    (_three_non_batch_dims, r"3 non-batch dims; the shift is broadcast over exactly one"),
    (_exp_accumulates, r"the exp is itself an accumulator of GraphNode_\w+"),
    (_axis_outside_the_tile, r"softmax axis -2 is not in its tile \(32,\)"),
    (_block_holds_half_the_row,
     r"the softmax axis is spread over grid blocks \(128 of 256 per block\)"),
    (_single_row, r"its tile \(1, 32\) holds a single row"),
    (_export_in_a_loop_that_holds_the_row,
     r"exported max 'm' would accumulate over a loop that does not tile the softmax axis"),
    (_one_element_steps, r"one element of the softmax axis per iteration"),
    (_no_rolling_form,
     r"no rolling form: x reads the exp next to the finished reduction y"),
    (_wrong_trip_count,
     r"GraphNode_\w+ iterates 4 times but the softmax axis is stepped 32 of 256"),
    (_read_outside, r"read by \['div_\d+'\] outside the loop that tiles its softmax axis"),
    (_reduction_not_in_the_loop,
     r"\['nope'\] sum\(s\) the softmax axis but not as accumulator\(s\) of the loop"),
    (_accumulator_not_analysed,
     r"accumulator mma_\d+ depends on the exp but is not one of its analysed reductions"),
    (_degree_two, r"accumulator reduction_\w+ has degree 2"),
    (_accumulator_starts_elsewhere,
     r"accumulator reduction_\w+ starts at 1.0; a rescaled accumulator must start at zero"),
    (_already_rescaled, r"accumulator reduction_\w+ is already rescaled by div_\d+"),
    (_rows_do_not_lead,
     r"accumulator mma_\d+ tile \(32, 32\) does not lead with the row tile \(64,\)"),
    (_unit_extent_accumulator, r"accumulator mma_\d+ has a unit extent in its tile \(64, 1\)"),
]


def _doctored(a, cand, doctor):
    tiled = _pristine(a.ir, cand)
    tier, owner, exp = _find(tiled, a.exp.name)
    t = SimpleNamespace(tiled=tiled, tier=tier, owner=owner, exp=exp)
    for role in ("rowsum", "value"):
        if hasattr(a, role):
            setattr(t, role, _find(tiled, getattr(a, role).name)[2])
    doctor(t)
    return t


def _assert_refused(t, text, name):
    before = _names(t.tiled)
    with pytest.raises(stab.StabilizeRefused) as raised:
        stab.stabilize(t.tiled)
    assert re.fullmatch(rf"{name}: {text}.*", str(raised.value)), str(raised.value)
    assert isinstance(raised.value, ScheduleRejected)
    # nothing was edited: same nodes, still locked, the record still there
    assert _names(t.tiled) == before and t.tiled.locked
    assert _find(t.tiled, name)[2].stabilize is not None


@pytest.mark.parametrize(("doctor", "text"), ROLLING_REFUSALS,
                         ids=[d.__name__.lstrip("_") for d, _ in ROLLING_REFUSALS])
def test_looped_tiling_refused(small, doctor, text):
    t = _doctored(small, small.rolling, doctor)
    assert t.owner is not None
    _assert_refused(t, text, small.exp.name)


def _block_holds_half_the_row_main(t):
    _retile(t.tier, t.exp, tile=(1, 64, 128))


def _single_row_main(t):
    _retile(t.tier, t.exp, tile=(1, 1, 256))


MAIN_REFUSALS = [
    (_block_holds_half_the_row_main,
     r"the softmax axis is spread over grid blocks \(128 of 256 per block\)"),
    (_single_row_main, r"its tile \(1, 1, 256\) holds a single row"),
]


@pytest.mark.parametrize(("doctor", "text"), MAIN_REFUSALS,
                         ids=[d.__name__.lstrip("_") for d, _ in MAIN_REFUSALS])
def test_unlooped_tiling_refused(small, doctor, text):
    t = _doctored(small, small.in_tile, doctor)
    assert t.owner is None
    _assert_refused(t, text, small.exp.name)


def _export_missing(t):
    _record(t, export="nope")


def _export_is_a_sum(t):
    _record(t, export="rowsum_0")


def _export_reads_something_else(t):
    stab._wire(_find(t.tiled, "mma_0")[2], t.max, 0)


def _export_reduces_another_axis(t):
    # The axis the max was built with. Its `axis_arg` is the emitted `dim`, which the
    # tiler re-counts on the stored tile and the pass does not read.
    t.max.flat_axis = 0


def _export_is_not_float32(t):
    t.max.out_dtype = "float16"


def _export_is_not_carried(t):
    t.tier.accum_nodes.remove(t.max)


EXPORT_REFUSALS = [
    (_export_missing, r"exported max 'nope' is not in GraphNode_\w+, the tier of the exp"),
    (_export_is_a_sum, r"exported max 'rowsum_0' is a ReduceSum, not a ReduceMax"),
    (_export_reads_something_else,
     r"exported max 'reduce_max_2_exp_0' does not read the operand of the exp"),
    (_export_reduces_another_axis,
     r"exported max 'reduce_max_2_exp_0' reduces axis 0, the softmax axis is 2"),
    (_export_is_not_float32, r"exported max 'reduce_max_2_exp_0' is float16, not float32"),
    (_export_is_not_carried,
     r"exported max 'reduce_max_2_exp_0' is not an accumulator of GraphNode_\w+"),
]


@pytest.mark.parametrize(("doctor", "text"), EXPORT_REFUSALS,
                         ids=[d.__name__.lstrip("_") for d, _ in EXPORT_REFUSALS])
def test_exported_max_refused(exported, doctor, text):
    tiled = _pristine(exported.ir, exported.cand)
    tier, owner, exp = _find(tiled, "exp_0")
    t = SimpleNamespace(tiled=tiled, tier=tier, owner=owner, exp=exp,
                        max=_find(tiled, "reduce_max_2_exp_0")[2])
    doctor(t)
    _assert_refused(t, text, "exp_0")


def test_one_accumulator_scaled_by_two_exps_is_refused():
    """Two softmaxes whose records both claim the shared sum, with the obstacle the analysis
    writes for that taken away: the pass itself still keeps one factor per accumulator."""
    obstacle = "the shared sum is also scaled by the other exp"
    ir = S.softmax_ir(S.two_softmaxes, dict(R=256, C=4096),
                      dict(rolling=obstacle), dict(rolling=obstacle))
    first, second = S.exps(ir)
    for exp in (first, second):
        exp.stabilize = dataclasses.replace(exp.stabilize, rolling="")
    shared = set(dict(first.stabilize.scaled)) & set(dict(second.stabilize.scaled))
    assert len(shared) == 1
    texts = set()
    for cand in search_tiling(ir, topk=1000):
        try:
            tiled = apply_tiling(cand)
        except stab.StabilizeRefused as e:
            texts.add(stab._reason_key(e))
            continue
        assert {s.case for s in stab.sites(tiled)} == {"in-tile"}
    (name,) = shared
    wanted = {
        f"{x}: accumulator {name} is also scaled by {y}; the rolling form keeps one factor "
        "per accumulator"
        for x, y in ((first.name, second.name), (second.name, first.name))
    }
    assert texts & wanted, texts


def test_a_taken_node_name_is_a_bug_not_a_refusal(small):
    tiled = _pristine(small.ir, small.rolling)
    next(n for n in tiled.nodes if n.name.startswith("div_")).name = f"{small.exp.name}_shifted"
    with pytest.raises(AssertionError, match="already taken in this graph"):
        stab.stabilize(tiled)


def test_every_codegen_caller_gets_the_rewrite(prefill):
    """The pass is the tail of ``apply_tiling``, so a caller that runs the tiler and the
    codegen passes itself -- the snapshot mirror does -- emits the same stabilized kernel
    as the default build, not a plain exp against a stabilized reference."""
    from tests.pipeline._pipeline import _build_code

    code = _build_code(prefill.cand, False)
    assert code == build_candidate_code(prefill.cand)
    assert f"ukernel_rescale_1dims_rr(t_{prefill.exp.name}_alpha" in code
