"""The lazily rescaled rolling softmax: ``cleave.schedule.stabilize`` on wide accumulators (CPU).

The rolling form multiplies every accumulator that sums the exp by ``alpha = exp(m_prev - m)``
on every iteration. That is nearly free while a thread keeps its share of the accumulator in
registers, and it is a load, a multiply and a store per element once it does not: a thread has
at most 255 registers. For such a tiling the rewrite builds the *lazy* form instead: the shift
is ``g(m) = 8 * floor(m / 8)``, which moves only when the running max crosses a multiple of 8,
and a thread skips the rescale while its own rows did not move.

Pinned here: which tilings get it (and that an exported max never does), the tier and the
text it builds, and that the kernel lowers without a GPU.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.lang as cl
import cleave.schedule.stabilize as stab
from cleave.ir.graph_utils import tmap_by_name
from cleave.ir.ops import NEG_FLT_MAX, Exp, GridFloor, Matmul, ReduceMax, ReduceSum, Sub
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling

from tests import _softmax as S
from tests._lowering import lower_to_cuda
from tests.graph.apply_tiling.test_stabilize import _find, _loops, _macro_calls, _pick, _steps

WIDE = dict(H=8, SQ=64, SKV=1024, D=64, DV=512)


def wide_value_attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                         v: cl.Tensor["H", "SKV", "DV"]):
    """Flash form whose values are wider than the keys: the accumulator of ``P @ V`` is
    ``SQ x DV``, 256 elements a thread at a 64 x 512 tile over 128 threads."""
    e = S.stable_exp((q @ k.mT) / q.shape[-1] ** 0.5)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


@pytest.fixture(scope="module")
def wide():
    ir = S.softmax_ir(wide_value_attention, WIDE)
    matmuls = [n for n in ir.nodes if type(n) is Matmul]
    rowsum = next(n for n in ir.nodes if type(n) is ReduceSum)
    steps = {matmuls[0].name: 64, rowsum.name: 32, matmuls[-1].name: 32}
    return dict(
        ir=ir, exp=S.exps(ir)[0], rowsum=rowsum, value=matmuls[-1],
        lazy=_pick(ir, [1, 64, 512], threads=128, **steps),
        eager=_pick(ir, [1, 64, 512], threads=256, **steps),
        narrow=_pick(ir, [1, 64, 64], threads=128, **steps),
    )


def test_the_threshold_is_what_a_thread_cannot_hold_in_registers():
    assert stab.LAZY_MIN_ELEMENTS == 256  # a thread has 255 registers at most


def test_only_a_tiling_whose_accumulator_overflows_the_registers_is_lazy(wide):
    """One softmax, three tilings: 64 x 512 over 128 threads is 256 elements a thread and
    gets the lazy form; the same tile over 256 threads (128 a thread) and a 64 x 64 tile do
    not, and neither does any tiling of the D = 64 attention the demos run."""
    sites = {name: stab.sites(apply_tiling(wide[name]))[0] for name in ("lazy", "eager", "narrow")}
    assert all(site.case == "rolling" for site in sites.values())
    assert [sites[name].lazy for name in ("lazy", "eager", "narrow")] == [True, False, False]

    every = {}
    for cand in search_tiling(wide["ir"], topk=1000):
        try:
            (site,) = stab.sites(apply_tiling(cand))
        except stab.ScheduleRejected:
            continue
        if site.case == "rolling":
            every.setdefault((tuple(cand.output_tile), cand.n_threads), set()).add(site.lazy)
    lazy = sorted(key for key, kinds in every.items() if kinds == {True})
    assert lazy == [((1, 64, 512), 128)]
    assert all(kinds == {False} for key, kinds in every.items() if key not in lazy)

    plain = S.softmax_ir(S.flash_attention, dict(H=32, SQ=64, SKV=4096, D=64))
    kinds = set()
    for cand in search_tiling(plain, topk=1000):
        try:
            kinds |= {(s.case, s.lazy) for s in stab.sites(apply_tiling(cand))}
        except stab.ScheduleRejected:
            pass
    assert kinds == {("rolling", False)}


def test_no_thread_count_no_lazy_form(wide):
    """Decided from the kernel's thread count; a tiled graph that does not say stays eager."""
    tiled = apply_tiling(wide["eager"])
    (loop, owner), = _loops(tiled)
    tmaps = tmap_by_name(loop)
    accs = [n for n in loop.accum_nodes if n.name in (wide["rowsum"].name, wide["value"].name)]
    assert stab._lazy(accs, tmaps, 128) is True  # 64 * 512 / 128 = 256
    assert stab._lazy(accs, tmaps, 129) is False
    assert stab._lazy(accs, tmaps, 256) is False and stab._lazy(accs, tmaps, None) is False
    assert stab._lazy([], tmaps, 128) is False


def test_lazy_tier(wide):
    exp, rowsum, value = wide["exp"], wide["rowsum"], wide["value"]
    tiled = apply_tiling(wide["lazy"])
    (loop, owner), = _loops(tiled)
    m, m_prev, grid, step, alpha, shifted = (
        _find(tiled, f"{exp.name}_{suffix}")[2]
        for suffix in ("max", "max_prev", "max_grid", "max_step", "alpha", "shifted")
    )
    tiled_exp = _find(tiled, exp.name)[2]

    # the same declared state as the eager form: one carried max, one lag node, one factor
    assert [n.name for n in loop.accum_nodes] == [rowsum.name, value.name, m.name]
    assert [n.name for n in loop.lag_nodes] == [m_prev.name]
    assert {acc.name: f.name for acc, f in loop.accum_scale.items()} == {
        rowsum.name: alpha.name, value.name: alpha.name}
    assert m.accum_init == NEG_FLT_MAX and type(m) is ReduceMax

    # ... but the shift is g(m), the lag node g of the previous max, and the scores are
    # shifted by g(m): the raw max feeds nothing else
    assert type(m_prev) is GridFloor and type(grid) is GridFloor
    assert m_prev.inputs[0].src_node is m and grid.inputs[0].src_node is m
    assert sorted(e.dst_node.name for e in m.outputs if e.dst_node is not owner) == sorted(
        [m_prev.name, grid.name])
    assert type(shifted) is Sub and shifted.inputs[1].src_node is grid
    assert tiled_exp.inputs[0].src_node is shifted
    assert [e.src_node for e in step.inputs] == [m_prev, grid]
    assert type(alpha) is Exp and alpha.inputs[0].src_node is step and alpha.unit_guard is True
    for node in (m, m_prev, grid, step, alpha, shifted):
        assert node.out_dtype == "float32", node.name
    tmaps = tmap_by_name(loop)
    assert [int(t) for t in tmaps[grid.name].tile_size] == [
        int(t) for t in tmaps[m.name].tile_size]

    site = tiled_exp.stabilized
    assert site == stab.Site(exp.name, owner.name, "rolling", axis=1, full=1024, block=1024,
                             step=32, scaled=(rowsum.name, value.name), lazy=True)
    assert stab.overhead_bytes(tiled) == 4 * 64 * 5
    # the flat graph gains nothing and keeps its record
    assert exp.stabilize is not None
    assert not [n for n in wide["ir"].nodes if n.name.startswith(f"{exp.name}_")]


def test_lazy_kernel_text(wide):
    e, l, o = wide["exp"].name, wide["rowsum"].name, wide["value"].name
    code = build_candidate_code(wide["lazy"])
    calls = [re.sub(r"PlaceHolder_\w+", "P", c) for c in _macro_calls(code, f"GraphNode_{l}_iter")]
    scores = re.search(r"ukernel_reduce_max_axis2_rr\((\w+), ", " ".join(calls)).group(1)
    assert calls[-11:] == [
        f"ukernel_grid_floor_1dims_rr(t_{e}_max, t_{e}_max_prev)",
        f"ukernel_reduce_max_axis2_rr({scores}, t_{e}_max)",
        f"ukernel_grid_floor_1dims_rr(t_{e}_max, t_{e}_max_grid)",
        f"ukernel_sub_broadcast_0_2dims_rrr({scores}, t_{e}_max_grid, t_{e}_shifted)",
        f"ukernel_exp_2dims_rr(t_{e}_shifted, t_{e})",
        f"ukernel_sub_1dims_rrr(t_{e}_max_prev, t_{e}_max_grid, t_{e}_max_step)",
        f"ukernel_exp_1dims_rr(t_{e}_max_step, t_{e}_alpha)",
        f"ukernel_rescale_if_moved_1dims_rr(t_{e}_alpha, t_{l})",
        f"ukernel_reduction_axis2_rr(t_{e}, t_{l})",
        f"ukernel_rescale_if_moved_lead1_2dims_rr(t_{e}_alpha, t_{o})",
        f"ukernel_mma_rsr(t_{e}_cast, P, t_{o})",
    ]
    assert f"T.fill(t_{e}_max, -3.4028234663852886e38)" in code
    assert "def ukernel_rescale_1dims_rr" not in code and "ukernel_copy_1dims_rr" not in code

    # the same softmax where a thread holds half as much: the eager text, no grid, no guard
    eager = build_candidate_code(wide["eager"])
    assert "grid_floor" not in eager and "if_moved" not in eager and "alloc_var" not in eager
    assert f"ukernel_copy_1dims_rr(t_{e}_max, t_{e}_max_prev)" in eager
    assert f"ukernel_rescale_lead1_2dims_rr(t_{e}_alpha, t_{o})" in eager


def test_an_exported_max_is_never_lazy():
    """A kernel that exports its max (a split partial) returns sums in the frame of that
    max: the reference of its three outputs says so. Its accumulators are rescaled eagerly
    however wide they are."""
    ir = S.raw_statistics_graph(8, 64, 1024, 512)
    exp = S.exps(ir)[0]
    assert exp.stabilize.export is not None
    wide_enough = 0
    for cand in search_tiling(ir, topk=1000):
        try:
            tiled = apply_tiling(cand)
        except stab.ScheduleRejected:
            continue
        for site in stab.sites(tiled):
            if site.case != "rolling":
                continue
            assert site.lazy is False, _steps(cand)
            loop = next(tier for tier, owner in _loops(tiled) if owner.name == site.tier)
            accs = [n for n in loop.accum_nodes if n.name in site.scaled]
            wide_enough += stab._lazy(accs, tmap_by_name(loop), cand.n_threads)
    assert wide_enough >= 1  # ... and some of them are wide: it is the export that decides


def test_lazy_kernel_lowers_without_a_gpu(wide):
    e = wide["exp"].name
    cuda = lower_to_cuda(build_candidate_code(wide["lazy"]))
    assert "-0x1.fffffep+127f" in cuda
    # g of the previous max is taken before the max of this tile is folded in
    snapshot = cuda.index(f"t_{e}_max_prev[", cuda.index(f"t_{e}_max_prev[") + 1)
    fold = cuda.index("tl::MaxOp")
    assert snapshot < fold < cuda.index(f"t_{e}_max_grid[", cuda.index(f"t_{e}_max_grid[") + 1)
    assert len(re.findall(r"floorf\(", cuda)) == 2
    # each thread decides for itself, from its own entries of alpha, and both accumulators
    # are multiplied inside the branch
    guards = re.findall(r"if \((low\w*) < 0x1p\+0f", cuda)
    assert len(guards) == 2 and len(set(guards)) == 2
    assert len(re.findall(rf"if \(\(t_{e}_alpha\[\w+\] == 0x1p\+0f", cuda)) == 2  # exactly 1?
    assert len(re.findall(r"= min\(low\w*, condval\w*\)", cuda)) == 2
    assert len(re.findall(rf"\* t_{e}_alpha\[", cuda)) == 2
    first = cuda.index(f"if ({guards[0]} <")
    assert cuda.index(f"expf(t_{e}_max_step[") < first < cuda.index("tl::SumOp")
