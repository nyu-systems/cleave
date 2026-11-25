"""Softmax graphs that are legal but must not get a looped kernel (CPU).

Each program below keeps its max in the flat graph, so a tiling that holds the whole row in
one tile is served by the flat graph's own max (which may also run a loop of its own ahead
of what reads it: two passes). None of them has a carried form, each for a
different reason, and for each of them an early version of the hand-derived pass built
rolling kernels that were wrong:

``two_softmaxes``      one sum sits under two maxes; one factor per accumulator (those
                       versions rescaled the shared sum for one of them).
``column_weighted``    the max is indexed by the accumulator's TRAILING axis
                       (``O[m, n] *= alpha[m]`` was emitted for a factor indexed by ``n``).
``self_weighted``      the same with ``v = x^T``: one input dim is the accumulator's rows
                       and the max's, and the factor still belongs to the trailing axis
                       (the assembly built ``ACC[i0, i1] *= S[i0]`` for it).
``max_normalized``     a max of the shifted exp: its accumulator starts at ``-inf``.
``sum_next_to_exp``    the exp is read next to its own finished row sum.
``unscaled_attention`` traced order, the max and the exp reading the QK^T matmul itself.

Every candidate goes through the default build. Anything but a ``ScheduleRejected``
propagates: a refused tiling is a result, a crash is a bug.

Who refuses is pinned per case as a histogram of reasons. The classification refuses the
last three (T5, T3, T3): no record is ever written for them, the pins stay. The first
three are classified as streaming obstacles: there the engine finds no rule
(``tests/update``), and the assembly refuses even a record put on them by hand -- the
refusals below are the assembly's, on records that a bug in the engine would have to write.
"""

import re
from collections import Counter

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.schedule.streaming as stream
from cleave.core.errors import ScheduleRejected
from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops import Matmul, ReduceMax
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling
from cleave.schedule.tiling import schedule_kernel

from tests import _explicit_softmax as X

# the traced order's row sum: a conflict of its own, which the engine refuses (E14)
SUM_UNDER = "is computed from"

OUTSIDE = (ScheduleRejected,
           r"\w+ is computed per iteration of GraphNode_\w+ but also read by \w+ outside it")
SINGLE_ROW_LOOP = (stream.StreamingRefused, r"reduce_max_\w+: its tile \(N,\) holds a single row")
TWO_REPAIRS = (stream.StreamingRefused,
               r"reduce_max_\w+: accumulator \w+ is already repaired; one repair per accumulator")
NOT_LEADING = (stream.StreamingRefused,
               r"reduce_max_\w+: accumulator mma_\w+ tile \(N, N\) does not lead with the row "
               r"tile \(N,\)")
NOT_ITS_DIMS = (stream.StreamingRefused,
                r"reduce_max_\w+: accumulator mma_\w+ tile \(N, N\) does not lead with the row "
                r"tile \(N,\): its dim N is not dim N of reduce_max_\w+")
ALSO_ITS_TRAILING_DIM = (stream.StreamingRefused,
                         r"reduce_max_\w+: accumulator mma_\w+ tile \(N, N\) does not lead with "
                         r"the row tile \(N,\): dim N of reduce_max_\w+ is also its dim N")
# The tiler's own refusals of a matmul whose two output dims are one input dim (x^T @ e).
ONE_DIM_TWO_AXES = (ScheduleRejected,
                    r"Expected exactly one launch-axis mapping for iter dim N, got \[N, N\]\.")
SLICE_AND_WHOLE = (ScheduleRejected,
                   r"Dim N is held whole but launch axes \[N, N\] index it and the grid walks "
                   r"\[N\]: one tile cannot be a slice and the whole\.")

# A tile that holds one row in the main tier: the build refuses its unit extent
# (``refuse_unit_extents``) for every graph, with a max or without.
UNIT_ROW = (ScheduleRejected,
            r"sub_\d+ indexes N dims but its tile \(N, N\) is stored with N: a unit extent that "
            r"is not a batch dim is squeezed out of the tile")

# name -> (program, shapes, expectation per obstacle (None: classified, gets a record),
#          kernels built (all one tile), refusals as [(who, count)])
CASES = {
    "two_softmaxes": (X.two_softmaxes, dict(R=256, C=4096), (None, None), 1,
                      [(TWO_REPAIRS, 27), (SINGLE_ROW_LOOP, 6), (UNIT_ROW, 1)]),
    "column_weighted": (X.column_weighted, dict(M=128, K=4096, N=256), (None,), 0,
                        [(NOT_LEADING, 34), (NOT_ITS_DIMS, 12)]),
    # (the tilings that read ``x`` with one tile per reader are not proposed:
    # DefaultPolicy._reads_placeholder_unevenly; 26 and the 6 SLICE_AND_WHOLE ones were,
    # and such a block built by hand is still refused:
    # test_a_block_that_holds_a_dim_whole_and_slices_it_is_refused)
    "self_weighted": (X.self_weighted, dict(K=4096, N=128), (None,), 0,
                      [(ONE_DIM_TWO_AXES, 8), (ALSO_ITS_TRAILING_DIM, 4)]),
    "max_normalized": (X.max_normalized, dict(R=256, C=4096, D=64),
                       ("must be sums or matmuls that start at zero",), 0, []),
    "sum_next_to_exp": (X.sum_next_to_exp, dict(R=256, C=4096), ("two passes", SUM_UNDER), 14,
                        [(UNIT_ROW, 7)]),
    # (64 query rows: no 256-thread tiling, TCPolicy._register_operand_layout_conflict)
    "unscaled_attention": (X.unscaled_attention, dict(H=32, SQ=64, SKV=256, D=64),
                           ("two passes", SUM_UNDER), 4, [(OUTSIDE, 12)]),
}


def _try(cand):
    """``(tiled, None)`` or ``(None, (exception class, folded reason))``: the tiling the
    build makes of ``cand`` (``schedule_kernel``: ``apply_tiling``, the assembly at its
    tail, and the refusal of unit extents no kernel compiles for)."""
    try:
        return schedule_kernel(cand), None
    except ScheduleRejected as e:
        return None, (type(e), stream._reason_key(e))


def _census(cands):
    """``(kernels, reasons)`` over ``cands``: source per built one, and why not."""
    kernels, reasons = [], Counter()
    for cand in cands:
        tiled, refusal = _try(cand)
        if tiled is None:
            reasons[refusal] += 1
        else:
            kernels.append((cand, stream.sites(tiled), build_candidate_code(cand)))
    return kernels, reasons


def _histogram(reasons, expected):
    """``reasons`` as counts per expected refusal; a reason none of them matches is kept
    under its own text, so it fails the comparison by name."""
    out = Counter()
    for (cls, text), n in reasons.items():
        match = next((who for who, _ in expected
                      if cls is who[0] and re.fullmatch(who[1], text)), None)
        out[match if match is not None else (cls, text)] += n
    return out


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("force", [False, True], ids=["search", "forced-fusion"])
def test_no_looped_kernel(name, force):
    fn, shapes, refused, n_built, refusals = CASES[name]
    ir = X.streamed_ir(fn, shapes, *refused)
    cands = search_tiling(ir, topk=1000, force_fusion=force)
    kernels, reasons = _census(cands)
    n_maxes = sum(type(n) is ReduceMax for n in ir.nodes)

    assert len(kernels) == n_built, reasons
    for cand, sites, code in kernels:
        # nothing carried: no snapshot, no repair, no sentinel. (A max may still run its
        # own loop, ahead of what reads it, from the identity of max: two passes.)
        assert sites == []
        assert "ukernel_rescale" not in code and "_prev" not in code
        assert "-3.4028234663852886e38" not in code
        calls = [line for line in code.splitlines() if line.lstrip().startswith("ukernel_")]
        assert sum(c.lstrip().startswith("ukernel_reduce_max") for c in calls) == n_maxes
    # every other candidate is refused, by whom and why the case says
    assert _histogram(reasons, refusals) == Counter(dict(refusals))
    assert len(kernels) + sum(reasons.values()) == len(cands)
    # the filter keeps exactly the kernels that build
    accept = stream.streaming_accept(ir)
    assert accept is not None
    kept = [c for c in search_tiling(ir, topk=1000, force_fusion=force) if accept(c)]
    assert len(kept) == n_built
    assert accept.reasons == Counter({text: n for (_, text), n in reasons.items()})


def test_search_the_engine_gives_the_trailing_axis_no_rule():
    """What keeps a wrong record off ``column_weighted``, which the classification lets
    through: the engine, run by lowering on the searched graphs, proves no update for the
    value matmul (its rows are not the max's), so the max keeps its pin with that reason.
    (The shared sum of ``two_softmaxes`` is refused by the assembly whatever the engine
    says: ``test_shared_sum_is_never_repaired_twice``.)"""
    import cleave.lang as cl
    from cleave import pipeline as P
    from cleave.frontend import trace
    from cleave.pipeline.plans import lower

    program = trace(X.column_weighted, dict(M=128, K=4096, N=256), cl.f16)
    graphs = [g for g in P.superoptimize(program, P.SuperoptConfig(cache=False))
              if not g.is_split and not g.invalid_split]
    assert graphs
    for g in graphs:
        (ir,) = lower(g)
        assert not stream.recorded(ir), g.summary()
        assert ir.streaming_refusals, g.summary()


def test_shared_sum_is_never_repaired_twice():
    """The two records, each proven on its own, keep one factor per accumulator: the shared
    sum is refused in every tiling that loops it, whichever max is looked at first."""
    ir = X.streamed_ir(X.two_softmaxes, dict(R=256, C=4096), None, None)
    first, second = X.obstacles_of(ir)
    (shared,) = set(first.loops) & set(second.loops)
    kernels, reasons = _census(search_tiling(ir, topk=1000))
    assert all(sites == [] for _, sites, _ in kernels)
    looped = [text for (cls, text), _ in reasons.items() if cls is stream.StreamingRefused
              and "already repaired" in text]
    assert looped and all(f"accumulator {shared} " in text for text in looped)
    # one record alone is assembled: the shared sum gets the factor of that max
    ir = X.streamed_ir(X.two_softmaxes, dict(R=256, C=4096), None, None)
    next(n for n in ir.nodes if n.name == second.full).streaming = None
    kernels, _ = _census(search_tiling(ir, topk=1000))
    assert {s.full for _, sites, _ in kernels for s in sites} <= {first.full}


def test_max_of_the_shifted_exp_is_never_repaired(monkeypatch):
    """Without the classification's refusal (T5 taken away), the record reaches the
    assembly, which sees an accumulator that starts at ``-inf`` and refuses to repair it."""
    monkeypatch.setattr(stream, "_not_a_sum", lambda full, d: "")
    fn, shapes, _, _, _ = CASES["max_normalized"]
    ir = X.streamed_ir(fn, shapes)
    outer, inner = (n.name for n in X.maxes(ir))
    kernels, reasons = _census(search_tiling(ir, topk=1000))
    assert not kernels
    starts = (stream.StreamingRefused,
              rf"{outer}: accumulator {inner} starts at -inf; a repaired accumulator must "
              r"start at zero")
    assert _histogram(reasons, [(starts, 22)]) == Counter({starts: 22})


def test_factor_on_the_trailing_axis_is_refused_on_a_consistent_loop(monkeypatch):
    """``(v @ e) / sum(e, 0)`` with a record by hand: every looped tiling the tiler builds
    has a loop that steps the softmax axis and holds the value matmul as an accumulator.
    That accumulator leads with the rows of ``v``, the max is indexed by the columns of
    ``x``: the assembly refuses where the two tile extents differ (by the tiles) and where
    they are equal (by the dims themselves), and edits nothing."""
    ir = X.streamed_ir(X.column_weighted, dict(M=128, K=4096, N=256))
    (m,) = X.maxes(ir)
    (matmul,) = [n for n in ir.nodes if type(n) is Matmul]
    by_kind = Counter()
    for cand in search_tiling(ir, topk=1000):
        with monkeypatch.context() as patch:
            patch.setattr(stream, "assemble", lambda tiled, flat_graph=None: tiled)
            try:
                tiled = apply_tiling(cand)  # the loop, as the tiler builds it
            except ScheduleRejected:
                continue
        (loop,) = [tier for tier, owner in graph_tiers(tiled) if owner]
        assert matmul.name in {n.name for n in loop.accum_nodes}
        before = [n.name for n in loop.nodes]
        with pytest.raises(stream.StreamingRefused) as refusal:
            stream.assemble(tiled, ir)
        assert [n.name for n in loop.nodes] == before and not loop.accum_scale
        rows, cols = (int(t) for t in cand.output_tile)
        by_kind["equal" if rows == cols else "unequal"] += 1
        wanted = NOT_ITS_DIMS if rows == cols else NOT_LEADING
        assert re.fullmatch(wanted[1], stream._reason_key(refusal.value)), refusal.value
    assert by_kind["equal"] >= 4 and by_kind["unequal"] >= 4, by_kind


def test_a_dim_that_is_both_the_rows_and_the_trailing_axis_is_refused(monkeypatch):
    """``(x^T @ e) / sum(e, 0)`` with a record by hand. The accumulator's rows come from the
    columns of ``x``, and so do the max's rows, so dim 0 of the accumulator IS dim 0 of the
    max -- but the columns of ``x`` are also the accumulator's trailing axis, the one the
    factor belongs to (``acc[n1, n2] = sum_k x[k, n1] * exp(x[k, n2] - m[n2])``). A repair
    of the leading dim would emit ``ACC[i0, i1] *= S[i0]``. Every looped tiling the tiler
    builds is refused, and nothing is edited."""
    ir = X.streamed_ir(X.self_weighted, dict(K=4096, N=128))
    (m,) = X.maxes(ir)
    (matmul,) = [n for n in ir.nodes if type(n) is Matmul]
    assert stream._other_lead(ir, m, matmul) == f"dim 0 of {m.name} is also its dim 1"
    looped = 0
    for cand in search_tiling(ir, topk=1000):
        with monkeypatch.context() as patch:
            patch.setattr(stream, "assemble", lambda tiled, flat_graph=None: tiled)
            try:
                tiled = apply_tiling(cand)
            except ScheduleRejected:
                continue
        (loop,) = [tier for tier, owner in graph_tiers(tiled) if owner]
        assert matmul.name in {n.name for n in loop.accum_nodes}
        before = [n.name for n in loop.nodes]
        with pytest.raises(stream.StreamingRefused) as refusal:
            stream.assemble(tiled, ir)
        assert [n.name for n in loop.nodes] == before and not loop.accum_scale
        assert re.fullmatch(ALSO_ITS_TRAILING_DIM[1], stream._reason_key(refusal.value)), \
            refusal.value
        looped += 1
    assert looped == 4


def test_a_block_that_holds_a_dim_whole_and_slices_it_is_refused():
    """``self_weighted`` reads ``x`` on both axes of its output, so a block of 64 rows and
    128 columns would read ``x`` with one tile of 64 columns for one operand and the whole
    128 for the other. The policy does not propose such a block
    (``DefaultPolicy._reads_placeholder_unevenly``); built by hand from the policy's own
    rating, with the steps of every candidate it does propose, the build refuses it, as it
    did when the policy proposed it (6 of them), and never builds it."""
    from cleave.arch import get_arch
    from cleave.ir.graph import OutputNode
    from cleave.schedule.policy.default import TCPolicy

    ir = X.streamed_ir(X.self_weighted, dict(K=4096, N=128))
    cands = search_tiling(ir, topk=1000)
    assert {tuple(c.output_tile) for c in cands} == {(64, 64), (128, 128)}
    policy = TCPolicy([n for n in ir.nodes if isinstance(n, OutputNode)], get_arch("cuda"),
                      flat_graph=ir)
    reasons = Counter()
    for cand in cands:
        by_hand = policy.compute_tile_dict(
            [64, 128], {node: dict(steps) for node, steps in cand.rstep_map.items()})
        by_hand.n_threads = cand.n_threads
        tiled, refusal = _try(by_hand)
        assert tiled is None
        reasons[refusal] += 1
    assert _histogram(reasons, [(SLICE_AND_WHOLE, len(cands))]) == Counter(
        {SLICE_AND_WHOLE: len(cands)})


SERVED = {  # every family the assembly serves: its accumulators lead with the max's dims
    "flash_attention": (X.flash_attention, dict(H=32, SQ=64, SKV=256, D=64)),
    "one_head": (X.flash_attention, dict(H=1, SQ=64, SKV=256, D=64)),
    "flash_and_its_max": (X.flash_and_its_max, dict(H=32, SQ=64, SKV=256, D=64)),
    "flash_rows": (X.flash_rows, dict(R=256, C=4096, D=64)),
    "column_average": (X.column_average, dict(H=4, K=4096, N=256)),
    "column_average_2d": (X.column_average_2d, dict(K=4096, N=256)),
}


@pytest.mark.parametrize("name", sorted(SERVED))
def test_the_provenance_check_passes_every_served_family(name):
    fn, shapes = SERVED[name]
    ir = X.streamed_ir(fn, shapes)
    for ob in X.obstacles_of(ir):
        full = next(n for n in ir.nodes if n.name == ob.full)
        for acc in ob.loops:
            node = next(n for n in ir.nodes if n.name == acc)
            assert stream._other_lead(ir, full, node) is None, (ob.full, acc)
