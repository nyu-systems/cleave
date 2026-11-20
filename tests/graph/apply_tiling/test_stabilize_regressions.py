"""Softmax graphs that are legal but must not get a rolling kernel (CPU).

Each program below has an ``exp`` whose scale cancels, so its reference is the stable one
and a tiling that holds the whole softmax axis in one tile is served. None of them has a
rolling form, each for a different reason, and for each of them a design prototype built
rolling kernels that were wrong:

``two_softmaxes``     one sum is scaled by two softmaxes; the rolling form keeps a single
                      factor per accumulator (the prototypes rescaled the shared sum by one
                      alpha).
``column_weighted``   the factor lives on the accumulator's trailing axis
                      (``O[m, n] *= alpha[m]`` was emitted for a factor indexed by ``n``).
``max_normalized``    a max of the scaled value: its accumulator starts at ``-inf`` and the
                      first ``alpha`` is 0.
``sum_next_to_exp``   the exp is read next to its own finished row sum.
``unscaled_attention``  traced order, the exp reading the QK^T matmul itself: here it is
                      the IN-TILE rewrite that used to fail, by replacing the accumulation
                      edge with a plain one.

Every candidate of every program goes through the default build. Anything but a
``ScheduleRejected`` propagates: a refused tiling is a result, a crash is a bug.

Who refuses is pinned per case, as a histogram of reasons, because "no kernel" alone does
not say the rewrite had a say: a looped tiling of ``sum_next_to_exp`` and of
``unscaled_attention`` is already refused by the tiler (the exp would be read outside its
loop), and every tiling the policy proposes for ``column_weighted`` by the tiler's
loop-extent guard (the policy steps the matmul and the column sum differently). For
``column_weighted`` the rewrite is therefore also tried on tilings that step both
reductions together, which the policy does not propose today.
"""

import dataclasses
import re
from collections import Counter

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.schedule.stabilize as stab
from cleave.arch import get_arch
from cleave.core.errors import ScheduleRejected
from cleave.ir.graph import OutputNode
from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops import Matmul, ReduceMax, ReduceSum
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling, tiling
from cleave.schedule.policy.default import TCPolicy

from tests import _softmax as S

NEXT_TO_SUM = "reads the exp next to the finished reduction"
SHARED = "the shared sum is also scaled by the other exp; one factor per accumulator"
TRAILING = "the factor is not on the leading axes"
TAKES_MAX = "takes a max of the scaled value"

# The refusals, by who raises them (numbers are folded to N by ``_reason_key``).
OUTSIDE = (ScheduleRejected,
           r"exp_\d+ is computed per iteration of GraphNode_\w+ but also read by \w+ outside it")
MISCOUNT = (ScheduleRejected,
            r"GraphNode_\w+ runs N iterations but dim N of \w+ is cut into N tiles \(N of N\)")
SINGLE_ROW = (stab.StabilizeRefused, r"exp_\d+: its tile \(N, N\) holds a single row")
TRIP_COUNT = (stab.StabilizeRefused,
              r"exp_\d+: GraphNode_\w+ iterates N times but the softmax axis is stepped N of N")
NOT_LEADING = (stab.StabilizeRefused,
               r"exp_\d+: accumulator \w+ tile \(N, N\) does not lead with the row tile \(N,\)")
TWO_OWNERS = (stab.StabilizeRefused,
              r"exp_\d+: accumulator \w+ is also scaled by exp_\d+; the rolling form keeps "
              r"one factor per accumulator")
FROM_MINUS_INF = (stab.StabilizeRefused,
                  r"exp_\d+: accumulator \w+ starts at -inf; a rescaled accumulator must "
                  r"start at zero")


def _no_rolling_form(obstacle):
    return (stab.StabilizeRefused, rf"exp_\d+: no rolling form: {re.escape(obstacle)}")


# name -> (program, shapes, one expectation per exp, kernels built (all in-tile),
#          refusals as [(who, count)])
CASES = {
    "two_softmaxes": (S.two_softmaxes, dict(R=256, C=4096),
                      (dict(rolling=SHARED), dict(rolling=SHARED)), 1,
                      [(_no_rolling_form(SHARED), 35), (SINGLE_ROW, 8)]),
    "column_weighted": (S.column_weighted, dict(M=128, K=4096, N=256),
                        (dict(axis=-2, rolling=TRAILING),), 0,
                        [(MISCOUNT, 52)]),
    "max_normalized": (S.max_normalized, dict(R=256, C=4096, D=64),
                       (dict(rolling=TAKES_MAX),), 0,
                       [(_no_rolling_form(TAKES_MAX), 34)]),
    "sum_next_to_exp": (S.sum_next_to_exp, dict(R=256, C=4096),
                        (dict(rolling=f"add {NEXT_TO_SUM}"),), 2,
                        [(OUTSIDE, 21), (SINGLE_ROW, 1)]),
    "unscaled_attention": (S.unscaled_attention, dict(H=32, SQ=64, SKV=256, D=64),
                           (dict(rolling=f"div {NEXT_TO_SUM}"),), 8,
                           [(OUTSIDE, 24)]),
}


def _try(cand):
    """``(tiled, None)`` or ``(None, (exception class, folded reason))``."""
    try:
        return apply_tiling(cand), None
    except ScheduleRejected as e:
        return None, (type(e), stab._reason_key(e))


def _census(cands):
    """``(kernels, reasons)`` over ``cands``: source per built one, and why not."""
    kernels, reasons = [], Counter()
    for cand in cands:
        tiled, refusal = _try(cand)
        if tiled is None:
            reasons[refusal] += 1
        else:
            kernels.append(
                (cand, {s.case for s in stab.sites(tiled)}, build_candidate_code(cand))
            )
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


def _without_obstacle(ir):
    for exp in S.exps(ir):
        exp.stabilize = dataclasses.replace(exp.stabilize, rolling="")
    return ir


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("force", [False, True], ids=["search", "forced-fusion"])
def test_no_rolling_kernel(name, force):
    fn, shapes, softmax, n_built, refusals = CASES[name]
    ir = S.softmax_ir(fn, shapes, *softmax)
    cands = search_tiling(ir, topk=1000, force_fusion=force)
    kernels, reasons = _census(cands)
    # one max per softmax, next to whatever max the program takes itself
    n_maxes = len(S.exps(ir)) + sum(type(n) is ReduceMax for n in ir.nodes)

    assert len(kernels) == n_built, reasons
    for cand, cases, code in kernels:
        # served in place: a per-block max, nothing carried, nothing rescaled
        assert cases == {"in-tile"}
        assert "ukernel_rescale" not in code and "T.fill" not in code
        calls = [line for line in code.splitlines() if line.lstrip().startswith("ukernel_")]
        assert sum(c.lstrip().startswith("ukernel_reduce_max") for c in calls) == n_maxes
    # every other candidate is refused, by whom and why the case says
    assert _histogram(reasons, refusals) == Counter(dict(refusals))
    assert len(kernels) + sum(reasons.values()) == len(cands)
    # the filter keeps exactly the kernels that build
    accept = stab.stabilize_accept(ir)
    kept = [c for c in search_tiling(ir, topk=1000, force_fusion=force) if accept(c)]
    assert len(kept) == n_built
    assert accept.reasons == Counter({text: n for (_, text), n in reasons.items()})


def test_shared_sum_is_never_rescaled():
    """With the obstacle the analysis records taken away, the tilings that loop the shared
    sum reach the rewrite itself, and it still refuses them."""
    fn, shapes, softmax, n_built, _ = CASES["two_softmaxes"]
    ir = _without_obstacle(S.softmax_ir(fn, shapes, *softmax))
    kernels, reasons = _census(search_tiling(ir, topk=1000))
    assert len(kernels) == n_built
    assert all("ukernel_rescale" not in code for _, _, code in kernels)
    expected = [(TWO_OWNERS, 35), (SINGLE_ROW, 8)]
    assert _histogram(reasons, expected) == Counter(dict(expected))


def test_max_of_the_scaled_value_is_never_rescaled():
    """Same for the max: without the recorded obstacle the rewrite sees an accumulator
    that starts at ``-inf`` and refuses to rescale it."""
    fn, shapes, softmax, _, _ = CASES["max_normalized"]
    ir = _without_obstacle(S.softmax_ir(fn, shapes, *softmax))
    kernels, reasons = _census(search_tiling(ir, topk=1000))
    assert not kernels
    assert _histogram(reasons, [(FROM_MINUS_INF, 34)]) == Counter({FROM_MINUS_INF: 34})


# ----------------------------------------------------- column_weighted, past the tiler


def _column_weighted():
    fn, shapes, softmax, _, _ = CASES["column_weighted"]
    ir = S.softmax_ir(fn, shapes, *softmax)
    (matmul,) = [n for n in ir.nodes if type(n) is Matmul]
    (colsum,) = [n for n in ir.nodes if type(n) is ReduceSum]
    return ir, matmul, colsum


def _stepped_together(ir, cands):
    """One tiling per output tile and step of ``cands`` in which EVERY reduction takes
    that step: the loop then counts the tiles of both the matmul and the column sum."""
    for node in ir.nodes:
        if isinstance(node, Matmul):
            node.set_tag("tensorcore_config", True)
    outputs = [n for n in ir.nodes if isinstance(n, OutputNode)]
    policy = TCPolicy(outputs, get_arch("cuda"), flat_graph=ir)
    built, seen = [], set()
    for cand in cands:
        for step in sorted({int(s) for v in cand.rstep_map.values() for s in (v or {}).values()}):
            key = (tuple(cand.output_tile), step)
            if key in seen:
                continue
            seen.add(key)
            rstep = {n: ({k: step for k in v} if v else v) for n, v in cand.rstep_map.items()}
            together = policy.compute_tile_dict(list(cand.output_tile), rstep)
            if together.valid:
                built.append((together, step))
    return built


def test_factor_on_the_trailing_axis_is_refused_by_the_rewrite(monkeypatch):
    """``(v @ e) / sum(e, 0)`` on the policy's own tilings, with the tiler's loop-extent
    guard taken out so that they reach the rewrite: it refuses every one of them, with the
    obstacle the analysis records, and without that obstacle because the loop does not
    count the tiles of the softmax axis."""
    ir, _, _ = _column_weighted()
    cands = search_tiling(ir, topk=1000)
    monkeypatch.setattr(tiling, "_refuse_loops_that_miscount_their_tiles", lambda main: None)

    _, reasons = _census(cands)
    wanted = [(_no_rolling_form(TRAILING), 52)]
    assert _histogram(reasons, wanted) == Counter(dict(wanted))

    _, reasons = _census(search_tiling(_without_obstacle(ir), topk=1000))
    assert _histogram(reasons, [(TRIP_COUNT, 52)]) == Counter({TRIP_COUNT: 52})


def test_factor_on_the_trailing_axis_is_refused_on_a_consistent_loop():
    """The same graph with the matmul and the column sum stepped together: tilings the
    tiler builds (its guard untouched), with a loop that does tile the softmax axis.

    The recorded obstacle is what refuses them. Tile sizes cannot stand in for it: the
    rewrite's own check of the accumulator's tile against the row tile only tells the two
    axes apart while their tile extents differ, so it is exercised here on those tilings
    only.
    """
    ir, matmul, colsum = _column_weighted()
    (exp,) = S.exps(ir)
    record = exp.stabilize
    together = [(c, step) for c, step in _stepped_together(ir, search_tiling(ir, topk=1000))
                if step > 1]  # one element per iteration is refused before anything else
    assert len(together) >= 8

    unequal = 0
    for cand, step in together:
        exp.stabilize = None
        (loop,) = [tier for tier, owner in graph_tiers(apply_tiling(cand)) if owner]
        assert loop.extent == 4096 // step  # the tiler builds it, and it counts its tiles
        assert {n.name for n in loop.accum_nodes} == {matmul.name, colsum.name}

        exp.stabilize = record
        assert _try(cand) == (None, (stab.StabilizeRefused,
                                     f"{exp.name}: no rolling form: {TRAILING}"))

        rows, cols = (int(t) for t in cand.output_tile)
        if rows != cols:
            unequal += 1
            exp.stabilize = dataclasses.replace(record, rolling="")
            tiled, refusal = _try(cand)
            assert tiled is None and refusal[0] is stab.StabilizeRefused
            assert re.fullmatch(NOT_LEADING[1], refusal[1]), refusal
    exp.stabilize = record
    assert unequal >= 4

