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
"""

from collections import Counter

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.schedule.stabilize as stab
from cleave.core.errors import ScheduleRejected
from cleave.ir.ops import ReduceMax
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling

from tests import _softmax as S

NEXT_TO_SUM = "reads the exp next to the finished reduction"
SHARED = "the shared sum is also scaled by the other exp; one factor per accumulator"

# name -> (program, shapes, one expectation per exp, kernels built, of them rolling)
CASES = {
    "two_softmaxes": (S.two_softmaxes, dict(R=256, C=4096),
                      (dict(rolling=SHARED), dict(rolling=SHARED)), 1),
    "column_weighted": (S.column_weighted, dict(M=128, K=4096, N=256),
                        (dict(axis=-2, rolling="the factor is not on the leading axes"),), 0),
    "max_normalized": (S.max_normalized, dict(R=256, C=4096, D=64),
                       (dict(rolling="takes a max of the scaled value"),), 0),
    "sum_next_to_exp": (S.sum_next_to_exp, dict(R=256, C=4096),
                        (dict(rolling=f"add {NEXT_TO_SUM}"),), 2),
    "unscaled_attention": (S.unscaled_attention, dict(H=32, SQ=64, SKV=256, D=64),
                           (dict(rolling=f"div {NEXT_TO_SUM}"),), 8),
}


def _census(ir, force):
    """``(kernels, reasons)`` over every raw candidate: source per built one, and why not."""
    kernels, reasons = [], Counter()
    for cand in search_tiling(ir, topk=1000, force_fusion=force):
        try:
            tiled = apply_tiling(cand)
        except ScheduleRejected as e:
            reasons[stab._reason_key(e)] += 1
            continue
        kernels.append((cand, {s.case for s in stab.sites(tiled)}, build_candidate_code(cand)))
    return kernels, reasons


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("force", [False, True], ids=["search", "forced-fusion"])
def test_no_rolling_kernel(name, force):
    fn, shapes, softmax, n_built = CASES[name]
    ir = S.softmax_ir(fn, shapes, *softmax)
    kernels, reasons = _census(ir, force)
    # one max per softmax, next to whatever max the program takes itself
    n_maxes = len(S.exps(ir)) + sum(type(n) is ReduceMax for n in ir.nodes)

    assert len(kernels) == n_built, reasons
    for cand, cases, code in kernels:
        # served in place: a per-block max, nothing carried, nothing rescaled
        assert cases == {"in-tile"}
        assert "ukernel_rescale" not in code and "T.fill" not in code
        calls = [line for line in code.splitlines() if line.lstrip().startswith("ukernel_")]
        assert sum(c.lstrip().startswith("ukernel_reduce_max") for c in calls) == n_maxes
    assert reasons, "every candidate was built; the case tests nothing"
    # the filter keeps exactly the kernels that build
    accept = stab.stabilize_accept(ir)
    kept = [c for c in search_tiling(ir, topk=1000, force_fusion=force) if accept(c)]
    assert len(kept) == n_built and accept.reasons == reasons


def test_shared_sum_is_never_rescaled():
    """With the obstacle the analysis records taken away, the tilings that loop the shared
    sum reach the rewrite itself, and it still refuses them."""
    import dataclasses

    fn, shapes, softmax, _ = CASES["two_softmaxes"]
    ir = S.softmax_ir(fn, shapes, *softmax)
    for exp in S.exps(ir):
        exp.stabilize = dataclasses.replace(exp.stabilize, rolling="")
    kernels, reasons = _census(ir, False)
    assert all("ukernel_rescale" not in code for _, _, code in kernels)
    assert sum(n for text, n in reasons.items() if "is also scaled by" in text) == 35
