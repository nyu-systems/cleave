"""An exact softmax over an axis that is not the last one, taken in place (CPU).

A tile that holds the whole softmax axis gets ``m = max(s)`` in place. When that axis is not
the last axis of the scores there were two tilings the pass refused, because a reduction that
is not looped used to be emitted with the ``dim`` of the flat graph:

* the exp sits in the main tier and a unit dim in front of the softmax axis is squeezed out
  of the stored tile (a batched ``softmax(x, 1)``): the flat ``dim`` is one too many;
* the exp sits in a loop that does not tile its softmax axis (attention normalized over the
  queries: the exp is inside the loop of the value matmul).

The tiler now gives every reduction the ``dim`` of the tile it reads
(``rebase_reduction_axis``), in the main tier as in a loop, and the max the pass adds gets
its ``dim`` the same way. So both tilings are served; the refusal is gone. Their kernels are
validated on a GPU in ``tests/frontend/test_exact_softmax.py``.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.lang as cl
import cleave.schedule.stabilize as stab
from cleave.ir.ops import ReduceMax, ReduceSum, Sub
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling

from tests import _softmax as S
from tests._lowering import lower_to_cuda
from tests.graph.apply_tiling.test_stabilize import _find, _macro_calls

NEXT_TO_SUM = "reads the exp next to the finished reduction"


def softmax_mid(x: cl.Tensor["H", "C", "R"]):
    """Softmax over the middle axis of a batched input."""
    e = S.stable_exp(x, 1)
    return e / cl.sum(e, 1, keepdim=True)


def softmax_over_queries(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                         v: cl.Tensor["H", "SKV", "D"]):
    """Attention normalized over the queries: every key's column sums to 1."""
    e = S.stable_exp((q @ k.mT) / q.shape[-1] ** 0.5, -2)
    return (e / cl.sum(e, -2, keepdim=True)) @ v


def _accepted(ir, threads=128):
    accept = stab.stabilize_accept(ir)
    cands = [c for c in search_tiling(ir, topk=1000, accept=accept) if c.n_threads == threads]
    return cands, accept.reasons


def test_the_refusal_is_gone():
    import inspect

    assert "is not the last axis" not in inspect.getsource(stab)


def test_in_tile_in_the_main_tier_behind_a_squeezed_unit_dim():
    ir = S.lowered(softmax_mid, dict(H=4, C=256, R=64))
    (exp,) = S.exps(ir)
    assert exp.stabilize.axes == (-2,) and NEXT_TO_SUM in exp.stabilize.rolling
    cands, reasons = _accepted(ir)
    assert len(cands) >= 3
    # what is still refused is every tiling that would need the rolling form (the exp is
    # read next to its finished sum) and the single-row tile
    assert all("also read by" in r or "holds a single row" in r for r in reasons), reasons
    for cand in cands:
        tiled = apply_tiling(cand)
        tier, owner, m = _find(tiled, f"{exp.name}_max")
        site = _find(tiled, exp.name)[2].stabilized
        exp_tile = tuple(int(t) for t in tier.tmap[_find(tiled, exp.name)[2]].tile_size)
        # the batch dim is a unit extent in front of the softmax axis: squeezed in storage
        assert owner is None and exp_tile[0] == 1 and exp_tile[1] == 256
        assert site == stab.Site(exp.name, "main", "in-tile", axis=1, full=256, block=256,
                                 step=256)
        # flat axis 1 of (H, C, R); dim 0 of the stored (C, r) tile, like the sum next to it
        assert type(m) is ReduceMax and (m.flat_axis, m.axis_arg) == (1, 0)
        (rowsum,) = [n for n in tier.nodes if type(n) is ReduceSum]
        assert (rowsum.flat_axis, rowsum.axis_arg) == (1, 0)
        shifted = _find(tiled, f"{exp.name}_shifted")[2]
        assert type(shifted) is Sub and shifted.broadcast == 1
        assert tuple(int(t) for t in tier.tmap[m].tile_size) == (1, exp_tile[2])

    code = build_candidate_code(cands[0])
    main = [re.sub(r"PlaceHolder_\w+", "P", c) for c in _macro_calls(code, "main_kernel_iter")]
    e = exp.name
    assert main[:3] == [
        f"ukernel_reduce_max_axis0_gr_clear(P, t_{e}_max)",
        f"ukernel_sub_broadcast_1_2dims_rrr(P, t_{e}_max, t_{e}_shifted)",
        f"ukernel_exp_2dims_rr(t_{e}_shifted, t_{e})",
    ]
    assert "T.reduce_max(A_ptr, B_ptr, dim=0, clear=True)" in code
    cuda = lower_to_cuda(code)
    assert cuda.index("tl::MaxOp") < cuda.index("expf(") < cuda.index("tl::SumOp")


def test_in_tile_inside_a_loop_that_does_not_tile_the_softmax_axis():
    ir = S.lowered(softmax_over_queries, dict(H=32, SQ=64, SKV=1024, D=64))
    (exp,) = S.exps(ir)
    assert exp.stabilize.axes == (-2,)
    cands, reasons = _accepted(ir)
    assert len(cands) >= 8 and not reasons
    looped = 0
    for cand in cands:
        tiled = apply_tiling(cand)
        tier, owner, m = _find(tiled, f"{exp.name}_max")
        site = _find(tiled, exp.name)[2].stabilized
        assert site.case == "in-tile" and site.full == site.block == site.step == 64
        assert not tier.lag_nodes and not tier.accum_scale  # nothing is carried
        assert "accum_init" not in vars(m)
        if owner is None:
            continue
        looped += 1
        # the loop is the value matmul's, over the keys; the exp tile is (queries, keys)
        assert site.tier == owner.name and site.axis == 0
        assert (m.flat_axis, m.axis_arg) == (1, 0)
        (colsum,) = [n for n in tier.nodes if type(n) is ReduceSum]
        assert (colsum.flat_axis, colsum.axis_arg) == (1, 0)
        assert _find(tiled, f"{exp.name}_shifted")[2].broadcast == 1
    assert looped >= 8

    cand = next(c for c in cands if _find(apply_tiling(c), exp.name)[1] is not None)
    owner = _find(apply_tiling(cand), exp.name)[1]
    code = build_candidate_code(cand)
    calls = [re.sub(r"PlaceHolder_\w+", "P", c) for c in _macro_calls(code, f"{owner.name}_iter")]
    e = exp.name
    at = next(i for i, c in enumerate(calls) if c.startswith("ukernel_reduce_max_axis0_rr_clear("))
    scores = re.match(r"ukernel_reduce_max_axis0_rr_clear\((\w+), ", calls[at]).group(1)
    assert calls[at: at + 3] == [
        f"ukernel_reduce_max_axis0_rr_clear({scores}, t_{e}_max)",
        f"ukernel_sub_broadcast_1_2dims_rrr({scores}, t_{e}_max, t_{e}_shifted)",
        f"ukernel_exp_2dims_rr(t_{e}_shifted, t_{e})",
    ]
    assert "T.fill" not in code and "rescale" not in code
    cuda = lower_to_cuda(code)
    assert cuda.index("tl::MaxOp") < cuda.index("expf(") < cuda.index("tl::SumOp")
