"""A softmax over an axis that is not the last one, its row held whole (CPU).

The max is the flat graph's own reduction, so where its row is whole in a tile it is
emitted the way the tiler emits every reduction: with the ``dim`` of the tile it reads
(``rebase_reduction_axis``), in the main tier as in a loop. Two tilings the hand-derived
pass had to be taught are served this way: the max in the main tier behind a unit dim
that is squeezed out of the stored tile (a batched ``softmax(x, 1)``), and the max inside a
loop that does not step its dim (attention normalized over the queries: the max is in the
loop of the value matmul, over the keys). Both programs return a softmax or divide before
the value matmul: their max has a record, but a tiling that loops its row is refused (the
exp is computed per iteration and also read outside the loop), and one that holds the row
whole uses no record (``tiling.records_in_use``). Their kernels are validated on a GPU in
``test_streaming_gpu.py``.
"""

import re

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.schedule.streaming as stream
from cleave.ir.graph_utils import tmap_by_name
from cleave.ir.ops import ReduceMax, ReduceSum, Sub
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling

from tests import _explicit_softmax as X
from tests._lowering import lower_to_cuda
from tests.graph.apply_tiling.test_streaming import _find, _macro_calls


def _accepted(ir, threads=128):
    accept = stream.streaming_accept(ir)
    assert accept is not None  # the graph has an obstacle: a refused one installs it too
    cands = [c for c in search_tiling(ir, topk=1000, accept=accept) if c.n_threads == threads]
    return cands, accept.reasons


def test_one_tile_in_the_main_tier_behind_a_squeezed_unit_dim():
    ir = X.streamed_ir(X.softmax_mid, dict(H=4, C=256, R=64))
    (m,) = X.maxes(ir)
    assert m.flat_axis == 1
    cands, reasons = _accepted(ir)
    assert len(cands) >= 3
    # what is refused are the tilings the tiler refuses for any graph: a unit extent that is
    # not a batch dim, and a loop over the row, whose exp the division reads outside it
    e = X.exps(ir)[0].name
    assert {r.split(" ")[0] for r in reasons} == {"sub_2", e}, reasons
    assert all("unit extent" in r or f"{e} is computed per iteration of " in r
               for r in reasons), reasons
    whole = []
    for cand in cands:
        tiled = apply_tiling(cand)
        assert stream.sites(tiled) == [] and not tiled.lag_nodes and not tiled.accum_scale
        e = X.exps(ir)[0].name
        tier, owner, _ = _find(tiled, e)
        assert owner is None
        tiles = tmap_by_name(tier)
        exp_tile = tuple(int(t) for t in tiles[e].tile_size)
        # the batch dim is a unit extent in front of the softmax axis: squeezed in storage
        assert exp_tile[0] == 1 and exp_tile[1] == 256
        (rowsum,) = [n for n in tier.nodes if type(n) is ReduceSum]
        assert (rowsum.flat_axis, rowsum.axis_arg) == (1, 0)
        assert next(n for n in tier.nodes if type(n) is Sub).broadcast == 1
        max_tier, max_owner, node = _find(tiled, m.name)
        assert type(node) is ReduceMax and "accum_init" not in vars(node)
        if max_owner is None:
            # flat axis 1 of (H, C, R); dim 0 of the stored (C, r) tile, like the sum
            assert (node.flat_axis, node.axis_arg) == (1, 0)
            assert tuple(int(t) for t in tiles[node.name].tile_size) == (1, exp_tile[2])
            whole.append(cand)
        else:
            # or the max runs a loop of its own over the row, ahead of what reads it: two
            # passes, from the identity of max (nothing is carried for anyone else)
            assert [n.name for n in max_tier.accum_nodes] == [m.name]
            assert all(n is node or n.is_placeholder() for n in max_tier.nodes)
    assert whole

    code = build_candidate_code(whole[0])
    main = [re.sub(r"PlaceHolder_\w+", "P", c) for c in _macro_calls(code, "main_kernel_iter")]
    e = X.exps(ir)[0].name
    assert main[:3] == [
        f"ukernel_reduce_max_axis0_gr_clear(P, t_{m.name})",
        f"ukernel_sub_broadcast_1_2dims_rrr(P, t_{m.name}, t_sub_2)",
        f"ukernel_exp_2dims_rr(t_sub_2, t_{e})",
    ]
    assert "T.reduce_max(A_ptr, B_ptr, dim=0, clear=True)" in code
    cuda = lower_to_cuda(code)
    assert cuda.index("tl::MaxOp") < cuda.index("expf(") < cuda.index("tl::SumOp")


def test_one_tile_inside_a_loop_that_does_not_step_the_softmax_axis():
    ir = X.streamed_ir(X.softmax_over_queries, dict(H=32, SQ=64, SKV=1024, D=64))
    (m,) = X.maxes(ir)
    assert m.flat_axis == 1
    cands, reasons = _accepted(ir)
    # every tiling that steps the max and its sum (over the queries) in the loop of the value
    # matmul (over the keys) is refused: that loop does not step the dim the max reduces
    assert len(cands) >= 8 and reasons
    assert all(re.fullmatch(rf"{m.name}: the loop of GraphNode_mma_\d+ carries it but does not "
                            r"step the dim it reduces", r) for r in reasons), reasons
    looped = 0
    for cand in cands:
        tiled = apply_tiling(cand)
        tier, owner, node = _find(tiled, m.name)
        assert stream.sites(tiled) == []
        assert not tier.lag_nodes and not tier.accum_scale  # nothing is carried
        assert "accum_init" not in vars(node) and getattr(node, "streaming", None) is None
        if owner is None:
            continue
        looped += 1
        # the loop is the value matmul's, over the keys; the max reads (queries, keys)
        assert not any(node is a for a in tier.accum_nodes)
        assert (node.flat_axis, node.axis_arg) == (1, 0)
        (colsum,) = [n for n in tier.nodes if type(n) is ReduceSum]
        assert (colsum.flat_axis, colsum.axis_arg) == (1, 0)
        assert next(n for n in tier.nodes if type(n) is Sub).broadcast == 1
    assert looped >= 8

    cand = next(c for c in cands if _find(apply_tiling(c), m.name)[1] is not None)
    owner = _find(apply_tiling(cand), m.name)[1]
    code = build_candidate_code(cand)
    calls = [re.sub(r"PlaceHolder_\w+", "P", c) for c in _macro_calls(code, f"{owner.name}_iter")]
    at = next(i for i, c in enumerate(calls) if c.startswith("ukernel_reduce_max_axis0_rr_clear("))
    scores = re.match(r"ukernel_reduce_max_axis0_rr_clear\((\w+), ", calls[at]).group(1)
    shifted = next(n.name for n in ir.nodes if type(n) is Sub)
    e = X.exps(ir)[0].name
    assert calls[at: at + 3] == [
        f"ukernel_reduce_max_axis0_rr_clear({scores}, t_{m.name})",
        f"ukernel_sub_broadcast_1_2dims_rrr({scores}, t_{m.name}, t_{shifted})",
        f"ukernel_exp_2dims_rr(t_{shifted}, t_{e})",
    ]
    assert "T.fill" not in code and "rescale" not in code
    cuda = lower_to_cuda(code)
    assert cuda.index("tl::MaxOp") < cuda.index("expf(") < cuda.index("tl::SumOp")
