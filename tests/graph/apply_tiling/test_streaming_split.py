"""The pair built from the loop state, tiled (CPU).

``pipeline.plans.state_pair`` builds a split of the flash graph from its loop state: a
partial that returns each split's accumulators and its own max, and a combine that takes
the max over the splits and repairs each partial by the proven rule. Both are flat graphs
with a max in front of their sums and each carries the record derived for it, so
``apply_tiling`` serves both with the one assembly every streaming obstacle goes through.
Pinned here is what it builds for them over every tiling of the GQA pair (32 query heads
over 8 kv heads, 64 queries, 4 splits of 64 keys):

* the partial reuses the max it returns as its carried max where the keys of a split are
  looped -- an accumulator of the loop behind the two sums it repairs, started at the most
  negative finite float32 -- and takes it in place where a tile holds the split; either way
  it returns it in float32, in front of its two float32 sums;
* the combine carries a running max over the split axis where it loops over the partials
  (in a plain loop: it weights the partials it loads, so no asynchronous copy may refill
  one in flight), takes it in place where a tile holds all of them, and is refused where a
  step holds one partial or a tile a single row.

The loop bodies (both looped stages, and the combine that holds every partial) are pinned
statement by statement, and lowered to CUDA source.

``test_search_*``: the pair is built from a searched graph (root conftest marks).
"""

import re
from collections import Counter

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import ScheduleRejected
from cleave.frontend import trace
from cleave.ir.graph import AccumEdge
from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops import NEG_FLT_MAX, ReduceMax
from cleave.pipeline.plans import lower
from cleave.runtime.compile import build_candidate
from cleave.runtime.reference import output_order
from cleave.schedule import apply_tiling, search_tiling
from cleave.schedule.streaming import _reason_key, sites, streaming_accept

from tests._lowering import lower_to_cuda
from tests.graph.apply_tiling.test_streaming import _calls, _find, _macro_calls, _steps

SHAPES = dict(HQ=32, HKV=8, SQ=64, SKV=256, D=64)
FACTOR = 4
# accumulator, row sums, max. Looped: the split axis first (the two vectors padded).
SPLIT_FIRST = (((1, 0, 2), 1), ((1, 0, 2, 3), 0), ((1, 0, 2), 1))
LOGICAL = (((0, 1, 2), 0), ((0, 1, 2, 3), 0), ((0, 1, 2), 0))
FILL = "-3.4028234663852886e38"


def grouped(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"],
            v: cl.Tensor["HKV", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


@pytest.fixture(scope="module")
def pair():
    prog = trace(grouped, SHAPES, cl.f16)
    graphs = P.superoptimize(prog, P.SuperoptConfig(split=("SKV",), cache=False))
    (graph,) = [g for g in graphs if g.summary().endswith("sum>matmul>div")]
    return lower(graph, FACTOR)


def _tilings(ir):
    """``(candidate, tiled graph | refusal)`` for every tiling the policy proposes."""
    out = []
    for cand in search_tiling(ir, topk=1000):
        try:
            out.append((cand, apply_tiling(cand)))
        except ScheduleRejected as refusal:
            out.append((cand, refusal))
    return out


def _loop_kinds(code):
    return re.findall(r"for i0 in T\.(Pipelined|serial)\(", code)


def _plain(calls):
    """Macro calls with the placeholders' names folded."""
    return [re.sub(r"PlaceHolder_\w+", "P", c) for c in calls]


def _statements(code, macro):
    """The macro calls and copies of one macro body, in order, each on one line."""
    body = " ".join(_calls(code, macro))
    found = re.findall(r"(?:ukernel_\w+|T\.copy)\([^()]*\)", body)
    return [re.sub(r"\(\s+", "(", re.sub(r",?\s+\)", ")", s)) for s in found]


# ---------------------------------------------------------------------------- the partial


def test_search_every_partial_tiling_serves_the_max_it_returns(pair):
    partial, _ = pair
    (m,) = [n for n in partial.nodes if type(n) is ReduceMax]
    record = m.streaming
    tilings = _tilings(partial)
    assert len(tilings) == 16 and not [t for _, t in tilings if isinstance(t, ScheduleRejected)]
    accept = streaming_accept(partial)
    cases = Counter()
    for cand, tiled in tilings:
        assert accept(cand)
        found = sites(tiled)
        tier, owner, mt = _find(tiled, m.name)
        names = [n.name for t, _ in graph_tiers(tiled) for n in t.nodes]
        assert names.count(m.name) == 1  # no second max
        shift = [o for o in output_order(tiled) if o.boundary.role == "shift"]
        assert len(shift) == 1 and str(shift[0].get_dtype(0)) == "float32"
        (edge,) = shift[0].inputs
        code, layouts = build_candidate(cand)
        if found:
            cases["looped"] += 1
            (site,) = found
            assert (site.full, site.step, site.block) == (m.name, 32, 64)
            assert site.tier == owner.name and owner.name == "GraphNode_reduction_3_10"
            # the carried max: an accumulator of the loop that steps the keys, behind the
            # two sums it repairs, read by the output through that loop
            assert sorted(n.name for n in tier.accum_nodes[:2]) == sorted(record.loops)
            assert tier.accum_nodes[2] is mt and mt.accum_init == NEG_FLT_MAX
            assert [n.name for n in tier.lag_nodes] == [f"{m.name}_prev"]
            assert sorted(a.name for a in tier.accum_scale) == sorted(record.loops)
            assert isinstance(edge, AccumEdge) and edge.accum_node is mt
            assert layouts == SPLIT_FIRST and tier.loop_kind == "pipelined"
            assert _loop_kinds(code) == ["Pipelined"]
            assert f"T.fill(t_{m.name}, {FILL})" in code
        else:
            cases["one tile"] += 1
            assert edge.src_node is mt and layouts == LOGICAL
            assert FILL not in code and "_prev" not in code
        # three stores, in the order the kernel returns its outputs: the max, then the two
        # sums. All float32: relative to a split's own max its sums are not small, and
        # float16 would not hold them
        signature = code[code.index("def main_kernel("):code.index("with T.Kernel")]
        outs = re.findall(r"(OutputNod\w): T\.Tensor\(\(.*?\), \"(\w+)\"\)", signature)
        assert outs == [("OutputNodf", "float32"), ("OutputNode", "float32"),
                        ("OutputNodg", "float32")]
    assert cases == {"looped": 8, "one tile": 8}


def test_search_the_looped_partial_emits_the_loop_of_an_attention_kernel(pair):
    """What the loop does with the max it returns: snapshot, fold in this tile's max, shift
    the scores by it, the proven factor, each sum repaired right before it accumulates.
    The hand-derived pass's rolling kernel, with the flat graph's own names."""
    partial, _ = pair
    cand, tiled = next((c, t) for c, t in _tilings(partial)
                       if sites(t) and c.n_threads == 128 and list(c.output_tile) == [1, 64, 32])
    code, _ = build_candidate(cand)
    (site,) = sites(tiled)
    assert _plain(_macro_calls(code, f"{site.tier}_iter")) == [
        "ukernel_mma_ssr_transB_clear(P, P, t_mma_5)",
        "ukernel_div_const_8_0_2dims_rr(t_mma_5, t_div_6)",
        "ukernel_copy_1dims_rr(t_reduce_max_3_7, t_reduce_max_3_7_prev)",
        "ukernel_reduce_max_axis3_rr(t_div_6, t_reduce_max_3_7)",
        "ukernel_sub_broadcast_0_2dims_rrr(t_div_6, t_reduce_max_3_7, t_sub_8)",
        "ukernel_exp_2dims_rr(t_sub_8, t_exp_9)",
        "ukernel_sub_1dims_rrr(t_reduce_max_3_7_prev, t_reduce_max_3_7, t_reduce_max_3_7_upd0)",
        "ukernel_exp_1dims_rr(t_reduce_max_3_7_upd0, t_reduce_max_3_7_upd1)",
        "ukernel_rescale_1dims_rr(t_reduce_max_3_7_upd1, t_reduction_3_10)",
        "ukernel_reduction_axis3_rr(t_exp_9, t_reduction_3_10)",
        "ukernel_rescale_lead1_2dims_rr(t_reduce_max_3_7_upd1, t_mma_11)",
        "ukernel_mma_rsr(t_exp_9_cast, P, t_mma_11)",
    ]
    main = _calls(code, "main_kernel_iter")
    assert f"T.fill(t_reduce_max_3_7, {FILL})" in main
    assert "T.clear(t_reduction_3_10)" in main and "T.clear(t_mma_11)" in main
    # TileLang keeps the start value (the most negative finite float32, not -inf) and the
    # max the loop folds in
    cuda = lower_to_cuda(code)
    assert "-0x1.fffffep+127f" in cuda and "tl::MaxOp" in cuda


# ---------------------------------------------------------------------------- the combine


def test_search_the_combine_carries_the_max_over_the_splits_in_a_plain_loop(pair):
    _, combine = pair
    (m,) = [n for n in combine.nodes if type(n) is ReduceMax]
    assert m.name == "state_max" and m.streaming is not None
    loops = m.streaming.loops  # the combined sums
    accept = streaming_accept(combine)
    cases, refused = Counter(), Counter()
    for cand, tiled in _tilings(combine):
        # the max over the splits steps with the two sums it repairs
        steps = set(_steps(cand).values())
        assert len(steps) == 1 and set(_steps(cand)) == {m.name, *loops}
        (step,) = steps
        rows = list(cand.output_tile)[1]
        if isinstance(tiled, ScheduleRejected):
            assert not accept(cand)
            refused[_reason_key(tiled)] += 1
            assert step == 1 or rows == 1, (cand.output_tile, step)
            continue
        if not accept(cand):  # built, and refused by the build after it
            with pytest.raises(ScheduleRejected) as info:
                build_candidate(cand)
            refused[_reason_key(info.value)] += 1
            assert step == 1 or rows == 1, (cand.output_tile, step)
            continue
        assert rows > 1
        code, layouts = build_candidate(cand)
        assert layouts == (((0, 1, 2), 0),)  # the output, stored as it is read
        found = sites(tiled)
        if found:
            cases["looped"] += 1
            (site,) = found
            assert (site.full, site.extent, site.step, site.block) == (m.name, FACTOR, 2, FACTOR)
            assert sorted(site.repaired) == sorted(loops)
            tier, owner, mt = _find(tiled, m.name)
            assert site.tier == owner.name and step == 2
            assert sorted(n.name for n in tier.accum_nodes) == sorted([m.name, *loops])
            assert [n.name for n in tier.lag_nodes] == [f"{m.name}_prev"]
            assert tier.loop_kind == "serial" and _loop_kinds(code) == ["serial"]
            assert mt.accum_init == NEG_FLT_MAX and f"T.fill(t_state_max, {FILL})" in code
        else:
            cases["one tile"] += 1
            assert step == FACTOR and _find(tiled, m.name)[1] is None  # in the block's tier
            assert _loop_kinds(code) == [] and FILL not in code
    assert cases == {"looped": 6, "one tile": 6}
    assert refused == {
        "state_max: one element of the reduced dim per iteration; the tile would lose that "
        "axis": 7,
        "state_max: its tile (N,) holds a single row": 1,
        # one query row a tile, all splits in it: the weights keep a unit extent
        "state_upd0 indexes N dims but its tile (N, N, N) is stored with N: a unit extent "
        "that is not a batch dim is squeezed out of the tile": 1}


def test_search_the_looped_combine_repairs_the_partials_and_its_own_sums(pair):
    """Per iteration: the running max over the splits loaded so far, each partial weighted
    by ``exp(m_i - M)`` (the proven factor with ``m := m_i``, ``m' := M``), and the sums
    repaired by ``exp(M_prev - M)`` before they accumulate (the combine's own rule): the
    online merge of the partials, found rather than written."""
    _, combine = pair
    cand, tiled = next((c, t) for c, t in _tilings(combine)
                       if not isinstance(t, ScheduleRejected) and sites(t))
    code, _ = build_candidate(cand)
    (site,) = sites(tiled)
    assert _plain(_macro_calls(code, f"{site.tier}_iter")) == [
        "ukernel_copy_1dims_rr(t_state_max, t_state_max_prev)",
        "ukernel_reduce_max_axis0_rr(P, t_state_max)",
        "ukernel_sub_broadcast_1_2dims_rrr(P, t_state_max, t_state_upd0)",
        "ukernel_exp_2dims_rr(t_state_upd0, t_state_upd1)",
        "ukernel_mul_2dims_srr(P, t_state_upd1, t_reduction_2_8_repaired)",
        "ukernel_sub_1dims_rrr(t_state_max_prev, t_state_max, t_state_max_upd0)",
        "ukernel_exp_1dims_rr(t_state_max_upd0, t_state_max_upd1)",
        "ukernel_rescale_1dims_rr(t_state_max_upd1, t_reduction_2_8_combined)",
        "ukernel_reduction_axis0_rr(t_reduction_2_8_repaired, t_reduction_2_8_combined)",
        "ukernel_mul_broadcast_0_3dims_srr(P, t_state_upd1, t_mma_9_repaired)",
        "ukernel_rescale_lead1_2dims_rr(t_state_max_upd1, t_mma_9_combined)",
        "ukernel_reduction_axis0_rr(t_mma_9_repaired, t_mma_9_combined)",
    ]
    # each partial is loaded once, the max of the splits once for its two readers, and
    # the float32 partials are weighted as they arrive; the epilogue runs after the loop
    assert _statements(code, f"{site.tier}_iter") == [
        "T.copy(PlaceHolder_state_reduction_2_8, PlaceHolder_state_reduction_2_88)",
        "T.copy(PlaceHolder_state_max, PlaceHolder_state_maxx)",
        "ukernel_copy_1dims_rr(t_state_max, t_state_max_prev)",
        "ukernel_reduce_max_axis0_rr(PlaceHolder_state_maxx, t_state_max)",
        "ukernel_sub_broadcast_1_2dims_rrr(PlaceHolder_state_maxx, t_state_max, t_state_upd0)",
        "ukernel_exp_2dims_rr(t_state_upd0, t_state_upd1)",
        "ukernel_mul_2dims_srr(PlaceHolder_state_reduction_2_88, t_state_upd1, "
        "t_reduction_2_8_repaired)",
        "ukernel_sub_1dims_rrr(t_state_max_prev, t_state_max, t_state_max_upd0)",
        "ukernel_exp_1dims_rr(t_state_max_upd0, t_state_max_upd1)",
        "ukernel_rescale_1dims_rr(t_state_max_upd1, t_reduction_2_8_combined)",
        "ukernel_reduction_axis0_rr(t_reduction_2_8_repaired, t_reduction_2_8_combined)",
        "T.copy(PlaceHolder_state_mma_9, PlaceHolder_state_mma_99)",
        "ukernel_mul_broadcast_0_3dims_srr(PlaceHolder_state_mma_99, t_state_upd1, "
        "t_mma_9_repaired)",
        "ukernel_rescale_lead1_2dims_rr(t_state_max_upd1, t_mma_9_combined)",
        "ukernel_reduction_axis0_rr(t_mma_9_repaired, t_mma_9_combined)",
    ]
    assert _statements(code, "main_kernel_iter") == [
        "ukernel_div_broadcast_0_resolve_rhs_layout_2dims_rrr(t_mma_9_combined, "
        "t_reduction_2_8_combined, t_div_10)",
        "T.copy(t_div_10, OutputNode)",
    ]
    # a plain loop down to the CUDA source: no asynchronous copy of a partial that the
    # previous iteration may still be reading (no TMA load, no barrier), and the start value
    # of the running max
    cuda = lower_to_cuda(code)
    assert "-0x1.fffffep+127f" in cuda and "tma_load" not in cuda and "mbarrier" not in cuda


def test_search_the_combine_that_holds_every_partial_takes_the_max_in_place(pair):
    """A tile with all the splits in it: the max over them, the weights and both weighted
    sums in the block's own tier, each sum cleared as it is taken, then the epilogue; no
    loop and no carried state."""
    _, combine = pair
    cand = next(c for c, t in _tilings(combine) if not isinstance(t, ScheduleRejected)
                and not sites(t) and list(c.output_tile) == [1, 16, 64])
    code, _ = build_candidate(cand)
    assert _statements(code, "main_kernel_iter") == [
        "ukernel_reduce_max_axis0_gr_clear(PlaceHolder_state_max, t_state_max)",
        "ukernel_sub_broadcast_1_2dims_grr(PlaceHolder_state_max, t_state_max, t_state_upd0)",
        "ukernel_exp_2dims_rr(t_state_upd0, t_state_upd1)",
        "ukernel_mul_broadcast_0_3dims_grr(PlaceHolder_state_mma_9, t_state_upd1, "
        "t_mma_9_repaired)",
        "ukernel_reduction_axis0_rr_clear(t_mma_9_repaired, t_mma_9_combined)",
        "ukernel_mul_2dims_grr(PlaceHolder_state_reduction_2_8, t_state_upd1, "
        "t_reduction_2_8_repaired)",
        "ukernel_reduction_axis0_rr_clear(t_reduction_2_8_repaired, t_reduction_2_8_combined)",
        "ukernel_div_broadcast_0_resolve_rhs_layout_2dims_rrr(t_mma_9_combined, "
        "t_reduction_2_8_combined, t_div_10)",
        "T.copy(t_div_10, OutputNode)",
    ]
    assert _loop_kinds(code) == [] and "_prev" not in code and FILL not in code
    assert "tl::MaxOp" in lower_to_cuda(code)
