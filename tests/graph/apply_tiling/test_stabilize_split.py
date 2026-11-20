"""The exact softmax of a split pair on tiled graphs (CPU).

``cleave.lower.scale.stabilize_split`` leaves two marked exps: the partial's, whose record
names the max the partial exports, and the combine's weights, a softmax over the split axis.
``apply_tiling`` serves both with the pass every marked exp goes through; pinned here is what
it builds for them, over every tiling of the pair lowered from a stored GQA split structure
(64 query heads, 64 queries, 4 splits of 64 keys):

* the partial reuses the exported max as its shift -- the carried max where the keys of a
  split are looped, started at the most negative finite float32 and never rescaled lazily
  -- and returns it in float32, in front of its two sums;
* the combine carries a running max over the split axis where it loops over the partials
  (in a plain loop: it weights the partials it loads), takes it in place where a tile
  holds all of them, and is refused where a step holds one partial or a tile a single row.
"""

import re
from collections import Counter

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.core.errors import ScheduleRejected
from cleave.ir.graph import AccumEdge, OutputNode
from cleave.ir.graph_utils import graph_tiers
from cleave.ir.ops import NEG_FLT_MAX, Exp, ReduceMax
from cleave.lower.scale import SHIFT
from cleave.runtime.compile import build_candidate
from cleave.runtime.reference import output_order
from cleave.schedule import apply_tiling, search_tiling
from cleave.schedule.stabilize import _reason_key, sites, stabilize_accept

from tests._lowering import lower_to_cuda
from tests.graph.apply_tiling.test_stabilize import _calls, _find, _steps
from tests.lower.test_split_softmax import _exact_pair

# shift, row sums, values. Looped: the split axis first (the two vectors padded).
SPLIT_FIRST = (((1, 0, 2), 1), ((1, 0, 2), 1), ((1, 0, 2, 3), 0))
LOGICAL = (((0, 1, 2), 0), ((0, 1, 2), 0), ((0, 1, 2, 3), 0))


@pytest.fixture(scope="module")
def pair():
    return _exact_pair(1)


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


def _statements(code, macro):
    """The macro calls and copies of one macro body, in order, each on one line."""
    body = " ".join(_calls(code, macro))
    found = re.findall(r"(?:ukernel_\w+|T\.copy)\([^()]*\)", body)
    return [re.sub(r"\(\s+", "(", re.sub(r",?\s+\)", ")", s)) for s in found]


# ---------------------------------------------------------------------------- the partial

def test_every_partial_tiling_reuses_the_exported_max(pair):
    partial, _ = pair
    (exp,) = [n for n in partial.nodes if type(n) is Exp]
    export = exp.stabilize.export
    tilings = _tilings(partial)
    assert len(tilings) == 16 and not [t for _, t in tilings if isinstance(t, ScheduleRejected)]
    accept = stabilize_accept(partial)
    cases = Counter()
    for cand, tiled in tilings:
        assert accept(cand)
        (site,) = sites(tiled)
        cases[site.case] += 1
        names = [n.name for tier, _ in graph_tiers(tiled) for n in tier.nodes]
        assert names.count(export) == 1 and f"{exp.name}_max" not in names
        tier, owner, m = _find(tiled, export)
        assert type(m) is ReduceMax and site.tier == ("main" if owner is None else owner.name)
        assert not site.lazy  # the sums stay in the frame of the max the kernel returns
        shift = [o for o in output_order(tiled) if o.boundary.role == SHIFT]
        assert len(shift) == 1 and str(shift[0].get_dtype(0)) == "float32"
        (edge,) = shift[0].inputs
        code, layouts = build_candidate(cand)
        if site.case == "rolling":
            assert site.step == 32 and site.block == site.full == 64
            # the carried max: an accumulator of the loop that tiles the keys, behind the
            # two sums it rescales (the value matmul made the loop, as in a solo kernel),
            # read by the output through that loop
            assert sorted(n.name for n in tier.accum_nodes[:2]) == sorted(
                dict(exp.stabilize.scaled))
            assert tier.accum_nodes[2] is m and owner.name == "GraphNode_mma_10"
            assert m.accum_init == NEG_FLT_MAX
            assert [n.name for n in tier.lag_nodes] == [f"{exp.name}_max_prev"]
            assert sorted(a.name for a in tier.accum_scale) == sorted(dict(exp.stabilize.scaled))
            assert isinstance(edge, AccumEdge) and edge.accum_node is m
            assert layouts == SPLIT_FIRST and tier.loop_kind == "pipelined"
            assert _loop_kinds(code) == ["Pipelined"]
            assert f"T.fill(t_{export}, -3.4028234663852886e38)" in code
        else:
            assert site.step == site.block == site.full == 64
            assert edge.src_node is m and layouts == LOGICAL
            assert "-3.4028234663852886e38" not in code and "_max_prev" not in code
        # three stores, in the order the kernel returns its outputs: the float32 shift,
        # then the two sums under the names the cut gave them
        signature = code[code.index("def main_kernel("):code.index("with T.Kernel")]
        outs = re.findall(r"(OutputNod\w): T\.Tensor\(\(.*?\), \"(\w+)\"\)", signature)
        assert outs == [("OutputNodg", "float32"), ("OutputNode", "float16"),
                        ("OutputNodf", "float16")]
    assert cases == {"rolling": 8, "in-tile": 8}


def test_the_rolling_partial_emits_the_loop_of_an_attention_kernel(pair):
    """What the loop does with the exported max: snapshot, fold in this tile's max, shift
    the scores by it, rescale both sums by ``alpha`` before they accumulate."""
    partial, _ = pair
    (exp,) = [n for n in partial.nodes if type(n) is Exp]
    e, m = exp.name, exp.stabilize.export
    rowsum, value = sorted(dict(exp.stabilize.scaled), key=lambda n: n.startswith("mma"))
    cand = next(c for c, t in _tilings(partial)
                if sites(t)[0].case == "rolling" and c.n_threads == 128
                and list(c.output_tile) == [1, 64, 32] and _steps(c)["mma_5"] == 64)
    code, _ = build_candidate(cand)
    loop = _find(apply_tiling(cand), e)[1].name
    assert [c for c in _statements(code, f"{loop}_iter") if c.startswith("ukernel_")] == [
        "ukernel_mma_ssr_transB_clear(PlaceHolder_input_100000011, PlaceHolder_input_100000022, "
        "t_mma_5)",
        "ukernel_div_const_8_0_2dims_rr(t_mma_5, t_div_6)",
        "ukernel_add_const_minus_6_25_2dims_rr(t_div_6, t_add_7)",
        f"ukernel_copy_1dims_rr(t_{m}, t_{e}_max_prev)",
        f"ukernel_reduce_max_axis3_rr(t_add_7, t_{m})",
        f"ukernel_sub_broadcast_0_2dims_rrr(t_add_7, t_{m}, t_{e}_shifted)",
        f"ukernel_exp_2dims_rr(t_{e}_shifted, t_{e})",
        f"ukernel_sub_1dims_rrr(t_{e}_max_prev, t_{m}, t_{e}_max_step)",
        f"ukernel_exp_1dims_rr(t_{e}_max_step, t_{e}_alpha)",
        f"ukernel_rescale_lead1_2dims_rr(t_{e}_alpha, t_{value})",
        f"ukernel_mma_rsr(t_{e}_cast, PlaceHolder_input_100000033, t_{value})",
        f"ukernel_rescale_1dims_rr(t_{e}_alpha, t_{rowsum})",
        f"ukernel_reduction_axis3_rr(t_{e}, t_{rowsum})",
    ]
    cuda = lower_to_cuda(code)
    assert "-0x1.fffffep+127f" in cuda and "tl::MaxOp" in cuda


# ---------------------------------------------------------------------------- the combine

def test_every_combine_tiling_is_served_or_refused_with_its_reason(pair):
    _, combine = pair
    accept = stabilize_accept(combine)
    served, refused = Counter(), Counter()
    for cand, tiled in _tilings(combine):
        step = set(_steps(cand).values())
        assert len(step) == 1  # the two sums are stepped together
        (step,) = step
        rows = list(cand.output_tile)[1]
        if isinstance(tiled, ScheduleRejected):
            assert not accept(cand)
            refused[_reason_key(tiled)] += 1
            assert step == 1 or rows == 1, (cand.output_tile, step)
            continue
        assert accept(cand) and rows > 1
        (site,) = sites(tiled)
        served[(site.case, step)] += 1
        code, layouts = build_candidate(cand)
        assert layouts == (((0, 1, 2), 0),)
        assert (site.full, site.block) == (4, 4) and site.step == step
        if site.case == "rolling":
            # a running max over the partials, in a plain loop
            tier, owner, m = _find(tiled, "exp_shift_max")
            assert m.accum_init == NEG_FLT_MAX and tier.loop_kind == "serial"
            assert sorted(n.name for n in tier.accum_nodes) == [
                "exp_shift_max", "reduction_1_2", "reduction_1_3"]
            assert _loop_kinds(code) == ["serial"]
            assert "T.fill(t_exp_shift_max, -3.4028234663852886e38)" in code
        else:
            assert _find(tiled, "exp_shift_max")[1] is None  # in the block's own tier
            assert _loop_kinds(code) == [] and "-3.4028234663852886e38" not in code
    assert served == {("rolling", 2): 6, ("in-tile", 4): 6}
    assert refused == {
        "exp_shift: one element of the softmax axis per iteration; the tile would lose that "
        "axis": 6,
        "exp_shift: its tile (N,) holds a single row": 2,
        "exp_shift: its tile (N, N, N) holds a single row": 1,
    }


def test_the_rolling_combine_weights_each_partial_by_its_shift(pair):
    """``M = max(M, max_i m_i)``, ``w = exp(m - M)``, both sums rescaled by
    ``alpha = exp(M_prev - M)`` and fed ``w * l`` / ``w * o``: the weights are float32, and
    the value partials take theirs by the lead broadcast (one weight per row of the tile).
    The shift is loaded once for its two readers; the partials are cast to float32 on the
    way to their product."""
    _, combine = pair
    cand = next(c for c, t in _tilings(combine) if not isinstance(t, ScheduleRejected)
                and sites(t)[0].case == "rolling" and list(c.output_tile) == [1, 16, 64])
    code, _ = build_candidate(cand)
    loop = _find(apply_tiling(cand), "exp_shift")[1].name
    assert _statements(code, f"{loop}_iter") == [
        "T.copy(PlaceHolder_input_10000001, PlaceHolder_input_100000011)",
        "T.copy(PlaceHolder_shift, PlaceHolder_shiftt)",
        "ukernel_copy_1dims_rr(t_exp_shift_max, t_exp_shift_max_prev)",
        "ukernel_reduce_max_axis0_rr(PlaceHolder_shiftt, t_exp_shift_max)",
        "ukernel_sub_broadcast_1_2dims_rrr(PlaceHolder_shiftt, t_exp_shift_max, "
        "t_exp_shift_shifted)",
        "ukernel_exp_2dims_rr(t_exp_shift_shifted, t_exp_shift)",
        "T.copy(PlaceHolder_input_100000011, PlaceHolder_input_100000011_cast)",
        "ukernel_mul_2dims_rrr(PlaceHolder_input_100000011_cast, t_exp_shift, "
        "t_reduction_1_2_weighted)",
        "ukernel_sub_1dims_rrr(t_exp_shift_max_prev, t_exp_shift_max, t_exp_shift_max_step)",
        "ukernel_exp_1dims_rr(t_exp_shift_max_step, t_exp_shift_alpha)",
        "ukernel_rescale_1dims_rr(t_exp_shift_alpha, t_reduction_1_2)",
        "ukernel_reduction_axis0_rr(t_reduction_1_2_weighted, t_reduction_1_2)",
        "T.copy(PlaceHolder_input_10000002, PlaceHolder_input_100000022)",
        "T.copy(PlaceHolder_input_100000022, PlaceHolder_input_100000022_cast)",
        "ukernel_mul_broadcast_0_3dims_rrr(PlaceHolder_input_100000022_cast, t_exp_shift, "
        "t_reduction_1_3_weighted)",
        "ukernel_rescale_lead1_2dims_rr(t_exp_shift_alpha, t_reduction_1_3)",
        "ukernel_reduction_axis0_rr(t_reduction_1_3_weighted, t_reduction_1_3)",
    ]
    main = "\n".join(_calls(code, "main_kernel_iter"))
    assert "ukernel_div_broadcast_0_resolve_rhs_layout_2dims_rrr(t_reduction_1_3, " \
           "t_reduction_1_2, t_div_4)" in main
    cuda = lower_to_cuda(code)
    assert "-0x1.fffffep+127f" in cuda and "tma_load" not in cuda and "mbarrier" not in cuda


def test_the_combine_that_holds_every_partial_takes_the_max_in_place(pair):
    _, combine = pair
    cand = next(c for c, t in _tilings(combine) if not isinstance(t, ScheduleRejected)
                and sites(t)[0].case == "in-tile" and list(c.output_tile) == [1, 16, 64])
    code, _ = build_candidate(cand)
    assert _statements(code, "main_kernel_iter") == [
        "ukernel_reduce_max_axis0_gr_clear(PlaceHolder_shift, t_exp_shift_max)",
        "ukernel_sub_broadcast_1_2dims_grr(PlaceHolder_shift, t_exp_shift_max, "
        "t_exp_shift_shifted)",
        "ukernel_exp_2dims_rr(t_exp_shift_shifted, t_exp_shift)",
        "T.copy(PlaceHolder_input_10000002, PlaceHolder_input_10000002_cast)",
        "ukernel_mul_broadcast_0_3dims_rrr(PlaceHolder_input_10000002_cast, t_exp_shift, "
        "t_reduction_1_3_weighted)",
        "ukernel_reduction_axis0_rr_clear(t_reduction_1_3_weighted, t_reduction_1_3)",
        "T.copy(PlaceHolder_input_10000001, PlaceHolder_input_10000001_cast)",
        "ukernel_mul_2dims_rrr(PlaceHolder_input_10000001_cast, t_exp_shift, "
        "t_reduction_1_2_weighted)",
        "ukernel_reduction_axis0_rr_clear(t_reduction_1_2_weighted, t_reduction_1_2)",
        "ukernel_div_broadcast_0_resolve_rhs_layout_2dims_rrr(t_reduction_1_3, "
        "t_reduction_1_2, t_div_4)",
        "T.copy(t_div_4, OutputNode)",
    ]
    assert "main_kernel" in lower_to_cuda(code)


def test_the_combine_output_has_no_role(pair):
    """Only the tensor the rewrite added to the cut carries a role; a role on the combine's
    output would ask a declaration keyed by role for a bound nobody gave."""
    partial, combine = pair
    assert [o.boundary.role for o in combine.nodes if isinstance(o, OutputNode)] == [""]
    assert [o.boundary.role for o in output_order(partial)] == [SHIFT, "", ""]
