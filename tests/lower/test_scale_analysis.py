"""The scale analysis: which ``Exp`` of a lowered graph is a softmax exp, and why not.

Programs carry the exact-softmax request (``tests/_softmax.py``) and are lowered through
``pipeline.plans.lower`` as their identity graph; the rules that the frontend cannot reach are
checked on hand-built IR. No kernel is involved: the verdicts are plain data on the flat graph.
"""

import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave.core.errors import LoweringRejected
from cleave.frontend import trace
from cleave.frontend.program import Op
from cleave.ir.graph import OutputNode, PlaceHolderNode
from cleave.ir.ops import (Copy, Div, Exp, Exp2, MaskedDiv, Matmul, Mul, ReduceMax, ReduceSum,
                           Stabilize)
from cleave.lower import scale as S
from cleave.lower.scale import Reduction, Scale, ScaleError, analyse, annotate, exps, record_for
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet
from tests import _graphs as G
from tests._lowering import run_lowered
from tests._softmax import stable_exp as sexp

softmax = cl.softmax  # the requested exp, its sum, the division

# ------------------------------------------------------------------ programs: softmax exps


def sm_last(x: cl.Tensor["R", "C"]):
    return softmax(x, -1)


def sm_first(x: cl.Tensor["R", "C"]):
    return softmax(x, 0)


def sm_middle(x: cl.Tensor["B", "R", "C"]):
    return softmax(x, 1)


def sm_batched_last(x: cl.Tensor["B", "R", "C"]):
    return softmax(x, -1)


def flash(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    e = sexp(x, -1)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def recip(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    e = sexp(x, -1)
    return (e @ v) * (1.0 / cl.sum(e, -1, keepdim=True))


def scaled_constants(x: cl.Tensor["R", "C"]):
    e = sexp(x, -1)
    return (e * 2.0) / cl.sum(e / 4.0, -1, keepdim=True)


def traced_attention(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"],
                     v: cl.Tensor["HKV", "SKV", "D"]):
    return softmax((q @ k.mT) / q.shape[-1] ** 0.5, -1) @ v


def flash_attention(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"],
                    v: cl.Tensor["HKV", "SKV", "D"]):
    e = sexp((q @ k.mT) / q.shape[-1] ** 0.5, -1)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def attention_plus_exp(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"],
                       v: cl.Tensor["HKV", "SKV", "D"], b: cl.Tensor["HQ", "SQ", "D"]):
    return flash_attention(q, k, v) + cl.exp(b)


def two(a: cl.Tensor["R", "C"], b: cl.Tensor["R", "C"]):
    pa, pb = sexp(a, -1), sexp(b, -1)
    s = cl.sum(pa * pb, -1, keepdim=True)
    return s / (cl.sum(pa, -1, keepdim=True) * cl.sum(pb, -1, keepdim=True))


def two_softmaxes(a: cl.Tensor["R", "C"], b: cl.Tensor["C", "D"]):
    return softmax(a, -1) @ softmax(b, -1)


def addl(x: cl.Tensor["R", "C"]):
    e = sexp(x, -1)
    l = cl.sum(e, -1, keepdim=True)
    return cl.sum(e + l, -1, keepdim=True) / l


def emax(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    e = sexp(x, -1)
    return (e @ v) / cl.max(e, -1, keepdim=True)


def colsum(x: cl.Tensor["R", "C"]):
    return cl.sum(softmax(x, -1), 0, keepdim=True)


def scaled_b(v: cl.Tensor["M", "K"], x: cl.Tensor["K", "N"]):
    e = sexp(x, 0)
    return (v @ e) / cl.sum(e, 0, keepdim=True)


def mixed_axes(x: cl.Tensor["R", "C"]):
    # the record says dim -1, the graph sums over dim 0: the axis is derived, not trusted
    e = sexp(x, -1)
    return cl.sum(e * (1.0 / cl.sum(e, 0, keepdim=True)), -1, keepdim=True)


RC, RCD = dict(R=16, C=24), dict(R=16, C=24, D=8)
MHA = dict(HQ=2, HKV=2, SQ=8, SKV=48, D=16)
GQA = dict(HQ=4, HKV=2, SQ=8, SKV=48, D=16)
NEXT_TO = r"{} reads the exp next to the finished reduction {}"

# name: (fn, sizes, [(exp shape, axes, classes of the scaled nodes, rolling pattern or "")])
ANNOTATED = {
    "last": (sm_last, RC, [((16, 24), (-1,), [ReduceSum], NEXT_TO.format(r"div_\d+", r"reduction_1_\d+"))]),
    "first": (sm_first, RC, [((16, 24), (-2,), [ReduceSum], NEXT_TO.format(r"div_\d+", r"reduction_0_\d+"))]),
    "first_square": (sm_first, dict(R=16, C=16), [((16, 16), (-2,), [ReduceSum], "next to")]),
    # one element per trailing axis: the divide still names the axis its divisor spans
    "first_unit": (sm_first, dict(R=1, C=1), [((1, 1), (-2,), [ReduceSum], "next to")]),
    "last_unit": (sm_last, dict(R=1, C=1), [((1, 1), (-1,), [ReduceSum], "next to")]),
    "middle": (sm_middle, dict(B=3, R=16, C=24), [((3, 16, 24), (-2,), [ReduceSum], "next to")]),
    "middle_square": (sm_middle, dict(B=3, R=16, C=16), [((3, 16, 16), (-2,), [ReduceSum], "next to")]),
    "middle_unit": (sm_middle, dict(B=4, R=1, C=1), [((4, 1, 1), (-2,), [ReduceSum], "next to")]),
    "batched_last": (sm_batched_last, dict(B=3, R=16, C=24), [((3, 16, 24), (-1,), [ReduceSum], "next to")]),
    "flash": (flash, RCD, [((16, 24), (-1,), [Matmul, ReduceSum], "")]),
    "reciprocal": (recip, RCD, [((16, 24), (-1,), [Matmul, ReduceSum], "")]),
    "scaled_constants": (scaled_constants, RC, [((16, 24), (-1,), [ReduceSum], "next to")]),
    "flash_attention": (flash_attention, MHA, [((2, 8, 48), (-1,), [Matmul, ReduceSum], "")]),
    "flash_attention_gqa": (flash_attention, GQA, [((4, 8, 48), (-1,), [Matmul, ReduceSum], "")]),
    "traced_attention": (traced_attention, MHA, [
        ((2, 8, 48), (-1,), [ReduceSum], NEXT_TO.format(r"div_\d+", r"reduction_2_\d+"))]),
    "attention_plus_exp": (attention_plus_exp, MHA, [
        ((2, 8, 48), (-1,), [Matmul, ReduceSum], ""), ((2, 8, 16), None, None, None)]),
    "two": (two, RC, [
        ((16, 24), (-1,), [ReduceSum, ReduceSum],
         r"reduction_1_\d+ is also scaled by exp_3; the rolling form keeps one factor per accumulator"),
        ((16, 24), (-1,), [ReduceSum, ReduceSum], r"reduction_1_\d+ is also scaled by exp_2")]),
    "two_softmaxes": (two_softmaxes, RCD, [((16, 24), (-1,), [ReduceSum], "next to"),
                                           ((24, 8), (-1,), [ReduceSum], "next to")]),
    "addl": (addl, RC, [((16, 24), (-1,), [ReduceSum, ReduceSum],
                         NEXT_TO.format(r"add_\d+", r"reduction_1_\d+"))]),
    "emax": (emax, RCD, [((16, 24), (-1,), [Matmul, ReduceMax],
                          r"reduce_max_1_\d+ takes a max of the scaled value; a rescaled "
                          "accumulator must start at zero")]),
    "colsum": (colsum, RC, [((16, 24), (-1,), [ReduceSum], "next to")]),
    "scaled_b": (scaled_b, dict(M=8, K=24, N=16), [
        ((24, 16), (-2,), [Matmul, ReduceSum],
         r"mma_\d+ carries the factor on axes \(None, 1\), not on its leading axes")]),
    "mixed_axes": (mixed_axes, RC, [((16, 24), (-2,), [ReduceSum], "next to")]),
}

# ------------------------------------------------------------------ programs: refused


def exp_ratio(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]):
    return sexp(x, -1) / sexp(y, -1)


def merged_with_exp(x: cl.Tensor["R", "C"]):
    # what the search makes of ``softmax(x) * exp(x)``: one exp behind both uses
    e = sexp(x, -1)
    return (e / cl.sum(e, -1, keepdim=True)) * e


def two_axes(x: cl.Tensor["R", "C"], w: cl.Tensor["R", "C"]):
    e = sexp(x, -1)
    return (cl.sum(cl.sum(e * w, -1, keepdim=True), -2, keepdim=True)
            / cl.sum(cl.sum(e, -1, keepdim=True), -2, keepdim=True))


def cube(x: cl.Tensor["R", "C"]):
    e = sexp(x, -1)
    return (e * e) / cl.sum(e, -1, keepdim=True)


def squares(x: cl.Tensor["R", "C"]):
    e = sexp(x, -1)
    return (e * e) / cl.sum(e * e, -1, keepdim=True)


def squares_over_columns(x: cl.Tensor["R", "C"]):
    e = sexp(x, -1)
    return (e * e) / cl.sum(e * e, 0, keepdim=True)


def squares_first_axis(x: cl.Tensor["R", "C"]):
    e = sexp(x, 0)
    return (e * e) / cl.sum(e * e, 0, keepdim=True)


def inverse_sum(x: cl.Tensor["R", "C"]):
    e = sexp(x, -1)
    return cl.sum(1.0 / e, -1, keepdim=True) * cl.sum(e, -1, keepdim=True)


def vector(x: cl.Tensor["C"], w: cl.Tensor["C"]):
    e = sexp(x, 0)
    return cl.sum(e * w, 0, keepdim=True) / cl.sum(e, 0, keepdim=True)


def batch_axis(x: cl.Tensor["B", "R", "C"], w: cl.Tensor["B", "R", "C"]):
    e = sexp(x, 0)
    return cl.sum(e * w, 0, keepdim=True) / cl.sum(e, 0, keepdim=True)


# The plain softmax of the three shapes above divides by a value no binary op can read (a
# divisor that is broadcast along two axes, along the only axis of a vector, along a batch
# axis): the conversion refuses it before the analysis runs.

def two_axes_softmax(x: cl.Tensor["R", "C"]):
    e = sexp(x, -1)
    return e / cl.sum(cl.sum(e, -1, keepdim=True), -2, keepdim=True)


def vector_softmax(x: cl.Tensor["C"]):
    return softmax(x, 0)


def batch_axis_softmax(x: cl.Tensor["B", "R", "C"]):
    return softmax(x, 0)


def unit_row(x: cl.Tensor["B", "F"], v: cl.Tensor["ONE", "F"]):
    # out[i, j] = (v . e_j) / sum(e_i) keeps the factor; with B == 4 on both axes the divide
    # used to lower as a product along the columns, where the factor cancels
    e = sexp(x, -1)
    return (v @ e.mT) / cl.sum(e, -1, keepdim=True)


def kv_side(q: cl.Tensor["HQ", "S", "D"], k: cl.Tensor["HKV", "S", "D"]):
    return q @ sexp(k, -1).mT


def gram(x: cl.Tensor["R", "C"]):
    e = sexp(x, -1)
    return e @ e.mT


def different_axes(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "R"], u: cl.Tensor["R", "C"]):
    e = sexp(x, -1)
    return (e @ v) * (u @ e.mT)


def _unary_of(op):
    def fn(x: cl.Tensor["R", "C"]):
        e = sexp(x, -1)
        return op(e) / cl.sum(e, -1, keepdim=True)
    return fn


def add_tensor(x: cl.Tensor["R", "C"]):
    e = sexp(x, -1)
    return (e + x) / cl.sum(e, -1, keepdim=True)


def sub_tensor(x: cl.Tensor["R", "C"]):
    e = sexp(x, -1)
    return (x - e) / cl.sum(e, -1, keepdim=True)


def two_requests_one_cancels(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]):
    return softmax(x, -1) * sexp(y, -1)


def one_request_recorded_twice(x: cl.Tensor["R", "C"]):
    return softmax(x, -1) * sexp(x, -1)


# name: (fn, sizes, the reason after "softmax cannot be stabilized: ")
REFUSED = {
    "exp_ratio": (exp_ratio, RC, r"exp_2: nothing sums it, so there is no softmax axis"),
    "softmax_times_its_exp": (merged_with_exp, RC, r"exp_1: output of mul_4 keeps degree 1 in its scale"),
    "two_axes": (two_axes, RC, r"exp_2: the factor must be constant along axes \[0, 1\]; one softmax "
                               "axis is supported"),
    "cube": (cube, RC, r"exp_1: output of div_4 keeps degree 1 in its scale"),
    "squares": (squares, RC, r"exp_1: reduction_1_4 accumulates a value of degree 2"),
    "squares_over_columns": (squares_over_columns, RC, r"exp_1: reduction_0_4 accumulates a value of degree 2"),
    "squares_first_axis": (squares_first_axis, RC, r"exp_1: reduction_0_4 accumulates a value of degree 2"),
    "inverse_sum": (inverse_sum, RC, r"exp_1: reduction_1_3 accumulates a value of degree -1"),
    "vector": (vector, dict(C=24), r"exp_2: the exp has 1 non-batch axes; the softmax axis and one "
                                   "row axis are supported"),
    "batch_axis": (batch_axis, dict(B=3, R=16, C=24), r"exp_2: softmax axis 0 is a batch axis of the exp"),
    "kv_side": (kv_side, dict(HQ=4, HKV=2, S=8, D=16),
                r"mma_3: Matmul broadcasts its scaled operand 1 over the batch \(\(2,\) -> \(4,\)\)"),
    "gram": (gram, RC, r"mma_2: Matmul of two scaled values"),
    "different_axes": (different_axes, RC, r"mul_6: Mul of two scaled values on different axes "
                                           r"\(0, None\) vs \(None, 0\)"),
    "add_constant": (_unary_of(lambda e: e + 1.0), RC, r"add_2: Add of a constant and a scaled value"),
    "add_tensor": (add_tensor, RC, r"add_2: Add of degrees 1 and 0"),
    "sub_tensor": (sub_tensor, RC, r"sub_2: Sub of degrees 0 and 1"),
    "silu": (_unary_of(cl.silu), RC, r"silu_2: Silu of a scaled value"),
    "sqrt": (_unary_of(cl.sqrt), RC, r"sqrt_2: Sqrt of a scaled value"),
    "rsqrt": (_unary_of(cl.rsqrt), RC, r"rsqrt_3: RSqrt of a scaled value"),
    "exp": (_unary_of(cl.exp), RC, r"exp_2: Exp of a scaled value"),
    "two_requests_one_cancels": (two_requests_one_cancels, RC,
                                 r"2 softmax request\(s\), 1 exp\(s\) of this graph can be stabilized "
                                 r"\(exp_5: nothing sums it, so there is no softmax axis\)"),
    "one_request_recorded_twice": (one_request_recorded_twice, RC,
                                   r"exp_4: nothing sums it, so there is no softmax axis"),
}

BINARY_RULE = ("; a binary op keeps the shape of its first operand and broadcasts the second "
               "along one of its last two axes")
# name: (fn, sizes, the reason after "graph 0: ValueError: "): refused by the conversion
UNREADABLE = {
    "two_axes_softmax": (two_axes_softmax, RC,
                         r"kn_div_op at index 4: the operand \(\) is broadcast along axes 0, 1 of the "
                         r"first operand \(16, 24\) \(dims \(16, 24\) and \(1, 1\)\)" + BINARY_RULE),
    "vector_softmax": (vector_softmax, dict(C=24),
                       r"kn_div_op at index 3: the operand \(\) is broadcast along axis 0 of the first "
                       r"operand \(24,\) \(dims \(24,\) and \(1,\)\)" + BINARY_RULE),
    "batch_axis_softmax": (batch_axis_softmax, dict(B=3, R=16, C=24),
                           r"kn_div_op at index 3: the operand \(16, 24\) is broadcast along axis 0 of "
                           r"the first operand \(3, 16, 24\) \(dims \(3, 16, 24\) and \(1, 16, 24\)\)"
                           + BINARY_RULE),
    "unit_row": (unit_row, dict(B=4, F=2, ONE=1),
                 r"kn_div_op at index 5: the operand \(4,\) has extent 4 where the first operand "
                 r"\(1, 4\) has 1 \(dims \(1, 4\) and \(4, 1\)\)" + BINARY_RULE),
}


def _lower(fn, sizes):
    prog = trace(fn, sizes, cl.f32)
    (graph,) = GraphSet.identity(prog)
    (ir,) = lower(graph)
    return prog, ir


def _node(ir, name):
    return next(n for n in ir.nodes if n.name == name)


# ------------------------------------------------------------------ verdict table

@pytest.mark.parametrize("name", ANNOTATED)
def test_softmax_exps_are_annotated(name):
    fn, sizes, expected = ANNOTATED[name]
    _, ir = _lower(fn, sizes)
    found = exps(ir)
    assert [tuple(int(s) for s in e.get_shape()) for e in found] == [shape for shape, *_ in expected]
    for exp, (_, axes, classes, rolling) in zip(found, expected):
        record = exp.stabilize
        if axes is None:  # a hand-written exp next to the softmax stays a plain exp
            assert record is None
            continue
        assert isinstance(record, Stabilize) and record.axes == axes and record.export is None
        assert [type(_node(ir, n)) for n, _ in record.scaled] == classes
        assert all(degree == 1 for _, degree in record.scaled)
        if rolling:
            assert re.search(rolling, record.rolling), record.rolling
        else:
            assert record.rolling == ""


@pytest.mark.parametrize("name", REFUSED)
def test_other_exps_are_refused_with_the_node_that_breaks(name):
    fn, sizes, reason = REFUSED[name]
    prog = trace(fn, sizes, cl.f32)
    (graph,) = GraphSet.identity(prog)
    with pytest.raises(LoweringRejected) as e:
        lower(graph)
    assert re.fullmatch(r"graph 0: softmax cannot be stabilized: " + reason, str(e.value)), e.value


@pytest.mark.parametrize("name", UNREADABLE)
def test_divisor_no_binary_op_can_read_is_refused_by_the_conversion(name):
    """Not the analysis' call: such a graph used to reach it with an operand of another
    shape wired into the divide (``unit_row`` even passed it, on the wrong axis)."""
    fn, sizes, reason = UNREADABLE[name]
    prog = trace(fn, sizes, cl.f32)
    (graph,) = GraphSet.identity(prog)
    with pytest.raises(LoweringRejected) as e:
        lower(graph)
    assert re.fullmatch(r"graph 0: ValueError: " + reason, str(e.value)), e.value


def test_plain_exp_next_to_a_softmax_keeps_its_reason():
    _, ir = _lower(attention_plus_exp, MHA)
    softmax_exp, plain = exps(ir)
    assert softmax_exp.stabilize is not None and plain.stabilize is None
    with pytest.raises(ScaleError, match=r"add_\d+: Add of degrees 0 and 1"):
        record_for(ir, plain)


def test_flash_and_traced_differ_only_in_the_rolling_form():
    """Both orders cancel; only ``(e @ v) / sum(e)`` can be accumulated tile by tile."""
    _, flash_ir = _lower(flash_attention, MHA)
    _, traced_ir = _lower(traced_attention, MHA)
    (f,), (t,) = exps(flash_ir), exps(traced_ir)
    assert f.stabilize.axes == t.stabilize.axes == (-1,)
    assert f.stabilize.rolling == "" and "next to the finished reduction" in t.stabilize.rolling
    # the traced order divides first: the value matmul reads a finished softmax, not a scale
    assert [type(_node(traced_ir, n)) for n, _ in t.stabilize.scaled] == [ReduceSum]


def test_lowering_twice_gives_equal_records_on_fresh_nodes():
    prog = trace(flash_attention, MHA, cl.f32)
    (graph,) = GraphSet.identity(prog)
    (a,), (b,) = lower(graph), lower(graph)
    (ea,), (eb,) = exps(a), exps(b)
    assert ea is not eb and ea.stabilize == eb.stabilize


def test_unrequested_programs_are_left_alone():
    """The record is written for a requesting program only: same graph, no mark."""
    def twin(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
        e = cl.exp(x)
        return (e @ v) / cl.sum(e, -1, keepdim=True)

    _, ir = _lower(twin, RCD)
    (exp,) = exps(ir)
    assert exp.stabilize is None
    assert record_for(ir, exp).rolling == ""  # the analysis itself does not need the request


# ------------------------------------------------------------------ report and fixed point

def test_report_of_the_flash_form():
    _, ir = _lower(flash, RCD)
    (exp,) = exps(ir)
    report = analyse(ir, exp)
    mma = next(n for n in ir.nodes if type(n) is Matmul)
    rowsum = next(n for n in ir.nodes if type(n) is ReduceSum)
    assert (report.exp, report.rank, report.const) == (exp.name, 2, (1,))
    assert report.reductions == (Reduction(mma.name, "Matmul", 0, 1, 1),
                                 Reduction(rowsum.name, "ReduceSum", 0, 1, 1))
    # the factor varies with the row only, and sits on the leading axis of both sums
    assert report.scaled == {exp.name: Scale(1, (0, None)), mma.name: Scale(1, (0, None)),
                             rowsum.name: Scale(1, (0,))}
    assert [s.deg for s in report.outputs.values()] == [0]


def test_const_is_a_fixed_point():
    """``sum(sum(e, -1), -2)``: the second axis is only found once the first is constant."""
    prog = trace(two_axes, RC, cl.f32)
    ir = _plain_ir(prog)
    (exp,) = exps(ir)
    assert analyse(ir, exp).const == (0, 1)


def test_axis_refusals_wait_for_the_fixed_point():
    """In the first pass ``e / sum(e)`` has its operands on different axes: the sum has lost
    the softmax axis, the exp has not been told yet that the factor is constant along it."""
    ir = _plain_ir(trace(sm_last, RC, cl.f32))
    (exp,) = exps(ir)
    *_, found, _, pending = S._propagate(ir, exp, set(), None)
    assert found == {1} and len(pending) == 1 and "on different axes" in pending[0]
    *_, found, _, pending = S._propagate(ir, exp, {1}, None)
    assert found == set() and pending == []
    assert analyse(ir, exp).const == (1,)


def test_seeded_inputs_carry_the_factor():
    """``analyse(graph, seeds=...)``: placeholders that arrive scaled (a combine's inputs)."""
    o = Div((4, 16), broadcast=0, name="div_0")
    num, den = (e.src_node for e in sorted(o.inputs, key=lambda e: e.dst_id))
    OutputNode(o, dtype="float32")
    graph = G._finalize(o)
    seeds = {num.name: Scale(1, (0, None)), den.name: Scale(1, (0,))}
    report = analyse(graph, None, seeds)
    assert report.exp is None and report.const == ()
    assert [s for s in report.outputs.values()] == [Scale(0, (None, None))]
    half = analyse(graph, None, {num.name: Scale(1, (0, None))})
    assert [s for s in half.outputs.values()] == [Scale(1, (0, None))]
    assert half.scaled == {"div_0": Scale(1, (0, None))}
    with pytest.raises(ScaleError, match="div_0: Div of two scaled values on different axes"):
        analyse(graph, None, {num.name: Scale(1, (None, 0)), den.name: Scale(1, (0,))})


# ------------------------------------------------------------------ rules on hand-built IR

def _chain(*nodes):
    """Wire ``nodes`` in a row (operand 0 each) and finish the graph at the last one."""
    for src, dst in zip(nodes, nodes[1:]):
        G._connect(src, dst, 0, 0)
    OutputNode(nodes[-1], dtype="float32")
    return G._finalize(nodes[-1])


def _normalized(exp, *between, axis=-1):
    """``f(exp) / rowsum(exp)`` with the ops of ``between`` on the numerator."""
    dims = tuple(exp.dims)
    rowsum = ReduceSum(dims, axis=axis, name="rowsum_0")
    div = Div(dims, broadcast=0 if axis in (-1, len(dims) - 1) else 1, name="div_0")
    G._connect(exp, rowsum, 0, 0)
    for src, dst in zip((exp, *between), (*between, div)):
        G._connect(src, dst, 0, 0)
    G._connect(rowsum, div, 0, 1)
    OutputNode(div, dtype="float32")
    return G._finalize(div)


def test_hand_built_softmax_is_recognised():
    exp = Exp((4, 16), name="exp_0")
    graph = _normalized(exp)
    record = record_for(graph, exp)
    assert record == Stabilize((-1,), (("rowsum_0", 1),),
                               "div_0 reads the exp next to the finished reduction rowsum_0")


def test_unknown_class_on_a_scaled_value_has_no_identity_fall_through():
    exp = Exp((4, 16), name="exp_0")
    with pytest.raises(ScaleError, match="exp2_0: Exp2 has no scale rule"):
        record_for(_normalized(exp, Exp2((4, 16), name="exp2_0")), exp)
    exp = Exp((4, 16), name="exp_0")
    with pytest.raises(ScaleError, match="copy_0: Copy has no scale rule"):
        record_for(_normalized(exp, Copy((4, 16), name="copy_0")), exp)


def test_rules_match_the_exact_class():
    """``MaskedDiv`` is a ``Div`` subclass with its own semantics: no rule is inherited."""
    exp = Exp((4, 16), name="exp_0")
    rowsum = ReduceSum((4, 16), name="rowsum_0")
    div = MaskedDiv((4, 16), broadcast=0, name="mdiv_0")
    G._connect(exp, rowsum, 0, 0)
    G._connect(exp, div, 0, 0)
    G._connect(rowsum, div, 0, 1)
    OutputNode(div, dtype="float32")
    with pytest.raises(ScaleError, match="mdiv_0: MaskedDiv has no scale rule"):
        record_for(G._finalize(div), exp)


def test_any_class_is_fine_on_an_unscaled_value():
    exp = Exp((4, 16), name="exp_0")
    rowsum = ReduceSum((4, 16), name="rowsum_0")
    div = Div((4, 16), broadcast=0, name="div_0")
    G._connect(exp, rowsum, 0, 0)
    G._connect(exp, div, 0, 0)
    G._connect(rowsum, div, 0, 1)
    graph = _chain(div, Copy((4, 16), name="copy_0"), Exp2((4, 16), name="exp2_0"))
    assert record_for(graph, exp).axes == (-1,)


def _self_ratio(exp, axis):
    """``rowsum / rowsum``: degree 0 at the output, whatever the shape of the exp."""
    dims = tuple(exp.dims)
    rowsum = ReduceSum(dims, axis=axis, name="rowsum_0")
    ratio = Div(tuple(rowsum.output_dims), name="div_0")
    G._connect(exp, rowsum, 0, 0)
    G._connect(rowsum, ratio, 0, 0)
    G._connect(rowsum, ratio, 0, 1)
    OutputNode(ratio, dtype="float32")
    return G._finalize(ratio)


def test_softmax_over_a_batch_axis_is_refused():
    exp = Exp((3, 4, 16), name="exp_0")  # one batch axis
    with pytest.raises(ScaleError, match="exp_0: softmax axis 0 is a batch axis of the exp"):
        record_for(_self_ratio(exp, 0), exp)
    exp = Exp((3, 4, 16), name="exp_0")
    assert record_for(_self_ratio(exp, 1), exp).axes == (-2,)
    exp = Exp((3, 4, 16), name="exp_0")
    assert record_for(_self_ratio(exp, 2), exp).axes == (-1,)


def test_scaled_operand_of_another_shape_is_refused():
    """The axis maps hold for the operand an op declares. The conversion no longer wires any
    other (``tests/lower/test_broadcast_axis.py``); a hand-built graph still can."""
    exp = Exp((3, 16, 24), name="exp_0")
    colsum = ReduceSum((3, 16, 24), axis=0, name="colsum_0")  # (16, 24)
    div = Div((3, 16, 24), name="div_0")  # declares a (3, 16, 24) divisor
    G._connect(exp, colsum, 0, 0)
    G._connect(exp, div, 0, 0)
    G._connect(colsum, div, 0, 1)
    OutputNode(div, dtype="float32")
    with pytest.raises(ScaleError, match=r"^div_0: scaled operand 1 has shape \(16, 24\), Div takes "
                                         r"\(3, 16, 24\)$"):
        record_for(G._finalize(div), exp)


@pytest.mark.parametrize("dims, n_batch, count", [((2, 4, 16), 0, 3), ((2, 4, 16), 2, 1)])
def test_exp_needs_the_softmax_axis_and_one_row_axis(dims, n_batch, count):
    exp = Exp(dims, name="exp_0", n_batch_dims=n_batch)
    with pytest.raises(ScaleError, match=f"exp_0: the exp has {count} non-batch axes"):
        record_for(_self_ratio(exp, 2), exp)


def test_analysed_exp_must_not_read_a_scaled_value():
    exp = Exp((4, 16), name="exp_0")
    graph = _normalized(exp)
    (ph,) = [n for n in graph.nodes if isinstance(n, PlaceHolderNode)]
    with pytest.raises(ScaleError, match="exp_0: the analysed exp reads a scaled value"):
        analyse(graph, exp, {ph.name: Scale(1, (0, None))})


# ------------------------------------------------------------------ the graph-level rule

class _Program:
    """What ``annotate`` reads of a program: its requests and whether it has a plain exp."""

    def __init__(self, requests, plain_exp):
        self.stabilize_requests = tuple((i, 1) for i in range(requests))
        self.ops = ([Op("exp", (i,), (("stabilize", 1),), 100 + i) for i in range(requests)]
                    + [Op("exp", (50,), (), 200)] * plain_exp)


def _two_exp_graph():
    """One softmax exp and one exp whose factor does not cancel (it is added)."""
    _, ir = _lower(attention_plus_exp, MHA)
    for exp in exps(ir):
        exp.stabilize = None
    return ir


def test_annotate_marks_what_cancels_and_leaves_a_handwritten_exp():
    ir = _two_exp_graph()
    marked = annotate(ir, _Program(requests=1, plain_exp=1))
    assert [m.name for m in marked] == [exps(ir)[0].name]
    assert exps(ir)[1].stabilize is None


def test_annotate_refuses_when_fewer_exps_cancel_than_were_requested():
    ir = _two_exp_graph()
    with pytest.raises(ScaleError, match=r"2 softmax request\(s\), 1 exp\(s\) of this graph can be "
                                         r"stabilized \(add_\d+: Add of degrees 0 and 1\)"):
        annotate(ir, _Program(requests=2, plain_exp=1))
    assert [e.stabilize for e in exps(ir)] == [None, None]  # nothing is half-marked


def test_annotate_refuses_a_plain_exp_the_program_never_wrote():
    """Every exp of the program is a softmax exp, so an exp left plain is a requested one."""
    ir = _two_exp_graph()
    with pytest.raises(ScaleError, match=r"^add_\d+: Add of degrees 0 and 1$"):
        annotate(ir, _Program(requests=1, plain_exp=0))
    assert [e.stabilize for e in exps(ir)] == [None, None]


def test_annotate_refuses_a_graph_without_a_softmax_exp():
    ir = _plain_ir(trace(cube, RC, cl.f32))
    with pytest.raises(ScaleError, match=r"^exp_1: output of div_4 keeps degree 1 in its scale$"):
        annotate(ir, _Program(requests=1, plain_exp=0))

    def no_exp(x: cl.Tensor["R", "C"]):
        return x * 2.0

    with pytest.raises(ScaleError, match="^the lowered graph has no exp$"):
        annotate(_plain_ir(trace(no_exp, RC, cl.f32)), _Program(requests=1, plain_exp=0))


def _plain_ir(prog):
    """The lowered identity graph without the hook (``to_ir`` only)."""
    from cleave.lower.convert import to_ir

    return to_ir(prog.to_graph_structure(), dtype="float32", out_dtype="float32")


# ------------------------------------------------------------------ IR fixtures

@pytest.mark.parametrize("build, scaled, rolling", [
    (lambda: G.mha_bhsd_graph(1, 4, 64, 256, 64), ("mma_1", "rowsum_0"), ""),
    (lambda: G.gqa_bhsd_graph(1, 8, 2, 64, 256, 64), ("mma_1", "rowsum_0"), ""),
    (lambda: G.mqa_bhsd_graph(1, 8, 1, 64, 256, 64), ("mma_1", "rowsum_0"), ""),
    (lambda: G.qknorm_graph(1, 8, 2, 64, 256, 64), ("mma_1", "rowsum_0"), ""),
    (lambda: G.mla_prefill_bhsd_graph(1, 4, 64, 256, 48, 32), ("mma_1", "rowsum_0"), ""),
    (lambda: G.mla_decode_bhsd_graph(1, 16, 256, 64, 16), ("mma_1", "rowsum_0"), ""),
    (lambda: G.matmul_softmax_graph(1, 4, 64, 256, 64), ("rowsum_0",),
     "div_0 reads the exp next to the finished reduction rowsum_0"),
], ids=["mha", "gqa", "mqa", "qknorm", "mla_prefill", "mla_decode", "matmul_softmax"])
def test_fixed_shift_fixtures_have_a_softmax_exp(build, scaled, rolling):
    """The fixed shift in front of the exp is an unscaled value: the analysis sees through it.
    Grouped (GQA / MQA) batch broadcast sits on the K / V side, never on the scaled operand."""
    graph = build()
    (exp,) = exps(graph)
    assert record_for(graph, exp) == Stabilize((-1,), tuple((n, 1) for n in scaled), rolling)
    assert exp.stabilize is None  # record_for decides, annotate writes


@pytest.mark.parametrize("build", [lambda: G.mha_split_partial_graph(1, 4, 4, 64, 64, 64),
                                   lambda: G.gqa_split_partial_graph(1, 8, 2, 4, 64, 64, 64)],
                         ids=["mha", "gqa"])
def test_split_partial_keeps_the_factor_in_both_outputs(build):
    """Not a solo softmax: each split's sums are relative to that split's own scale."""
    graph = build()
    (exp,) = exps(graph)
    with pytest.raises(ScaleError, match="exp_0: output of rowsum_0 keeps degree 1 in its scale"):
        record_for(graph, exp)
    report = analyse(graph, exp)
    assert report.const == (3,)
    assert [s.deg for s in report.outputs.values()] == [1, 1]
    # ...on the same leading axes, which is what a combine can undo
    assert [tuple(a for a in s.ax if a is not None) for s in report.outputs.values()] == [(0, 1, 2)] * 2


# ------------------------------------------------------------------ false-cancel sweep

SWEEP = {name: (fn, sizes) for name, (fn, sizes, _) in {**ANNOTATED, **REFUSED, **UNREADABLE}.items()}
UNSCALED = {"b"}  # the operand of a hand-written exp: nothing keeps that one finite at x100


def _inputs(prog, scale):
    torch.manual_seed(0)
    return [torch.randn(*s.shape) * (1.0 if s.name in UNSCALED else scale) for s in prog.inputs]


def _with_and_without(ir, prog, inputs):
    marked = [n for n in ir.nodes if getattr(n, "stabilize", None) is not None]
    (annotated,) = run_lowered(ir, prog, inputs)
    saved = [(n, n.stabilize) for n in marked]
    for n, _ in saved:
        n.stabilize = None
    try:
        (plain,) = run_lowered(ir, prog, inputs)
    finally:
        for n, record in saved:
            n.stabilize = record
    return annotated, plain


def _distance(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


@pytest.mark.parametrize("name", SWEEP)
def test_no_false_cancel(name):
    """A marked Exp must not change the graph: evaluate it with and without the records.

    On benign inputs the plain ``exp`` is exact, so any difference beyond rounding would be a
    wrong "the shift cancels". A refused program has nothing marked and nothing to compare.
    """
    fn, sizes = SWEEP[name]
    prog = trace(fn, sizes, cl.f32)
    (graph,) = GraphSet.identity(prog)
    try:
        (ir,) = lower(graph)
    except LoweringRejected:
        assert name in REFUSED or name in UNREADABLE
        return
    assert name in ANNOTATED
    inputs = _inputs(prog, 1.0)
    annotated, plain = _with_and_without(ir, prog, inputs)
    assert torch.isfinite(plain).all() and torch.isfinite(annotated).all()
    assert _distance(annotated, plain) <= 1e-6
    # the record is what keeps the graph finite on large logits
    large, _ = _with_and_without(ir, prog, _inputs(prog, 100.0))
    assert torch.isfinite(large).all()


@pytest.mark.parametrize("name", ["cube", "softmax_times_its_exp", "squares_over_columns"])
def test_sweep_has_teeth(name):
    """Mark by hand an Exp the analysis refuses: the sweep's measure sees the factor."""
    fn, sizes = SWEEP[name]
    prog = trace(fn, sizes, cl.f32)
    ir = _plain_ir(prog)
    (exp,) = exps(ir)
    with pytest.raises(ScaleError):
        record_for(ir, exp)
    exp.stabilize = Stabilize((-1,))
    annotated, plain = _with_and_without(ir, prog, _inputs(prog, 1.0))
    assert _distance(annotated, plain) > 1e-2
