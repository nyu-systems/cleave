"""Axes of a lowered graph come from the structure, never from extents that happen to agree.

``e / sum(e, dim)`` lowers to ``Div(e, rowsum)`` with ``broadcast == 0`` (the operand spans
the second-to-last axis) or ``broadcast == 1`` (it spans the last one). With equal trailing
extents the two value shapes are the same, so the index has to come from the structure:
the symbolic dims of the two tensors, else the axis the operand's producer reduced.
A reduction drops its axis from the value while the structure keeps it as a unit dim, so
the axis of a second reduction has to be re-counted as well.
The lowered graph is compared with ``Program.reference``, which never sees the IR.

A binary op keeps the shape of its first operand and broadcasts the second along one of its
last two axes. A program that asks for anything else is refused with the op and the reason:
it used to lower to an op with an operand of another shape wired in, or -- where two extents
happened to agree -- to an op on the wrong axis. An op that no output reads is not lowered.
"""

import random
import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import LoweringRejected, SpecError
from cleave.frontend import trace
from cleave.frontend.program import Program
from cleave.ir.graph import OutputNode, PlaceHolderNode
from cleave.ir.ops import Add, Div, Exp, Matmul, Mul, ReduceMax, ReduceSum, Sub
from cleave.lower.convert import to_ir
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.reference import execute, placeholder_order

SHAPES = [
    # (annotation dims, sizes, softmax dim, expected broadcast index of the Div)
    (("R", "C"), dict(R=16, C=24), -1, 0),
    (("R", "C"), dict(R=16, C=24), 0, 1),
    (("R", "C"), dict(R=16, C=16), -1, 0),
    (("R", "C"), dict(R=16, C=16), 0, 1),  # off by 1.8x of the max before the fix
    (("N", "N"), dict(N=16), -1, 0),
    (("N", "N"), dict(N=16), 0, 1),  # one symbol for both axes
    (("B", "R", "C"), dict(B=3, R=16, C=24), 1, 1),
    (("B", "R", "C"), dict(B=3, R=16, C=16), -1, 0),
    (("B", "R", "C"), dict(B=3, R=16, C=16), 1, 1),  # off by 4.5x of the max before the fix
    (("B", "N", "N"), dict(B=3, N=16), 1, 1),
]
UNIT_SHAPES = [
    # One element along the softmax axis, the row axis or both: either index reads the same
    # values, but the index still names the axis the divisor is broadcast along. The search
    # drops a sum over a unit axis, so these are identity only.
    (("R", "C"), dict(R=1, C=1), -1, 0),
    (("R", "C"), dict(R=1, C=1), 0, 1),  # said 0 before: the two shapes cannot tell
    (("B", "R", "C"), dict(B=4, R=1, C=1), -1, 0),
    (("B", "R", "C"), dict(B=4, R=1, C=1), 1, 1),
    (("R", "C"), dict(R=1, C=16), 0, 1),
    (("R", "C"), dict(R=16, C=1), -1, 0),
    (("R", "C"), dict(R=16, C=1), 0, 1),
    (("R", "C"), dict(R=1, C=16), -1, 0),
]


def _ids(shapes):
    return [f"{''.join(dims)}_{'x'.join(str(s[d]) for d in dims)}_dim{dim}" for dims, s, dim, _ in shapes]


IDS = _ids(SHAPES)


def _program(dims, sizes, dim):
    def softmax(x):
        e = cl.exp(x)  # the shift-free form: exp / sum / div and nothing else
        return e / cl.sum(e, dim, keepdim=True)

    softmax.__annotations__ = {"x": cl.Tensor[dims]}
    return trace(softmax, sizes, cl.f32)


def _check(ir, prog, dim, want_index):
    (div,) = [n for n in ir.nodes if type(n) is Div]
    assert div.broadcast == want_index
    torch.manual_seed(0)
    x = torch.randn(*prog.inputs[0].shape)
    (got,) = execute(ir, [x])
    (want,) = prog.reference(x)
    torch.testing.assert_close(want, torch.softmax(x, dim), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got.reshape(want.shape), want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dims, sizes, dim, index", SHAPES + UNIT_SHAPES, ids=_ids(SHAPES + UNIT_SHAPES))
def test_identity_graph_matches_program(dims, sizes, dim, index):
    prog = _program(dims, sizes, dim)
    (graph,) = GraphSet.identity(prog)
    (ir,) = lower(graph)
    _check(ir, prog, dim, index)


@pytest.mark.parametrize("dims, sizes, dim, index", SHAPES, ids=IDS)
def test_search_graphs_match_program(dims, sizes, dim, index):
    """Searched structures carry symbols, and one extent may stand behind two of them."""
    prog = _program(dims, sizes, dim)
    graphs = [g for g in P.superoptimize(prog, P.SuperoptConfig(cache=False))
              if not g.is_split and not g.invalid_split]
    assert graphs
    for graph in graphs:
        (ir,) = lower(graph)
        _check(ir, prog, dim, index)


# ------------------------------------------------------------------ the three rules

def _t(guid, dims):
    return {"num_dims": len(dims), "dim": list(dims), "guid": guid}


def _softmax_structure(lhs, reduced, axis, *, tail=()):
    """``exp(x) / f(sum(exp(x), axis))`` with the rowsum's dims written as ``reduced``."""
    ops = [
        {"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [_t(1, lhs)]},
        {"op_type": "kn_exp_op", "input_tensors": [_t(1, lhs)], "output_tensors": [_t(2, lhs)]},
        {"op_type": f"kn_reduce_sum_{axis}_op", "input_tensors": [_t(2, lhs)],
         "output_tensors": [_t(3, reduced)]},
    ]
    guid = 3
    for op in tail:  # unary / constant-binary ops between the reduction and the divide
        ops.append({"input_tensors": [_t(guid, reduced)], "output_tensors": [_t(guid + 1, reduced)],
                    **op})
        guid += 1
    ops += [
        {"op_type": "kn_div_op", "input_tensors": [_t(2, lhs), _t(guid, reduced)],
         "output_tensors": [_t(guid + 1, lhs)]},
        {"op_type": "kn_output_op", "input_tensors": [_t(guid + 1, lhs)], "output_tensors": []},
    ]
    return ops


def _div_index(structure, symbol_map=None):
    ir = to_ir(structure, dtype="float32", out_dtype="float32", symbol_map=symbol_map)
    (div,) = [n for n in ir.nodes if type(n) is Div]
    return div.broadcast


@pytest.mark.parametrize("axis, reduced, index", [(1, (4, 1), 0), (0, (1, 8), 1),
                                                  (1, (4,), 0), (0, (8,), 1)])
def test_symbolic_dims_decide(axis, reduced, index):
    # symbols 4 and 8 both stand for 16: the value shapes cannot tell the axes apart
    structure = _softmax_structure((4, 8), reduced, axis)
    assert _div_index(structure, {4: 16, 8: 16}) == index


@pytest.mark.parametrize("axis, index", [(1, 0), (0, 1)])
def test_reduced_axis_decides_when_the_dims_do_not(axis, index):
    # one symbol for both axes and no unit dim kept: only the producer knows
    assert _div_index(_softmax_structure((4, 4), (4,), axis), {4: 16}) == index
    assert _div_index(_softmax_structure((16, 16), (16,), axis)) == index


@pytest.mark.parametrize("axis, index", [(1, 0), (0, 1)])
def test_reduced_axis_is_forwarded_through_unary_and_constant_ops(axis, index):
    tail = ({"op_type": "kn_mul_op", "scalar": 2.0, "scalar_on_left": False},
            {"op_type": "kn_sqrt_op"},
            {"op_type": "kn_div_op", "scalar": 1.0, "scalar_on_left": True})
    assert _div_index(_softmax_structure((16, 16), (16,), axis, tail=tail)) == index
    # fused to an rsqrt: the fused op forwards the axis as well
    assert _div_index(_softmax_structure((16, 16), (16,), axis, tail=tail[1:])) == index


def test_undecidable_broadcast_is_refused_not_guessed():
    structure = [
        {"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [_t(1, (16, 16))]},
        {"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [_t(2, (16,))]},
        {"op_type": "kn_mul_op", "input_tensors": [_t(1, (16, 16)), _t(2, (16,))],
         "output_tensors": [_t(3, (16, 16))]},
        {"op_type": "kn_output_op", "input_tensors": [_t(3, (16, 16))], "output_tensors": []},
    ]
    with pytest.raises(ValueError, match="kn_mul_op at index 2.*broadcast"):
        to_ir(structure, dtype="float32", out_dtype="float32")
    # distinct extents were never ambiguous
    structure[1]["output_tensors"] = [_t(2, (24,))]
    structure[0]["output_tensors"] = structure[2]["output_tensors"] = [_t(1, (16, 24))]
    structure[2]["input_tensors"] = [_t(1, (16, 24)), _t(2, (24,))]
    structure[2]["output_tensors"] = [_t(3, (16, 24))]
    structure[3]["input_tensors"] = [_t(3, (16, 24))]
    ir = to_ir(structure, dtype="float32", out_dtype="float32")
    (mul,) = [n for n in ir.nodes if type(n) is Mul]
    assert mul.broadcast == 1


def test_unit_extents_leave_nothing_to_guess():
    # (1, 1) * (1,) with nothing to go by: both indices read the one element
    structure = [
        {"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [_t(1, (1, 1))]},
        {"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [_t(2, (1,))]},
        {"op_type": "kn_mul_op", "input_tensors": [_t(1, (1, 1)), _t(2, (1,))],
         "output_tensors": [_t(3, (1, 1))]},
        {"op_type": "kn_output_op", "input_tensors": [_t(3, (1, 1))], "output_tensors": []},
    ]
    ir = to_ir(structure, dtype="float32", out_dtype="float32")
    (mul,) = [n for n in ir.nodes if type(n) is Mul]
    assert mul.broadcast == 0
    (got,) = execute(ir, [torch.full(_shape(n), 3.0) for n in placeholder_order(ir)])
    assert got.tolist() == [[9.0]]


def _rowsum_then(rowsum, operand):
    """``sum(x (16, 24), 1) / y`` with the row sums written as ``rowsum`` and ``y`` as
    ``operand``: tensors of two ranks, so the dims do not line up position by position."""
    return [
        {"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [_t(1, (16, 24))]},
        {"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [_t(2, operand)]},
        {"op_type": "kn_reduce_sum_1_op", "input_tensors": [_t(1, (16, 24))],
         "output_tensors": [_t(3, rowsum)]},
        {"op_type": "kn_div_op", "input_tensors": [_t(3, rowsum), _t(2, operand)],
         "output_tensors": [_t(4, rowsum)]},
        {"op_type": "kn_output_op", "input_tensors": [_t(4, rowsum)], "output_tensors": []},
    ]


def test_input_with_the_first_operands_elements_takes_its_shape():
    # row sums (16,) / an input (16, 1): wiring declares the input (16,), as it always did
    ir = to_ir(_rowsum_then((16,), (16, 1)), dtype="float32", out_dtype="float32")
    (div,) = [n for n in ir.nodes if type(n) is Div]
    assert div.broadcast == -1
    shapes = {n.input_index: _shape(n) for n in ir.nodes if isinstance(n, PlaceHolderNode)}
    assert shapes == {0: (16, 24), 1: (16,)}
    # ... but not an input with other elements
    with pytest.raises(ValueError, match=r"kn_div_op at index 3: an operand of shape \(16, 24\) is "
                                         r"neither the shape \(16,\) of the first operand nor that "
                                         r"shape without one of its last two axes"):
        to_ir(_rowsum_then((16,), (16, 24)), dtype="float32", out_dtype="float32")


def test_dims_that_contradict_the_value_are_refused():
    # the divide lists the column sums (24,) as a vector over the rows (symbol 4)
    structure = [
        {"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [_t(1, (4, 8))]},
        {"op_type": "kn_reduce_sum_0_op", "input_tensors": [_t(1, (4, 8))],
         "output_tensors": [_t(2, (8,))]},
        {"op_type": "kn_div_op", "input_tensors": [_t(1, (4, 8)), _t(2, (4,))],
         "output_tensors": [_t(3, (4, 8))]},
        {"op_type": "kn_output_op", "input_tensors": [_t(3, (4, 8))], "output_tensors": []},
    ]
    with pytest.raises(ValueError, match=r"kn_div_op at index 2: by its dims the operand \(24,\) is "
                                         r"broadcast along axis 1 of \(16, 24\), where its shape "
                                         r"does not fit"):
        to_ir(structure, dtype="float32", out_dtype="float32", symbol_map={4: 16, 8: 24})


def test_symbolic_rule_table():
    from cleave.lower.convert import _broadcast_from_symbols as rule

    assert rule((12, 8), (12, 1), 2) == 0 and rule((12, 8), (1, 8), 2) == 1
    assert rule((20, 12, 8), (20, 12, 1), 3) == 0 and rule((20, 12, 8), (20, 1, 8), 3) == 1
    assert rule((12, 8), (12,), 2) == 0 and rule((12, 8), (8,), 2) == 1
    # a unit batch axis of the value is not a reduced axis
    assert rule((1, 12, 8), (1, 12, 1), 3) == 0 and rule((1, 12, 8), (1, 1, 8), 3) == 1
    # both tensors still list an axis reduced before them: the values have dropped it
    assert rule((1, 20, 12, 8), (1, 20, 12, 1), 3) == 0
    assert rule((1, 20, 12, 8), (1, 20, 1, 8), 3) == 1
    assert rule((1, 20, 12, 8), (1, 20, 12, 1), 4) == 0  # ... unless the value has rank 4
    # ... and a unit batch axis next to the reduced one (one head): the dims do not say
    # which of the two the value still has, and it does not matter
    assert rule((1, 1, 12, 8), (1, 1, 12, 1), 3) == 0
    assert rule((1, 1, 12, 8), (1, 1, 1, 8), 3) == 1
    assert rule((1, 5, 1, 7), (1, 5, 1, 1), 3) == 0
    assert rule((1, 1, 1, 12, 8), (1, 1, 1, 12, 1), 3) == 0  # two reduced, one unit axis
    # ... unless it does: without the first unit the operand spans the last axis, without
    # the other one it spans nothing the lhs has
    assert rule((1, 7, 1, 7), (1, 1, 1, 7), 3) is None
    assert rule((1, 1, 12, 8), (1, 1, 12, 1), 1) is None  # more axes dropped than are listed
    # nothing to go by
    assert rule((8, 8), (8,), 2) is None and rule((8, 8), (1, 1), 2) is None
    assert rule((20, 12, 8), (1, 12, 8), 3) is None  # a reduced batch axis
    assert rule((2, 20, 12, 8), (20, 12), 3) is None


def test_split_combine_divides_by_the_row_sums():
    """The combine of split attention as ``lower/process.py`` hands it to ``to_ir``.

    ``sum_split(o) / sum_split(l)`` with S_Q == D: the two structure tensors are
    ``(1, BH, SQ, D)`` and ``(1, BH, SQ, 1)`` while the values have rank 3 and 2.
    """
    bh, split, sq, d = 20, 2, 28, 44  # symbols; SQ and D both stand for 64
    structure = [
        {"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [_t(1, (bh, split, sq))]},
        {"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [_t(2, (bh, split, sq, d))]},
        {"op_type": "kn_reduce_sum_1_op", "input_tensors": [_t(1, (split, bh, sq, 1))],
         "output_tensors": [_t(3, (1, bh, sq, 1))]},
        {"op_type": "kn_reduce_sum_1_op", "input_tensors": [_t(2, (split, bh, sq, d))],
         "output_tensors": [_t(4, (1, bh, sq, d))]},
        {"op_type": "kn_div_op", "input_tensors": [_t(4, (1, bh, sq, d)), _t(3, (1, bh, sq, 1))],
         "output_tensors": [_t(5, (1, bh, sq, d))]},
        {"op_type": "kn_output_op", "input_tensors": [_t(5, (1, bh, sq, d))], "output_tensors": []},
    ]
    ir = to_ir(structure, dtype="float32", out_dtype="float32",
               symbol_map={bh: 4, split: 4, sq: 64, d: 64})
    (div,) = [n for n in ir.nodes if type(n) is Div]
    assert div.broadcast == 0
    torch.manual_seed(0)
    by_shape = {(4, 4, 64): torch.rand(4, 4, 64) + 0.5, (4, 4, 64, 64): torch.randn(4, 4, 64, 64)}
    order = [tuple(int(v) for v in n.get_shape(0)) for n in placeholder_order(ir)]
    (got,) = execute(ir, [by_shape[shape] for shape in order])
    want = by_shape[(4, 4, 64, 64)].sum(1) / by_shape[(4, 4, 64)].sum(1).unsqueeze(-1)
    torch.testing.assert_close(got.reshape(want.shape), want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("sq, d", [(64, 64), (64, 128)], ids=["SQ == D", "SQ != D"])
def test_split_combine_with_one_head_divides_by_the_row_sums(sq, d):
    """The same combine for one head, as the search writes it: a unit dim is the literal
    ``1``, like the dim a reduction leaves behind, so the two structure tensors are
    ``(1, 1, SQ, D)`` and ``(1, 1, SQ, 1)`` for values of rank 3 and 2.

    With S_Q == D the value shapes do not say which axis the row sums span, and the dims
    used to be read with both units taken for reduced axes: every split graph of one-head
    attention (the decode demo's shape with H = 1) was refused, "cannot tell which axis of
    (1, 64, 64) the operand (1, 64) is broadcast along".
    """
    split, s_q, s_d = 2, 12, 8  # symbols; a unit dim has none
    structure = [
        {"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [_t(1, (1, split, s_q))]},
        {"op_type": "kn_input_op", "input_tensors": [],
         "output_tensors": [_t(2, (1, split, s_q, s_d))]},
        {"op_type": "kn_reduce_sum_1_op", "input_tensors": [_t(1, (split, 1, s_q, 1))],
         "output_tensors": [_t(3, (1, 1, s_q, 1))]},
        {"op_type": "kn_reduce_sum_1_op", "input_tensors": [_t(2, (split, 1, s_q, s_d))],
         "output_tensors": [_t(4, (1, 1, s_q, s_d))]},
        {"op_type": "kn_div_op",
         "input_tensors": [_t(4, (1, 1, s_q, s_d)), _t(3, (1, 1, s_q, 1))],
         "output_tensors": [_t(5, (1, 1, s_q, s_d))]},
        {"op_type": "kn_output_op", "input_tensors": [_t(5, (1, 1, s_q, s_d))],
         "output_tensors": []},
    ]
    ir = to_ir(structure, dtype="float32", out_dtype="float32",
               symbol_map={split: 4, s_q: sq, s_d: d, 1: 1})
    (div,) = [n for n in ir.nodes if type(n) is Div]
    assert div.broadcast == 0
    torch.manual_seed(0)
    by_shape = {(1, 4, sq): torch.rand(1, 4, sq) + 0.5, (1, 4, sq, d): torch.randn(1, 4, sq, d)}
    order = [tuple(int(v) for v in n.get_shape(0)) for n in placeholder_order(ir)]
    (got,) = execute(ir, [by_shape[shape] for shape in order])
    want = by_shape[(1, 4, sq, d)].sum(1) / by_shape[(1, 4, sq)].sum(1).unsqueeze(-1)
    torch.testing.assert_close(got.reshape(want.shape), want, rtol=1e-5, atol=1e-6)


def test_a_reduced_batch_axis_is_refused():
    # (B, M, N) / sum over B: the divisor is (M, N), which no broadcast index describes;
    # with B == M == N it used to pass for a row sum.
    structure = _softmax_structure((16, 16, 16), (1, 16, 16), 0)
    with pytest.raises(ValueError, match="broadcast"):
        to_ir(structure, dtype="float32", out_dtype="float32")


# ------------------------------------------------------------------ chained reductions

@pytest.mark.parametrize("sizes", [dict(B=4, R=6, C=8), dict(B=8, R=8, C=8)], ids=["4x6x8", "8x8x8"])
@pytest.mark.parametrize("first, second", [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)])
def test_second_reduction_reduces_the_axis_the_program_names(sizes, first, second):
    """``sum(sum(x, 0), 1)`` reduced the wrong axis; ``(0, 2)`` and ``(1, 2)`` did not lower."""
    def fn(x: cl.Tensor["B", "R", "C"]):
        return cl.sum(cl.sum(x, first, keepdim=True), second, keepdim=True)

    prog = trace(fn, sizes, cl.f32)
    (graph,) = GraphSet.identity(prog)
    (ir,) = lower(graph)
    torch.manual_seed(0)
    x = torch.randn(*prog.inputs[0].shape)
    (got,) = execute(ir, [x])
    (want,) = prog.reference(x)
    assert got.numel() == want.numel()
    torch.testing.assert_close(got.reshape(want.shape), want, rtol=1e-5, atol=1e-5)


def test_reducing_an_axis_twice_is_refused():
    def fn(x: cl.Tensor["B", "R", "C"]):
        return cl.sum(cl.sum(x, 0, keepdim=True), 0, keepdim=True)

    prog = trace(fn, dict(B=4, R=6, C=8), cl.f32)
    (graph,) = GraphSet.identity(prog)
    with pytest.raises(LoweringRejected, match="axis 0 of .* was already reduced away"):
        lower(graph)


# ------------------------------------------------------------------ what a binary op cannot do

def _shape(node, i=0):
    return tuple(int(s) for s in node.get_shape(i))


def _lower(fn, sizes):
    prog = trace(fn, sizes, cl.f32)
    (graph,) = GraphSet.identity(prog)
    (ir,) = lower(graph)
    return prog, ir


def _run(ir, prog, inputs):
    """The flat graph on the program's inputs, each in its placeholder's own layout: an
    input is bound by declaration and viewed in the kernel's shape (``Kernel.__call__``)."""
    by_index = {n.input_index: n for n in ir.nodes if isinstance(n, PlaceHolderNode)}
    feed = {n.name: inputs[i].reshape(_shape(n)) for i, n in by_index.items()}
    return execute(ir, [feed[n.name] for n in placeholder_order(ir)])


def _assert_computes_the_program(ir, prog):
    """Every binary op and reduction reads the operand shapes it declares, every input holds
    the program's elements, and the result is the program's."""
    for n in ir.nodes:
        if type(n) in (Add, Sub, Mul, Div, ReduceSum):
            for e in n.inputs:
                declared = tuple(int(d) for d in n.input_buffers[e.dst_id].shape)
                assert _shape(e.src_node, e.src_id) == declared, (n.name, e.dst_id)
    torch.manual_seed(0)
    inputs = [torch.rand(*s.shape) + 0.5 for s in prog.inputs]
    (got,) = _run(ir, prog, inputs)
    (want,) = prog.reference(*inputs)
    assert got.numel() == want.numel()
    torch.testing.assert_close(got.reshape(want.shape), want, rtol=1e-5, atol=1e-6)


def unit_lhs(b: cl.Tensor["ONE", "N"], x: cl.Tensor["N", "M"]):
    return b * cl.sum(x, -1, keepdim=True)  # (1, N) * (N, 1): an (N, N) result


def outer(x: cl.Tensor["N", "M"], y: cl.Tensor["M", "N"]):
    return cl.sum(x, 1, keepdim=True) * cl.sum(y, 0, keepdim=True)  # (N, 1) * (1, N)


def reduced_batch_axis(x: cl.Tensor["B", "K", "K2", "N"]):
    return x / cl.sum(x, 1, keepdim=True)  # (B, K, K, N) / (B, 1, K, N)


def lhs_grows(x: cl.Tensor["R", "C"]):
    return cl.sum(x, -1, keepdim=True) * x  # (R, 1) * (R, C)


def batch_softmax(x: cl.Tensor["B", "R", "C"]):
    e = cl.exp(x)
    return e / cl.sum(e, 0, keepdim=True)


# The same refusals where the softmax has its max: ``cl.softmax`` records ``x - max(x)``, so
# a softmax over a vector or a batch axis is refused at that subtraction, the first op that
# broadcasts.
def _stable_exp(x, dim):
    return cl.exp(x - cl.max(x, dim, keepdim=True))


def vector_softmax(x: cl.Tensor["C"]):
    return cl.softmax(x, 0)


def batch_axis_softmax(x: cl.Tensor["B", "R", "C"]):
    return cl.softmax(x, 0)


def two_axes_softmax(x: cl.Tensor["R", "C"]):
    e = _stable_exp(x, -1)
    return e / cl.sum(cl.sum(e, -1, keepdim=True), -2, keepdim=True)


def unit_row_softmax(x: cl.Tensor["B", "F"], v: cl.Tensor["ONE", "F"]):
    # out[i, j] = (v . e_j) / sum(e_i): with B == 4 on both axes the divide used to lower as
    # a product along the columns
    e = _stable_exp(x, -1)
    return (v @ e.mT) / cl.sum(e, -1, keepdim=True)


def column_input(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "ONE"]):
    return x * y


def row_input(x: cl.Tensor["R", "C"], y: cl.Tensor["ONE", "C"]):
    return x + y


def total(x: cl.Tensor["R", "C"]):
    return x / cl.sum(cl.sum(x, -1, keepdim=True), 0, keepdim=True)


def vector_total(x: cl.Tensor["C"]):
    return x / cl.sum(x, 0, keepdim=True)


def rows_of_rowsums(x: cl.Tensor["B", "R", "C"]):
    l = cl.sum(x, -1, keepdim=True)
    return l / cl.sum(l, 1, keepdim=True)


def unit_op_output(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "ONE"]):
    return cl.sum(x, -1, keepdim=True) / (y * 2.0)


def input_two_layouts(x: cl.Tensor["R", "ONE"]):
    return cl.sum(x, -1, keepdim=True) - x


RULE = ("; a binary op keeps the shape of its first operand and broadcasts the second along one "
        "of its last two axes")
SMALL = dict(R=4, C=6, ONE=1)
# name: (fn, sizes, the reason after "graph 0: ValueError: ")
REFUSED = {
    # Lowered before, to an op that computes something else: equal extents hid it.
    # (1, N) * rowsum(x (N, M)) was read as a product along the columns: (1, N) for (N, N)
    "unit_lhs": (unit_lhs, dict(ONE=1, N=4, M=6),
                 r"kn_mul_op at index 3: the operand \(4,\) has extent 4 where the first operand "
                 r"\(1, 4\) has 1 \(dims \(1, 4\) and \(4, 1\)\)" + RULE),
    # the outer product of a row sum and a column sum: elementwise, (N,) for (N, N)
    "outer": (outer, dict(N=4, M=6),
              r"kn_mul_op at index 4: the operand \(4,\) has extent 4 where the first operand "
              r"\(4,\) has 1 \(dims \(4, 1\) and \(1, 4\)\)" + RULE),
    # divided by the sums of another row: off by 0.44 of the max
    "reduced_batch_axis": (reduced_batch_axis, dict(B=3, K=4, K2=4, N=6),
                           r"kn_div_op at index 2: the operand \(3, 4, 6\) is broadcast along axis "
                           r"1 of the first operand \(3, 4, 4, 6\) \(dims \(3, 4, 4, 6\) and "
                           r"\(3, 1, 4, 6\)\)" + RULE),
    # the (R, C) input was declared (R,)
    "lhs_grows": (lhs_grows, dict(R=4, C=6),
                  r"kn_mul_op at index 2: the operand \(4, 6\) has extent 6 where the first operand "
                  r"\(4,\) has 1 \(dims \(4, 1\) and \(4, 6\)\)" + RULE),
    # Lowered before, to an op with an operand of another shape wired in: no tiling builds.
    "batch_softmax": (batch_softmax, dict(B=3, R=16, C=24),
                      r"kn_div_op at index 3: the operand \(16, 24\) is broadcast along axis 0 of "
                      r"the first operand \(3, 16, 24\) \(dims \(3, 16, 24\) and \(1, 16, 24\)\)"
                      + RULE),
    "total": (total, dict(R=4, C=6),
              r"kn_div_op at index 3: the operand \(\) is broadcast along axes 0, 1 of the first "
              r"operand \(4, 6\) \(dims \(4, 6\) and \(1, 1\)\)" + RULE),
    "vector_total": (vector_total, dict(C=6),
                     r"kn_div_op at index 2: the operand \(\) is broadcast along axis 0 of the first "
                     r"operand \(6,\) \(dims \(6,\) and \(1,\)\)" + RULE),
    "rows_of_rowsums": (rows_of_rowsums, dict(B=3, R=4, C=6),
                        r"kn_div_op at index 3: an operand of shape \(3,\) cannot be broadcast over "
                        r"\(3, 4\), a value with 1 batch axes"),
    "unit_op_output": (unit_op_output, SMALL,
                       r"kn_div_op at index 4: the operand \(4, 1\) and the first operand \(4,\) hold "
                       r"the same elements with different unit axes \(dims \(4, 1\) and \(4, 1\)\)"),
    "input_two_layouts": (input_two_layouts, dict(R=4, ONE=1),
                          r"input 1 is read as \(4, 1\) by reduction_1_1 and as \(4,\) by sub_2; an "
                          r"input has one layout"),
    # softmaxes written with their max
    "vector_softmax": (vector_softmax, dict(C=24),
                       r"kn_sub_op at index 2: the operand \(\) is broadcast along axis 0 of the first "
                       r"operand \(24,\) \(dims \(24,\) and \(1,\)\)" + RULE),
    "batch_axis_softmax": (batch_axis_softmax, dict(B=3, R=16, C=24),
                           r"kn_sub_op at index 2: the operand \(16, 24\) is broadcast along axis 0 of "
                           r"the first operand \(3, 16, 24\) \(dims \(3, 16, 24\) and \(1, 16, 24\)\)"
                           + RULE),
    "two_axes_softmax": (two_axes_softmax, dict(R=16, C=24),
                         r"kn_div_op at index 6: the operand \(\) is broadcast along axes 0, 1 of the "
                         r"first operand \(16, 24\) \(dims \(16, 24\) and \(1, 1\)\)" + RULE),
    "unit_row_softmax": (unit_row_softmax, dict(B=4, F=2, ONE=1),
                         r"kn_div_op at index 7: the operand \(4,\) has extent 4 where the first "
                         r"operand \(1, 4\) has 1 \(dims \(1, 4\) and \(4, 1\)\)" + RULE),
}


@pytest.mark.parametrize("name", REFUSED)
def test_operand_a_binary_op_cannot_read_is_refused(name):
    fn, sizes, reason = REFUSED[name]
    prog = trace(fn, sizes, cl.f32)
    (graph,) = GraphSet.identity(prog)
    with pytest.raises(LoweringRejected) as e:
        lower(graph)
    assert re.fullmatch("graph 0: ValueError: " + reason, str(e.value)), e.value


def keepdim_input(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "ONE"]):
    return cl.sum(x, -1, keepdim=True) / y  # y has the row sums' elements: declared (R,)


def keepdim_input_twice(a: cl.Tensor["R", "C"], b: cl.Tensor["R", "C"], y: cl.Tensor["R", "ONE"]):
    return cl.sum(a, -1, keepdim=True) / y + cl.sum(b, -1, keepdim=True) / y


def unit_column_lhs(y: cl.Tensor["R", "ONE"], x: cl.Tensor["R", "C"]):
    return y * cl.sum(x, -1, keepdim=True)


def unit_row_lhs(y: cl.Tensor["ONE", "C"], x: cl.Tensor["R", "C"]):
    return y / cl.sum(x, 0, keepdim=True)


def kept_and_dropped_unit(z: cl.Tensor["B", "ONE", "N"], x: cl.Tensor["B", "G", "ONE"]):
    # the divisor (B, 1) has dropped the reduced axis and kept its own unit axis
    return z / cl.sum(x, 1, keepdim=True)


def same_shape(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]):
    return (x - y) * y


# name: (fn, sizes, broadcast of each tensor-tensor op in graph order, shape of each input)
UNIT_AXES = {
    "keepdim_input": (keepdim_input, SMALL, [-1], [(4, 6), (4,)]),
    "keepdim_input_twice": (keepdim_input_twice, SMALL, [-1, -1, -1], [(4, 6), (4, 6), (4,)]),
    "unit_column_lhs": (unit_column_lhs, SMALL, [0], [(4, 1), (4, 6)]),
    "unit_row_lhs": (unit_row_lhs, SMALL, [1], [(1, 6), (4, 6)]),
    "kept_and_dropped_unit": (kept_and_dropped_unit, dict(B=3, G=5, N=6, ONE=1), [0],
                              [(3, 1, 6), (3, 5, 1)]),
    "same_shape": (same_shape, dict(R=4, C=6), [-1, -1], [(4, 6), (4, 6)]),
    # an input broadcast along the axis it keeps as a unit (a log-sum-exp ``(H, SQ, 1)`` under
    # the scores of an attention backward): declared the way the op reads it
    "column_input": (column_input, SMALL, [0], [(4, 6), (4,)]),
    "row_input": (row_input, SMALL, [1], [(4, 6), (6,)]),
}


@pytest.mark.parametrize("name", UNIT_AXES)
def test_unit_axes_do_not_refuse_what_the_op_can_read(name):
    """Unit axes are interchangeable: the same elements in the same order are the same operand."""
    fn, sizes, broadcasts, input_shapes = UNIT_AXES[name]
    prog, ir = _lower(fn, sizes)
    ops = [n for n in ir.nodes if type(n) in (Add, Sub, Mul, Div) and n.const is None]
    assert [n.broadcast for n in ops] == broadcasts
    by_index = {n.input_index: _shape(n) for n in ir.nodes if isinstance(n, PlaceHolderNode)}
    assert [by_index[i] for i in range(len(prog.inputs))] == input_shapes
    _assert_computes_the_program(ir, prog)


def test_matmul_operand_that_lost_a_matrix_axis_is_refused():
    """``rowsum.mT @ rowsum`` per batch entry: the two (B, K) values were read as matrices and
    contracted over the batch (2.4x of the max off the program)."""
    def outer_per_batch(x: cl.Tensor["B", "M", "K"]):
        l = cl.sum(x, 1, keepdim=True)
        return l.mT @ l

    prog = trace(outer_per_batch, dict(B=4, M=6, K=3), cl.f32)
    (graph,) = GraphSet.identity(prog)
    with pytest.raises(LoweringRejected, match=r"kn_matmul_op at index 2: the first operand \(4, 3\) "
                                               r"has lost a matrix axis to a reduction "
                                               r"\(dims \(4, 1, 3\)\)"):
        lower(graph)


def test_vector_operand_of_a_matmul_still_lowers():
    """``x @ rowsum(w)``: a value of rank 1 is a vector, as the search of ``(x @ w).sum(-1)``
    writes it (demo/matmul_rowsum.py)."""
    def matvec(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
        return x @ cl.sum(w, -1, keepdim=True)

    prog, ir = _lower(matvec, dict(M=4, K=6, N=5))
    (mma,) = [n for n in ir.nodes if type(n) is Matmul]
    operands = [_shape(e.src_node, e.src_id) for e in sorted(mma.inputs, key=lambda e: e.dst_id)]
    assert operands == [(4, 6), (6,)]
    _assert_computes_the_program(ir, prog)


def a_lost_its_batch_axis(x: cl.Tensor["B", "M", "K"]):
    return cl.sum(x, 0, keepdim=True) @ x.mT


def a_lost_its_batch_axis_two_inputs(x: cl.Tensor["B", "M", "K"], y: cl.Tensor["B", "N", "K"]):
    return cl.sum(x, 0, keepdim=True) @ y.mT


def b_lost_its_batch_axis(x: cl.Tensor["B", "M", "K"], y: cl.Tensor["B", "K", "N"]):
    return x @ cl.sum(y, 0, keepdim=True)


# name: (fn, sizes, which operand, its value shape, the matmul's output, the operand's dims)
LOST_BATCH_AXIS = {
    "a": (a_lost_its_batch_axis, dict(B=4, M=6, K=3), "first", (6, 3), (4, 6, 6), (1, 6, 3)),
    "a_two_inputs": (a_lost_its_batch_axis_two_inputs, dict(B=4, M=6, K=3, N=5),
                     "first", (6, 3), (4, 6, 5), (1, 6, 3)),
    "b": (b_lost_its_batch_axis, dict(B=4, M=6, K=3, N=5), "second", (3, 5), (4, 6, 5), (1, 3, 5)),
}


@pytest.mark.parametrize("name", LOST_BATCH_AXIS)
def test_matmul_operand_computed_without_its_batch_axes_is_refused(name):
    """``sum(x, 0) @ x.mT``: the flat graph is right, but the matmul declares the operand
    with a unit batch axis the value does not have, and the tiling policy tiles the value as
    the matmul reads it: once per batch entry of the output. None of the 16 plans of such a
    graph built, each dying in whichever guard its tiles tripped first ("runs 8 iterations
    but dim 0 ... is cut into 4 tiles", "Expected exactly one launch-axis mapping")."""
    fn, sizes, which, shape, out, dims = LOST_BATCH_AXIS[name]
    prog = trace(fn, sizes, cl.f32)
    (graph,) = GraphSet.identity(prog)
    with pytest.raises(LoweringRejected) as refusal:
        lower(graph)
    assert str(refusal.value).endswith(
        f": the {which} operand {shape} is computed without the batch axes of its matmul, "
        f"whose output is {out} (dims {dims}); only an input is read over batch axes it "
        "does not have")
    assert "kn_matmul_op at index " in str(refusal.value)


def both_lost_the_batch_axis(x: cl.Tensor["B", "M", "K"], y: cl.Tensor["B", "K", "N"]):
    return cl.sum(x, 0, keepdim=True) @ cl.sum(y, 0, keepdim=True)


def both_lost_a_middle_batch_axis(x: cl.Tensor["B", "H", "M", "K"],
                                  y: cl.Tensor["B", "H", "K", "N"]):
    return cl.sum(x, 1, keepdim=True) @ cl.sum(y, 1, keepdim=True)


def lost_axis_against_a_unit_batch_value(x: cl.Tensor["B", "M", "K"],
                                         w: cl.Tensor["ONE", "K", "N"]):
    return cl.exp(cl.sum(x, 0, keepdim=True)) @ cl.exp(w)


def a_lost_axis_against_a_unit_batch_input(x: cl.Tensor["B", "M", "K"],
                                           w: cl.Tensor["ONE", "K", "N"]):
    return cl.sum(x, 0, keepdim=True) @ w


def b_lost_axis_against_a_unit_batch_input(x: cl.Tensor["ONE", "M", "K"],
                                           y: cl.Tensor["B", "K", "N"]):
    return x @ cl.sum(y, 0, keepdim=True)


# fn: (sizes, value shapes of the two operands, the matmul's output)
LOST_UNIT_BATCH_AXIS = {
    both_lost_the_batch_axis: (dict(B=4, M=6, K=3, N=5), [(6, 3), (3, 5)], (1, 6, 5)),
    both_lost_a_middle_batch_axis: (dict(B=2, H=4, M=6, K=3, N=5),
                                    [(2, 6, 3), (2, 3, 5)], (2, 1, 6, 5)),
    lost_axis_against_a_unit_batch_value: (dict(B=4, M=6, K=3, N=5, ONE=1),
                                           [(6, 3), (1, 3, 5)], (1, 6, 5)),
    a_lost_axis_against_a_unit_batch_input: (dict(B=4, M=6, K=3, N=5, ONE=1),
                                             [(6, 3), (1, 3, 5)], (1, 6, 5)),
    b_lost_axis_against_a_unit_batch_input: (dict(B=4, M=6, K=3, N=5, ONE=1),
                                             [(1, 6, 3), (3, 5)], (1, 6, 5)),
}


@pytest.mark.parametrize("fn", LOST_UNIT_BATCH_AXIS, ids=[fn.__name__ for fn in LOST_UNIT_BATCH_AXIS])
def test_matmul_operand_without_a_batch_axis_the_output_has_one_entry_of_still_lowers(fn):
    """The refusal above is for an operand that would be read once per batch entry. Where
    the output has one entry along every axis the value lacks -- both operands lost it,
    or the other one is broadcast over it -- the declared operand is the stored one with
    unit axes, and these programs were lowered, tiled and compiled before the refusal
    existed. It refused them all, by rank alone."""
    sizes, operands, out = LOST_UNIT_BATCH_AXIS[fn]
    prog, ir = _lower(fn, sizes)
    (mma,) = [n for n in ir.nodes if type(n) is Matmul]
    assert [_shape(e.src_node, e.src_id)
            for e in sorted(mma.inputs, key=lambda e: e.dst_id)] == operands
    assert _shape(mma) == out
    _assert_computes_the_program(ir, prog)


def test_matmul_operand_without_a_batch_axis_is_refused_by_entries_not_by_rank():
    """The line between the two: the same operand, one batch entry in the output or four."""
    def one_entry(x: cl.Tensor["B", "M", "K"], y: cl.Tensor["ONE", "K", "N"]):
        return cl.sum(x, 0, keepdim=True) @ cl.exp(y)

    def four_entries(x: cl.Tensor["B", "M", "K"], y: cl.Tensor["B", "K", "N"]):
        return cl.sum(x, 0, keepdim=True) @ cl.exp(y)

    _lower(one_entry, dict(B=4, M=6, K=3, N=5, ONE=1))
    (graph,) = GraphSet.identity(trace(four_entries, dict(B=4, M=6, K=3, N=5), cl.f32))
    with pytest.raises(LoweringRejected, match=r"the first operand \(6, 3\) is computed "
                                               r"without the batch axes of its matmul, "
                                               r"whose output is \(4, 6, 5\)"):
        lower(graph)


def unit_batch_input_b(x: cl.Tensor["B", "M", "K"], w: cl.Tensor["ONE", "K", "N"]):
    return x @ w


def unit_batch_input_a(w: cl.Tensor["ONE", "M", "K"], y: cl.Tensor["B", "K", "N"]):
    return w @ y


def unit_batch_value_a(w: cl.Tensor["ONE", "M", "K"], y: cl.Tensor["B", "K", "N"]):
    return cl.exp(w) @ y


def unit_batch_value_b(x: cl.Tensor["B", "M", "K"], w: cl.Tensor["ONE", "K", "N"]):
    return x @ cl.exp(w)


UNIT_BATCH = [unit_batch_input_a, unit_batch_input_b, unit_batch_value_a, unit_batch_value_b]


@pytest.mark.parametrize("fn", UNIT_BATCH, ids=[fn.__name__ for fn in UNIT_BATCH])
def test_matmul_operand_with_a_unit_batch_axis_still_lowers(fn):
    """What the refusal above leaves alone: an operand that has every axis the matmul
    declares, with extent 1 where it is broadcast over the batch -- input or computed."""
    prog, ir = _lower(fn, dict(B=4, M=6, K=3, N=5, ONE=1))
    (mma,) = [n for n in ir.nodes if type(n) is Matmul]
    operands = [_shape(e.src_node, e.src_id) for e in sorted(mma.inputs, key=lambda e: e.dst_id)]
    assert sorted(len(shape) for shape in operands) == [3, 3]
    assert sorted(shape[0] for shape in operands) == [1, 4]
    _assert_computes_the_program(ir, prog)


def test_vector_against_a_batched_matrix_is_refused():
    """``x(B, M, K) @ v`` with ``v`` a vector of K: the Matmul declared the vector with the
    batch, ``(B, 1, K)`` for K elements, so not even the reference executor could run the
    flat graph, and no plan built."""
    def batched_matvec(x: cl.Tensor["B", "M", "K"]):
        v = cl.sum(cl.sum(x, 1, keepdim=True), 0, keepdim=True)
        return x @ v.mT

    def batched_vecmat(x: cl.Tensor["B", "K", "N"]):
        v = cl.sum(cl.sum(x, 2, keepdim=True), 0, keepdim=True)
        return v.mT @ x

    for fn, sizes, which, other in ((batched_matvec, dict(B=4, M=6, K=3), "second", (4, 6, 3)),
                                    (batched_vecmat, dict(B=4, K=3, N=5), "first", (4, 3, 5))):
        prog = trace(fn, sizes, cl.f32)
        (graph,) = GraphSet.identity(prog)
        with pytest.raises(LoweringRejected, match=(
                rf"kn_matmul_op at index 3: the {which} operand is a vector \(3,\) and the "
                rf"other a batched matrix {re.escape(str(other))} .*; only a matmul without "
                r"batch axes takes a vector$")):
            lower(graph)


# ------------------------------------------------------------------ ops no output reads

def test_op_no_output_reads_does_not_reshape_a_live_input():
    """``a + x`` is never used; built, it was wired last and declared ``x`` with its own
    (1, 1): the live ``exp(x)`` was lowered on one element of a four-element input."""
    def fn(a: cl.Tensor["ONE", "ONE"], x: cl.Tensor["R", "ONE"]):
        a + x
        return cl.exp(x)

    prog, ir = _lower(fn, dict(R=4, ONE=1))
    assert [type(n) for n in ir.nodes] == [PlaceHolderNode, Exp, OutputNode]
    (ph,), (exp,) = ([n for n in ir.nodes if type(n) is kind] for kind in (PlaceHolderNode, Exp))
    assert _shape(ph) == _shape(exp) == (4, 1) and ph.input_index == 1
    assert [e.dst_node for e in ph.outputs] == [exp]  # no edge into an op that was dropped
    torch.manual_seed(0)
    x = torch.randn(4, 1)
    (got,) = execute(ir, [x])
    torch.testing.assert_close(got, prog.reference(torch.zeros(1, 1), x)[0], rtol=1e-6, atol=1e-6)


def test_op_no_output_reads_cannot_refuse_the_graph():
    """A reduction of an axis that is already gone, an operand no binary op can read and a
    matmul of two batched row sums: none of them is part of what the program computes."""
    def fn(x: cl.Tensor["B", "R", "C"], y: cl.Tensor["ONE", "R", "C"]):
        l = cl.sum(x, 1, keepdim=True)
        cl.sum(l, 1, keepdim=True)
        x * y  # broadcast along the batch axis
        l.mT @ l
        return cl.exp(x)

    prog, ir = _lower(fn, dict(B=3, R=4, C=4, ONE=1))
    assert [type(n) for n in ir.nodes] == [PlaceHolderNode, Exp, OutputNode]
    _assert_computes_the_program(ir, prog)


def test_the_same_ops_refuse_the_graph_once_an_output_reads_them():
    def fn(x: cl.Tensor["B", "R", "C"], y: cl.Tensor["ONE", "R", "C"]):
        return cl.exp(x * y)

    prog = trace(fn, dict(B=3, R=4, C=4, ONE=1), cl.f32)
    (graph,) = GraphSet.identity(prog)
    with pytest.raises(LoweringRejected, match=r"kn_mul_op at index 2: the operand \(1, 4, 4\) is "
                                               r"broadcast along axis 0"):
        lower(graph)


# ------------------------------------------------------------------ whatever lowers is right

SIZES = {"A": 4, "B": 4, "C": 6, "E": 3, "U": 1}  # equal extents and unit extents on purpose


def _random_program(seed):
    """Up to six elementwise / reduction ops over one to three inputs of rank 2 or 3; the
    last value is the output, so most programs also have ops that no output reads."""
    rng = random.Random(seed)
    prog = Program(f"random{seed}", dtype=cl.f32)
    for name, size in SIZES.items():
        prog.dim(name, size)
    rank = rng.choice([2, 2, 3])
    values = [prog.input(f"x{i}", [rng.choice(list(SIZES)) for _ in range(rank)])
              for i in range(rng.choice([1, 2, 2, 3]))]
    for _ in range(rng.randint(1, 6)):
        kind = rng.choice(["binary", "binary", "binary", "scalar", "exp", "sum", "sum", "max"])
        a, b = rng.choice(values), rng.choice(values)
        try:
            if kind == "binary":
                value = prog._binary(rng.choice(["add", "sub", "mul", "div"]), a, b)
            elif kind == "scalar":
                value = prog._binary(rng.choice(["add", "mul", "div"]), a, rng.choice([2.0, 0.5]))
            elif kind == "exp":
                value = a.exp()
            else:
                value = getattr(a, kind)(rng.randrange(a.ndim), keepdim=True)
        except SpecError:
            continue  # shapes that do not broadcast
        values.append(value)
    if len(values) == len(prog.inputs):
        return None
    prog.output(values[-1])
    return prog


@pytest.mark.parametrize("first", range(0, 600, 100))
def test_random_programs_lower_to_what_they_compute_or_are_refused(first):
    """No third outcome: a program either has a typed refusal, or its flat graph reads every
    operand in the shape it declares and computes ``Program.reference``."""
    lowered = 0
    for seed in range(first, first + 100):
        prog = _random_program(seed)
        if prog is None:
            continue
        (graph,) = GraphSet.identity(prog)
        try:
            (ir,) = lower(graph)
        except LoweringRejected:
            continue
        lowered += 1
        for n in ir.nodes:
            if type(n) in (Add, Sub, Mul, Div, ReduceSum, ReduceMax):
                for e in n.inputs:
                    declared = tuple(int(d) for d in n.input_buffers[e.dst_id].shape)
                    assert _shape(e.src_node, e.src_id) == declared, (seed, n.name, e.dst_id)
        torch.manual_seed(seed)
        inputs = [torch.rand(*s.shape) + 0.5 for s in prog.inputs]
        (want,) = prog.reference(*inputs)
        (got,) = _run(ir, prog, inputs)
        assert got.numel() == want.numel(), (seed, tuple(got.shape), tuple(want.shape))
        if torch.isfinite(want).all():  # a chain of exps overflows on its own
            error = float((got.reshape(want.shape) - want).abs().max())
            assert error <= 1e-4 * max(float(want.abs().max()), 1e-6), (seed, error)
    assert lowered >= 50  # the property is about programs that lower
