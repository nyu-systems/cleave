"""Axes of a lowered graph come from the structure, never from extents that happen to agree.

``e / sum(e, dim)`` lowers to ``Div(e, rowsum)`` with ``broadcast == 0`` (the operand spans
the second-to-last axis) or ``broadcast == 1`` (it spans the last one). With equal trailing
extents the two value shapes are the same, so the index has to come from the structure:
the symbolic dims of the two tensors, else the axis the operand's producer reduced.
A reduction drops its axis from the value while the structure keeps it as a unit dim, so
the axis of a second reduction has to be re-counted as well.
The lowered graph is compared with ``Program.reference``, which never sees the IR.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.frontend import trace
from cleave.ir.ops import Div, Mul
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
IDS = [f"{''.join(dims)}_{'x'.join(str(s[d]) for d in dims)}_dim{dim}" for dims, s, dim, _ in SHAPES]


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


@pytest.mark.parametrize("dims, sizes, dim, index", SHAPES, ids=IDS)
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
    from cleave.core.errors import LoweringRejected

    def fn(x: cl.Tensor["B", "R", "C"]):
        return cl.sum(cl.sum(x, 0, keepdim=True), 0, keepdim=True)

    prog = trace(fn, dict(B=4, R=6, C=8), cl.f32)
    (graph,) = GraphSet.identity(prog)
    with pytest.raises(LoweringRejected, match="axis 0 of .* was already reduced away"):
        lower(graph)
