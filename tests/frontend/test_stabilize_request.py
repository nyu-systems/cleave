"""The exact-softmax request: a plain ``exp`` record tagged with the softmax axis.

The search and the lowering see ``exp / sum / div``; the tag only says that the program asked
for the shift-invariant softmax. It changes the fingerprint and ``Program.reference``.
"""

import pytest

torch = pytest.importorskip("torch")

import cleave.lang as cl
from cleave.core.errors import SpecError
from cleave.frontend import Program, trace
from tests._softmax import stable_exp


def requested(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1)


def handwritten(x: cl.Tensor["R", "C"]):
    e = cl.exp(x)
    return e / cl.sum(e, -1, keepdim=True)


def attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
              v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, -1) @ v


SIZES = {"R": 4, "C": 32}


def test_record_is_a_tagged_exp():
    prog = trace(requested, SIZES, cl.f32)
    assert [r.kind for r in prog.ops] == ["input", "exp", "reduce_sum", "div"]
    exp = prog.ops[1]
    assert exp.args == (0,) and exp.attrs == (("stabilize", 1),)  # dim normalised to >= 0
    assert prog.stabilize_requests == ((0, 1),)
    assert "exp(%0) stabilize=1" in repr(prog)
    # nothing else carries the request
    assert trace(handwritten, SIZES, cl.f32).stabilize_requests == ()


def test_requests_are_deduplicated_and_dim_zero_counts():
    p = Program("p", cl.f32)
    x, y = p.input("x", [4, 8]), p.input("y", [4, 8])
    for t, dim in ((x, 0), (x, -2), (x, 1), (y, 0), (x, 0)):
        stable_exp(t, dim)
    # dim 0 is a request like any other (it is falsy, the record is not)
    assert p.stabilize_requests == ((0, 0), (0, 1), (1, 0))
    assert bool(p.stabilize_requests)


def test_request_is_validated():
    p = Program("p", cl.f32)
    x = p.input("x", [4, 8])
    with pytest.raises(SpecError, match="softmax: dim 2 out of range for rank 2"):
        stable_exp(x, 2)
    with pytest.raises(SpecError, match="softmax: dim -3 out of range"):
        stable_exp(x, -3)
    with pytest.raises(SpecError, match="softmax on a transposed view"):
        stable_exp(x.mT, -1)
    with pytest.raises(SpecError, match="another Program"):
        p._stable_exp(Program("q", cl.f32).input("x", [4, 8]), -1)
    assert p.stabilize_requests == () and [r.kind for r in p.ops] == ["input"]


def test_search_and_lowering_see_a_plain_exp():
    prog, plain = trace(requested, SIZES, cl.f32), trace(handwritten, SIZES, cl.f32)
    assert prog.to_graph_structure() == plain.to_graph_structure()
    assert [o["op_type"] for o in prog.to_graph_structure()] == [
        "kn_input_op", "kn_exp_op", "kn_reduce_sum_1_op", "kn_div_op", "kn_output_op"]
    assert prog.op_count == plain.op_count == 4

    def ops(p):
        return [o["op_type"] for o in p.to_tile_graph().get_graph_structure()]

    assert ops(prog) == ops(plain)


def test_fingerprint_separates_the_request():
    """The request must never share a search or kernel cache entry with a plain exp."""
    prog, plain = (trace(f, SIZES, cl.f32) for f in (requested, handwritten))
    assert prog.fingerprint() != plain.fingerprint()

    def over(dim):
        def f(x: cl.Tensor["R", "C"]):
            return cl.softmax(x, dim)
        return trace(f, SIZES, cl.f32).fingerprint()

    assert over(-1) == over(1) == prog.fingerprint()
    assert over(0) != over(1)


@pytest.mark.parametrize("shape, dim", [((4, 32), -1), ((4, 32), 0), ((16, 16), 0),
                                        ((3, 16, 16), 1), ((3, 8, 24), -1)])
@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_reference_is_the_exact_softmax(shape, dim, scale):
    p = Program("p", cl.f32)
    x = p.input("x", list(shape))
    p.output(cl.softmax(x, dim))
    torch.manual_seed(0)
    data = torch.randn(*shape) * scale
    (got,) = p.reference(data)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, torch.softmax(data, dim), rtol=1e-6, atol=1e-7)


def test_reference_of_the_plain_twin_overflows():
    """What the request buys: the same records without the tag are not finite at x100."""
    torch.manual_seed(0)
    data = torch.randn(4, 32) * 100
    (plain,) = trace(handwritten, SIZES, cl.f32).reference(data)
    (exact,) = trace(requested, SIZES, cl.f32).reference(data)
    assert not torch.isfinite(plain).all()
    assert torch.isfinite(exact).all()


@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_attention_reference(scale):
    shapes = dict(H=2, SQ=8, SKV=48, D=16)
    prog = trace(attention, shapes, cl.f32)
    assert [r.kind for r in prog.ops] == ["input", "input", "input", "matmul", "div", "exp",
                                          "reduce_sum", "div", "matmul"]
    assert prog.stabilize_requests == ((4, 2),)
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, n, 16) for n in (8, 48, 48))
    q = q * scale
    (got,) = prog.reference(q, k, v)
    want = torch.softmax((q @ k.mT) / 4.0, -1) @ v
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
