"""The graph reference of a softmax is exact by structure (CPU, torch only).

``cl.softmax`` records its max, so the stable value comes from the graph itself: the
``ReduceMax``, the ``Sub`` and the ``Exp`` each have their own per-node reference, and an
``Exp`` is the plain ``exp`` whatever it reads. An ``Exp`` behind ``x - max(x)`` is then the
exact softmax at any logit scale (``cleave.runtime.inputs.softmax_axis`` names it), and an
``exp / sum`` without the max overflows where its logits do. The graphs are the programs'
own identity graphs (``tests/_explicit_softmax.py::lowered``).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave.frontend import trace
from cleave.ir.ops import CausalMaskedExp, Exp, MaskedExp
from cleave.pipeline.plans import _lower_structure, bind_inputs
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.inputs import softmax_axis
from cleave.runtime.reference import REFERENCES, execute, placeholder_order, values
from tests._explicit_softmax import lowered, stable_exp


def softmax_rows(x: cl.Tensor["R", "C"]):
    e = stable_exp(x, -1)
    return e / cl.sum(e, -1, keepdim=True)


def softmax_cols(x: cl.Tensor["R", "C"]):
    e = stable_exp(x, 0)
    return e / cl.sum(e, 0, keepdim=True)


def softmax_mid(x: cl.Tensor["B", "R", "C"]):
    e = stable_exp(x, 1)
    return e / cl.sum(e, 1, keepdim=True)


def softmax_last3(x: cl.Tensor["B", "R", "C"]):
    e = stable_exp(x, -1)
    return e / cl.sum(e, -1, keepdim=True)


def plain_rows(x: cl.Tensor["R", "C"]):
    e = cl.exp(x)
    return e / cl.sum(e, -1, keepdim=True)


def exact_next_to_plain(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]):
    """``softmax(x) + exp(y) / sum(exp(y))``: one exp behind its max, one plain."""
    e, f = stable_exp(x, -1), cl.exp(y)
    return e / cl.sum(e, -1, keepdim=True) + f / cl.sum(f, -1, keepdim=True)


def _x(*shape, scale=1.0):
    torch.manual_seed(0)
    return torch.randn(*shape) * scale


@pytest.mark.parametrize("fn, sizes, dim, axis", [
    (softmax_rows, dict(R=4, C=32), -1, -1), (softmax_cols, dict(R=4, C=32), 0, -2),
    (softmax_cols, dict(R=16, C=16), 0, -2), (softmax_last3, dict(B=3, R=8, C=24), -1, -1),
    (softmax_mid, dict(B=3, R=8, C=24), 1, -2), (softmax_mid, dict(B=3, R=16, C=16), 1, -2),
])
@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_an_exp_behind_x_minus_its_max_is_the_exact_softmax(fn, sizes, dim, axis, scale):
    ir = lowered(fn, sizes, cl.f32)
    (exp,) = [n for n in ir.nodes if type(n) is Exp]
    assert softmax_axis(exp) == axis
    x = _x(*sizes.values(), scale=scale)
    (out,) = execute(ir, [x])
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.softmax(x, dim), rtol=1e-6, atol=1e-7)


def test_the_plain_twin_overflows_on_the_same_inputs():
    sizes = dict(R=4, C=32)
    x = _x(4, 32, scale=100.0)
    (plain,) = execute(lowered(plain_rows, sizes, cl.f32), [x])
    (exact,) = execute(lowered(softmax_rows, sizes, cl.f32), [x])
    assert not torch.isfinite(plain).all()
    assert torch.isfinite(exact).all()
    # ... and on benign inputs the two are the same function
    x = _x(4, 32)
    (plain,) = execute(lowered(plain_rows, sizes, cl.f32), [x])
    (exact,) = execute(lowered(softmax_rows, sizes, cl.f32), [x])
    torch.testing.assert_close(exact, plain, rtol=1e-6, atol=1e-7)


def test_exactness_is_per_node():
    """The exp behind its max is exact; the plain exp next to it is not rescued."""
    prog = trace(exact_next_to_plain, dict(R=4, C=32), cl.f32)
    (graph,) = GraphSet.identity(prog)
    (ir,) = _lower_structure(graph, graph.structure)
    assert sorted((softmax_axis(n) for n in ir.nodes if type(n) is Exp),
                  key=lambda a: (a is None, a)) == [-1, None]
    names = bind_inputs(ir, prog)  # the placeholder of x, then that of y

    def run(x, y):
        by_name = dict(zip(names, (x, y)))
        (out,) = execute(ir, [by_name[p.name] for p in placeholder_order(ir)])
        return out

    large, small = _x(4, 32, scale=100.0), _x(4, 32) + 1.0
    out = run(large, small)
    assert torch.isfinite(out).all()
    want = torch.softmax(large, -1) + torch.softmax(small, -1)
    torch.testing.assert_close(out, want, rtol=1e-6, atol=1e-7)
    assert not torch.isfinite(run(small, large)).all()


def test_storage_precision_is_modelled_before_the_shift():
    """A float16 operand is rounded as the kernel's buffer rounds it; the max, the
    subtraction and the exp are float32 values, as lowering types them."""
    ir = lowered(softmax_rows, dict(R=4, C=32), cl.f16)
    (exp,) = [n for n in ir.nodes if type(n) is Exp]
    x = _x(4, 32, scale=100.0)
    named, _ = values(ir, [x.half()])
    stored = x.half().float()
    want = (stored - stored.amax(-1, keepdim=True)).exp()
    assert named[exp.name].dtype == torch.float32
    torch.testing.assert_close(named[exp.name], want, rtol=1e-6, atol=1e-7)


def test_the_exp_reference_is_the_plain_exp():
    """Registered for the exact class only, and one function for every ``Exp``: whatever
    an exp reads, its reference is ``exp`` of it."""
    assert Exp in REFERENCES
    assert MaskedExp not in REFERENCES and CausalMaskedExp not in REFERENCES
    ir = lowered(softmax_rows, dict(R=4, C=32), cl.f32)
    (exp,) = [n for n in ir.nodes if type(n) is Exp]
    x = _x(4, 32, scale=100.0)
    named, _ = values(ir, [x])
    shifted = named[exp.inputs[0].src_node.name]
    torch.testing.assert_close(named[exp.name], shifted.exp(), rtol=0, atol=0)
    assert float(shifted.max()) == 0.0  # the max of each row is subtracted: exp <= 1
