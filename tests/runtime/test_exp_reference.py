"""The graph reference of an ``Exp`` honours its ``Stabilize`` record (CPU, torch only).

An annotated Exp evaluates ``exp(x - max over the softmax axis)``, per node; an Exp without
the record is the plain ``exp``. The record is set by hand here: what writes it is tested in
``tests/lower/test_scale_analysis.py``.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

from cleave.ir.graph import OutputNode
from cleave.ir.ops import Add, Div, Exp, ReduceSum, Stabilize
from cleave.runtime.reference import REFERENCES, execute, fingerprint_seed, placeholder_order
from tests import _graphs as G


def _softmax(dims, axis=-1, *, record=None, dtype="float32", name="exp_0"):
    """``exp(x) / sum(exp(x), axis)`` over one of the last two axes; returns (exp, div)."""
    last = axis in (-1, len(dims) - 1)
    exp = Exp(dims, name=name, dtype=dtype, out_dtype="float32")
    rowsum = ReduceSum(dims, axis=axis, name=f"rowsum_{name}")
    div = Div(dims, broadcast=0 if last else 1, name=f"div_{name}")
    G._connect(exp, rowsum, 0, 0)
    G._connect(exp, div, 0, 0)
    G._connect(rowsum, div, 0, 1)
    exp.stabilize = record
    return exp, div


def _graph(dims, axis=-1, **kw):
    _, div = _softmax(dims, axis, **kw)
    OutputNode(div, dtype="float32")
    return G._finalize(div)


def _x(*shape, scale=1.0):
    torch.manual_seed(0)
    return torch.randn(*shape) * scale


@pytest.mark.parametrize("dims, axis, axes", [((4, 32), -1, (-1,)), ((4, 32), 0, (-2,)),
                                              ((16, 16), 0, (-2,)), ((3, 8, 24), -1, (-1,)),
                                              ((3, 8, 24), 1, (-2,)), ((3, 16, 16), 1, (-2,))])
@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_annotated_exp_is_the_exact_softmax(dims, axis, axes, scale):
    x = _x(*dims, scale=scale)
    (out,) = execute(_graph(dims, axis, record=Stabilize(axes)), [x])
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.softmax(x, axis), rtol=1e-6, atol=1e-7)


def test_unannotated_twin_overflows_on_the_same_inputs():
    x = _x(4, 32, scale=100.0)
    (plain,) = execute(_graph((4, 32)), [x])
    (exact,) = execute(_graph((4, 32), record=Stabilize((-1,))), [x])
    assert not torch.isfinite(plain).all()
    assert torch.isfinite(exact).all()


def test_record_changes_no_benign_output():
    x = _x(4, 32)
    (plain,) = execute(_graph((4, 32)), [x])
    (exact,) = execute(_graph((4, 32), record=Stabilize((-1,))), [x])
    torch.testing.assert_close(exact, plain, rtol=1e-6, atol=1e-7)


def test_record_is_per_node():
    """``softmax(x) + softmax(y)`` with only the first Exp annotated: its neighbour is plain."""
    dims = (4, 32)
    exact, left = _softmax(dims, record=Stabilize((-1,)), name="exp_0")
    plain, right = _softmax(dims, name="exp_1")
    total = Add(dims, name="add_0")
    G._connect(left, total, 0, 0)
    G._connect(right, total, 0, 1)
    OutputNode(total, dtype="float32")
    graph = G._finalize(total)

    def run(exact_in, plain_in):
        by_name = {exact.inputs[0].src_node.name: exact_in, plain.inputs[0].src_node.name: plain_in}
        (out,) = execute(graph, [by_name[p.name] for p in placeholder_order(graph)])
        return out

    large, small = _x(*dims, scale=100.0), _x(*dims) + 1.0
    out = run(large, small)
    assert torch.isfinite(out).all()
    want = torch.softmax(large, -1) + torch.softmax(small, -1)
    torch.testing.assert_close(out, want, rtol=1e-6, atol=1e-7)
    # the neighbour is not rescued: the same large logits overflow there
    assert not torch.isfinite(run(small, large)).all()


def test_storage_precision_is_modelled_before_the_shift():
    """An fp16 operand is rounded as the kernel's buffer rounds it; the max and the exp
    are taken in the accumulation dtype."""
    x = _x(4, 32, scale=100.0)
    (out,) = execute(_graph((4, 32), record=Stabilize((-1,)), dtype="float16"), [x])
    torch.testing.assert_close(out, torch.softmax(x.half().float(), -1), rtol=1e-6, atol=1e-7)


def test_record_does_not_change_the_validation_seed():
    """The flat graph gains no node, so its validation inputs are those of the plain graph."""
    marked, plain = _graph((4, 32), record=Stabilize((-1,))), _graph((4, 32))
    assert fingerprint_seed(marked) == fingerprint_seed(plain)


def test_reference_is_registered_for_the_exact_class():
    from cleave.ir.ops import CausalMaskedExp, MaskedExp

    assert Exp in REFERENCES
    assert MaskedExp not in REFERENCES and CausalMaskedExp not in REFERENCES
