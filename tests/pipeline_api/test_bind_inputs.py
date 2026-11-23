"""CPU-only: kernel placeholders bind to Program inputs by input identity, not by shape."""

import cleave.lang as cl
from cleave.frontend import trace
from cleave.pipeline import GraphSet
from cleave.pipeline.plans import bind_inputs, lower


def _bound(fn, shapes):
    (g,) = GraphSet.identity(trace(fn, shapes, cl.f16))
    (ir,) = lower(g)
    return bind_inputs(ir, g)


def test_same_shaped_inputs_bind_by_identity():
    # b is read first, so its placeholder comes first; a square shape cannot tell them apart.
    def f(a: cl.Tensor["N", "N"], b: cl.Tensor["N", "N"]):
        return b @ a

    assert _bound(f, dict(N=128)) == ("PlaceHolder_input_1", "PlaceHolder_input_2")


def test_unread_input_binds_to_none():
    def f(a: cl.Tensor["M", "N"], b: cl.Tensor["M", "N"]):
        return a * 2.0

    assert _bound(f, dict(M=64, N=128)) == ("PlaceHolder_input_1", None)
