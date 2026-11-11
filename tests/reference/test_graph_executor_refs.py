import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")
tvm = pytest.importorskip("tvm")

from cleave.ir.graph import Graph, OutputNode, PlaceHolderNode
from cleave.ir.ops import (
    Accumulate,
    Copy,
    Div,
    Log2,
    MaskedDiv,
    MaskedLog2,
    wire_input,
)
from cleave.ir.graph_utils import find_topo_sort, gather_all_nodes
from cleave.runtime.reference import execute

SHAPE = (4, 8)


def _run(node, *inputs):
    for i, x in enumerate(inputs):
        ph = PlaceHolderNode(name=f"in_{i}")
        ph.set_shape(tuple(x.shape))
        ph.set_dtype(tvm.DataType("float32"))
        wire_input(ph, node, i)
    out = OutputNode(node, dtype="float32")
    graph = Graph(nodes=find_topo_sort(gather_all_nodes(out)))
    return execute(graph, list(inputs))[0]


def _rand(positive=False):
    x = torch.randn(SHAPE)
    return x.abs() + 0.5 if positive else x


def test_log2():
    x = _rand(positive=True)
    torch.testing.assert_close(_run(Log2(SHAPE), x), torch.log2(x))


def test_div_denom_floor_tensor():
    a, b = _rand(), _rand()
    out = _run(Div(SHAPE, denom_floor=0.25), a, b)
    torch.testing.assert_close(out, a / b.clamp_min(0.25))


def test_div_denom_floor_const_on_left():
    a = _rand()
    out = _run(Div(SHAPE, const=1.0, const_on_left=True, denom_floor=0.25), a)
    torch.testing.assert_close(out, 1.0 / a.clamp_min(0.25))


def test_div_const_right():
    a = _rand()
    torch.testing.assert_close(_run(Div(SHAPE, const=2.0), a), a / 2.0)


@pytest.mark.parametrize("op", [Copy, Accumulate])
def test_identity_ops(op):
    a = _rand()
    torch.testing.assert_close(_run(op(SHAPE), a), a)


def test_masked_div_is_refused_not_unmasked():
    with pytest.raises(NotImplementedError, match="MaskedDiv"):
        _run(MaskedDiv(SHAPE), _rand(), _rand(positive=True))


def test_masked_log2_is_refused_not_plain_log2():
    with pytest.raises(NotImplementedError, match="MaskedLog2"):
        _run(MaskedLog2(SHAPE), _rand(positive=True))
