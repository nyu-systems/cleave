import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")
tvm = pytest.importorskip("tvm")

from cleave.ir.graph import Graph, OutputNode, PlaceHolderNode
from cleave.ir.ops import (
    Accumulate,
    AccumulateMax,
    CausalMaskedScore,
    Copy,
    Div,
    Exp2,
    Log2,
    Lse2,
    MaskedDiv,
    MaskedExp,
    MaskedLog2,
    MaskedLse2,
    MaskedScore,
    Mul,
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


@pytest.mark.parametrize("op", [Copy, Accumulate, AccumulateMax])
def test_identity_ops(op):
    a = _rand()
    torch.testing.assert_close(_run(op(SHAPE), a), a)


def test_mul_broadcast_over_the_last_axis_of_a_three_dim_tile():
    """``broadcast == 0`` with three non-batch dims: B is A without its last axis."""
    a, b = torch.randn(2, 4, 8), torch.randn(2, 4)
    out = _run(Mul((2, 4, 8), broadcast=0, n_batch_dims=0), a, b)
    torch.testing.assert_close(out, a * b[:, :, None])


def test_masked_div_is_refused_not_unmasked():
    with pytest.raises(NotImplementedError, match="MaskedDiv"):
        _run(MaskedDiv(SHAPE), _rand(), _rand(positive=True))


def test_masked_log2_is_refused_not_plain_log2():
    with pytest.raises(NotImplementedError, match="MaskedLog2"):
        _run(MaskedLog2(SHAPE), _rand(positive=True))


def test_exp2():
    x = _rand()
    torch.testing.assert_close(_run(Exp2(SHAPE), x), torch.exp2(x))


def test_lse2_is_the_log2_of_the_first_operand_plus_the_second():
    """``log2(l) + m``: with ``l = sum exp2(s - m)`` this is the base-2 log-sum-exp of ``s``."""
    ln2 = 0.6931471805599453
    scores = _rand() * 40.0  # far apart: the plain sum of exp2 would overflow float32
    m = scores.amax(dim=-1)
    l = torch.exp2(scores - m[:, None]).sum(dim=-1)
    out = _run(Lse2((SHAPE[0],)), l, m)
    torch.testing.assert_close(out, torch.log2(l) + m)
    # the same quantity by torch's own (max-subtracting) float32 logsumexp
    stable = torch.logsumexp(scores * ln2, dim=-1) / ln2
    torch.testing.assert_close(out, stable, rtol=1e-5, atol=1e-5)
    # an empty row: l = 0 and the max still at its finite start value give -inf, not NaN
    empty = _run(Lse2((2,)), torch.zeros(2), torch.full((2,), -3.4028234663852886e38))
    assert torch.isneginf(empty).all()


@pytest.mark.parametrize(
    "build",
    [
        lambda: MaskedLse2((SHAPE[0],)),
        lambda: MaskedScore(SHAPE),
        lambda: CausalMaskedScore(SHAPE),
        lambda: MaskedExp(SHAPE, None, shift=True),
    ],
    ids=["MaskedLse2", "MaskedScore", "CausalMaskedScore", "MaskedExp_shift"],
)
def test_masked_softmax_ops_are_refused_not_unmasked(build):
    """Their semantics depend on schedule scalars (the mask, the runtime scale): the
    executor refuses them rather than evaluating the unmasked stand-in."""
    node = build()
    inputs = [torch.rand(tuple(int(d) for d in e.src_node.get_shape(0))) + 0.5
              for e in node.inputs]
    with pytest.raises(NotImplementedError, match=type(node).__name__):
        _run(node, *inputs)

