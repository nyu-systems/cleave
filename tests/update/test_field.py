"""The field the update search tests in is the first pass's own: the evaluator of
``pipeline.update.field`` on a lowered flat graph gives, element for element, the
fingerprint the native superoptimizer computes for the same program
(``CyKNGraph.output_fingerprints``, the probabilistic verifier's own evaluation).
CPU only, no search.
"""

import pytest

torch = pytest.importorskip("torch")

import cleave.lang as cl
from cleave.frontend import trace
from cleave.pipeline.plans import _lower_structure
from cleave.pipeline.superopt import GraphSet
from cleave.pipeline.update import field
from cleave.runtime.reference import placeholder_order


def softmax_v(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    return cl.softmax(x, dim=-1) @ v


def flash(q: cl.Tensor["H", "S", "D"], k: cl.Tensor["H", "T", "D"],
          v: cl.Tensor["H", "T", "E"]):
    s = (q @ k.mT) / 8.0
    e = cl.exp(s - cl.max(s, -1, keepdim=True))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def norm_matmul(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    return cl.rms_norm(x, dim=-1) @ w


def scalars(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]):
    """silu, sqrt, a constant on either side of - and of /, broadcasting, two outputs."""
    a = cl.silu(x) * y - 1.0
    b = 2.0 - x / (y + 0.5)
    return cl.sum(a * b, -1, keepdim=True), cl.sqrt(x * x + 1.0)


def normed_scores(q: cl.Tensor["H", "S", "D"], k: cl.Tensor["H", "T", "D"]):
    """rsqrt and a mean."""
    qn = q * cl.rsqrt((q * q).mean(-1, keepdim=True) + 1e-6)
    return qn @ k.mT


CASES = [(softmax_v, dict(R=4, C=8, D=3)), (flash, dict(H=2, S=3, T=8, D=4, E=5)),
         (norm_matmul, dict(M=3, K=8, N=5)), (scalars, dict(R=3, C=6)),
         (normed_scores, dict(H=2, S=3, T=5, D=4))]


@pytest.mark.parametrize("fn, shapes", CASES, ids=[c[0].__name__ for c in CASES])
def test_the_field_is_the_first_pass_fingerprint(fn, shapes):
    prog = trace(fn, shapes, cl.f16)
    native = prog.to_tile_graph().cygraph.kn_graph().output_fingerprints()
    (graph,) = list(GraphSet.identity(prog))
    (ir,) = _lower_structure(graph, graph.structure)
    inputs = field.native_inputs([tuple(int(d) for d in p.get_shape(0))
                                  for p in placeholder_order(ir)])
    _, outputs = field.values(ir, inputs)
    assert len(outputs) == len(native)
    for mine, theirs in zip(outputs, native):
        assert mine.reshape(-1).tolist() == theirs


def test_exp_is_a_homomorphism_and_zero_has_inverse_one():
    x = torch.randint(0, field.PQ, (64,))
    y = torch.randint(0, field.PQ, (64,))
    assert torch.equal(field.exp(field.add(x, y)), field.mul(field.exp(x), field.exp(y)))
    assert torch.equal(field.exp(field.sub(x, y)), field.div(field.exp(x), field.exp(y)))
    zero = torch.zeros(1, dtype=torch.int64)
    assert torch.equal(field.div(torch.tensor([5]), zero), torch.tensor([5]))  # as natively


def normalized(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    return (x / cl.sum(x, -1, keepdim=True)) @ v


def test_the_zero_divisors_are_where_a_value_divides_by_zero_in_a_residue():
    """``(x / sum(x)) @ v``: the quotient divides by zero in a residue where the row sum is
    zero in it, and the product reads every element of its row; the inputs and the sum
    itself divide by nothing."""
    (graph,) = list(GraphSet.identity(trace(normalized, dict(R=64, C=8, D=3), cl.f16)))
    (ir,) = _lower_structure(graph, graph.structure)
    shapes = [tuple(int(d) for d in p.get_shape(0)) for p in placeholder_order(ir)]
    names = {type(n).__name__: n.name for n in ir.nodes}
    for seed in range(4):
        env, _, zeros = field.values(ir, field.random_inputs(shapes, seed), with_zeros=True)
        want = field.zero_bits(env[names["ReduceSum"]]).reshape(-1, 1)
        assert not zeros[names["ReduceSum"]].any()
        assert torch.equal(zeros[names["Div"]], want.expand(-1, 8))
        assert torch.equal(zeros[names["Matmul"]], want.expand(-1, 3))
    hits = [field.zero_bits(field.values(ir, field.random_inputs(shapes, s))[0][
        names["ReduceSum"]]).any() for s in range(4)]
    assert any(hits)  # 64 rows a draw: a row sum that is zero in a residue comes up
    # bit 1 is P, bit 2 is Q
    assert field.zero_bits(torch.tensor([0, field.P, field.Q, 1])).tolist() == [3, 1, 2, 0]


def test_an_op_without_field_semantics_is_no_field():
    from cleave.ir.ops import Exp2

    (graph,) = list(GraphSet.identity(trace(softmax_v, dict(R=2, C=4, D=2), cl.f16)))
    (ir,) = _lower_structure(graph, graph.structure)
    node = next(n for n in ir.nodes if type(n).__name__ == "Exp")
    node.__class__ = Exp2  # an op the first pass does not fingerprint
    with pytest.raises(field.NoField):
        field.values(ir, field.random_inputs([(2, 4), (4, 2)], 0))
