"""The lowered, annotated flat graph computes what the program says, at x1 and at x100 (CPU).

Validation compares a kernel with the reference of its lowered flat graph, never with the
program, so this is the link that makes "the kernel validated" mean "the program is right":
``execute(plans.lower(identity graph))`` against ``Program.reference``, which never sees the IR.
At x100 logits both must stay finite; the plain twin of the same graph does not.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave.frontend import trace
from cleave.ir.ops import Exp
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.inputs import DISTRIBUTIONS
from cleave.runtime.reference import build_inputs_and_reference
from tests._lowering import run_lowered
from tests._softmax import stable_exp


def _softmax(dims, dim):
    def softmax(x):
        return cl.softmax(x, dim)

    softmax.__annotations__ = {"x": cl.Tensor[dims]}
    return softmax


def flash(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"], v: cl.Tensor["HKV", "SKV", "D"]):
    e = stable_exp((q @ k.mT) / q.shape[-1] ** 0.5, -1)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def traced(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"], v: cl.Tensor["HKV", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, -1) @ v


def flash_plus_exp(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"],
                   v: cl.Tensor["HKV", "SKV", "D"], b: cl.Tensor["HQ", "SQ", "D"]):
    return flash(q, k, v) + cl.exp(b)


def two_softmaxes(a: cl.Tensor["R", "C"], b: cl.Tensor["C", "D"]):
    return cl.softmax(a, -1) @ cl.softmax(b, -1)


def flash_twin(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"], v: cl.Tensor["HKV", "SKV", "D"]):
    e = cl.exp((q @ k.mT) / q.shape[-1] ** 0.5)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


MHA = dict(HQ=2, HKV=2, SQ=8, SKV=48, D=16)
GQA = dict(HQ=4, HKV=2, SQ=8, SKV=48, D=16)
SQUARE = dict(HQ=2, HKV=2, SQ=16, SKV=16, D=16)

# name: (fn, sizes, inputs that hold logits (scaled at x100), number of softmax exps)
CASES = {
    "last_rect": (_softmax(("R", "C"), -1), dict(R=16, C=24), {"x"}, 1),
    "last_square": (_softmax(("R", "C"), -1), dict(R=16, C=16), {"x"}, 1),
    "last_one_symbol": (_softmax(("N", "N"), -1), dict(N=16), {"x"}, 1),
    "first_rect": (_softmax(("R", "C"), 0), dict(R=16, C=24), {"x"}, 1),
    "first_square": (_softmax(("R", "C"), 0), dict(R=16, C=16), {"x"}, 1),
    "first_one_symbol": (_softmax(("N", "N"), 0), dict(N=16), {"x"}, 1),
    "middle_rect": (_softmax(("B", "R", "C"), 1), dict(B=3, R=16, C=24), {"x"}, 1),
    "middle_square": (_softmax(("B", "R", "C"), 1), dict(B=3, R=16, C=16), {"x"}, 1),
    "batched_last": (_softmax(("B", "R", "C"), -1), dict(B=3, R=16, C=16), {"x"}, 1),
    "flash_mha": (flash, MHA, {"q"}, 1),
    "flash_gqa": (flash, GQA, {"q"}, 1),
    "flash_square": (flash, SQUARE, {"q"}, 1),
    "traced_mha": (traced, MHA, {"q"}, 1),
    "traced_gqa": (traced, GQA, {"q"}, 1),
    "traced_square": (traced, SQUARE, {"q"}, 1),
    "flash_plus_exp": (flash_plus_exp, MHA, {"q"}, 1),
    "two_softmaxes": (two_softmaxes, dict(R=16, C=24, D=8), {"a", "b"}, 2),
}


def _lowered(fn, sizes, dtype=cl.f32):
    prog = trace(fn, sizes, dtype)
    (graph,) = GraphSet.identity(prog)
    (ir,) = lower(graph)
    return prog, ir


def _inputs(prog, logits, scale):
    torch.manual_seed(0)
    return [torch.randn(*s.shape) * (scale if s.name in logits else 1.0) for s in prog.inputs]


@pytest.mark.parametrize("scale", [1.0, 100.0], ids=["x1", "x100"])
@pytest.mark.parametrize("name", CASES)
def test_lowered_graph_equals_program_reference(name, scale):
    fn, sizes, logits, n_softmax = CASES[name]
    prog, ir = _lowered(fn, sizes)
    assert sum(n.stabilize is not None for n in ir.nodes if type(n) is Exp) == n_softmax
    inputs = _inputs(prog, logits, scale)
    (got,) = run_lowered(ir, prog, inputs)
    (want,) = prog.reference(*inputs)
    assert torch.isfinite(want).all() and torch.isfinite(got).all()
    torch.testing.assert_close(got.reshape(want.shape), want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", CASES)
def test_unmarked_graph_is_not_finite_at_x100(name):
    """What the annotation is for: drop the records and the same graph overflows."""
    fn, sizes, logits, _ = CASES[name]
    prog, ir = _lowered(fn, sizes)
    for node in ir.nodes:
        if type(node) is Exp:
            node.stabilize = None
    (got,) = run_lowered(ir, prog, _inputs(prog, logits, 100.0))
    assert not torch.isfinite(got).all()


@pytest.mark.parametrize("sizes", [dict(HQ=32, HKV=32, SQ=64, SKV=256, D=64),
                                   dict(HQ=32, HKV=8, SQ=64, SKV=256, D=64)], ids=["mha", "gqa"])
@pytest.mark.parametrize("fn", [flash, traced], ids=["flash", "traced"])
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_validation_reference_is_finite_on_adversarial_inputs(fn, sizes, distribution):
    """The reference every tiling of the graph is validated against, in the program's fp16,
    on the inputs the validation generates (``runtime/inputs.py``)."""
    _, ir = _lowered(fn, sizes, cl.f16)
    _, reference, _ = build_inputs_and_reference(ir, "float16", device="cpu", distribution=distribution)
    assert all(torch.isfinite(r).all() for r in reference)
    assert all(float(r.float().abs().max()) > 0 for r in reference)


# ``constant`` is adversarial to a running max (it never moves), not to a plain exp: its
# logits are normal-sized, so the plain twin's reference is finite on it.
@pytest.mark.parametrize("distribution",
                         [d for d in DISTRIBUTIONS if d not in ("normal", "constant")])
def test_validation_reference_of_the_plain_twin_is_not(distribution):
    _, ir = _lowered(flash_twin, dict(HQ=32, HKV=32, SQ=64, SKV=256, D=64), cl.f16)
    _, reference, _ = build_inputs_and_reference(ir, "float16", device="cpu", distribution=distribution)
    assert not all(torch.isfinite(r).all() for r in reference)


def test_validation_reference_of_the_plain_twin_is_finite_on_constant_rows():
    _, ir = _lowered(flash_twin, dict(HQ=32, HKV=32, SQ=64, SKV=256, D=64), cl.f16)
    _, reference, _ = build_inputs_and_reference(ir, "float16", device="cpu", distribution="constant")
    assert all(torch.isfinite(r).all() for r in reference)
