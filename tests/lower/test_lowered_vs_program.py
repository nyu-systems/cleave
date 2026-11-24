"""The lowered flat graph computes what the program says, at x1 and at x100 (CPU).

Validation compares a kernel with the reference of its lowered flat graph, never with the
program, so this is the link that makes "the kernel validated" mean "the program is right":
``execute(plans.lower(identity graph))`` against ``Program.reference``, which never sees the IR.
The max of every softmax is a node of the lowered graph (``exp(x - max(x))``, which
``runtime.inputs.softmax_axis`` recognizes). At x100 logits both must stay finite; the same
graph with its maxes taken as zero (the plain ``exp``) does not.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave.frontend import trace
from cleave.ir.ops import Exp, ReduceMax
from cleave.pipeline.plans import bind_inputs, lower
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.inputs import DISTRIBUTIONS, softmax_axis
from cleave.runtime.reference import build_inputs_and_reference, placeholder_order, values
from tests._explicit_softmax import stable_exp
from tests._lowering import run_lowered


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
    exact = [n for n in ir.nodes if type(n) is Exp and softmax_axis(n) is not None]
    assert len(exact) == n_softmax == len([n for n in ir.nodes if type(n) is ReduceMax])
    inputs = _inputs(prog, logits, scale)
    (got,) = run_lowered(ir, prog, inputs)
    (want,) = prog.reference(*inputs)
    assert torch.isfinite(want).all() and torch.isfinite(got).all()
    torch.testing.assert_close(got.reshape(want.shape), want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", CASES)
def test_the_graph_without_its_maxes_is_not_finite_at_x100(name):
    """What the max is for: the same graph with every max given as zero -- the plain
    ``exp`` of the logits -- overflows."""
    fn, sizes, logits, _ = CASES[name]
    prog, ir = _lowered(fn, sizes)
    maxes = [n for n in ir.nodes if type(n) is ReduceMax]
    assert maxes
    by_name = dict(zip(bind_inputs(ir, prog), _inputs(prog, logits, 100.0)))
    inputs = [by_name[n.name] for n in placeholder_order(ir)]
    zero = {m.name: torch.zeros(tuple(int(s) for s in m.get_shape(0))) for m in maxes}
    _, (got,) = values(ir, inputs, given=zero)
    assert not torch.isfinite(got).all()
    _, (exact,) = values(ir, inputs)
    assert torch.isfinite(exact).all()


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


# ----------------------------------------- programs the scale analysis refused, served
#
# With a request on a plain exp, the hand-derived pass proved at lowering that the shift of
# every exp cancels, and refused a program where it did not (the square of the numerator, a
# Gram matrix of the weights, a nonlinear op of them, ...). With the max in the graph there is
# nothing to prove: such a program is an ordinary graph whose exp reads ``x - max(x)``, and
# it is served where its rows fit one tile, like a softmax that is returned.


def _weights_of(op):
    def fn(x: cl.Tensor["R", "C"]):
        e = stable_exp(x, -1)
        return op(e) / cl.sum(e, -1, keepdim=True)
    return fn


def cube(x: cl.Tensor["R", "C"]):
    e = stable_exp(x, -1)
    return (e * e) / cl.sum(e, -1, keepdim=True)


def squares(x: cl.Tensor["R", "C"]):
    e = stable_exp(x, -1)
    return (e * e) / cl.sum(e * e, -1, keepdim=True)


def squares_first_axis(x: cl.Tensor["R", "C"]):
    e = stable_exp(x, 0)
    return (e * e) / cl.sum(e * e, 0, keepdim=True)


def softmax_times_its_exp(x: cl.Tensor["R", "C"]):
    e = stable_exp(x, -1)
    return (e / cl.sum(e, -1, keepdim=True)) * e


def gram(x: cl.Tensor["R", "C"]):
    e = stable_exp(x, -1)
    return e @ e.mT


def different_axes(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "R"], u: cl.Tensor["R", "C"]):
    e = stable_exp(x, -1)
    return (e @ v) * (u @ e.mT)


def kv_side(q: cl.Tensor["HQ", "S", "D"], k: cl.Tensor["HKV", "S", "D"]):
    return q @ stable_exp(k, -1).mT


def add_tensor(x: cl.Tensor["R", "C"]):
    e = stable_exp(x, -1)
    return (e + x) / cl.sum(e, -1, keepdim=True)


def sub_tensor(x: cl.Tensor["R", "C"]):
    e = stable_exp(x, -1)
    return (x - e) / cl.sum(e, -1, keepdim=True)


RC, RC_TILE = dict(R=16, C=24), dict(R=64, C=256)
# name: (fn, sizes, sizes at which a tile holds the rows (float16), inputs that hold logits)
FORMERLY_REFUSED = {
    "cube": (cube, RC, RC_TILE, {"x"}),
    "squares": (squares, RC, RC_TILE, {"x"}),
    "squares_first_axis": (squares_first_axis, RC, RC_TILE, {"x"}),
    "softmax_times_its_exp": (softmax_times_its_exp, RC, RC_TILE, {"x"}),
    "gram": (gram, RC, RC_TILE, {"x"}),
    # at C = 256 a block that holds the rows needs 180 KB of shared memory, over the cap:
    # the exp, read transposed by ``u @ e.mT``, is staged through shared as well
    "different_axes": (different_axes, RC, dict(R=64, C=128), {"x"}),
    "kv_side": (kv_side, dict(HQ=4, HKV=2, S=8, D=16), dict(HQ=4, HKV=2, S=64, D=64), {"k"}),
    "add_constant": (_weights_of(lambda e: e + 1.0), RC, RC_TILE, {"x"}),
    "add_tensor": (add_tensor, RC, RC_TILE, {"x"}),
    "sub_tensor": (sub_tensor, RC, RC_TILE, {"x"}),
    "silu": (_weights_of(cl.silu), RC, RC_TILE, {"x"}),
    "sqrt": (_weights_of(cl.sqrt), RC, RC_TILE, {"x"}),
    "exp": (_weights_of(cl.exp), RC, RC_TILE, {"x"}),
}


@pytest.mark.parametrize("scale", [1.0, 100.0], ids=["x1", "x100"])
@pytest.mark.parametrize("name", FORMERLY_REFUSED)
def test_a_program_the_scale_analysis_refused_is_the_program(name, scale):
    fn, sizes, _, logits = FORMERLY_REFUSED[name]
    prog, ir = _lowered(fn, sizes)
    (exp,) = [n for n in ir.nodes if type(n) is Exp and softmax_axis(n) is not None]
    inputs = _inputs(prog, logits, scale)
    (got,) = run_lowered(ir, prog, inputs)
    (want,) = prog.reference(*inputs)
    assert torch.isfinite(want).all() and torch.isfinite(got).all()
    torch.testing.assert_close(got.reshape(want.shape), want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", FORMERLY_REFUSED)
def test_a_program_the_scale_analysis_refused_has_plans_where_a_tile_holds_its_rows(name):
    """At 64x256 in float16 (the scale analysis refused each: "softmax cannot be stabilized: ...")."""
    from cleave import pipeline as P

    fn, _, sizes, _ = FORMERLY_REFUSED[name]
    plans = P.schedule(GraphSet.identity(trace(fn, sizes, cl.f16)), P.ScheduleConfig(topk=4))
    assert len(plans) >= 1, plans.summary()
