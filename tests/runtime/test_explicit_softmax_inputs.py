"""Which exp is an exact softmax, by structure, where update rules are searched (CPU).

With the max in the flat graph there is no record on the exp: an exp is the exp of an exact
softmax when it reads ``x - max(x, axis)`` -- a ``Sub`` whose second operand is a
``ReduceMax`` of its first (``runtime.inputs.softmax_axis``). Its argument is then never
positive, whatever the logits are, which is the claim the adversarial input cases test.
From that fact the inputs are shaped: which matmul makes the scores, which of its operands
is the key, which input is the logits, and whether a plain ``exp`` next to the softmax
keeps the cases away. A hand-written ``exp(x - max(x))`` is the same graph as
``cl.softmax`` and makes the same claim.

The reference every tiling is held to is the lowered flat graph's, never the program's:
``execute(lower(identity graph))`` is ``Program.reference`` at x1 and at x100 for a softmax
over the last, the middle and the first axis.
"""

import functools

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.frontend import trace
from cleave.ir.ops import Exp, Matmul, ReduceMax, Sub
from cleave.ir.update_rule import use_searched_updates
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.inputs import (
    ADVERSARIAL,
    DISTRIBUTIONS,
    adversarial_cases,
    key_operand,
    logit_inputs,
    score_matmuls,
    softmax_axis,
    why_not_adversarial,
)
from cleave.runtime.reference import build_inputs_and_reference

from tests import _explicit_softmax as X
from tests._lowering import run_lowered

SMALL = dict(H=32, SQ=64, SKV=256, D=64)

# The structural claim is the searched path's: every test runs with the switch on, and the
# ones about the switch turn it off where they say so.
pytestmark = pytest.mark.usefixtures("searched_updates")


def _lowered(fn, sizes, dtype=cl.f16):
    """``(program, lowered identity graph)``, traced and lowered with the switch on:
    ``cl.softmax`` records the explicit max."""
    with use_searched_updates():
        prog = trace(fn, sizes, dtype)
        (graph,) = GraphSet.identity(prog)
        (ir,) = lower(graph)
    return prog, ir


def _exps(ir):
    return [n for n in ir.nodes if type(n) is Exp]


# ---------------------------------------------------------------------------- softmax_axis


def softmax_last(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1)


def softmax_middle(x: cl.Tensor["B", "R", "C"]):
    return cl.softmax(x, 1)


def softmax_first(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, 0)


def query_softmax(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                  v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-2) @ v


def key_softmax(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


@pytest.mark.parametrize(("fn", "sizes", "axis"), [
    (softmax_last, dict(R=16, C=24), -1),
    (softmax_middle, dict(B=3, R=16, C=24), -2),
    (softmax_first, dict(R=16, C=24), -2),
    (query_softmax, dict(H=2, SQ=8, SKV=48, D=16), -2),
    (key_softmax, dict(H=2, SQ=8, SKV=48, D=16), -1),
], ids=["last", "middle", "first", "over the queries", "over the keys"])
def test_softmax_axis_is_read_off_the_explicit_max(fn, sizes, axis):
    _, ir = _lowered(fn, sizes)
    (exp,) = _exps(ir)
    assert getattr(exp, "stabilize", None) is None  # no record: the structure says it
    assert softmax_axis(exp) == axis
    (m,) = [n for n in ir.nodes if type(n) is ReduceMax]
    assert m.flat_axis - len(exp.get_shape(0)) == axis
    # only the exp: the max, the subtraction and everything else are not softmax exps
    assert [n.name for n in ir.nodes if softmax_axis(n) is not None] == [exp.name]


def test_softmax_axis_is_a_claim_only_where_update_rules_are_searched():
    """The structure is the same graph with the switch off (the max is written out); the
    claim it makes is the searched path's, so the hand-derived path's graphs keep theirs."""
    ir = X.lowered(X.flash_attention, SMALL)
    (exp,) = _exps(ir)
    with use_searched_updates():
        assert softmax_axis(exp) == -1
    with use_searched_updates(False):
        assert softmax_axis(exp) is None


def shifted_by_another_max(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]):
    return cl.exp(x - cl.max(y, -1, keepdim=True))


def shifted_by_a_constant(x: cl.Tensor["R", "C"]):
    return cl.exp(x - 3.0)


def shifted_by_the_max_of_a_sum(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]):
    return cl.exp(x - cl.max(x + y, -1, keepdim=True))


@pytest.mark.parametrize("fn", [shifted_by_another_max, shifted_by_a_constant,
                                shifted_by_the_max_of_a_sum])
def test_an_exp_of_anything_else_is_no_softmax(fn):
    sizes = dict(R=16, C=24)
    _, ir = _lowered(fn, sizes)
    (exp,) = _exps(ir)
    assert softmax_axis(exp) is None
    assert why_not_adversarial(ir) or adversarial_cases(ir) == ()


def test_a_hand_written_max_is_the_softmax_cl_softmax_records():
    """``exp(x - max(x))`` written by hand traces to the graph ``cl.softmax`` records under
    the switch, and is held to the same claim."""
    _, by_hand = _lowered(X.softmax_rows, dict(R=16, C=24))
    _, recorded = _lowered(softmax_last, dict(R=16, C=24))
    shape = [(type(n).__name__, tuple(int(s) for s in n.get_shape(0))) for n in by_hand.nodes
             if not n.is_output()]
    assert shape == [(type(n).__name__, tuple(int(s) for s in n.get_shape(0)))
                     for n in recorded.nodes if not n.is_output()]
    for ir in (by_hand, recorded):
        assert [softmax_axis(n) for n in _exps(ir)] == [-1]
        assert adversarial_cases(ir) == ADVERSARIAL and not why_not_adversarial(ir)


# ------------------------------------------------------------------- score matmuls, key, logits


def test_the_key_operand_follows_the_softmax_axis():
    """The scores have two readers, the max and the subtraction it feeds: that is one path,
    and the softmax axis of the exp behind it decides which operand is the key."""
    for fn, key in ((key_softmax, 1), (query_softmax, 0)):
        _, ir = _lowered(fn, SMALL)
        (scores,) = score_matmuls(ir)
        assert scores is [n for n in ir.nodes if type(n) is Matmul][0]
        assert key_operand(ir, scores) == key, fn.__name__
        assert adversarial_cases(ir) == ADVERSARIAL and not why_not_adversarial(ir)


def query_softmax_and_its_max(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                              v: cl.Tensor["H", "SKV", "D"]):
    s = (q @ k.mT) / q.shape[-1] ** 0.5
    m = cl.max(s, -2, keepdim=True)
    e = cl.exp(s - m)
    return (e / cl.sum(e, -2, keepdim=True)) @ v, m


def key_softmax_and_its_max(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                            v: cl.Tensor["H", "SKV", "D"]):
    s = (q @ k.mT) / q.shape[-1] ** 0.5
    m = cl.max(s, -1, keepdim=True)
    e = cl.exp(s - m)
    return (e @ v) / cl.sum(e, -1, keepdim=True), m


def test_the_key_operand_of_a_softmax_that_also_returns_its_max():
    """The max may be read by more than the subtraction (a split partial returns it): the
    scores still have one path, and the softmax axis still decides. (It used to be taken
    for the last axis as soon as the max had a second reader: the cases of a softmax over
    the queries were then laid along the keys.)"""
    for fn, key in ((key_softmax_and_its_max, 1), (query_softmax_and_its_max, 0)):
        _, ir = _lowered(fn, SMALL)
        (m,) = [n for n in ir.nodes if type(n) is ReduceMax]
        assert len(m.outputs) == 2
        (scores,) = score_matmuls(ir)
        assert key_operand(ir, scores) == key, fn.__name__
        assert adversarial_cases(ir) == ADVERSARIAL and not why_not_adversarial(ir)


def test_the_key_operand_without_the_claim_is_the_last_axis():
    """With the switch off the same explicit graph makes no claim: the last axis, as for any
    graph that makes none."""
    _, ir = _lowered(query_softmax, SMALL)
    (scores,) = score_matmuls(ir)
    with use_searched_updates(False):
        assert key_operand(ir, scores) == 1


def test_logits_that_are_an_input():
    _, ir = _lowered(X.flash_rows, dict(R=64, C=1024, D=64))
    ((placeholder, axis, sign),) = logit_inputs(ir)
    (exp,) = _exps(ir)
    (sub,) = [n for n in ir.nodes if type(n) is Sub]
    assert exp.inputs[0].src_node is sub and sub.inputs[0].src_node is placeholder
    assert (axis, sign) == (1, 1.0) and placeholder.is_placeholder()
    assert score_matmuls(ir) == [] and adversarial_cases(ir) == ADVERSARIAL


def test_search_logits_of_the_split_combine_are_its_maxes():
    """The combine of a split built from the loop state reads the max of every split as an
    input; its exp reads ``m_i - max_i m_i``: its logits are that input, along the split
    axis, and all six cases apply to the combine alone."""
    from cleave.lower.state_split import combine_graph

    from tests.update._attention import attention, flash

    program = trace(attention, dict(H=4, SQ=8, SKV=256, D=16), cl.f16)
    graph = flash(P.superoptimize(program, P.SuperoptConfig(max_ops=11, cache=False)))
    (ir,) = lower(graph)
    (full,) = [n for n in ir.nodes if getattr(n, "streaming", None) is not None]
    combine = combine_graph(ir, full.name, full.streaming.rules, 4, split_pos=1,
                            out_dtype=graph.dtype)
    found = logit_inputs(combine)
    assert len(found) == 1
    ((placeholder, axis, sign),) = found
    assert placeholder.boundary.role == "shift" and (axis, sign) == (1, 1.0)
    assert [softmax_axis(n) for n in _exps(combine)] == [-2]
    assert adversarial_cases(combine) == ADVERSARIAL and not why_not_adversarial(combine)


def softmax_plus_exp(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1) + cl.exp(x)


def test_a_plain_exp_next_to_a_softmax_makes_no_claim():
    """The plain ``exp(x)`` reads the logits of the softmax: no case is run, and the graph
    says why (its reference would overflow by the program's own definition)."""
    _, ir = _lowered(softmax_plus_exp, dict(R=64, C=256))
    plain, exact = sorted(_exps(ir), key=lambda n: softmax_axis(n) is not None)
    assert softmax_axis(plain) is None and softmax_axis(exact) == -1
    assert adversarial_cases(ir) == ()
    assert why_not_adversarial(ir) == (
        f"input {logit_inputs(ir)[0][0].name!r} also feeds {plain.name!r}, an exp that is not "
        "stabilized: on shaped logits its reference overflows")


def flash_next_to_an_exp(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                         v: cl.Tensor["H", "SKV", "D"], b: cl.Tensor["H", "SQ", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v + cl.exp(b)


def test_a_plain_exp_of_another_input_leaves_the_cases_alone():
    """Its own input is not shaped, and it does not count as a softmax: the scores are."""
    _, ir = _lowered(flash_next_to_an_exp, SMALL)
    assert [softmax_axis(n) for n in _exps(ir)].count(None) == 1
    (scores,) = score_matmuls(ir)
    assert scores is [n for n in ir.nodes if type(n) is Matmul][0]
    assert adversarial_cases(ir) == ADVERSARIAL and not why_not_adversarial(ir)


def test_the_reference_publishing_holds_an_explicit_softmax_to_every_case():
    """``pipeline.build``: a graph whose exp reads ``x - max(x)`` is held to the adversarial
    cases whoever wrote it; with the switch off a graph without a request makes no claim."""
    from cleave.pipeline.build import _reference_fn

    prog, ir = _lowered(X.flash_attention, SMALL)
    published = []

    def publish(reference_cache_dir, graph, dtype, distributions, **kw):
        published.append(tuple(distributions))
        return "ok"

    reference = _reference_fn(("normal",), prog, publish)
    reference("dir", ir, "float16")
    with use_searched_updates(False):
        reference("dir", ir, "float16")
    assert published == [("normal",) + ADVERSARIAL, ("normal",)]


# ------------------------------------------------------- the lowered graph is the program


def _softmax(dims, dim):
    def softmax(x):
        return cl.softmax(x, dim)

    softmax.__annotations__ = {"x": cl.Tensor[dims]}
    return softmax


# name: (fn, sizes, inputs that hold logits (scaled at x100))
CASES = {
    "last": (_softmax(("R", "C"), -1), dict(R=16, C=24), {"x"}),
    "last_square": (_softmax(("R", "C"), -1), dict(R=16, C=16), {"x"}),
    "middle": (_softmax(("B", "R", "C"), 1), dict(B=3, R=16, C=24), {"x"}),
    "first": (_softmax(("R", "C"), 0), dict(R=16, C=24), {"x"}),
    "first_one_symbol": (_softmax(("N", "N"), 0), dict(N=16), {"x"}),
    "attention": (key_softmax, dict(H=2, SQ=8, SKV=48, D=16), {"q"}),
    "over_the_queries": (query_softmax, dict(H=2, SQ=8, SKV=48, D=16), {"q"}),
    "first_axis_summed": (X.column_average, dict(H=2, K=48, N=8), {"x"}),
}


@functools.lru_cache(maxsize=None)
def _case(name, dtype=cl.f32):
    fn, sizes, _ = CASES[name]
    return _lowered(fn, sizes, dtype)


def _inputs(prog, logits, scale):
    torch.manual_seed(0)
    return [torch.randn(*s.shape) * (scale if s.name in logits else 1.0) for s in prog.inputs]


@pytest.mark.parametrize("scale", [1.0, 100.0], ids=["x1", "x100"])
@pytest.mark.parametrize("name", CASES)
def test_lowered_graph_equals_program_reference(name, scale):
    """The program's own reference evaluates the recorded max and subtraction; the lowered
    graph evaluates its ``ReduceMax``, ``Sub`` and ``Exp`` nodes, each by its own reference.
    The two are the same float32 arithmetic in the same order: equal, at x1 and at x100,
    where a plain exp of the logits would overflow."""
    prog, ir = _case(name)
    assert [r.kind for r in prog._records].count("reduce_max") == 1
    assert sum(softmax_axis(n) is not None for n in _exps(ir)) == 1
    inputs = _inputs(prog, CASES[name][2], scale)
    (got,) = run_lowered(ir, prog, inputs)
    (want,) = prog.reference(*inputs)
    assert torch.isfinite(want).all() and torch.isfinite(got).all()
    assert torch.equal(got.reshape(want.shape), want), float(
        (got.reshape(want.shape) - want).abs().max())


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("name", ["attention", "over_the_queries", "last", "first_axis_summed"])
def test_validation_reference_is_finite_on_adversarial_inputs(name, distribution):
    """The reference every tiling of the graph is validated against, in the program's fp16,
    on the inputs the validation generates."""
    _, ir = _case(name, cl.f16)
    _, reference, _ = build_inputs_and_reference(ir, "float16", device="cpu",
                                                 distribution=distribution)
    assert all(torch.isfinite(r).all() for r in reference)
    assert all(float(r.float().abs().max()) > 0 for r in reference)
