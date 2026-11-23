"""What ``cl.softmax`` records, and which exp of a lowered graph is an exact softmax (CPU).

``cl.softmax`` records the max it is defined with, ``exp(x - max(x)) / sum``: the same
program as the max written out by hand. Lowering asks the update search for every
streaming obstacle of the graph, and an exp that reads ``x - max(x)`` is the exp of an
exact softmax by structure, whoever wrote it.
"""

import pytest

torch = pytest.importorskip("torch")

import cleave.lang as cl
from cleave.frontend import trace


def softmax(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1)


def softmax_by_hand(x: cl.Tensor["R", "C"]):
    e = cl.exp(x - cl.max(x, -1, keepdim=True))
    return e / cl.sum(e, -1, keepdim=True)


def _kinds(program):
    return [r.kind for r in program.ops]


def test_softmax_records_the_max():
    program = trace(softmax, {"R": 4, "C": 32}, cl.f32)
    assert _kinds(program) == ["input", "reduce_max", "sub", "exp", "reduce_sum", "div"]
    assert program.ops[1].attr("dim") == 1 and program.ops[4].attr("dim") == 1
    # exactly what the max written out by hand records: one program, one fingerprint
    by_hand = trace(softmax_by_hand, {"R": 4, "C": 32}, cl.f32)
    assert _kinds(by_hand) == _kinds(program)
    assert by_hand.fingerprint() == program.fingerprint()
    x = torch.randn(4, 32) * 30
    torch.testing.assert_close(program.reference(x)[0], torch.softmax(x, -1))


def test_the_eager_softmax_is_the_max_written_out():
    x = torch.randn(4, 32) * 30
    eager = cl.softmax(x, -1)
    e = torch.exp(x - x.amax(-1, keepdim=True))
    assert torch.equal(eager, e / e.sum(-1, keepdim=True))
    torch.testing.assert_close(eager, torch.softmax(x, -1))


def test_the_kernel_key_is_made_of_the_program_and_its_configs():
    """``pipeline.compile._kernel_key``: the validation schema, the program's fingerprint,
    its input and output names, the budget and the configs -- nothing else."""
    import dataclasses

    from cleave.pipeline import cache
    from cleave.pipeline.compile import _kernel_key, resolve_configs

    program = trace(softmax_by_hand, {"R": 4, "C": 32}, cl.f32)
    so, sc, ev = resolve_configs(program)

    def fields(cfg, *drop):
        return {k: v for k, v in dataclasses.asdict(cfg).items() if k not in drop}

    names = [[s.name for s in program.inputs], [o.name for o in program.outputs]]
    assert _kernel_key(program, so, sc, ev) == cache.digest(
        "kernel", cache.VALIDATION, program.fingerprint(), names, so.resolve_max_ops(program),
        fields(so, "threads", "cache", "max_ops"), fields(sc),
        fields(ev, "max_parallel", "timeout_s"))


def test_lowering_asks_for_the_update_rules(monkeypatch):
    """Every solo graph that is lowered goes through ``_derive_updates``, a softmax of
    ``cl.softmax`` and one written by hand alike."""
    from cleave.pipeline import plans
    from cleave.pipeline.superopt import GraphSet

    def derive(*args, **kwargs):
        raise AssertionError("update rules were asked for")

    monkeypatch.setattr(plans, "_derive_updates", derive)
    for fn in (softmax, softmax_by_hand):
        (graph,) = GraphSet.identity(trace(fn, {"R": 16, "C": 64}, cl.f16))
        with pytest.raises(AssertionError, match="asked for"):
            plans.lower(graph)


# --------------------------------------------------------- which exp is an exact softmax


def attention_over_queries_by_hand(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                                   v: cl.Tensor["H", "SKV", "D"]):
    """Normalized over the queries (``dim=-2``), the max written out."""
    s = (q @ k.mT) / q.shape[-1] ** 0.5
    e = cl.exp(s - cl.max(s, -2, keepdim=True))
    return (e / cl.sum(e, -2, keepdim=True)) @ v


def softmax_fixed_shift(x: cl.Tensor["R", "C"]):
    e = cl.exp(x + -4.0)
    return e / cl.sum(e, -1, keepdim=True)


def softmax_of_another_max(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]):
    e = cl.exp(x - cl.max(y, -1, keepdim=True))  # not the max of what it is subtracted from
    return e / cl.sum(e, -1, keepdim=True)


def test_an_explicit_max_is_an_exact_softmax():
    """``softmax_axis``: the exp reads ``x - max(x)``. The axis decides which operand of the
    score matmul is the key, a logit input is found along it, and the reference publishing
    holds the graph to every adversarial case."""
    from cleave.ir.ops import Exp, Matmul
    from cleave.pipeline.build import _reference_fn
    from cleave.runtime.inputs import ADVERSARIAL, key_operand, logit_inputs, softmax_axis

    from ._attention import flash_by_hand, lowered, node_of

    shapes = dict(H=4, SQ=8, SKV=32, D=16)
    ir = lowered(flash_by_hand, shapes)
    rows = lowered(softmax_by_hand, {"R": 16, "C": 64})
    queries = lowered(attention_over_queries_by_hand, shapes)
    held = []

    def publish(cache_dir, graph, dtype, distributions):
        held.append(distributions)

    reference = _reference_fn(("normal",), trace(flash_by_hand, shapes, cl.f16), publish)

    assert softmax_axis(node_of(ir, Exp)) == -1
    assert softmax_axis(node_of(queries, Exp)) == -2
    ((placeholder, axis, sign),) = logit_inputs(rows)
    assert placeholder.is_placeholder() and (axis, sign) == (1, 1.0)
    # the softmax runs over the first operand's rows: the two operands change places
    assert key_operand(queries, node_of(queries, Matmul)) == 0
    assert key_operand(ir, node_of(ir, Matmul)) == 1
    reference("unused", ir, "float16")
    assert held == [("normal",) + ADVERSARIAL]
    assert reference.held[id(ir)] == (("normal",) + ADVERSARIAL, "")


def test_only_x_minus_its_own_max_is_an_exact_softmax_by_structure():
    from cleave.ir.ops import Exp
    from cleave.runtime.inputs import logit_inputs, softmax_axis

    from ._attention import lowered, node_of

    for fn in (softmax_fixed_shift, softmax_of_another_max):
        ir = lowered(fn, {"R": 16, "C": 64})
        assert softmax_axis(node_of(ir, Exp)) is None and logit_inputs(ir) == []
    for fn in (softmax_by_hand, softmax):
        ir = lowered(fn, {"R": 16, "C": 64})
        assert softmax_axis(node_of(ir, Exp)) == -1
