"""The one switch of the searched update rules (``CLEAVE_SEARCHED_UPDATES``), CPU only.

Off by default: ``cl.softmax`` records the request of the hand-derived pass and nothing of
the searched path runs. On (the ``searched_updates`` fixture, or the environment variable),
``cl.softmax`` records the max it is defined with.
"""

import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.frontend import trace
from cleave.ir import update_rule
from cleave.ir.update_rule import SWITCH_ENV, searched_updates, use_searched_updates


def softmax(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1)


def softmax_by_hand(x: cl.Tensor["R", "C"]):
    e = cl.exp(x - cl.max(x, -1, keepdim=True))
    return e / cl.sum(e, -1, keepdim=True)


def _kinds(program):
    return [r.kind for r in program.ops]


def test_the_switch_is_off_by_default():
    assert os.environ.get(SWITCH_ENV, "") in ("", "0")
    assert searched_updates() is False


def test_use_searched_updates_restores_the_setting():
    before = os.environ.get(SWITCH_ENV)
    with use_searched_updates():
        assert searched_updates() is True and os.environ[SWITCH_ENV] == "1"
        with use_searched_updates(False):
            assert searched_updates() is False and os.environ[SWITCH_ENV] == "0"
        assert searched_updates() is True and os.environ[SWITCH_ENV] == "1"
    assert searched_updates() is False and os.environ.get(SWITCH_ENV) == before
    with pytest.raises(RuntimeError, match="inside"):
        with use_searched_updates():
            raise RuntimeError("inside")
    assert searched_updates() is False and os.environ.get(SWITCH_ENV) == before


def test_the_fixture_switches_it_on(searched_updates):
    assert update_rule.searched_updates() is True


@pytest.mark.parametrize("value, on", [("1", True), ("yes", True), ("0", False), ("", False),
                                       (None, False)])
def test_the_environment_variable_sets_it_at_import(value, on):
    """A process started with the variable (a demo, a worker started inside
    ``use_searched_updates``) begins with that setting."""
    env = {k: v for k, v in os.environ.items() if k != SWITCH_ENV}
    env["CUDA_VISIBLE_DEVICES"] = ""
    if value is not None:
        env[SWITCH_ENV] = value
    out = subprocess.run(
        [sys.executable, "-c",
         "from cleave.ir.update_rule import searched_updates; print(searched_updates())"],
        env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == str(on)


def test_without_the_switch_softmax_records_the_request():
    program = trace(softmax, {"R": 4, "C": 32}, cl.f32)
    assert _kinds(program) == ["input", "exp", "reduce_sum", "div"]
    assert program.stabilize_requests == ((program.ops[0].out, 1),)


def test_with_the_switch_softmax_records_the_max(searched_updates):
    program = trace(softmax, {"R": 4, "C": 32}, cl.f32)
    assert _kinds(program) == ["input", "reduce_max", "sub", "exp", "reduce_sum", "div"]
    assert not program.stabilize_requests
    assert program.ops[1].attr("dim") == 1 and program.ops[3].attr("stabilize") is None
    # exactly what the max written out by hand records: one program, one fingerprint
    by_hand = trace(softmax_by_hand, {"R": 4, "C": 32}, cl.f32)
    assert _kinds(by_hand) == _kinds(program)
    assert by_hand.fingerprint() == program.fingerprint()
    x = torch.randn(4, 32) * 30
    torch.testing.assert_close(program.reference(x)[0], torch.softmax(x, -1))


def test_the_eager_softmax_does_not_depend_on_the_switch():
    x = torch.randn(4, 32) * 30
    off = cl.softmax(x, -1)
    with use_searched_updates():
        on = cl.softmax(x, -1)
    assert torch.equal(off, on)
    torch.testing.assert_close(off, torch.softmax(x, -1))


def test_a_saved_kernel_is_not_shared_between_the_two_settings():
    """One program -- the max written out by hand -- builds different kernels under the two
    settings and is held to different input cases, so the kernel cache keys differ; the
    default setting keeps the key it had."""
    from cleave.pipeline import cache
    from cleave.pipeline.compile import _kernel_key, resolve_configs

    program = trace(softmax_by_hand, {"R": 4, "C": 32}, cl.f32)
    so, sc, ev = resolve_configs(program)
    off = _kernel_key(program, so, sc, ev)
    with use_searched_updates():
        on = _kernel_key(program, so, sc, ev)
    assert off != on and off == _kernel_key(program, so, sc, ev)

    import dataclasses

    def fields(cfg, *drop):
        return {k: v for k, v in dataclasses.asdict(cfg).items() if k not in drop}

    names = [[s.name for s in program.inputs], [o.name for o in program.outputs]]
    assert off == cache.digest("kernel", cache.VALIDATION, program.fingerprint(), names,
                               so.resolve_max_ops(program),
                               fields(so, "threads", "cache", "max_ops"), fields(sc),
                               fields(ev, "max_parallel", "timeout_s"))


def test_without_the_switch_no_update_rule_is_derived(monkeypatch):
    """Lowering a graph with an explicit max does not reach the engine while the switch is
    off: no record, no refusal list, the pin and its text as they were."""
    from cleave.pipeline import plans
    from cleave.pipeline.superopt import GraphSet
    from cleave.schedule.streaming import streaming_accept

    def derive(*args, **kwargs):
        raise AssertionError("update rules were asked for")

    monkeypatch.setattr(plans, "_derive_updates", derive)
    (graph,) = GraphSet.identity(trace(softmax_by_hand, {"R": 16, "C": 64}, cl.f16))
    (ir,) = plans.lower(graph)
    assert not hasattr(ir, "streaming_refusals") and not hasattr(ir, "derivations")
    assert not [n for n in ir.nodes if getattr(n, "streaming", None) is not None]
    assert streaming_accept(ir) is None  # no filter either: the tilings are the policy's
    with use_searched_updates():  # ... and with it on, lowering asks for the rules
        with pytest.raises(AssertionError, match="asked for"):
            plans.lower(graph)


# ------------------------------------------------- which exp is an exact softmax, and when


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


def test_an_explicit_max_is_an_exact_softmax_only_with_the_switch():
    """``softmax_axis``: the record of the hand-derived pass always; the structural fact
    (the exp reads ``x - max(x)``) only where update rules are searched. With the switch
    off a max written by hand makes no claim, as before."""
    from cleave.ir.ops import Exp, Matmul
    from cleave.pipeline.build import _reference_fn
    from cleave.runtime.inputs import ADVERSARIAL, key_operand, logit_inputs, softmax_axis

    from ._attention import flash_by_hand, lowered, node_of

    shapes = dict(H=4, SQ=8, SKV=32, D=16)
    ir = lowered(flash_by_hand, shapes)
    exp = node_of(ir, Exp)
    rows = lowered(softmax_by_hand, {"R": 16, "C": 64})
    queries = lowered(attention_over_queries_by_hand, shapes)
    held = []

    def publish(cache_dir, graph, dtype, distributions):
        held.append(distributions)

    reference = _reference_fn(("normal",), trace(flash_by_hand, shapes, cl.f16), publish)

    assert softmax_axis(exp) is None
    assert logit_inputs(rows) == []
    assert key_operand(queries, node_of(queries, Matmul)) == 1
    reference("unused", ir, "float16")
    assert held == [("normal",)] and reference.held[id(ir)] == (("normal",), "")

    with use_searched_updates():
        assert softmax_axis(exp) == -1
        assert softmax_axis(node_of(queries, Exp)) == -2
        ((placeholder, axis, sign),) = logit_inputs(rows)
        assert placeholder.is_placeholder() and (axis, sign) == (1, 1.0)
        # the softmax runs over the first operand's rows: the two operands change places
        assert key_operand(queries, node_of(queries, Matmul)) == 0
        assert key_operand(ir, node_of(ir, Matmul)) == 1
        reference("unused", ir, "float16")
        assert held[-1] == ("normal",) + ADVERSARIAL
        assert reference.held[id(ir)] == (("normal",) + ADVERSARIAL, "")
    assert softmax_axis(exp) is None and logit_inputs(rows) == []


def test_only_x_minus_its_own_max_is_an_exact_softmax_by_structure(searched_updates):
    from cleave.ir.ops import Exp
    from cleave.runtime.inputs import logit_inputs, softmax_axis

    from ._attention import lowered, node_of

    for fn in (softmax_fixed_shift, softmax_of_another_max):
        ir = lowered(fn, {"R": 16, "C": 64})
        assert softmax_axis(node_of(ir, Exp)) is None and logit_inputs(ir) == []
    ir = lowered(softmax_by_hand, {"R": 16, "C": 64})
    assert softmax_axis(node_of(ir, Exp)) == -1


def test_the_record_of_the_hand_derived_pass_counts_whatever_the_switch_says():
    from cleave.ir.ops import Exp
    from cleave.runtime.inputs import logit_inputs, softmax_axis

    from ._attention import lowered, node_of

    ir = lowered(softmax, {"R": 16, "C": 64})  # traced with the switch off: the request
    exp = node_of(ir, Exp)
    assert exp.stabilize is not None
    off = (softmax_axis(exp), [(p.name, a, s) for p, a, s in logit_inputs(ir)])
    with use_searched_updates():
        on = (softmax_axis(exp), [(p.name, a, s) for p, a, s in logit_inputs(ir)])
    assert off == on and off[0] == -1 and len(off[1]) == 1
