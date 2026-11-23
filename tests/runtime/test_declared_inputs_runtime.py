"""Declared inputs at validation time (CPU): the ``derive`` hook of ``make_inputs``.

A kernel is validated on inputs drawn from its own lowered graph (``make_inputs``), not on
``Program.random_inputs``; the inputs a Program declares (``equals=`` / ``sample=``) are
recomputed from the drawn ones after any distribution is applied, by
``Program.derive_inputs``. ``tests/runtime/test_input_distributions.py`` pins the stream of
graphs without declarations, which this hook leaves alone (it is None for them).
"""

import pytest

torch = pytest.importorskip("torch")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import SpecError
from cleave.frontend import Program
from cleave.pipeline.build import derive_for
from cleave.runtime import inputs as rt_inputs
from cleave.runtime import reference as rt

from tests.frontend import _programs as programs

DTYPE = "float16"


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    """program name -> (Program, its first solo plan): one search each, on the CPU."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CLEAVE_CACHE_ROOT", str(tmp_path_factory.mktemp("cache")))
        out = {}
        for declared in (True, False):
            program = programs.lse_attention_builder(declared=declared)
            graphs = P.superoptimize(program, P.SuperoptConfig(split=()))
            plan = P.schedule(graphs, P.ScheduleConfig(topk=1))[0]
            assert plan.inputs is not None
            out[declared] = (program, plan)
        return out


def _by_name(plan, tensors):
    """Program input name -> tensor, through the plan's placeholder binding."""
    graph = plan.tile.flat_graph
    position = {ph.name: i for i, ph in enumerate(rt.placeholder_order(graph))}
    return {s.name: tensors[position[ph]] for s, ph in zip(plan.program.inputs, plan.inputs)}


def _lse(x, w):
    scores = x.float() @ w.float()
    return torch.logsumexp(scores, -1, keepdim=True).expand_as(scores)


def test_a_program_without_declarations_has_no_hook(plans):
    program, plan = plans[False]
    graph = plan.tile.flat_graph
    assert rt.declared_inputs_hook(program, plan.inputs, graph) is None
    assert derive_for(plan, graph) is None
    assert derive_for(None, graph) is None  # a graph no plan claims


def test_declared_inputs_are_recomputed_from_the_drawn_ones(plans):
    program, plan = plans[True]
    graph = plan.tile.flat_graph
    derive = derive_for(plan, graph)
    assert derive is not None
    drawn = rt.make_inputs(graph, DTYPE, device="cpu")
    derived = rt.make_inputs(graph, DTYPE, device="cpu", derive=derive)
    a, b = _by_name(plan, drawn), _by_name(plan, derived)
    # every input is still drawn (same stream); only the declared one is replaced
    for name in ("x", "w", "v"):
        assert torch.equal(a[name], b[name]), name
    assert a["t"].dtype == b["t"].dtype == torch.float32  # the fp32 input stays fp32
    assert not torch.equal(a["t"], b["t"])
    assert b["t"].shape == a["t"].shape and b["t"].is_contiguous()
    torch.testing.assert_close(b["t"], _lse(b["x"], b["w"]), rtol=1e-5, atol=1e-5)
    # ... and exactly what the Program derives from the same values
    want = program.derive_inputs([b["x"], b["w"], None, b["v"]])[2]
    assert torch.equal(b["t"], want)


def test_the_reference_is_computed_from_the_derived_inputs(plans):
    _, plan = plans[True]
    graph = plan.tile.flat_graph
    derive = derive_for(plan, graph)
    _, noise, _ = rt.build_inputs_and_reference(graph, DTYPE, device="cpu")
    _, real, _ = rt.build_inputs_and_reference(graph, DTYPE, device="cpu", derive=derive)
    # drawn as noise, exp(x @ w - t) overflows and the second matmul makes NaN of it ...
    assert torch.isnan(noise[0]).any()
    # ... while a real log-sum-exp keeps every probability <= 1 and the output finite
    assert torch.isfinite(real[0]).all()
    inputs = _by_name(plan, rt.make_inputs(graph, DTYPE, device="cpu", derive=derive))
    probabilities = torch.exp(inputs["x"].float() @ inputs["w"].float() - inputs["t"])
    assert probabilities.max() <= 1 + 1e-5
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(probabilities.shape[0]))


def test_declared_inputs_are_derived_after_the_distribution(plans, monkeypatch):
    _, plan = plans[True]
    graph = plan.tile.flat_graph
    derive = derive_for(plan, graph)
    seen = []

    def rewrite(graph_, inputs, distribution):  # stands in for an adversarial generator
        seen.append(distribution)
        return tuple(x * 2 for x in inputs)

    monkeypatch.setattr(rt_inputs, "apply_distribution", rewrite)
    normal = _by_name(plan, rt.make_inputs(graph, DTYPE, device="cpu"))
    got = _by_name(plan, rt.make_inputs(graph, DTYPE, device="cpu", derive=derive,
                                        distribution="large_logits"))
    assert seen == ["large_logits"]
    assert torch.equal(got["x"], normal["x"] * 2)
    # t is the log-sum-exp of the rewritten x and w, not of the normal ones
    torch.testing.assert_close(got["t"], _lse(got["x"], got["w"]), rtol=1e-5, atol=1e-5)


def test_the_hook_refuses_what_it_cannot_bind(plans):
    program, plan = plans[True]
    graph = plan.tile.flat_graph
    with pytest.raises(SpecError, match="could not be bound"):
        rt.declared_inputs_hook(program, None, graph)
    with pytest.raises(SpecError, match="one kernel input per Program input"):
        rt.declared_inputs_hook(program, plan.inputs[:-1], graph)
    with pytest.raises(SpecError, match="one kernel input per Program input"):
        rt.declared_inputs_hook(program, (plan.inputs[0],) * len(plan.inputs), graph)
    with pytest.raises(SpecError, match="one kernel input per Program input"):
        rt.declared_inputs_hook(program, (*plan.inputs[:-1], "not_a_placeholder"), graph)


def test_a_derived_dtype_must_match_its_placeholder(plans):
    """An fp32 declared input is never fed to an fp16 kernel parameter (or the reverse)."""
    _, plan = plans[False]  # its t placeholder is fp32
    graph = plan.tile.flat_graph
    p = Program("lse_attention_f16_t")
    M, K, N, D = p.dims(M=64, K=64, N=128, D=32)
    p.input("x", (M, K)), p.input("w", (K, N))
    p.input("t", (M, N), cl.f16, sample=lambda x, w: x @ w)
    p.input("v", (N, D))
    derive = rt.declared_inputs_hook(p, plan.inputs, graph)
    with pytest.raises(SpecError, match="'t' is declared torch.float16 but its kernel"):
        rt.make_inputs(graph, DTYPE, device="cpu", derive=derive)


def test_a_combine_stage_draws_all_its_inputs(plans):
    _, plan = plans[True]

    class Combine:
        role, program, inputs = "combine", plan.program, None

    assert derive_for(Combine, plan.tile.flat_graph) is None


def test_build_publishes_the_reference_with_the_plans_derivation(plans):
    """``build``'s reference_fn: ``publish_reference`` exactly as before (``derive=None``)
    for a program without declarations, with the plan's hook for one with them."""
    from cleave.pipeline.build import _publish_reference

    for declared in (False, True):
        _, plan = plans[declared]
        graph = plan.tile.flat_graph
        calls = []

        def publish(*args, **kwargs):
            calls.append((args, kwargs))
            return 0, 0

        assert _publish_reference(publish, ("normal",), {id(graph): plan}, "dir", graph,
                                  DTYPE) == (0, 0)
        ((args, kwargs),) = calls
        assert args == ("dir", graph, DTYPE) and kwargs["distributions"] == ("normal",)
        assert (kwargs["derive"] is not None) == declared
