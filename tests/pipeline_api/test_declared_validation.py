"""GPU: kernels of a Program with a declared input are validated and timed on derived values.

``lse_attention`` is ``exp(x @ w - t) @ v`` with an fp32 ``t`` declared (``sample=``) as the
row log-sum-exp of ``x @ w``. Drawn as noise, ``t`` makes every kernel's output (and the
reference's) NaN; derived, the same kernels validate. ``t`` is a free input to the search, so
nothing about the search changes; the derivation happens where the inputs are drawn.
"""

import pytest

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import NonFiniteOutput
from cleave.pipeline.build import derive_for
from cleave.runtime import reference as rt
from cleave.runtime import validate
from cleave.runtime.evaluate import benchmark_results
from tests._tolerances import tolerance_for
from tests.frontend import _programs as programs

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

TOLERANCE = tolerance_for("lse_attention")
EVAL = P.EvalConfig(warmup=2, iters=20, max_parallel=2, tolerance=TOLERANCE)


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    return tmp_path


def _plans(declared):
    program = programs.lse_attention_builder(declared=declared)
    graphs = P.superoptimize(program, P.SuperoptConfig(split=()))
    return P.schedule(graphs, P.ScheduleConfig(topk=3))


def _by_name(plan, inputs_by_placeholder):
    return {s.name: inputs_by_placeholder[ph] for s, ph in zip(plan.program.inputs, plan.inputs)}


def _assert_real_lse(named):
    scores = named["x"].float() @ named["w"].float()
    want = torch.logsumexp(scores, -1, keepdim=True).expand_as(scores)
    assert named["t"].dtype == torch.float32
    torch.testing.assert_close(named["t"], want, rtol=1e-5, atol=1e-5)


def test_validation_and_timing_use_the_derived_inputs(cache_root, monkeypatch):
    published = []
    original = validate.publish_reference

    def recording(cache_dir, graph, dtype, distributions=("normal",), derive=None):
        result = original(cache_dir, graph, dtype, distributions=distributions, derive=derive)
        published.append((graph, derive, validate._load_reference(cache_dir)))
        return result

    monkeypatch.setattr(validate, "publish_reference", recording)
    plans = _plans(declared=True)
    built = P.build(plans, config=EVAL)

    (graph, derive, cases), = published  # one lowered graph, one reference for all its plans
    assert derive is not None and all(p.tile.flat_graph is graph for p in plans)
    # the one case is what every candidate was validated on and timed on (``_check_kernel``)
    ((distribution, inputs_by_name, reference),) = cases
    assert distribution == "normal" and torch.isfinite(reference[0]).all()
    _assert_real_lse(_by_name(plans[0], inputs_by_name))

    ok = [r for r in built.results if r.run.ok]
    assert ok, [r.run.error for r in built.results]
    assert all(r.run.latency_ms > 0 for r in ok)
    assert all(0 <= r.run.measured[0].rel <= TOLERANCE.rtol for r in ok)

    # The serial re-time draws its own inputs: derived the same way.
    seen = []

    def derive_and_record(g):
        hook = derive_for(next(p for p in plans if p.tile.flat_graph is g), g)

        def run(inputs):
            out = hook(inputs)
            seen.append(out)
            return out
        return run

    timed = benchmark_results([r.run for r in ok], "float16", warmup=2, iters=20,
                              derive_for=derive_and_record)
    assert all(t.ok and t.latency_ms > 0 for t in timed)
    (inputs,) = seen  # drawn once per graph
    names = [ph.name for ph in rt.placeholder_order(graph)]
    _assert_real_lse(_by_name(plans[0], dict(zip(names, inputs))))


def test_noise_where_an_input_is_declared_fails_every_kernel(cache_root):
    """The same program with ``t`` undeclared: its kernels are not wrong, its inputs are."""
    built = P.build(_plans(declared=False), config=EVAL)
    compiled = [r for r in built.results if r.run.stage != "compile"]
    assert compiled and not any(r.run.ok for r in compiled)
    for r in compiled:
        assert r.run.stage == "validate" and isinstance(r.run.error, NonFiniteOutput), r.run.error
        assert "NaN in the kernel output" in str(r.run.error)


def test_the_compiled_kernel_takes_its_fp32_input(cache_root):
    program = programs.lse_attention_builder()
    kernel = P.compile(program, split=False, schedule=P.ScheduleConfig(topk=3), eval=EVAL)
    x, w, t, v = program.random_inputs(seed=1)
    assert t.dtype == torch.float32 and x.dtype == torch.float16
    out = kernel(x, w, t, v)
    (ref,) = program.reference(x, w, t, v)
    assert out.dtype == ref.dtype == torch.float16 and torch.isfinite(out).all()
    validate.compare_outputs([out], [ref], TOLERANCE)
