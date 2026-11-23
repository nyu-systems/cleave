"""GPU: a per-output search, merged, scheduled, built and called end to end.

``per_output_pair``: ``e = exp(x @ w * c)``; outputs ``a = e @ v`` and ``b = rowsum(e)``; ``y``
is declared ``equals= x @ w``. Searched per output (each part reads x, w and keeps y), the merged
graph of the parts' first graphs computes ``e`` once for both outputs, reads no ``y`` (the kernel
takes it and ignores it), and validates on derived inputs.
"""

import pytest

import cleave.lang as cl
from cleave import pipeline as P
from cleave.frontend.program import Program
from cleave.runtime import validate
from tests._tolerances import tolerance_for

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

TOLERANCE = tolerance_for("per_output_pair")
EVAL = P.EvalConfig(warmup=2, iters=20, max_parallel=4, tolerance=TOLERANCE)
SUPEROPT = P.SuperoptConfig(threads=4)


def pair() -> Program:
    p = Program("per_output_pair", cl.f16)
    m, k, n, d = p.dims(M=128, K=64, N=256, D=64)
    x, w, v = p.input("x", (m, k)), p.input("w", (k, n)), p.input("v", (n, d))
    p.input("y", (m, n), equals=lambda x, w: x @ w)
    e = cl.exp(x @ w * 0.125)
    p.output(e @ v, "a")
    p.output(cl.sum(e, -1, keepdim=True), "b")
    return p


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    return tmp_path


def test_merged_graphs_build_and_validate(cache_root):
    graphs = P.superoptimize(pair(), SUPEROPT)
    assert len(graphs.parts) == 2 and len(graphs) == len(graphs.parts[0]) * len(graphs.parts[1])
    (shared,) = [g for g in graphs if g.parts == (0, 0)]
    assert shared.ops.count("exp") == 1
    built = P.build(P.schedule(graphs, P.ScheduleConfig(topk=8)), config=EVAL)
    ok = [r for r in built.results if r.run.ok]
    assert any(r.plan.graph is shared for r in ok), [str(r.run.error)[:200] for r in built.results]
    for r in ok:
        assert all(m.rel <= TOLERANCE.rtol for m in r.run.measured)


def test_the_compiled_kernel_returns_the_program_outputs(cache_root):
    program = pair()
    kernel = P.compile(program, superopt=SUPEROPT, schedule=P.ScheduleConfig(topk=8), eval=EVAL)
    inputs = program.random_inputs(seed=1)
    out = kernel(*inputs)
    ref = program.reference(*inputs)
    assert [tuple(o.shape) for o in out] == [tuple(r.shape) for r in ref]
    validate.compare_outputs(list(out), ref, TOLERANCE)
