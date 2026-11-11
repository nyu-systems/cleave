"""GPU: compile() end to end on a small matmul, and split-K partial/combine plans."""

import time

import pytest

import cleave.lang as cl
from cleave.frontend import trace
from cleave import pipeline as P
from tests.frontend import _programs as programs

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

SHAPES = dict(M=128, N=128, K=256)
FAST = dict(schedule=P.ScheduleConfig(topk=3), eval=P.EvalConfig(warmup=2, iters=20, max_parallel=4))


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    return tmp_path


def _close(out, ref):
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=5e-2)


def test_compile_matmul_end_to_end(cache_root, tmp_path):
    kernel = P.compile(programs.matmul, shapes=SHAPES, dtype=cl.f16, split=False, **FAST)
    report = kernel.report
    assert kernel.variant == "solo" and kernel.latency_us > 0 and kernel.unrunnable is None
    assert len(report.correct) >= 1 and set(report.timings) >= {"superoptimize", "schedule", "build"}
    assert "def main_kernel" in kernel.source or "main_kernel" in kernel.source
    assert "superoptimize" in report.summary() and "benchmark" in report.summary()

    prog = trace(programs.matmul, SHAPES, cl.f16)
    x, w = prog.random_inputs(seed=1)
    ref = prog.reference(x, w)[0]
    _close(kernel(x, w), ref)
    _close(kernel(w=w, x=x), ref)
    with pytest.raises(ValueError):
        kernel(w, x)  # wrong shapes are rejected, not silently transposed

    loaded = P.load(kernel.save(tmp_path / "saved"))
    assert loaded.latency_us == kernel.latency_us and loaded.input_names == ["x", "w"]
    _close(loaded(x, w), ref)

    t0 = time.perf_counter()
    again = P.compile(programs.matmul, shapes=SHAPES, dtype=cl.f16, split=False, **FAST)
    assert again.meta["cache"] == "hit" and time.perf_counter() - t0 < 30
    _close(again(x, w), ref)


def test_split_k_schedule_and_chained_kernel(cache_root):
    prog = trace(programs.matmul, SHAPES, cl.f16)
    graphs = P.superoptimize(prog, P.SuperoptConfig(split=("K",)))
    assert any(g.is_split for g in graphs) and any(not g.is_split for g in graphs)
    assert graphs.checkpoint and str(cache_root) in graphs.checkpoint

    plans = P.schedule(graphs, P.ScheduleConfig(topk=3, split_k=(4, 8)))
    roles = {(p.variant, p.role) for p in plans}
    assert {("solo", "main"), ("split-k=4", "partial"), ("split-k=4", "combine"),
            ("split-k=8", "partial"), ("split-k=8", "combine")} <= roles
    assert all(p.inputs is not None for p in plans if p.role != "combine")
    assert not plans.refused, plans.summary()
    cached = P.superoptimize(prog, P.SuperoptConfig(split=("K",)))
    assert cached.from_cache and len(cached) == len(graphs)

    # Build and time only split-k=4, then run partial -> combine as one Kernel.
    split4 = plans.filter(lambda p: p.variant == "split-k=4")
    report = P.evaluate(split4, P.EvalConfig(warmup=2, iters=20, max_parallel=4))
    kernel = report.best("split-k=4")
    assert [s.role for s in kernel.stages] == ["partial", "combine"]
    parts = report.variants()["split-k=4"].members
    assert kernel.latency_us == pytest.approx(sum(m.latency_us for m in parts))
    x, w = prog.random_inputs(seed=2)
    _close(kernel(x, w), prog.reference(x, w)[0])


def test_session_scopes_compile(tmp_path, monkeypatch):
    import cleave

    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path / "env"))
    with cleave.Session(cache_root=tmp_path / "session", max_parallel=2):
        kernel = P.compile(programs.matmul, shapes=SHAPES, dtype=cl.f16, split=False, **FAST)
    assert str(tmp_path / "session") in kernel.meta["saved"]
    assert str(tmp_path / "session") in kernel.report.plans.graphs.checkpoint
    (built,) = kernel.kernels
    assert built.source == kernel.source and built.entry == "main_kernel"
    assert len(built.inputs) == 2 and len(built.outputs) == 1
    assert [p.role for p in built.params][-1] == "output" and built.plan is kernel.plan
    best = min(kernel.report.timed, key=lambda r: r.latency_ms)
    assert best.kernel is built and best.source == built.source
    x, w = trace(programs.matmul, SHAPES, cl.f16).random_inputs(seed=3)
    _close(kernel(x, w), x.float() @ w.float())
