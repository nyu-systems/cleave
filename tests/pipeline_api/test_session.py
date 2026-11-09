"""CPU-only: Session scoping (cache root, parallelism, target) and BuiltKernel persistence."""

import importlib
from pathlib import Path

import pytest

import cleave
from cleave.core import Settings, current_session, current_target
from cleave.core.session import Session, active_session
from cleave.pipeline import cache
from cleave.pipeline.build import BuiltKernel, KernelParam, pass_configs_of
from cleave.pipeline.kernel import Kernel, Param, Stage, load

build_mod = importlib.import_module("cleave.pipeline.build")  # the package exports build()


def test_session_is_exported_lazily():
    assert cleave.Session is Session


def test_default_session_reads_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    assert active_session() is None
    s = current_session()
    assert s.cache_root == tmp_path and s.max_parallel is None
    assert s.target is current_target()
    assert cache.root() == tmp_path / "persistent"


def test_session_scopes_cache_root_and_nests(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path / "env"))
    with Session(cache_root=tmp_path / "outer", max_parallel=4) as outer:
        assert current_session() is outer
        assert cache.root() == tmp_path / "outer" / "persistent"
        with Session(cache_root=tmp_path / "inner"):
            assert cache.root() == tmp_path / "inner" / "persistent"
        assert current_session() is outer
    assert active_session() is None
    assert cache.root() == tmp_path / "env" / "persistent"


def test_session_settings_and_validation(tmp_path):
    s = Session(settings=Settings(compile_timeout_secs=7.0, cache_root=tmp_path))
    assert s.settings.compile_timeout_secs == 7.0 and s.cache_root == tmp_path
    with pytest.raises(ValueError):
        Session(max_parallel=0)


def test_session_target_reaches_current_target():
    default = current_target()
    with Session(target="cuda") as s:
        assert current_target() is s.target and s.target is not default
        assert current_target("other") is not s.target
    assert current_target() is default


def test_build_reads_session_parallelism(monkeypatch):
    seen = {}

    def fake_run_candidates(cands, dtype, **kw):
        seen.update(kw)
        return []

    from cleave.runtime import evaluate as runtime

    monkeypatch.setattr(runtime, "run_candidates", fake_run_candidates)
    build_mod.build([])
    assert seen["max_parallel"] == 8
    with Session(max_parallel=3, settings=Settings(compile_timeout_secs=None)):
        build_mod.build([])
        assert seen["max_parallel"] == 3 and seen["timeout_s"] == 0
        build_mod.build([], workers=2)
        assert seen["max_parallel"] == 2


SRC = 'PASS_CONFIGS = {\n "tl.disable_tma_lower": True,\n}\n\ndef main_kernel():\n    pass\n'


def test_pass_configs_of_reads_the_declared_literal():
    assert pass_configs_of(SRC) == {"tl.disable_tma_lower": True}
    assert pass_configs_of("def main_kernel():\n    pass\n") == {}


def test_kernel_save_load_keeps_built_kernel(tmp_path):
    params = (KernelParam("input_1", "float16", (4, 8), "input"),
              KernelParam("out", "float16", (4,), "output"))
    built = BuiltKernel(SRC, params=params, out_idx=(-1,), pass_configs=pass_configs_of(SRC))
    stage = Stage("main", SRC, [-1], [Param("input_1", "input", 0, (4, 8))], "g0", 1.5, built)
    k = Kernel([stage], name="k", input_names=["x"], input_shapes=[(4, 8)])
    assert k.source == SRC and k.kernels == [built]
    loaded = load(k.save(tmp_path / "k"))
    (lb,) = loaded.kernels
    assert lb.source == SRC and lb.params == params and lb.out_idx == (-1,)
    assert lb.pass_configs == {"tl.disable_tma_lower": True} and lb.inputs == [params[0]]
    assert loaded.stages[0].params == stage.params and loaded.stages[0].latency_us == 1.5


def test_load_without_built_kernel_metadata(tmp_path):
    stage = Stage("main", SRC, [-1], [], "g0")
    k = Kernel([stage], name="k", input_names=[], input_shapes=[])
    loaded = load(k.save(Path(tmp_path) / "k"))
    assert loaded.kernels == [None] and loaded.source == SRC


def test_session_is_per_thread_and_reentrant(tmp_path):
    import threading

    s = Session(cache_root=tmp_path)
    seen = []
    with s:
        t = threading.Thread(target=lambda: seen.append(active_session()))
        t.start()
        t.join()
        with s:
            assert active_session() is s
        assert active_session() is s
    assert seen == [None] and active_session() is None


def test_closed_worker_pool_refuses_work():
    from cleave.runtime.workers import WorkerPool

    pool = WorkerPool()
    pool.close()
    with pytest.raises(RuntimeError):
        pool.acquire()
