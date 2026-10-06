import multiprocessing as mp
import time
from types import SimpleNamespace

import pytest

import cleave.codegen as codegen_module
import cleave.passes as passes_module
import cleave.schedule as schedule_module
import cleave.util.runtime as runtime


@pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="requires fork multiprocessing support",
)
def test_compile_and_run_candidates_parallel_times_out_and_reports_progress(
    monkeypatch, capsys
):
    class FakeCodeGen:
        def gen(self, graph):
            return graph.code

    def identity(graph, **_kwargs):
        return graph

    def fake_compile_and_run(code_str, *, out_idx=None, reference_cache_dir=None):
        del out_idx, reference_cache_dir
        if code_str == "slow":
            time.sleep(1.0)
            return "cuda_slow"
        time.sleep(0.05)
        return f"cuda_{code_str}"

    monkeypatch.setattr(
        runtime, "_compile_and_run_mp_context", lambda: mp.get_context("fork")
    )
    monkeypatch.setattr(runtime, "compile_and_run", fake_compile_and_run)
    # The parent normally builds the reference from the graph; stub it out so the
    # fake SimpleNamespace candidates never touch the executor or CUDA.
    monkeypatch.setattr(
        runtime, "publish_reference", lambda cache_dir, graph, dtype: ([-1], 0)
    )
    monkeypatch.setattr(codegen_module, "TilelangCodeGen", FakeCodeGen)
    monkeypatch.setattr(schedule_module, "apply_tiling", lambda cand: cand)
    monkeypatch.setattr(passes_module, "annotate_args", identity)
    monkeypatch.setattr(passes_module, "annotate_tinfo", identity)
    monkeypatch.setattr(passes_module, "annotate_tc_layout", identity)
    monkeypatch.setattr(
        passes_module,
        "annotate_fragment_layout_conflicts",
        identity,
    )
    monkeypatch.setenv("COMPILATION_PROGRESS_LOG", "true")
    monkeypatch.setenv("CLEAVE_COMPILE_AND_RUN_TIMEOUT_SECS", "0.3")
    monkeypatch.setenv("CLEAVE_COMPILE_PROGRESS_HEARTBEAT_SECS", "0.1")

    candidates = [
        SimpleNamespace(code="fast_a", rstep_map={"id": 1}, output_tile=(1,), flat_graph=None),
        SimpleNamespace(code="slow", rstep_map={"id": 2}, output_tile=(2,), flat_graph=None),
        SimpleNamespace(code="fast_b", rstep_map={"id": 3}, output_tile=(3,), flat_graph=None),
    ]
    success_info = []
    error_graphs = []
    errors = []

    runtime.compile_and_run_candidates_parallel(
        candidates,
        dtype="float16",
        max_parallel=2,
        success_info=success_info,
        error_graphs=error_graphs,
        errors=errors,
    )

    assert [entry[3] for entry in success_info] == ["fast_a", "fast_b"]
    assert [graph[0].code for graph in error_graphs] == ["slow"]
    assert len(errors) == 1
    assert isinstance(errors[0], TimeoutError)
    assert "candidate 2/3" in str(errors[0])

    output = capsys.readouterr().out
    assert "[cleave]" in output
    assert "2 running. Slowest: candidate 2/3" in output
    assert "[cleave] timed out candidate 2/3" in output


@pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="requires fork multiprocessing support",
)
def test_compile_and_run_candidates_parallel_progress_log_disabled_by_default(
    monkeypatch, capsys
):
    class FakeCodeGen:
        def gen(self, graph):
            return graph.code

    def identity(graph, **_kwargs):
        return graph

    def fake_compile_and_run(code_str, *, out_idx=None, reference_cache_dir=None):
        del out_idx, reference_cache_dir
        time.sleep(0.05)
        return f"cuda_{code_str}"

    monkeypatch.setattr(
        runtime, "_compile_and_run_mp_context", lambda: mp.get_context("fork")
    )
    monkeypatch.setattr(runtime, "compile_and_run", fake_compile_and_run)
    monkeypatch.setattr(
        runtime, "publish_reference", lambda cache_dir, graph, dtype: ([-1], 0)
    )
    monkeypatch.setattr(codegen_module, "TilelangCodeGen", FakeCodeGen)
    monkeypatch.setattr(schedule_module, "apply_tiling", lambda cand: cand)
    monkeypatch.setattr(passes_module, "annotate_args", identity)
    monkeypatch.setattr(passes_module, "annotate_tinfo", identity)
    monkeypatch.setattr(passes_module, "annotate_tc_layout", identity)
    monkeypatch.setattr(
        passes_module,
        "annotate_fragment_layout_conflicts",
        identity,
    )
    monkeypatch.delenv("COMPILATION_PROGRESS_LOG", raising=False)
    monkeypatch.setenv("CLEAVE_COMPILE_PROGRESS_HEARTBEAT_SECS", "0.01")

    success_info = []
    error_graphs = []
    errors = []
    candidates = [
        SimpleNamespace(code="fast_a", rstep_map={"id": 1}, output_tile=(1,), flat_graph=None),
        SimpleNamespace(code="fast_b", rstep_map={"id": 2}, output_tile=(2,), flat_graph=None),
    ]

    runtime.compile_and_run_candidates_parallel(
        candidates,
        dtype="float16",
        max_parallel=2,
        success_info=success_info,
        error_graphs=error_graphs,
        errors=errors,
    )

    assert [entry[3] for entry in success_info] == ["fast_a", "fast_b"]
    assert error_graphs == []
    assert errors == []

    output = capsys.readouterr().out
    assert output == "Compiling and running 2 scheduled graphs:\n"
