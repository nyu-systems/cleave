"""CPU-only: configs, presets, the max_ops="auto" rule, cache keys, CLI parsing."""

import dataclasses

import pytest

import cleave.lang as cl
from cleave.core.errors import SpecError
from cleave.frontend import trace
from cleave.pipeline import (
    PRESETS,
    EvalConfig,
    ScheduleConfig,
    SuperoptConfig,
    auto_max_ops,
    get_preset,
    list_presets,
    resolve_configs,
)
from cleave.pipeline import cache
from tests.frontend import _programs as P


def test_presets_are_frozen_and_listed():
    assert set(list_presets()) == {"fast", "default", "exhaustive"}
    for p in PRESETS.values():
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.schedule.topk = 1
    assert get_preset("default").schedule == ScheduleConfig()
    assert get_preset("fast").schedule.topk < ScheduleConfig().topk
    with pytest.raises(SpecError):
        get_preset("nope")


def test_config_normalization_and_replace():
    assert SuperoptConfig(split="K").split == ("K",)
    assert SuperoptConfig(split=cl.Split("SKV", (16, 32))).split == ("SKV",)
    sc = ScheduleConfig(split_k=[8, "16"])
    assert sc.split_k == (8, 16) and sc.replace(topk=5).topk == 5 and sc.topk == 1000
    with pytest.raises(SpecError):
        ScheduleConfig(fusion="maybe")
    with pytest.raises(SpecError):
        SuperoptConfig(max_ops=0)
    assert EvalConfig().max_parallel == 8


# Hand-set budgets in demo/*.py (13 = engine default where the demo sets none).
DEMO_MAX_OPS = [
    (P.matmul, dict(M=128, N=128, K=4096), (), 13),
    (P.matmul, dict(M=128, N=128, K=4096), ("K",), 13),
    (P.rmsnorm_matmul, dict(M=128, N=6144, K=4096), ("K",), 13),
    (P.attention, dict(HQ=32, SQ=64, HKV=32, SKV=4096, D=64), ("SKV",), 14),  # mha/gqa/mqa decode
    (P.attention, dict(HQ=32, SQ=4096, HKV=32, SKV=4096, D=64), ("SKV",), 14),  # prefill split
    (P.qknorm, dict(HQ=64, SQ=4096, HKV=8, SKV=4096, D=128), (), 26),
    (P.qknorm, dict(HQ=64, SQ=4096, HKV=8, SKV=4096, D=128), ("SKV",), 28),
]


@pytest.mark.parametrize("fn,shapes,split,expected", DEMO_MAX_OPS)
def test_auto_max_ops_reproduces_demo_budgets(fn, shapes, split, expected):
    prog = trace(fn, shapes, cl.f16)
    assert SuperoptConfig(split=split).resolve_max_ops(prog) == expected
    assert SuperoptConfig(max_ops=7, split=split).resolve_max_ops(prog) == 7


def test_auto_max_ops_floor():
    # Demos that hand-set budgets below the engine default (prefill unsplit 12,
    # matmul_matmul_split 5/8) get the default floor of 13 instead: a superset search.
    assert auto_max_ops(3, 1, False) == 13
    assert auto_max_ops(24, 1, True) == 28


def test_superopt_cache_key_is_stable_and_content_addressed(monkeypatch, tmp_path):
    a = trace(P.matmul, dict(M=128, N=128, K=4096), cl.f16)
    b = trace(P.matmul, dict(M=128, N=128, K=4096), cl.f16)
    c = trace(P.matmul, dict(M=128, N=128, K=2048), cl.f16)
    key = cache.superopt_key(a, 13, ("K",), False)
    assert len(key) == 64 and key == cache.superopt_key(b, 13, ("K",), False)
    assert key != cache.superopt_key(c, 13, ("K",), False)
    assert key != cache.superopt_key(a, 14, ("K",), False)
    assert key != cache.superopt_key(a, 13, (), False)
    assert key != cache.superopt_key(a, 13, ("K",), True)
    monkeypatch.setattr(cache, "build_id", lambda: "other-build")
    assert key != cache.superopt_key(a, 13, ("K",), False)
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    path = cache.checkpoint_path(key)
    assert str(path).startswith(str(tmp_path / "persistent" / "superopt")) and key in str(path)


def test_resolve_configs_split():
    prog = trace(P.matmul, dict(M=128, N=128, K=4096), cl.f16)
    so, sc, _ = resolve_configs(prog, split=cl.Split("K", (4, 8)))
    assert so.split == ("K",) and sc.split_k == (4, 8)
    so, sc, _ = resolve_configs(prog, split="K")
    assert so.split == ("K",) and sc.split_k == (4, 8, 16)
    so, sc, _ = resolve_configs(prog, preset="fast", split=False)
    assert so.split == () and sc.split_k == () and sc.topk == PRESETS["fast"].schedule.topk
    so, sc, _ = resolve_configs(prog, schedule=ScheduleConfig(split_k=(32,)), split="K")
    assert sc.split_k == (32,)


def test_cli_split_parsing():
    from cleave.examples import parse_split

    s = parse_split("K:4,8,16")
    assert (s.dim, s.factors) == ("K", (4, 8, 16))
    assert parse_split("SKV").dim == "SKV"
    assert parse_split("none") is False


def test_identity_graphset_is_unsplit_program():
    from cleave.pipeline import GraphSet

    prog = trace(P.rmsnorm_matmul, dict(M=128, N=256, K=512), cl.f16)
    gs = GraphSet.identity(prog)
    assert len(gs) == 1 and not gs[0].is_split and gs[0].kind == "solo"
    assert gs[0].ops[-1] == "matmul"


def test_kernel_cache_key_covers_io_names():
    from cleave.frontend import Program
    from cleave.pipeline.compile import _kernel_key

    def prog(x, w):
        p = Program("mm")
        p.dims(M=16, K=32, N=8)
        p.output(p.input(x, ("M", "K")) @ p.input(w, ("K", "N")))
        return p

    cfgs = resolve_configs(prog("x", "w"))
    assert prog("x", "w").fingerprint() == prog("a", "b").fingerprint()
    assert _kernel_key(prog("x", "w"), *cfgs) != _kernel_key(prog("a", "b"), *cfgs)


def test_compile_rejects_dtype_for_a_program():
    import cleave

    prog = trace(P.matmul, dict(M=16, N=8, K=32), cl.f16)
    with pytest.raises(TypeError, match="dtype"):
        cleave.compile(prog, dtype=cl.bf16)


def test_schedule_module_is_callable(monkeypatch):
    import cleave
    import cleave.pipeline

    monkeypatch.setattr(cleave.pipeline, "schedule", lambda *a, **k: ("pipeline", a, k))
    assert cleave.schedule("g", topk=1) == ("pipeline", ("g",), {"topk": 1})
    assert callable(cleave.schedule.search_tiling) and callable(cleave.schedule.apply_tiling)


def test_invalid_split_graph_is_refused_not_lowered():
    import cleave
    from cleave.pipeline import GraphSet, SearchedGraph

    g = SearchedGraph(0, [], None, False, None, ("K",), invalid_split=True)
    plans = cleave.schedule(GraphSet([g]), ScheduleConfig(topk=1))
    assert not plans and g.kind == "invalid-split"
    assert "no partial/combine cut" in plans.refused[0].reason
