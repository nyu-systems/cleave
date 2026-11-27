"""CPU-only: configs, presets, the max_ops="auto" rule, cache keys, CLI parsing."""

import dataclasses
import importlib
import types

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

# the package exports the functions build() and compile() under the modules' names
build_mod = importlib.import_module("cleave.pipeline.build")
compile_mod = importlib.import_module("cleave.pipeline.compile")


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


# The auto rule on the programs of demo/*.py (13 = engine default where the rule gives
# less). ``cl.softmax`` records its max and the subtraction (two ops more than the request of
# the hand-derived pass), and the rule follows. A program split over a dim its max reduces
# is searched unsplit (``pipeline.superopt``), at the budget of ``split=()``: the first row
# of attention is the budget the decode demos search at.
DEMO_MAX_OPS = [
    (P.matmul, dict(M=128, N=128, K=4096), (), 13),
    (P.matmul, dict(M=128, N=128, K=4096), ("K",), 13),
    (P.rmsnorm_matmul, dict(M=128, N=6144, K=4096), ("K",), 13),
    (P.attention, dict(HQ=32, SQ=64, HKV=32, SKV=4096, D=64), (), 13),  # decode, --split none
    (P.attention, dict(HQ=32, SQ=64, HKV=32, SKV=4096, D=64), ("SKV",), 15),  # mha/gqa/mqa decode
    (P.attention, dict(HQ=32, SQ=4096, HKV=32, SKV=4096, D=64), ("SKV",), 15),  # prefill split
    (P.qknorm, dict(HQ=64, SQ=4096, HKV=8, SKV=4096, D=128), (), 27),
    (P.qknorm, dict(HQ=64, SQ=4096, HKV=8, SKV=4096, D=128), ("SKV",), 29),
]


@pytest.mark.parametrize("fn,shapes,split,expected", DEMO_MAX_OPS)
def test_auto_max_ops_reproduces_demo_budgets(fn, shapes, split, expected):
    prog = trace(fn, shapes, cl.f16)
    assert SuperoptConfig(split=split).resolve_max_ops(prog) == expected
    assert SuperoptConfig(max_ops=7, split=split).resolve_max_ops(prog) == 7


def test_auto_max_ops_floor():
    # Demos that hand-set budgets below the engine default (prefill unsplit 11,
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


def test_cache_schema_retires_kernels_built_before_a_scheduler_fix(monkeypatch):
    """A kernel-cache hit returns the saved kernel as it is, so a fix that changes what the
    same program and configs build has to move the key: the schema is in every one."""
    from cleave.pipeline.compile import _kernel_key

    prog = trace(P.matmul, dict(M=128, N=128, K=4096), cl.f16)
    cfgs = resolve_configs(prog)
    assert cache.SCHEMA >= 43
    key = _kernel_key(prog, *cfgs)
    monkeypatch.setattr(cache, "SCHEMA", 1)
    assert _kernel_key(prog, *cfgs) != key


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


def test_kernel_cache_key_covers_validation():
    """A saved kernel is returned without validating, so everything that decides what
    "validated" means is in its key: the tolerance, the input distributions, and the
    comparator's own version."""
    from cleave.pipeline.compile import _kernel_key

    prog = trace(P.matmul, dict(M=16, N=8, K=32), cl.f16)
    so, sc, ev = resolve_configs(prog)

    def key(config):
        return _kernel_key(prog, so, sc, config)

    declared = ev.replace(tolerance=cl.Tolerance(rtol=1e-3))
    assert key(declared) == key(ev.replace(tolerance=cl.Tolerance(rtol=1e-3)))
    assert key(declared) != key(ev.replace(tolerance=cl.Tolerance(rtol=2e-3)))
    assert key(declared) != key(ev.replace(tolerance=cl.Tolerance(rtol=1e-3, atol=1e-6)))
    # ... declared by boundary role too: every bound of it is in the key
    by_role = ev.replace(tolerance={"": cl.Tolerance(rtol=1e-3), "shift": cl.Tolerance(rtol=1e-6)})
    assert key(by_role) != key(declared)
    assert key(by_role) == key(ev.replace(tolerance=dict(by_role.tolerance)))
    assert key(by_role) != key(ev.replace(tolerance={**by_role.tolerance,
                                                     "shift": cl.Tolerance(rtol=2e-6)}))
    assert key(declared) != key(declared.replace(distributions=("normal", "large_logits")))
    assert key(declared) == key(declared.replace(max_parallel=2, timeout_s=5.0))
    # 1: elementwise assert_close with a 0.5/0.5 default. 3: the partial of a program
    # split over the softmax axis of an input was held to no adversarial case.
    assert cache.VALIDATION >= 7
    before = key(declared)
    cache.VALIDATION += 1
    try:
        assert key(declared) != before
    finally:
        cache.VALIDATION -= 1


def test_compile_and_build_require_a_declared_tolerance(monkeypatch):
    """There is no default bound: a dense program without one is refused before the search."""
    import cleave

    prog = trace(P.matmul, dict(M=16, N=8, K=32), cl.f16)
    monkeypatch.setattr(compile_mod, "superoptimize",
                        lambda *a, **k: pytest.fail("searched without a tolerance"))
    with pytest.raises(SpecError, match="no validation tolerance declared"):
        cleave.compile(prog, preset="fast")
    with pytest.raises(SpecError, match="no validation tolerance declared"):
        cleave.compile(prog, eval=EvalConfig(warmup=1, iters=1))
    assert EvalConfig().tolerance is None and EvalConfig().distributions == ("normal",)

    plan = types.SimpleNamespace(tile=None, program=prog)
    with pytest.raises(SpecError, match="no validation tolerance declared"):
        build_mod.build([plan])
    with pytest.raises(SpecError, match="input distributions must be drawn from"):
        build_mod.build([plan], config=EvalConfig(tolerance=cl.Tolerance(rtol=1e-3),
                                                  distributions=("normal", "bogus")))
    assert len(build_mod.build([])) == 0  # nothing to validate, nothing to declare


def test_a_per_output_tolerance_must_fit_every_plan(monkeypatch):
    """A split partial has more outputs than its combine and the solo plans, and all
    candidates share one declaration: a sequence that cannot fit them all is refused by
    ``build()`` before any candidate runs, not by every mismatching candidate at validate."""
    import cleave.runtime.evaluate as evaluate_mod
    import cleave.runtime.reference as reference_mod

    one, two = object(), object()
    outputs = {id(one): ["o"], id(two): ["l", "o"]}
    monkeypatch.setattr(reference_mod, "output_order", lambda graph: outputs[id(graph)])
    ran = []
    monkeypatch.setattr(evaluate_mod, "run_candidates",
                        lambda tiles, dtype, **kw: ran.append(kw["tolerance"]) or
                        [types.SimpleNamespace(code=None) for _ in tiles])

    def plan(graph, role):
        return types.SimpleNamespace(tile=types.SimpleNamespace(flat_graph=graph), role=role,
                                     label=f"g0 {role}", program=None)

    solo, partial, combine = plan(one, "main"), plan(two, "partial"), plan(one, "combine")
    t = cl.Tolerance(rtol=5e-3)

    def build(plans, tolerance):
        return build_mod.build(plans, config=EvalConfig(tolerance=tolerance))

    # one Tolerance covers every output of every plan
    assert len(build([solo, partial, combine], t)) == 3
    # one per output is fine while every plan has that many
    assert len(build([partial], (t, t))) == 1 and len(build([solo, combine], [t])) == 2
    assert ran == [t, (t, t), [t]]
    for plans, tolerance, message in (
            ([solo, partial, combine], (t, t),
             r"2 tolerances were declared, but plan 'g0 main' \(main\) has 1 output"),
            ([partial, combine], (t,),
             r"1 tolerances were declared, but plan 'g0 partial' \(partial\) has 2 output"),
            ([partial, solo], [t, t], "A per-output tolerance must fit every plan"),
            ([solo], (t, 1e-3), "must be a Tolerance or a sequence of them"),
            ([solo], 1e-3, "must be a Tolerance or a sequence of them")):
        with pytest.raises(SpecError, match=message):
            build(plans, tolerance)
    assert len(ran) == 3  # nothing ran for the refused declarations


def test_a_tolerance_by_role_gives_the_shift_its_own_bound(monkeypatch):
    """The split partial of an exact softmax returns its float32 shift next to its float16
    sums. One ``Tolerance`` is the bound of the outputs a program or its cut defines, so a
    program with such plans declares its tolerance by boundary role; ``build()`` refuses
    anything that leaves an output of some plan without a bound, before any candidate runs."""
    import cleave.runtime.evaluate as evaluate_mod
    import cleave.runtime.reference as reference_mod
    from cleave.ir.graph import Boundary

    def out(index, role=""):
        return types.SimpleNamespace(boundary=Boundary(index, role))

    one, three = object(), object()
    outputs = {id(one): [out(0)], id(three): [out(2, "shift"), out(0), out(1)]}
    monkeypatch.setattr(reference_mod, "output_order", lambda graph: outputs[id(graph)])
    ran = []
    monkeypatch.setattr(evaluate_mod, "run_candidates",
                        lambda tiles, dtype, **kw: ran.append(kw["tolerance"]) or
                        [types.SimpleNamespace(code=None) for _ in tiles])

    def plan(graph, role):
        return types.SimpleNamespace(tile=types.SimpleNamespace(flat_graph=graph), role=role,
                                     label=f"g0 {role}", program=None)

    solo, partial, combine = plan(one, "main"), plan(three, "partial"), plan(one, "combine")
    t, shift = cl.Tolerance(rtol=5e-3), cl.Tolerance(rtol=2e-6)

    def build(plans, tolerance):
        return build_mod.build(plans, config=EvalConfig(tolerance=tolerance))

    by_role = {"": t, "shift": shift}
    assert len(build([solo, partial, combine], by_role)) == 3
    assert len(build([solo, combine], by_role)) == 2  # an entry no plan needs is no error
    assert len(build([solo, combine], t)) == 2  # no plan returns a shift: one bound is enough
    assert len(build([partial], (shift, t, t))) == 1  # one per output, by position
    assert ran == [by_role, by_role, t, (shift, t, t)]
    for plans, tolerance, message in (
            ([solo, partial, combine], t,
             r"build\(\): plan 'g0 partial' \(partial\): one Tolerance covers the outputs a "
             r"program or its cut defines, and this kernel also returns outputs of role "
             r"\['shift'\]"),
            ([solo, partial], {"": t},
             r"plan 'g0 partial' \(partial\): no validation tolerance declared for the outputs "
             r"of role \['shift'\]"),
            ([solo, partial], {"shift": shift},
             r"plan 'g0 main' \(main\): no validation tolerance declared for the outputs of "
             r"role \[''\]"),
            ([solo], {"": 5e-3}, "maps each boundary role"),
            ([solo, partial], (t,), "A per-output tolerance must fit every plan")):
        with pytest.raises(SpecError, match=message):
            build(plans, tolerance)
    assert len(ran) == 4  # nothing ran for the refused declarations


def test_compile_takes_the_tolerance_from_jit_options_or_the_keyword(monkeypatch):
    import cleave

    seen = []

    def stop(program, so, sc, ev):
        seen.append(ev.tolerance)
        raise RuntimeError("stop")

    monkeypatch.setattr(compile_mod, "_kernel_key", stop)

    @cleave.jit(tolerance=cl.Tolerance(rtol=1e-3))
    def mm(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
        return x @ w

    shapes = dict(M=16, N=16, K=64)
    for call in (lambda: cleave.compile(mm, shapes=shapes),
                 lambda: mm.compile(**shapes),
                 lambda: cleave.compile(mm, shapes=shapes, tolerance=cl.Tolerance(rtol=5e-3)),
                 lambda: cleave.compile(mm.fn, shapes=shapes, preset="fast",
                                        tolerance=cl.Tolerance(rtol=2e-3))):
        with pytest.raises(RuntimeError, match="stop"):
            call()
    assert [t.rtol for t in seen] == [1e-3, 1e-3, 5e-3, 2e-3]


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
