"""``cleave.compile``: superoptimize -> schedule -> build -> evaluate -> best Kernel."""

from __future__ import annotations

import dataclasses
import time
from typing import Callable

from ..core.errors import CleaveError
from . import cache
from .build import build, evaluate
from ..frontend.tensor import as_dtype, split_names
from .config import EvalConfig, ScheduleConfig, SuperoptConfig, get_preset
from .kernel import Kernel, load
from .plans import schedule as schedule_
from .superopt import superoptimize

DEFAULT_SPLIT_FACTORS = (4, 8, 16)


def as_program(fn_or_program, shapes=None, dtype=None):
    if hasattr(fn_or_program, "to_tile_graph"):
        if shapes:
            raise TypeError("shapes= is only for functions; this is already a Program")
        if dtype is not None and as_dtype(dtype) != fn_or_program.dtype:
            raise TypeError(f"dtype={dtype} differs from the Program's {fn_or_program.dtype}; "
                            "set it when tracing")
        return fn_or_program
    from ..frontend import trace

    return trace(fn_or_program, shapes or {}, dtype)


def _split_factors(split) -> tuple[int, ...]:
    if split is None or isinstance(split, str):
        return ()
    items = split if isinstance(split, (tuple, list)) else (split,)
    out = []
    for s in items:
        for f in getattr(s, "factors", ()) or ():
            if int(f) not in out:
                out.append(int(f))
    return tuple(out)


def resolve_configs(program, *, preset="default", superopt=None, schedule=None, eval=None,
                    split=None):
    """Preset defaults, overridden by explicit configs, then by ``split`` (None: Program's own)."""
    base = get_preset(preset)
    so, sc, ev = superopt or base.superopt, schedule or base.schedule, eval or base.eval
    if split is None:
        split = getattr(program, "split", ()) or None
    if split is not None and split is not False:
        dims = split_names(split)
        so = so.replace(split=dims) if superopt is None or not superopt.split else so
        if dims and not sc.split_k:
            sc = sc.replace(split_k=_split_factors(split) or DEFAULT_SPLIT_FACTORS)
    elif split is False:
        so = so.replace(split=())
    return so, sc, ev


def _kernel_key(program, so: SuperoptConfig, sc: ScheduleConfig, ev: EvalConfig) -> str:
    def fields(cfg, *drop):
        return {k: v for k, v in dataclasses.asdict(cfg).items() if k not in drop}

    # names are not in the fingerprint but are baked into the Kernel's call signature
    names = [[s.name for s in program.inputs], [o.name for o in program.outputs]]
    return cache.digest("kernel", program.fingerprint(), names, so.resolve_max_ops(program),
                        fields(so, "threads", "cache", "max_ops"), fields(sc),
                        fields(ev, "max_parallel", "timeout_s"))


def compile(fn_or_program, *, shapes: dict | None = None, dtype=None, preset="default",
            superopt: SuperoptConfig | None = None, schedule: ScheduleConfig | None = None,
            eval: EvalConfig | None = None, split=None, cache_kernels: bool | None = None,
            progress: Callable[[str, object], None] | None = None, **paged_options) -> Kernel:
    """Compile a traced function or ``Program`` to its fastest validated kernel.

    ``split``: dim name / ``cl.Split`` / tuple; ``None``: the Program's own, ``False``: off.
    ``cache_kernels`` (default ``superopt.cache``) reuses a saved kernel. Paged programs go to
    ``pipeline.dynamic.compile_dynamic`` (``hints``, ``workloads``, ...).
    """
    from ..frontend.jit import JitFunction
    from ..frontend.paged import is_paged

    if isinstance(fn_or_program, JitFunction):  # @jit(**options) are defaults for explicit args
        opts = dict(fn_or_program.options)
        if preset == "default":
            preset = opts.pop("preset", preset)
        superopt = superopt or opts.pop("superopt", None)
        schedule = schedule or opts.pop("schedule", None)
        eval = eval or opts.pop("eval", None)
        for k in ("preset", "superopt", "schedule", "eval"):
            opts.pop(k, None)
        paged_options = {**opts, **paged_options}
        if split is None:
            split = fn_or_program.split
    if is_paged(fn_or_program):
        from .dynamic import compile_dynamic

        return compile_dynamic(fn_or_program, dtype=dtype, preset=preset, superopt=superopt,
                               schedule=schedule, eval=eval, split=split, progress=progress,
                               **paged_options)
    if paged_options:
        raise TypeError(f"compile() got unexpected options {sorted(paged_options)}")
    program = as_program(fn_or_program, shapes, dtype)
    so, sc, ev = resolve_configs(program, preset=preset, superopt=superopt, schedule=schedule,
                                 eval=eval, split=split)
    use_cache = so.cache if cache_kernels is None else cache_kernels
    note = progress or (lambda stage, value: None)
    kdir = cache.kernel_dir(getattr(program, "name", "kernel"), _kernel_key(program, so, sc, ev))
    if use_cache and (kdir / "kernel.json").exists():
        try:
            kernel = load(kdir)
        except Exception:  # noqa: BLE001 - a stale or partial entry is simply recompiled
            kernel = None
        if kernel is not None:
            kernel.meta["cache"] = "hit"
            note("cache", kernel)
            return kernel

    t0 = time.perf_counter()
    graphs = superoptimize(program, so)
    note("superoptimize", graphs)
    t1 = time.perf_counter()
    plans = schedule_(graphs, sc)
    note("schedule", plans)
    t2 = time.perf_counter()
    built = build(plans, config=ev)
    note("build", built)
    report = evaluate(built, ev)
    report.timings = {"superoptimize": t1 - t0, "schedule": t2 - t1, **report.timings}
    note("evaluate", report)
    try:
        kernel = report.best()
    except CleaveError as e:
        e.report = report  # examples.main prints the stage report of a failed compile
        raise
    kernel.meta["cache"] = "miss"
    if use_cache:
        kernel.save(kdir)
        kernel.meta["saved"] = str(kdir)
    return kernel

