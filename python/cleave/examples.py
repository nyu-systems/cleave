"""Demo CLI: ``cleave.examples.main(fn_or_factory, shapes=..., ...)`` turns a demo into a script.

Flags: one per shape dim (traced functions) or per factory keyword (Program factories), plus
``--preset --topk --split DIM:f1,f2 --max-ops --fusion --warmup --iters --max-parallel
--no-cache --save DIR -v --json``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing
import sys
import time

from .core.errors import CleaveError
from .pipeline import compile as _compile
from .pipeline import resolve_configs
from .pipeline.config import PRESETS, EvalConfig, ScheduleConfig, SuperoptConfig, get_preset

_FIELDS = {cls: set(cls.__dataclass_fields__) - {"cache", "split"}
           for cls in (SuperoptConfig, ScheduleConfig, EvalConfig)}


_RESERVED = {"preset", "topk", "split", "max_ops", "fusion", "warmup", "iters", "max_parallel",
             "no_cache", "save", "verbose", "json"}


def table(rows, header=None) -> str:
    from .pipeline.report import _fmt

    rows = [tuple(f"{c:.2f}" if isinstance(c, float) else c for c in r) for r in rows]
    header = header or tuple(f"c{i}" for i in range(len(rows[0]) if rows else 0))
    return _fmt(rows, header)


def parse_split(text: str):
    """``K:4,8,16`` -> ``cl.Split("K", (4, 8, 16))``; ``K`` -> default factors; ``none`` -> False."""
    from .frontend.tensor import Split

    if text.lower() in ("none", "off", "false", "no", ""):
        return False
    dim, _, factors = text.partition(":")
    if not factors:
        return Split(dim)
    return Split(dim, tuple(int(f) for f in factors.split(",") if f))


def _is_traced(fn) -> bool:
    from .frontend import JitFunction
    from .frontend.trace import signature_dims

    if isinstance(fn, JitFunction):
        return True
    try:
        return bool(signature_dims(fn))
    except Exception:  # noqa: BLE001
        return False


def _bool(text: str) -> bool:
    if text.lower() in ("1", "true", "yes", "on"):
        return True
    if text.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")


def _target() -> str:
    try:
        import torch

        major, minor = torch.cuda.get_device_capability()
        return f"{torch.cuda.get_device_name()} (sm_{major}{minor})"
    except Exception:  # noqa: BLE001
        return "?"


def _parser(fn, traced, shapes, factory_params, presets, doc):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("problem")
    if traced:
        for name, size in shapes.items():
            g.add_argument(f"--{name}", type=int, default=None, help=f"dim {name} (default {size})")
    else:
        for name, default in factory_params.items():
            if name in _RESERVED:
                continue
            kind = _bool if isinstance(default, bool) else type(default) if default is not None else str
            g.add_argument(f"--{name}", dest=f"p_{name}", type=kind, default=None,
                           help=f"(default {default})")
    c = ap.add_argument_group("compile")
    c.add_argument("--preset", choices=sorted(set(PRESETS) | set(presets or {})), default=None)
    c.add_argument("--topk", type=int, default=None)
    c.add_argument("--split", action="append", default=None, metavar="DIM:F1,F2",
                   help="split dim and split-K factors (repeatable); 'none' disables")
    c.add_argument("--max-ops", dest="max_ops", type=int, default=None)
    c.add_argument("--fusion", choices=("auto", "force", "off"), default=None)
    c.add_argument("--warmup", type=int, default=None)
    c.add_argument("--iters", type=int, default=None)
    c.add_argument("--max-parallel", dest="max_parallel", type=int, default=None)
    c.add_argument("--no-cache", dest="no_cache", action="store_true")
    c.add_argument("--save", default=None, metavar="DIR")
    c.add_argument("-v", "--verbose", action="store_true")
    c.add_argument("--json", action="store_true", help="print the report as JSON")
    return ap


def _apply(cfgs, overrides: dict):
    so, sc, ev = cfgs
    pick = lambda cls: {k: v for k, v in overrides.items() if k in _FIELDS[cls]}  # noqa: E731
    return so.replace(**pick(SuperoptConfig)), sc.replace(**pick(ScheduleConfig)), ev.replace(**pick(EvalConfig))


def _print_report(report, verbose: bool):
    print(report.summary())
    if not verbose:
        return
    if report.timed:
        print(report.table(top=10))
    seen = set()
    for r in report.failures:
        if r.error_type not in seen:
            seen.add(r.error_type)
            lines = str(r.error).strip().splitlines() or [""]
            print(f"  first {r.error_type} ({r.label}): {lines[-1][:300]}")


def main(fn_or_factory, shapes: dict | None = None, dtype=None, presets: dict | None = None,
         split=None, expect_ok: int = 0, argv=None, **defaults):
    """Run a demo: parse flags, compile, print the per-stage report; returns the Kernel."""
    if multiprocessing.parent_process() is not None:
        return None  # re-imported by a spawned candidate worker: do nothing
    fn = fn_or_factory
    traced = _is_traced(fn)
    shapes = dict(shapes or {})
    factory_params = {}
    if not traced:
        sig = inspect.signature(fn)
        factory_params = {n: (defaults[n] if n in defaults else
                              None if p.default is inspect.Parameter.empty else p.default)
                          for n, p in sig.parameters.items()}
    doc = inspect.getdoc(sys.modules.get(getattr(fn, "__module__", ""), None)) or inspect.getdoc(fn)
    args = _parser(fn, traced, shapes, factory_params, presets, doc).parse_args(argv)

    # defaults < demo preset < flags.  Keys are shape dims / factory params / config fields.
    preset_name = args.preset or "default"
    merged = dict(defaults)
    merged.update((presets or {}).get(preset_name, {}))
    flags = {k: getattr(args, k) for k in ("topk", "max_ops", "fusion", "warmup", "iters",
                                          "max_parallel") if getattr(args, k) is not None}
    merged.update(flags)
    if traced:
        sizes = {**shapes, **{k: v for k, v in merged.items() if k in shapes}}
        sizes.update({k: getattr(args, k) for k in shapes if getattr(args, k) is not None})
        from .frontend import trace

        program = trace(fn, sizes, dtype)
    else:
        kw = {**{k: v for k, v in factory_params.items() if v is not None},
              **{k: v for k, v in merged.items() if k in factory_params}}
        kw.update({k: getattr(args, f"p_{k}") for k in factory_params
                   if k not in _RESERVED and getattr(args, f"p_{k}") is not None})
        program = fn(**kw)
        sizes = kw
    if args.split:
        split = tuple(parse_split(s) for s in args.split)
        split = False if any(s is False for s in split) else split
    elif "split" in merged:
        split = merged["split"]
    base = get_preset(preset_name if preset_name in PRESETS else "default")
    so, sc, ev = _apply(resolve_configs(program, preset=base, split=split), merged)
    if args.no_cache:
        so = so.replace(cache=False)

    name = getattr(program, "name", getattr(fn, "__name__", "kernel"))
    dims = " ".join(f"{k}={v}" for k, v in (getattr(program, "sizes", None) or sizes).items())
    print(f"{name}  {dims} {getattr(program, 'dtype', '')}  target={_target()}  preset={preset_name}",
          flush=True)

    def progress(stage, value):
        if args.verbose:
            print(f"[{stage}] {value}", flush=True)

    t0 = time.perf_counter()
    try:
        kernel = _compile(program, superopt=so, schedule=sc, eval=ev,
                          split=False if not so.split else None,
                          cache_kernels=not args.no_cache, progress=progress)
    except CleaveError as e:
        if getattr(e, "report", None) is None:
            raise
        _print_report(e.report, True)
        raise SystemExit(f"{name}: {e}") from None
    total = time.perf_counter() - t0
    report = kernel.report
    if report is not None:
        _print_report(report, args.verbose)
    else:
        print(f"cache hit: {kernel.meta.get('loaded_from')}  best {kernel.latency_us:.2f} us  {kernel.label}")
    print(f"compile time {total:.2f} s; best {kernel.variant} {kernel.latency_us:.2f} us"
          + (f"  [call: {kernel.unrunnable}]" if kernel.unrunnable else ""))
    if kernel.meta.get("saved"):
        print(f"cached: {kernel.meta['saved']}")
    if args.save:
        print(f"saved: {kernel.save(args.save)}")
    if args.json:
        payload = {"name": name, "sizes": dict(getattr(program, "sizes", {}) or {}),
                   "variant": kernel.variant, "latency_us": kernel.latency_us, "label": kernel.label,
                   "compile_s": total, "report": None if report is None else report.to_dict()}
        print(json.dumps(payload, indent=1, default=str))
    n_ok = len(report.correct) if report is not None else expect_ok
    if n_ok < expect_ok:
        raise SystemExit(f"expected >= {expect_ok} correct candidates, got {n_ok}")
    return kernel
