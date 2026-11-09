"""Shared CLI for the frontend-style dynamic demos (``demo/dynamic/<family>.py``)."""

from __future__ import annotations

import json
import time
from pathlib import Path


def add_args(ap) -> None:
    ap.add_argument("--topk", type=int, default=400)
    ap.add_argument("--split", default="", help="split-KV factors to search too, e.g. 4,8,16")
    ap.add_argument("--max-workloads", type=int, default=6,
                    help="workloads to TIME each candidate on (all are validated)")
    ap.add_argument("--skip-workloads", action="store_true", help="search and page only")
    ap.add_argument("--filter-before-topk", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="validate/time on the family's synthetic workloads, not the captured ones")
    ap.add_argument("--dump-best", default=None, metavar="DIR",
                    help="write the winner to DIR/<op>.py (with --split also .split/.nosplit/.hybrid.json)")
    ap.add_argument("-v", "--verbose", action="store_true")


def main(doc, presets, build, argv, default, **run_kw):
    """``python -m demo.dynamic.<family> [--preset P] ...`` for a family's ``build(preset)``."""
    import argparse

    ap = argparse.ArgumentParser(description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(presets), default=default)
    add_args(ap)
    args = ap.parse_args(argv)
    fn, hints = build(args.preset)
    return run(fn, args, hints=hints, **run_kw)


def run(fn, args, *, hints=None, tolerance=None, synthetic=None):
    """``synthetic``: the ``cl.workloads`` spec ``--synthetic`` selects (default: captured)."""
    import cleave
    import cleave.lang as cl
    from cleave.frontend.paged import trace_paged
    from cleave.pipeline import ScheduleConfig

    program = trace_paged(fn, hints=hints)
    attention = program.attention()
    spec = attention.paging_spec()
    print(f"=== {program.name}\nspec: {spec}", flush=True)
    factors = tuple(int(f) for f in args.split.split(",") if f.strip())
    split = cl.Split(attention.kv_len.name, factors) if factors else None
    schedule = ScheduleConfig(topk=args.topk)
    t0 = time.perf_counter()
    if args.skip_workloads:
        from cleave.pipeline import resolve_configs, superoptimize
        from cleave.pipeline.dynamic import schedule_paged

        body = attention.dense_body()
        so, sc, _ = resolve_configs(body, schedule=schedule)
        plans = schedule_paged(superoptimize(body, so), spec, sc,
                               filter_before_topk=args.filter_before_topk)
        print(plans.summary(), f"\nelapsed {time.perf_counter() - t0:.1f}s")
        return plans

    def progress(stage, value):
        if args.verbose or stage == "split":
            print(f"[{stage}] {value}", flush=True)

    if args.synthetic and synthetic is None:
        raise SystemExit(f"{program.name}: this family has no synthetic workloads")
    workloads = synthetic if args.synthetic else None
    kernel = cleave.compile(fn, schedule=schedule, split=split, hints=hints, workloads=workloads,
                            tolerance=tolerance, max_workloads=args.max_workloads,
                            filter_before_topk=args.filter_before_topk, progress=progress)
    report = kernel.report
    print(report.summary())
    print(report.table())
    print(f"BEST {program.name}: {kernel.label} at {kernel.latency_us:.2f}us geomean", flush=True)
    if args.dump_best:
        out = Path(args.dump_best)
        out.mkdir(parents=True, exist_ok=True)
        files = {f"{program.name}.py": kernel.source}
        if factors:
            for tag, cand in (("split", report.best(split=True)), ("nosplit", report.best(split=False))):
                if cand is not None:
                    files[f"{program.name}.{tag}.py"] = cand.source
        if kernel.dispatch:
            files[f"{program.name}.hybrid.json"] = json.dumps(
                dict(kernel.dispatch, operator=program.name, split=f"{program.name}.split.py",
                     single=f"{program.name}.nosplit.py"), indent=1) + "\n"
        for name, text in files.items():
            (out / name).write_text(text, encoding="utf-8")
            print(f"  wrote {out / name}")
    print(f"elapsed {time.perf_counter() - t0:.1f}s\n")
    return kernel
