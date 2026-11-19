"""Measure every operator's validation error and print the tolerance each should declare.

    python -m tests._measure_tolerances --gpus 4,5,6,7 --out DIR    # measure, then report
    python -m tests._measure_tolerances --report DIR                # the table again

Run from the repo root on an otherwise idle set of GPUs. Every run sets
``CLEAVE_VALIDATION_LOG``, so each validated kernel appends the error the comparator measured
(``cleave.runtime.validate.compare_outputs``), whether it passed or not; nothing here depends
on the tolerances currently declared. Per seed (``CLEAVE_TEST_SEED``; "default" leaves it
unset) it runs

  demos     ``demo/<name>.py --preset fast --no-cache ...``, one GPU each;
  dynamic   ``python -m demo.dynamic <operator> --topk 4 --filter-before-topk``, one GPU each
            (every candidate on every captured workload);
  tests     the suites that validate kernels, one xdist worker per GPU. They are an ordinary
            pytest run and fail where a declared tolerance is exceeded -- that is a finding,
            the measurement itself is in the log either way.

The table has one row per operator and output: a histogram of the measured errors, the
largest error of the candidates that are clearly valid, and the tolerance that follows from
it by the rule in ``recommend``. Declare that value (``demo/<name>.py``,
``cleave.dynamic.registry``, ``tests/_tolerances.py``); never one chosen to make a kernel pass.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import queue
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from tests._tolerances import CEILING  # no operator may declare a looser bound

ROOT = Path(__file__).resolve().parents[1]
SAFETY = 4.0  # declared tolerance >= SAFETY x the largest valid error
# Errors are bimodal (valid kernels near one ulp, wrong ones above 1e-1). An error is
# "clearly valid" when SAFETY x it still fits under the ceiling; anything between that and
# WRONG is reported as ambiguous and has to be looked at by a person.
VALID = CEILING / SAFETY
WRONG = 1e-1
ULP = {"float16": 2.0**-10, "bfloat16": 2.0**-7, "float32": 2.0**-23}

# demo -> the runs that cover its plans (fast settings; lora and qknorm keep the fast
# preset's topk because few of their tilings survive, and mla_decode needs its whole
# tiling pool: its first valid tiling ranks far below the top 32).
FAST = ["--preset", "fast", "--topk", "8"]
SPLIT = ["--topk", "8", "--warmup", "4", "--iters", "100"]
DEMOS = {
    "matmul": [FAST, FAST + ["--split", "K:4,8"]],
    "matmul_matmul": [FAST, ["--preset", "split", *SPLIT]],
    "matmul_rowsum": [FAST],
    "lora": [["--preset", "fast"]],
    "swiglu": [FAST],
    "rmsnorm_matmul": [FAST, ["--preset", "split-k", *SPLIT]],
    "qknorm": [["--preset", "fast"]],
    "mha_decode": [FAST],
    "mha_decode_exact": [["--preset", "fast", "--topk", "16"]],
    "gqa_decode": [FAST],
    "mqa_decode": [FAST],
    "mha_prefill": [FAST],
    "gqa_prefill": [FAST],
    "mqa_prefill": [FAST],
    "mla_decode": [["--preset", "fast", "--topk", "1000"]],
    "mla_prefill": [FAST],
    "staged": [["--topk", "8"]],
}
GPU_TESTS = [
    "tests/schedule/end2end", "tests/graph/apply_tiling", "tests/graph/handwritten",
    "tests/reference", "tests/runtime/test_run_candidates.py", "tests/pipeline_api/test_e2e.py",
    "tests/frontend/test_exact_softmax.py", "tests/frontend_dynamic/test_paged_pipeline.py",
]


# ----------------------------------------------------------------------------- the rule
def step125(x: float) -> float:
    """The smallest of {1, 2, 5} x 10^k that is >= ``x``."""
    if x <= 0:
        return 0.0
    k = math.floor(math.log10(x))
    for mantissa in (1, 2, 5, 10):
        value = mantissa * 10.0**k
        if value >= x * (1 - 1e-9):
            return float(f"{value:.0e}")
    raise AssertionError(x)


def recommend(largest_valid: float, dtype: str, relative: bool) -> float:
    """The 1-2-5 step at or above ``SAFETY`` x the largest clearly-valid error.

    A row-relative bound is also never below one ulp of the output dtype: two correct
    results may differ by one ulp of the row max, whatever a few seeds happened to measure.
    """
    floor = ULP.get(dtype, 0.0) if relative else 0.0
    return min(CEILING, step125(max(SAFETY * largest_valid, floor)))


# ----------------------------------------------------------------------------- running
def _jobs(seeds, only, names=None):
    from cleave.dynamic import OPERATORS

    py = sys.executable
    wanted = lambda name: names is None or name in names  # noqa: E731
    for seed in seeds:
        if "demos" in only:
            for name, runs in DEMOS.items():
                for i, args in enumerate(runs if wanted(name) else ()):
                    yield (f"demo:{name}", f"demo-{name}-{i}", seed,
                           [py, f"demo/{name}.py", *args, "--max-parallel", "8"]
                           + ([] if name == "staged" else ["--no-cache"]))
        if "dynamic" in only:
            for name in filter(wanted, sorted(OPERATORS)):
                yield (f"dynamic:{name}", f"dynamic-{name}", seed,
                       [py, "-m", "demo.dynamic", name, "--topk", "4", "--filter-before-topk",
                        "--max-workloads", "1"])


def _run(context, tag, seed, cmd, out, env_extra):
    env = dict(os.environ, CLEAVE_VALIDATION_LOG=str(out / "validation" / f"{tag}__{seed}.jsonl"),
               CLEAVE_CACHE_ROOT=str(out / "cache"), **env_extra)
    env.pop("CLEAVE_TEST_SEED", None)
    env.pop("CLEAVE_VALIDATION_CONTEXT", None)
    if context:
        env["CLEAVE_VALIDATION_CONTEXT"] = context
    if seed != "default":
        env["CLEAVE_TEST_SEED"] = seed
    log = out / "logs" / f"{tag}__{seed}.log"
    t0 = time.time()
    with open(log, "w") as f:
        code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"[{time.strftime('%H:%M:%S')}] {tag} seed={seed} exit={code} {time.time() - t0:.0f}s",
          flush=True)
    return code


def measure(out: Path, gpus, seeds, only, names=None):
    for sub in ("validation", "logs", "cache"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    free = queue.Queue()
    for gpu in gpus:
        free.put(gpu)

    def one(job):
        gpu = free.get()
        try:
            return _run(*job, out, {"CUDA_VISIBLE_DEVICES": gpu})
        finally:
            free.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:
        list(pool.map(one, list(_jobs(seeds, only, names))))
    if "tests" in only:
        for seed in seeds:  # one worker per GPU: packing more causes candidate timeouts
            _run("", "tests", seed,
                 [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *GPU_TESTS],
                 out, {"CLEAVE_GPU_POOL": ",".join(gpus)})


# ----------------------------------------------------------------------------- reporting
def _test_operators():
    """pytest node id (without the phase) -> the operator whose tolerance it validates at."""
    from tests.graph.handwritten._cases import CASES as handwritten
    from tests.schedule.end2end.test_end2end import CASES as end2end

    def builder_of(path):
        return re.search(r"^from tests\._graphs import (\w+)$", (ROOT / path).read_text(), re.M)[1]

    def operator(node):
        path, _, test = node.partition("::")
        param = (re.search(r"\[(.*)\]$", test) or [None, None])[1]
        if path == "tests/schedule/end2end/test_end2end.py":
            return end2end[param][0].__name__
        if path == "tests/graph/handwritten/test_handwritten.py":
            case = handwritten[param]
            return f"dynamic:{param}" if case.out_idx is not None else case.source.__name__
        if path in ("tests/graph/handwritten/test_empty_request.py",
                    "tests/graph/handwritten/test_splitkv_decode.py"):
            return f"dynamic:{param or 'gqa_paged_decode_h32_kv8_d128_ps1'}"
        if path.endswith("test_gqa_paged_decode_h32_kv8_d128_ps1.py"):
            return "dynamic:gqa_paged_decode_h32_kv8_d128_ps1"
        if path == "tests/frontend_dynamic/test_paged_pipeline.py":
            return f"dynamic:gqa_paged_decode_h32_kv8_d128_ps{param}"
        if path.startswith(("tests/graph/apply_tiling/", "tests/reference/")):
            return builder_of(path)
        if path == "tests/runtime/test_run_candidates.py":
            return "matmul_graph"
        if path == "tests/pipeline_api/test_e2e.py":
            return "matmul"
        if path == "tests/frontend/test_exact_softmax.py":
            return "softmax" if "standalone" in test else "exact_attention"
        return node

    return operator


def load(out: Path) -> dict:
    """operator -> output index -> list of (dtype, relative?, value | None, kind, seed)."""
    operator_of = _test_operators()
    table = defaultdict(lambda: defaultdict(list))
    for path in sorted((out / "validation").glob("*.jsonl")):
        seed = path.stem.rpartition("__")[2]
        for line in path.read_text().splitlines():
            record = json.loads(line)
            context = record["context"]
            if not context.startswith(("demo:", "dynamic:")):
                node = context.rpartition(" (")[0]
                if not node.startswith("tests/") or "test_compare_outputs" in node \
                        or node.startswith("tests/test_runtime_tolerance.py") \
                        or "threaded_into_workers" in node:
                    continue  # comparator unit tests and deliberate failures
                context = "test:" + operator_of(node)
                context = context.replace("test:dynamic:", "dynamic:")
            for i, o in enumerate(record["outputs"]):
                relative = not (o["rtol"] == 0 and o["atol"] > 0)
                value = o["rel"] if relative else o["abs"]
                table[context][i].append((o["dtype"], relative, value, o["kind"], seed))
    return table


BUCKETS = ["0", "<1e-4", "<1e-3", "<1e-2", "<1e-1", ">=1e-1", "non-finite", "layout"]


def _bucket(value, kind):
    if kind == "non_finite":
        return "non-finite"
    if value is None:
        return "layout"  # wrong shape or dtype: nothing was measured
    if value == 0:
        return "0"
    return next((b for b, hi in zip(BUCKETS[1:5], (1e-4, 1e-3, 1e-2, 1e-1)) if value < hi), ">=1e-1")


def rows(table):
    for context in sorted(table, key=lambda c: (c.split(":")[0] != "demo", c)):
        for i, entries in sorted(table[context].items()):
            dtype, relative = entries[0][0], entries[0][1]
            counts = defaultdict(int)
            for _, _, value, kind, _ in entries:
                counts[_bucket(value, kind)] += 1
            values = [v for _, _, v, kind, _ in entries if v is not None and kind != "non_finite"]
            valid = [v for v in values if v <= VALID]
            ambiguous = sorted(v for v in values if VALID < v < WRONG)
            largest = max(valid, default=None)
            yield {
                "operator": context, "output": i, "dtype": dtype,
                "bound": "rtol" if relative else "atol",
                "seeds": sorted({e[4] for e in entries}), "n": len(entries),
                "histogram": {b: counts[b] for b in BUCKETS if counts[b]},
                "largest_valid": largest, "ambiguous": ambiguous,
                "tolerance": None if largest is None else recommend(largest, dtype, relative),
            }


def markdown(table) -> str:
    lines = [
        "| operator | out | dtype | bound | seeds | validated | error histogram | largest valid "
        "| tolerance |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows(table):
        hist = ", ".join(f"{k}: {v}" for k, v in r["histogram"].items())
        largest = "-" if r["largest_valid"] is None else f"{r['largest_valid']:.2e}"
        tol = "-" if r["tolerance"] is None else f"{r['bound']}={r['tolerance']:g}"
        if r["ambiguous"]:
            tol += f" (AMBIGUOUS: {len(r['ambiguous'])} in ({VALID:g}, {WRONG:g}), " \
                   f"max {r['ambiguous'][-1]:.2e})"
        lines.append(f"| {r['operator']} | {r['output']} | {r['dtype']} | {r['bound']} | "
                     f"{len(r['seeds'])} | {r['n']} | {hist} | {largest} | {tol} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="measure into DIR (validation/, logs/, cache/)")
    ap.add_argument("--report", type=Path, metavar="DIR", help="only print the table of DIR")
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids to use")
    ap.add_argument("--seeds", default="default,1,2")
    ap.add_argument("--only", default="demos,dynamic,tests")
    ap.add_argument("--names", default=None,
                    help="comma-separated demos / dynamic operators to restrict the runs to")
    args = ap.parse_args(argv)
    out = args.report or args.out
    if out is None:
        ap.error("give --out DIR to measure or --report DIR")
    if args.report is None:
        measure(out, args.gpus.split(","), args.seeds.split(","), args.only.split(","),
                args.names.split(",") if args.names else None)
    text = markdown(load(out))
    (out / "tolerances.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
