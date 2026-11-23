"""Measure every operator's validation error and print the tolerance each should declare.

    python -m tests._measure_tolerances --gpus 4,5,6,7 --out DIR    # measure, then report
    python -m tests._measure_tolerances --report DIR                # the table again
    python -m tests._measure_tolerances --report DIR --pin          # ... and pin it in-tree

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

The table has one row per operator and output: a histogram of the measured row-relative
errors, the largest error of the candidates that are clearly valid, and the tolerance that
follows from it by the rule in ``recommend``. Declare that value (``demo/<name>.py``,
``cleave.dynamic.registry``, ``tests/_tolerances.py``); never one chosen to make a kernel pass.

``--pin`` writes the measurement to ``tests/measured_errors.json``.
``tests/test_tolerance_declarations.py`` and ``tests/demo/test_demos.py`` hold every
declaration to what the rule gives for that file, so after re-measuring the failing tests
list exactly the declarations to change.
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

from cleave.runtime.validate import Tolerance
from tests._tolerances import CEILING, FP32_CEILING  # no operator may declare a looser bound

ROOT = Path(__file__).resolve().parents[1]
MEASURED = ROOT / "tests" / "measured_errors.json"  # what the declarations are pinned to
SAFETY = 4.0  # declared tolerance >= SAFETY x the largest valid error
# ... except for bf16. It has 8 significand bits, so one ulp is 7.8e-3 of the row max, and
# that is what a valid kernel measures (7.8e-3 .. 9.4e-3). 4x it is the ceiling itself, which
# for any row whose max is above 0.25 is looser than the elementwise 0.01 + 0.01|ref| the
# paged operators were held to before. 2x gives 2e-2, i.e. 2.56 ulps.
COARSE_SAFETY = {"bfloat16": 2.0}
WRONG = 1e-1
ULP = {"float16": 2.0**-10, "bfloat16": 2.0**-7, "float32": 2.0**-23}


def ceiling(dtype: str) -> float:
    """The loosest rtol an output of ``dtype`` may declare."""
    return FP32_CEILING if dtype == "float32" else CEILING


def safety(dtype: str) -> float:
    return COARSE_SAFETY.get(dtype, SAFETY)


def valid_limit(dtype: str) -> float:
    """The largest error that counts as "clearly valid": ``safety`` x it still fits under the
    ceiling. Errors are bimodal (valid kernels within a few ulp, wrong ones above WRONG);
    anything between this and WRONG is reported as ambiguous, for a person to look at."""
    return ceiling(dtype) / safety(dtype)


# demo -> the runs that cover its plans (fast settings; lora and qknorm keep the fast
# preset's topk because few of their tilings survive, mla_decode needs its whole tiling
# pool: its first valid tiling ranks far below the top 32, and mha_backward's top 8 tilings
# all fail to compile).
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
    "mha_backward": [["--preset", "fast"]],
    "staged": [["--topk", "8"]],
}
GPU_TESTS = [
    "tests/schedule/end2end", "tests/graph/apply_tiling", "tests/graph/handwritten",
    "tests/reference", "tests/runtime/test_run_candidates.py", "tests/pipeline_api/test_e2e.py",
    "tests/frontend/test_exact_softmax.py", "tests/frontend_dynamic/test_paged_pipeline.py",
    "tests/pipeline_api/test_declared_validation.py", "tests/runtime/test_inter_outputs.py",
    "tests/pipeline_api/test_per_output_gpu.py",
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


def recommend(largest_valid: float, dtype: str) -> float:
    """The rtol for an output: the 1-2-5 step at or above ``safety(dtype)`` x its largest
    clearly-valid row-relative error.

    Never below one ulp of the output dtype (two correct results may differ by one ulp of
    the row max, whatever a few seeds happened to measure) and never above its ceiling.
    """
    return min(ceiling(dtype), step125(max(safety(dtype) * largest_valid, ULP.get(dtype, 0.0))))


# The absolute part of a bound that has one (the paged lse, an fp32 log2-sum-exp): what an
# lse of exactly 0 can be off by, SAFETY fp32 ulps of a sum of 1 in log2 units. It is fixed,
# never measured: the lse error is proportional to |lse| (2 .. 25 on the captured workloads,
# hundreds at x100 logits, where one fp32 ulp alone is 3e-5), so an absolute bound sized on
# one logit scale is wrong on every other. The measured part of the bound is its rtol.
ATOL_FLOOR = step125(SAFETY * ULP["float32"] / math.log(2))  # 1e-6


def expected_tolerances(outputs) -> list:
    """Per measured output (an entry of ``measured_errors.json``), the ``Tolerance`` the rule
    gives, or None for an output no valid kernel was measured on."""
    return [
        None if o["rel"] is None else
        Tolerance(rtol=recommend(o["rel"], o["dtype"]), atol=ATOL_FLOOR if o["absolute"] else 0.0)
        for o in outputs
    ]


def expected_tolerance(outputs) -> Tolerance:
    """The single ``Tolerance`` an operator declares for all its outputs: the loosest of what
    the rule gives each (a dense program's plans do not all have the same outputs)."""
    rtols = [t.rtol for t in expected_tolerances(outputs) if t is not None]
    assert rtols and not any(o["absolute"] for o in outputs), outputs
    return Tolerance(rtol=max(rtols))


def expected_declaration(declared, outputs):
    """What ``declared`` must equal for an operator measured as ``outputs``: one
    ``Tolerance`` for every output (``expected_tolerance``), or, when it declares a sequence,
    one per output (``expected_tolerances``; every output must have been measured)."""
    if isinstance(declared, Tolerance):
        return expected_tolerance(outputs)
    per_output = expected_tolerances(outputs)
    assert None not in per_output, outputs
    return tuple(per_output)


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
        if path == "tests/pipeline_api/test_declared_validation.py":
            return "lse_attention"
        if path == "tests/runtime/test_inter_outputs.py":
            return "inter_pair"
        if path == "tests/pipeline_api/test_per_output_gpu.py":
            return "per_output_pair"
        return node

    return operator


def load(out: Path) -> dict:
    """operator -> output index -> list of (dtype, absolute part?, rel, abs, kind, seed)."""
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
                        or "threaded_into_workers" in node \
                        or "test_noise_where_an_input_is_declared" in node:
                    continue  # comparator unit tests and deliberate failures
                context = "test:" + operator_of(node)
                context = context.replace("test:dynamic:", "dynamic:")
            for i, o in enumerate(record["outputs"]):
                # Every output is judged by its row-relative error, whatever bound it was
                # validated at when this was logged; "absolute" only records that the
                # operator declares an absolute part as well (ATOL_FLOOR).
                table[context][i].append(
                    (o["dtype"], o["atol"] > 0, o["rel"], o["abs"], o["kind"], seed))
    return table


_DECADES = [(f"<1e-{k}", 10.0**-k) for k in (6, 5, 4, 3, 2, 1)]
BUCKETS = ["0", *(name for name, _ in _DECADES), ">=1e-1", "non-finite", "layout"]


def _bucket(value, kind):
    if kind == "non_finite":
        return "non-finite"
    if value is None:
        return "layout"  # wrong shape or dtype: nothing was measured
    if value == 0:
        return "0"
    return next((name for name, hi in _DECADES if value < hi), ">=1e-1")


def rows(table):
    for context in sorted(table, key=lambda c: (c.split(":")[0] != "demo", c)):
        for i, entries in sorted(table[context].items()):
            dtype = entries[0][0]
            absolute = any(e[1] for e in entries)
            counts = defaultdict(int)
            for _, _, rel, _, kind, _ in entries:
                counts[_bucket(rel, kind)] += 1
            measured = [(rel, abs_) for _, _, rel, abs_, kind, _ in entries
                        if rel is not None and kind != "non_finite"]
            limit = valid_limit(dtype)
            valid = [(rel, abs_) for rel, abs_ in measured if rel <= limit]
            largest = max((rel for rel, _ in valid), default=None)
            yield {
                "operator": context, "output": i, "dtype": dtype, "absolute": absolute,
                "seeds": sorted({e[5] for e in entries}), "n": len(entries),
                "histogram": {b: counts[b] for b in BUCKETS if counts[b]},
                "largest_valid": largest,
                "largest_valid_abs": max((abs_ for _, abs_ in valid), default=None),
                "ambiguous": sorted(rel for rel, _ in measured if limit < rel < WRONG),
                "rtol": None if largest is None else recommend(largest, dtype),
                "atol": ATOL_FLOOR if absolute else 0.0,
            }


def measured_errors(table) -> dict:
    """operator -> per output ``{dtype, rel, abs, absolute, seeds, validated}``: the largest
    clearly-valid error of each output, which is all the rule needs (``MEASURED``)."""
    out = defaultdict(list)
    for r in rows(table):
        assert r["output"] == len(out[r["operator"]]), r
        out[r["operator"]].append({
            "dtype": r["dtype"], "rel": r["largest_valid"], "abs": r["largest_valid_abs"],
            "absolute": r["absolute"], "seeds": len(r["seeds"]), "validated": r["n"],
        })
    return dict(out)


def markdown(table) -> str:
    lines = [
        "| operator | out | dtype | seeds | validated | row-relative error histogram "
        "| largest valid | largest valid abs | tolerance |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows(table):
        hist = ", ".join(f"{k}: {v}" for k, v in r["histogram"].items())
        largest = "-" if r["largest_valid"] is None else f"{r['largest_valid']:.2e}"
        largest_abs = "-" if r["largest_valid_abs"] is None else f"{r['largest_valid_abs']:.2e}"
        tol = "-" if r["rtol"] is None else f"rtol={r['rtol']:g}" + (
            f", atol={r['atol']:g}" if r["atol"] else "")
        if r["ambiguous"]:
            tol += f" (AMBIGUOUS: {len(r['ambiguous'])} in ({valid_limit(r['dtype']):g}, " \
                   f"{WRONG:g}), max {r['ambiguous'][-1]:.2e})"
        lines.append(f"| {r['operator']} | {r['output']} | {r['dtype']} | {len(r['seeds'])} | "
                     f"{r['n']} | {hist} | {largest} | {largest_abs} | {tol} |")
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
    ap.add_argument("--pin", action="store_true",
                    help=f"also write the measurement to {MEASURED.relative_to(ROOT)}, which "
                         "the declaration tests hold every declared tolerance to")
    args = ap.parse_args(argv)
    out = args.report or args.out
    if out is None:
        ap.error("give --out DIR to measure or --report DIR")
    if args.report is None:
        measure(out, args.gpus.split(","), args.seeds.split(","), args.only.split(","),
                args.names.split(",") if args.names else None)
    table = load(out)
    text = markdown(table)
    (out / "tolerances.md").write_text(text + "\n")
    pinned = json.dumps(measured_errors(table), indent=1, sort_keys=True) + "\n"
    (out / MEASURED.name).write_text(pinned)
    if args.pin:
        MEASURED.write_text(pinned)
    print(text)


if __name__ == "__main__":
    main()
