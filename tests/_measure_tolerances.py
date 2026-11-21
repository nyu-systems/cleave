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
            (every candidate on every captured workload and on every adversarial one: the
            softmax of every paged layer is exact, so an operator validates on both);
  tests     the suites that validate kernels, one xdist worker per GPU. They are an ordinary
            pytest run and fail where a declared tolerance is exceeded -- that is a finding,
            the measurement itself is in the log either way.

The table has one row per operator and output: a histogram of the measured row-relative
errors, the largest error of the candidates that are clearly valid, and the tolerance that
follows from it by the rule in ``recommend``. An output with a boundary role (the float32
shift a split partial of an exact softmax returns) has a row of its own, under its role,
whatever place it has among a kernel's outputs; the other outputs are numbered among
themselves. That is the operator's error on its normal
inputs. Where its softmax is exact (a dense program that records the request, every
dynamic operator) its kernels also run on adversarial softmax inputs, at the same bound,
and those comparisons are counted without feeding it (``ADVERSARIAL_CASE``).
Declare that value (``demo/<name>.py``,
``cleave.dynamic.registry``, ``tests/_tolerances.py``); never one chosen to make a kernel pass.

``--pin`` writes the measurement to ``tests/measured_errors.json``.
``tests/test_tolerance_declarations.py`` and ``tests/demo/test_demos.py`` hold every
declaration to what the rule gives for that file, so after re-measuring the failing tests
list exactly the declarations to change.

A measurement of a part (``--only``, ``--names``) pins the operators it measured in full
and leaves the others as they are pinned: a demo's operator is fed by that demo's runs, a
test operator by the GPU tests (always the whole list), a dynamic operator by its
``demo.dynamic`` run and by the GPU tests. The directory records what was run
(``ran.json``), so ``--report DIR --pin`` pins the same operators later, and a directory
that does not say, or that ran fewer than the three seeds, pins nothing.

    python -m tests._measure_tolerances --gpus 0,1,2,3 --out DIR --only tests --pin
    python -m tests._measure_tolerances --gpus 0,1,2,3 --out DIR --only demos \
        --names mha_decode,gqa_decode --pin
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
RAN = "ran.json"  # in a measurement directory: what the measurement ran
SEEDS = ("default", "1", "2")  # what a pinned measurement covers
KINDS = ("demos", "dynamic", "tests")
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
    "tests/pipeline_api/test_split_attention.py",
    "tests/pipeline_api/test_exact_split_attention.py",
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
#
# The rtol is measured where it is what bounds the output. A comparison in which every
# entry is within the floor is inside ``rtol * rowmax + ATOL_FLOOR`` whatever the rtol, so
# it says nothing about the rtol: its row-relative error is the floor's share of a small
# row, not an error that grows with the row. Measured on gqa_paged_decode_h32_kv8_d128_ps1
# with logits that only fall along the keys (the lse of a row is then close to 0): errors
# of 3e-7 .. 4e-7 are 5e-7 .. 7e-7 of a row max of 0.6, which as an rtol input would ask
# for 5e-6, while the declared 1e-6 * rowmax + 1e-6 holds those rows with 3x to spare.
# Such a comparison is counted and held to the declared bound like any other; it does not
# feed ``largest valid`` (``feeds_rtol``). One entry over the floor and it feeds in full.
ATOL_FLOOR = step125(SAFETY * ULP["float32"] / math.log(2))  # 1e-6


def feeds_rtol(absolute: bool, abs_error: float) -> bool:
    """Whether a valid comparison's row-relative error is an input to the output's rtol."""
    return not (absolute and abs_error <= ATOL_FLOOR)


# A dense program's tolerance is declared from its error on the normal inputs. Once its
# softmax is exact its kernels are also validated on the adversarial softmax inputs
# (``cleave.runtime.inputs.ADVERSARIAL``), at that same bound: those comparisons are counted
# and held to the declaration like any other, and they are no input to it. A bound is never
# loosened for an adversarial case; a kernel that needs more than the bound there fails.
# (Measured on this operator family: a row dominated by its last keys is off by up to two
# fp16 ulps of its max, 1.2e-3 .. 1.9e-3, against 9.8e-4 .. 1.2e-3 on normal inputs; as an
# input that would turn every declared 5e-3 into 1e-2.)
# Such a comparison is logged with the case in its ``what`` ("increasing inputs") and is
# given this kind in the table, passed or not (a non-finite one stays non-finite: nothing
# was measured). One that is over the bound is a finding, and the table says so.
#
# A dynamic operator's tolerance is declared from its error on its normal inputs: its
# captured workloads and the hand-made requests that leave the inputs as generated. Its
# softmax is exact, so its kernels are also validated on adversarial softmax inputs -- the
# registry's adversarial workloads and the hand-made requests that transform the inputs --
# at that same bound: those comparisons are counted and held to the declaration like any
# other, and they are no input to it. A bound is never loosened for an adversarial case; a
# kernel that needs more than the bound there fails.
# (Measured on these operators: the lse of logits x100 or of a 128-unit ramp along the keys
# is off by 4.1e-7 .. 6.9e-7 of the row max for the GQA layers and by 5.0e-6 for the MLA
# prefill, where the float32 reference itself is 3.2e-6 from what the inputs allow; on
# normal inputs the same kernels measure 2.3e-7 .. 5.7e-7 and 1.5e-6. As inputs the first
# figures would turn six of the nine declared bounds into the next 1-2-5 step.)
# Such a comparison is logged with its workload in ``what``, and the workload's key ends
# in ``@<case>`` (``wl_16_1070_1020@large_logits``, ``tile_edges@decreasing``). It is given
# this kind in the table, passed or not (a non-finite one stays non-finite: nothing was
# measured). One that is over the bound is a finding, and the table says so.
ADVERSARIAL_CASE = "adversarial"


def _adversarial_whats() -> frozenset:
    from cleave.runtime.inputs import ADVERSARIAL

    return frozenset(f"{name} inputs" for name in ADVERSARIAL)


def is_adversarial(context: str, what: str) -> bool:
    """Whether a logged comparison ran on adversarial softmax inputs: a dynamic operator on
    an adversarial workload (by its key), a dense program on one of its adversarial cases
    (by name)."""
    if context.startswith("dynamic:"):
        return "@" in what
    return what in _adversarial_whats()


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
    assert not any(o.get("role") for o in outputs), "declare this operator by role"
    rtols = [t.rtol for t in expected_tolerances(outputs) if t is not None]
    assert rtols and not any(o["absolute"] for o in outputs), outputs
    return Tolerance(rtol=max(rtols))


def expected_tolerances_by_role(outputs) -> dict:
    """What an operator with an output that has a boundary role declares: one ``Tolerance``
    per role. ``""`` is the loosest of what the rule gives the outputs without a role, as
    for any other operator; each role's is what the rule gives the outputs of that role."""
    by_role = {}
    for o in outputs:
        by_role.setdefault(o.get("role", ""), []).append(o)
    assert set(by_role) > {""}, outputs
    return {role: expected_tolerance([{k: v for k, v in o.items() if k != "role"}
                                      for o in entries])
            for role, entries in by_role.items()}


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
    (out / RAN).write_text(json.dumps(
        {"seeds": list(seeds), "only": sorted(only), "names": names}) + "\n")


def measured_in_full(ran):
    """``operator -> bool``: whether a measurement that ran ``ran`` (the ``ran.json`` of its
    directory) ran everything that feeds that operator's entry."""
    only, names = set(ran["only"]), ran["names"]

    def in_full(operator: str) -> bool:
        kind, _, name = operator.partition(":")
        named = names is None or name in names
        if kind == "demo":
            return "demos" in only and named
        if kind == "test":  # the GPU tests run as one list, whatever --names says
            return "tests" in only
        # a dynamic operator: its demo.dynamic run, and the GPU tests that validate it
        return {"dynamic", "tests"} <= only and named

    return in_full


def pin(out: Path, measured: dict) -> dict:
    """Pin ``measured`` (``measured_errors`` of the measurement in ``out``) in ``MEASURED``;
    the pinned file's content.

    Only what was measured in full is pinned: the operators every feeder of which the
    measurement ran (``measured_in_full``), over the three seeds. Such an operator's entry
    is replaced, or dropped if nothing validated at it any more; every other operator
    keeps the entry it has. A measurement that ran everything therefore replaces the file.
    """
    record = out / RAN
    if not record.exists():
        raise SystemExit(f"{out} does not say what was run (no {RAN}): it pins nothing. "
                         "Measure again with --out.")
    ran = json.loads(record.read_text())
    if sorted(ran["seeds"]) != sorted(SEEDS):
        raise SystemExit(f"{out} ran the seeds {ran['seeds']}; a pinned measurement covers "
                         f"{list(SEEDS)}")
    in_full = measured_in_full(ran)
    pinned = json.loads(MEASURED.read_text()) if MEASURED.exists() else {}
    kept = {operator: entry for operator, entry in pinned.items() if not in_full(operator)}
    fresh = {operator: entry for operator, entry in measured.items() if in_full(operator)}
    partial = sorted(set(measured) - set(fresh))
    content = {**kept, **fresh}
    MEASURED.write_text(json.dumps(content, indent=1, sort_keys=True) + "\n")
    print(f"pinned {len(fresh)} operator(s) measured in full; {len(kept)} keep their entry"
          + (f"; measured in part and not pinned: {', '.join(partial)}" if partial else ""),
          file=sys.stderr)
    return content


# ----------------------------------------------------------------------------- reporting
def _single_row_builder(param):
    from tests.graph.apply_tiling.test_unit_extents import SINGLE_ROW

    return SINGLE_ROW[param][0].__name__


# Tiling test files that validate at more than one operator: test -> the graph builder it
# passes to ``tolerance_for`` (a name, or a function of the test's parameter id). Every
# other file under tests/graph/apply_tiling and tests/reference imports its one builder on
# a line of its own.
SEVERAL_BUILDERS = {
    "tests/graph/apply_tiling/test_reduction_axes.py": {
        "test_unlooped_sum_over_a_front_named_axis_validates":
            "batched_matmul_splitk_combine_graph",
        "test_unlooped_sum_over_a_loops_accumulator_validates": "matmul_rowsum_graph",
    },
    "tests/graph/apply_tiling/test_unit_extents.py": {
        "test_a_loop_over_a_single_row_block_validates": _single_row_builder,
        "test_a_block_that_computes_one_element_validates": "rowsum_graph",
    },
    "tests/graph/apply_tiling/test_rmsnorm_matmul_split_combine.py": {
        "test_apply_tiling_rmsnorm_matmul_split_combine": "rmsnorm_matmul_split_combine_graph",
    },
}


def _test_operators():
    """pytest node id (without the phase) -> the operator whose tolerance it validates at."""
    from tests.graph.handwritten._cases import CASES as handwritten
    from tests.schedule.end2end.test_end2end import CASES as end2end

    def builder_of(path, test, param):
        """The graph builder a tiling test validates at: the one its file imports, or, in a
        file that validates at several, the one that test passes to ``tolerance_for``."""
        if path in SEVERAL_BUILDERS:
            tests = SEVERAL_BUILDERS[path]
            name = test.partition("[")[0]
            assert name in tests, (
                f"{path}::{name} logs a validation, and its file validates at several "
                f"operators: say which one in SEVERAL_BUILDERS")
            builder = tests[name]
            return builder if isinstance(builder, str) else builder(param)
        found = re.search(r"^from tests\._graphs import (\w+)$", (ROOT / path).read_text(), re.M)
        assert found, (
            f"{path} logs a validation and does not import one graph builder on a line of "
            f"its own: say which operator each of its tests validates at in SEVERAL_BUILDERS")
        return found[1]

    def operator(node):
        path, _, test = node.partition("::")
        param = (re.search(r"\[(.*)\]$", test) or [None, None])[1]
        if path == "tests/schedule/end2end/test_end2end.py":
            return end2end[param][0].__name__
        if path == "tests/graph/handwritten/test_handwritten.py":
            case = handwritten[param]
            return f"dynamic:{param}" if case.out_idx is not None else case.source.__name__
        if path in ("tests/graph/handwritten/test_empty_request.py",
                    "tests/graph/handwritten/test_splitkv_decode.py",
                    "tests/graph/handwritten/test_paged_exact_softmax.py"):
            # a second parameter (the combine step of a split pair) leads the id: "2-<name>"
            name = re.sub(r"^\d+-", "", param or "")
            return f"dynamic:{name or 'gqa_paged_decode_h32_kv8_d128_ps1'}"
        if path.endswith("test_gqa_paged_decode_h32_kv8_d128_ps1.py"):
            return "dynamic:gqa_paged_decode_h32_kv8_d128_ps1"
        if path == "tests/frontend_dynamic/test_paged_pipeline.py":
            return f"dynamic:gqa_paged_decode_h32_kv8_d128_ps{param}"
        if path.startswith(("tests/graph/apply_tiling/", "tests/reference/")):
            return builder_of(path, test, param)
        if path == "tests/runtime/test_run_candidates.py":
            return "matmul_graph"
        if path == "tests/pipeline_api/test_e2e.py":
            return "matmul"
        if path == "tests/pipeline_api/test_split_attention.py":
            return "attention"
        if path == "tests/pipeline_api/test_exact_split_attention.py":
            return "exact_split_attention"
        if path == "tests/frontend/test_exact_softmax.py":
            if "middle_axis" in test:
                return "softmax_mid"
            return "softmax" if "standalone" in test else "exact_attention"
        return node

    return operator


def load(out: Path) -> dict:
    """operator -> output -> list of (dtype, absolute part?, rel, abs, kind, seed).

    An output is its place among the outputs of its kernel that have no boundary role, or
    the role of one that has: a split partial's ``"shift"`` is one output of the operator,
    wherever the kernel returns it, and the sums next to it are numbered as in a partial
    without it."""
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
                        or "test_without_the_running_max" in node \
                        or "test_gpu_unstabilized_" in node:
                    # comparator unit tests and deliberate failures (the last two run
                    # kernels without the exact softmax: an operator's kernel as it was
                    # before, and a dense program's tilings around a plain exp)
                    continue
                context = "test:" + operator_of(node)
                context = context.replace("test:dynamic:", "dynamic:")
            # a dense program on one of its adversarial softmax cases, or a dynamic
            # operator on one of its adversarial softmax workloads
            hard = is_adversarial(context, record.get("what", ""))
            place = 0
            for o in record["outputs"]:
                # Every output is judged by its row-relative error, whatever bound it was
                # validated at when this was logged; "absolute" only records that the
                # operator declares an absolute part as well (ATOL_FLOOR).
                kind = ADVERSARIAL_CASE if hard and o["kind"] != "non_finite" else o["kind"]
                key = o.get("role") or place
                place += not o.get("role")
                table[context][key].append(
                    (o["dtype"], o["atol"] > 0, o["rel"], o["abs"], kind, seed))
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
        # numbered outputs first, then the ones under a role
        for i, entries in sorted(table[context].items(),
                                 key=lambda item: (isinstance(item[0], str), item[0])):
            dtype = entries[0][0]
            absolute = any(e[1] for e in entries)
            counts = defaultdict(int)
            for _, _, rel, _, kind, _ in entries:
                counts[_bucket(rel, kind)] += 1
            measured = [(rel, abs_, kind) for _, _, rel, abs_, kind, _ in entries
                        if rel is not None and kind != "non_finite"]
            limit = valid_limit(dtype)
            # Adversarial comparisons the rtol has to cover: like any other, one within the
            # floor alone is inside the bound whatever the rtol.
            hard = [rel if feeds_rtol(absolute, abs_) else 0.0
                    for rel, abs_, kind in measured if kind == ADVERSARIAL_CASE]
            measured = [(rel, abs_) for rel, abs_, kind in measured if kind != ADVERSARIAL_CASE]
            valid = [(rel, abs_) for rel, abs_ in measured if rel <= limit]
            feeding = [rel for rel, abs_ in valid if feeds_rtol(absolute, abs_)]
            # No valid comparison at all: nothing was measured. Valid ones that the floor
            # covers entirely: nothing the rtol has to cover, which is an error of 0.
            largest = max(feeding, default=0.0 if valid else None)
            yield {
                "operator": context, "output": i, "dtype": dtype, "absolute": absolute,
                "seeds": sorted({e[5] for e in entries}), "n": len(entries),
                "histogram": {b: counts[b] for b in BUCKETS if counts[b]},
                "largest_valid": largest,
                "largest_valid_abs": max((abs_ for _, abs_ in valid), default=None),
                "ambiguous": sorted(rel for rel, _ in measured if limit < rel < WRONG),
                "within_floor": len(valid) - len(feeding),
                # an operator's passes on adversarial inputs: counted, no input to rtol
                "adversarial": len(hard), "largest_adversarial": max(hard, default=None),
                "rtol": None if largest is None else recommend(largest, dtype),
                "atol": ATOL_FLOOR if absolute else 0.0,
            }


def measured_errors(table) -> dict:
    """operator -> per output ``{dtype, rel, abs, absolute, seeds, validated}``: the largest
    clearly-valid error of each output, which is all the rule needs (``MEASURED``). The
    numbered outputs come first, in order; an output under a boundary role follows, with
    its ``role``.

    ``rel`` is taken over the comparisons that feed the rtol (``feeds_rtol``, and for a
    dense program or a dynamic operator its normal inputs only: ``ADVERSARIAL_CASE``); ``abs``
    over the valid comparisons on normal inputs and ``validated`` over all of them."""
    out = defaultdict(list)
    for r in rows(table):
        role = r["output"] if isinstance(r["output"], str) else None
        assert role or r["output"] == len(out[r["operator"]]), r
        out[r["operator"]].append({
            "dtype": r["dtype"], "rel": r["largest_valid"], "abs": r["largest_valid_abs"],
            "absolute": r["absolute"], "seeds": len(r["seeds"]), "validated": r["n"],
            **({"role": role} if role else {}),
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
        if r["within_floor"]:
            tol += f" ({r['within_floor']} within the floor alone)"
        if r["adversarial"]:
            over = r["rtol"] is None or r["largest_adversarial"] > r["rtol"]
            tol += (f" ({r['adversarial']} on adversarial inputs, largest "
                    f"{r['largest_adversarial']:.2e}: "
                    + ("OVER THE BOUND, a finding" if over else "held to the bound, no input to it")
                    + ")")
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
                    help=f"also pin what was measured in full in tests/{MEASURED.name}, which "
                         "the declaration tests hold every declared tolerance to")
    args = ap.parse_args(argv)
    out = args.report or args.out
    if out is None:
        ap.error("give --out DIR to measure or --report DIR")
    if args.report is None:
        only = args.only.split(",")
        if not set(only) <= set(KINDS):
            ap.error(f"--only takes {', '.join(KINDS)}, got {args.only!r}")
        measure(out, args.gpus.split(","), args.seeds.split(","), only,
                args.names.split(",") if args.names else None)
    table = load(out)
    text = markdown(table)
    (out / "tolerances.md").write_text(text + "\n")
    measured = measured_errors(table)
    (out / MEASURED.name).write_text(json.dumps(measured, indent=1, sort_keys=True) + "\n")
    if args.pin:
        pin(out, measured)
    print(text)


if __name__ == "__main__":
    main()
