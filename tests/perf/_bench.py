"""Time the checked-in expected kernels, one workload at a time.

This measures **codegen output only**. The kernel under test is the expected TileLang
module in ``tests/graph/handwritten/expected_kernels/``, compiled and run directly --
no schedule, no tiling search, no superoptimizer. That is deliberate: a regression
here can only come from the emitted kernel changing, which is what the expected-kernel files
pin and what a reviewer can diff.

The numbers are wall clock over a warmed kernel, not SOL/CUPTI time, so they are
comparable only against another run of this same harness on the same GPU. They are
not the operator's speedup against FlashInfer and must not be quoted as one.

Outputs are allocated once, outside the timing loop, and passed in. A kernel compiled
with ``out_idx`` allocates them on every call instead, which costs more than these
kernels take to run and would mask everything this harness exists to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

# The operator table and its workloads are the same ones the numeric tests and the dynamic
# demo use. At module scope because `parametrize` in test_kernel_perf_regression.py needs
# the keys at collection time.
from cleave.dynamic.registry import EXPECTED_KERNELS_DIR as EXPECTED_KERNELS
from cleave.dynamic.registry import OPERATORS, keyed_workloads
# Shared with the dynamic autotune loop; this gate passes `budget_us=inf` to keep the
# recorded baselines' exact settings (the autotune loop caps the work per measurement).
from cleave.runtime.autotune import geomean, ramp, time_kernel


def _compiled_expected_kernel(operator: str):
    """The expected kernel for ``operator``, compiled with the pass configs it declares."""
    from cleave.runtime.compile import _compile, _load_main_kernel_from_code

    source = (EXPECTED_KERNELS / f"{operator}.py").read_text(encoding="utf-8")
    # out_idx=None: every output is already a declared parameter of these kernels, so
    # the caller owns the buffers and no allocation happens per call.
    return _compile(_load_main_kernel_from_code(source), None)


def time_operator(
    operator: str,
    *,
    warmup: int = 50,
    iters: int = 20,
    repeats: int = 5,
    max_workloads: int = 0,
    ramp_seconds: float = 1.5,
) -> dict[str, float]:
    """Microseconds per launch for a sample of ``operator``'s workloads.

    Deliberately keeps ``iters=20`` and no time budget, unlike the autotune loop: the
    recorded baselines and this gate's null distribution (0.999-1.001x) were established
    at that setting, and re-establishing them is a separate piece of work.
    """
    kernel = _compiled_expected_kernel(operator)

    timings: dict[str, float] = {}
    ramped = False
    for key, inputs, reference in keyed_workloads(operator, keep=max_workloads)():
        outputs = [torch.empty_like(tensor) for tensor in reference]
        args = (*inputs, *outputs)
        if not ramped:
            ramp(kernel, args, ramp_seconds)
            ramped = True
        timings[key], _ = time_kernel(
            kernel,
            args,
            warmup=warmup,
            iters=iters,
            repeats=repeats,
            budget_us=float("inf"),
        )
        del inputs, reference, outputs, args
        torch.cuda.empty_cache()
    assert timings, f"{operator} yielded no workloads"
    return timings


def device_name() -> str:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


BASELINES = Path(__file__).resolve().parent / "baselines"


def device_slug(name: str | None = None) -> str:
    """File-name form of a GPU model, e.g. 'NVIDIA H100 80GB HBM3' -> 'h100-80gb-hbm3'."""
    name = (name if name is not None else device_name()).lower()
    if name.startswith("nvidia "):
        name = name[len("nvidia ") :]
    return "-".join(name.split())


def baseline_path(name: str | None = None) -> Path:
    """Where this GPU's recorded latencies live. Only H100 is recorded today."""
    return BASELINES / f"{device_slug(name)}.json"


def measure(operators=None, **kwargs) -> dict:
    """Benchmark every operator and return a JSON-serialisable record."""
    results = {}
    for operator in list(operators or OPERATORS):
        timings = time_operator(operator, **kwargs)
        results[operator] = {
            "workloads": timings,
            "geomean_us": geomean(timings.values()),
        }
    return {"device": device_name(), "operators": results}


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(record: dict, path) -> None:
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
