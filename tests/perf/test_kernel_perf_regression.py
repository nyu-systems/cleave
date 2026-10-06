"""Guard the generated kernels' latency against this GPU's recorded baseline.

Cleave against itself: the kernel under test is the checked-in golden TileLang in
``tests/graph/handwritten/expected_kernels/``, compiled and run directly, and the
reference is a recorded run of the same harness on the same GPU model. Nothing here
compares against another library -- for that, see
``bench/bench_dynamic.py``.

Baselines are per GPU model (``tests/perf/baselines/<model>.json``) and only H100 is
recorded today, so the test skips on any other device rather than comparing latencies
that are not comparable. Record a new one with

    python -c "from tests.perf import _bench; _bench.dump( \\
        _bench.measure(max_workloads=4), _bench.baseline_path())"

on an otherwise idle GPU. There is no recorder script: the ``maint/`` one this docstring
used to point at was removed deliberately, so re-recording is a two-line call rather than
a tool.

It also skips under ``pytest-xdist``: the default suite runs ``-n auto``, and a timing
assertion sharing a GPU with a dozen compile workers would fail for reasons that have
nothing to do with the kernels. Run it on its own:

    pytest tests/perf -n 0 -q

``CLEAVE_PERF_THRESHOLD`` (default 1.25) is the allowed slowdown, ``CLEAVE_PERF_BASELINE``
overrides the record path, and ``CLEAVE_PERF_MAX_WORKLOADS`` (default 4) sets how many
stratified workloads per operator are timed. The threshold is deliberately loose: a stored
record ages against driver and library changes, so it catches a structural regression in
codegen rather than a few percent of drift.
"""

from __future__ import annotations

import os

import pytest

# Before importing `_bench`: it reaches the operator registry through a module that imports
# torch at module scope, so without this the whole file errors at collection instead of
# skipping on a machine that cannot time anything.
pytest.importorskip("torch")

from tests.perf import _bench  # noqa: E402

THRESHOLD = float(os.environ.get("CLEAVE_PERF_THRESHOLD", "1.25"))
MAX_WORKLOADS = int(os.environ.get("CLEAVE_PERF_MAX_WORKLOADS", "4"))


@pytest.fixture(scope="module")
def baseline():
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("perf timing needs the GPU to itself; run 'pytest tests/perf -n 0'")
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to time a kernel")

    device = _bench.device_name()
    path = os.environ.get("CLEAVE_PERF_BASELINE") or _bench.baseline_path(device)
    if not os.path.exists(path):
        pytest.skip(
            f"no recorded latencies for {device!r} ({path}); record one with "
            "_bench.dump(_bench.measure(max_workloads=4), _bench.baseline_path())"
        )
    record = _bench.load(path)
    if record.get("device") != device:
        pytest.skip(
            f"{path} was recorded on {record.get('device')!r}, this GPU is {device!r}"
        )
    return record


# The suite runs --reruns 10; retrying a timing assertion until it passes is
# min-of-eleven, which would let a real regression through most of the time.
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize("operator", sorted(_bench.OPERATORS))
def test_expected_kernel_latency(operator, baseline):
    recorded = baseline["operators"].get(operator)
    if recorded is None:
        pytest.skip(f"{operator} is absent from the baseline record")

    measured = _bench.time_operator(operator, max_workloads=MAX_WORKLOADS)
    current = _bench.geomean(measured.values())
    ratio = current / recorded["geomean_us"]

    # Per-workload detail, so a failure names the workload that moved rather than only
    # the aggregate it moved.
    detail = [
        f"{name} {recorded['workloads'][name]:.2f} -> {measured[name]:.2f}us "
        f"({measured[name] / recorded['workloads'][name]:.3f}x)"
        for name in sorted(measured)
        if recorded["workloads"].get(name)
    ]
    report = (
        f"{operator}: geomean {recorded['geomean_us']:.2f} -> {current:.2f}us "
        f"({ratio:.3f}x)\n  " + "\n  ".join(detail)
    )
    print(report)
    assert ratio <= THRESHOLD, f"slowdown exceeds {THRESHOLD:.3f}x\n{report}"
