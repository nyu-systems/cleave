"""Tolerances, the on-disk reference a worker validates against, and the check itself."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import os
from pathlib import Path

from .gpu import _gpu_exec_guard
from .reference import (
    build_inputs_and_reference,
    order_inputs_for_kernel,
    reconcile_output_layout,
)
from .timer import mean_launch_ms


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float
    equal_nan: bool = False


# The loose default bound; per-operator tolerances tighten it.
DEFAULT_TOLERANCE = Tolerance(rtol=0.5, atol=0.5)


def _live_bytes(tensors):
    return sum(t.element_size() * t.nelement() for t in tensors)


@dataclass(frozen=True)
class BenchOptions:
    warmup: int = 16
    iters: int = 1000


@contextlib.contextmanager
def _stage(name):
    """Tag an escaping exception with the pipeline stage it came from."""
    try:
        yield
    except BaseException as exc:
        if getattr(exc, "_cleave_stage", None) is None:
            try:
                exc._cleave_stage = name
            except Exception:  # noqa: BLE001 - some builtins reject attributes
                pass
        raise


# How many launches one captured CUDA graph holds. Replaying the graph in batches
# amortises the replay's own launch cost over many kernel launches.
_BENCHMARK_GRAPH_BATCH = 100


def benchmark_compiled_kernel(
    kernel,
    input_tensors,
    *,
    warmup_iters=0,
    profile_iters=1,
):
    """Mean milliseconds per launch of ``kernel(*input_tensors)``.

    Launches are captured into a CUDA graph and timed by replaying it (see
    :mod:`cleave.runtime.timer`); if capture fails this warns and times a plain launch loop.
    """
    return mean_launch_ms(
        kernel, input_tensors, warmup=warmup_iters, iters=profile_iters,
        batch=_BENCHMARK_GRAPH_BATCH,
    )


_REFERENCE_FILE_NAME = "reference.pt"


def _reference_cache_file(reference_cache_dir):
    return Path(reference_cache_dir) / _REFERENCE_FILE_NAME


def publish_reference(reference_cache_dir, graph, dtype):
    """Build the (inputs, reference) once in the parent and persist them to disk.

    The cleave ``graph`` wraps TVM objects that need not cross the ``spawn``
    boundary; every candidate worker instead loads the plain tensors written here.
    GPU work is serialized by ``_gpu_exec_guard``. Returns ``(out_idx, io_bytes)``
    where ``io_bytes`` is the input+output footprint used to size parallelism.
    """
    import torch

    with _gpu_exec_guard():
        inputs_by_name, reference, out_idx = build_inputs_and_reference(graph, dtype)
        path = _reference_cache_file(reference_cache_dir)
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        os.makedirs(reference_cache_dir, exist_ok=True)
        torch.save(
            {"inputs_by_name": inputs_by_name, "reference": list(reference)}, tmp_path
        )
        os.replace(tmp_path, path)
        io_bytes = _live_bytes(inputs_by_name.values()) + _live_bytes(reference)
        del inputs_by_name, reference
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return out_idx, io_bytes


def _load_reference(reference_cache_dir):
    import torch

    payload = torch.load(
        _reference_cache_file(reference_cache_dir),
        map_location="cuda",
        weights_only=True,
    )
    return dict(payload["inputs_by_name"]), list(payload["reference"])


def _as_list(reference):
    return list(reference) if isinstance(reference, (tuple, list)) else [reference]


def _validate_kernel_outputs(
    kernel, inputs, reference, *, tolerance: Tolerance = DEFAULT_TOLERANCE
):
    """Run ``kernel(*inputs)`` and assert it matches ``reference`` within tolerance.

    Shared comparison core: reconciles each reference to the kernel's physical
    output shape and compares at ``tolerance``, defaulting to the loose
    0.5/0.5 bound. Must run inside a ``_gpu_exec_guard``.
    """
    import torch

    actual = kernel(*inputs)
    actual = list(actual) if isinstance(actual, (tuple, list)) else [actual]
    assert len(actual) == len(reference), (
        f"Kernel returned {len(actual)} outputs but the reference has "
        f"{len(reference)}."
    )
    reference = [
        reconcile_output_layout(ref, act.shape)
        for ref, act in zip(reference, actual)
    ]
    torch.testing.assert_close(
        actual,
        reference,
        rtol=tolerance.rtol,
        atol=tolerance.atol,
        equal_nan=tolerance.equal_nan,
    )
    torch.cuda.synchronize()
    del actual, reference
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _check_kernel(
    kernel, *, graph=None, dtype=None, reference_cache_dir=None,
    tolerance: Tolerance = DEFAULT_TOLERANCE, benchmark: BenchOptions | None = None,
):
    """Validate ``kernel`` against the graph reference, then optionally time it.

    Either ``reference_cache_dir`` (worker mode: load the parent-built tensors) or
    ``graph`` + ``dtype`` (standalone mode: build the reference in-process) must be
    provided. Timing happens under the same GPU lock hold as validation. Returns the
    mean latency in ms, or None without ``benchmark``.
    """
    with _gpu_exec_guard():
        with _stage("validate"):
            if reference_cache_dir is not None:
                inputs_by_name, reference = _load_reference(reference_cache_dir)
            else:
                inputs_by_name, reference, _ = build_inputs_and_reference(graph, dtype)

            inputs = order_inputs_for_kernel(inputs_by_name, kernel)
            _validate_kernel_outputs(kernel, inputs, reference, tolerance=tolerance)
        if benchmark is None:
            return None
        with _stage("bench"):
            return benchmark_compiled_kernel(
                kernel, inputs, warmup_iters=benchmark.warmup,
                profile_iters=benchmark.iters,
            )


def _run_and_validate_kernel(
    kernel, *, graph=None, dtype=None, reference_cache_dir=None,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
):
    _check_kernel(
        kernel, graph=graph, dtype=dtype, reference_cache_dir=reference_cache_dir,
        tolerance=tolerance,
    )
    return kernel.get_kernel_source()
