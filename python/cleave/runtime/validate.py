"""Tolerances, the comparator, the on-disk reference a worker validates against, and the check."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

from ..core.errors import LaunchCrash, NonFiniteOutput, OutputMismatch, SpecError
from ..core.settings import Settings
from .gpu import _gpu_exec_guard
from .reference import (
    build_inputs_and_reference,
    order_inputs_for_kernel,
    reconcile_output_layout,
)
from .timer import mean_launch_ms


@dataclass(frozen=True)
class Tolerance:
    """The bound one kernel output is held to: ``|actual - ref| <= rtol * scale + atol``.

    ``scale`` is the largest finite ``|ref|`` of the entry's *row*: unit axes are dropped and
    the row is the last remaining axis (a vector is one row). ``rtol`` is therefore relative
    to the row, not to the entry, and an all-zero row demands ``|actual| <= atol``. ``atol``
    is absolute: the part of a bound that does not scale with the row. An operator's
    measured error belongs in ``rtol``; ``atol`` is 0, or a small fixed floor where a row of
    zeros is a legitimate value that cannot be reproduced bit for bit (a log-sum-exp of 0).

    There is no default bound anywhere: every operator declares its own, from a measured
    error (see :func:`compare_outputs`). A sequence of tolerances declares one per output.
    """

    rtol: float
    atol: float = 0.0

    def __post_init__(self):
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise SpecError(f"Tolerance.{name} must be a finite number >= 0, got {value!r}")


@dataclass(frozen=True)
class MeasuredError:
    """What comparing one output against its reference measured (finite reference entries).

    ``rel`` is the largest ``|actual - ref| / scale`` over the rows whose scale is not zero
    (the quantity ``Tolerance.rtol`` bounds); ``abs`` the largest ``|actual - ref|``.
    """

    rel: float
    abs: float

    def to_dict(self) -> dict:
        return {"rel": self.rel, "abs": self.abs}


def worst(a, b):
    """Per output, the larger of two measurements (either may be None)."""
    if a is None or b is None:
        return a if b is None else b
    return tuple(MeasuredError(max(x.rel, y.rel), max(x.abs, y.abs)) for x, y in zip(a, b))


def require_tolerance(tolerance, what="this kernel"):
    """``tolerance``, or the error for a kernel nobody declared a bound for."""
    if tolerance is None:
        raise SpecError(
            f"no validation tolerance declared for {what}: there is no default bound; pass "
            "tolerance=Tolerance(rtol=..., atol=...) (one, or one per output), set from the "
            "operator's measured error"
        )
    return tolerance


def tolerances_for(tolerance, n_outputs):
    """One ``Tolerance`` per output from a single one or a sequence of ``n_outputs``."""
    require_tolerance(tolerance)
    if isinstance(tolerance, Tolerance):
        return (tolerance,) * n_outputs
    try:
        tolerances = tuple(tolerance)
    except TypeError:
        tolerances = (tolerance,)
    if not all(isinstance(t, Tolerance) for t in tolerances):
        raise SpecError(f"tolerance must be a Tolerance or a sequence of them, got {tolerance!r}")
    if len(tolerances) != n_outputs:
        raise SpecError(
            f"{len(tolerances)} tolerances were declared for {n_outputs} outputs; declare one "
            "for all outputs or exactly one per output"
        )
    return tolerances


def _compare_one(index, actual, reference, tol):
    """``(measured | None, kind | None, message)`` for one output; never raises on values."""
    import torch

    if tuple(actual.shape) != tuple(reference.shape):
        return None, "mismatch", (
            f"output {index}: shape {tuple(actual.shape)} differs from the reference's "
            f"{tuple(reference.shape)}"
        )
    if actual.dtype != reference.dtype:
        return None, "mismatch", (
            f"output {index}: dtype {actual.dtype} differs from the reference's {reference.dtype}"
        )
    if actual.numel() == 0:
        return MeasuredError(0.0, 0.0), None, ""

    wide = torch.promote_types(actual.dtype, torch.float32)
    act, ref = actual.to(wide), reference.to(wide)
    n_nan_act, n_nan_ref = int(torch.isnan(act).sum()), int(torch.isnan(ref).sum())
    finite = torch.isfinite(ref)
    # A non-finite reference entry is a defined answer (an empty row's -inf lse) and must be
    # matched exactly; an infinity the reference does not have is a defect; a NaN always is.
    unmatched = int(((~finite) & (act != ref) & ~torch.isnan(ref)).sum())
    stray = int((finite & torch.isinf(act)).sum())
    non_finite = ""
    if n_nan_act or n_nan_ref:
        non_finite = (
            f"output {index}: {n_nan_act} NaN in the kernel output, {n_nan_ref} in the "
            f"reference (of {act.numel()} entries)"
        )
    elif unmatched or stray:
        non_finite = (
            f"output {index}: {stray} infinite kernel entries where the reference is finite, "
            f"{unmatched} non-finite reference entries not reproduced (of {act.numel()})"
        )

    # Everything below only looks at the entries that are finite on both sides (all of them
    # where the reference is finite, once the checks above have passed).
    comparable = finite & torch.isfinite(act)
    zero = torch.zeros((), dtype=wide, device=act.device)
    err = torch.where(comparable, (act - ref).abs(), zero)
    mag = torch.where(finite, ref.abs(), zero)
    del act, ref
    # The row scale must not depend on whether a kernel keeps or drops a unit axis.
    dims = [d for d in err.shape if d != 1]
    err, mag = err.reshape(dims), mag.reshape(dims)
    scale = mag.max() if len(dims) <= 1 else mag.amax(dim=-1, keepdim=True)
    del mag
    # No division: a zero scale (an all-zero row) leaves only atol.
    n_bad = int((err > scale * tol.rtol + tol.atol).sum())
    live = scale > 0
    rel = torch.where(live, err / torch.where(live, scale, torch.ones_like(scale)), zero)
    measured = MeasuredError(float(rel.max()), float(err.max()))
    if non_finite:
        # Say whether the kernel is otherwise right: one stray infinity and a kernel that is
        # wrong everywhere are different findings.
        return None, "non_finite", (
            f"{non_finite}; of the {int(comparable.sum())} entries finite on both sides {n_bad} "
            f"are out of tolerance (row-normalized error {measured.rel:.3e})"
        )
    if not n_bad:
        return measured, None, ""
    return measured, "mismatch", (
        f"output {index}: {n_bad} of {err.numel()} entries out of tolerance; max |actual-ref| "
        f"{measured.abs:.3e}, row-normalized {measured.rel:.3e}; bound rtol={tol.rtol:g} * "
        f"rowmax|ref| + atol={tol.atol:g}"
    )


def _log_validation(what, actual, tolerances, results):
    path = Settings.from_env().validation_log
    if path is None:
        return
    record = {
        "context": os.environ.get("CLEAVE_VALIDATION_CONTEXT")
        or os.environ.get("PYTEST_CURRENT_TEST", ""),
        "what": what,
        "kind": next((kind for _, kind, _ in results if kind), None),
        "outputs": [
            {
                "shape": list(act.shape), "dtype": str(act.dtype).removeprefix("torch."),
                "rtol": tol.rtol, "atol": tol.atol, "kind": kind,
                "rel": None if measured is None else measured.rel,
                "abs": None if measured is None else measured.abs,
            }
            for act, tol, (measured, kind, _) in zip(actual, tolerances, results)
        ],
    }
    line = (json.dumps(record, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    # One O_APPEND write per record: lines from concurrent workers do not interleave.
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def compare_outputs(actual, reference, tolerance, *, what: str = ""):
    """Hold every kernel output to its reference; the one comparison every validation uses.

    Returns one :class:`MeasuredError` per output. Raises ``OutputMismatch`` when a finite
    output is outside its :class:`Tolerance` (or has the wrong shape, dtype or count) and
    ``NonFiniteOutput`` for any NaN, an infinity the reference does not have, or a
    non-finite reference entry that is not reproduced exactly; non-finite entries take no
    part in the row scale. ``tolerance`` is one ``Tolerance`` or one per output. ``what``
    names the kernel in the message and in the ``CLEAVE_VALIDATION_LOG`` record.
    """
    actual, reference = list(actual), list(reference)
    prefix = f"{what}: " if what else ""
    if len(actual) != len(reference):
        raise OutputMismatch(
            f"{prefix}the kernel returned {len(actual)} outputs but the reference has "
            f"{len(reference)}"
        )
    tolerances = tolerances_for(tolerance, len(actual))
    results = [
        _compare_one(i, act, ref, tol)
        for i, (act, ref, tol) in enumerate(zip(actual, reference, tolerances))
    ]
    _log_validation(what, actual, tolerances, results)
    measured = tuple(m for m, _, _ in results)
    for kind, error_cls in (("non_finite", NonFiniteOutput), ("mismatch", OutputMismatch)):
        messages = [message for _, k, message in results if k == kind]
        if messages:
            error = error_cls(prefix + "; ".join(messages))
            error.measured = measured if all(m is not None for m in measured) else None
            raise error
    return measured


def launch(kernel, args):
    """``kernel(*args)``; a fault it raises, now or at the next sync, is a ``LaunchCrash``."""
    import torch

    try:
        result = kernel(*args)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - typed, so it is not counted as a mismatch
        first = (str(exc).strip().splitlines() or [""])[0][:200]
        raise LaunchCrash(f"kernel launch raised {type(exc).__name__}: {first}") from exc
    return result


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


def _reference_cases(graph, dtype, distributions, derive=None):
    """``([(distribution, inputs_by_name, reference), ...], out_idx)``, one case per name.

    ``derive`` recomputes the declared inputs of each case after its distribution is applied
    (``cleave.runtime.reference.declared_inputs_hook``); None draws every input.
    """
    cases, out_idx = [], None
    for distribution in distributions:
        inputs_by_name, reference, out_idx = build_inputs_and_reference(
            graph, dtype, distribution=distribution, derive=derive
        )
        cases.append((distribution, inputs_by_name, list(reference)))
    if not cases:
        raise SpecError("validation needs at least one input distribution")
    return cases, out_idx


def publish_reference(reference_cache_dir, graph, dtype, distributions=("normal",), derive=None):
    """Build the (inputs, reference) cases once in the parent and persist them to disk.

    The cleave ``graph`` wraps TVM objects that need not cross the ``spawn``
    boundary; every candidate worker instead loads the plain tensors written here.
    There is one case per input distribution (``cleave.runtime.inputs.DISTRIBUTIONS``);
    a kernel is validated on all of them and timed on the first. ``derive``: the declared
    inputs, see :func:`_reference_cases`.
    GPU work is serialized by ``_gpu_exec_guard``. Returns ``(out_idx, io_bytes)``
    where ``io_bytes`` is the input+output footprint used to size parallelism.
    """
    import torch

    with _gpu_exec_guard():
        cases, out_idx = _reference_cases(graph, dtype, distributions, derive)
        path = _reference_cache_file(reference_cache_dir)
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        os.makedirs(reference_cache_dir, exist_ok=True)
        torch.save(
            {
                "cases": [
                    {"distribution": name, "inputs_by_name": inputs, "reference": reference}
                    for name, inputs, reference in cases
                ]
            },
            tmp_path,
        )
        os.replace(tmp_path, path)
        io_bytes = sum(
            _live_bytes(inputs.values()) + _live_bytes(reference)
            for _, inputs, reference in cases
        )
        del cases
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
    return [
        (case["distribution"], dict(case["inputs_by_name"]), list(case["reference"]))
        for case in payload["cases"]
    ]


def _as_list(reference):
    return list(reference) if isinstance(reference, (tuple, list)) else [reference]


def _validate_kernel_outputs(kernel, inputs, reference, *, tolerance, what: str = ""):
    """Run ``kernel(*inputs)`` and hold it to ``reference``; the measured error per output.

    Shared comparison core: reconciles each reference to the kernel's physical output
    shape and compares with :func:`compare_outputs` at the operator's declared
    ``tolerance`` (there is no default). Must run inside a ``_gpu_exec_guard``.
    """
    import torch

    require_tolerance(tolerance)
    actual = _as_list(launch(kernel, inputs))
    if len(actual) == len(reference):
        reference = [
            reconcile_output_layout(ref, act.shape)
            for ref, act in zip(reference, actual)
        ]
    measured = compare_outputs(actual, reference, tolerance, what=what)
    del actual, reference
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return measured


def _check_kernel(
    kernel, *, graph=None, dtype=None, reference_cache_dir=None,
    tolerance=None, benchmark: BenchOptions | None = None, distributions=("normal",),
    derive=None,
):
    """Validate ``kernel`` against the graph reference, then optionally time it.

    Either ``reference_cache_dir`` (worker mode: load the parent-built tensors) or
    ``graph`` + ``dtype`` (standalone mode: build the reference in-process, one case per
    name in ``distributions``, declared inputs recomputed by ``derive``) must be provided.
    A missing ``tolerance`` is an error.
    Timing happens under the same GPU lock hold as validation, on the first case's inputs.
    Returns ``(measured error per output, mean latency in ms or None)``.
    """
    with _stage("validate"):
        require_tolerance(tolerance)
    with _gpu_exec_guard():
        with _stage("validate"):
            if reference_cache_dir is not None:
                cases = _load_reference(reference_cache_dir)
            else:
                cases, _ = _reference_cases(graph, dtype, distributions, derive)

            measured, timing_inputs = None, None
            for distribution, inputs_by_name, reference in cases:
                inputs = order_inputs_for_kernel(inputs_by_name, kernel)
                if timing_inputs is None:
                    timing_inputs = inputs
                measured = worst(measured, _validate_kernel_outputs(
                    kernel, inputs, reference, tolerance=tolerance,
                    what="" if distribution == "normal" else f"{distribution} inputs",
                ))
        if benchmark is None:
            return measured, None
        with _stage("bench"):
            return measured, benchmark_compiled_kernel(
                kernel, timing_inputs, warmup_iters=benchmark.warmup,
                profile_iters=benchmark.iters,
            )


def _run_and_validate_kernel(
    kernel, *, graph=None, dtype=None, reference_cache_dir=None, tolerance=None,
):
    _check_kernel(
        kernel, graph=graph, dtype=dtype, reference_cache_dir=reference_cache_dir,
        tolerance=tolerance,
    )
    return kernel.get_kernel_source()
