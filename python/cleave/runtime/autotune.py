"""Pick the best of several candidate kernels for one dynamic (paged/ragged/MLA) operator.

A paged operator's reference is its captured workloads (real inputs + references), not its
dense graph, so the caller supplies a workload factory; each workload is built once and
shared by every candidate, in this process. Timing is CUDA-graph replay (these kernels run
2-6us, under the Python launch cost); it ranks candidates against each other and overstates
speedups vs other implementations by ~10-15%.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .validate import DEFAULT_TOLERANCE, Tolerance
from .gpu import _gpu_exec_guard
from .timer import capture_launch_graph, launch_rounds, replay_rounds


def ramp(kernel, args, seconds: float) -> None:
    """Run the kernel for ``seconds`` first: the first process to measure reads high
    (host-side warm-up: CPU governor, page cache, the launch path)."""
    import time

    import torch

    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(50):
            kernel(*args)
        torch.cuda.synchronize()


def time_replay(kernel, args, *, iters: int, repeats: int) -> float | None:
    """Microseconds of GPU time per launch from replaying a captured graph (min over
    ``repeats``), or None when capture is unavailable."""
    graph = capture_launch_graph(kernel, args, REPLAY_BATCH)
    if graph is None:
        return None
    rounds = replay_rounds(graph, iters, repeats=repeats, warmup=3)
    return min(ms / (iters * REPLAY_BATCH) * 1e3 for ms in rounds)


# How many launches one captured graph holds. Enough that replay overhead is amortised.
REPLAY_BATCH = 20

# Relative margin below which this harness cannot order two candidates (the measured spread
# of identical candidates).
TIE_BAND = 0.005


def workload_key(axes) -> str:
    """A stable name for a workload, so inserting one does not re-map every baseline."""
    return "wl_" + "_".join(str(int(a)) for a in axes)


def sampled(indices, keep: int):
    """``keep`` workloads spread across ``indices`` (a prefix would sample only the smallest,
    launch-bound shapes); all of them for ``keep=0``."""
    assert keep >= 0, f"keep must not be negative, got {keep}"
    if not keep or keep >= len(indices):
        return list(indices)
    step = (len(indices) - 1) / (keep - 1) if keep > 1 else 0
    return sorted({indices[round(i * step)] for i in range(keep)})


def geomean(values) -> float:
    values = [v for v in values if v > 0]
    return math.exp(statistics.fmean(math.log(v) for v in values))


@dataclass
class WorkloadCandidate:
    """One candidate kernel for an operator, and what measuring it found out."""

    label: str
    source: str
    origin: str = "searched"  # for the report: searched, derived split-K, tuned (reference)
    tuned: bool = False  # it is the operator's tuned schedule
    eligible: bool = True  # False: measured and reported only, never selected
    split: int | None = None  # split-KV factor of a partial+combine pair
    tiling: object = None  # the dense tiling read off the searched graph, if any
    kernel: object = None
    checked: int = 0
    failure: str | None = None  # first wrong/failed workload; set means excluded
    latency_us: dict = field(default_factory=dict)
    used_replay: bool = True

    @property
    def correct(self) -> bool:
        return self.failure is None

    @property
    def geomean_us(self) -> float:
        timings = [v for v in self.latency_us.values() if v > 0]
        return geomean(timings) if timings else float("inf")


def time_kernel(
    kernel,
    args,
    *,
    warmup: int = 50,
    iters: int = 20,
    repeats: int = 5,
    budget_us: float = 40_000.0,
) -> tuple:
    """``(us per launch, used graph replay)``. ``budget_us`` caps the work per measurement
    by shrinking ``iters`` after a pilot round (workloads span three orders of magnitude)."""
    import torch

    for _ in range(warmup):
        kernel(*args)
    torch.cuda.synchronize()

    if budget_us != float("inf") and iters > 1:
        pilot = time_replay(kernel, args, iters=1, repeats=1)
        if pilot is not None and pilot > 0:
            affordable = int(budget_us // (REPLAY_BATCH * pilot))
            iters = max(1, min(iters, affordable))

    elapsed = time_replay(kernel, args, iters=iters, repeats=repeats)
    if elapsed is not None:
        return elapsed, True

    # No capture: a launch loop (flagged, it measures dispatch too), budgeted by one timing.
    if budget_us != float("inf") and iters > 1:
        import time as _time

        start = _time.perf_counter()
        kernel(*args)
        torch.cuda.synchronize()
        one_us = (_time.perf_counter() - start) * 1e6
        if one_us > 0:
            iters = max(1, min(iters, int(budget_us // (REPLAY_BATCH * one_us)) or 1))
    rounds = launch_rounds(kernel, args, iters * REPLAY_BATCH, repeats=repeats)
    return min(ms / (iters * REPLAY_BATCH) * 1e3 for ms in rounds), False


def _assert_context_survived(label: str, cause: BaseException) -> None:
    """Raise if a candidate's failure left the CUDA context unusable."""
    import torch

    try:
        torch.zeros(1, device="cuda").add_(1)
        torch.cuda.synchronize()
    except Exception as probe:  # noqa: BLE001
        raise RuntimeError(
            f"{label} left the CUDA context unusable ({type(cause).__name__}: "
            f"{str(cause).strip().splitlines()[0][:120] if str(cause).strip() else ''}), "
            "so no later candidate can be judged; re-run without it"
        ) from probe


def _compiled(source: str):
    from .compile import _compile, _load_main_kernel_from_code

    # out_idx=None: dynamic kernels take their outputs as parameters (caller-owned buffers).
    return with_scratch(_compile(_load_main_kernel_from_code(source), None))


# Split-KV partial buffers: `Partial_<name>_split` / `Partial_<name>_combine`, one tensor.
_PARTIAL_PREFIX = "Partial_"
_PARTIAL_SUFFIXES = ("_split", "_combine")


def with_scratch(kernel):
    """``kernel`` callable as ``(*inputs, *outputs)``, owning any split-KV partial buffers
    (allocated once per shape, symbolic extents bound from the inputs)."""
    scratch_params = _scratch_params(kernel)
    if not scratch_params:
        return kernel
    import torch

    from tvm import tirx as tir

    adapter = kernel.adapter
    params = list(adapter.params)
    dyn = getattr(adapter, "dynamic_symbolic_map", {}) or {}
    n_params = len(params)
    n_scratch = len(scratch_params)
    scratch_idx = {idx for idx, _, _ in scratch_params}
    # Every symbol a partial's shape uses, and which (input index, dim) binds it.
    binders = {}
    for key, entry in dyn.items():
        buffer_idx, shape_idx = entry[0], entry[1]
        if buffer_idx < n_params and buffer_idx not in scratch_idx:
            binders.setdefault(str(key), (buffer_idx, shape_idx))
    cache = {}

    def _shape_of(param, user_args):
        shape = []
        for dim in param.shape:
            if isinstance(dim, tir.IntImm):
                shape.append(int(dim.value))
            elif isinstance(dim, tir.Var):
                buffer_idx, shape_idx = binders[str(dim)]
                shape.append(int(user_args[buffer_idx].shape[shape_idx]))
            else:
                shape.append(int(dim))
        return tuple(shape)

    def call(*args):
        assert len(args) == n_params - n_scratch, (
            f"expected {n_params - n_scratch} arguments (the operator's inputs and "
            f"outputs), got {len(args)}"
        )
        full, user = [], iter(args)
        for idx, param in enumerate(params):
            if idx in scratch_idx:
                full.append(None)
            else:
                full.append(next(user))
        for idx, base, dtype in scratch_params:
            shape = _shape_of(params[idx], full)
            key = (base, shape)
            if key not in cache:
                device = next(a for a in full if isinstance(a, torch.Tensor)).device
                cache[key] = torch.empty(shape, dtype=dtype, device=device)
            full[idx] = cache[key]
        return kernel(*full)

    call.kernel = kernel
    call.scratch_params = scratch_params
    return call


def param_index(kernel, buffer_name: str) -> int:
    """Position of ``buffer_name`` among the caller arguments (scratch partials excluded)."""
    inner = getattr(kernel, "kernel", kernel)
    prim_func = inner.adapter.prim_func
    scratch = {idx for idx, _, _ in _scratch_params(inner)}
    position = 0
    for idx, param in enumerate(prim_func.params):
        if idx in scratch:
            continue
        buffer = prim_func.buffer_map.get(param)
        if buffer is not None and buffer.name == buffer_name:
            return position
        position += 1
    raise KeyError(f"{buffer_name!r} is not a parameter of this kernel")


def _scratch_params(kernel):
    """``(param index, base name, torch dtype)`` for every partial buffer of ``kernel``."""
    from .reference import torch_dtype

    adapter = kernel.adapter
    prim_func = adapter.prim_func
    found = []
    for idx, param in enumerate(prim_func.params):
        buffer = prim_func.buffer_map.get(param)
        if buffer is None or not buffer.name.startswith(_PARTIAL_PREFIX):
            continue
        base = buffer.name
        for suffix in _PARTIAL_SUFFIXES:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        found.append((idx, base, torch_dtype(str(buffer.dtype))))
    return found


def compile_and_validate_over_workloads(
    candidates: Sequence[WorkloadCandidate],
    workloads_fn: Callable[[], Iterable[tuple]],
    success_info: list,
    errors: list,
    *,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
) -> None:
    """Compile every candidate and check it on every workload; never raises per candidate.

    ``success_info`` gets the candidates correct everywhere, ``errors`` ``(label, exception)``.
    """
    import torch

    seen_workloads = 0
    live = []
    for candidate in candidates:
        try:
            candidate.kernel = _compiled(candidate.source)
            live.append(candidate)
        except Exception as error:  # noqa: BLE001 - a search reports, it does not stop
            first = str(error).strip().splitlines()
            candidate.failure = f"compile: {type(error).__name__}" + (
                f": {first[0][:80]}" if first else ""
            )
            errors.append((candidate.label, error))

    stream = workloads_fn()
    while True:
        # Inside the guard: building a workload is what allocates (tens of GiB).
        with _gpu_exec_guard():
            try:
                key, inputs, reference = next(stream)
            except StopIteration:
                break
            seen_workloads += 1
            remaining = [c for c in live if c.correct]
            if not remaining:
                del inputs, reference
                break
            outputs = [torch.empty_like(tensor) for tensor in reference]
            for candidate in remaining:
                # NaN-poisoned per candidate: unwritten elements must fail (prefill references
                # are mostly zeros, so zeroed buffers would pass a kernel that wrote nothing).
                for tensor in outputs:
                    tensor.fill_(float("nan"))
                try:
                    candidate.kernel(*inputs, *outputs)
                    torch.cuda.synchronize()
                    for got, want in zip(outputs, reference):
                        torch.testing.assert_close(
                            got.float(),
                            want.float(),
                            rtol=tolerance.rtol,
                            atol=tolerance.atol,
                            equal_nan=tolerance.equal_nan,
                        )
                except Exception as error:  # noqa: BLE001
                    first = str(error).strip().splitlines()
                    candidate.failure = (
                        f"{key}: {type(error).__name__}"
                        + (f": {first[0][:80]}" if first else "")
                    )
                    errors.append((candidate.label, error))
                    # A poisoned context would blame every later candidate: stop.
                    _assert_context_survived(candidate.label, error)
                else:
                    candidate.checked += 1
        del inputs, reference, outputs
        torch.cuda.empty_cache()

    assert seen_workloads, (
        "the workload factory yielded nothing, so every candidate would be 'correct' on no "
        "evidence at all"
    )
    success_info.extend(c for c in live if c.correct)


def benchmark_over_workloads(
    success_info: Sequence[WorkloadCandidate],
    workloads_fn: Callable[[], Iterable[tuple]],
    *,
    warmup: int = 50,
    iters: int = 20,
    repeats: int = 5,
    ramp_seconds: float = 1.5,
    budget_us: float = 40_000.0,
) -> tuple:
    """Time every validated candidate on every workload, filling ``latency_us`` in place.

    Returns ``(best, best_geomean_us, best_idx)`` (microseconds).
    """
    import torch

    if not success_info:
        raise ValueError("No correct candidates to benchmark.")

    ramped = False
    for key, inputs, reference in workloads_fn():
        outputs = [torch.empty_like(tensor) for tensor in reference]
        args = (*inputs, *outputs)
        with _gpu_exec_guard():
            if not ramped:
                ramp(success_info[0].kernel, args, ramp_seconds)
                ramped = True
            # One discarded measurement per candidate first: otherwise the first candidate
            # timed on a workload reads ~1.6% low.
            for candidate in success_info:
                time_kernel(
                    candidate.kernel,
                    args,
                    warmup=warmup,
                    iters=1,
                    repeats=1,
                    budget_us=budget_us,
                )
            for candidate in success_info:
                elapsed, used_replay = time_kernel(
                    candidate.kernel,
                    args,
                    warmup=warmup,
                    iters=iters,
                    repeats=repeats,
                    budget_us=budget_us,
                )
                candidate.latency_us[key] = elapsed
                candidate.used_replay = candidate.used_replay and used_replay
        del inputs, reference, outputs, args
        torch.cuda.empty_cache()

    timed = [c for c in success_info if c.latency_us]
    assert timed, "no candidate was timed; the workload factory yielded nothing"
    best = pick_best(rank(timed))
    # By identity: equal-field dataclasses compare equal.
    return best, best.geomean_us, next(
        i for i, c in enumerate(success_info) if c is best
    )


def tie_band_members(ranked, *, tie_band: float = TIE_BAND) -> list:
    """The candidates this harness cannot distinguish from the fastest one."""
    if not ranked:
        return []
    return [c for score, c in ranked if score / ranked[0][0] - 1 <= tie_band]


def pick_best(ranked, *, tie_band: float = TIE_BAND):
    """The fastest *eligible* candidate (reference-only ones are ranked, never picked)."""
    if not ranked:
        raise ValueError("No correct candidates to pick from.")
    eligible = [c for _, c in ranked if c.eligible]
    if not eligible:
        raise ValueError(
            "No eligible candidate: every correct kernel was entered for reference only."
        )
    return eligible[0]


def rank(success_info: Sequence[WorkloadCandidate]) -> list:
    """``(geomean_us, candidate)`` fastest first."""
    return sorted(((c.geomean_us, c) for c in success_info), key=lambda pair: pair[0])


def format_ranking(
    success_info: Sequence[WorkloadCandidate],
    rejected: Sequence[WorkloadCandidate] = (),
    *,
    tie_band: float = TIE_BAND,
) -> str:
    """The candidate table, the winner, and every reason to distrust the winner."""
    ranked = rank(success_info)
    if not ranked:
        return "  no correct candidate to rank"
    best_geomean = ranked[0][0]
    eligible = [c for _, c in ranked if c.eligible]
    best = eligible[0] if eligible else ranked[0][1]  # the table still prints without a winner
    keys = list(best.latency_us)

    width = max((len(c.label) for c in list(success_info) + list(rejected)), default=10)
    head = f"  {'#':>2}  {'candidate':{width}}  {'origin':17}  {'correct':>8}"
    head += "".join(f"  {k[-14:]:>14}" for k in keys)
    head += f"  {'geomean':>9}  {'vs #1':>8}"
    lines = [head]
    for index, (score, candidate) in enumerate(ranked, start=1):
        row = (
            f"  {index:>2}  {candidate.label:{width}}  {candidate.origin:17}"
            f"  {str(candidate.checked):>8}"
        )
        for key in keys:
            cell = candidate.latency_us.get(key)
            marker = "" if candidate.used_replay else "*"
            row += f"  {('%.2f%s' % (cell, marker)) if cell else '-':>14}"
        row += f"  {score:9.2f}  {score / best_geomean:7.3f}x"
        lines.append(row)
    for candidate in rejected:
        lines.append(
            f"   -  {candidate.label:{width}}  {candidate.origin:17}  "
            f"{'EXCLUDED':>8}  {candidate.failure}"
        )

    # Does the winner actually win everywhere, or only on aggregate?
    per_workload = {
        key: min(success_info, key=lambda c: c.latency_us.get(key, float("inf")))
        for key in keys
    }
    losers = {k: c.label for k, c in per_workload.items() if c is not best}
    lines.append(
        f"  winner: {best.label} ({best.origin}), geomean {best.geomean_us:.2f}us"
    )
    if losers:
        lines.append(
            f"  does NOT dominate: {len(losers)} of {len(keys)} timed workloads go to "
            + ", ".join(sorted(set(losers.values())))
        )
    else:
        lines.append(f"  dominates all {len(keys)} timed workloads")
    tied = tie_band_members(ranked, tie_band=tie_band)
    if len(tied) > 1:
        lines.append(
            f"  within {tie_band:.0%} of the fastest (a tie for this harness): "
            + ", ".join(c.label for c in tied)
        )
        if any(not c.eligible for c in tied):
            lines.append(
                "  (a reference candidate is inside that band; it is reported, not selected)"
            )
    tuned = [c for c in success_info if c.tuned]
    if tuned:
        position = next(
            i for i, (_, c) in enumerate(ranked, start=1) if c is tuned[0]
        )
        # gap > 1 means the tuned point took LONGER, i.e. the winner is that much faster.
        gap = tuned[0].geomean_us / best.geomean_us
        lines.append(
            f"  tuned point ({'eligible' if tuned[0].eligible else 'reference only'}): "
            f"rank {position} of {len(ranked)}, {tuned[0].geomean_us:.2f}us -- the winner is "
            + (
                f"{gap:.3f}x faster"
                if gap > 1
                else f"{1 / gap:.3f}x slower"
                if gap < 1
                else "the same speed"
            )
        )
    if any(not c.used_replay for c in success_info):
        lines.append("  * launch-loop reading: measures dispatch too, not comparable")
    lines.append(
        "  NB: graph replay on fixed buffers, so these overstate a speedup by 10-15% "
        "against SOL/CUPTI; they rank candidates, they are not vs-FlashInfer numbers"
    )
    return "\n".join(lines)
