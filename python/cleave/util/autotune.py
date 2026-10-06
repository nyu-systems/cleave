"""Pick the best of several candidate kernels for one dynamic operator.

The dynamic (paged/ragged/MLA) counterpart of ``runtime.compile_and_run_candidates_parallel``
and ``runtime.benchmark_graphs``: same shape -- validate every candidate, benchmark the
survivors, return the winner -- against a different kind of reference.

It is a separate module rather than a branch of those two because a paged operator's
reference is not derivable from its dense graph. ``graph_executor`` has no semantics for
``PagedGather``/``MaskedExp`` at all, and the paged kernel's signature (five to seven
tensors, a runtime scale, two outputs) does not match the dense graph's three-and-one. What
stands in for it is the operator's *captured workloads*: real inputs with references
computed by the benchmark's own definition. So the caller supplies a workload factory and
this module never learns which operators exist -- the table lives in the tests, next to the
captured axes it came from.

The other reason is process structure. ``compile_and_run_candidates_parallel`` ships one
``torch.save``'d reference to each spawned worker; here the reference is a *stream* of up to
48 workloads whose largest single case allocates 25 GB, and the whole economy of the loop is
that each workload is built once and shared by every candidate. That does not survive being
serialised per worker, and it does not need to: after the access layer refuses the tilings it
cannot represent, one operator has between one and nine candidates, not a thousand.

Timing is CUDA-graph replay, because these kernels run 2-6us while the Python dispatch in
front of a launch costs ~6.5us -- a launch loop measures the dispatch and reports a 2.2x
difference in kernel cost as no difference at all. The bias that buys: replaying over fixed
buffers keeps addresses and caches warm, which **overstates a speedup by 10-15%** against
SOL/CUPTI, which rotates addresses and reports a median. These numbers rank candidates
against each other, in one process, on one GPU. They are not vs-FlashInfer speedups; see
``bench/bench_dynamic.py`` for that claim.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .runtime import DEFAULT_TOLERANCE, Tolerance, _gpu_exec_guard


def ramp(kernel, args, seconds: float) -> None:
    """Run the kernel for a while before the first measurement.

    The first process to measure otherwise reports inflated latencies. On this H100 the SM
    clock already sits at its ceiling when idle, so the effect is host-side -- CPU
    governor, page cache, and warming the Python/tvm_ffi launch path -- rather than a GPU
    clock ramp, but it is real and it lands entirely on whichever side of an A/B runs
    first.
    """
    import time

    import torch

    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(50):
            kernel(*args)
        torch.cuda.synchronize()


def time_replay(kernel, args, *, iters: int, repeats: int) -> float | None:
    """Microseconds of GPU time per launch, measured by replaying a captured graph.

    A plain launch loop cannot measure these kernels: they run in 2-6us and the Python
    dispatch in front of each launch costs ~6.5us, so a loop reports the dispatch and
    flattens a 2.2x difference in kernel cost to nothing. Capturing once and replaying
    takes Python out of the timed region.

    The usual caveat about CUDA graphs -- that replaying on fixed buffers overstates a
    speedup, because it reuses addresses and keeps caches warm -- is about comparing
    against another implementation. Here both sides of the comparison are the same kernel
    built from different commits and measured the same way, so the bias is common to both
    and cancels in the ratio.

    Returns None when capture is unavailable, so the caller can fall back.
    """
    import torch

    try:
        graph = torch.cuda.CUDAGraph()
        # A capture needs its own stream, and the kernel must have run once already so
        # any lazy allocation happens outside the capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            kernel(*args)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            for _ in range(REPLAY_BATCH):
                kernel(*args)
    except Exception:  # noqa: BLE001 - capture is an optimisation, not a requirement
        return None

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    rounds = []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            graph.replay()
        stop.record()
        torch.cuda.synchronize()
        rounds.append(start.elapsed_time(stop) / (iters * REPLAY_BATCH) * 1e3)
    return min(rounds)


# How many launches one captured graph holds. Enough that replay overhead is amortised.
REPLAY_BATCH = 20

# How close two candidates have to be before this harness stops claiming to order them.
# Set from the measured resolution of the loop below: timing several *identical* candidates
# and reading the spread. That spread used to be ~1.6% and position-dependent, which is why
# `benchmark_over_workloads` now discards one measurement per candidate per workload; with
# that in place the residual is what this band has to cover.
TIE_BAND = 0.005


def workload_key(axes) -> str:
    """A stable name for a workload, so inserting one does not re-map every baseline."""
    return "wl_" + "_".join(str(int(a)) for a in axes)


def sampled(indices, keep: int):
    """``keep`` workloads spread across the captured range, not the first ``keep``.

    ``WORKLOADS`` is ordered smallest-first, so taking a prefix samples only degenerate
    shapes -- one query against one key -- whose cost is entirely launch overhead. A
    stratified sample keeps the small end represented while also timing shapes where the
    GPU is actually busy.
    """
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
    # Which stage produced it, for the report: a candidate the search proposed, one the
    # search proposed that turns out to be the tuned schedule, or the tuned schedule
    # supplied by the caller because the search could not reach it.
    origin: str = "searched"
    # Whether this candidate is the operator's tuned schedule. A flag rather than a test on
    # `origin`, which is prose: `"tuned" in origin` also matched "untuned".
    tuned: bool = False
    # Whether it may WIN. A hand-written schedule is entered as a reference -- measured and
    # ranked so the report can say how far the search is from it -- but selecting it would
    # mean the pipeline's output is not the pipeline's output. Only the search's own
    # candidates are eligible.
    eligible: bool = True
    # The split-KV factor when the source is a partial+combine pair, else None. Read by the
    # report and by the demo when it picks the best *single* kernel next to the winner.
    split: int | None = None
    # The dense tiling the access layer read off the searched graph, when the caller has
    # one (a `DenseAttention`/`DenseMLA`); lets a split-KV pair be derived from a
    # single-kernel candidate's tiling when the search itself proposes no split-K body.
    tiling: object = None
    kernel: object = None
    checked: int = 0
    # First workload this candidate got wrong, or failed to compile/launch on. Set means
    # excluded from the ranking -- a kernel that is wrong has no meaningful latency.
    failure: str | None = None
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
    """Microseconds per launch, and whether graph replay was used.

    ``budget_us`` caps the work per measurement by shrinking ``iters`` once a pilot round has
    said how slow the kernel is. It exists because these operators' workloads span three
    orders of magnitude: the largest 64-page slab workload runs 4.8ms, where the default
    2000 launches cost 10 seconds for a single cell. Measured, dropping that case to one
    round changes the reading by 0.007%, because a 4.8ms kernel needs no amortisation --
    whereas a 2.5us kernel very much does, which is why ``REPLAY_BATCH`` is never reduced.
    """
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

    # Capture unavailable: fall back to a launch loop and say so, because such a reading
    # measures the dispatch as much as the kernel and must not be compared with the rest.
    # The pilot above returns None here too, so shrink from a direct timing instead --
    # otherwise the fallback runs the full launch count and costs what the budget forbids.
    if budget_us != float("inf") and iters > 1:
        import time as _time

        start = _time.perf_counter()
        kernel(*args)
        torch.cuda.synchronize()
        one_us = (_time.perf_counter() - start) * 1e6
        if one_us > 0:
            iters = max(1, min(iters, int(budget_us // (REPLAY_BATCH * one_us)) or 1))
    rounds = []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters * REPLAY_BATCH):
            kernel(*args)
        stop.record()
        torch.cuda.synchronize()
        rounds.append(start.elapsed_time(stop) / (iters * REPLAY_BATCH) * 1e3)
    return min(rounds), False


def _assert_context_survived(label: str, cause: BaseException) -> None:
    """Re-raise if a candidate's failure left the CUDA context unusable.

    ``torch.cuda.empty_cache()`` still returns cleanly on a poisoned context, so without
    this probe the sweep carries on and records every subsequent candidate as broken.
    """
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
    from .runtime import _compile, _load_main_kernel_from_code

    # out_idx=None: every dynamic kernel declares its outputs as parameters, so the caller
    # owns the buffers. Compiled with out_idx it would allocate them on every call, which
    # costs more than these kernels take to run.
    return with_scratch(_compile(_load_main_kernel_from_code(source), None))


# The parameter-name pattern of a split-KV kernel's internal buffers: the fused emitter
# names a partial `Partial_<name>_split` for the half that writes it and
# `Partial_<name>_combine` for the half that reads it, and both bind to one tensor.
_PARTIAL_PREFIX = "Partial_"
_PARTIAL_SUFFIXES = ("_split", "_combine")


def with_scratch(kernel):
    """``kernel`` with the operator's own signature, owning any split-KV partials.

    A single-kernel operator is returned as it is. A split-KV module's prim_func takes,
    between the operator's inputs and its outputs, the fp32 partial tensors its two
    kernels hand to each other -- parameters the operator's definition knows nothing
    about. This wraps such a kernel in a callable taking exactly ``(*inputs, *outputs)``
    like every other candidate, so validation, timing, and the benchmark wrapper need no
    special case. The partials are allocated once per distinct shape and reused: the
    symbolic extents in their declared shapes (the batch) are resolved from the operator's
    inputs the same way the runtime resolves an output's, and a fresh allocation lands on
    the first call for a shape, which under every harness here is an untimed one.
    """
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
        # Positions are relative to the full parameter list, which interleaves the
        # scratch; walk it once, pulling user arguments in order.
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
    """Position of the buffer named ``buffer_name`` among the kernel's *caller* arguments.

    Counted over the operator's own signature -- scratch partials excluded -- so it indexes
    the ``(*inputs, *outputs)`` a wrapped kernel is called with.
    """
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


def max_request_tiles(indptr, kv_tile: int, page_size: int = 1) -> int:
    """The longest request's KV window, in tiles, read off a CSR indptr.

    What decides whether splitting a decode's KV window pays: a request of one or two tiles
    has nothing to split, and a fixed split then only adds the combine launch. Costs one
    device-to-host read; call it where that is untimed.
    """
    if indptr.numel() < 2:
        return 0
    lengths = indptr[1:] - indptr[:-1]
    longest = int(lengths.max().item()) * page_size
    return -(-longest // kv_tile)


def dispatch_by_tiles(
    split_kernel, single_kernel, *, indptr_index: int, kv_tile: int, min_tiles: int,
    page_size: int = 1,
):
    """One callable running the split-KV pair or the single kernel, per workload.

    The rule is FlashInfer's, made explicit: split when the longest request has at least
    ``min_tiles`` KV tiles, else run the single kernel. The decision is a property of the
    workload's *values* (its request lengths), so it is taken once per distinct set of
    argument shapes -- on the first call, which under every harness here is an untimed one
    (SOL's round-0 check, autotune's warmup) -- and then reused. Both callers run one set of
    lengths per set of shapes: SOL one workload per process, autotune the captured workloads,
    whose axes are all distinct. The indptr is read back to the host for that one decision,
    never in the timed launches.
    """
    decided = {}

    def call(*args):
        key = tuple(
            (tuple(a.shape), str(a.dtype)) if hasattr(a, "shape") else ("scalar",)
            for a in args
        )
        use_split = decided.get(key)
        if use_split is None:
            tiles = max_request_tiles(args[indptr_index], kv_tile, page_size)
            use_split = decided[key] = tiles >= min_tiles
        return (split_kernel if use_split else single_kernel)(*args)

    call.split_kernel = split_kernel
    call.single_kernel = single_kernel
    call.min_tiles = min_tiles
    call.kv_tile = kv_tile
    return call


def _scratch_params(kernel):
    """``(param index, base name, torch dtype)`` for every partial buffer of ``kernel``."""
    from .graph_executor import torch_dtype

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
    """Compile every candidate and check it against every workload's reference.

    Out-parameter contract borrowed from ``compile_and_run_candidates_parallel`` so a caller
    reads the same way in both flows: ``success_info`` collects the candidates that compiled
    and matched everywhere, ``errors`` collects ``(label, exception)`` for the rest. A bad
    candidate never raises -- a search reports, it does not stop.

    Every workload, not a sample. Measured, this costs 1-3 seconds per operator no matter how
    many candidates there are, because the reference is built once per workload and shared, so
    there is nothing to buy by sampling. What it catches in practice is candidates that never
    get as far as a number: on the nine shipped operators, 14 failed to compile and 2 failed
    at launch (asking for 272 KB of dynamic shared memory against Hopper's 227 KB), and not
    one compiled and then computed wrong results. So this pass is insurance rather than a
    filter that has had to fire -- but a timing loop that skipped it would rank whatever it
    was handed, and the check is stricter than the benchmark's own gate (every element at
    1e-2, where SOL allows 1% of elements to miss).
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
        # Inside the guard: building a workload is what allocates (up to 31 GiB live for the
        # 64-token-page operator), so two unguarded builders would OOM an 80 GB card even
        # though neither is launching anything yet.
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
                # Poisoned with NaN per candidate, not zeroed. The buffers are shared, so a
                # candidate that fails to write a region would otherwise inherit its
                # predecessor's correct values -- and zeroing does not fix that, because for
                # the prefill operators the reference itself is ~99.9% zeros (a fully-masked
                # causal window is defined to be 0), so a kernel that wrote *nothing* would
                # agree with it. NaN agrees with nothing, and `equal_nan=False` in the
                # tolerance turns any unwritten element into a failure. Measured: every
                # shipped kernel writes every element, so this costs nothing.
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
                    # A device-side assert or an illegal access poisons the context for the
                    # whole process, so every later candidate would fail too and be blamed
                    # for this one's bug -- and the caller's tuned candidate is typically
                    # last, i.e. maximally blamed. Verified: three trivial launches after a
                    # gather assert all raise. Nothing here can recover it, so stop.
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
    """Time every validated candidate on every workload the factory yields.

    Returns ``(best, best_geomean_us, best_idx)`` -- the shape ``benchmark_graphs`` returns,
    but in **microseconds**, where that one returns milliseconds: these kernels run 2-6us and
    would print as zeros. Each candidate's ``latency_us`` is filled in place, because the
    scalar is not the whole answer: candidates routinely split the workloads between them, and
    hiding the matrix behind the winner would hide that.
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
            # One discarded measurement per candidate per workload, before any that counts.
            # Without it the first candidate timed on a workload reads systematically low --
            # measured at 1.6% over five *identical* candidates, position 0 fastest in every
            # trial, which is the width of TIE_BAND and would decide near-ties by ordering
            # alone. It lands on the small workloads (a 2.4us case steps to 2.6us and stays
            # there) and not on the busy ones, so it biases precisely the shapes an
            # equal-weight geomean is most sensitive to.
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
    # By identity: `WorkloadCandidate` is an eq=True dataclass, so `.index` would match the
    # first candidate with equal fields rather than this one.
    return best, best.geomean_us, next(
        i for i, c in enumerate(success_info) if c is best
    )


def tie_band_members(ranked, *, tie_band: float = TIE_BAND) -> list:
    """The candidates this harness cannot distinguish from the fastest one."""
    if not ranked:
        return []
    return [c for score, c in ranked if score / ranked[0][0] - 1 <= tie_band]


def pick_best(ranked, *, tie_band: float = TIE_BAND):
    """The fastest *eligible* candidate by geomean.

    Ineligible candidates -- a hand-written schedule entered for comparison -- are ranked and
    reported but cannot win. Otherwise a table headed "best kernel found" could name a kernel
    the search cannot produce, which is the opposite of what it claims. ``tie_band`` no longer
    decides anything here; it only tells the report which margins are too small to defend.
    """
    if not ranked:
        raise ValueError("No correct candidates to pick from.")
    eligible = [c for _, c in ranked if c.eligible]
    if not eligible:
        raise ValueError(
            "No eligible candidate: every correct kernel was entered for reference only."
        )
    return eligible[0]


def rank(success_info: Sequence[WorkloadCandidate]) -> list:
    """``(geomean_us, candidate)`` fastest first, by geomean over the timed workloads.

    Geomean because it is what the benchmark of record aggregates and what the perf gate
    compares, so this ordering is the one that matters downstream. Its weakness is worth
    knowing: the captured workloads are mostly small, where cost is fixed per-CTA overhead
    rather than throughput, so the geomean can crown a candidate that loses every large
    shape. ``format_ranking`` prints that when it happens instead of leaving it implied.
    """
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
    best = pick_best(ranked, tie_band=tie_band)
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
