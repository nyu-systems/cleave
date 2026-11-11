"""CUDA-graph-replay kernel timing shared by ``runtime`` and ``autotune``.

Capturing ``batch`` back-to-back launches once and replaying the graph keeps Python
dispatch out of the timed region, whatever TileLang execution backend built the kernel.
Callers pick their own batch, round count and reduction (runtime: batch 100, one round,
mean ms; autotune: batch 20, ``repeats`` rounds, min us).
"""

from __future__ import annotations

import warnings


def capture_launch_graph(kernel, args, batch: int):
    """A CUDA graph of ``batch`` back-to-back ``kernel(*args)`` launches, or None."""
    import torch

    try:
        graph = torch.cuda.CUDAGraph()
        # Capture needs its own stream, and the kernel must have run once already so
        # any lazy initialisation or allocation happens outside the capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            kernel(*args)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            for _ in range(batch):
                kernel(*args)
    except Exception:  # noqa: BLE001 - capture is an optimisation, not a requirement
        return None
    return graph


def time_rounds(body, repeats: int = 1) -> list[float]:
    """CUDA-event milliseconds of ``body()`` for each of ``repeats`` rounds."""
    import torch

    rounds = []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        body()
        stop.record()
        torch.cuda.synchronize()
        rounds.append(start.elapsed_time(stop))
    return rounds


def replay_rounds(graph, replays: int, *, repeats: int = 1, warmup: int = 1) -> list[float]:
    """Milliseconds per round of ``replays`` graph replays, after ``warmup`` untimed ones."""
    import torch

    for _ in range(warmup):
        graph.replay()  # the first replay also uploads the graph
    torch.cuda.synchronize()

    def body():
        for _ in range(replays):
            graph.replay()

    return time_rounds(body, repeats)


def launch_rounds(kernel, args, launches: int, *, repeats: int = 1) -> list[float]:
    """Milliseconds per round of ``launches`` plain launches (includes Python dispatch)."""

    def body():
        for _ in range(launches):
            kernel(*args)

    return time_rounds(body, repeats)


def mean_launch_ms(kernel, args, *, warmup: int = 0, iters: int = 1, batch: int = 100) -> float:
    """Mean ms per launch over at least ``iters`` launches, replayed in graphs of ``batch``.

    Falls back to a plain launch loop (with a warning) when capture fails, because that
    reading includes dispatch overhead and is not comparable with replay readings.
    """
    if warmup < 0:
        raise ValueError(f"warmup_iters must be >= 0, got {warmup}")
    if iters <= 0:
        raise ValueError(f"profile_iters must be > 0, got {iters}")

    import torch

    args = list(args)
    for _ in range(warmup):
        kernel(*args)
    torch.cuda.synchronize()

    batch = min(iters, batch)
    graph = capture_launch_graph(kernel, args, batch)
    if graph is not None:
        replays = -(-iters // batch)  # ceil: time at least `iters` launches
        (elapsed,) = replay_rounds(graph, replays)
        return elapsed / (replays * batch)

    warnings.warn(
        "CUDA graph capture failed for this kernel; timing a plain launch loop, "
        "which includes per-call Python dispatch overhead",
        RuntimeWarning,
        stacklevel=3,
    )
    (elapsed,) = launch_rounds(kernel, args, iters)
    return elapsed / iters
