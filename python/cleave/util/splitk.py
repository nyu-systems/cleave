from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..integration.graph_processing import is_split_graph, process_graphs
from .runtime import (
    CompilerPassOptions,
    benchmark_graphs,
    compile_and_run_candidates_parallel,
)


@dataclass(frozen=True)
class SplitKStageSpec:
    label: str
    graph_index: int
    search_once_per_factor: bool = False


@dataclass
class SplitKStageBenchmarkResult:
    label: str
    latency_ms: float
    best_idx: int
    best_info: Any


@dataclass
class SplitKCompileResult:
    label: str
    latency_ms: float
    split_factor: int | None
    best_idx: int | None = None
    stage_results: list[SplitKStageBenchmarkResult] = field(default_factory=list)


@dataclass
class SplitKCompileSummary:
    solo_result: SplitKStageBenchmarkResult | None
    split_results: list[SplitKCompileResult]
    best_result: SplitKCompileResult


@dataclass
class _CandidateCompileGroup:
    label: str
    candidates: list[Any] = field(default_factory=list)
    success_info: list[Any] = field(default_factory=list)
    error_graphs: list[Any] = field(default_factory=list)
    errors: list[Any] = field(default_factory=list)


@dataclass
class _SplitKCompilation:
    split_factor: int
    groups: list[_CandidateCompileGroup]


def _search_tiling_candidates(
    graph,
    *,
    topk: int,
    force_fusion: bool,
):
    from ..schedule import search_tiling

    return search_tiling(
        graph,
        topk=topk,
        force_fusion=force_fusion,
    )


def make_mnk_splitk_stage_specs(
    *,
    partial_label: str = "partial",
    combine_label: str = "combine",
    partial_graph_index: int = 0,
    combine_graph_index: int = 1,
) -> list[SplitKStageSpec]:
    """Build stage specs for the common matmul-style split-k layout [M, N, K]."""
    return [
        SplitKStageSpec(
            label=partial_label,
            graph_index=partial_graph_index,
        ),
        SplitKStageSpec(
            label=combine_label,
            graph_index=combine_graph_index,
            search_once_per_factor=True,
        ),
    ]


def _collect_candidate_groups(
    graph_structures,
    *,
    symbol_map: dict[int, int] | None,
    split_factors: Sequence[int],
    topk: int,
    force_schedule_fusion: bool,
    stage_specs: Sequence[SplitKStageSpec],
    include_solo: bool,
    solo_label: str | None,
) -> tuple[_CandidateCompileGroup | None, list[_SplitKCompilation]]:
    solo_group = None
    if include_solo:
        solo_group = _CandidateCompileGroup(label=solo_label or "solo")

    solo_structures = [gs for gs in graph_structures if not is_split_graph(gs)]
    split_structures = [gs for gs in graph_structures if is_split_graph(gs)]

    if solo_group is not None:
        for graph_bundle in process_graphs(solo_structures, symbol_map=symbol_map):
            assert (
                len(graph_bundle) == 1
            ), "Expected one graph for solo candidate generation, got %d" % len(
                graph_bundle
            )
            solo_group.candidates.extend(
                _search_tiling_candidates(
                    graph_bundle[0],
                    topk=topk,
                    force_fusion=force_schedule_fusion,
                )
            )

    if split_structures and not stage_specs:
        raise ValueError("stage_specs is required when split-k graph structures exist.")

    split_compilations = []
    for split_factor in [int(factor) for factor in split_factors]:
        compilation = _SplitKCompilation(
            split_factor=split_factor,
            groups=[
                _CandidateCompileGroup(
                    label=f"split-k={split_factor} {spec.label}",
                )
                for spec in stage_specs
            ],
        )

        initialized_once = set()
        for graph_bundle in process_graphs(
            split_structures,
            symbol_map=symbol_map,
            split_factors=[split_factor],
        ):
            for spec_idx, (spec, group) in enumerate(
                zip(stage_specs, compilation.groups)
            ):
                assert spec.graph_index < len(
                    graph_bundle
                ), "Stage %s expects graph index %d, but bundle only has %d graphs" % (
                    spec.label,
                    spec.graph_index,
                    len(graph_bundle),
                )
                if spec.search_once_per_factor and spec_idx in initialized_once:
                    continue

                group.candidates.extend(
                    _search_tiling_candidates(
                        graph_bundle[spec.graph_index],
                        topk=topk,
                        force_fusion=force_schedule_fusion,
                    )
                )
                if spec.search_once_per_factor:
                    initialized_once.add(spec_idx)

        if any(group.candidates for group in compilation.groups):
            split_compilations.append(compilation)

    return solo_group, split_compilations


def _compile_candidate_group(
    group: _CandidateCompileGroup,
    max_parallel: int,
    dtype: str,
    compiler_pass_options: CompilerPassOptions | None = None,
):
    if not group.candidates:
        print("No %s candidates were generated." % group.label)
        return

    compile_and_run_candidates_parallel(
        group.candidates,
        dtype,
        max_parallel,
        group.success_info,
        group.error_graphs,
        group.errors,
        compiler_pass_options=compiler_pass_options,
    )


def _log_candidate_group(group: _CandidateCompileGroup):
    print(
        "%s: %d candidates, %d successful, %d failed"
        % (
            group.label,
            len(group.candidates),
            len(group.success_info),
            len(group.errors),
        )
    )


def _benchmark_candidate_group(
    group: _CandidateCompileGroup,
    *,
    dtype: str,
    warmup_iters: int,
    profile_iters: int,
) -> SplitKStageBenchmarkResult | None:
    if not group.success_info:
        print(
            "Skipping benchmark for %s because nothing compiled successfully."
            % group.label
        )
        return None

    best_info, best_latency_ms, best_idx = benchmark_graphs(
        group.success_info,
        dtype,
        warmup_iters=warmup_iters,
        profile_iters=profile_iters,
    )

    print(
        "Best %s candidate idx: %d, latency: %.4f ms"
        % (group.label, best_idx, best_latency_ms)
    )

    return SplitKStageBenchmarkResult(
        label=group.label,
        latency_ms=best_latency_ms,
        best_idx=best_idx,
        best_info=best_info,
    )


def _benchmark_splitk_compilation(
    compilation: _SplitKCompilation,
    *,
    dtype: str,
    warmup_iters: int,
    profile_iters: int,
) -> SplitKCompileResult | None:
    stage_results = []
    for group in compilation.groups:
        stage_result = _benchmark_candidate_group(
            group,
            dtype=dtype,
            warmup_iters=warmup_iters,
            profile_iters=profile_iters,
        )
        if stage_result is None:
            print(
                "Skipping split-k=%d overall selection because one stage has no successful candidates."
                % compilation.split_factor
            )
            return None
        stage_results.append(stage_result)

    total_latency_ms = sum(result.latency_ms for result in stage_results)
    latency_breakdown = " + ".join(
        "%s %.4f ms" % (result.label, result.latency_ms) for result in stage_results
    )
    print(
        "Best split-k=%d compilation latency: %.4f ms (%s)"
        % (compilation.split_factor, total_latency_ms, latency_breakdown)
    )

    return SplitKCompileResult(
        label=f"split-k={compilation.split_factor}",
        latency_ms=total_latency_ms,
        split_factor=compilation.split_factor,
        stage_results=stage_results,
    )


def _select_best_overall(
    solo_result: SplitKStageBenchmarkResult | None,
    split_results: list[SplitKCompileResult],
    solo_label: str,
) -> SplitKCompileResult:
    candidates = list(split_results)
    if solo_result is not None:
        candidates.append(
            SplitKCompileResult(
                label=solo_label,
                latency_ms=solo_result.latency_ms,
                split_factor=None,
                best_idx=solo_result.best_idx,
            )
        )

    if not candidates:
        raise ValueError("No candidates compiled successfully.")

    return min(candidates, key=lambda result: result.latency_ms)


def _format_split_candidate_counts(
    split_compilations: Sequence[_SplitKCompilation],
) -> str:
    if not split_compilations:
        return "no split candidates"

    summaries = []
    for compilation in split_compilations:
        prefix = f"split-k={compilation.split_factor} "
        stage_counts = ", ".join(
            "%s=%d"
            % (
                group.label.removeprefix(prefix),
                len(group.candidates),
            )
            for group in compilation.groups
        )
        total_candidates = sum(len(group.candidates) for group in compilation.groups)
        summaries.append(
            "split-k=%d: %d total (%s)"
            % (
                compilation.split_factor,
                total_candidates,
                stage_counts,
            )
        )
    return "; ".join(summaries)


def run_splitk_compile(
    graph_structures,
    *,
    symbol_map: dict[int, int] | None,
    split_factors: Sequence[int],
    topk: int,
    max_parallel: int,
    force_schedule_fusion: bool,
    stage_specs: Sequence[SplitKStageSpec],
    include_solo: bool = False,
    solo_label: str | None = None,
    dtype: str = "float16",
    warmup_iters: int = 16,
    profile_iters: int = 1000,
    compiler_pass_options: CompilerPassOptions | None = None,
) -> SplitKCompileSummary:
    """Search, compile, benchmark, and select solo vs split-k compilations."""
    solo_group, split_compilations = _collect_candidate_groups(
        graph_structures,
        symbol_map=symbol_map,
        split_factors=split_factors,
        topk=topk,
        force_schedule_fusion=force_schedule_fusion,
        stage_specs=stage_specs,
        include_solo=include_solo,
        solo_label=solo_label,
    )

    print(
        "Candidate groups: %d solo candidates, %s"
        % (
            0 if solo_group is None else len(solo_group.candidates),
            _format_split_candidate_counts(split_compilations),
        )
    )

    # Future extraction candidate:
    #   compile_candidate_groups_parallel(groups)
    # Keep the grouped split-k policy behind a single high-level utility.
    if solo_group is not None:
        _compile_candidate_group(
            solo_group,
            max_parallel=max_parallel,
            dtype=dtype,
            compiler_pass_options=compiler_pass_options,
        )
        _log_candidate_group(solo_group)

    for compilation in split_compilations:
        for group in compilation.groups:
            _compile_candidate_group(
                group,
                max_parallel=max_parallel,
                dtype=dtype,
                compiler_pass_options=compiler_pass_options,
            )
            _log_candidate_group(group)

    solo_result = None
    if solo_group is not None:
        solo_result = _benchmark_candidate_group(
            solo_group,
            dtype=dtype,
            warmup_iters=warmup_iters,
            profile_iters=profile_iters,
        )

    split_results = []
    for compilation in split_compilations:
        result = _benchmark_splitk_compilation(
            compilation,
            dtype=dtype,
            warmup_iters=warmup_iters,
            profile_iters=profile_iters,
        )
        if result is not None:
            split_results.append(result)

    best_result = _select_best_overall(
        solo_result,
        split_results,
        solo_label=solo_group.label if solo_group is not None else "solo",
    )

    if best_result.split_factor is None:
        print(
            "Best overall compilation: %s, candidate idx: %d, latency: %.4f ms"
            % (best_result.label, best_result.best_idx, best_result.latency_ms)
        )
    else:
        print(
            "Best overall compilation: %s, latency: %.4f ms"
            % (best_result.label, best_result.latency_ms)
        )
        print(
            "  stage winners: %s"
            % ", ".join(
                "%s idx=%d" % (stage.label, stage.best_idx)
                for stage in best_result.stage_results
            )
        )

    return SplitKCompileSummary(
        solo_result=solo_result,
        split_results=split_results,
        best_result=best_result,
    )
