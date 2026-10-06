import argparse
import time

import cleave.superoptimizer as csopt

from cleave.integration import graph_structure_to_cleave
from cleave.schedule import search_tiling
from cleave.util import (
    benchmark_graphs,
    compile_and_run_candidates_parallel,
)


RMS_NORM_EPS = 1e-6


def build_qknorm(
    split_qk: bool,
    split_pv: bool,
    checkpoint_file: str,
    max_parallel: int,
    force_schedule_fusion: bool,
):
    start = time.perf_counter()

    print(
        "QKNorm Search with split_qk=%s, split_pv=%s"
        % ("true" if split_qk else "false", "true" if split_pv else "false")
    )

    B, H_Q, H_KV, S_Q, S_KV, HD = 1, 64, 8, 4096, 4096, 128
    q_can_split = (False, split_qk, False)
    k_can_split = (False, split_qk, split_pv)
    v_can_split = (False, split_pv, False)

    split_enabled = any([any(x) for x in [q_can_split, k_can_split, v_can_split]])
    graph = csopt.new_tile_graph(split_enabled=split_enabled)
    Q = graph.create_tiled_input((B * H_Q, S_Q, HD), q_can_split)
    K = graph.create_tiled_input((B * H_KV, S_KV, HD), k_can_split)
    V = graph.create_tiled_input((B * H_KV, S_KV, HD), v_can_split)

    q_sq = graph.mul(Q, Q)
    k_sq = graph.mul(K, K)
    q_mean = graph.div(graph.reduction(q_sq, 2), float(HD))
    k_mean = graph.div(graph.reduction(k_sq, 2), float(HD))
    Q_norm = graph.mul(Q, graph.div(1, graph.sqrt(graph.add(q_mean, RMS_NORM_EPS))))
    K_norm = graph.mul(K, graph.div(1, graph.sqrt(graph.add(k_mean, RMS_NORM_EPS))))

    norm_factor = HD**0.5
    A = graph.matmul(Q_norm, K_norm, trans_B=True)
    A = graph.div(A, norm_factor)
    A = graph.add(A, -50 / norm_factor)
    E = graph.exp(A)
    S = graph.reduction(E, 2)
    D = graph.div(E, S)
    O = graph.matmul(D, V)
    graph.create_tiled_output(O)

    symbolic_graph, symbol_map = csopt.create_symbolic_tile_graph(graph)

    config = symbolic_graph.setup_search_config()
    config.search_thread = 16
    # QKNorm's source graph has ~25 kernel operators (two RMSNorms built from
    # primitives + the attention block), well above the default budget of 13, so
    # the search cannot assemble an equivalent kernel graph without a larger
    # budget. Follow the attention-prefill convention (op count + margin; +2 more
    # when splitting introduces split/reduction ops).
    config.max_num_kernel_graph_op = 28 if split_enabled else 26

    optimized_graphs = symbolic_graph.search_tile_graph(checkpoint_file, False, True, config)

    candidates = []
    for optimized_graph in optimized_graphs:
        graph_structure = optimized_graph.get_graph_structure()
        cleave_graph = graph_structure_to_cleave(graph_structure, symbol_map=symbol_map)
        # topk is capped for QKNorm: the fused RMSNorm + attention kernel is heavy
        # to compile and only a fraction of tilings are WGMMA-schedulable, so a
        # smaller candidate pool keeps compile time bounded while still yielding
        # many working candidates (the surviving ones rank near the top).
        candidates.extend(
            search_tiling(cleave_graph, topk=1000, force_fusion=force_schedule_fusion)
        )

    error_graphs, errors = [], []
    success_info = []

    compile_and_run_candidates_parallel(
        candidates,
        "float16",
        max_parallel,
        success_info,
        error_graphs,
        errors,
    )

    end = time.perf_counter()
    print(
        "Successful candidates: %d, Failed candidates: %d"
        % (len(success_info), len(errors))
    )

    assert len(success_info) > 0, "Expected at least one successful candidate for QKNorm"

    _, best_latency_ms, best_idx = benchmark_graphs(
        success_info,
        "float16",
        warmup_iters=16,
        profile_iters=1000,
    )

    print("Compile Time = %.4fsec" % (end - start))
    print("Best candidate idx: %d, latency: %.4f ms" % (best_idx, best_latency_ms))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_qk", choices=["true", "false"], default="false")
    parser.add_argument("--split_pv", choices=["true", "false"], default="false")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_qknorm.json")
    parser.add_argument("--max_parallel", type=int, default=64)
    # QKNorm's fused RMSNorm + attention kernel is only schedulable under forced
    # fusion: the default (non-forced) tile policy prunes every candidate for
    # this graph, so fusion is on by default here.
    parser.add_argument(
        "--force_schedule_fusion", choices=["true", "false"], default="true"
    )
    args = parser.parse_args()

    build_qknorm(
        args.split_qk == "true",
        args.split_pv == "true",
        args.checkpoint,
        args.max_parallel,
        args.force_schedule_fusion == "true",
    )


if __name__ == "__main__":
    main()
