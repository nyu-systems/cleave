import argparse
import time

import cleave.superoptimizer as csopt

from cleave.integration import graph_structure_to_cleave
from cleave.schedule import search_tiling
from cleave.util import (
    benchmark_graphs,
    compile_and_run_candidates_parallel,
)


def build_mla_decode_bhsd(
    checkpoint_file: str,
    max_parallel: int,
    force_schedule_fusion: bool,
):
    start = time.perf_counter()

    print(
        "MLA Decode BHSD Search"
    )

    B, H, S_KV, HD, PE = 1, 128, 1024, 512, 64
    q_can_split = (False, False, False)
    kv_can_split = (False, False, False)

    split_enabled = any([any(x) for x in [q_can_split, kv_can_split]])
    graph = csopt.new_tile_graph(split_enabled=split_enabled)
    Q = graph.create_tiled_input((B, H, HD), q_can_split)
    KV = graph.create_tiled_input((B, S_KV, HD), kv_can_split)
    Q_PE = graph.create_tiled_input((B, H, PE), q_can_split)
    K_PE = graph.create_tiled_input((B, S_KV, PE), kv_can_split)

    norm_factor = 128**0.5
    A = graph.matmul(Q, KV, trans_B=True)
    A_PE = graph.matmul(Q_PE, K_PE, trans_B=True)
    A = graph.add(A, A_PE)
    A = graph.div(A, norm_factor)
    A = graph.add(A, -50 / norm_factor)
    E = graph.exp(A)
    S = graph.reduction(E, 2)
    D = graph.div(E, S)
    O = graph.matmul(D, KV)
    graph.create_tiled_output(O)

    symbolic_graph, symbol_map = csopt.create_symbolic_tile_graph(graph)

    config = symbolic_graph.setup_search_config()
    config.search_thread = 16

    optimized_graphs = symbolic_graph.search_tile_graph(checkpoint_file, False, True, config)

    cleave_graph = None
    candidates = []
    for optimized_graph in optimized_graphs:
        graph_structure = optimized_graph.get_graph_structure()
        cleave_graph = graph_structure_to_cleave(graph_structure, symbol_map=symbol_map)
        candidates.extend(
            search_tiling(cleave_graph, topk=1000, force_fusion=force_schedule_fusion)
        )

    error_graphs, errors = [], []
    success_info = []

    # candidates = [cand for cand in candidates if cand.output_tile[-1] == HD and cand.rstep_map[cand.flat_graph.nodes[2]]['v2'] == HD]

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

    assert (
        len(success_info) >= 1
    ), f"Expected at least 1 successful candidates for MLA, but got {len(success_info)}."

    best_info, best_latency_ms, best_idx = benchmark_graphs(
        success_info,
        "float16",
        warmup_iters=16,
        profile_iters=1000,
    )

    print("Compile Time = %.4fsec" % (end - start))
    print("Best candidate idx: %d, latency: %.4f ms" % (best_idx, best_latency_ms))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint_mla_decode_bhsd.json"
    )
    parser.add_argument("--max_parallel", type=int, default=32)
    parser.add_argument("--force_schedule_fusion", action="store_true", default=False)
    args = parser.parse_args()

    build_mla_decode_bhsd(
        args.checkpoint,
        args.max_parallel,
        args.force_schedule_fusion,
    )


if __name__ == "__main__":
    main()
