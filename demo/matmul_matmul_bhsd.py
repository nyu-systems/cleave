import argparse
import time

import cleave.superoptimizer as csopt

from cleave.integration import graph_structure_to_cleave
from cleave.schedule import search_tiling
from cleave.util import (
    benchmark_graphs,
    compile_and_run_candidates_parallel,
)


DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"


def build_matmul_matmul_bhsd(
    topk: int, checkpoint_file: str, max_parallel: int, force_schedule_fusion: bool
):
    start = time.perf_counter()

    B, H, S_Q, S_KV, D = 1, 32, 4096, 4096, 64
    q_can_split = (False, False, False)
    k_can_split = (False, False, False)
    v_can_split = (False, False, False)

    graph = csopt.new_tile_graph(
        split_enabled=[any(x) for x in [q_can_split, k_can_split, v_can_split]],
    )
    q = graph.create_tiled_input((B * H, S_Q, D), q_can_split)
    k = graph.create_tiled_input((B * H, S_KV, D), k_can_split)
    v = graph.create_tiled_input((B * H, S_KV, D), v_can_split)

    s = graph.matmul(q, k, trans_B=True)
    out = graph.matmul(s, v)
    graph.create_tiled_output(out)

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
            search_tiling(cleave_graph, topk=topk, force_fusion=force_schedule_fusion)
        )

    error_graphs, errors = [], []
    success_info = []

    compile_and_run_candidates_parallel(
        candidates,
        DTYPE,
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

    assert len(success_info) >= 19, "Expected 19 successful candidates."

    _, best_latency_ms, best_idx = benchmark_graphs(
        success_info,
        DTYPE,
        warmup_iters=16,
        profile_iters=1000,
    )

    print("Compile Time = %.4fsec" % (end - start))
    print("Best candidate idx: %d, latency: %.4f ms" % (best_idx, best_latency_ms))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint_matmul_matmul_bhsd.json"
    )
    parser.add_argument("--max_parallel", type=int, default=64)
    parser.add_argument("--force_schedule_fusion", action="store_true", default=False)
    args = parser.parse_args()
    build_matmul_matmul_bhsd(
        args.topk, args.checkpoint, args.max_parallel, args.force_schedule_fusion
    )


if __name__ == "__main__":
    main()
