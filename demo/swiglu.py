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


def build_swiglu(
    topk: int, checkpoint_file: str, max_parallel: int, force_schedule_fusion: bool
):
    start = time.perf_counter()

    M, N, K = 4096, 14336, 4096
    x_can_split = (False, False)
    w_can_split = (False, False)

    graph = csopt.new_tile_graph(
        split_enabled=[any(x) for x in [x_can_split, w_can_split]],
    )
    x = graph.create_tiled_input((M, K), x_can_split)
    w0 = graph.create_tiled_input((K, N), w_can_split)
    w1 = graph.create_tiled_input((K, N), w_can_split)

    mm0 = graph.matmul(x, w0)
    mm1 = graph.matmul(x, w1)
    silu = graph.silu(mm0)
    out = graph.mul(silu, mm1)
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
            search_tiling(cleave_graph, topk=1000, force_fusion=force_schedule_fusion)
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

    assert len(success_info) >= 21, "Expected 21 successful candidates for SwigLU"

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
    parser.add_argument("--checkpoint", type=str, default="checkpoint_swiglu.json")
    parser.add_argument("--max_parallel", type=int, default=64)
    parser.add_argument("--force_schedule_fusion", action="store_true", default=False)
    args = parser.parse_args()
    build_swiglu(
        args.topk, args.checkpoint, args.max_parallel, args.force_schedule_fusion
    )


if __name__ == "__main__":
    main()
