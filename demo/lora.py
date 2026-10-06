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


def build_lora(
    topk: int, checkpoint_file: str, max_parallel: int, force_schedule_fusion: bool
):
    start = time.perf_counter()

    # Mirage LoRA update O = X@W + (X@A)@B (rank R, no scaling), matching
    # mirage-project/mirage demo/demo_lora.py. Dims follow that LoRA layer
    # (hidden K = N = 4096, rank R = 16); M = 16 is Mirage's benchmark batch.
    M, N, K, R = 16, 4096, 4096, 16
    x_can_split = (False, False)
    w_can_split = (False, False)
    a_can_split = (False, False)
    b_can_split = (False, False)

    graph = csopt.new_tile_graph(
        split_enabled=[
            any(x) for x in [x_can_split, w_can_split, a_can_split, b_can_split]
        ],
    )
    x = graph.create_tiled_input((M, K), x_can_split)
    w = graph.create_tiled_input((K, N), w_can_split)
    a = graph.create_tiled_input((K, R), a_can_split)
    b = graph.create_tiled_input((R, N), b_can_split)

    base = graph.matmul(x, w)   # C = X @ W       base projection  (M, N)
    down = graph.matmul(x, a)   # D = X @ A       down projection  (M, R)
    up = graph.matmul(down, b)  # E = D @ B       up projection    (M, N)
    out = graph.add(base, up)   # O = C + E       (M, N)
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

    assert len(success_info) > 0, "Expected at least one successful candidate for LoRA"

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
    parser.add_argument("--checkpoint", type=str, default="checkpoint_lora.json")
    parser.add_argument("--max_parallel", type=int, default=64)
    parser.add_argument("--force_schedule_fusion", action="store_true", default=False)
    args = parser.parse_args()
    build_lora(
        args.topk, args.checkpoint, args.max_parallel, args.force_schedule_fusion
    )


if __name__ == "__main__":
    main()
