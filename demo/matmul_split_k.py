import argparse
import time

import cleave.superoptimizer as csopt

from cleave.util import make_mnk_splitk_stage_specs, run_splitk_compile


DTYPE = "float16"
SPLITK_STAGE_SPECS = make_mnk_splitk_stage_specs()


def build_matmul(
    topk: int,
    checkpoint_file: str,
    max_parallel: int,
    force_schedule_fusion: bool,
    split: bool,
):
    start = time.perf_counter()

    M, N, K = 128, 128, 4096
    SPLIT_FACTORS = [4, 8, 16]
    x_can_split = (False, split)
    w_can_split = (split, False)

    print(
        "MatMul Search (M=%d, N=%d, K=%d, split=%s)"
        % (M, N, K, "true" if split else "false")
    )

    graph = csopt.new_tile_graph(
        split_enabled=[any(x) for x in [x_can_split, w_can_split]],
    )
    x = graph.create_tiled_input((M, K), x_can_split)
    w = graph.create_tiled_input((K, N), w_can_split)

    out = graph.matmul(x, w)
    graph.create_tiled_output(out)

    symbolic_graph, symbol_map = csopt.create_symbolic_tile_graph(graph)

    config = symbolic_graph.setup_search_config()
    config.search_thread = 16

    optimized_graphs = symbolic_graph.search_tile_graph(checkpoint_file, False, True, config)

    graph_structures = [g.get_graph_structure() for g in optimized_graphs]

    run_splitk_compile(
        graph_structures,
        symbol_map=symbol_map,
        split_factors=SPLIT_FACTORS,
        topk=topk,
        max_parallel=max_parallel,
        force_schedule_fusion=force_schedule_fusion,
        stage_specs=SPLITK_STAGE_SPECS,
        include_solo=True,
        solo_label="solo matmul",
        dtype=DTYPE,
    )

    end = time.perf_counter()

    print("Compile Time = %.4fsec" % (end - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--split", choices=["true", "false"], default="true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_parallel", type=int, default=32)
    parser.add_argument("--force_schedule_fusion", action="store_true", default=False)
    args = parser.parse_args()

    split = args.split == "true"
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = "checkpoint_matmul_split_%s.json" % ("true" if split else "false")

    build_matmul(
        args.topk,
        checkpoint,
        args.max_parallel,
        args.force_schedule_fusion,
        split,
    )


if __name__ == "__main__":
    main()
