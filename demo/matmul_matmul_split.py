import argparse
import time

import cleave.superoptimizer as csopt

from cleave.util import SplitKStageSpec, run_splitk_compile


DTYPE = "float16"


SPLITK_STAGE_SPECS = [
    SplitKStageSpec(
        label="partial",
        graph_index=0,
    ),
    SplitKStageSpec(
        label="combine",
        graph_index=1,
        search_once_per_factor=True,
    ),
]


def build_matmul_matmul_split(
    topk: int,
    checkpoint_file: str,
    max_parallel: int,
    force_schedule_fusion: bool,
    split: bool,
):
    start = time.perf_counter()

    B, H, S_Q, S_KV, D = 1, 32, 64, 4096, 64
    SPLIT_FACTORS = [4, 8, 16, 32]
    q_can_split = (False, False, False)
    k_can_split = (False, split, False)
    v_can_split = (False, split, False)

    print(
        "MatMul+MatMul Decode BHSD Search "
        "(B=%d, H=%d, S_Q=%d, S_KV=%d, D=%d, split=%s)"
        % (B, H, S_Q, S_KV, D, "true" if split else "false")
    )

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
    config.max_num_kernel_graph_op = 8 if split else 5

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
        solo_label="solo matmul_matmul_bhsd",
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
        checkpoint = "checkpoint_matmul_matmul_split_%s.json" % (
            "true" if split else "false"
        )

    build_matmul_matmul_split(
        args.topk,
        checkpoint,
        args.max_parallel,
        args.force_schedule_fusion,
        split,
    )


if __name__ == "__main__":
    main()
