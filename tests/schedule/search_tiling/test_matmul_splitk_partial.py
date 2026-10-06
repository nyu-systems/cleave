import os
import shutil

from cleave.graph import OutputNode, Graph
from cleave.ops import Matmul, SplitView, wire_input
from cleave.schedule import search_tiling
from cleave.util import (
    gather_all_nodes,
    find_topo_sort,
)


DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"

LOG = True
MAX_PARALLEL = max(1, int(os.getenv("CLEAVE_MAX_PARALLEL", "64")))


def test_matmul_splitk_partial():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = os.path.join(script_dir, "log", file_name_without_ext)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    SPLIT_K, M, N, K_PARTIAL = 4, 128, 128, 1024
    K = SPLIT_K * K_PARTIAL
    split_a = SplitView(
        (M, K),
        axis=1,
        factor=SPLIT_K,
        dtype=DTYPE,
        name="split_a",
    )
    split_b = SplitView(
        (K, N),
        axis=0,
        factor=SPLIT_K,
        dtype=DTYPE,
        name="split_b",
    )
    node_0 = Matmul(
        (SPLIT_K, M, N, K_PARTIAL),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        warp_policy="rows",
    )
    wire_input(split_a, node_0, 0)
    wire_input(split_b, node_0, 1)
    output_nodes = [OutputNode(node_0, dtype=OUT_DTYPE)]
    nodes = gather_all_nodes(node_0)
    main_graph = Graph(nodes=find_topo_sort([n for n in nodes if n.is_output()]))

    candidates = search_tiling(main_graph, topk=1000)

    assert (
        len(candidates) > 0
    ), "Expected to find at least one tiling candidate for matmul with split-k partial, but found none."
