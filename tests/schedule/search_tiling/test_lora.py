import os
import shutil

from cleave.schedule import search_tiling

from tests._graphs import lora_graph


def test_lora():
    # LoRA (O = X@W + (X@A)@B) fuses matmuls with heterogeneous reduction
    # extents -- the base/down projections reduce over K while the up projection
    # reduces over the rank R. The tiling policy grouped the base (K) and up (R)
    # reductions together (shared X, no descendant reduction) and expanded their
    # reduction steps in lockstep, which raised "ValueError: <K-step> is not in
    # list" when a K-scale step was written into the R-extent axis. This exercises
    # that path: search_tiling must enumerate candidates without crashing.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = os.path.join(script_dir, "log", file_name_without_ext)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    M, N, K, R = 16, 4096, 4096, 16
    main_graph = lora_graph(M, N, K, R)

    candidates = search_tiling(main_graph, topk=1000)

    assert (
        len(candidates) > 0
    ), "Expected to find at least one tiling candidate for LoRA, but found none."
