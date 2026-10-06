import os
import shutil

from cleave.schedule import search_tiling

from tests._graphs import qknorm_graph


def test_qknorm():
    # QKNorm (RMSNorm on Q and K, then grouped-query flash attention) fuses two
    # RMSNorm reductions over the head dim with the QK^T / softmax / P@V attention
    # block. Its QK^T matmul takes two computed (RMS-normalized) operands, and the
    # RMSNorm reductions leave rank-2 (batch, seq) intermediates -- the schedule
    # only enumerates candidates for this fused shape under forced fusion. This
    # guards that search_tiling enumerates candidates without crashing.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = os.path.join(script_dir, "log", file_name_without_ext)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    B, H_Q, H_KV, S_Q, S_KV, HD = 1, 64, 8, 4096, 4096, 128
    main_graph = qknorm_graph(B, H_Q, H_KV, S_Q, S_KV, HD)

    candidates = search_tiling(main_graph, topk=32, force_fusion=True)

    assert (
        len(candidates) > 0
    ), "Expected to find at least one tiling candidate for QKNorm, but found none."
