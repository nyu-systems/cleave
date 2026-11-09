import pytest

from cleave.dynamic.dense import flash_attention_sd_graph
from cleave.schedule import search_tiling
from cleave.runtime.compile import CompilerPassOptions

from tests import _graphs as g
from tests.schedule.end2end._test_utils import run_end2end_candidates

DTYPE = "float16"
MLA_DECODE = {"compiler_pass_options": CompilerPassOptions(mla_decode_pass=True)}

# name -> (graph builder, dims, min_success, search_tiling / run options)
CASES = {
    "batched_matmul_splitk_combine": (g.batched_matmul_splitk_combine_graph, (2, 4, 128, 128), 80, {}),
    "batched_matmul_splitk_partial": (g.batched_matmul_splitk_partial_graph, (2, 4, 128, 128, 1024), 1, {}),
    "flash_attention_sd": (flash_attention_sd_graph, (4096, 4096, 64), 16, {}),
    "gqa_bhsd": (g.gqa_bhsd_graph, (1, 32, 8, 4096, 4096, 64), 16, {}),
    # Few candidates survive: the (M, R) = (16, 16) down-projection forces 32 threads.
    "lora": (g.lora_graph, (16, 4096, 4096, 16), 1, {}),
    "matmul": (g.matmul_graph, (128, 128, 4096), 18, {}),
    "matmul_matmul_bhsd": (g.matmul_matmul_bhsd_graph, (1, 32, 4096, 4096, 64), 19, {}),
    "matmul_matmul_small_k": (g.matmul_matmul_graph, (4096, 4096, 64), 16, {}),
    "matmul_matmul_small_n": (g.matmul_matmul_graph, (4096, 64, 64), 20, {}),
    "matmul_rowsum_small_k": (g.matmul_rowsum_graph, (4096, 4096, 64), 10, {}),
    "matmul_splitk_combine": (g.matmul_splitk_combine_graph, (4, 128, 128), 89, {}),
    "matmul_splitk_partial": (g.matmul_splitk_partial_graph, (4, 128, 128, 1024), 1, {}),
    "mha_bhsd": (g.mha_bhsd_graph, (1, 32, 4096, 4096, 64), 16, {}),
    "mha_split_combine": (g.mha_split_combine_graph, (32, 4, 4096, 64), 1, {}),
    "mha_split_partial": (g.mha_split_partial_graph, (1, 32, 4, 4096, 1024, 64), 1, {}),
    "mla_decode_bhsd": (g.mla_decode_bhsd_graph, (1, 128, 4096, 512, 64), 5, MLA_DECODE),
    "mla_prefill_bhsd": (g.mla_prefill_bhsd_graph, (1, 128, 4096, 4096, 192, 128), 13, {}),
    "mqa_bhsd": (g.mqa_bhsd_graph, (1, 32, 1, 4096, 4096, 64), 16, {}),
    # Fused RMSNorm-into-GQA attention (dims of demo/qknorm.py) only tiles validly at the
    # full sequence length and under forced fusion; the few survivors rank near the top.
    "qknorm": (g.qknorm_graph, (1, 64, 8, 4096, 4096, 128), 1, {"topk": 32, "force_fusion": True}),
    "rmsnorm_matmul": (g.rmsnorm_matmul_graph, (4096, 6144, 4096), 25, {}),
    "rmsnorm_matmul_split_combine": (g.rmsnorm_matmul_split_combine_graph, (4, 1024, 6144, 4096), 1, {}),
    "rmsnorm_matmul_split_partial": (g.rmsnorm_matmul_split_partial_graph, (4, 1024, 6144, 1024), 1, {}),
    "rowsum": (g.rowsum_graph, (4096, 4096), 28, {}),
    "rowsum_div": (g.rowsum_div_graph, (4096, 4096), 1, {}),
    "swiglu": (g.swiglu_graph, (4096, 14336, 4096), 21, {}),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_end2end(name):
    build, dims, min_success, options = CASES[name]
    options = dict(options)
    pass_options = options.pop("compiler_pass_options", None)
    candidates = search_tiling(build(*dims), **{"topk": 1000, **options})
    run_end2end_candidates(
        name, candidates, DTYPE, min_success=min_success,
        compiler_pass_options=pass_options,
    )
