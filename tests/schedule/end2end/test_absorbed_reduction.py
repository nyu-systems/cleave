"""Tilings with an absorbed reduction compute the right values (GPU).

Each case steps a score contraction (``Q K^T`` over ``D``, or the rope ``Q_pe K_pe^T`` over
``PE``) below its extent while the ``S_KV`` loop iterates, so the contraction is absorbed into
that loop. ``apply_tiling`` used to slice the contraction axis with the ``S_KV`` counter: the
kernel contracted one block of the axis per ``S_KV`` step and read past the tile. Attention
outputs are small, so those kernels still passed a loose elementwise bound; the bound here is
row-normalized and computed in this file.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pytest

from cleave.dynamic.dense import flash_attention_sd_graph
from cleave.runtime.compile import CompilerPassOptions, build_candidate_code

from tests import _graphs as g
from tests.graph.apply_tiling._test_utils import build_tiled_candidate

DTYPE = "float16"
# max|actual - ref| <= ROW_RTOL * max|ref| over every output row (the last axis). A correct
# kernel differs from the torch reference by one fp16 ulp of the row max (measured 8.1e-4 to
# 9.8e-4 on these cases); the mis-tiled kernels measured 0.32 to 2.2.
ROW_RTOL = 2e-3

# name -> (graph builder, dims, output tile, reduction steps, absorbed reduction, pass options)
CASES = {
    "flash_attention_sd": (
        flash_attention_sd_graph, (4096, 4096, 64), [128, 64],
        {"mma_0": 32, "mma_1": 128, "rowsum_0": 128}, "mma_0", None,
    ),
    # The tiling demo/mha_prefill.py used to pick as its fastest kernel.
    "mha_bhsd": (
        g.mha_bhsd_graph, (1, 32, 4096, 4096, 64), [1, 128, 64],
        {"mma_0": 32, "mma_1": 128, "rowsum_0": 128}, "mma_0", None,
    ),
    "mla_decode_bhsd": (
        g.mla_decode_bhsd_graph, (1, 128, 4096, 512, 64), [1, 64, 512],
        {"mma_0": 512, "mma_pe": 32, "mma_1": 32, "rowsum_0": 32}, "mma_pe",
        CompilerPassOptions(mla_decode_pass=True),
    ),
}


def _worst_row_error(name, code):
    """``(worst row-normalized error, non-finite count)`` of kernel ``code`` on case ``name``.

    Runs in its own process: a kernel that reads out of bounds can fault, and the fault takes
    the process's CUDA context with it.
    """
    import torch

    from cleave.runtime.compile import _compile, _load_main_kernel_from_code
    from cleave.runtime.gpu import _gpu_exec_guard
    from cleave.runtime.reference import (
        build_inputs_and_reference,
        order_inputs_for_kernel,
        reconcile_output_layout,
    )

    build, dims = CASES[name][:2]
    with _gpu_exec_guard():
        inputs, (reference,), out_idx = build_inputs_and_reference(build(*dims), DTYPE)
    kernel = _compile(_load_main_kernel_from_code(code), out_idx)
    with _gpu_exec_guard():
        actual = kernel(*order_inputs_for_kernel(inputs, kernel))
        torch.cuda.synchronize()
        reference = reconcile_output_layout(reference, actual.shape).double()
        nonfinite = int((~torch.isfinite(actual)).sum())
        error = (actual.double() - reference).abs().amax(dim=-1)
        return float((error / reference.abs().amax(dim=-1)).max()), nonfinite


@pytest.mark.parametrize("name", sorted(CASES))
def test_absorbed_reduction_matches_reference(name):
    build, dims, output_tile, steps, absorbed, options = CASES[name]
    graph = build(*dims)
    ops = {n.name: n for n in graph.nodes if not n.is_placeholder() and not n.is_output()}
    rstep = {op: {} for op in ops}
    for op, step in steps.items():
        rstep[op] = {ops[op].raxis[0].var.name: step}
    # The case is of the mis-tiled class: the absorbed contraction is stepped below its extent.
    assert steps[absorbed] < int(ops[absorbed].raxis[0].dom.extent)

    cand = build_tiled_candidate(graph, output_tile, rstep)
    code = build_candidate_code(cand, options)

    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=spawn) as pool:
        worst, nonfinite = pool.submit(_worst_row_error, name, code).result()

    assert nonfinite == 0, f"{nonfinite} NaN/inf entries in the kernel output"
    assert worst <= ROW_RTOL, f"worst row-normalized error {worst:.3e} > {ROW_RTOL:g}"
