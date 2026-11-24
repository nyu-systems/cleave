"""Tilings with an absorbed reduction compute the right values (GPU).

Each case steps a score contraction (``Q K^T`` over ``D``, or the rope ``Q_pe K_pe^T`` over
``PE``) below its extent while the ``S_KV`` loop iterates, so the contraction is absorbed into
that loop. Sliced with the ``S_KV`` counter, the contraction axis would be contracted one block
per ``S_KV`` step, reading past the tile (the mis-tiled kernels measured 0.32 to 2.2 of the
row max). Attention outputs are small, so such a kernel can pass a loose elementwise bound;
each kernel here is held by ``compare_outputs`` to the row-relative bound this test declares
(``tests/_tolerances.py``, ``absorbed_reduction``), and its comparison is logged like every
other validation's (``CLEAVE_VALIDATION_LOG``).
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pytest

from cleave.dynamic.dense import flash_attention_sd_graph
from cleave.runtime.compile import CompilerPassOptions, build_candidate_code

from tests import _graphs as g
from tests._tolerances import tolerance_for
from tests.graph.apply_tiling._test_utils import build_tiled_candidate

DTYPE = "float16"
# max|actual - ref| <= rtol * max|ref| over every output row (the last axis). A correct kernel
# differs from the torch reference by one fp16 ulp of the row max (measured 8.1e-4 to 9.8e-4
# on these cases); the mis-tiled kernels measured 0.32 to 2.2.
TOLERANCE = tolerance_for("absorbed_reduction")

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


def held(actual, reference, what):
    """``(row-normalized error or None, "" or why it fails)``: ``compare_outputs`` at
    ``TOLERANCE``, in plain values (they cross a process boundary)."""
    from cleave.core.errors import NonFiniteOutput, OutputMismatch
    from cleave.runtime.validate import compare_outputs

    try:
        (measured,) = compare_outputs([actual], [reference], TOLERANCE, what=what)
    except (NonFiniteOutput, OutputMismatch) as e:
        found = getattr(e, "measured", None)
        return (found[0].rel if found else None), f"{type(e).__name__}: {e}"
    return measured.rel, ""


def _held_kernel(name, code):
    """``held`` for kernel ``code`` on case ``name``.

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
        return held(actual, reconcile_output_layout(reference, actual.shape), name)


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
        worst, problem = pool.submit(_held_kernel, name, code).result()

    assert not problem, problem
    assert 0 <= worst <= TOLERANCE.rtol


def test_the_bound_holds_a_right_kernel_and_fails_a_mis_tiled_one():
    """The comparison a case is held to, on made-up outputs: the reference itself passes;
    a kernel 0.32 of the row max off (the smallest mis-tiled error measured) and one with a
    NaN fail, in the program's dtype (no float64)."""
    import torch

    reference = (torch.randn(64, 128, generator=torch.Generator().manual_seed(0))
                 .to(torch.float16))
    assert held(reference.clone(), reference, "same") == (0.0, "")
    rows = reference.float().abs().amax(dim=-1, keepdim=True)
    off = (reference.float() + 0.32 * rows).to(torch.float16)
    worst, problem = held(off, reference, "mis-tiled")
    assert problem.startswith("OutputMismatch: mis-tiled: output 0") and worst > 0.3
    nan = reference.clone()
    nan[3, 5] = float("nan")
    assert held(nan, reference, "nan")[1].startswith("NonFiniteOutput: nan: ")
