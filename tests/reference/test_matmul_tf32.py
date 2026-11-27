"""GPU: a float32 matmul is the float32 product of its operands as TF32 reads them.

``Matmul.tensor_core_format`` says ``tf32`` for a float32 program
(``tests/pipeline_api/test_matmul_operands.py`` pins the declaration and the report on
CPU). This file pins what the kernel does with it, on the device:

* the instruction reads each operand with the leading 10 fraction bits of its significand
  and drops the other 13 -- it truncates, it does not round. A kernel returns for ``(x, w)``
  bit for bit what it returns for the two with those bits cleared;
* nothing else is lost. On operands that have no bits to drop the kernel is the float32
  product, to the accuracy of a float32 sum over ``K`` (1.0e-6 of the row maximum at
  ``K = 128``), and it is held to the bound measured there;
* on the operands as given it is off by one TF32 ulp of the row maximum, 1e-3: fifty times
  the loosest bound a float32 output may declare. That is why a float32 program's matmul
  has no declared tolerance on ordinary inputs, and why the report names the format.
"""

import pytest
import torch

import cleave.lang as cl
from cleave.frontend import trace
from cleave.ir.ops import Matmul
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.compile import _compile, _load_main_kernel_from_code, build_candidate
from cleave.runtime.gpu import _gpu_exec_guard
from cleave.runtime.reference import execute, graph_out_indices, make_inputs
from cleave.runtime.reference import order_inputs_for_kernel, placeholder_order
from cleave.runtime.validate import compare_outputs, launch
from cleave.schedule import search_tiling

from tests._tolerances import FP32_CEILING, tolerance_for

SIZES = dict(M=256, K=128, N=256)
DROPPED_BITS = 23 + 1 - Matmul.SIGNIFICANT_BITS["tf32"]  # of a float32 significand: 13


def matmul(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    return x @ w


def as_tf32(tensor):
    """``tensor`` with the significand bits a TF32 operand does not have cleared."""
    return (tensor.view(torch.int32) & ~((1 << DROPPED_BITS) - 1)).view(torch.float32)


def row_error(actual, reference):
    scale = reference.abs().amax(dim=-1, keepdim=True)
    return float(((actual - reference).abs() / scale).max())


def kernels(topk=4):
    """``(ir, compiled kernel)`` for the first tilings of the float32 matmul."""
    (graph,) = list(GraphSet.identity(trace(matmul, SIZES, cl.f32)))
    (ir,) = lower(graph)
    for cand in search_tiling(ir, topk=topk):
        code, _ = build_candidate(cand)
        yield ir, _compile(_load_main_kernel_from_code(code), graph_out_indices(ir))


def run(ir, kernel, operands):
    by_name = {ph.name: t for ph, t in zip(placeholder_order(ir), operands)}
    out = launch(kernel, order_inputs_for_kernel(by_name, kernel))
    out = out[0] if isinstance(out, (list, tuple)) else out
    return out.reshape(SIZES["M"], SIZES["N"])


def test_gpu_a_float32_matmul_is_the_product_of_what_tf32_keeps_of_its_operands():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    assert DROPPED_BITS == 13
    assert not torch.backends.cuda.matmul.allow_tf32  # the reference multiplies in float32
    tolerance = tolerance_for("matmul_tf32")
    n = 0
    for ir, kernel in kernels():
        operands = make_inputs(ir, "float32")
        kept = tuple(as_tf32(t) for t in operands)
        assert all(not torch.equal(k, t) for k, t in zip(kept, operands))
        with _gpu_exec_guard():
            as_given, on_kept = run(ir, kernel, operands), run(ir, kernel, kept)
            # the 13 low bits of an operand never reach the product
            assert torch.equal(as_given, on_kept)
            # ... and that is all the kernel loses
            compare_outputs([on_kept], execute(ir, list(kept)), tolerance)
            # against the product of the operands as given it is a TF32 ulp off
            (product,) = execute(ir, list(operands))
            assert row_error(as_given, product) > 10 * FP32_CEILING
        n += 1
    assert n == 4
