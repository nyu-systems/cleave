"""Validation has no default bound, and holds a kernel to the bound its operator declares.

The comparison itself is pinned on CPU in ``tests/runtime/test_compare_outputs.py``; this is
the kernel-facing half (``_validate_kernel_outputs`` and the entry points above it).
"""

import pytest
import torch

from cleave.core.errors import LaunchCrash, OutputMismatch, SpecError, ValidationError
from cleave.dynamic import OPERATORS
from cleave.runtime import compile as runtime_compile
from cleave.runtime import validate
from cleave.runtime.gpu import _gpu_exec_guard
from cleave.runtime.validate import Tolerance

MISSING = "no validation tolerance declared"


class FakeKernel:
    def __init__(self, *outputs):
        self.outputs = outputs

    def __call__(self, *inputs):
        del inputs
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs


def _validate(outputs, reference, **kwargs):
    with _gpu_exec_guard():
        return validate._validate_kernel_outputs(FakeKernel(*outputs), (), reference, **kwargs)


def test_there_is_no_default_tolerance():
    """A kernel nobody declared a bound for is an error at every entry point.

    (Until Step 0 of the numerical-stability pass this test pinned the opposite: a loose
    0.5/0.5 default that every undeclared operator fell back to.)
    """
    assert not hasattr(validate, "DEFAULT_TOLERANCE")
    tensor = torch.zeros(4, device="cuda")
    with pytest.raises(TypeError, match="tolerance"):
        _validate([tensor], [tensor])
    with pytest.raises(SpecError, match=MISSING):
        _validate([tensor], [tensor], tolerance=None)
    with pytest.raises(SpecError, match=MISSING):
        validate._check_kernel(FakeKernel(tensor), graph=None, dtype="float16")
    # check_code refuses before it compiles anything, so the source is never even loaded.
    with pytest.raises(SpecError, match=MISSING) as error:
        runtime_compile.check_code("not a kernel", out_idx=[-1])
    assert error.value._cleave_stage == "validate"
    with pytest.raises(SpecError, match=MISSING):
        runtime_compile.compile_and_run_prim_func(None, out_idx=[-1], workloads_fn=tuple)


def test_an_all_zeros_output_no_longer_passes_for_small_outputs():
    # Attention outputs are well under 0.5 in magnitude, so the old 0.5/0.5 default accepted
    # a kernel that wrote nothing. Relative to the row, zeros are 100% wrong.
    reference = (0.1 * torch.randn(8, 64, device="cuda")).half()
    assert float(reference.abs().max()) < 0.5
    with pytest.raises(OutputMismatch) as error:
        _validate([torch.zeros_like(reference)], [reference], tolerance=Tolerance(rtol=5e-2))
    (measured,) = error.value.measured
    assert measured.rel == pytest.approx(1.0)


def test_validate_kernel_outputs_honors_a_per_operator_tolerance():
    """``out`` is held to its row and ``lse`` absolutely, each at the operator's own bound."""
    out_tol, lse_tol = OPERATORS["gqa_paged_decode_h32_kv8_d128_ps1"].tolerance
    assert out_tol.rtol > 0 and lse_tol.atol > 0
    out = torch.randn(4, 32, 128, device="cuda")
    lse = 10 * torch.randn(4, 32, device="cuda")
    lse[1, 3] = float("-inf")  # an empty request: defined, and to be reproduced exactly
    lse[0, 0] = 0.0  # the perturbed entry: a few 1e-6 are not representable next to a 10
    scale = out.abs().amax(dim=-1, keepdim=True)

    def run(out_error, lse_error):
        """Kernel outputs off by that many tolerances at one entry of each output."""
        got_out, got_lse = out.clone(), lse.clone()
        got_out[2, 5, 7] += out_error * out_tol.rtol * scale[2, 5, 0]
        got_lse[0, 0] += lse_error * lse_tol.atol
        return _validate([got_out, got_lse], [out, lse], tolerance=(out_tol, lse_tol))

    measured_out, measured_lse = run(0.5, 0.5)
    assert measured_out.rel == pytest.approx(0.5 * out_tol.rtol, rel=1e-3)
    assert measured_lse.abs == pytest.approx(0.5 * lse_tol.atol, rel=1e-3)
    with pytest.raises(OutputMismatch, match="output 0"):
        run(2.0, 0.5)
    with pytest.raises(OutputMismatch, match="output 1"):
        run(0.5, 2.0)
    # The outputs do not lend each other slack: the lse gets no relative bound and the
    # output no absolute one.
    assert lse_tol.rtol == 0 and out_tol.atol == 0
    # A declaration for the wrong number of outputs is an error, not a silent broadcast.
    with pytest.raises(SpecError, match="3 tolerances were declared for 2 outputs"):
        _validate([out, lse], [out, lse], tolerance=(out_tol, lse_tol, lse_tol))


def test_a_kernel_that_cannot_be_launched_is_a_launch_crash_not_a_mismatch():
    class Faulting:
        def __call__(self, *inputs):
            raise RuntimeError("CUDA error: an illegal memory access was encountered\nmore text")

    reference = torch.zeros(4, device="cuda")
    with _gpu_exec_guard(), pytest.raises(LaunchCrash) as error:
        validate._validate_kernel_outputs(Faulting(), (), [reference], tolerance=Tolerance(rtol=1e-3))
    assert str(error.value) == (
        "kernel launch raised RuntimeError: CUDA error: an illegal memory access was encountered")
    assert isinstance(error.value, ValidationError) and isinstance(error.value, AssertionError)
    assert isinstance(error.value.__cause__, RuntimeError)
