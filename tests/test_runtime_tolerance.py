import torch

from tests.graph.handwritten._test_utils import GATHER_TOLERANCE

import cleave.util.runtime as runtime
from cleave.util.env import _gpu_exec_guard


class FakeKernel:
    def __init__(self, output):
        self.output = output

    def __call__(self, *inputs):
        del inputs
        return self.output


def _captured_tolerance(monkeypatch, **kwargs):
    seen = {}

    def fake_assert_close(actual, reference, rtol, atol, equal_nan):
        del actual, reference
        seen["rtol"] = rtol
        seen["atol"] = atol
        seen["equal_nan"] = equal_nan

    monkeypatch.setattr(torch.testing, "assert_close", fake_assert_close)
    with _gpu_exec_guard():
        tensor = torch.zeros(4, device="cuda")
        runtime._validate_kernel_outputs(FakeKernel(tensor), (), [tensor], **kwargs)
    return seen


def test_validate_kernel_outputs_keeps_the_legacy_bound_by_default(monkeypatch):
    assert _captured_tolerance(monkeypatch) == {
        "rtol": 0.5,
        "atol": 0.5,
        "equal_nan": False,
    }


def test_validate_kernel_outputs_honors_a_per_operator_tolerance(monkeypatch):
    # equal_nan stays False: an empty attention window is defined to be 0/-inf, so a NaN
    # is a real defect and must not be allowed to match one in the reference.
    assert _captured_tolerance(monkeypatch, tolerance=GATHER_TOLERANCE) == {
        "rtol": 0.01,
        "atol": 0.01,
        "equal_nan": False,
    }
