"""The split-KV decode pairs run, and agree with the operators' references.

The emitted text is pinned by ``tests/graph/apply_paging/test_split_decode.py``; this is
the numeric half. Each pair is compiled through ``autotune._compiled`` -- which is what the
demos and the benchmark use, and what hides the fp32 partial buffers behind the operator's
own signature -- and checked at 1e-2 on a stratified sample of the captured workloads plus
a hand-built one with an empty request in the middle. The empty request is the case the
split arithmetic has to get right with no work at all: every split runs zero tiles, stores
zero partials, and the combine has to turn 0/0 into the operator's defined 0 and -inf.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tilelang")
torch = pytest.importorskip("torch")

from tests.graph.apply_paging.test_split_decode import CASES, split_kernel_source
from tests.graph.handwritten import _paged_reference as ref
from tests.graph.handwritten._paged_workloads import OPERATORS, keyed_workloads
from tests.graph.handwritten._test_utils import GATHER_TOLERANCE
from tests.graph.handwritten.test_empty_request import (
    DECODE_KV_LENS,
    MLA_SM_SCALE,
)
from cleave.util.autotune import _compiled
from cleave.util.env import _gpu_exec_guard

# How many captured workloads each pair is checked on: a stratified sample, so the tiny
# batch-1 shapes, the batch-16 middle and the 50k-token batch-64 tail are all represented.
KEEP = 6


def _check(kernel, inputs, reference):
    outputs = [torch.full_like(t, float("nan")) for t in reference]
    kernel(*inputs, *outputs)
    torch.cuda.synchronize()
    for got, want in zip(outputs, reference):
        torch.testing.assert_close(
            got.float(),
            want.float(),
            rtol=GATHER_TOLERANCE.rtol,
            atol=GATHER_TOLERANCE.atol,
            equal_nan=GATHER_TOLERANCE.equal_nan,
        )


@pytest.mark.parametrize("operator", sorted(CASES))
def test_split_pair_matches_the_reference(operator):
    kernel = _compiled(split_kernel_source(operator))
    assert getattr(kernel, "scratch_params", None), "a split pair carries partial buffers"
    seen = 0
    for _key, inputs, reference in keyed_workloads(operator, keep=KEEP)():
        with _gpu_exec_guard():
            _check(kernel, inputs, reference)
        seen += 1
        del inputs, reference
        torch.cuda.empty_cache()
    assert seen == KEEP


def _empty_request_workload(operator):
    spec = OPERATORS[operator]
    kv_lens = DECODE_KV_LENS
    if operator.startswith("mla"):
        wl = (len(kv_lens), *spec.dims, sum(kv_lens) + 4, sum(kv_lens) + 4)
        inputs = ref.mla_paged_decode_inputs(wl, kv_lens, scale=MLA_SM_SCALE)
        return inputs, ref.mla_paged_decode_reference(wl, kv_lens)(*inputs)
    pages = sum(-(-n // spec.page_size) for n in kv_lens)
    wl = (len(kv_lens), *spec.dims[:3], pages + 4, pages + 4)
    inputs = ref.gqa_paged_decode_inputs(wl, kv_lens, page_size=spec.page_size)
    return inputs, ref.gqa_paged_decode_reference(wl, kv_lens, page_size=spec.page_size)(*inputs)


@pytest.mark.parametrize("operator", sorted(CASES))
def test_split_pair_handles_an_empty_request(operator):
    kernel = _compiled(split_kernel_source(operator))
    with _gpu_exec_guard():
        inputs, reference = _empty_request_workload(operator)
        assert torch.isneginf(reference[1]).any(), "the workload must contain an empty request"
        _check(kernel, inputs, reference)


def test_scratch_wrapper_reuses_one_allocation_per_shape():
    """The partials are allocated once per batch size and shared by the two names."""
    from tests.graph.apply_paging._test_utils import operator_kernel_source

    kernel = _compiled(split_kernel_source("gqa_paged_decode_h32_kv8_d128_ps1"))
    params = kernel.scratch_params
    assert [p[1] for p in params] == [
        "Partial_ROWSUM", "Partial_ROWSUM", "Partial_VALUE", "Partial_VALUE"
    ]
    assert all(p[2] == torch.float32 for p in params)
    _key, inputs, reference = next(keyed_workloads("gqa_paged_decode_h32_kv8_d128_ps1", keep=1)())
    with _gpu_exec_guard():
        outputs = [torch.empty_like(t) for t in reference]
        kernel(*inputs, *outputs)
        torch.cuda.synchronize()
        # The second call for the same shapes allocates nothing: what a timing loop replays
        # is the launch alone.
        held = torch.cuda.memory_allocated()
        kernel(*inputs, *outputs)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == held
        # Too few or too many arguments is a caller bug, not a silent misbinding.
        with pytest.raises(AssertionError, match="arguments"):
            kernel(*inputs)
    # A single-kernel source comes back as the compiled kernel itself.
    single = _compiled(operator_kernel_source("gqa_paged_decode_h32_kv8_d128_ps1"))
    assert not hasattr(single, "scratch_params")
