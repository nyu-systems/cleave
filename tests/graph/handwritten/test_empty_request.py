"""Every operator's answer for a request with no work: 0 in the output, -inf in the lse.

The captured SOL-ExecBench workloads only ever contain non-empty requests, so nothing else
in the suite reaches this path -- but each definition's own reference specifies it
(``if ps >= pe: output.zero_(); continue`` over a ``-inf``-filled lse), and SOL-ExecBench's
``check_tensor_sanity`` rejects NaN unconditionally, so a 0/0 leaking through as NaN would
fail the benchmark outright rather than merely differ.

An empty *query* request additionally exercises the device work-list scan, which must
contribute zero tiles for it and still map every later request correctly.
"""

import pytest

from tests.graph.handwritten import _paged_reference as ref
from tests.graph.handwritten._test_utils import (
    GATHER_TOLERANCE,
    _load_main_kernel_from_code,
    ensure_main_graph_threads,
)
from tests.graph.handwritten.handwritten_schedules import (
    gqa_paged_decode_h32_kv4_d128_ps1,
    gqa_paged_decode_h32_kv8_d128_ps1,
    gqa_paged_decode_h32_kv8_d128_ps64,
    gqa_paged_prefill_causal_h32_kv4_d128_ps1,
    gqa_paged_prefill_causal_h32_kv8_d128_ps1,
    gqa_ragged_prefill_causal_h32_kv4_d128,
    gqa_ragged_prefill_causal_h32_kv8_d128,
    mla_paged_decode_h16_ckv512_kpe64_ps1,
    mla_paged_prefill_causal_h16_ckv512_kpe64_ps1,
)
from cleave.codegen import TilelangCodeGen
from cleave.util.runtime import compile_and_run_prim_func

MLA_SM_SCALE = 0.1352337747812271

# One empty request among non-empty neighbours, so a mis-scan shows up as a wrong answer
# for the requests around it rather than only as a missing one.
DECODE_KV_LENS = [5, 0, 7]
PREFILL_Q_LENS = [3, 0, 5]
PREFILL_KV_LENS = [6, 0, 9]

# (schedule module, dims, out_idx, builder)
CASES = {
    "gqa_paged_decode_h32_kv8_d128_ps1": (
        gqa_paged_decode_h32_kv8_d128_ps1, (32, 8, 128), [6, 7], "gqa_decode", 1),
    "gqa_paged_decode_h32_kv4_d128_ps1": (
        gqa_paged_decode_h32_kv4_d128_ps1, (32, 4, 128), [6, 7], "gqa_decode", 1),
    "gqa_paged_decode_h32_kv8_d128_ps64": (
        gqa_paged_decode_h32_kv8_d128_ps64, (32, 8, 128, 64), [7, 8], "gqa_decode", 64),
    "mla_paged_decode_h16_ckv512_kpe64_ps1": (
        mla_paged_decode_h16_ckv512_kpe64_ps1, (16, 512, 64), [7, 8], "mla_decode", 1),
    "gqa_ragged_prefill_causal_h32_kv8_d128": (
        gqa_ragged_prefill_causal_h32_kv8_d128, (32, 8, 128), [6, 7], "gqa_ragged", 1),
    "gqa_ragged_prefill_causal_h32_kv4_d128": (
        gqa_ragged_prefill_causal_h32_kv4_d128, (32, 4, 128), [6, 7], "gqa_ragged", 1),
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1": (
        gqa_paged_prefill_causal_h32_kv8_d128_ps1, (32, 8, 128), [7, 8], "gqa_paged_pre", 1),
    "gqa_paged_prefill_causal_h32_kv4_d128_ps1": (
        gqa_paged_prefill_causal_h32_kv4_d128_ps1, (32, 4, 128), [7, 8], "gqa_paged_pre", 1),
    "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": (
        mla_paged_prefill_causal_h16_ckv512_kpe64_ps1, (16, 512, 64), [8, 9], "mla_paged_pre", 1),
}


def _workload(kind, dims, page_size):
    """One workload whose middle request is empty, plus its reference."""
    if kind == "gqa_decode":
        kv_lens = DECODE_KV_LENS
        pages = sum(-(-n // page_size) for n in kv_lens)
        wl = (len(kv_lens), *dims[:3], pages + 4, pages + 4)
        return (ref.gqa_paged_decode_inputs(wl, kv_lens, page_size=page_size),
                ref.gqa_paged_decode_reference(wl, kv_lens, page_size=page_size))
    if kind == "mla_decode":
        kv_lens = DECODE_KV_LENS
        wl = (len(kv_lens), *dims, sum(kv_lens) + 4, sum(kv_lens) + 4)
        return (ref.mla_paged_decode_inputs(wl, kv_lens, scale=MLA_SM_SCALE),
                ref.mla_paged_decode_reference(wl, kv_lens))
    q_lens, kv_lens = PREFILL_Q_LENS, PREFILL_KV_LENS
    total_q, n_idx = sum(q_lens), sum(kv_lens)
    if kind == "gqa_ragged":
        wl = (*dims, total_q, n_idx)
        return (ref.gqa_ragged_prefill_inputs(wl, q_lens, kv_lens),
                ref.gqa_ragged_prefill_reference(wl, q_lens, kv_lens))
    wl = (*dims, n_idx + 4, n_idx + 4, total_q)
    if kind == "mla_paged_pre":
        return (ref.mla_paged_prefill_inputs(wl, q_lens, kv_lens, scale=MLA_SM_SCALE),
                ref.mla_paged_prefill_reference(wl, q_lens, kv_lens))
    return (ref.gqa_paged_prefill_inputs(wl, q_lens, kv_lens),
            ref.gqa_paged_prefill_reference(wl, q_lens, kv_lens))


@pytest.mark.parametrize("operator", sorted(CASES))
def test_empty_request(operator):
    schedule, dims, out_idx, kind, page_size = CASES[operator]
    graph = schedule.get_handwritten_scheduled_graph(dims=dims)
    ensure_main_graph_threads(graph)
    main_kernel = _load_main_kernel_from_code(TilelangCodeGen().gen(graph))

    def workloads_fn():
        inputs, reference = _workload(kind, dims, page_size)
        yield inputs, reference(*inputs)

    compile_and_run_prim_func(
        main_kernel,
        dtype="bfloat16",
        workloads_fn=workloads_fn,
        out_idx=out_idx,
        tolerance=GATHER_TOLERANCE,
    )
