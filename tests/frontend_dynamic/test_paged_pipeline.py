"""The paged pipeline: searched plans reproduce the expected kernels; compile -> Kernel runs."""

import pytest

pytest.importorskip("tilelang")

import cleave
import cleave.lang as cl
from cleave.core.errors import SpecError
from cleave.dynamic import OPERATORS
from cleave.frontend.paged import trace_paged
from demo.dynamic.gqa_paged_decode import build, gqa_paged_decode

CASES = {
    "kv8": "gqa_paged_decode_h32_kv8_d128_ps1",
    "kv4": "gqa_paged_decode_h32_kv4_d128_ps1",
    "ps64": "gqa_paged_decode_h32_kv8_d128_ps64",
}


def _attention(preset):
    fn, hints = build(preset)
    return trace_paged(fn, hints=hints).attention()


@pytest.mark.expected
@pytest.mark.parametrize("name", sorted(OPERATORS))
def test_pinned_tiling_reproduces_expected(name):
    from cleave.dynamic.dense import mla_paged_kernel_source, paged_kernel_source
    from demo.dynamic.__main__ import family_of

    module, preset = family_of(name)
    fn, hints = module.build(preset)
    att = trace_paged(fn, hints=hints).attention()
    op = OPERATORS[name]
    emit = mla_paged_kernel_source if att.is_mla else paged_kernel_source
    source = emit(att.paging_spec(), **op.tuned_tiling)
    assert source == op.expected_kernel_path.read_text(encoding="utf-8")


@pytest.mark.gpu
@pytest.mark.search
@pytest.mark.expected
@pytest.mark.parametrize("preset", sorted(CASES))
def test_searched_plans_reproduce_expected(preset):
    from cleave.pipeline import ScheduleConfig, SuperoptConfig, superoptimize
    from cleave.pipeline.dynamic import schedule_paged

    att = _attention(preset)
    graphs = superoptimize(att.dense_body(), SuperoptConfig())
    plans = schedule_paged(graphs, att.paging_spec(), ScheduleConfig(topk=100))
    expected = OPERATORS[CASES[preset]].expected_kernel_path.read_text(encoding="utf-8")
    tuned = [p for p in plans if p.source == expected]
    assert len(tuned) == 1, plans.summary()
    assert (tuned[0].dense.query_rows, tuned[0].dense.kv_tile) == (16, 64)
    assert plans.refused and all(r.variant == "solo" for r in plans.refused)
    assert len({p.source for p in plans}) == len(plans)


@pytest.mark.search
@pytest.mark.expected
@pytest.mark.parametrize("preset", sorted(CASES))
def test_filtered_top_k_slots_are_distinct_kernels(preset):
    """With the paging filter ahead of the cut, each of four slots is a different kernel.

    A score matmul stepped below the head dim is scheduled over all of it, which makes that
    tiling the kernel of the one that steps the head dim whole. The filter refuses it rather
    than let it take that one's slot, so ``topk=4`` -- what ``test_compile_end_to_end``
    searches -- still reaches the 64-token KV tile the expected kernel is tuned at.
    """
    from cleave.pipeline import ScheduleConfig, SuperoptConfig, superoptimize
    from cleave.pipeline.dynamic import schedule_paged

    att = _attention(preset)
    graphs = superoptimize(att.dense_body(), SuperoptConfig())
    plans = schedule_paged(graphs, att.paging_spec(), ScheduleConfig(topk=4),
                           filter_before_topk=True)
    tilings = [(p.dense.kv_tile, p.dense.query_rows, p.dense.n_threads) for p in plans]
    assert tilings == [(32, 16, 128), (32, 16, 256), (64, 16, 128), (64, 16, 256)], plans.summary()
    expected = OPERATORS[CASES[preset]].expected_kernel_path.read_text(encoding="utf-8")
    assert [p.source == expected for p in plans].count(True) == 1


@pytest.mark.gpu
@pytest.mark.search
@pytest.mark.slow
@pytest.mark.parametrize("page", [1, 64])
def test_compile_end_to_end(page, tmp_path):
    import torch

    from cleave.dynamic import reference
    from cleave.pipeline import ScheduleConfig
    from cleave.pipeline.dynamic import PagedKernel

    fn = gqa_paged_decode(PAGE=page)
    hints = None if page == 1 else cl.ScheduleHints(grid_order=("head", "batch"), pass_configs={})
    wl = cl.workloads.paged_decode(batch=(1, 3), kv_len=(5, 300), samples=2)
    kernel = cleave.compile(fn, schedule=ScheduleConfig(topk=4), filter_before_topk=True,
                            hints=hints, workloads=wl, max_workloads=1)
    assert isinstance(kernel, PagedKernel) and kernel.report.correct
    assert kernel.latency_us > 0

    lens = [700, 1, 129, 64]
    pages = sum(-(-n // page) for n in lens)
    dims = (len(lens), 32, 8, 128, pages + 3, pages)
    inputs = reference.gqa_paged_decode_inputs(dims, lens, page_size=page)
    want = reference.gqa_paged_decode_reference(dims, lens, page_size=page)(*inputs)
    args = inputs if page > 1 else [*inputs[:5], None, inputs[5]]
    out, lse = kernel(*args)
    torch.testing.assert_close(out.float(), want[0].float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, want[1], rtol=1e-2, atol=1e-2)
    named = dict(zip(["q", "k_cache", "v_cache", "kv_indptr", "kv_indices", "kv_last_page_len",
                      "sm_scale"], args))
    out2, _ = kernel(**named)
    assert torch.equal(out, out2)
    loaded = cleave.load(kernel.save(tmp_path / "paged"))
    assert type(loaded) is PagedKernel and torch.equal(loaded(*args)[0], out)


def test_resolve_workloads_has_no_fallback_tolerance():
    """The bound is the one given, else the registry operator's own; never a generic one."""
    from cleave.pipeline.dynamic import resolve_workloads
    from demo.dynamic.gqa_paged_prefill import gqa_paged_prefill

    registry = _attention("kv8")
    op = OPERATORS[CASES["kv8"]]
    custom = trace_paged(gqa_paged_decode(H_Q=16, H_KV=4)).attention()  # no such operator
    assert custom.program.name not in OPERATORS
    wl = cl.workloads.paged_decode(batch=(1, 3), kv_len=(5, 300), samples=2)
    own = (cl.Tolerance(rtol=1e-2), cl.Tolerance(rtol=1e-6, atol=1e-6))

    assert resolve_workloads(registry)[1] is op.tolerance
    # custom workloads for a registry operator are still held to that operator's bound
    assert resolve_workloads(registry, wl)[1] is op.tolerance
    assert resolve_workloads(registry, wl, own)[1] is own
    assert resolve_workloads(custom, wl, own)[1] is own
    for args in ((custom,), (custom, wl), (custom, None, None, True)):
        with pytest.raises(SpecError, match="no validation tolerance declared"):
            resolve_workloads(*args)
    # adversarial workloads exist for the registry's captured workloads only
    for args in ((registry, wl, None, True), (custom, wl, own, True), (custom, None, own, True)):
        with pytest.raises(SpecError, match="adversarial workloads are defined for the registry"):
            resolve_workloads(*args)
    # a prefill program with neither is told about the workloads first, as before
    prefill = trace_paged(gqa_paged_prefill(H_Q=16, H_KV=4)).attention()
    assert prefill.is_prefill and prefill.program.name not in OPERATORS
    for tolerance in (None, own):
        with pytest.raises(SpecError, match="no captured .* workloads; pass workloads="):
            resolve_workloads(prefill, None, tolerance)


def test_a_paged_program_that_cannot_be_validated_is_refused_before_the_search(monkeypatch):
    import cleave.pipeline.dynamic as dynamic_mod

    monkeypatch.setattr(dynamic_mod, "superoptimize",
                        lambda *a, **k: pytest.fail("searched a program it cannot validate"))
    custom = gqa_paged_decode(H_Q=16, H_KV=4)
    wl = cl.workloads.paged_decode(batch=(1, 3), kv_len=(5, 300), samples=2)
    for options in ({}, {"workloads": wl}, {"preset": "fast", "split": False}):
        with pytest.raises(SpecError, match="no validation tolerance declared"):
            cleave.compile(custom, **options)
    with pytest.raises(SpecError, match="adversarial workloads are defined for the registry"):
        cleave.compile(custom, workloads=wl, adversarial=True,
                       tolerance=(cl.Tolerance(rtol=1e-2), cl.Tolerance(rtol=1e-6, atol=1e-6)))


def test_mla_split_pairs_are_derived_from_single_tilings():
    from cleave.dynamic.dense import KV_PROXY, mla_attention_sd_graph, tiled_dense_attention
    from cleave.dynamic.paging import read_dense_mla
    from cleave.pipeline.dynamic import PagedPlan, _derive_split

    op = OPERATORS["mla_paged_decode_h16_ckv512_kpe64_ps1"]
    dense = mla_attention_sd_graph(16, KV_PROXY, 512, 64, dtype="bfloat16", out_dtype="bfloat16")
    steps = {"mma_0": {"v2": 512}, "mma_pe": {"v2": 64}, "mma_1": {"v2": 64}, "rowsum_0": {"v1": 64}}
    tiled = tiled_dense_attention(dense, 16, 512, 64, steps=steps, n_threads=256)
    solo = PagedPlan(0, "solo", "main", None, "L", 0, None, None, dense=read_dense_mla(tiled))
    added, refused = [], []
    _derive_split(op.spec, [solo], (4, 8), lambda *a: added.append(a), refused)
    assert not refused and [a[6] for a in added] == [4, 8]
    assert [a[8] for a in added] == ["derived split-K"] * 2
    assert all("Partial_" in a[4] for a in added)
