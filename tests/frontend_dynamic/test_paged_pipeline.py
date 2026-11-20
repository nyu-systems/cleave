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


# ------------------------------------------------- which workloads validate which candidate
class _Indptr:
    """Stands in for a workload's KV indptr tensor: one request of 4096 keys."""

    def tolist(self):
        return [0, 4096]


def _recorded_compile(monkeypatch, preset, **options):
    """``cleave.compile`` of a registry paged program with the search and the GPU stubbed
    out: ``(kernel, calls)``, ``calls`` being what was validated / timed on which workload
    keys, in order, as ``(stage, candidate labels, keys)``.

    The stubs stand at the edges only. The searched plans are two single kernels and, with a
    split, one pair per factor (made-up sources, so the registry's tuned kernel is entered
    as a reference candidate as well); the registry operator's workloads are replaced by
    their keys. Everything in between is the real ``compile_dynamic``.
    """
    import cleave.pipeline.dynamic as dynamic_mod
    import cleave.runtime.autotune as autotune
    from cleave.dynamic.paging import DenseAttention
    from cleave.dynamic.registry import DynamicOperator
    from cleave.pipeline.plans import PlanSet
    from cleave.runtime.autotune import sampled, workload_key

    calls = []

    class Graphs(tuple):
        from_cache = True

        def filter(self, fn):
            return self

    def plans(graphs, spec, config, *, split_graphs=(), filter_before_topk=False):
        def dense(kv):
            return DenseAttention(query_rows=16, kv_tile=kv, head_dim=128, n_threads=128,
                                  dtype="bfloat16", accum_dtype="float32", out_dtype="bfloat16")

        found = [dynamic_mod.PagedPlan(0, "solo", "main", None, f"solo kv={kv}", rank, None,
                                       None, source=f"solo {kv}", dense=dense(kv))
                 for rank, kv in enumerate((64, 32))]
        found += [dynamic_mod.PagedPlan(0, f"split-kv={f}", "main", None, f"pair S={f}", 0, f,
                                        None, source=f"pair {f}", dense=dense(64),
                                        origin="searched split-K")
                  for f in config.split_k]
        return PlanSet(found, graphs, config)

    def keyed_workloads(self, *, keep=0, adversarial=False):
        wanted = [self.workloads[i] for i in sampled(range(len(self.workloads)), keep)]

        def factory():
            for axes in wanted:
                yield workload_key(axes), (_Indptr(),), ()
            if adversarial:
                for distribution, axes in self.adversarial:
                    yield f"{workload_key(axes)}@{distribution}", (_Indptr(),), ()

        return factory

    def validate(candidates, workloads_fn, success, errors, *, tolerance=None):
        assert tolerance is not None
        keys = [key for key, *_ in workloads_fn()]
        calls.append(("validate", [c.label for c in candidates], keys))
        for c in candidates:
            c.kernel, c.checked = object(), len(keys)
        success.extend(candidates)

    def benchmark(success, workloads_fn, **kw):
        keys = [key for key, *_ in workloads_fn()]
        calls.append(("time", [c.label for c in success], keys))
        for c in success:
            # the pair is faster on these, the first single kernel next
            c.latency_us.update(dict.fromkeys(keys, 1.0 if c.split else 2.0 + len(c.label)))

    monkeypatch.setattr(dynamic_mod, "superoptimize", lambda *a, **k: Graphs())
    monkeypatch.setattr(dynamic_mod, "schedule_paged", plans)
    monkeypatch.setattr(DynamicOperator, "keyed_workloads", keyed_workloads)
    monkeypatch.setattr(autotune, "compile_and_validate_over_workloads", validate)
    monkeypatch.setattr(autotune, "benchmark_over_workloads", benchmark)
    monkeypatch.setattr(autotune, "param_index", lambda kernel, name: 0)
    fn, hints = build(preset)
    return cleave.compile(fn, hints=hints, **options), calls


def _adversarial(keys):
    return [k for k in keys if "@" in k]


def test_a_registry_operator_is_validated_on_its_adversarial_workloads(monkeypatch):
    """A registry operator's kernels are held to its adversarial workloads without being
    asked: the softmax of every paged layer is exact, so they are part of what it passes."""
    op = OPERATORS[CASES["kv8"]]
    assert len(op.adversarial) == 12
    captured = len(op.workloads)

    kernel, calls = _recorded_compile(monkeypatch, "kv8", split=False, max_workloads=3)
    validations = [c for c in calls if c[0] == "validate"]
    # one pass: both searched kernels and the registry's tuned kernel (a reference)
    assert [labels for _, labels, _ in validations] == [
        ["solo kv=64", "solo kv=32", "tuned kv=64 q=16"]]
    keys = validations[0][2]
    assert len(keys) == captured + 12 and len(_adversarial(keys)) == 12
    assert _adversarial(keys) == keys[captured:], "after the captured ones"
    assert {k.partition("@")[2] for k in _adversarial(keys)} == {
        "large_logits", "large_scale", "increasing", "dominant_last"}
    assert kernel.report.n_workloads == captured + 12
    summary = kernel.report.summary()
    assert f"3 of 3 correct over {captured + 12} workloads" in summary
    assert "split pairs over" not in summary
    # they are never timed
    (_, _, timed), = [c for c in calls if c[0] == "time"]
    assert len(timed) == 3 and not _adversarial(timed)

    # every registry operator, not one: the same pass for the kv4 program ...
    other = OPERATORS[CASES["kv4"]]
    assert len(other.adversarial) == 12
    _, calls = _recorded_compile(monkeypatch, "kv4", split=False)
    assert [(stage, len(_adversarial(keys))) for stage, _, keys in calls] == [
        ("validate", 12), ("time", 0)]
    # ... and asking for them by name changes nothing
    _, calls = _recorded_compile(monkeypatch, "kv4", split=False, adversarial=True)
    assert [(stage, len(_adversarial(keys))) for stage, _, keys in calls] == [
        ("validate", 12), ("time", 0)]


def test_split_pairs_are_validated_on_the_adversarial_workloads_too(monkeypatch):
    """A split-KV pair is exact like the single kernel -- its partial hands its running max
    to the combine, which weighs the splits by it -- so it is held to the same workloads,
    in the same single validation pass, and stays eligible for the split/single dispatch."""
    from cleave.dynamic.paging import CombineTiling, DenseAttention, build_split_decode
    from cleave.ir.graph import GraphNode, PlaceHolderNode
    from cleave.ir.ops import MaskedExp

    op = OPERATORS[CASES["kv8"]]
    captured = len(op.workloads)
    # what makes one pass right: the pair's partial is exact and the combine takes its max
    dense = DenseAttention(query_rows=16, kv_tile=64, head_dim=128, n_threads=128,
                           dtype="bfloat16", accum_dtype="float32", out_dtype="bfloat16")
    partial, combine = build_split_decode(
        op.spec, dense, 4, CombineTiling(split=4, split_step=1, cols=128, n_threads=128))
    shifted = [n.shift for g in partial.nodes if isinstance(g, GraphNode)
               for n in g.graph.nodes if isinstance(n, MaskedExp)]
    assert shifted == [True]
    assert [n.name for n in combine.args if isinstance(n, PlaceHolderNode)] == [
        "PlaceHolder_ROWSUM", "PlaceHolder_VALUE", "PlaceHolder_MAX"]

    split = cl.Split("kv_len", (4, 8))
    everyone = ["solo kv=64", "solo kv=32", "pair S=4", "pair S=8", "tuned kv=64 q=16"]
    kernel, calls = _recorded_compile(monkeypatch, "kv8", split=split, max_workloads=3)
    validations = [(labels, keys) for stage, labels, keys in calls if stage == "validate"]
    # one pass over every candidate, in candidate order, on captured + adversarial
    assert [labels for labels, _ in validations] == [everyone]
    keys = validations[0][1]
    assert len(keys) == captured + 12 and _adversarial(keys) == keys[captured:]
    assert len(_adversarial(keys)) == 12
    # timed together on captured workloads; the pair is eligible and the dispatch is
    # calibrated on captured workloads only
    timings = [(labels, keys) for stage, labels, keys in calls if stage == "time"]
    assert timings[0][0] == everyone
    assert all(not _adversarial(keys) for _, keys in timings)
    report = kernel.report
    assert report.best(split=True) is not None and report.best(split=False) is not None
    assert report.dispatch and report.dispatch["n_workloads"] == captured
    assert len(timings) == 2 and len(timings[1][1]) == captured  # the calibration
    assert set(kernel.dispatch_sources) == {"split", "single"}
    # every candidate passed the same workloads, and the report says so in one figure
    assert {c.checked for c in report.correct} == {captured + 12}
    assert report.n_workloads == captured + 12
    summary = report.summary()
    assert f"5 of 5 correct over {captured + 12} workloads" in summary
    assert "split pairs over" not in summary

    # asking for the adversarial workloads by name is the same pass
    _, again = _recorded_compile(monkeypatch, "kv8", split=split, max_workloads=3,
                                 adversarial=True)
    assert again == calls


def test_custom_workloads_are_not_extended_with_adversarial_ones(monkeypatch):
    """The adversarial table belongs to the registry's captured workloads; a caller's own
    workloads are validated as given."""
    own = lambda: iter([("mine_0", (_Indptr(),), ()), ("mine_1", (_Indptr(),), ())])  # noqa: E731
    _, calls = _recorded_compile(monkeypatch, "kv8", split=False, workloads=own)
    assert [keys for _, _, keys in calls] == [["mine_0", "mine_1"]] * 2
    _, calls = _recorded_compile(monkeypatch, "kv8", split=cl.Split("kv_len", (4,)),
                                 workloads=own)
    validations = [(labels, keys) for stage, labels, keys in calls if stage == "validate"]
    assert validations == [
        (["solo kv=64", "solo kv=32", "pair S=4", "tuned kv=64 q=16"], ["mine_0", "mine_1"])]


def test_a_topk_that_holds_no_accepted_tiling_says_how_to_get_past_it(monkeypatch):
    """The tilings are ranked by cost before the access layer reads them, so a small topk
    can hold none it accepts (``mla_paged_decode`` at ``topk=100``: the 512-wide head dim
    is split in every one). The refusal names the two ways out; with the filter already
    on there is none to name."""
    import cleave.pipeline.dynamic as dynamic_mod
    from cleave.core.errors import CleaveError
    from cleave.pipeline import ScheduleConfig
    from cleave.pipeline.plans import PlanSet, Refusal

    class Graphs(tuple):
        from_cache = True

    def nothing(graphs, spec, config, *, split_graphs=(), filter_before_topk=False):
        refused = [Refusal(0, "solo", "main", ValueError("the head dim is tiled"))] * 3
        return PlanSet([], graphs, config, refused)

    monkeypatch.setattr(dynamic_mod, "superoptimize", lambda *a, **k: Graphs())
    monkeypatch.setattr(dynamic_mod, "schedule_paged", nothing)
    fn, hints = build("kv8")
    with pytest.raises(CleaveError) as info:
        cleave.compile(fn, hints=hints, schedule=ScheduleConfig(topk=100))
    first, *rest = str(info.value).splitlines()
    assert first.endswith(
        "the access layer refused every tiling of the 100 ranked first; raise topk, or pass "
        "filter_before_topk=True to rank only the tilings the layer accepts")
    assert "3x ValueError: the head dim is tiled" in "\n".join(rest)
    with pytest.raises(CleaveError) as info:
        cleave.compile(fn, hints=hints, schedule=ScheduleConfig(topk=100),
                       filter_before_topk=True)
    assert str(info.value).splitlines()[0].endswith("the access layer refused every tiling")


# ----------------------------------------------------------------- a cache with no row
def test_a_cache_without_a_row_is_answered_without_a_launch():
    """Every request of a batch may have no key, and the key tensors of a ragged batch then
    have no row at all. The operator defines the answer -- 0, and -inf in the lse -- but the
    kernel cannot be launched to give it: its caches are TMA sources, and a zero-sized tensor
    has no descriptor (TileLang raises ``InternalError: Invalid TMA descriptor arguments ...
    globalAddress must be non-null`` before the kernel starts). ``PagedKernel`` answers such
    a call itself, and launches as usual as soon as the caches have a row.

    It answers only what the inputs define. Caches without a row under a KV table that
    gives requests keys, or one cache with rows next to one without, are refused: with K
    filled and V empty the lse is finite, so -inf would be a wrong answer, not a missing one."""
    import torch

    from cleave.pipeline.dynamic import PagedKernel
    from demo.dynamic.gqa_ragged_prefill import build as build_ragged

    fn, hints = build_ragged("kv8")
    attention = trace_paged(fn, hints=hints).attention()
    launched = []
    kernel = PagedKernel(attention, attention.paging_spec(), "the kernel's source",
                         compiled=lambda *args: launched.append(args))
    kernel._slots = attention.inputs  # the stub is called in the traced order

    q = torch.randn(13, 32, 128).to(torch.bfloat16)
    qo_indptr = torch.tensor([0, 4, 4, 13], dtype=torch.int32)
    kv_indptr = torch.zeros(4, dtype=torch.int32)  # three requests, no key in any
    k = v = torch.empty(0, 8, 128, dtype=torch.bfloat16)
    out, lse = kernel(q, k, v, qo_indptr, kv_indptr, 0.088)
    assert not launched
    assert out.shape == (13, 32, 128) and out.dtype == torch.bfloat16 and not out.any()
    assert lse.shape == (13, 32) and lse.dtype == torch.float32
    assert torch.isneginf(lse).all()
    # by name as well, and with a table that starts anywhere: it spans no entry
    named = dict(q=q, k=k, v=v, qo_indptr=qo_indptr, kv_indptr=kv_indptr + 7, sm_scale=0.088)
    out, lse = kernel(**named)
    assert not launched and not out.any() and torch.isneginf(lse).all()

    # the same empty caches under a table that gives the requests 12 keys: no answer
    keyed = torch.tensor([0, 5, 9, 12], dtype=torch.int32)
    with pytest.raises(ValueError, match="kv_indptr gives the requests 12 entries of caches "
                                         r"that have no row \(k, v\)"):
        kernel(q, k, v, qo_indptr, keyed, 0.088)
    # one cache with rows and one without, whatever the table says: the lse of the first
    # is finite, and there is nothing to weigh by it
    rows = torch.zeros(12, 8, 128, dtype=torch.bfloat16)
    for table in (keyed, kv_indptr):
        with pytest.raises(ValueError, match="v has no row but k does"):
            kernel(q, rows, v, qo_indptr, table, 0.088)
        with pytest.raises(ValueError, match="k has no row but v does"):
            kernel(**dict(named, k=k, v=rows, kv_indptr=table))
    assert not launched

    # one spare row in each and the kernel is launched as before, on outputs it has to fill
    k = v = torch.zeros(1, 8, 128, dtype=torch.bfloat16)
    kernel(q, k, v, qo_indptr, kv_indptr, 0.088)
    (args,) = launched
    assert len(args) == 8 and args[1] is k and args[5] == 0.088
    assert [tuple(t.shape) for t in args[6:]] == [(13, 32, 128), (13, 32)]
    # ... also when the table gives the requests keys: that is the kernel's to answer
    kernel(q, rows, rows, qo_indptr, keyed, 0.088)
    assert len(launched) == 2 and launched[1][4] is keyed
