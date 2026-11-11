"""Paged attention traced from a cl function: derived PagingSpec, dense body, errors."""

import pytest

import cleave.lang as cl
from cleave.core.errors import SpecError
from cleave.dynamic import OPERATORS
from cleave.frontend.paged import is_paged, trace_paged
from demo.dynamic.__main__ import family_of
from demo.dynamic.gqa_paged_decode import PRESETS, build, gqa_paged_decode

CASES = {
    "kv8": "gqa_paged_decode_h32_kv8_d128_ps1",
    "kv4": "gqa_paged_decode_h32_kv4_d128_ps1",
    "ps64": "gqa_paged_decode_h32_kv8_d128_ps64",
}


def attention(preset="kv8"):
    fn, hints = build(preset)
    return trace_paged(fn, hints=hints).attention()


def registry_attention(name):
    module, preset = family_of(name)
    fn, hints = module.build(preset)
    return trace_paged(fn, hints=hints).attention()


@pytest.mark.parametrize("name", sorted(OPERATORS))
def test_every_registry_operator_has_a_frontend_demo(name):
    att = registry_attention(name)
    assert att.program.name == name
    assert att.paging_spec() == OPERATORS[name].spec
    assert att.kind == OPERATORS[name].spec.kind


@pytest.mark.parametrize("preset", sorted(CASES))
def test_spec_equals_registry(preset):
    att = attention(preset)
    assert att.program.name == CASES[preset]
    assert att.paging_spec() == OPERATORS[CASES[preset]].spec


def test_kernel_inputs_drop_unread_last_page_len():
    assert [p.name for p in attention("kv8").inputs] == [
        "q", "k_cache", "v_cache", "kv_indptr", "kv_indices", "sm_scale"]
    assert "kv_last_page_len" in [p.name for p in attention("ps64").inputs]


def test_schedule_hints_map_onto_spec_knobs():
    hints = cl.ScheduleHints(grid_order=["head", "batch"], rasterization=None, warp_policy="even",
                             pass_configs={})
    spec = trace_paged(gqa_paged_decode(), hints=hints).attention().paging_spec()
    assert (spec.grid_order, spec.rasterization, spec.warp_policy, spec.pass_configs) == (
        ("head", "batch"), None, "even", {})
    with pytest.raises(SpecError, match="unknown schedule hints"):
        cl.ScheduleHints(tile=64)
    assert PRESETS["kv4"]["hints"] == cl.ScheduleHints(grid_order=("head", "batch"))


def test_is_paged():
    fn, _ = build()
    assert is_paged(fn) and is_paged(trace_paged(fn))

    def dense(x: cl.Tensor["M", "K"]):
        return x

    assert not is_paged(dense)


def test_emit_lse_follows_outputs():
    b, n = cl.dims("batch n_idx", dynamic=True)

    def op(q: cl.Tensor[b, 32, 128], k_cache: cl.Tensor["num_pages", 1, 8, 128],
           v_cache: cl.Tensor["num_pages", 1, 8, 128], kv_indptr: cl.Tensor[b + 1, cl.i32],
           kv_indices: cl.Tensor[n, cl.i32], sm_scale: cl.Scalar[cl.f32]):
        pages = cl.PageTable(kv_indptr, kv_indices)
        k, v = cl.paged_gather(k_cache, pages), cl.paged_gather(v_cache, pages)
        p = cl.softmax((q.view(b, 8, 4, 128) @ k.mT) * sm_scale, dim=-1)
        return (p @ v).view(b, 32, 128)

    spec = trace_paged(op).attention().paging_spec()
    assert spec.emit_lse is False and spec.num_kv_heads == 8


def test_slab_needs_last_page_len():
    b, n = cl.dims("batch n_idx", dynamic=True)

    def op(q: cl.Tensor[b, 32, 128], k_cache: cl.Tensor["num_pages", 64, 8, 128],
           v_cache: cl.Tensor["num_pages", 64, 8, 128], kv_indptr: cl.Tensor[b + 1, cl.i32],
           kv_indices: cl.Tensor[n, cl.i32], sm_scale: cl.Scalar[cl.f32]):
        pages = cl.PageTable(kv_indptr, kv_indices)
        k, v = cl.paged_gather(k_cache, pages), cl.paged_gather(v_cache, pages)
        p, lse = cl.softmax((q.view(b, 8, 4, 128) @ k.mT) * sm_scale, return_lse=True)
        return (p @ v).view(b, 32, 128), lse.view(b, 32)

    with pytest.raises(SpecError, match="last_page_len"):
        trace_paged(op).attention().paging_spec()


def test_unscaled_scores_are_refused():
    b, n = cl.dims("batch n_idx", dynamic=True)

    def op(q: cl.Tensor[b, 32, 128], k_cache: cl.Tensor["num_pages", 1, 8, 128],
           v_cache: cl.Tensor["num_pages", 1, 8, 128], kv_indptr: cl.Tensor[b + 1, cl.i32],
           kv_indices: cl.Tensor[n, cl.i32], sm_scale: cl.Scalar[cl.f32]):
        pages = cl.PageTable(kv_indptr, kv_indices)
        k, v = cl.paged_gather(k_cache, pages), cl.paged_gather(v_cache, pages)
        p = cl.softmax(q.view(b, 8, 4, 128) @ k.mT)
        return (p @ v).view(b, 32, 128)

    with pytest.raises(SpecError, match="cl.Scalar"):
        trace_paged(op).attention()


def test_static_programs_reject_paged_ops():
    import cleave

    prog = cleave.Program("p", cl.f16)
    t = prog.input("x", (16, 16))
    with pytest.raises(SpecError):
        t.view(256)
    with pytest.raises(SpecError):
        cl.softmax(t, return_lse=True)


# ---------------------------------------------------------------- dense body


class _Recorder:
    """Stands in for csopt's tile graph: logs every builder call with its arguments."""

    def __init__(self, log):
        self.log, self.n = log, 0

    def _call(self, name, *args, **kw):
        norm = tuple(a.tag if isinstance(a, _Value) else a for a in args)
        self.log.append((name, norm, tuple(sorted(kw.items()))))
        self.n += 1
        return _Value(self.n)

    def __getattr__(self, name):
        return lambda *a, **k: self._call(name, *a, **k)


class _Value:
    def __init__(self, tag):
        self.tag = tag


def _driver_gqa_graph(query_rows, head_dim, kv_proxy, split=False):
    """The former demo driver's dense GQA body (decode and prefill), verbatim."""
    import cleave.superoptimizer as csopt

    graph = csopt.new_tile_graph(split_enabled=[False, split, split])
    q = graph.create_tiled_input((query_rows, head_dim), (False, False), dtype="bfloat16")
    k = graph.create_tiled_input((kv_proxy, head_dim), (split, False), dtype="bfloat16")
    v = graph.create_tiled_input((kv_proxy, head_dim), (split, False), dtype="bfloat16")
    scores = graph.div(graph.matmul(q, k, trans_B=True), head_dim**0.5)
    weights = graph.exp(scores)
    denom = graph.reduce_sum(weights, 1)
    graph.create_tiled_output(graph.matmul(graph.div(weights, denom), v))
    return graph


def _driver_mla_graph(query_rows, ckv, kpe, kv_proxy, split=False):
    """The former demo driver's dense MLA body, verbatim."""
    import cleave.superoptimizer as csopt

    graph = csopt.new_tile_graph(split_enabled=[False, split, False, split])
    q_nope = graph.create_tiled_input((query_rows, ckv), (False, False), dtype="bfloat16")
    ckv_cache = graph.create_tiled_input((kv_proxy, ckv), (split, False), dtype="bfloat16")
    q_pe = graph.create_tiled_input((query_rows, kpe), (False, False), dtype="bfloat16")
    kpe_cache = graph.create_tiled_input((kv_proxy, kpe), (split, False), dtype="bfloat16")
    scores = graph.add(graph.matmul(q_nope, ckv_cache, trans_B=True),
                       graph.matmul(q_pe, kpe_cache, trans_B=True))
    weights = graph.exp(graph.div(scores, ckv**0.5))
    denom = graph.reduce_sum(weights, 1)
    graph.create_tiled_output(graph.matmul(graph.div(weights, denom), ckv_cache))
    return graph


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("name", sorted(OPERATORS))
def test_dense_body_replays_the_driver_builder_calls(monkeypatch, name, split):
    import cleave.superoptimizer as csopt

    def record(log):
        def new_tile_graph(**kw):
            log.append(("new_tile_graph", (), tuple(sorted(kw.items()))))
            return _Recorder(log)

        return new_tile_graph

    t = OPERATORS[name].tuned_tiling
    want, got = [], []
    monkeypatch.setattr(csopt, "new_tile_graph", record(want))
    if OPERATORS[name].is_mla:
        _driver_mla_graph(t["query_rows"], t["ckv"], t["kpe"], 4096, split=split)
    else:
        _driver_gqa_graph(t["query_rows"], t["head_dim"], 4096, split=split)
    monkeypatch.setattr(csopt, "new_tile_graph", record(got))
    registry_attention(name).dense_body().to_tile_graph(split=("kv_len",) if split else ())
    assert got == want and len(got) == (14 if OPERATORS[name].is_mla else 11)


# ---------------------------------------------------------------- MLA decode
MLA = "mla_paged_decode_h16_ckv512_kpe64_ps1"


def _mla():
    from demo.dynamic.mla_paged_decode import build as build_mla

    return trace_paged(build_mla()[0]).attention()


def test_mla_spec_equals_registry():
    att = _mla()
    assert att.is_mla and att.program.name == MLA
    assert att.paging_spec() == OPERATORS[MLA].spec
    with pytest.raises(SpecError, match="schedule knob"):
        att.paging_spec(cl.ScheduleHints(grid_order=("head", "batch")))
    knobs = cl.ScheduleHints(index_window_tiles=4, prefetch=False, rasterization=1)
    spec = att.paging_spec(knobs)
    assert (spec.index_window_tiles, spec.prefetch, spec.rasterization) == (4, False, 1)


# ---------------------------------------------------------------- prefill / ragged
def _prefill(causal=True, ragged_q=True, ragged_kv=False):
    b, tq, tkv, n = cl.dims("batch total_q total_kv n_idx", dynamic=True)
    ql, kl = cl.Dim("q_len", dynamic=True), cl.Dim("kv_len", dynamic=True, hint=4096)
    cache = [tkv, 8, 128] if ragged_kv else ["num_pages", 1, 8, 128]

    def op(q: cl.Tensor[tq if ragged_q else b, 32, 128], k: cl.Tensor[tuple(cache)],
           v: cl.Tensor[tuple(cache)], qo_indptr: cl.Tensor[b + 1, cl.i32],
           kv_indptr: cl.Tensor[b + 1, cl.i32], kv_indices: cl.Tensor[n, cl.i32],
           sm_scale: cl.Scalar[cl.f32]):
        pages = cl.PageTable(kv_indptr, None if ragged_kv else kv_indices, length=kl)
        kk = cl.paged_gather(k, pages).view(b, 8, 1, kl, 128)
        vv = cl.paged_gather(v, pages).view(b, 8, 1, kl, 128)
        if ragged_q:
            qq = cl.paged_gather(q, cl.PageTable(qo_indptr, length=ql)).view(b, 8, 4, ql, 128)
        else:
            qq = q.view(b, 8, 4, 1, 128)
        p, lse = cl.softmax((qq @ kk.mT) * sm_scale, causal=causal, return_lse=True)
        return (p @ vv).view(tq, 32, 128), lse.view(tq, 32)

    return trace_paged(op).attention()


def test_prefill_kind_and_outputs_follow_the_trace():
    from cleave.dynamic.paging import PAGED_PREFILL_CAUSAL, RAGGED_PREFILL_CAUSAL

    att = _prefill()
    assert att.is_prefill and att.causal and att.kind == PAGED_PREFILL_CAUSAL
    spec = att.paging_spec()
    assert (spec.query_indptr_input, spec.tokens_dim, spec.cache_inputs) == (
        "QO_INDPTR", "total_q", ("K", "V"))
    assert [[getattr(d, "name", d) for d in dims] for dims in att.output_dims()] == [
        ["total_q", 32, 128], ["total_q", 32]]
    assert att.query_rows == 64
    assert _prefill(ragged_kv=True).kind == RAGGED_PREFILL_CAUSAL


@pytest.mark.parametrize("kw, match", [
    (dict(causal=False), "causal"),
    (dict(ragged_q=False), "causal"),
    (dict(ragged_q=False, causal=False, ragged_kv=True), "ragged"),
])
def test_unbuilt_prefill_forms_are_refused(kw, match):
    with pytest.raises(SpecError, match=match):
        _prefill(**kw).paging_spec()


def test_ragged_table_takes_no_last_page_len():
    with pytest.raises(SpecError, match="last_page_len"):
        cl.PageTable("indptr", last_page_len="lens")


def test_dispatch_is_saved_with_the_kernel(tmp_path):
    from cleave.pipeline import load
    from cleave.pipeline.dynamic import PagedKernel

    att = attention("kv8")
    table = {"min_tiles": 3, "kv_tile": 64, "geomean_us": {"hybrid": 1.0}}
    kernel = PagedKernel(att, att.paging_spec(), "# best", label="L", variant="solo",
                         latency_us=1.0, dispatch=table,
                         dispatch_sources={"split": "# pair", "single": "# solo"})
    loaded = load(kernel.save(tmp_path / "k"))
    assert loaded.dispatch == table
    assert loaded.dispatch_sources == {"split": "# pair", "single": "# solo"}


# ---------------------------------------------------------------- doors
def test_trace_and_jit_compile_route_paged(monkeypatch):
    import cleave
    import cleave.pipeline
    from cleave.frontend.paged import PagedProgram

    fn, _ = build("ps64")
    prog = cleave.trace(fn)
    assert isinstance(prog, PagedProgram) and prog.dtype == cl.bf16
    assert fn.trace().attention().paging_spec() == prog.attention().paging_spec()
    with pytest.raises(SpecError, match="dynamic"):
        cleave.trace(fn, {"batch": 4})
    split = cleave.trace(fn, split=cl.Split("kv_len", (4,))).split
    assert split == (cl.Split("kv_len", (4,)),)

    seen = []
    monkeypatch.setattr(cleave.pipeline, "compile", lambda f, **kw: seen.append((f, kw)) or "K")
    assert fn.compile(preset="fast") == "K" and seen[0][0] is fn and seen[0][1]["preset"] == "fast"
    with pytest.raises(TypeError, match="dynamic"):
        fn.compile(batch=4)


def test_paged_kernel_save_load_round_trip(tmp_path):
    from cleave.pipeline import load
    from cleave.pipeline.dynamic import PagedKernel

    att = attention("ps64")
    kernel = PagedKernel(att, att.paging_spec(), "# source", label="L", variant="solo",
                         latency_us=1.5, meta={"expected": None})
    loaded = load(kernel.save(tmp_path / "k"))
    assert isinstance(loaded, PagedKernel) and loaded.source == "# source"
    assert loaded.spec == kernel.spec and loaded.input_names == kernel.input_names
    assert (loaded.label, loaded.variant, loaded.latency_us) == ("L", "solo", 1.5)
    assert [p.name for p in loaded.attention.inputs] == [p.name for p in att.inputs]
