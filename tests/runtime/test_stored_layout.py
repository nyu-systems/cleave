"""The stored layout of a kernel's outputs travels next to the source, never inside it.

``apply_tiling`` records ``(axes, pad)`` per output; ``build_candidate`` returns it with the
source, ``run_candidates`` hands it to the worker and puts it on the result, and validation
lays the reference out with it instead of guessing from the extents of two shapes.

``test_gpu_*`` needs a CUDA device (root conftest marks); everything else runs on CPU,
the worker's validation included (``check_on_cpu``).
"""

import multiprocessing as mp
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cleave.core.errors import OutputMismatch, ValidationError
from cleave.runtime.compile import build_candidate, build_candidate_code
from cleave.runtime.evaluate import run_candidates
from cleave.runtime.reference import (
    is_transposed,
    reconcile_output_layout,
    stored_shape,
    to_logical,
    to_stored,
    with_hand_off,
)
from cleave.runtime.validate import BenchOptions, Tolerance
from cleave.runtime.workers import WorkerPool

# A split partial stores (split, B*H, S_Q[, D]) for the logical (B*H, split, S_Q[, D]);
# the row sum is padded to the rank of the value output.
ROWSUM, VALUE = ((1, 0, 2), 1), ((1, 0, 2, 3), 0)


def _arange(*shape):
    return torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)


@pytest.mark.parametrize("heads, split", [(32, 32), (32, 16), (4, 4), (4, 8)])
def test_to_stored_and_to_logical_are_inverse_transpositions(heads, split):
    rowsum, value = _arange(heads, split, 64), _arange(heads, split, 64, 64)
    stored_rowsum, stored_value = to_stored(rowsum, ROWSUM), to_stored(value, VALUE)
    assert stored_rowsum.shape == (split, heads, 64, 1) == stored_shape(rowsum.shape, ROWSUM)
    assert stored_value.shape == (split, heads, 64, 64) == stored_shape(value.shape, VALUE)
    # position (s, b) of the stored tensor is entry (b, s) of the logical one
    assert torch.equal(stored_rowsum[3, 1, :, 0], rowsum[1, 3])
    assert torch.equal(stored_value[3, 1], value[1, 3])
    assert torch.equal(stored_value, value.movedim(1, 0))

    for stored, layout, logical in ((stored_rowsum, ROWSUM, rowsum), (stored_value, VALUE, value)):
        back = to_logical(stored.contiguous(), layout, logical.shape)
        assert torch.equal(back, logical) and back.is_contiguous()
    # A reshape is what the hand-off used to do. When the split extent equals the heads the
    # two shapes are equal and it hands the tensor over as it is: transposed.
    reshaped = stored_value.reshape(value.shape)
    assert not torch.equal(reshaped, value)


def test_identity_layout_only_pads():
    x = _arange(32, 4096)
    assert torch.equal(to_stored(x, ((0, 1), 1)), x.unsqueeze(-1))
    assert torch.equal(to_logical(x.unsqueeze(-1), ((0, 1), 1), x.shape), x)
    assert stored_shape((32, 8, 4096), ((0, 1, 2), 1)) == (32, 8, 4096, 1)


def test_layout_and_shape_mismatches_are_errors():
    x = _arange(4, 8, 16)
    with pytest.raises(ValueError, match="not a permutation"):
        to_stored(x, ((0, 0, 2), 0))
    with pytest.raises(ValueError, match="not a permutation"):
        to_stored(x, ((1, 0), 0))  # a record for a rank-2 output
    with pytest.raises(ValueError, match="not a permutation"):
        stored_shape(x.shape, ((0, 1, 2), -1))
    with pytest.raises(ValueError, match="is not an output of logical shape"):
        to_logical(x, ((1, 0, 2), 0), x.shape)  # x is (4, 8, 16), the record says (8, 4, 16)


def test_reconcile_uses_the_record_where_extents_cannot_tell():
    # split == B*H: both layouts have the same shape. Without the record the reference is
    # returned as it is and a correct kernel is compared transposed.
    value = _arange(32, 32, 64, 64)
    assert reconcile_output_layout(value, (32, 32, 64, 64)) is value
    stored = reconcile_output_layout(value, (32, 32, 64, 64), VALUE)
    assert torch.equal(stored, value.movedim(1, 0)) and not torch.equal(stored, value)
    rowsum = _arange(32, 32, 64)
    assert torch.equal(reconcile_output_layout(rowsum, (32, 32, 64, 1)), rowsum.unsqueeze(-1))
    assert torch.equal(reconcile_output_layout(rowsum, (32, 32, 64, 1), ROWSUM),
                       rowsum.movedim(1, 0).unsqueeze(-1))

    # S_Q == D: the extents do not determine the permutation and the extent path gives up.
    value = _arange(32, 16, 64, 64)
    with pytest.raises(ValueError, match="Ambiguous output layout"):
        reconcile_output_layout(value, (16, 32, 64, 64))
    assert torch.equal(reconcile_output_layout(value, (16, 32, 64, 64), VALUE),
                       value.movedim(1, 0))


def test_reconcile_without_a_record_still_matches_distinct_extents():
    # Hand-written kernels have no schedule to record anything; their extents are distinct.
    value = _arange(32, 4, 128, 64)
    by_extent = reconcile_output_layout(value, (4, 32, 128, 64))
    assert torch.equal(by_extent, reconcile_output_layout(value, (4, 32, 128, 64), VALUE))
    rowsum = _arange(32, 4, 128)
    assert torch.equal(reconcile_output_layout(rowsum, (4, 32, 128, 1)),
                       reconcile_output_layout(rowsum, (4, 32, 128, 1), ROWSUM))


def test_reconcile_refuses_a_record_that_is_not_the_kernels_layout():
    with pytest.raises(ValueError, match="but the kernel returned"):
        reconcile_output_layout(_arange(32, 16, 64, 64), (32, 16, 64, 64), VALUE)
    with pytest.raises(ValueError, match="but the kernel returned"):
        reconcile_output_layout(_arange(32, 16, 64), (16, 32, 64), ROWSUM)  # pad of 1 missing


def test_a_transposed_output_is_timed_with_its_hand_off():
    """A kernel that stores an output transposed is only ever read through ``to_logical``,
    which copies; that copy is part of what a launch costs, so it is in what gets timed.
    Padding alone is a view and leaves the kernel as it is."""
    assert is_transposed(ROWSUM) and is_transposed(VALUE)
    assert not is_transposed(((0, 1, 2), 1)) and not is_transposed(((0, 1), 0))

    rowsum, value = _arange(4, 8, 64), _arange(4, 8, 64, 64)
    stored = (to_stored(rowsum, ROWSUM).contiguous(), to_stored(value, VALUE).contiguous())
    kernel = lambda *inputs: stored  # noqa: E731
    timed = with_hand_off(kernel, (ROWSUM, VALUE), [rowsum.shape, value.shape])
    assert timed is not kernel
    handed = timed()
    assert all(torch.equal(got, want) and got.is_contiguous()
               for got, want in zip(handed, (rowsum, value)))

    # nothing transposed (a solo kernel, a combine, a single-output partial), or no record
    for layouts in (None, (), (((0, 1, 2), 1), ((0, 1, 2, 3), 0))):
        assert with_hand_off(kernel, layouts, [rowsum.shape, value.shape]) is kernel
    single = lambda *inputs: stored[1]  # noqa: E731 - a kernel returning one tensor
    (only,) = with_hand_off(single, (VALUE,), [value.shape])()
    assert torch.equal(only, value)
    with pytest.raises(ValueError, match="2 stored layouts for 1 outputs"):
        with_hand_off(kernel, (ROWSUM, VALUE), [value.shape])


# ------------------------------------------------------------------ build_candidate (CPU)

def _split_partial_candidate():
    from cleave.schedule import search_tiling
    from tests._graphs import mha_split_partial_graph

    return search_tiling(mha_split_partial_graph(1, 32, 4, 4096, 1024, 64), topk=1)[0]


def test_build_candidate_returns_the_layouts_next_to_the_source():
    pytest.importorskip("tilelang")
    cand = _split_partial_candidate()
    code, layouts = build_candidate(cand)
    assert layouts == (ROWSUM, VALUE)  # in the order the kernel returns its outputs
    assert code == build_candidate_code(cand)
    assert "OutputNode_0: T.Tensor((4, 32, 4096, 1)" in code
    assert "OutputNode_1: T.Tensor((4, 32, 4096, 64)" in code
    # out-of-band: generated text is hashed and diffed, the record is not part of it
    assert "stored_layouts" not in code and "(1, 0, 2" not in code


# ------------------------------------------------------------------ carriage (fork pool)

fork = pytest.mark.skipif("fork" not in mp.get_all_start_methods(),
                          reason="requires fork multiprocessing support")


def _no_reference(cache_dir, graph, dtype):
    return [-1], 0


def _build_with_layouts(cand):
    return cand.code, cand.layouts


def _build_source_only(cand):
    return cand.code


def _check_takes_layouts(code_str, *, out_idx, reference_cache_dir, tolerance=None,
                         benchmark=None, stored_layouts=None):
    if code_str == "wrong":
        raise AssertionError("mismatch")
    return repr(stored_layouts), 0.0


def _check_without_the_keyword(code_str, *, out_idx, reference_cache_dir, tolerance=None,
                               benchmark=None):
    return f"cuda_{code_str}", 0.0


def _candidates(*codes):
    return [SimpleNamespace(code=c, layouts=(ROWSUM, VALUE), rstep_map={"id": i},
                            output_tile=(i,), flat_graph=None)
            for i, c in enumerate(codes, start=1)]


@fork
def test_run_candidates_carries_the_layouts_to_the_worker_and_the_result():
    with WorkerPool(mp.get_context("fork")) as pool:
        ok, wrong = run_candidates(
            _candidates("a", "wrong"), "float16", build_fn=_build_with_layouts,
            check_fn=_check_takes_layouts, reference_fn=_no_reference, pool=pool, max_parallel=1)
    assert ok.ok and ok.cuda_src == repr((ROWSUM, VALUE))  # what the worker was handed
    assert ok.stored_layouts == (ROWSUM, VALUE) and ok.code == "a"
    # a failed candidate keeps its layouts too: the report can still describe its kernel
    assert isinstance(wrong.error, ValidationError) and wrong.stored_layouts == (ROWSUM, VALUE)


@fork
def test_a_build_fn_without_layouts_and_a_check_fn_without_the_keyword_keep_working():
    with WorkerPool(mp.get_context("fork")) as pool:
        (plain,) = run_candidates(
            _candidates("a"), "float16", build_fn=_build_source_only,
            check_fn=_check_without_the_keyword, reference_fn=_no_reference, pool=pool)
        (unset,) = run_candidates(
            _candidates("a"), "float16", build_fn=_build_source_only,
            check_fn=_check_takes_layouts, reference_fn=_no_reference, pool=pool)
    assert plain.ok and plain.cuda_src == "cuda_a" and plain.stored_layouts is None
    assert unset.ok and unset.cuda_src == "None" and unset.stored_layouts is None


# ------------------------------------------------------------------ validation

class _FakeKernel:
    """Stands in for a compiled kernel without inputs: it returns what it was given."""

    def __init__(self, *outputs):
        self.outputs = outputs
        self.adapter = SimpleNamespace(prim_func=SimpleNamespace(params=[], buffer_map={}),
                                       result_idx=[])

    def __call__(self, *inputs):
        return self.outputs

    def get_kernel_source(self):
        return "cuda source"


@pytest.fixture
def check_on_cpu(monkeypatch):
    """The candidate check of a worker with the device taken out of it.

    ``check(reference, kernel)`` registers one candidate and returns the source text to
    hand to ``check_code``: "compiling" that text yields ``kernel``, and the reference the
    worker would load from disk is ``reference``. The GPU lock and the CUDA cache calls do
    nothing. Everything between the task a worker receives and the comparison is the real
    code: ``_run_task`` -> ``check_code`` -> ``_check_kernel`` ->
    ``_validate_kernel_outputs`` -> ``reconcile_output_layout`` -> ``compare_outputs``.
    """
    import contextlib

    from cleave.runtime import compile as runtime_compile
    from cleave.runtime import validate

    registered = {}

    def register(reference, kernel):
        code = f"candidate {len(registered)}"
        registered[code] = (kernel, [("normal", {}, list(reference))])
        return code

    monkeypatch.setattr(runtime_compile, "_create_parallel_env", lambda: None)
    monkeypatch.setattr(runtime_compile, "_load_main_kernel_from_code", lambda code: code)
    monkeypatch.setattr(runtime_compile, "_compile", lambda code, out_idx: registered[code][0])
    monkeypatch.setattr(validate, "_gpu_exec_guard", contextlib.nullcontext)
    monkeypatch.setattr(validate, "_load_reference", lambda code: registered[code][1])
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    return register


@pytest.mark.parametrize("heads, split, without_the_record", [
    (4, 4, "OutputMismatch"),  # split == B*H: one shape for both layouts, compared transposed
    (4, 8, "ValueError"),      # S_Q == D: "Ambiguous output layout reconciliation"
    (32, 32, "OutputMismatch"),
    (32, 16, "ValueError"),
])
def test_a_worker_validates_a_split_first_partial_only_with_its_record(
        check_on_cpu, heads, split, without_the_record):
    """The carriage of the record into validation, without a device: a kernel that is right
    passes the worker's check exactly when the task carries its stored layouts. Without
    them these are the two failures every static split-attention partial had."""
    from cleave.runtime.compile import check_code
    from cleave.runtime.workers import _run_task

    torch.manual_seed(0)
    rowsum, value = torch.rand(heads, split, 64).half(), torch.randn(heads, split, 64, 64).half()
    kernel = _FakeKernel(to_stored(rowsum, ROWSUM).contiguous(),
                         to_stored(value, VALUE).contiguous())
    code = check_on_cpu([rowsum, value], kernel)
    task = {"check_fn": check_code, "code_str": code, "out_idx": [-2, -1],
            "reference_cache_dir": code, "tolerance": Tolerance(rtol=1e-3), "benchmark": None}

    result = _run_task({**task, "stored_layouts": (ROWSUM, VALUE)})
    assert result["ok"], result.get("error_message")
    assert result["cuda_src"] == "cuda source" and result["latency_ms"] is None
    assert [m.rel for m in result["measured"]] == [0.0, 0.0]

    for bare in (task, {**task, "stored_layouts": None}):  # no record: the extent path
        result = _run_task(bare)
        assert not result["ok"] and result["stage"] == "validate"
        assert result["error_type"] == without_the_record
        assert ("Ambiguous output layout" in result["error_message"]) == (heads != split)

    # the record of another kernel is an error, never a silent fallback to the extents
    result = _run_task({**task, "stored_layouts": (VALUE,)})
    assert not result["ok"] and "1 stored layouts were recorded" in result["error_message"]
    result = _run_task({**task, "stored_layouts": (((0, 1, 2), 1), ((0, 1, 2, 3), 0))})
    if heads == split:  # the logical layout has the same shape: a real mismatch
        assert result["error_type"] == "OutputMismatch"
    else:
        assert "but the kernel returned" in result["error_message"]


def test_gpu_a_split_first_partial_validates_only_with_its_record():
    """The two failures of a static split-attention partial, on a kernel that is right.

    ``split == B*H``: the extents cannot tell the layouts apart and the reference used to be
    compared transposed. ``S_Q == D`` with any other split: the extent matching used to
    raise. With the record both validate, exactly.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from cleave.runtime import validate
    from cleave.runtime.gpu import _gpu_exec_guard

    torch.manual_seed(0)
    tolerance = Tolerance(rtol=1e-3)
    for heads, split in ((4, 4), (4, 8)):
        rowsum = torch.rand(heads, split, 64, device="cuda").half()
        value = torch.randn(heads, split, 64, 64, device="cuda").half()
        kernel = _FakeKernel(to_stored(rowsum, ROWSUM).contiguous(),
                             to_stored(value, VALUE).contiguous())
        with _gpu_exec_guard():
            measured = validate._validate_kernel_outputs(
                kernel, (), [rowsum, value], tolerance=tolerance,
                stored_layouts=(ROWSUM, VALUE))
            assert [m.rel for m in measured] == [0.0, 0.0]
            error = OutputMismatch if heads == split else ValueError
            with pytest.raises(error, match="out of tolerance|Ambiguous output layout"):
                validate._validate_kernel_outputs(kernel, (), [rowsum, value],
                                                  tolerance=tolerance)
            with pytest.raises(ValueError, match="2 outputs"):
                validate._validate_kernel_outputs(kernel, (), [rowsum, value],
                                                  tolerance=tolerance, stored_layouts=(VALUE,))


def test_a_worker_times_a_split_first_partial_with_its_hand_off(check_on_cpu, monkeypatch):
    """The latency a worker reports for a split-first partial covers the launch and the
    copy to the logical layout; the sum of a split variant's stage timings is then what a
    chained call costs. A kernel that stores its outputs as the graph has them is timed
    bare."""
    from cleave.runtime import validate
    from cleave.runtime.compile import check_code
    from cleave.runtime.workers import _run_task

    timed = []

    def fake_benchmark(kernel, inputs, *, warmup_iters, profile_iters):
        outputs = kernel(*inputs)
        timed.append((kernel, [tuple(out.shape) for out in outputs]))
        return 0.125

    monkeypatch.setattr(validate, "benchmark_compiled_kernel", fake_benchmark)
    rowsum, value = torch.rand(4, 8, 64).half(), torch.randn(4, 8, 64, 64).half()
    partial = _FakeKernel(to_stored(rowsum, ROWSUM).contiguous(),
                          to_stored(value, VALUE).contiguous())
    logical = _FakeKernel(rowsum.unsqueeze(-1), value)  # the layout of GQA at split == H_KV
    task = {"check_fn": check_code, "out_idx": [-2, -1], "tolerance": Tolerance(rtol=1e-3),
            "benchmark": BenchOptions(warmup=1, iters=2)}

    code = check_on_cpu([rowsum, value], partial)
    result = _run_task({**task, "code_str": code, "reference_cache_dir": code,
                        "stored_layouts": (ROWSUM, VALUE)})
    assert result["ok"] and result["latency_ms"] == 0.125
    code = check_on_cpu([rowsum, value], logical)
    result = _run_task({**task, "code_str": code, "reference_cache_dir": code,
                        "stored_layouts": (((0, 1, 2), 1), ((0, 1, 2, 3), 0))})
    assert result["ok"] and result["latency_ms"] == 0.125

    (with_copy, handed), (bare, returned) = timed
    assert with_copy is not partial and handed == [(4, 8, 64), (4, 8, 64, 64)]
    assert bare is logical and returned == [(4, 8, 64, 1), (4, 8, 64, 64)]


def test_the_serial_re_time_includes_the_hand_off_too(monkeypatch):
    """``benchmark_results`` re-times a result in this process; it must time what the
    workers time, or a re-timed split partial would lose its hand-off again."""
    import contextlib

    from cleave.runtime import evaluate
    from cleave.runtime.reference import output_order
    from tests._graphs import mha_split_partial_graph

    graph = mha_split_partial_graph(1, 4, 8, 64, 16, 64)  # B*H=4, split 8, S_Q=64, D=64
    logical = [tuple(int(s) for s in node.get_shape(0)) for node in output_order(graph)]
    assert logical == [(4, 8, 64), (4, 8, 64, 64)]
    kernel = _FakeKernel(*(torch.zeros(stored_shape(shape, layout))
                           for shape, layout in zip(logical, (ROWSUM, VALUE))))
    timed = []

    def fake_benchmark(timed_kernel, inputs, *, warmup_iters, profile_iters):
        timed.append([tuple(out.shape) for out in timed_kernel(*inputs)])
        return 0.25

    monkeypatch.setattr(evaluate, "_gpu_exec_guard", contextlib.nullcontext)
    monkeypatch.setattr(evaluate, "_load_main_kernel_from_code", lambda code: code)
    monkeypatch.setattr(evaluate, "_compile", lambda code, out_idx: kernel)
    monkeypatch.setattr(evaluate, "make_inputs", lambda graph, dtype: [])
    monkeypatch.setattr(evaluate, "order_inputs_for_kernel", lambda inputs, kernel: [])
    monkeypatch.setattr(evaluate, "benchmark_compiled_kernel", fake_benchmark)
    for name in ("synchronize", "empty_cache", "ipc_collect"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)

    cand = SimpleNamespace(flat_graph=graph)
    results = [
        evaluate.CandidateResult(cand, "ok", code="partial", stored_layouts=(ROWSUM, VALUE)),
        evaluate.CandidateResult(cand, "ok", code="partial"),  # built without the record
    ]
    retimed = evaluate.benchmark_results(results, "float16", warmup=1, iters=2)
    assert [r.latency_ms for r in retimed] == [0.25, 0.25] and all(r.ok for r in retimed)
    assert timed == [logical, [(8, 4, 64, 1), (8, 4, 64, 64)]]
