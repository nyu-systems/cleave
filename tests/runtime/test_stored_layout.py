"""The stored layout of a kernel's outputs travels next to the source, never inside it.

``apply_tiling`` records ``(axes, pad)`` per output; ``build_candidate`` returns it with the
source, ``run_candidates`` hands it to the worker and puts it on the result, and validation
lays the reference out with it instead of guessing from the extents of two shapes.

``test_gpu_*`` needs a CUDA device (root conftest marks); everything else runs on CPU.
"""

import multiprocessing as mp
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cleave.core.errors import OutputMismatch, ValidationError
from cleave.runtime.compile import build_candidate, build_candidate_code
from cleave.runtime.evaluate import run_candidates
from cleave.runtime.reference import (
    reconcile_output_layout,
    stored_shape,
    to_logical,
    to_stored,
)
from cleave.runtime.validate import Tolerance
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


# ------------------------------------------------------------------ validation (GPU)

class _FakeKernel:
    def __init__(self, *outputs):
        self.outputs = outputs

    def __call__(self, *inputs):
        return self.outputs


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
