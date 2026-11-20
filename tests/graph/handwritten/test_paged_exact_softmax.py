"""The exact (running-max) softmax of the paged decode operators that have one, on a GPU.

The captured workloads and the adversarial table run in ``test_handwritten.py`` and in the
end-to-end test of the operator. This file adds the hand-made cases neither contains --
requests with no keys at all, lengths on the KV tile's edges, a negative scale, a max that
sits in the first tile and never moves -- and the other direction: the same operator built
*without* the running max is right on a captured workload and non-finite on every
adversarial one, so those workloads are what tells the two kernels apart.

Everything is held to the operator's declared tolerance with the comparator every other
validation uses: an empty row's lse is -inf and has to be reproduced exactly, its output
exactly 0, and a NaN anywhere is a failure.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tilelang")
torch = pytest.importorskip("torch")

from cleave.core.errors import NonFiniteOutput
from cleave.dynamic import OPERATORS, operator_kernel_source
from cleave.dynamic import reference as ref
from cleave.runtime.autotune import _compiled, workload_key
from cleave.runtime.gpu import _gpu_exec_guard
from cleave.runtime.inputs import LARGE_LOGIT_FACTOR
from cleave.runtime.validate import compare_outputs

# The single-kernel GQA paged decode operators whose softmax is exact so far.
EXACT = sorted(
    name for name, op in OPERATORS.items()
    if op.exact_softmax and name.startswith("gqa_paged_decode") and op.page_size == 1
)
# Around the 64-key tile of the tuned kernels: one key, one short of a tile, exactly one,
# one more (a last tile holding a single key), two tiles, none, and many.
EDGE_KV_LENS = [1, 63, 64, 65, 128, 0, 700]


def _check(operator, kernel, inputs, reference):
    outputs = [torch.full_like(t, float("nan")) for t in reference]  # NaN unless written
    kernel(*inputs, *outputs)
    torch.cuda.synchronize()
    return compare_outputs(outputs, reference, OPERATORS[operator].tolerance)


def _workload(operator, kv_lens, *, scale=None, distribution="normal", flip_scale=False):
    """``(inputs, reference)`` for hand-picked request lengths.

    ``flip_scale`` negates the softmax scale *after* the inputs were shaped, so the logits
    of ``distribution`` come out mirrored: an increasing ramp becomes a decreasing one.
    """
    op = OPERATORS[operator]
    wl = (len(kv_lens), *op.dims[:3], sum(kv_lens) + 4, sum(kv_lens) + 4)
    inputs = ref.gqa_paged_decode_inputs(wl, kv_lens, scale=scale)
    inputs = ref.apply_distribution(operator, inputs, distribution)
    if flip_scale:
        inputs = [*inputs[:-1], -float(inputs[-1])]
    return inputs, ref.gqa_paged_decode_reference(wl, kv_lens)(*inputs)


def _empty_rows(kv_lens, reference):
    """The reference's own answer for the requests with no keys: 0 and -inf."""
    out, lse = reference
    empty = torch.tensor([n == 0 for n in kv_lens], device=lse.device)
    assert torch.equal(torch.isneginf(lse).all(dim=1), empty)
    assert not out[empty].any() and torch.isfinite(lse[~empty]).all()
    return int(empty.sum())


@pytest.mark.parametrize("operator", EXACT)
def test_exact_softmax_on_hand_made_requests(operator):
    kernel = _compiled(operator_kernel_source(operator))
    head_dim = OPERATORS[operator].dims[2]
    base = float(head_dim) ** -0.5
    cases = {
        # zero KV tiles for every block: the max keeps its start value, l = 0
        "all_empty": dict(kv_lens=[0, 0, 0]),
        "tile_edges": dict(kv_lens=EDGE_KV_LENS),
        "tile_edges@large_logits": dict(kv_lens=EDGE_KV_LENS, distribution="large_logits"),
        "tile_edges@large_scale": dict(kv_lens=EDGE_KV_LENS, distribution="large_scale"),
        "tile_edges@increasing": dict(kv_lens=EDGE_KV_LENS, distribution="increasing"),
        # at 65 keys the dominant one is alone in its tile; at 64 it closes the only tile
        "tile_edges@dominant_last": dict(kv_lens=EDGE_KV_LENS, distribution="dominant_last"),
        # a negative scale: the max has to be taken after scaling, not before
        "tile_edges@negative_scale": dict(kv_lens=EDGE_KV_LENS, scale=-base),
        "tile_edges@negative_large_scale": dict(
            kv_lens=EDGE_KV_LENS, scale=-base * LARGE_LOGIT_FACTOR),
        # decreasing along the keys: the max is in the first tile and never moves again,
        # so every later rescale factor is exp2(0) and must leave the sums as they are
        "tile_edges@decreasing": dict(
            kv_lens=EDGE_KV_LENS, distribution="increasing", flip_scale=True),
        # one long request among empty ones: 16k keys, 256 tiles
        "long@increasing": dict(kv_lens=[0, 16384, 0], distribution="increasing"),
        "long@decreasing": dict(
            kv_lens=[0, 16384, 0], distribution="increasing", flip_scale=True),
    }
    for name, kw in cases.items():
        with _gpu_exec_guard():
            inputs, reference = _workload(operator, **kw)
            n_empty = _empty_rows(kw["kv_lens"], reference)
            assert n_empty == kw["kv_lens"].count(0)
            try:
                _check(operator, kernel, inputs, reference)
            except AssertionError as error:
                raise AssertionError(f"{operator} on {name}: {error}") from error
        del inputs, reference
        torch.cuda.empty_cache()


@pytest.mark.parametrize("operator", EXACT)
def test_the_adversarial_logits_are_out_of_reach_of_a_plain_exp(operator):
    """What the adversarial workloads are for: their logits overflow fp32 ``exp`` (88.7)."""
    with _gpu_exec_guard():
        for distribution in ("large_scale", "increasing", "dominant_last"):
            _inputs, (_out, lse) = _workload(operator, [700, 65], distribution=distribution)
            # lse >= the largest logit, in log2 units
            assert float(lse.max()) > 88.7 * 1.4426950408889634, distribution


@pytest.mark.parametrize("operator", EXACT)
def test_without_the_running_max_the_adversarial_workloads_fail(operator, monkeypatch):
    """The operator's kernel as it was before its softmax was exact: ``exp2(s * scale2)``.

    It is right on a captured workload and non-finite on every adversarial distribution.
    That is the defect the running max fixes, and it is what gives the adversarial
    workloads their teeth: a kernel that loses the max cannot pass them.
    """
    from cleave.dynamic.paging.builders import decode

    op = OPERATORS[operator]
    exact = operator_kernel_source(operator)
    monkeypatch.setattr(decode, "_EXACT_SOFTMAX", frozenset())
    plain = operator_kernel_source(operator)
    assert "T.reduce_max(" in exact and "T.exp2(A[m0, m1] - M[m0])" in exact
    assert "T.reduce_max(" not in plain and "T.exp2(A[m0, m1] * scale2)" in plain

    kernels = {"exact": _compiled(exact), "plain": _compiled(plain)}
    captured = op.workloads[len(op.workloads) // 2]
    with _gpu_exec_guard():
        inputs, reference = op.reference(captured)
        for kernel in kernels.values():
            _check(operator, kernel, inputs, reference)
    del inputs, reference
    failed = {}
    for distribution, axes in op.adversarial:
        key = f"{workload_key(axes)}@{distribution}"
        with _gpu_exec_guard():
            inputs, reference = op.reference(axes, distribution=distribution)
            _check(operator, kernels["exact"], inputs, reference)
            with pytest.raises(NonFiniteOutput) as info:
                _check(operator, kernels["plain"], inputs, reference)
            failed[key] = str(info.value)
        del inputs, reference
        torch.cuda.empty_cache()
    assert len(failed) == len(op.adversarial) == 12
