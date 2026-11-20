"""The exact (running-max) softmax of every paged operator, on a GPU.

The captured workloads and the adversarial table run in ``test_handwritten.py``, in
``test_splitkv_decode.py`` and in the end-to-end test of the first operator. This file adds
the hand-made cases none of them contains -- requests with no keys at all, lengths on the
KV tile's edges, causal windows that open late or not at all, a negative scale, a max that
sits in the first tile and never moves -- for every operator and for the split-KV pair of
every decode operator, and the other direction: the same kernel with the running max
taken out of its text is right on every captured workload and non-finite on the
adversarial ones, so those workloads are what tells the two kernels apart.

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
from tests.graph.apply_paging.test_split_decode import CASES as SPLIT_CASES
from tests.graph.apply_paging.test_split_decode import split_kernel_source

LOG2E = 1.4426950408889634
# Around the 64-key tile of the decode kernels: one key, one short of a tile, exactly one,
# one more (a last tile holding a single key), two tiles, none, and many. For a split-KV
# pair at 8 splits the same lengths leave trailing splits empty (65 keys are 2 tiles, so 6
# splits run none), every split empty (0), and the tiles dealt unevenly (700 keys are 11).
EDGE_KV_LENS = [1, 63, 64, 65, 128, 0, 700]
# (query tokens, keys) per request of a causal prefill. Row r of a request sees keys
# 0 .. kv_len - q_len + r: one key in the first row of a square request, a prefix that
# crosses KV tiles, a window that opens only at row 10 (kv_len < q_len: the rows before it
# have no key and are defined as 0 and -inf), no keys at all, and two requests longer than
# one query tile of any of the prefill kernels (16, 8, 64 and 1 tokens).
#
# No request has between 2 and a dozen keys. A ramp (``increasing``) spans 128 logit units
# whatever the request's length, so the keys of a 5-key request are 32 apart; mirrored
# (``decreasing``), the output of an MLA row -- whose values are the keys themselves, the
# first of them 0 -- is then the exp(-32 a) tail of its second key: 1e-38 for some heads,
# the bottom of float32's range, where a kernel compiled with fast math flushes to zero and
# the reference keeps a denormal. That is float32 underflow, not the softmax.
EDGE_PREFILL_LENS = [(1, 1), (24, 24), (3, 70), (40, 30), (20, 0), (17, 145), (70, 100)]


def _check(operator, kernel, inputs, reference):
    outputs = [torch.full_like(t, float("nan")) for t in reference]  # NaN unless written
    kernel(*inputs, *outputs)
    torch.cuda.synchronize()
    return compare_outputs(outputs, reference, OPERATORS[operator].tolerance)


def _is_prefill(operator) -> bool:
    return "prefill" in operator


def _scale(operator) -> float:
    """The softmax scale the operator's captured workloads pass."""
    op = OPERATORS[operator]
    return float(op.scale) if op.scale is not None else float(op.dims[2]) ** -0.5


def _workload(operator, lens, *, scale=None, distribution="normal", flip_scale=False):
    """``(inputs, reference)`` for hand-picked request lengths: KV lengths for a decode
    operator, ``(q_len, kv_len)`` pairs for a prefill one.

    ``flip_scale`` negates the softmax scale *after* the inputs were shaped, so the logits
    of ``distribution`` come out mirrored: an increasing ramp becomes a decreasing one.
    """
    op = OPERATORS[operator]
    scale = _scale(operator) if scale is None else scale
    if operator.startswith("gqa_paged_decode"):
        pages = sum(-(-n // op.page_size) for n in lens)
        wl = (len(lens), *op.dims[:3], pages + 4, pages + 4)
        inputs = ref.gqa_paged_decode_inputs(wl, lens, page_size=op.page_size, scale=scale)
        expected = ref.gqa_paged_decode_reference(wl, lens, page_size=op.page_size)
    elif operator.startswith("mla_paged_decode"):
        wl = (len(lens), *op.dims, sum(lens) + 4, sum(lens) + 4)
        inputs = ref.mla_paged_decode_inputs(wl, lens, scale=scale)
        expected = ref.mla_paged_decode_reference(wl, lens)
    else:
        q_lens, kv_lens = [q for q, _ in lens], [kv for _, kv in lens]
        total_q, n_idx = sum(q_lens), sum(kv_lens)
        if operator.startswith("gqa_ragged"):
            # Four spare key rows no request owns, like the spare pages of the paged
            # operators: with every request empty the key tensors would have no row at
            # all, and a zero-sized tensor has no TMA descriptor to launch the kernel with.
            wl = (*op.dims, total_q, n_idx + 4)
            inputs = ref.gqa_ragged_prefill_inputs(wl, q_lens, kv_lens, scale=scale)
            expected = ref.gqa_ragged_prefill_reference(wl, q_lens, kv_lens)
        elif operator.startswith("mla_paged_prefill"):
            wl = (*op.dims, n_idx + 4, n_idx + 4, total_q)
            inputs = ref.mla_paged_prefill_inputs(wl, q_lens, kv_lens, scale=scale)
            expected = ref.mla_paged_prefill_reference(wl, q_lens, kv_lens)
        else:
            wl = (*op.dims, n_idx + 4, n_idx + 4, total_q)
            inputs = ref.gqa_paged_prefill_inputs(wl, q_lens, kv_lens, scale=scale)
            expected = ref.gqa_paged_prefill_reference(wl, q_lens, kv_lens)
    inputs = ref.apply_distribution(operator, inputs, distribution)
    if flip_scale:
        inputs = [*inputs[:-1], -float(inputs[-1])]
    return inputs, expected(*inputs)


def _rows_without_a_key(operator, lens) -> list:
    """Per output row, whether it has no live key: a decode request with none, or a prefill
    row whose causal window is empty."""
    if not _is_prefill(operator):
        return [n == 0 for n in lens]
    return [kv_len - q_len + row < 0 or kv_len == 0
            for q_len, kv_len in lens for row in range(q_len)]


def _empty_rows(operator, lens, reference) -> int:
    """The reference's own answer for the rows with no key: exactly 0 and -inf."""
    out, lse = reference
    empty = torch.tensor(_rows_without_a_key(operator, lens), device=lse.device)
    assert empty.shape == lse.shape[:1]
    assert torch.equal(torch.isneginf(lse).all(dim=1), empty)
    assert torch.equal(torch.isneginf(lse).any(dim=1), empty)
    assert not out[empty].any() and torch.isfinite(lse[~empty]).all()
    return int(empty.sum())


def _hand_made_cases(operator) -> dict:
    """name -> ``_workload`` arguments, and how many rows without a key each must have."""
    prefill = _is_prefill(operator)
    edges = EDGE_PREFILL_LENS if prefill else EDGE_KV_LENS
    base = _scale(operator)
    # one long request among empty ones: 256 decode tiles; a prefill whose rows see between
    # 3701 and 4000 keys
    long = [(0, 0), (300, 4000), (5, 0)] if prefill else [0, 16384, 0]
    return {
        # zero KV tiles for every block: the max keeps its start value, l = 0
        "all_empty": dict(lens=[(4, 0), (0, 0), (9, 0)] if prefill else [0, 0, 0]),
        "tile_edges": dict(lens=edges),
        "tile_edges@large_logits": dict(lens=edges, distribution="large_logits"),
        "tile_edges@large_scale": dict(lens=edges, distribution="large_scale"),
        "tile_edges@increasing": dict(lens=edges, distribution="increasing"),
        # at 65 keys the dominant one is alone in its tile; at 64 it closes the only tile.
        # Under a causal mask only the last query row sees it: one row's max jumps in the
        # last tile while every other row of the same block has that column masked.
        "tile_edges@dominant_last": dict(lens=edges, distribution="dominant_last"),
        # a negative scale: the max has to be taken after scaling, not before
        "tile_edges@negative_scale": dict(lens=edges, scale=-base),
        "tile_edges@negative_large_scale": dict(lens=edges, scale=-base * LARGE_LOGIT_FACTOR),
        # decreasing along the keys: the max is in the first tile and never moves again,
        # so every later rescale factor is exp2(0) and must leave the sums as they are
        "tile_edges@decreasing": dict(lens=edges, distribution="increasing", flip_scale=True),
        "long@increasing": dict(lens=long, distribution="increasing"),
        "long@decreasing": dict(lens=long, distribution="increasing", flip_scale=True),
    }


def _run_hand_made(operator, kernel):
    for name, kw in _hand_made_cases(operator).items():
        with _gpu_exec_guard():
            inputs, reference = _workload(operator, **kw)
            n_empty = _empty_rows(operator, kw["lens"], reference)
            assert n_empty == sum(_rows_without_a_key(operator, kw["lens"]))
            if name == "all_empty":
                assert n_empty == reference[1].shape[0] > 0
            elif name.startswith("tile_edges"):
                assert 0 < n_empty < reference[1].shape[0]
            try:
                _check(operator, kernel, inputs, reference)
            except AssertionError as error:
                raise AssertionError(f"{operator} on {name}: {error}") from error
        del inputs, reference
        torch.cuda.empty_cache()


@pytest.mark.parametrize("operator", sorted(OPERATORS))
def test_exact_softmax_on_hand_made_requests(operator):
    _run_hand_made(operator, _compiled(operator_kernel_source(operator)))


@pytest.mark.parametrize("operator", sorted(SPLIT_CASES))
@pytest.mark.parametrize("step", [1, 2])
def test_exact_split_pair_on_hand_made_requests(operator, step):
    """The split-KV pair on the same requests: each split carries its own running max, and
    the combine weighs the splits by them -- one partial per iteration (the elementwise
    running max) and two (a reduction over the slice). Trailing splits that run no tile,
    requests in which no split runs one, and a dominant key that lands in one split."""
    kernel = _compiled(split_kernel_source(operator, step=step))
    assert [p[1] for p in kernel.scratch_params][::2] == [
        "Partial_ROWSUM", "Partial_VALUE", "Partial_MAX"]
    _run_hand_made(operator, kernel)


@pytest.mark.parametrize("operator", sorted(OPERATORS))
def test_the_adversarial_logits_are_out_of_reach_of_a_plain_exp(operator):
    """What the adversarial workloads are for: their logits overflow fp32 ``exp`` (88.7)."""
    lens = [(3, 700), (2, 65)] if _is_prefill(operator) else [700, 65]
    with _gpu_exec_guard():
        for distribution in ("large_scale", "increasing", "dominant_last"):
            _inputs, (_out, lse) = _workload(operator, lens, distribution=distribution)
            # lse >= the largest logit, in log2 units
            assert float(lse.max()) > 88.7 * LOG2E, distribution


# The running max, taken out of an exact kernel's text: P = exp2(score), nothing is
# rescaled, lse = log2(l). That is the kernel every operator had before its softmax was
# exact. (The max is still computed; nothing reads it.)
_WITHOUT_THE_MAX = (
    ("T.exp2(A[m0, m1] - M[m0])", "T.exp2(A[m0, m1])"),
    ("T.log2(A[i0]) + M[i0]", "T.log2(A[i0])"),
    ("ACC[i0] *= S[i0]", "ACC[i0] *= 1.0"),
    ("ACC[i0, i1] *= S[i0]", "ACC[i0, i1] *= 1.0"),
)


def _without_the_running_max(source: str) -> str:
    plain = source
    for old, new in _WITHOUT_THE_MAX:
        plain = plain.replace(old, new)
    assert "- M[" not in plain and "+ M[" not in plain and "*= S[" not in plain
    # one shifted exp, one lse and at least one rescale were there to take out
    assert source.count(_WITHOUT_THE_MAX[0][0]) == source.count(_WITHOUT_THE_MAX[1][0]) == 1
    assert source.count("*= S[") >= 1 and "T.reduce_max(" in source
    return plain


def _longest_request(operator, inputs) -> int:
    """Keys of the longest request, off the KV indptr (in pages for a slab operator)."""
    family = next(f for f in ref._SCORE_INPUTS if operator.startswith(f))
    ptr = inputs[ref._SCORE_INPUTS[family][1]].tolist()
    return max(b - a for a, b in zip(ptr, ptr[1:]))


@pytest.mark.parametrize("operator", sorted(OPERATORS))
def test_without_the_running_max_the_adversarial_workloads_fail(operator):
    """The operator's kernel as it was before its softmax was exact: ``exp2(s * scale2)``.

    It is right on a captured workload and non-finite on the adversarial ones. That is the
    defect the running max fixes, and it is what gives the adversarial workloads their
    teeth: a kernel that loses the max cannot pass them. The one adversarial workload it
    can pass is an ``increasing`` one whose requests hold a single key each -- there is
    nothing to increase along, so its logits are 0.
    """
    op = OPERATORS[operator]
    exact = operator_kernel_source(operator)
    plain = _without_the_running_max(exact)

    kernels = {"exact": _compiled(exact), "plain": _compiled(plain)}
    captured = op.workloads[len(op.workloads) // 2]
    with _gpu_exec_guard():
        inputs, reference = op.reference(captured)
        for kernel in kernels.values():
            _check(operator, kernel, inputs, reference)
    del inputs, reference
    failed, passed = {}, []
    for distribution, axes in op.adversarial:
        key = f"{workload_key(axes)}@{distribution}"
        with _gpu_exec_guard():
            inputs, reference = op.reference(axes, distribution=distribution)
            _check(operator, kernels["exact"], inputs, reference)
            flat = (distribution == "increasing" and op.page_size == 1
                    and _longest_request(operator, inputs) == 1)
            if flat:
                _check(operator, kernels["plain"], inputs, reference)
                passed.append(key)
            else:
                with pytest.raises(NonFiniteOutput) as info:
                    _check(operator, kernels["plain"], inputs, reference)
                failed[key] = str(info.value)
        del inputs, reference
        torch.cuda.empty_cache()
    assert len(failed) + len(passed) == len(op.adversarial) == 12
    # at most the smallest captured workload is too small to carry a ramp
    assert len(passed) <= 1 and len(failed) >= 11
    assert {key.partition("@")[2] for key in failed} == {
        "large_logits", "large_scale", "increasing", "dominant_last"}
