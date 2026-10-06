"""Inputs and references for the paged/ragged attention operators.

Each ``*_inputs`` builds exactly the tensors the operator's SOL-ExecBench definition
declares, in the declared order, and each ``*_reference`` closure takes that same
argument list -- so the reference reads as the definition's own ``run`` and the kernel's
parameter list is checked by construction rather than by comment.

A query row whose causal window is empty gets **0 in the output and -inf in the lse**,
which is what the definitions themselves produce (they preallocate ``zeros``/``-inf`` and
skip such requests) and what every flash-attention implementation writes. It matters
beyond taste: SOL-ExecBench's ``check_tensor_sanity`` rejects NaN unconditionally.
"""

from __future__ import annotations

import torch

SEED = 0
BF16 = torch.bfloat16
NEG_INF = float("-inf")


def _indptr(lengths):
    out = [0]
    for n in lengths:
        out.append(out[-1] + n)
    return out


def _i32(values, device="cuda"):
    return torch.tensor(list(values), device=device, dtype=torch.int32)


def _randn(*shape, dtype=BF16, device="cuda", gen=None):
    return torch.randn(*shape, device=device, dtype=dtype, generator=gen)


def _generator(device="cuda"):
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    return gen


def _page_table(num_pages, count, device="cuda"):
    """A random permutation of page ids, so the gather is genuinely scattered."""
    gen = _generator(device)
    return torch.randperm(num_pages, device=device, generator=gen)[:count].to(
        torch.int32
    )


def _pad_indices(kv_indices, n_idx):
    """Grow the page table to the workload's declared ``num_kv_indices`` extent."""
    short = n_idx - kv_indices.numel()
    assert short >= 0, (kv_indices.numel(), n_idx)
    if not short:
        return kv_indices
    pad = torch.zeros(short, device=kv_indices.device, dtype=torch.int32)
    return torch.cat([kv_indices, pad])


# Live fp32 logits are held under this many elements. The captured workloads reach 16k
# query tokens against 16k keys, whose full (n_q, n_head, n_kv) logits would be tens of GiB.
CHUNK_ELEMS = 1 << 27


def _q_chunk(n_q, n_head, n_kv):
    return max(1, min(n_q, CHUNK_ELEMS // max(1, n_head * n_kv)))


def _finish(logits):
    """Softmax weights and the 2-based lse of one (n_q, n_head, n_kv) logit block.

    Rows with no live column come back with zero weight everywhere and a -inf lse,
    instead of the NaN a plain softmax over all -inf would give.
    """
    lse = torch.logsumexp(logits, dim=-1) / torch.log(torch.tensor(2.0))
    weights = torch.softmax(logits, dim=-1)
    blind = torch.isneginf(lse)
    if blind.any():
        weights = weights.masked_fill(blind[..., None], 0.0)
    return weights, lse


def _attend(q, k, v, scale, mask=None):
    """Reference attention for one request: fp32 softmax plus the 2-based lse.

    ``q`` is (n_q, n_head, d), ``k``/``v`` are (n_kv, n_head, d). ``mask`` is a
    (n_q, n_kv) bool of KEEP positions. Query rows are processed in chunks so the live
    logits stay bounded; the arithmetic per row is unchanged.
    """
    n_q, n_head, _ = q.shape
    kf, vf = k.float(), v.float()
    step = _q_chunk(n_q, n_head, k.shape[0])
    outs, lses = [], []
    for lo in range(0, n_q, step):
        hi = min(lo + step, n_q)
        logits = torch.einsum("qhd,khd->qhk", q[lo:hi].float(), kf) * scale
        if mask is not None:
            logits = logits.masked_fill(~mask[lo:hi, None, :], NEG_INF)
        weights, lse = _finish(logits)
        lses.append(lse)
        outs.append(torch.einsum("qhk,khd->qhd", weights, vf))
        del logits, weights
    return torch.cat(outs), torch.cat(lses)


def _mla_attend_rows(qn, qp, kc, kp, scale, keep=None):
    """MLA over a block of query rows: (n_q, n_head, ckv) out, (n_q, n_head) lse."""
    n_q, n_head, _ = qn.shape
    kcf, kpf = kc.float(), kp.float()
    step = _q_chunk(n_q, n_head, kc.shape[0])
    outs, lses = [], []
    for lo in range(0, n_q, step):
        hi = min(lo + step, n_q)
        logits = (
            torch.einsum("qhd,kd->qhk", qn[lo:hi].float(), kcf)
            + torch.einsum("qhd,kd->qhk", qp[lo:hi].float(), kpf)
        ) * scale
        if keep is not None:
            logits = logits.masked_fill(~keep[lo:hi, None, :], NEG_INF)
        weights, lse = _finish(logits)
        lses.append(lse)
        outs.append(torch.einsum("qhk,kd->qhd", weights, kcf))
        del logits, weights
    return torch.cat(outs), torch.cat(lses)


def _causal_keep(q_len, kv_len, device="cuda"):
    """End-aligned causal mask: keep ``kv_pos <= kv_len - q_len + q_pos``."""
    q_pos = torch.arange(q_len, device=device)[:, None]
    kv_pos = torch.arange(kv_len, device=device)[None, :]
    return kv_pos <= (kv_len - q_len) + q_pos


def _blank(rows, heads, dim, device):
    """The definitions' own preallocation: zero output, -inf lse."""
    out = torch.zeros(rows, heads, dim, device=device, dtype=BF16)
    lse = torch.full((rows, heads), NEG_INF, device=device, dtype=torch.float32)
    return out, lse


def _paged_tokens(kv_indices, base, kv_len, page_size, device):
    """The cache rows one request's pages cover, in order, clipped to ``kv_len``."""
    n_pages = -(-kv_len // page_size)
    pages = kv_indices[base : base + n_pages].long()
    if page_size == 1:
        return pages, torch.zeros_like(pages)
    slot = torch.arange(page_size, device=device)
    page_of = pages[:, None].expand(-1, page_size).reshape(-1)[:kv_len]
    slot_of = slot[None, :].expand(pages.numel(), -1).reshape(-1)[:kv_len]
    return page_of, slot_of


# --------------------------------------------------------------- GQA paged decode
def gqa_paged_decode_inputs(dims, kv_lens, page_size=1, scale=None):
    batch, num_qo_heads, num_kv_heads, head_dim, num_pages, n_idx = dims[:6]
    gen = _generator()
    pages_per_req = [-(-n // page_size) for n in kv_lens]
    kv_indices = _pad_indices(_page_table(num_pages, sum(pages_per_req)), n_idx)
    kv_indptr = _i32(_indptr(pages_per_req))
    cache = lambda: _randn(  # noqa: E731
        num_pages, page_size, num_kv_heads, head_dim, gen=gen
    )
    args = [
        _randn(batch, num_qo_heads, head_dim, gen=gen),
        cache(),
        cache(),
        kv_indptr,
        kv_indices,
    ]
    if page_size != 1:
        # The last page of a request is partially filled, and only this tensor says by
        # how much: the page count alone rounds the length up.
        args.append(_i32([n - (p - 1) * page_size for n, p in zip(kv_lens, pages_per_req)]))
    args.append(float(head_dim) ** -0.5 if scale is None else float(scale))
    return args


def gqa_paged_decode_reference(dims, kv_lens, page_size=1):
    batch, num_qo_heads, num_kv_heads, head_dim, num_pages, _n_idx = dims[:6]
    group = num_qo_heads // num_kv_heads

    def reference(q, k_cache, v_cache, kv_indptr, kv_indices, *rest):
        sm_scale = rest[-1]
        out, lse = _blank(batch, num_qo_heads, head_dim, q.device)
        for b, kv_len in enumerate(kv_lens):
            if kv_len <= 0:
                continue
            page_of, slot_of = _paged_tokens(
                kv_indices, int(kv_indptr[b]), kv_len, page_size, q.device
            )
            kk = k_cache[page_of, slot_of].repeat_interleave(group, dim=1)
            vv = v_cache[page_of, slot_of].repeat_interleave(group, dim=1)
            o, l = _attend(q[b : b + 1], kk, vv, sm_scale)
            out[b] = o[0].to(BF16)
            lse[b] = l[0]
        return [out, lse]

    return reference


# ------------------------------------------------------- GQA paged prefill causal
def gqa_paged_prefill_inputs(dims, q_lens, kv_lens, scale=None):
    num_qo_heads, num_kv_heads, head_dim, num_pages, n_idx, total_q = dims[:6]
    gen = _generator()
    kv_indptr = _indptr(kv_lens)
    kv_indices = _pad_indices(_page_table(num_pages, kv_indptr[-1]), n_idx)
    cache = lambda: _randn(num_pages, 1, num_kv_heads, head_dim, gen=gen)  # noqa: E731
    return [
        _randn(total_q, num_qo_heads, head_dim, gen=gen),
        cache(),
        cache(),
        _i32(_indptr(q_lens)),
        _i32(kv_indptr),
        kv_indices,
        float(head_dim) ** -0.5 if scale is None else float(scale),
    ]


def gqa_paged_prefill_reference(dims, q_lens, kv_lens):
    num_qo_heads, num_kv_heads, head_dim, _num_pages, _n_idx, total_q = dims[:6]
    group = num_qo_heads // num_kv_heads

    def reference(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        out, lse = _blank(total_q, num_qo_heads, head_dim, q.device)
        for r, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
            if q_len <= 0 or kv_len <= 0:
                continue
            qs, qe = int(qo_indptr[r]), int(qo_indptr[r + 1])
            pages = kv_indices[int(kv_indptr[r]) : int(kv_indptr[r]) + kv_len].long()
            kk = k_cache[pages, 0].repeat_interleave(group, dim=1)
            vv = v_cache[pages, 0].repeat_interleave(group, dim=1)
            o, l = _attend(
                q[qs:qe], kk, vv, sm_scale, _causal_keep(q_len, kv_len, q.device)
            )
            out[qs:qe] = o.to(BF16)
            lse[qs:qe] = l
        return [out, lse]

    return reference


# ------------------------------------------------------ GQA ragged prefill causal
def gqa_ragged_prefill_inputs(dims, q_lens, kv_lens, scale=None):
    num_qo_heads, num_kv_heads, head_dim, total_q, total_kv = dims[:5]
    gen = _generator()
    return [
        _randn(total_q, num_qo_heads, head_dim, gen=gen),
        _randn(total_kv, num_kv_heads, head_dim, gen=gen),
        _randn(total_kv, num_kv_heads, head_dim, gen=gen),
        _i32(_indptr(q_lens)),
        _i32(_indptr(kv_lens)),
        float(head_dim) ** -0.5 if scale is None else float(scale),
    ]


def gqa_ragged_prefill_reference(dims, q_lens, kv_lens):
    num_qo_heads, num_kv_heads, head_dim, total_q, _total_kv = dims[:5]
    group = num_qo_heads // num_kv_heads

    def reference(q, k, v, qo_indptr, kv_indptr, sm_scale):
        out, lse = _blank(total_q, num_qo_heads, head_dim, q.device)
        for r, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
            if q_len <= 0 or kv_len <= 0:
                continue
            qs, qe = int(qo_indptr[r]), int(qo_indptr[r + 1])
            ks, ke = int(kv_indptr[r]), int(kv_indptr[r + 1])
            kk = k[ks:ke].repeat_interleave(group, dim=1)
            vv = v[ks:ke].repeat_interleave(group, dim=1)
            o, l = _attend(
                q[qs:qe], kk, vv, sm_scale, _causal_keep(q_len, kv_len, q.device)
            )
            out[qs:qe] = o.to(BF16)
            lse[qs:qe] = l
        return [out, lse]

    return reference


# --------------------------------------------------------------- MLA paged decode
def mla_paged_decode_inputs(dims, kv_lens, scale=None):
    batch, num_qo_heads, ckv, kpe, num_pages, n_idx = dims[:6]
    gen = _generator()
    kv_indices = _pad_indices(_page_table(num_pages, sum(kv_lens)), n_idx)
    return [
        _randn(batch, num_qo_heads, ckv, gen=gen),
        _randn(batch, num_qo_heads, kpe, gen=gen),
        _randn(num_pages, 1, ckv, gen=gen),
        _randn(num_pages, 1, kpe, gen=gen),
        _i32(_indptr(kv_lens)),
        kv_indices,
        float(ckv + kpe) ** -0.5 if scale is None else float(scale),
    ]


def mla_paged_decode_reference(dims, kv_lens):
    batch, num_qo_heads, ckv, _kpe, _num_pages, _n_idx = dims[:6]

    def reference(
        q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale
    ):
        out, lse = _blank(batch, num_qo_heads, ckv, q_nope.device)
        for b, kv_len in enumerate(kv_lens):
            if kv_len <= 0:
                continue
            base = int(kv_indptr[b])
            pages = kv_indices[base : base + kv_len].long()
            o, l = _mla_attend_rows(
                q_nope[b : b + 1],
                q_pe[b : b + 1],
                ckv_cache[pages, 0],
                kpe_cache[pages, 0],
                sm_scale,
            )
            out[b] = o[0].to(BF16)
            lse[b] = l[0]
        return [out, lse]

    return reference


# ------------------------------------------------------ MLA paged prefill causal
def mla_paged_prefill_inputs(dims, q_lens, kv_lens, scale=None):
    num_qo_heads, ckv, kpe, num_pages, n_idx, total_q = dims[:6]
    gen = _generator()
    kv_indptr = _indptr(kv_lens)
    kv_indices = _pad_indices(_page_table(num_pages, kv_indptr[-1]), n_idx)
    return [
        _randn(total_q, num_qo_heads, ckv, gen=gen),
        _randn(total_q, num_qo_heads, kpe, gen=gen),
        _randn(num_pages, 1, ckv, gen=gen),
        _randn(num_pages, 1, kpe, gen=gen),
        _i32(_indptr(q_lens)),
        _i32(kv_indptr),
        kv_indices,
        float(ckv + kpe) ** -0.5 if scale is None else float(scale),
    ]


def mla_paged_prefill_reference(dims, q_lens, kv_lens):
    num_qo_heads, ckv, _kpe, _num_pages, _n_idx, total_q = dims[:6]

    def reference(
        q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale
    ):
        out, lse = _blank(total_q, num_qo_heads, ckv, q_nope.device)
        for r, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
            if q_len <= 0 or kv_len <= 0:
                continue
            qs, qe = int(qo_indptr[r]), int(qo_indptr[r + 1])
            pages = kv_indices[int(kv_indptr[r]) : int(kv_indptr[r]) + kv_len].long()
            o, l = _mla_attend_rows(
                q_nope[qs:qe],
                q_pe[qs:qe],
                ckv_cache[pages, 0],
                kpe_cache[pages, 0],
                sm_scale,
                _causal_keep(q_len, kv_len, q_nope.device),
            )
            out[qs:qe] = o.to(BF16)
            lse[qs:qe] = l
        return [out, lse]

    return reference
