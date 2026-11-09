"""MLA causal prefill over a paged latent cache (FlashInfer BatchMLAPagedAttention), bf16.

    python -m demo.dynamic.mla_paged_prefill
"""

import cleave
import cleave.lang as cl

batch, total_q, num_pages, n_idx = cl.dims("batch total_q num_pages n_idx", dynamic=True)
q_len = cl.Dim("q_len", dynamic=True)
kv_len = cl.Dim("kv_len", dynamic=True, hint=4096)


def mla_paged_prefill(H=16, CKV=512, KPE=64):
    @cleave.jit(name=f"mla_paged_prefill_causal_h{H}_ckv{CKV}_kpe{KPE}_ps1", dtype=cl.bf16)
    def op(
        q_nope: cl.Tensor[total_q, H, CKV],
        q_pe: cl.Tensor[total_q, H, KPE],
        ckv_cache: cl.Tensor[num_pages, 1, CKV],
        kpe_cache: cl.Tensor[num_pages, 1, KPE],
        qo_indptr: cl.Tensor[batch + 1, cl.i32],
        kv_indptr: cl.Tensor[batch + 1, cl.i32],
        kv_indices: cl.Tensor[n_idx, cl.i32],
        sm_scale: cl.Scalar[cl.f32],
    ):
        rows = cl.PageTable(qo_indptr, length=q_len)
        pages = cl.PageTable(kv_indptr, kv_indices, length=kv_len)
        qn, qp = cl.paged_gather(q_nope, rows), cl.paged_gather(q_pe, rows)  # (batch, H, q_len, w)
        c = cl.paged_gather(ckv_cache, pages).view(batch, 1, kv_len, CKV)
        r = cl.paged_gather(kpe_cache, pages).view(batch, 1, kv_len, KPE)
        p, lse = cl.softmax((qn @ c.mT + qp @ r.mT) * sm_scale, dim=-1, causal=True,
                            return_lse=True)
        return (p @ c).view(total_q, H, CKV), lse.view(total_q, H)

    return op


# one 16-row query block is too little work for 8 warps: no prefetch, a narrower window
HINTS = cl.ScheduleHints(index_window_tiles=4, prefetch=False, rasterization=1)
PRESETS = {"h16": dict()}


def build(preset="h16"):
    return mla_paged_prefill(**PRESETS[preset]), HINTS


def main(argv=None, preset=None):
    from demo.dynamic._frontend import main as run_main

    return run_main(__doc__, PRESETS, build, argv, preset or "h16")


if __name__ == "__main__":
    main()
