"""GQA causal prefill over a paged KV cache (FlashInfer BatchPrefillWithPagedKVCache), bf16.

Queries are ragged (``qo_indptr``), keys/values paged (``kv_indptr``/``kv_indices``); the
causal mask is bottom-right aligned. Kind, head counts and input names come off the function.

    python -m demo.dynamic.gqa_paged_prefill                 # h32 kv8, page size 1
    python -m demo.dynamic.gqa_paged_prefill --preset kv4
"""

import cleave
import cleave.lang as cl

batch, total_q, num_pages, n_idx = cl.dims("batch total_q num_pages n_idx", dynamic=True)
q_len = cl.Dim("q_len", dynamic=True)  # per request, from qo_indptr
kv_len = cl.Dim("kv_len", dynamic=True, hint=4096)  # per request, from kv_indptr


def gqa_paged_prefill(H_Q=32, H_KV=8, D=128):
    G = H_Q // H_KV

    @cleave.jit(name=f"gqa_paged_prefill_causal_h{H_Q}_kv{H_KV}_d{D}_ps1", dtype=cl.bf16)
    def op(
        q: cl.Tensor[total_q, H_Q, D],
        k_cache: cl.Tensor[num_pages, 1, H_KV, D],
        v_cache: cl.Tensor[num_pages, 1, H_KV, D],
        qo_indptr: cl.Tensor[batch + 1, cl.i32],
        kv_indptr: cl.Tensor[batch + 1, cl.i32],
        kv_indices: cl.Tensor[n_idx, cl.i32],
        sm_scale: cl.Scalar[cl.f32],
    ):
        rows = cl.PageTable(qo_indptr, length=q_len)  # no indices: ragged query rows
        pages = cl.PageTable(kv_indptr, kv_indices, length=kv_len)
        qr = cl.paged_gather(q, rows).view(batch, H_KV, G, q_len, D)
        k = cl.paged_gather(k_cache, pages).view(batch, H_KV, 1, kv_len, D)
        v = cl.paged_gather(v_cache, pages).view(batch, H_KV, 1, kv_len, D)
        p, lse = cl.softmax((qr @ k.mT) * sm_scale, dim=-1, causal=True, return_lse=True)
        return (p @ v).view(total_q, H_Q, D), lse.view(total_q, H_Q)

    return op


PRESETS = {"kv8": dict(), "kv4": dict(H_KV=4)}


def build(preset="kv8"):
    return gqa_paged_prefill(**PRESETS[preset]), None


def main(argv=None, preset=None):
    from demo.dynamic._frontend import main as run_main

    return run_main(__doc__, PRESETS, build, argv, preset or "kv8")


if __name__ == "__main__":
    main()
