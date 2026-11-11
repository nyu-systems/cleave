"""GQA causal prefill over ragged (packed, unpaged) keys/values
(FlashInfer BatchPrefillWithRaggedKVCache), bf16.

Both the queries and the keys are packed per request (``qo_indptr``/``kv_indptr``, no page
table), so a KV tile is an affine slice.

    python -m demo.dynamic.gqa_ragged_prefill                # h32 kv8
    python -m demo.dynamic.gqa_ragged_prefill --preset kv4
"""

import cleave
import cleave.lang as cl

batch, total_q, total_kv = cl.dims("batch total_q total_kv", dynamic=True)
q_len = cl.Dim("q_len", dynamic=True)
kv_len = cl.Dim("kv_len", dynamic=True, hint=4096)


def gqa_ragged_prefill(H_Q=32, H_KV=8, D=128):
    G = H_Q // H_KV

    @cleave.jit(name=f"gqa_ragged_prefill_causal_h{H_Q}_kv{H_KV}_d{D}", dtype=cl.bf16)
    def op(
        q: cl.Tensor[total_q, H_Q, D],
        k: cl.Tensor[total_kv, H_KV, D],
        v: cl.Tensor[total_kv, H_KV, D],
        qo_indptr: cl.Tensor[batch + 1, cl.i32],
        kv_indptr: cl.Tensor[batch + 1, cl.i32],
        sm_scale: cl.Scalar[cl.f32],
    ):
        rows = cl.PageTable(qo_indptr, length=q_len)
        keys = cl.PageTable(kv_indptr, length=kv_len)  # ragged keys too
        qr = cl.paged_gather(q, rows).view(batch, H_KV, G, q_len, D)
        kr = cl.paged_gather(k, keys).view(batch, H_KV, 1, kv_len, D)
        vr = cl.paged_gather(v, keys).view(batch, H_KV, 1, kv_len, D)
        p, lse = cl.softmax((qr @ kr.mT) * sm_scale, dim=-1, causal=True, return_lse=True)
        return (p @ vr).view(total_q, H_Q, D), lse.view(total_q, H_Q)

    return op


# neighbouring blocks share keys, so the default L2 swizzle pays here
HINTS = cl.ScheduleHints(rasterization=None)
PRESETS = {"kv8": dict(), "kv4": dict(H_KV=4)}


def build(preset="kv8"):
    return gqa_ragged_prefill(**PRESETS[preset]), HINTS


def main(argv=None, preset=None):
    from demo.dynamic._frontend import main as run_main

    return run_main(__doc__, PRESETS, build, argv, preset or "kv8")


if __name__ == "__main__":
    main()
