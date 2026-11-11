"""MLA paged decode (DeepSeek latent attention, FlashInfer BatchMLAPagedAttention), bf16.

The compressed cache is both key and value; a RoPE cache adds a second score term.

    python -m demo.dynamic.mla_paged_decode --topk 800
"""

import cleave
import cleave.lang as cl

batch, num_pages, n_idx = cl.dims("batch num_pages n_idx", dynamic=True)
kv_len = cl.Dim("kv_len", dynamic=True, hint=4096)


def mla_paged_decode(H=16, CKV=512, KPE=64):
    @cleave.jit(name=f"mla_paged_decode_h{H}_ckv{CKV}_kpe{KPE}_ps1", dtype=cl.bf16)
    def op(
        qn: cl.Tensor[batch, H, CKV],
        qp: cl.Tensor[batch, H, KPE],
        ckv: cl.Tensor[num_pages, 1, CKV],
        kpe: cl.Tensor[num_pages, 1, KPE],
        kv_indptr: cl.Tensor[batch + 1, cl.i32],
        kvi: cl.Tensor[n_idx, cl.i32],
        sm_scale: cl.Scalar[cl.f32],
    ):
        pages = cl.PageTable(kv_indptr, kvi, length=kv_len)
        c, r = cl.paged_gather(ckv, pages), cl.paged_gather(kpe, pages)  # (batch, kv_len, width)
        p, lse = cl.softmax((qn @ c.mT + qp @ r.mT) * sm_scale, dim=-1, return_lse=True)
        return p @ c, lse

    return op


PRESETS = {"h16": dict()}


def build(preset="h16"):
    return mla_paged_decode(**PRESETS[preset]), None


def main(argv=None, preset=None):
    from demo.dynamic._frontend import main as run_main

    return run_main(__doc__, PRESETS, build, argv, preset or "h16")


if __name__ == "__main__":
    main()
