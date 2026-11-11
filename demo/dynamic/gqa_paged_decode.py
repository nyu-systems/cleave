"""GQA paged decode (FlashInfer BatchDecodeWithPagedKVCache signature), bf16.

kind / page size / head counts / emit_lse / input names are read off the function; the
dynamic KV length is searched at its hint and the paging pass makes it a runtime trip count.

    python -m demo.dynamic.gqa_paged_decode                  # h32 kv8, page size 1
    python -m demo.dynamic.gqa_paged_decode --preset kv4
    python -m demo.dynamic.gqa_paged_decode --preset ps64    # 64-token pages -> slab layer
    python -m demo.dynamic.gqa_paged_decode --split 4,8,16   # + split-KV kernel pairs
"""

import cleave
import cleave.lang as cl

batch, num_pages, n_idx = cl.dims("batch num_pages n_idx", dynamic=True)
kv_len = cl.Dim("kv_len", dynamic=True, hint=4096)  # per request, from kv_indptr


def gqa_paged_decode(H_Q=32, H_KV=8, D=128, PAGE=1):
    @cleave.jit(name=f"gqa_paged_decode_h{H_Q}_kv{H_KV}_d{D}_ps{PAGE}", dtype=cl.bf16)
    def op(
        q: cl.Tensor[batch, H_Q, D],
        k_cache: cl.Tensor[num_pages, PAGE, H_KV, D],
        v_cache: cl.Tensor[num_pages, PAGE, H_KV, D],
        kv_indptr: cl.Tensor[batch + 1, cl.i32],
        kv_indices: cl.Tensor[n_idx, cl.i32],
        kv_last_page_len: cl.Tensor[batch, cl.i32] | None,  # read iff PAGE > 1
        sm_scale: cl.Scalar[cl.f32],
    ):
        pages = cl.PageTable(kv_indptr, kv_indices, last_page_len=kv_last_page_len, length=kv_len)
        k, v = cl.paged_gather(k_cache, pages), cl.paged_gather(v_cache, pages)  # (batch, H_KV, kv_len, D)
        qg = q.view(batch, H_KV, H_Q // H_KV, D)
        p, lse = cl.softmax((qg @ k.mT) * sm_scale, dim=-1, return_lse=True)
        return (p @ v).view(batch, H_Q, D), lse.view(batch, H_Q)

    return op


PRESETS = {
    "kv8": dict(),
    # kv4 measured better with the kv head on the leading grid axis
    "kv4": dict(H_KV=4, hints=cl.ScheduleHints(grid_order=("head", "batch"))),
    # slab copies are predicated by tilelang's bounds check, so keep it on
    "ps64": dict(PAGE=64, hints=cl.ScheduleHints(grid_order=("head", "batch"), pass_configs={})),
}
# Validation/timing uses the registry's captured workloads; --synthetic uses these instead.
WORKLOADS = cl.workloads.paged_decode(batch=(1, 7, 64), kv_len=(1, 8192))
TOLERANCE = dict(rtol=1e-2, atol=1e-2)


def build(preset="kv8"):
    """``(fn, hints)`` for a preset."""
    kw = dict(PRESETS[preset])
    hints = kw.pop("hints", None)
    return gqa_paged_decode(**kw), hints


def main(argv=None, preset=None):
    from demo.dynamic._frontend import main as run_main

    return run_main(__doc__, PRESETS, build, argv, preset or "kv8",
                    tolerance=cl.Tolerance(**TOLERANCE), synthetic=WORKLOADS)


if __name__ == "__main__":
    main()
