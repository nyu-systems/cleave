"""MHA decode with the exact softmax exp(s - max(s)) / sum (B*H=32, S_Q=64, S_KV=256, D=64, fp16).

The row max must be complete before exp, so a block holds the whole S_KV row: no split, and
S_KV <= 256 schedules today (longer rows need online softmax, not implemented).

    python demo/mha_decode_exact.py
    python demo/mha_decode_exact.py --preset fast --topk 16
"""
import cleave
import cleave.lang as cl
from attention import exact_attention


@cleave.jit
def mha_decode_exact(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                     v: cl.Tensor["H", "SKV", "D"]):
    return exact_attention(q, k, v)


if __name__ == "__main__":
    cleave.examples.main(
        mha_decode_exact,
        shapes=dict(H=32, SQ=64, SKV=256, D=64),
        dtype=cl.f16,
        split=False,
        max_ops=14,
    )
