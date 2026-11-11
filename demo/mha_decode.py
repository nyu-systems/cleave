"""MHA decode, BHSD (B*H=32, S_Q=64, S_KV=4096, D=64, fp16), split S_KV by 16,32.

    python demo/mha_decode.py
    python demo/mha_decode.py --split none          # solo kernels only
    python demo/mha_decode.py --preset fast --topk 20
"""
import cleave
import cleave.lang as cl
from attention import attention


@cleave.jit
def mha_decode(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], v: cl.Tensor["H", "SKV", "D"]):
    return attention(q, k, v)


if __name__ == "__main__":
    cleave.examples.main(
        mha_decode,
        shapes=dict(H=32, SQ=64, SKV=4096, D=64),
        dtype=cl.f16,
        split=cl.Split("SKV", (16, 32)),
        max_ops=14,  # split or not (the auto rule gives 13 unsplit)
    )
