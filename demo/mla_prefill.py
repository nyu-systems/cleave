"""MLA prefill, BHSD (B*H=128, S_Q=S_KV=4096, qk head dim 192, v head dim 128, fp16), unsplit.

``--split SKV --max-ops 14`` splits k/v along S_KV.
"""
import cleave
import cleave.lang as cl
from attention import attention


@cleave.jit
def mla_prefill(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], v: cl.Tensor["H", "SKV", "DV"]):
    return attention(q, k, v)


if __name__ == "__main__":
    cleave.examples.main(
        mla_prefill,
        shapes=dict(H=128, SQ=4096, SKV=4096, D=192, DV=128),
        dtype=cl.f16,
        max_ops=12,  # the auto rule gives 13
    )
