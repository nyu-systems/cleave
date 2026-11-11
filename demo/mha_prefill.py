"""MHA prefill, BHSD (B*H=32, S_Q=S_KV=4096, D=64, fp16), unsplit.

``--split SKV --max-ops 14`` splits k/v along S_KV.
"""
import cleave
import cleave.lang as cl
from attention import attention


@cleave.jit
def mha_prefill(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], v: cl.Tensor["H", "SKV", "D"]):
    return attention(q, k, v)


if __name__ == "__main__":
    cleave.examples.main(
        mha_prefill,
        shapes=dict(H=32, SQ=4096, SKV=4096, D=64),
        dtype=cl.f16,
        max_ops=12,  # the auto rule gives 13
    )
