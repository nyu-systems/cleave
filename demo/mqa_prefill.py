"""MQA prefill, BHSD (32 query heads over 1 kv head, S_Q=S_KV=4096, D=64, fp16), unsplit.

``--split SKV --max-ops 14`` splits k/v along S_KV.
"""
import cleave
import cleave.lang as cl
from attention import attention


@cleave.jit
def mqa_prefill(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"], v: cl.Tensor["HKV", "SKV", "D"]):
    return attention(q, k, v)


if __name__ == "__main__":
    cleave.examples.main(
        mqa_prefill,
        shapes=dict(HQ=32, HKV=1, SQ=4096, SKV=4096, D=64),
        dtype=cl.f16,
        max_ops=12,  # the auto rule gives 13
    )
