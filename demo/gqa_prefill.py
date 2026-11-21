"""GQA prefill, BHSD (32 query heads over 8 kv heads, S_Q=S_KV=4096, D=64, fp16), unsplit.

``--split SKV`` splits k/v along S_KV.

The softmax is exact (see demo/mha_decode.py). The tolerance is declared by boundary role
because a split partial returns the max of its own split next to its sums: ``"shift"`` is
that float32 max, ``""`` every other output (all of them, unsplit).
"""
import cleave
import cleave.lang as cl
from attention import attention


@cleave.jit
def gqa_prefill(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"], v: cl.Tensor["HKV", "SKV", "D"]):
    return attention(q, k, v)


if __name__ == "__main__":
    cleave.examples.main(
        gqa_prefill,
        shapes=dict(HQ=32, HKV=8, SQ=4096, SKV=4096, D=64),
        dtype=cl.f16,
        tolerance={"": cl.Tolerance(rtol=5e-3), "shift": cl.Tolerance(rtol=5e-6)},
    )
