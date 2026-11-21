"""GQA decode, BHSD (32 query heads over 8 kv heads, S_Q=S_KV=4096, D=64, fp16), split S_KV by 4,8,16.

The softmax is exact (see demo/mha_decode.py), so a split partial returns the max of its own
split next to its sums: the tolerance is declared by boundary role, ``"shift"`` for that
float32 max and ``""`` for every other output.
"""
import cleave
import cleave.lang as cl
from attention import attention


@cleave.jit
def gqa_decode(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"], v: cl.Tensor["HKV", "SKV", "D"]):
    return attention(q, k, v)


if __name__ == "__main__":
    cleave.examples.main(
        gqa_decode,
        shapes=dict(HQ=32, HKV=8, SQ=4096, SKV=4096, D=64),
        dtype=cl.f16,
        split=cl.Split("SKV", (4, 8, 16)),
        tolerance={"": cl.Tolerance(rtol=5e-3), "shift": cl.Tolerance(rtol=5e-6)},
    )
