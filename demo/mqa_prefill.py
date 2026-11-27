"""MQA prefill, BHSD (32 query heads over 1 kv head, S_Q=S_KV=4096, D=64, fp16), unsplit.

The softmax is exact (see demo/mha_decode.py). ``--split SKV`` builds the split from the
loop state (see there too). The tolerance is declared by boundary role for that split:
``"shift"`` for the float32 max a split partial returns, ``""`` every other output.
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
        max_ops=11,  # the auto rule gives 13
        tolerance={"": cl.Tolerance(rtol=5e-3), "shift": cl.Tolerance(rtol=5e-6)},
    )
