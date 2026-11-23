"""MQA decode, BHSD (32 query heads over 1 kv head, S_Q=S_KV=4096, D=64, fp16); asks for a split of S_KV by 4,8,16.

The softmax is exact, and its split over S_KV is built from the loop state of the flash graph
(see demo/mha_decode.py): only K and V carry the keys, so each partial reads all query heads
next to its split of the kv heads' keys. The tolerance is declared by boundary role,
``"shift"`` for the float32 max a partial returns and ``""`` for every other output.
"""
import cleave
import cleave.lang as cl
from attention import attention


@cleave.jit
def mqa_decode(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"], v: cl.Tensor["HKV", "SKV", "D"]):
    return attention(q, k, v)


if __name__ == "__main__":
    cleave.examples.main(
        mqa_decode,
        shapes=dict(HQ=32, HKV=1, SQ=4096, SKV=4096, D=64),
        dtype=cl.f16,
        split=cl.Split("SKV", (4, 8, 16)),
        tolerance={"": cl.Tolerance(rtol=5e-3), "shift": cl.Tolerance(rtol=5e-6)},
    )
