"""MHA decode, BHSD (B*H=32, S_Q=64, S_KV=4096, D=64, fp16), split S_KV by 16,32.

``cl.softmax`` is the exact softmax, exp(s - max(s)) / sum: the search sees ``exp / sum /
div`` and no max, and each schedule gets the shift it needs. A tile that holds a whole S_KV
row takes the row max in place; a row that is looped over carries a running max and rescales
what it has accumulated (the online softmax of FlashAttention). At S_KV=4096 every unsplit
plan is of the second kind; ``--SKV 256`` has both. Split over the keys (the split-KV of
FlashDecoding), each partial kernel returns the max of its own split next to its sums, and
the combine kernel weights the partials by it. The kernels are validated on adversarial
logits as well (x100, increasing / decreasing along the keys, a spike on the last key,
constant rows, a common offset of -1e4).

The tolerance is declared by boundary role: ``"shift"`` is the float32 max a partial
returns, ``""`` every other output.

    python demo/mha_decode.py
    python demo/mha_decode.py --split none          # solo kernels only
    python demo/mha_decode.py --preset fast --topk 16
    python demo/mha_decode.py --SKV 256
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
        tolerance={"": cl.Tolerance(rtol=5e-3), "shift": cl.Tolerance(rtol=5e-6)},
    )
