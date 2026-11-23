"""MHA decode, BHSD (B*H=32, S_Q=64, S_KV=4096, D=64, fp16); asks for a split of S_KV by 16,32.

``cl.softmax`` is the exact softmax, exp(s - max(s)) / sum, and records its max: an ordinary
reduction of the searched graph. A tile that holds a whole S_KV row takes the row max in
place; where a tiling loops over the row, the loop carries the max and repairs what it has
accumulated with an update rule that is searched for and proven on the program's own graph
(the online softmax of FlashAttention, found rather than written). At S_KV=4096 every plan
is of the second kind; ``--SKV 256`` has both. The kernels are validated on adversarial
logits as well (x100, increasing / decreasing along the keys, a spike on the last key,
constant rows, a common offset of -1e4).

Split over the keys (the split-KV of FlashDecoding): the search does not split a graph whose
max reduces the split dim, and the partial / combine pair built from the loop state is not
wired in yet, so this demo runs unsplit for now and its report refuses each split factor
with that reason. The tolerance is declared by boundary role for that split: ``"shift"`` for
the float32 max a partial returns, ``""`` for every other output (all of them, unsplit).

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
