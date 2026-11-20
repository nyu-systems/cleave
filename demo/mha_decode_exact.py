"""MHA decode with the exact softmax exp(s - max(s)) / sum (B*H=32, S_Q=64, S_KV=4096, D=64, fp16).

``cl.softmax`` without a ``shift``: the search sees ``exp / sum / div`` and no max, and each
schedule gets the shift it needs. A tile that holds a whole S_KV row takes the row max in
place; a row that is looped over carries a running max and rescales what it has accumulated
(the online softmax of FlashAttention). At S_KV=4096 every plan is of the second kind;
``--SKV 256`` has both. The kernels are validated on adversarial logits as well (x100,
increasing / decreasing along the keys, a spike on the last key, constant rows, a common
offset of -1e4), which a fixed shift cannot hold. Unsplit: the exact softmax across a split
is not available yet.

    python demo/mha_decode_exact.py
    python demo/mha_decode_exact.py --preset fast --topk 16
    python demo/mha_decode_exact.py --SKV 256
"""
import cleave
import cleave.lang as cl
from attention import exact_attention


@cleave.jit
def mha_decode_exact(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                     v: cl.Tensor["H", "SKV", "D"]):
    return exact_attention(q, k, v)


if __name__ == "__main__":
    cleave.examples.main(
        mha_decode_exact,
        shapes=dict(H=32, SQ=64, SKV=4096, D=64),
        dtype=cl.f16,
        split=False,
        tolerance=cl.Tolerance(rtol=5e-3),
    )
