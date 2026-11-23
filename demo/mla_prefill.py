"""MLA prefill, BHSD (B*H=128, S_Q=S_KV=4096, qk head dim 192, v head dim 128, fp16), unsplit.

The softmax is exact (see demo/mha_decode.py). The demo is unsplit and declares one
tolerance, for the outputs of its solo kernels. Splitting k/v along S_KV (``--split SKV``)
is not offered at this shape: a split partial returns 128 x splits x 4096 x 128 values
(2 GB at 16 splits) and is validated on seven input cases, which comes to about 190 GB of
references on disk; and its max would need a tolerance declared by boundary role (see
demo/mha_prefill.py). The flag is still parsed, as for every demo, and the run is the
unsplit one: the split of a softmax is not built yet (see demo/mha_decode.py).
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
        max_ops=11,  # the auto rule gives 13
        tolerance=cl.Tolerance(rtol=5e-3),
    )
