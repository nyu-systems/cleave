"""QK-norm attention: RMSNorm(Q), RMSNorm(K), then attention (64 query heads over 8 kv heads,
S_Q=S_KV=4096, D=128, fp16). Forced schedule fusion by default: the unforced tile policy
prunes every candidate of the fused RMSNorm + attention kernel.

The softmax is exact (see demo/mha_decode.py). Its logits are computed here (from the
normalized q and k), not read from an input, so the adversarial input cases, which shape
inputs, cannot reach them: this demo's kernels are validated on normal inputs, and the
report says so.

The demo is unsplit and declares one tolerance, for the outputs of its solo kernels.
Splitting k/v along S_KV (``--split SKV``) is not offered: a split partial of an exact
softmax also returns its max, which needs a tolerance declared by boundary role (see
demo/mha_prefill.py). The flag is still parsed, as for every demo, and the run is the
unsplit one: the split of a softmax is not built yet (see demo/mha_decode.py).
"""
import cleave
import cleave.lang as cl
from attention import attention

EPS = 1e-6


@cleave.jit
def qknorm(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"], v: cl.Tensor["HKV", "SKV", "D"]):
    # q and k primitives interleaved (cl.rms_norm would emit q's chain, then k's).
    q_sq, k_sq = q * q, k * k
    q_ms, k_ms = q_sq.mean(-1, keepdim=True), k_sq.mean(-1, keepdim=True)
    qn = q * cl.rsqrt(q_ms + EPS)
    kn = k * cl.rsqrt(k_ms + EPS)
    return attention(qn, kn, v)


if __name__ == "__main__":
    cleave.examples.main(
        qknorm,
        shapes=dict(HQ=64, HKV=8, SQ=4096, SKV=4096, D=128),
        dtype=cl.f16,
        fusion="force",
        tolerance=cl.Tolerance(rtol=1e-2),
    )
