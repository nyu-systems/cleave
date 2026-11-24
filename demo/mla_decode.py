"""MLA decode (B=1, H=128, S_KV=1024, latent D=512, rope PE=64, fp16): the latent kv is both
K and V, and the rope parts add a second Q K^T term. Scale is 1/sqrt(128) (the nope head dim).
"""
import cleave
import cleave.lang as cl


@cleave.jit
def mla_decode(q: cl.Tensor["B", "H", "D"], kv: cl.Tensor["B", "SKV", "D"],
               q_pe: cl.Tensor["B", "H", "PE"], k_pe: cl.Tensor["B", "SKV", "PE"]):
    s = (q @ kv.mT + q_pe @ k_pe.mT) / 128 ** 0.5
    return cl.softmax(s, dim=-1) @ kv


if __name__ == "__main__":
    cleave.examples.main(
        mla_decode,
        shapes=dict(B=1, H=128, SKV=1024, D=512, PE=64),
        dtype=cl.f16,
        max_ops=13,  # the engine default; the auto rule gives 14
        tolerance=cl.Tolerance(rtol=5e-3),
    )
