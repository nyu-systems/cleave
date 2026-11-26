"""MLA decode (B=1, H=128, S_KV=1024, latent D=512, rope PE=64, fp16): the latent kv is both
K and V, and the rope parts add a second Q K^T term. Scale is 1/sqrt(128) (the nope head dim).

The softmax is exact (see demo/mha_decode.py): the loop over S_KV carries the max and
repairs its row sum and its value accumulator by the found factor ``exp(m - m')`` at every
iteration. The value accumulator of a block is 64 x 512 (256 elements a thread at 128
threads), so that repair is a pass over it per iteration; the best kernel takes about
280 us on an H100. (A form that repaired only when the max moved took 190 us, but no such
rule is derived: each accumulator's update is checked against that accumulator's own
values on every prefix of the keys.) The demo asks for no split.
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
        max_ops=14,  # the budget its graphs need with the max in them; the auto rule gives 16
        tolerance=cl.Tolerance(rtol=5e-3),
    )
