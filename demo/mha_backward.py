"""MHA backward: dQ, dK, dV of softmax(Q K^T / sqrt(d)) V in one fused kernel (B*H=32,
S_Q=S_KV=4096, D=64, fp16; L fp32), unsplit.

The program is the textbook backward, with ``delta = sum_SKV(P * dP)`` (a row reduction over the
whole S_KV row that feeds back into dS). Two inputs come from the forward pass and say so:

- ``L``, the forward's log-sum-exp (``P = exp(S - L)`` is the softmax), is ``cl.Sample(fn)``: fn is
  only how validation draws it, the input stays free for the search.
- ``o``, the forward's output, is ``cl.Equals(fn)``: the program does not read it, but the search
  may substitute it for its definition. Since ``o = P @ v``, ``sum_SKV(P * (dO @ v^T))`` equals
  ``D = sum_D(dO * o)``, a reduction over the head dim only, and the search finds that graph.

With a declared input and several outputs the search runs once per output and merges the
per-output graphs by sharing (S, P, dP, D and dS are common to all three). In the fused kernel a
thread block owns one tile along S_Q or S_KV and loops over the other: one group of outputs
accumulates in registers, the other across thread blocks with atomic adds.

    python demo/mha_backward.py
    python demo/mha_backward.py --preset fast
"""
import cleave
import cleave.lang as cl


def log_sum_exp(q, k):
    s = q @ k.mT * q.shape[-1] ** -0.5
    m = cl.max(s, -1, keepdim=True)
    return m + cl.log(cl.sum(cl.exp(s - m), -1, keepdim=True))


def forward_output(q, k, v, L):
    return cl.exp(q @ k.mT * q.shape[-1] ** -0.5 - L) @ v


@cleave.jit
def mha_backward(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                 v: cl.Tensor["H", "SKV", "D"], dO: cl.Tensor["H", "SQ", "D"],
                 L: cl.Tensor["H", "SQ", 1, cl.f32, cl.Sample(log_sum_exp)],
                 o: cl.Tensor["H", "SQ", "D", cl.Equals(forward_output)]):
    c = q.shape[-1] ** -0.5
    P = cl.exp(q @ k.mT * c - L)
    dP = dO @ v.mT
    delta = cl.sum(P * dP, -1, keepdim=True)
    dS = P * (dP - delta)
    return {"dQ": (dS @ k) * c, "dK": (dS.mT @ q) * c, "dV": P.mT @ dO}


if __name__ == "__main__":
    cleave.examples.main(
        mha_backward,
        shapes=dict(H=32, SQ=4096, SKV=4096, D=64),
        dtype=cl.f16,
        split=False,
        # The auto rule gives 19 for dQ and dK (~46 min of search at 24 threads); 17 (~5.5 min)
        # already finds the shared graph with D = sum_D(dO * o).
        max_ops=17,
        tolerance=(cl.Tolerance(rtol=5e-3),) * 3,  # placeholder, measured below
    )
