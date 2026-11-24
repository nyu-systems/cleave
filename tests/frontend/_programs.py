"""The demos' graphs written against ``cleave.lang`` (door 1: @jit, door 2: Program builder)."""

import cleave.lang as cl
from cleave.frontend import Program, jit

RMS_NORM_EPS = 1e-6


@jit
def matmul(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    return x @ w


@jit
def rmsnorm_matmul(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    return cl.rms_norm(x, dim=-1) @ w


@jit
def swiglu(x: cl.Tensor["M", "K"], w0: cl.Tensor["K", "N"], w1: cl.Tensor["K", "N"]):
    gate, up = x @ w0, x @ w1
    return cl.silu(gate) * up


@jit
def lora(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"], a: cl.Tensor["K", "R"], b: cl.Tensor["R", "N"]):
    return x @ w + (x @ a) @ b


@jit
def attention(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"], v: cl.Tensor["HKV", "SKV", "D"]):
    """MHA (HQ == HKV) and GQA (HQ = g * HKV) attention, as demo/attention.py writes it:
    ``cl.softmax`` is the exact softmax (the requested exp, its sum, the division)."""
    return cl.matmul(cl.softmax(cl.matmul(q, k.mT) / q.shape[-1] ** 0.5, dim=-1), v)


@jit
def matmul_rowsum(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    return (x @ w).sum(-1, keepdim=True)


@jit
def matmul_matmul(q: cl.Tensor["BH", "SQ", "D"], k: cl.Tensor["BH", "SKV", "D"], v: cl.Tensor["BH", "SKV", "D"]):
    return (q @ k.mT) @ v


@jit
def qknorm(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"], v: cl.Tensor["HKV", "SKV", "D"]):
    """Primitive order interleaved exactly like demo/qknorm.py."""
    d = q.shape[-1]
    q_sq, k_sq = q * q, k * k
    q_mean, k_mean = q_sq.sum(-1, keepdim=True) / float(d), k_sq.sum(-1, keepdim=True) / float(d)
    qn = q * cl.rsqrt(q_mean + RMS_NORM_EPS)
    kn = k * cl.rsqrt(k_mean + RMS_NORM_EPS)
    return attention.fn(qn, kn, v)


@jit
def qknorm_composite(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"], v: cl.Tensor["HKV", "SKV", "D"]):
    return attention(cl.rms_norm(q, eps=RMS_NORM_EPS), cl.rms_norm(k, eps=RMS_NORM_EPS), v)


def lora_builder(M=16, N=4096, K=4096, R=16) -> Program:
    p = Program("lora")
    M, K, N, R = p.dims(M=M, K=K, N=N, R=R)
    x, w = p.input("x", (M, K)), p.input("w", (K, N))
    a, b = p.input("a", (K, R)), p.input("b", (R, N))
    p.output(x @ w + (x @ a) @ b, name="y")
    return p


def rmsnorm_matmul_builder(M, N, K) -> Program:
    p = Program("rmsnorm_matmul")
    p.dims(M=M, K=K, N=N)
    x, w = p.input("x", ("M", "K")), p.input("w", ("K", "N"))
    p.output(cl.rms_norm(x) @ w)
    return p



def _row_lse(x, w):
    """``t`` for ``lse_attention``: the row log-sum-exp of ``x @ w``, along every row."""
    scores = x @ w
    top = cl.max(scores, -1, keepdim=True)
    return cl.log(cl.sum(cl.exp(scores - top), -1, keepdim=True)) + top + scores * 0.0


def lse_attention_builder(M=64, K=64, N=128, D=32, declared=True) -> Program:
    """``exp(x @ w - t) @ v`` with an fp32 ``t`` that must be the row log-sum-exp of ``x @ w``.

    With ``t`` declared (``sample=``) every probability is at most 1; drawn as noise instead,
    ``exp(x @ w - t)`` overflows fp16 and the second matmul turns that into NaN. ``t`` is
    full-shape ``[M, N]`` (an ``[M, 1]`` input does not lower yet); every input has its own
    shape (``bind_inputs`` matches same-shaped inputs by position).
    """
    p = Program("lse_attention")
    M, K, N, D = p.dims(M=M, K=K, N=N, D=D)
    x, w = p.input("x", (M, K)), p.input("w", (K, N))
    t = p.input("t", (M, N), cl.f32, **({"sample": _row_lse} if declared else {}))
    v = p.input("v", (N, D))
    p.output(cl.exp(x @ w - t) @ v, name="y")
    return p
