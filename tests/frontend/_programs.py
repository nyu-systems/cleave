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
    """MHA (HQ == HKV) and GQA (HQ = g * HKV) decode with the demos' fixed -50 max-shift."""
    n = q.shape[-1] ** 0.5
    return cl.matmul(cl.softmax(cl.matmul(q, k.mT) / n, dim=-1, shift=-50 / n), v)


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
