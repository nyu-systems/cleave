"""Attention math shared by the *_decode / *_prefill demos (a helper, not a demo)."""
import cleave.lang as cl


def attention(q, k, v):
    """softmax(Q K^T / sqrt(d)) V; the demos' fixed -50/sqrt(d) shift stands in for the row max."""
    n = q.shape[-1] ** 0.5
    return cl.softmax((q @ k.mT) / n, dim=-1, shift=-50 / n) @ v


def exact_attention(q, k, v):
    """softmax(Q K^T / sqrt(d)) V with the exact exp(s - max(s)) / sum."""
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v
