"""Attention math shared by the *_decode / *_prefill demos (a helper, not a demo)."""
import cleave.lang as cl


def attention(q, k, v):
    """softmax(Q K^T / sqrt(d)) V. ``cl.softmax`` is the exact exp(s - max(s)) / sum, at any S_KV."""
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v
