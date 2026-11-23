"""Pins computed before declared inputs existed: undeclared programs must not move.

Fingerprints, op counts and the ``random_inputs`` stream (sha256 of the raw bytes) of a few
existing programs, recorded at 9af4dd3 (before ``Program.input(equals=/sample=)``).
"""

import hashlib

import pytest
import torch

import cleave.lang as cl
from cleave.frontend import Program, trace

from . import _programs as P


def _exact_softmax_f32w():
    p = Program("exact")
    H, S, D = p.dims(H=2, S=64, D=32)
    x, w = p.input("x", (H, S, D)), p.input("w", (H, D, D), cl.f32)
    p.output(cl.softmax(x @ w, -1))
    return p


PROGRAMS = {
    "matmul": lambda: trace(P.matmul, dict(M=16, K=64, N=32)),
    "attention": lambda: trace(P.attention, dict(HQ=4, HKV=4, SQ=1, SKV=128, D=64)),
    "qknorm": lambda: trace(P.qknorm, dict(HQ=4, HKV=2, SQ=8, SKV=64, D=64)),
    "lora_builder": lambda: P.lora_builder(16, 256, 128, 16),
    "exact_softmax_f32w": _exact_softmax_f32w,
}

PINS = {  # name: (op_count, fingerprint, sha256[:16] of random_inputs(seed=3, device="cpu"))
    "matmul": (3, "8c9286f7441314f43b3009f91e43859a133b2bc7025b2481c17fce59f74aa25e", "517a979467082951"),
    "attention": (10, "a22609d51e062dc1c33a7f70a9311f9b0d27e025aa897e89d05da504dfaf3892", "05d633b95cb6af29"),
    "qknorm": (24, "6316a68c6c61a18f33d9ed6aa5dd272842d1a50fd5c89b9309db4c85ebda9d09", "6524ebb6c2bf49a8"),
    "lora_builder": (8, "0ab6318f8c248fab357cf7ba6fd18bb05146e08ea3ceb73b69bfac526b19ad1b", "339bc1ab9d060d50"),
    "exact_softmax_f32w": (8, "34a9d405fe3dc5b0db5946d661eb7d2565a865d644271f31ad00947be36a7d22",
                           "49a3cb5a7d1d627b"),
}


@pytest.mark.parametrize("name", sorted(PINS))
def test_undeclared_program_pins(name):
    p = PROGRAMS[name]()
    xs = p.random_inputs(seed=3, device="cpu")
    stream = hashlib.sha256(b"".join(x.contiguous().view(torch.uint8).numpy().tobytes() for x in xs))
    assert (p.op_count, p.fingerprint(), stream.hexdigest()[:16]) == PINS[name]
