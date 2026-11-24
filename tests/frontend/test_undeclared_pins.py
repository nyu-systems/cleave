"""Pins computed before declared inputs existed: undeclared programs must not move.

Fingerprints, op counts and the ``random_inputs`` stream (sha256 of the raw bytes) of a few
existing programs, recorded at 9af4dd3 (before ``Program.input(equals=/sample=)``). The
softmax programs are re-pinned where ``cl.softmax`` records its exact softmax differently
(their input streams are as recorded).
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
    "attention": (9, "4f3776ad9bf86d366b1dbd6aa9f02c1f1d2c1f91dc803192c7d30871d8aa2c15", "05d633b95cb6af29"),
    "qknorm": (23, "5a7005934c038a5c89b0eb8d1c25e6e21471197bb5df37988fc6b565b576f903", "6524ebb6c2bf49a8"),
    "lora_builder": (8, "0ab6318f8c248fab357cf7ba6fd18bb05146e08ea3ceb73b69bfac526b19ad1b", "339bc1ab9d060d50"),
    "exact_softmax_f32w": (6, "0a2a9f13089bd077ce776c6cbeb329f3eccf307cc354bec781da59227e3a3e92",
                           "49a3cb5a7d1d627b"),
}


@pytest.mark.parametrize("name", sorted(PINS))
def test_undeclared_program_pins(name):
    p = PROGRAMS[name]()
    xs = p.random_inputs(seed=3, device="cpu")
    stream = hashlib.sha256(b"".join(x.contiguous().view(torch.uint8).numpy().tobytes() for x in xs))
    assert (p.op_count, p.fingerprint(), stream.hexdigest()[:16]) == PINS[name]
