"""Declared inputs (``equals=`` / ``sample=``), ``cl.log`` and the unread-input refusal (CPU)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import cleave.lang as cl
from cleave.core.errors import SpecError
from cleave.frontend import Program, jit, trace

H, SQ, SKV, D = 2, 64, 96, 32
C = D ** -0.5


# ------------------------------------------------------------------ the attention backward


def log_sum_exp(q, k):
    s = q @ k.mT * C
    m = cl.max(s, -1, keepdim=True)
    return m + cl.log(cl.sum(cl.exp(s - m), -1, keepdim=True))


def forward_output(q, k, v, L):
    return cl.exp(q @ k.mT * C - L) @ v


def _backward(q, k, v, dO, L):
    """The input graph: ``delta = sum_SKV(P * dP)`` (``o`` is not read)."""
    P = cl.exp(q @ k.mT * C - L)
    dP = dO @ v.mT
    delta = cl.sum(P * dP, -1, keepdim=True)
    dS = P * (dP - delta)
    return {"dQ": (dS @ k) * C, "dK": (dS.mT @ q) * C, "dV": P.mT @ dO}


@jit
def attention_bwd(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                  v: cl.Tensor["H", "SKV", "D"], dO: cl.Tensor["H", "SQ", "D"],
                  L: cl.Tensor["H", "SQ", 1, cl.f32, cl.Sample(log_sum_exp)],
                  o: cl.Tensor["H", "SQ", "D", cl.Equals(forward_output)]):
    return _backward(q, k, v, dO, L)


SHAPES = dict(H=H, SQ=SQ, SKV=SKV, D=D)


def builder(dtype=cl.f16, equals=forward_output, sample=log_sum_exp, declared=True) -> Program:
    p = Program("attention_bwd", dtype)
    h, sq, skv, d = p.dims(**SHAPES)
    q, k = p.input("q", (h, sq, d)), p.input("k", (h, skv, d))
    v, dO = p.input("v", (h, skv, d)), p.input("dO", (h, sq, d))
    L = p.input("L", (h, sq, 1), cl.f32, **({"sample": sample} if declared else {}))
    p.input("o", (h, sq, d), **({"equals": equals} if declared else {}))
    for n, t in _backward(q, k, v, dO, L).items():
        p.output(t, n)
    return p


def _torch_lse(q, k):
    return torch.logsumexp(q.float() @ k.float().mT * C, -1, keepdim=True)


# ------------------------------------------------------------------ program structure


def test_builder_and_jit_give_the_same_program():
    a, b = builder(), attention_bwd.trace(**SHAPES)
    assert a.fingerprint() == b.fingerprint()
    assert a.op_count == b.op_count
    assert [(r.kind, r.args, r.attrs) for r in a._records] == [(r.kind, r.args, r.attrs) for r in b._records]
    assert [s.dtype for s in b.inputs] == [cl.f16] * 4 + [cl.f32, cl.f16]
    assert b.inputs[4].sample is not None and b.inputs[4].equals is None
    assert b.inputs[5].equals is not None and b.inputs[5].sample is None
    assert b.inputs[4].over == (0, 1) and b.inputs[5].over == (0, 1, 2, 4)
    assert dict(a.ops[4].attrs) == {"name": "L", "sample": a.inputs[4].sample.fingerprint(), "over": (0, 1)}
    assert dict(a.ops[5].attrs) == {"name": "o", "equals": a.inputs[5].equals.fingerprint(),
                                    "over": (0, 1, 2, 4)}


def test_declarations_move_only_the_fingerprint():
    plain, declared = builder(declared=False), builder()
    assert plain.op_count == declared.op_count  # the sub-programs' ops are not the parent's
    assert len(plain.outputs) == len(declared.outputs) == 3
    assert plain.fingerprint() != declared.fingerprint()
    assert plain.to_graph_structure() == declared.to_graph_structure()
    assert plain.can_split(("SKV",)) == declared.can_split(("SKV",))
    from cleave.pipeline.config import SuperoptConfig

    assert SuperoptConfig().resolve_max_ops(plain) == SuperoptConfig().resolve_max_ops(declared)


def test_fingerprint_depends_on_the_declared_function():
    def other_lse(q, k):
        s = q @ k.mT * (2 * C)
        m = cl.max(s, -1, keepdim=True)
        return m + cl.log(cl.sum(cl.exp(s - m), -1, keepdim=True))

    def other_forward(q, k, v):  # reads other inputs: ``over`` differs too
        return cl.exp(q @ k.mT * C) @ v

    base = builder().fingerprint()
    assert builder().fingerprint() == base
    assert builder(sample=other_lse).fingerprint() != base
    assert builder(equals=other_forward).fingerprint() != base
    # equals= vs sample= of the same function
    p = Program("x")
    M, K = p.dims(M=8, K=16)
    x = p.input("x", (M, K))
    p.input("y", (M, K), equals=lambda x: x * 2.0)
    p.output(x * 3.0)
    s = Program("x")
    M, K = s.dims(M=8, K=16)
    x = s.input("x", (M, K))
    s.input("y", (M, K), sample=lambda x: x * 2.0)
    s.output(x * 3.0)
    assert p.fingerprint() != s.fingerprint()


# ------------------------------------------------------------------ values


def test_random_inputs_derive_lse_and_forward_output():
    p = builder()
    q, k, v, dO, L, o = p.random_inputs(seed=1, device="cpu")
    assert L.dtype == torch.float32 and o.dtype == torch.float16
    # L is a real log-sum-exp (fp32 compute, different reduction order than torch.logsumexp)
    torch.testing.assert_close(L, _torch_lse(q, k), rtol=1e-6, atol=1e-6)
    # o is the declared forward output: fp32 compute, one cast to fp16
    expect = (torch.exp(q.float() @ k.float().mT * C - L) @ v.float()).half()
    assert torch.equal(o, expect)
    # ... and that is the softmax attention forward (up to the one fp16 rounding)
    soft = torch.softmax(q.float() @ k.float().mT * C, -1) @ v.float()
    torch.testing.assert_close(o.float(), soft, rtol=1e-3, atol=1e-3)


def test_random_inputs_keep_the_stream_of_undeclared_inputs():
    declared = builder().random_inputs(seed=5, device="cpu")
    plain = builder(declared=False).random_inputs(seed=5, device="cpu")
    for a, b in zip(declared[:4], plain[:4]):
        assert torch.equal(a, b)


def test_derive_inputs_recomputes_in_order():
    p = builder()
    xs = p.random_inputs(seed=2, device="cpu")
    again = p.derive_inputs([*xs[:4], None, None])
    assert all(torch.equal(a, b) for a, b in zip(xs, again))
    # recomputation from new sources: o follows the new L
    q, k, v, dO, L, o = p.derive_inputs([*xs[:4], xs[4], torch.zeros_like(xs[5])])
    assert torch.equal(o, (torch.exp(q.float() @ k.float().mT * C - L) @ v.float()).half())
    with pytest.raises(TypeError):
        p.derive_inputs(xs[:5])


def test_reference_is_the_attention_gradient():
    p = builder(cl.f32)
    q, k, v, dO, L, o = p.random_inputs(seed=3, device="cpu")
    dQ, dK, dV = p.reference(q, k, v, dO, L, o)
    qq, kk, vv = (x.clone().requires_grad_() for x in (q, k, v))
    out = torch.softmax(qq @ kk.mT * C, -1) @ vv
    out.backward(dO)
    torch.testing.assert_close(o, out.detach(), rtol=1e-5, atol=1e-5)
    for got, ref in ((dQ, qq.grad), (dK, kk.grad), (dV, vv.grad)):
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


def test_log_reference_and_eager():
    x = torch.rand(4, 8) + 0.5
    assert torch.equal(cl.log(x), torch.log(x))
    p = Program("log")
    M, N = p.dims(M=4, N=8)
    t = p.input("x", (M, N), cl.f32)
    p.output(cl.log(t))
    p.output(t.log() * 2.0)
    a, b = p.reference(x)
    assert torch.equal(a, torch.log(x)) and torch.equal(b, torch.log(x) * 2.0)
    with pytest.raises(SpecError, match="reference-only"):
        p.to_tile_graph()
    with pytest.raises(SpecError, match="reference-only"):
        p.to_graph_structure()


# ------------------------------------------------------------------ refusals


def _xy():
    p = Program("x")
    M, K = p.dims(M=8, K=16)
    x = p.input("x", (M, K))
    return p, x, M, K


def test_equals_and_sample_together():
    p, x, M, K = _xy()
    with pytest.raises(SpecError, match="not both"):
        p.input("y", (M, K), equals=lambda x: x, sample=lambda x: x)


def test_equals_reading_equals():
    p, x, M, K = _xy()
    p.input("y", (M, K), equals=lambda x: x * 2.0)
    with pytest.raises(SpecError, match="another equals"):
        p.input("z", (M, K), equals=lambda x, y: x + y)
    p.input("s", (M, K), sample=lambda y: y + 1.0)  # a sample may read an equals input
    p.input("w", (M, K), sample=lambda s: s * 2.0)
    p.input("e", (M, K), equals=lambda w: w * 3.0)  # equals reading a sample input is fine


def test_equals_must_come_after_its_sources():
    p, x, M, K = _xy()
    with pytest.raises(SpecError, match="not an earlier input"):
        p.input("y", (M, K), equals=lambda x, later: x + later)
    with pytest.raises(SpecError, match="not an earlier input"):

        @jit
        def f(x: cl.Tensor["M", "K"], y: cl.Tensor["M", "K", cl.Equals(lambda x, z: x + z)],
              z: cl.Tensor["M", "K"]):
            return x + y + z

        f.trace(M=8, K=16)


def test_declaration_result_checks():
    p, x, M, K = _xy()
    with pytest.raises(SpecError, match="gives shape"):
        p.input("y", (M, 1), equals=lambda x: x * 2.0)
    with pytest.raises(SpecError, match="must return one cl.Tensor"):
        p.input("y", (M, K), sample=lambda x: x.shape)
    with pytest.raises(SpecError, match="must be a function"):
        p.input("y", (M, K), sample=3)
    assert [s.name for s in p.inputs] == ["x"]  # refused declarations add nothing


def test_annotation_refusals():
    with pytest.raises(TypeError, match="more than one"):
        cl.Tensor["M", cl.Equals(abs), cl.Equals(abs)]
    spec = cl.Tensor["M", cl.Equals(abs), cl.Sample(abs)]

    @jit
    def both(x: cl.Tensor["M"], y: spec):
        return x + y

    with pytest.raises(SpecError, match="not both"):
        both.trace(M=8)


def test_declared_inputs_refused_in_paged_programs():
    @jit
    def paged(x: cl.Tensor["M", "K"], n: cl.Scalar[cl.i32],
              y: cl.Tensor["M", "K", cl.Equals(lambda x: x * 2.0)]):
        return x * y

    with pytest.raises(SpecError, match="paged"):
        trace(paged)


# ------------------------------------------------------------------ to_tile_graph


def _unread_program():
    p = Program("unread")
    M, K, N = p.dims(M=16, K=32, N=16)
    x, w = p.input("x", (M, K)), p.input("w", (K, N))
    p.input("z", (M, N))
    p.output(x @ w)
    return p


def test_unread_input_is_a_spec_error():
    with pytest.raises(SpecError, match="'z' is not read by any output"):
        _unread_program().to_tile_graph()
    # read only by dead compute: still no output depends on it
    p = Program("dead")
    M, K, N = p.dims(M=16, K=32, N=16)
    x, w, z = p.input("x", (M, K)), p.input("w", (K, N)), p.input("z", (M, N))
    z * 2.0
    p.output(x @ w)
    with pytest.raises(SpecError, match="'z' is not read"):
        p.to_tile_graph()


def test_unread_input_does_not_abort_the_symbolizer():
    """Regression: without the check, the symbolizer aborts the process (dimension_group.cc
    "Dimension not found in symbol map"); run in a subprocess to see the exit status."""
    code = ("import cleave.superoptimizer as csopt\n"
            "from cleave.core.errors import SpecError\n"
            "from tests.frontend.test_declared_inputs import _unread_program\n"
            "try:\n"
            "    g = _unread_program().to_tile_graph()\n"
            "except SpecError as e:\n"
            "    print('SpecError', e)\n"
            "else:\n"
            "    csopt.create_symbolic_tile_graph(g)\n")
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": os.pathsep.join([str(root / "python"), str(root), os.environ.get("PYTHONPATH", "")])}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("SpecError")


def test_unread_equals_input_is_not_searchable_yet():
    with pytest.raises(SpecError, match="does not support declared inputs yet"):
        builder().to_tile_graph()  # the input graph does not read ``o``


def test_read_declared_inputs_lower_as_plain_inputs():
    p = Program("read")
    M, K = p.dims(M=16, K=32)
    x = p.input("x", (M, K))
    s = p.input("s", (M, 1), cl.f32, sample=lambda x: cl.log(cl.sum(cl.exp(x), -1, keepdim=True)))
    y = p.input("y", (M, K), equals=lambda x, s: x - s)
    p.output(y * 2.0 + x)
    plain = Program("read")
    M, K = plain.dims(M=16, K=32)
    x = plain.input("x", (M, K))
    plain.input("s", (M, 1), cl.f32)
    y = plain.input("y", (M, K))
    plain.output(y * 2.0 + x)
    with pytest.raises(SpecError, match="'s' is not read"):
        p.to_tile_graph()  # a sample= input is a free input: unread is refused
    with pytest.raises(SpecError, match="'s' is not read"):
        plain.to_tile_graph()
    q = Program("read2")
    M, K = q.dims(M=16, K=32)
    x = q.input("x", (M, K))
    y = q.input("y", (M, K), equals=lambda x: x * 0.5)
    q.output(y + x)
    g = q.to_tile_graph()
    assert g is not None
