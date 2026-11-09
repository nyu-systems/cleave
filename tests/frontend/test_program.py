import os
import sys
import types
from pathlib import Path

import pytest
import torch

import cleave.lang as cl
from cleave.core.errors import SpecError
from cleave.frontend import Program, jit, trace

from . import _programs as P

SMALL_ATTN = dict(HQ=4, HKV=4, SQ=8, SKV=32, D=16)
SMALL_GQA = dict(HQ=8, HKV=2, SQ=8, SKV=32, D=16)


def _close(a, b, tol=2e-2):
    torch.testing.assert_close(a.float(), b.float(), rtol=tol, atol=tol)


# ------------------------------------------------------------------ reference


def test_reference_matmul():
    prog = P.matmul.trace(M=16, N=8, K=32)
    x, w = prog.random_inputs(device="cpu")
    (y,) = prog.reference(x, w)
    assert y.dtype == torch.float16 and y.shape == (16, 8)
    _close(y, x.float() @ w.float())
    _close(prog.reference(w=w, x=x)[0], y)


def test_reference_rmsnorm_matmul():
    prog = P.rmsnorm_matmul.trace(M=16, N=8, K=32)
    x, w = prog.random_inputs(device="cpu")
    xf = x.float()
    want = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True)) @ w.float()
    _close(prog.reference(x, w)[0], want)


@pytest.mark.parametrize("shapes", [SMALL_ATTN, SMALL_GQA], ids=["mha", "gqa"])
def test_reference_attention(shapes):
    prog = P.attention.trace(**shapes)
    q, k, v = prog.random_inputs(device="cpu")
    g = shapes["HQ"] // shapes["HKV"]
    kf, vf = (t.float().repeat_interleave(g, dim=0) for t in (k, v))
    s = q.float() @ kf.mT / shapes["D"] ** 0.5
    _close(prog.reference(q, k, v)[0], torch.softmax(s, -1) @ vf)
    # the decorated function itself, run eagerly on torch tensors, is the same reference
    _close(P.attention(q.float(), k.float(), v.float()), prog.reference(q, k, v)[0])


def test_reference_qknorm_and_eager():
    prog = P.qknorm.trace(**SMALL_GQA)
    ins = prog.random_inputs(device="cpu")
    _close(P.qknorm_composite.trace(**SMALL_GQA).reference(*ins)[0], prog.reference(*ins)[0])
    _close(P.qknorm(*(t.float() for t in ins)), prog.reference(*ins)[0])


def test_reference_rejects_bad_shape():
    prog = P.matmul.trace(M=16, N=8, K=32)
    with pytest.raises(SpecError):
        prog.reference(torch.zeros(16, 31), torch.zeros(31, 8))
    with pytest.raises(TypeError):
        prog.reference(torch.zeros(16, 32))


CASES = [
    (P.matmul, dict(M=16, N=8, K=32)),
    (P.rmsnorm_matmul, dict(M=16, N=32, K=64)),
    (P.swiglu, dict(M=16, N=32, K=64)),
    (P.lora, dict(M=16, N=32, K=64, R=16)),
    (P.attention, SMALL_ATTN),
    (P.attention, SMALL_GQA),
    (P.matmul_rowsum, dict(M=16, N=32, K=16)),
    (P.qknorm, SMALL_GQA),
]


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0].__name__}-{i}" for i, c in enumerate(CASES)])
def test_reference_matches_lowered_graph_executor(case):
    """Program.reference == graph_executor on structure_to_cleave(un-optimized program)."""
    from cleave.lower import to_ir
    from cleave.runtime import reference as ge

    jf, shapes = case
    prog = jf.trace(**shapes)
    graph = to_ir(prog.to_graph_structure(), dtype="float16", out_dtype="float16")
    ins = prog.random_inputs(seed=1, device="cpu")
    by_guid = {f"PlaceHolder_input_{r.out + 1}": ins[i] for i, r in enumerate(r for r in prog.ops if r.kind == "input")}
    (got,) = ge.execute(graph, [by_guid[ph.name] for ph in ge.placeholder_order(graph)])
    (want,) = prog.reference(*ins)
    _close(got.reshape(want.shape), want)


# ------------------------------------------------------------------ fingerprint


def test_fingerprint_stable_and_structural():
    a = P.attention.trace(**SMALL_ATTN).fingerprint()
    assert a == P.attention.trace(**SMALL_ATTN).fingerprint() and len(a) == 64
    assert a != P.attention.trace(**{**SMALL_ATTN, "SKV": 64}).fingerprint()
    assert P.lora_builder().fingerprint() == P.lora.trace(M=16, N=4096, K=4096, R=16).fingerprint()
    # dtype matters
    assert trace(P.matmul, dict(M=16, N=8, K=32), cl.bf16).fingerprint() != P.matmul.trace(
        M=16, N=8, K=32).fingerprint()


def test_fingerprint_independent_of_process():
    import subprocess

    code = ("from tests.frontend import _programs as P;"
            "print(P.attention.trace(HQ=4, HKV=4, SQ=8, SKV=32, D=16).fingerprint())")
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "python"), str(root),
                                                         os.environ.get("PYTHONPATH", "")])}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=root, env=env)
    assert out.stdout.strip() == P.attention.trace(**SMALL_ATTN).fingerprint()


# ------------------------------------------------------------------ splits / dims


def test_split_flags_by_dim_name():
    prog = P.attention.trace(**SMALL_ATTN)
    assert prog.can_split(("SKV",)) == [(False, False, False), (False, True, False), (False, True, False)]
    assert prog.can_split(cl.Split("D")) == [(False, False, True)] * 3
    assert prog.can_split() == [(False,) * 3] * 3
    with pytest.raises(SpecError, match="not dims"):
        prog.can_split("K")


def test_program_metadata():
    prog = P.attention.trace(**SMALL_ATTN)
    assert [i.name for i in prog.inputs] == ["q", "k", "v"]
    assert prog.inputs[1].dims[1] == cl.Dim("SKV", 32)
    assert prog.sizes == dict(HQ=4, SQ=8, D=16, HKV=4, SKV=32)
    assert prog.op_count == 10 and len(prog.outputs) == 1  # 3 inputs + 7 compute ops
    assert "matmul" in repr(prog)


def test_dynamic_dim_traced_at_hint():
    kv = cl.Dim("SKV", dynamic=True, hint=64)

    @jit
    def f(q: cl.Tensor["SQ", "D"], k: cl.Tensor[kv, "D"]):
        return q @ k.mT

    prog = f.trace(SQ=8, D=16)
    assert prog.inputs[1].shape == (64, 16) and prog.dynamic_dims == (kv,)
    assert f.trace(SQ=8, D=16, SKV=128).inputs[1].shape == (128, 16)


def test_builder_dims_and_literals():
    p = Program("t", dtype=cl.f32)
    M, K = p.dims(M=4, K=8)
    x = p.input("x", (M, K))
    w = p.input("w", ("K", 2), dtype=cl.f16)
    p.output(x @ w)
    assert p.inputs[1].dtype == cl.f16 and p.inputs[1].shape == (8, 2)
    with pytest.raises(SpecError):
        p.dim("M", 5)
    with pytest.raises(SpecError):
        p.input("y", ("Q",))


def test_trace_errors():
    with pytest.raises(SpecError, match="unbound"):
        P.matmul.trace(M=4, K=8)
    with pytest.raises(SpecError, match="not dims"):
        P.matmul.trace(M=4, K=8, N=2, Z=1)


# ------------------------------------------------------------------ op semantics


def _p():
    p = Program()
    p.dims(M=4, N=8)
    return p, p.input("a", ("M", "N")), p.input("b", ("M", "N"))


def test_scalar_ops_lowering():
    p, a, b = _p()
    for t in (a - 2, 3 + a, 2 * a, a / 2, 1 / a, -a, 1 - a, a ** 2):
        p.output(t)
    kinds = [(r.kind, r.args) for r in p.ops[2:]]
    assert kinds == [("add", (0, -2.0)), ("add", (0, 3.0)), ("mul", (0, 2.0)), ("div", (0, 2.0)),
                     ("div", (1.0, 0)), ("mul", (0, -1.0)), ("mul", (0, -1.0)), ("add", (8, 1.0)),
                     ("mul", (0, 0))]
    x, y = (torch.randn(4, 8) for _ in range(2))
    for got, want in zip(p.reference(x, y), (x - 2, 3 + x, 2 * x, x / 2, 1 / x, -x, 1 - x, x * x)):
        _close(got.float(), want, tol=1e-2)


def test_unsupported_ops_raise():
    p, a, b = _p()
    with pytest.raises(SpecError, match="keepdim"):
        a.sum(-1)
    with pytest.raises(SpecError, match="keepdim"):
        a.max(-1)
    with pytest.raises(SpecError, match="transposed"):
        a.T + 1
    with pytest.raises(SpecError, match="inner dims"):
        a @ b
    with pytest.raises(SpecError):
        bool(a)
    with pytest.raises(SpecError, match="paged program"):
        cl.paged_gather(a, cl.PageTable(a, b))


def test_eager_dispatch_on_torch():
    x = torch.rand(4, 8) + 0.5
    _close(cl.rsqrt(x), torch.rsqrt(x), tol=1e-5)
    _close(cl.softmax(x, -1, shift=-3.0), torch.softmax(x, -1), tol=1e-5)
    _close(cl.rms_norm(x, eps=1e-6), x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6), tol=1e-5)
    _close(cl.silu(x), torch.nn.functional.silu(x), tol=1e-5)


# ------------------------------------------------------------------ jit


def test_jit_compile_calls_pipeline_lazily(monkeypatch):
    calls = []

    def fake_compile(program, **kw):
        calls.append((program, kw))
        return lambda *ts: "kernel"

    monkeypatch.setitem(sys.modules, "cleave.pipeline", types.SimpleNamespace(compile=fake_compile))

    @jit(split=cl.Split("K", factors=(4, 8)), name="mm")
    def mm(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
        return x @ w

    assert mm.compile(M=16, N=8, K=32, preset="fast")() == "kernel"
    (prog, kw) = calls[0]
    assert prog.name == "mm" and prog.sizes == dict(M=16, K=32, N=8)
    assert kw == dict(preset="fast", split=cl.Split("K", (4, 8)))
    assert prog.split == (cl.Split("K", (4, 8)),)
    x, w = torch.randn(16, 32).half(), torch.randn(32, 8).half()
    assert mm(x, w) == "kernel" and mm(w=w, x=x) == "kernel"  # compiled shapes -> kernel
    x2 = torch.randn(4, 32)
    _close(mm(x2, w.float()), x2 @ w.float(), tol=1e-3)  # other shapes -> eager torch
    assert mm(x.float(), w.float()) != "kernel"  # other dtypes -> eager torch

    mm.compile(M=16, N=8, K=32, dtype=cl.bf16)
    assert calls[1][0].dtype == cl.bf16 and "dtype" not in calls[1][1]
    assert mm(x.bfloat16(), w.bfloat16()) == "kernel"


def test_trace_accepts_plain_function():
    def f(x: cl.Tensor["M", "K"], scale=2.0):
        return cl.exp(x * scale)

    prog = trace(f, dict(M=2, K=4), cl.f32)
    assert prog.dtype == cl.f32 and [r.kind for r in prog.ops] == ["input", "mul", "exp"]
