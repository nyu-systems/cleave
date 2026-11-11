"""Exact softmax ``exp(x - max(x)) / sum``: tensor-tensor sub and the max reduction.

``test_search_*`` run the superoptimizer, ``test_gpu_*`` compile kernels (root conftest marks).
"""

import pytest

torch = pytest.importorskip("torch")

import cleave
import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import SpecError
from cleave.frontend import Program, trace


def softmax(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1)


def softmax_shift(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1, shift=-4.0)


def attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
              v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


def _kinds(prog):
    return [r.kind for r in prog.ops]


def test_exact_softmax_traces_max_and_sub():
    prog = trace(softmax, {"R": 4, "C": 32}, cl.f32)
    assert _kinds(prog) == ["input", "reduce_max", "sub", "exp", "reduce_sum", "div"]
    assert prog.ops[1].attr("dim") == 1 and prog._values[prog.ops[1].out].shape == (4, 1)
    ops = [o["op_type"] for o in prog.to_graph_structure()]
    assert ops == ["kn_input_op", "kn_reduce_max_1_op", "kn_sub_op", "kn_exp_op",
                   "kn_reduce_sum_1_op", "kn_div_op", "kn_output_op"]
    x = torch.randn(4, 32) * 30
    torch.testing.assert_close(prog.reference(x)[0], torch.softmax(x, -1))


def test_shift_softmax_is_unchanged():
    prog = trace(softmax_shift, {"R": 4, "C": 32}, cl.f32)
    assert _kinds(prog) == ["input", "add", "exp", "reduce_sum", "div"]


def test_tensor_sub_and_max_builder():
    p = Program("p", cl.f32)
    a, b = p.input("a", [4, 8]), p.input("b", [4, 1])
    p.output(a - b)
    p.output(b - a)
    p.output(a.max(0, keepdim=True))
    p.output(cl.max(a, -1, keepdim=True))
    assert [(r.kind, r.args) for r in p.ops[2:]] == [
        ("sub", (0, 1)), ("sub", (1, 0)), ("reduce_max", (0,)), ("reduce_max", (0,))]
    assert [r.attr("dim") for r in p.ops[4:]] == [0, 1]
    x, y = torch.randn(4, 8), torch.randn(4, 1)
    want = (x - y, y - x, x.amax(0, keepdim=True), x.amax(-1, keepdim=True))
    for got, w in zip(p.reference(x, y), want):
        torch.testing.assert_close(got, w)
    with pytest.raises(SpecError, match="keepdim"):
        a.max(-1)
    with pytest.raises(SpecError, match="out of range"):
        a.max(2, keepdim=True)
    with pytest.raises(SpecError, match="broadcastable"):
        a - p.input("c", [4, 3])
    assert a.max(-1, keepdim=True).shape == (4, 1)


def test_eager_dispatch():
    x, y = torch.randn(4, 8) * 20, torch.randn(4, 8)
    torch.testing.assert_close(cl.softmax(x, -1), torch.softmax(x, -1))
    torch.testing.assert_close(cl.max(x, 1, keepdim=True), x.amax(1, keepdim=True))
    torch.testing.assert_close(cl.max(x), x.amax(-1))
    torch.testing.assert_close(cleave.jit(softmax)(x), torch.softmax(x, -1))
    assert torch.equal(x - y, x - y)


def test_paged_program_rejects_sub():
    from cleave.frontend.paged import PagedProgram

    p = PagedProgram.__new__(PagedProgram)
    with pytest.raises(SpecError, match="dense program"):
        p._binary("sub", None, None)


def test_search_exact_softmax():
    prog = trace(softmax, {"R": 16, "C": 256}, cl.f16)
    graphs = P.superoptimize(prog, P.SuperoptConfig(cache=False))
    assert len(graphs) >= 1 and all({"max", "sub"} <= set(g.ops) for g in graphs)


def test_search_max_row_limit_reason(tmp_path, monkeypatch):
    from cleave.core.errors import CleaveError
    from cleave.pipeline.plans import MAX_ROW_LIMIT

    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    shapes = dict(H=32, SQ=64, SKV=4096, D=64)
    prog = trace(attention, shapes, cl.f16)
    plans = P.schedule(P.superoptimize(prog, P.SuperoptConfig(max_ops=14, cache=False)),
                       P.ScheduleConfig(topk=2))
    assert not plans and plans.refused
    assert {str(r.error) for r in plans.refused} == {MAX_ROW_LIMIT}
    with pytest.raises(CleaveError, match="online softmax") as e:
        P.compile(attention, shapes=shapes, dtype=cl.f16, split=False, preset="fast",
                  superopt=P.SuperoptConfig(max_ops=14), schedule=P.ScheduleConfig(topk=2))
    assert MAX_ROW_LIMIT in e.value.report.summary()


# ------------------------------------------------------------------ GPU

FAST = dict(preset="fast", schedule=P.ScheduleConfig(topk=16),
            eval=P.EvalConfig(warmup=2, iters=20, max_parallel=4))


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    return tmp_path


def _gpu():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")


def test_gpu_exact_softmax_standalone(cache_root):
    _gpu()
    kernel = P.compile(softmax, shapes={"R": 64, "C": 1024}, dtype=cl.f16, split=False, **FAST)
    assert len(kernel.report.correct) >= 1
    x = (torch.randn(64, 1024) * 30).to("cuda", torch.float16)  # exp(x + c) would overflow
    torch.testing.assert_close(kernel(x).float(), torch.softmax(x.float(), -1),
                               rtol=2e-2, atol=2e-3)


def test_gpu_exact_softmax_attention(cache_root):
    _gpu()
    shapes = dict(H=32, SQ=64, SKV=256, D=64)
    kernel = P.compile(attention, shapes=shapes, dtype=cl.f16, split=False,
                       superopt=P.SuperoptConfig(max_ops=14), **FAST)
    assert kernel.variant == "solo" and len(kernel.report.correct) >= 1
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(s, generator=g).to("cuda", torch.float16)
               for s in [(32, 64, 64), (32, 256, 64), (32, 256, 64)])
    q = q * 4  # scores far from the fixed -50/sqrt(d) shift
    ref = torch.softmax((q.float() @ k.float().mT) / 8.0, -1) @ v.float()
    torch.testing.assert_close(kernel(q, k, v).float(), ref, rtol=2e-2, atol=2e-2)
