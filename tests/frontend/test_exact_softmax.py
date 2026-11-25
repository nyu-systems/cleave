"""Exact softmax ``exp(x - max(x)) / sum``.

``cl.softmax`` records ``reduce_max, sub, exp, reduce_sum, div``: the max written out, the
same program as ``cl.exp(x - cl.max(x))`` divided by its sum by hand. The max is an opaque
reduction of the searched graph; where a tiling loops over the softmax row under it, the
loop carries the max with a searched and proven update rule (the online softmax), and a
tile that holds the row whole takes the flat graph's own max. A softmax that is returned
itself stays limited to rows that fit one block.

``test_search_*`` run the superoptimizer, ``test_gpu_*`` compile kernels (root conftest marks).
"""

import re

import pytest

torch = pytest.importorskip("torch")

import cleave
import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import SpecError
from cleave.frontend import Program, trace
from tests._tolerances import tolerance_for


def softmax(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1)


def softmax_fixed_shift(x: cl.Tensor["R", "C"]):
    """A constant shift written out by hand: ``cl.softmax`` has no such parameter."""
    e = cl.exp(x + -4.0)
    return e / cl.sum(e, -1, keepdim=True)


def softmax_unstabilized(x: cl.Tensor["R", "C"]):
    """The twin of ``softmax`` without the max: a plain ``exp / sum / div``."""
    e = cl.exp(x)
    return e / cl.sum(e, -1, keepdim=True)


def softmax_mid(x: cl.Tensor["H", "C", "R"]):
    return cl.softmax(x, 1)


def attention_over_queries(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                           v: cl.Tensor["H", "SKV", "D"]):
    """Attention normalized over the queries: every key's column of weights sums to 1."""
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-2) @ v


def softmax_by_hand(x: cl.Tensor["R", "C"]):
    e = cl.exp(x - cl.max(x, -1, keepdim=True))
    return e / cl.sum(e, -1, keepdim=True)


def attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
              v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


def attention_by_hand(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                      v: cl.Tensor["H", "SKV", "D"]):
    """The max written out: what ``cl.softmax`` records."""
    s = (q @ k.mT) / q.shape[-1] ** 0.5
    e = cl.exp(s - cl.max(s, -1, keepdim=True))
    return (e / cl.sum(e, -1, keepdim=True)) @ v


def attention_unstabilized(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                           v: cl.Tensor["H", "SKV", "D"]):
    """The twin of ``attention`` without the max: a plain ``exp / sum / div``."""
    e = cl.exp((q @ k.mT) / q.shape[-1] ** 0.5)
    return (e / cl.sum(e, -1, keepdim=True)) @ v


def softmax_plus_exp(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1) + cl.exp(x)


def dual_softmax(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1) * cl.softmax(x, 0)


def _kinds(prog):
    return [r.kind for r in prog.ops]


def _records_a_max(prog) -> bool:
    return "reduce_max" in _kinds(prog)


def test_exact_softmax_records_the_max():
    prog = trace(softmax, {"R": 4, "C": 32}, cl.f32)
    assert _kinds(prog) == ["input", "reduce_max", "sub", "exp", "reduce_sum", "div"]
    assert prog.ops[1].attr("dim") == 1 and prog._values[prog.ops[1].out].shape == (4, 1)
    assert prog.ops[4].attr("dim") == 1 and prog._values[prog.ops[4].out].shape == (4, 1)
    # the search is handed the max and the subtraction as recorded
    ops = [o["op_type"] for o in prog.to_graph_structure()]
    assert ops == ["kn_input_op", "kn_reduce_max_1_op", "kn_sub_op", "kn_exp_op",
                   "kn_reduce_sum_1_op", "kn_div_op", "kn_output_op"]
    x = torch.randn(4, 32) * 30
    torch.testing.assert_close(prog.reference(x)[0], torch.softmax(x, -1))
    # the max is in the fingerprint: the program never shares a search or kernel cache
    # entry with the plain exp / sum, which makes no claim
    plain = trace(softmax_unstabilized, {"R": 4, "C": 32}, cl.f32)
    assert _kinds(plain) == ["input", "exp", "reduce_sum", "div"]
    assert plain.fingerprint() != prog.fingerprint()


def test_max_written_by_hand_traces_max_and_sub():
    """... which is ``cl.softmax``: the same records, the same fingerprint."""
    prog = trace(softmax_by_hand, {"R": 4, "C": 32}, cl.f32)
    assert _kinds(prog) == ["input", "reduce_max", "sub", "exp", "reduce_sum", "div"]
    assert prog.ops[1].attr("dim") == 1 and prog._values[prog.ops[1].out].shape == (4, 1)
    assert prog.fingerprint() == trace(softmax, {"R": 4, "C": 32}, cl.f32).fingerprint()
    ops = [o["op_type"] for o in prog.to_graph_structure()]
    assert ops == ["kn_input_op", "kn_reduce_max_1_op", "kn_sub_op", "kn_exp_op",
                   "kn_reduce_sum_1_op", "kn_div_op", "kn_output_op"]
    x = torch.randn(4, 32) * 30
    torch.testing.assert_close(prog.reference(x)[0], torch.softmax(x, -1))


def test_softmax_takes_no_shift():
    """The fixed shift is gone from the frontend, with no stand-in: ``cl.softmax`` is the
    exact softmax and nothing else, traced, eager and paged."""
    import inspect

    from cleave.frontend import composites, paged

    assert list(inspect.signature(composites.softmax).parameters) == ["x", "dim"]
    assert list(inspect.signature(cl.softmax).parameters) == ["x", "dim", "return_lse", "causal"]
    assert cl.softmax is paged.softmax
    p = Program("p", cl.f32)
    traced = p.input("x", [4, 32])
    for x in (torch.randn(4, 32), traced):
        with pytest.raises(TypeError, match="shift"):
            cl.softmax(x, -1, shift=-4.0)
        with pytest.raises(TypeError, match="shift"):
            composites.softmax(x, -1, shift=-4.0)
        # ... nor by position: the options behind ``dim`` are keyword-only, so a call
        # written for the old signature is not read as ``return_lse``
        with pytest.raises(TypeError):
            composites.softmax(x, -1, -4.0)
        with pytest.raises(TypeError):
            cl.softmax(x, -1, -4.0)
    assert [r.kind for r in p.ops] == ["input"]
    kinds = {name: param.kind for name, param in inspect.signature(cl.softmax).parameters.items()}
    assert kinds["return_lse"] is kinds["causal"] is inspect.Parameter.KEYWORD_ONLY


def test_constant_shift_written_by_hand_makes_no_claim():
    """``exp(x + c) / sum(exp(x + c))`` stays expressible: an add, a plain exp, the sum and
    the division. Its exp does not read ``x - max(x)``, so it is no exact softmax."""
    from cleave.pipeline.plans import lower
    from cleave.pipeline.superopt import GraphSet
    from cleave.runtime.inputs import softmax_axis

    prog = trace(softmax_fixed_shift, {"R": 4, "C": 32}, cl.f32)
    assert _kinds(prog) == ["input", "add", "exp", "reduce_sum", "div"]
    x = torch.randn(4, 32)
    torch.testing.assert_close(prog.reference(x)[0], torch.softmax(x, -1))
    (ir,) = lower(next(iter(GraphSet.identity(prog))))
    assert [softmax_axis(n) for n in ir.nodes if type(n).__name__ == "Exp"] == [None]


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
    """Every searched graph of ``cl.softmax`` contains its max: an opaque reduction of the
    search space, which the search does not move or split."""
    prog = trace(softmax, {"R": 16, "C": 256}, cl.f16)
    graphs = P.superoptimize(prog, P.SuperoptConfig(cache=False))
    assert len(graphs) >= 1 and all({"max", "sub"} <= set(g.ops) for g in graphs)
    assert all({"exp", "sum", "div"} <= set(g.ops) for g in graphs)


def test_search_max_written_by_hand_keeps_its_max():
    prog = trace(softmax_by_hand, {"R": 16, "C": 256}, cl.f16)
    graphs = P.superoptimize(prog, P.SuperoptConfig(cache=False))
    assert len(graphs) >= 1 and all({"max", "sub"} <= set(g.ops) for g in graphs)


def test_search_a_max_written_by_hand_is_carried_across_a_loop(tmp_path, monkeypatch):
    """A max written by hand is the max of ``cl.softmax``: at S_KV = 4096 the flash form the
    search finds carries it across the loop over the keys, and the traced order, whose value
    matmul reads the finished row sum, says why it has no tiling."""
    from cleave.pipeline.plans import ROW_LIMIT
    from cleave.schedule import apply_tiling
    from cleave.schedule.streaming import sites

    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    shapes = dict(H=32, SQ=64, SKV=4096, D=64)
    prog = trace(attention_by_hand, shapes, cl.f16)
    assert prog.fingerprint() == trace(attention, shapes, cl.f16).fingerprint()
    plans = P.schedule(P.superoptimize(prog, P.SuperoptConfig(max_ops=14, cache=False)),
                       P.ScheduleConfig(topk=2))
    assert len(plans) == 2 and all(sites(apply_tiling(p.tile)) for p in plans)
    assert {p.graph.summary() for p in plans} == {"solo matmul>div>max>sub>exp>sum>matmul>div"}
    (refusal,) = plans.refused
    assert str(refusal.error).startswith(f"{ROW_LIMIT}; it is not carried across a loop: ")
    assert "a finished reduction over the dim" in str(refusal.error)


def attention_weights(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1)


def attention_weights_by_hand(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"]):
    s = (q @ k.mT) / q.shape[-1] ** 0.5
    e = cl.exp(s - cl.max(s, -1, keepdim=True))
    return e / cl.sum(e, -1, keepdim=True)


def test_search_a_returned_softmax_is_not_promised_a_row_of_any_length(tmp_path, monkeypatch):
    """Attention weights over 4096 keys: the softmax is the output, so nothing sums it and
    no loop can carry it. Written with a max by hand or with ``cl.softmax`` (one program)
    it has no plan, and the refusal names the reason after the row limit: the exp, computed
    from the max for every key, is also read by the division, which is not summed over the
    keys. ``compile`` ends on that report."""
    from cleave.core.errors import CleaveError
    from cleave.pipeline.plans import ROW_LIMIT

    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    shapes = dict(H=32, SQ=64, SKV=4096, D=64)
    by_hand, exact = (trace(fn, shapes, cl.f16)
                      for fn in (attention_weights_by_hand, attention_weights))
    assert by_hand.fingerprint() == exact.fingerprint()
    plans = P.schedule(P.superoptimize(exact, P.SuperoptConfig(cache=False)),
                       P.ScheduleConfig(topk=2))
    assert not plans and plans.refused
    assert len(ROW_LIMIT) <= 160  # a refusal's reason is one line of that many characters
    assert "cl.softmax" not in ROW_LIMIT and "a row of any length" not in ROW_LIMIT
    for refusal in plans.refused:
        text = str(refusal.error)
        assert text.startswith(f"{ROW_LIMIT}; it is not carried across a loop: "), text
        assert re.search(r"exp_\d+, computed from reduce_max_\d+_\d+ for every element of the "
                         r"looped dim, is also read by div_\d+, which is not summed over that "
                         r"dim", text), text
    with pytest.raises(CleaveError) as e:
        P.compile(attention_weights_by_hand, shapes=shapes, dtype=cl.f16, split=False,
                  preset="fast", schedule=P.ScheduleConfig(topk=2),
                  tolerance=tolerance_for("softmax"))
    assert ROW_LIMIT in e.value.report.summary()


def test_search_long_rows_schedule_with_rolling_plans(tmp_path, monkeypatch):
    """The same attention through ``cl.softmax``: S_KV = 4096 schedules, and every plan
    carries the max over the loop that tiles the keys (the online softmax)."""
    from cleave.pipeline.plans import ROW_LIMIT
    from cleave.schedule import apply_tiling
    from cleave.schedule.streaming import sites

    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    shapes = dict(H=32, SQ=64, SKV=4096, D=64)
    prog = trace(attention, shapes, cl.f16)
    plans = P.schedule(P.superoptimize(prog, P.SuperoptConfig(cache=False)),
                       P.ScheduleConfig(topk=8))
    assert len(plans) == 8 and {p.variant for p in plans} == {"solo"}
    for plan in plans:
        (site,) = sites(apply_tiling(plan.tile))
        assert site.extent == site.block == 4096
        assert 1 < site.step < 4096 and len(site.repaired) == 2  # the row sum and P @ V
    # they are the flash form (the row sum next to the value matmul, the division last);
    # the traced order reads the exp next to its finished sum, so it has no looped form
    (flash,) = {p.graph.summary() for p in plans}
    assert flash == "solo matmul>div>max>sub>exp>sum>matmul>div"
    ((variant, text),) = [(r.variant, str(r.error)) for r in plans.refused]
    assert variant == "solo" and text.startswith(f"{ROW_LIMIT}; it is not carried across ")


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


def _hold_returned_kernel(kernel, fn, shapes, inputs, operator):
    """The kernel ``compile`` returned, on logits far larger than the ones it was validated
    and measured on, against the program's own reference at the bound the operator
    declares. (Logged as the ``large_logits`` case: held to the bound, no input to it.
    These checks compared elementwise at 2e-2 + 2e-2 |ref| or 2e-3 against a float32
    formula: 1.5% off everywhere passed.)"""
    from cleave.runtime.validate import compare_outputs

    reference = trace(fn, shapes, cl.f16).reference(*inputs)
    assert all(torch.isfinite(r).all() for r in reference)
    out = kernel(*inputs)
    assert torch.isfinite(out).all()
    return compare_outputs([out], reference, tolerance_for(operator), what="large_logits inputs")


def test_the_returned_kernel_check_has_the_declared_bound(monkeypatch):
    """CPU: what ``_hold_returned_kernel`` accepts. The reference itself passes; the
    reference 1.5% off, which the elementwise comparison it replaces accepted, does not."""
    from cleave.core.errors import OutputMismatch

    # (References and made-up tensors, no kernel: nothing here belongs in a measurement.)
    monkeypatch.delenv("CLEAVE_VALIDATION_LOG", raising=False)
    shapes = {"R": 64, "C": 1024}
    x = (torch.randn(64, 1024) * 30).to(torch.float16)
    (reference,) = trace(softmax, shapes, cl.f16).reference(x)
    _hold_returned_kernel(lambda x: reference.clone(), softmax, shapes, [x], "softmax")
    off = (reference.float() * 1.015).to(torch.float16)
    torch.testing.assert_close(off.float(), torch.softmax(x.float(), -1), rtol=2e-2, atol=2e-3)
    with pytest.raises(OutputMismatch):
        _hold_returned_kernel(lambda x: off, softmax, shapes, [x], "softmax")


def test_gpu_exact_softmax_standalone(cache_root):
    _gpu()
    kernel = P.compile(softmax, shapes={"R": 64, "C": 1024}, dtype=cl.f16, split=False,
                       tolerance=tolerance_for("softmax"), **FAST)
    assert len(kernel.report.correct) >= 1
    x = (torch.randn(64, 1024) * 30).to("cuda", torch.float16)  # exp(x + c) would overflow
    _hold_returned_kernel(kernel, softmax, {"R": 64, "C": 1024}, [x], "softmax")


def test_gpu_exact_softmax_attention(cache_root):
    _gpu()
    shapes = dict(H=32, SQ=64, SKV=256, D=64)
    kernel = P.compile(attention, shapes=shapes, dtype=cl.f16, split=False,
                       superopt=P.SuperoptConfig(max_ops=14),
                       tolerance=tolerance_for("exact_attention"), **FAST)
    assert kernel.variant == "solo" and len(kernel.report.correct) >= 1
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(s, generator=g).to("cuda", torch.float16)
               for s in [(32, 64, 64), (32, 256, 64), (32, 256, 64)])
    q = q * 4  # scores a constant shift chosen for unit-variance logits would not suit
    _hold_returned_kernel(kernel, attention, shapes, [q, k, v], "exact_attention")


def test_gpu_exact_softmax_over_a_middle_axis(cache_root):
    """``softmax(x, 1)`` of a batched input: the softmax axis is not the last one, and the
    batch dim in front of it is a unit extent the stored tile does not have. Every plan
    takes the max in place over the right axis of that tile."""
    _gpu()
    from cleave.runtime.inputs import ADVERSARIAL

    kernel = P.compile(softmax_mid, shapes=dict(H=4, C=256, R=64), dtype=cl.f16, split=False,
                       tolerance=tolerance_for("softmax_mid"), **FAST)
    report = kernel.report
    assert len(report.correct) >= 1
    assert all(r.cases == ("normal",) + ADVERSARIAL for r in report.results)
    x = (torch.randn(4, 256, 64) * 30).to("cuda", torch.float16)
    _hold_returned_kernel(kernel, softmax_mid, dict(H=4, C=256, R=64), [x], "softmax_mid")


def test_gpu_exact_softmax_over_the_queries(cache_root):
    """Attention normalized over the queries (``dim=-2``): the exp sits in the loop of the
    value matmul, which tiles the keys, and each iteration takes the max over the whole
    query axis of its tile."""
    _gpu()
    from cleave.runtime.inputs import ADVERSARIAL

    shapes = dict(H=32, SQ=64, SKV=1024, D=64)
    kernel = P.compile(attention_over_queries, shapes=shapes, dtype=cl.f16, split=False,
                       tolerance=tolerance_for("exact_attention"), **FAST)
    report = kernel.report
    assert kernel.variant == "solo" and len(report.correct) >= 1
    assert all(r.cases == ("normal",) + ADVERSARIAL for r in report.results)
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(s, generator=g).to("cuda", torch.float16)
               for s in [(32, 64, 64), (32, 1024, 64), (32, 1024, 64)])
    q = q * 40  # logits up to several hundred
    _hold_returned_kernel(kernel, attention_over_queries, shapes, [q, k, v], "exact_attention")


def test_gpu_exact_softmax_long_rows(cache_root):
    """S_KV = 4096: every plan carries a running max (the online softmax). The kernels are
    held to the adversarial softmax inputs by the build, and the returned one is checked
    here on logits a fixed shift cannot hold."""
    _gpu()
    shapes = dict(H=32, SQ=64, SKV=4096, D=64)
    kernel = P.compile(attention, shapes=shapes, dtype=cl.f16, split=False,
                       tolerance=tolerance_for("exact_attention"),
                       **{**FAST, "schedule": P.ScheduleConfig(topk=4)})
    assert kernel.variant == "solo" and len(kernel.report.correct) >= 1
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(s, generator=g).to("cuda", torch.float16)
               for s in [(32, 64, 64), (32, 4096, 64), (32, 4096, 64)])
    q = q * 40  # logits up to several hundred: exp of them overflows fp32
    _hold_returned_kernel(kernel, attention, shapes, [q, k, v], "exact_attention")


def test_gpu_softmax_next_to_an_exp_of_its_logits(cache_root):
    """``cl.softmax(x) + cl.exp(x)``: the softmax's exp reads ``x - max(x)`` and the other
    one ``x``, two different values the search keeps apart; the second is the plain exp the
    program wrote. The returned kernel is checked on logits around -100, where ``exp`` is
    0 in float32 and only the exact softmax has anything to divide by."""
    _gpu()
    shapes = dict(R=64, C=256)
    kernel = P.compile(softmax_plus_exp, shapes=shapes, dtype=cl.f16, split=False,
                       tolerance=tolerance_for("softmax_plus_exp"), **FAST)
    report = kernel.report
    assert kernel.variant == "solo" and len(report.correct) >= 1
    assert not [r.label for r in report.results
                if r.error_type in ("OutputMismatch", "NonFiniteOutput")]
    # the plain exp reads the same input, so no adversarial case is run, and the report says why
    assert all(r.cases == ("normal",) and "an exp that is not stabilized" in r.cases_note
               for r in report.results)
    x = (torch.randn(64, 256) - 100.0).to("cuda", torch.float16)
    (measured,) = _hold_returned_kernel(kernel, softmax_plus_exp, shapes, [x], "softmax_plus_exp")
    assert float(kernel(x).float().sum(-1).min()) > 0.99  # the rows are softmax rows


def test_gpu_two_softmaxes_of_one_input(cache_root):
    """``cl.softmax(x, -1) * cl.softmax(x, 0)``: two maxes, two exps, each a softmax over
    its own axis, and the kernels are held to every adversarial case."""
    _gpu()
    from cleave.runtime.inputs import ADVERSARIAL

    shapes = dict(R=16, C=32)
    kernel = P.compile(dual_softmax, shapes=shapes, dtype=cl.f16, split=False,
                       tolerance=tolerance_for("dual_softmax"), **FAST)
    report = kernel.report
    assert kernel.variant == "solo" and len(report.correct) >= 1
    assert not [r.label for r in report.results
                if r.error_type in ("OutputMismatch", "NonFiniteOutput")]
    assert all(r.cases == ("normal",) + ADVERSARIAL and not r.cases_note for r in report.results)
    x = (torch.randn(16, 32) * 30).to("cuda", torch.float16)
    _hold_returned_kernel(kernel, dual_softmax, shapes, [x], "dual_softmax")


def wide_value_attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                         v: cl.Tensor["H", "SKV", "DV"]):
    """Attention whose values are wider than its keys (the shape of MLA decode's output)."""
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


def test_gpu_exact_softmax_wide_values_are_rescaled_every_iteration_and_validate(cache_root):
    """A 64 x 512 value accumulator over 128 threads is 256 elements a thread: more than a
    thread's registers. (The hand-derived pass shifted such tilings by a max rounded down
    to a grid and rescaled only when that moved; the searched rule is checked against each
    accumulator's own reference, which that form does not meet, so it is not found.) Such a
    tiling repairs its accumulator every iteration, and is held to every adversarial case
    like any exact kernel: the ramps move the max in every tile, constant rows never again
    after the first."""
    _gpu()
    from cleave.runtime.inputs import ADVERSARIAL
    from cleave.schedule import apply_tiling
    from cleave.schedule.streaming import sites

    shapes = dict(H=8, SQ=64, SKV=1024, D=64, DV=512)
    prog = trace(wide_value_attention, shapes, cl.f16)
    plans = P.schedule(P.superoptimize(prog, P.SuperoptConfig()), P.ScheduleConfig(topk=1000))
    wide = plans.filter(lambda p: list(p.tile.output_tile) == [1, 64, 512]
                        and bool(sites(apply_tiling(p.tile))))
    assert len(wide) >= 1 and {p.n_threads for p in wide} >= {128}
    ev = P.EvalConfig(warmup=2, iters=20, max_parallel=4,
                      tolerance=tolerance_for("exact_attention"))
    built = P.build(wide, config=ev)
    # the claim is about 256 elements a thread: a 128-thread kernel validates (a 256-thread
    # one, 128 elements a thread, does not stand in for it)
    assert [r for r in built.correct if r.plan.n_threads == 128], built
    assert all(r.cases == ("normal",) + ADVERSARIAL for r in built)
    assert not [r for r in built if r.error_type in ("NonFiniteOutput", "OutputMismatch")], built
    for r in built.correct:
        assert "ukernel_rescale_lead1_2dims_rr" in r.source


def _kernel_nans(error) -> int:
    """How many NaN the comparator counted in the kernel's own output at x100, and that the
    failure is about the ``large_logits`` case."""
    found = re.search(r"large_logits inputs: output 0: (\d+) NaN in the kernel output", str(error))
    assert found, str(error)[:300]
    return int(found.group(1))


def test_gpu_unstabilized_twin_is_caught_at_x100(cache_root):
    """The validation has teeth: the twin without the max is a fine kernel on normal
    inputs and fails on logits 100 times larger, classified as non-finite by the comparator.

    The twin's own reference is NaN there as well (a plain ``exp`` overflows in torch too),
    and a NaN on either side is a non-finite failure. So the verdict alone would not tell a
    kernel that overflows from one that does not: what is asserted is that the *kernel's*
    output holds NaN."""
    _gpu()
    shapes = dict(H=32, SQ=64, SKV=256, D=64)
    so, sc = P.SuperoptConfig(max_ops=14), P.ScheduleConfig(topk=4)
    ev = P.EvalConfig(warmup=2, iters=20, max_parallel=4,
                      tolerance=tolerance_for("exact_attention"))

    twin = trace(attention_unstabilized, shapes, cl.f16)
    assert not _records_a_max(twin)
    plans = P.schedule(P.superoptimize(twin, so), sc)
    normal = P.build(plans, config=ev)
    assert len(normal.correct) >= 1
    assert all(r.cases == ("normal",) for r in normal)  # no claim, nothing added
    large = P.build(plans, config=ev.replace(distributions=("normal", "large_logits")))
    assert not large.correct
    for before, after in zip(normal.results, large.results):
        if before.ok:  # the same kernel: only the inputs changed
            assert after.stage == "validate" and after.error_type == "NonFiniteOutput", after
            assert _kernel_nans(after.error) > 0


def test_gpu_unstabilized_kernel_fails_against_the_exact_reference(cache_root):
    """The teeth at the level of one kernel: the tilings of the plain twin (the same
    attention without the max) that the build schedules, held to the exact program's
    reference, which is finite at x100. Each such kernel passes on normal inputs and is
    rejected at x100 for the NaN it returns, against a reference that has none.

    (The twin has no max, so no accept filter: its top tilings loop the row sum while the
    division outside the loop reads the exp, which ``apply_tiling`` refuses with its typed
    reason. The filter here keeps what the build schedules, as ``streaming_accept`` does for
    the exact graph.)"""
    _gpu()
    from cleave.core.errors import ScheduleRejected
    from cleave.pipeline.plans import lower
    from cleave.pipeline.superopt import GraphSet
    from cleave.runtime.evaluate import run_candidates
    from cleave.runtime.reference import placeholder_order
    from cleave.runtime.validate import publish_reference
    from cleave.schedule.tiling import schedule_kernel

    def builds(cand) -> bool:
        try:
            schedule_kernel(cand)
        except ScheduleRejected:
            return False
        return True

    shapes = dict(H=32, SQ=64, SKV=256, D=64)
    (exact,) = lower(next(iter(GraphSet.identity(trace(attention, shapes, cl.f16)))))
    plans = P.schedule(GraphSet.identity(trace(attention_unstabilized, shapes, cl.f16)),
                       P.ScheduleConfig(topk=4, hints={"accept": builds}))
    assert len(plans) >= 1
    # the twin's kernels read the exact graph's inputs, by name
    names = sorted(n.name for n in placeholder_order(exact))
    assert all(sorted(n.name for n in placeholder_order(p.tile.flat_graph)) == names
               for p in plans)

    def exact_reference(reference_cache_dir, graph, dtype):
        return publish_reference(reference_cache_dir, exact, dtype,
                                 distributions=("normal", "large_logits"))

    runs = run_candidates(
        [p.tile for p in plans], "float16", max_parallel=4,
        tolerance=tolerance_for("exact_attention"), reference_fn=exact_reference)
    validated = [r for r in runs if r.ok or r.stage == "validate"]
    assert validated, [(r.stage, str(r.error)[:200]) for r in runs]
    for run in validated:
        # the normal case comes first and passed: the failure names the second case
        assert not run.ok and type(run.error).__name__ == "NonFiniteOutput", run
        assert _kernel_nans(run.error) > 0
        assert re.search(r"NaN in the kernel output, 0 in the reference", str(run.error))


def test_gpu_exact_attention_is_held_to_every_adversarial_case(cache_root, tmp_path, monkeypatch,
                                                              request):
    """The exact program is held to the adversarial inputs by the build itself, whatever
    was asked for, and passes: every kernel that is ok was compared on each case."""
    _gpu()
    import json
    import os
    from pathlib import Path

    from cleave.runtime.inputs import ADVERSARIAL

    log = os.environ.get("CLEAVE_VALIDATION_LOG")  # the session's own, when it keeps one
    if not log:
        log = str(tmp_path / "validation.jsonl")
        monkeypatch.setenv("CLEAVE_VALIDATION_LOG", log)
    shapes = dict(H=32, SQ=64, SKV=256, D=64)
    ev = P.EvalConfig(warmup=2, iters=20, max_parallel=4,
                      tolerance=tolerance_for("exact_attention"))
    exact = trace(attention, shapes, cl.f16)
    assert _records_a_max(exact) and ev.distributions == ("normal",)
    built = P.build(P.schedule(P.superoptimize(exact, P.SuperoptConfig(max_ops=14)),
                               P.ScheduleConfig(topk=4)), config=ev)
    assert len(built.correct) >= 1
    assert all(r.cases == ("normal",) + ADVERSARIAL and not r.cases_note for r in built)
    records = [json.loads(line) for line in Path(log).read_text().splitlines()]
    mine = [r for r in records if request.node.name in r["context"]]
    passed = {case: sum(1 for r in mine if r["what"] == case and r["kind"] is None)
              for case in ["", *(f"{d} inputs" for d in ADVERSARIAL)]}
    assert all(n >= len(built.correct) for n in passed.values()), passed
    assert not [r for r in mine if r["kind"] == "non_finite"]
