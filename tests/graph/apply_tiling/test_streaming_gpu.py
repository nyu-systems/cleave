"""The searched path end to end on a GPU: every family the assembly serves, validated.

Under the switch (``searched_updates``) ``cl.softmax`` records the explicit max; the flat
search keeps it, lowering derives the update rule of every streaming obstacle, the tiler
carries the max where a loop steps its row, and each kernel is validated against the
reference of its own flat graph -- which a graph whose exp reads ``x - max(x)`` is, by
structure, on normal inputs and on every adversarial softmax case its inputs can be shaped
into (``runtime.inputs``). Each test holds its kernels to the bound its operator declares
(``tests/_tolerances.py``); none of them is loosened here.

The full MHA prefill census on a GPU (68 plans, against the hand-derived pass per tiling)
is an evidence run of the package, not a test: it takes the better part of an hour.
"""

import pytest

torch = pytest.importorskip("torch")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.frontend import trace
from cleave.runtime.inputs import ADVERSARIAL
from cleave.schedule.streaming import sites
from cleave.schedule.tiling import schedule_kernel

from tests import _explicit_softmax as X
from tests._tolerances import tolerance_for

EVERY_CASE = ("normal",) + ADVERSARIAL
FAST = dict(preset="fast", schedule=P.ScheduleConfig(topk=16),
            eval=P.EvalConfig(warmup=2, iters=20, max_parallel=4))

pytestmark = pytest.mark.usefixtures("searched_updates")


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    return tmp_path


def _gpu():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")


def _looped(plan) -> bool:
    return bool(sites(schedule_kernel(plan.tile)))


def _held(report, cases=EVERY_CASE, note=""):
    """Every kernel that compiled is right on every case it was held to, and those are the
    cases the test says."""
    assert all(r.cases == cases and r.cases_note.startswith(note) for r in report), [
        (r.label, r.cases, r.cases_note) for r in report]
    wrong = [(r.label, r.error_type, str(r.error)[-300:]) for r in report
             if r.error_type in ("OutputMismatch", "NonFiniteOutput")]
    assert not wrong, wrong
    assert report.correct


def _build(plans, tolerance, **kw):
    return P.build(plans, config=P.EvalConfig(warmup=2, iters=20, max_parallel=4,
                                              tolerance=tolerance, **kw))


def attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
              v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


def gqa(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"],
        v: cl.Tensor["HKV", "SKV", "D"]):
    return cl.matmul(cl.softmax(cl.matmul(q, k.mT) / q.shape[-1] ** 0.5, dim=-1), v)


def test_gpu_a_short_row_has_one_tile_and_looped_kernels(cache_root):
    """S_KV = 256: the flash form has tilings that hold the row whole (the flat graph's own
    max) and tilings that loop over it (the max carried); the traced order only the first
    kind. One of each kind and graph, held to every case."""
    _gpu()
    tolerance = tolerance_for("exact_attention")
    program = trace(attention, dict(H=32, SQ=64, SKV=256, D=64), cl.f16)
    graphs = P.superoptimize(program, P.SuperoptConfig(max_ops=11))
    plans = P.schedule(graphs, P.ScheduleConfig(topk=1000))
    picked, seen = [], set()
    for plan in plans:
        kind = (plan.graph_index, _looped(plan), plan.n_threads)
        if kind not in seen:
            seen.add(kind)
            picked.append(plan)
    assert {(g, looped) for g, looped, _ in seen} == {
        (g.index, looped) for g in graphs for looped in (False, True)
        if looped is False or g.summary().endswith("matmul>div")}
    report = _build(plans.filter(lambda p: any(p is q for q in picked)), tolerance)
    _held(report)
    assert {(r.plan.graph_index, _looped(r.plan)) for r in report.correct} == {
        (g, looped) for g, looped, _ in seen}


@pytest.mark.parametrize("hkv", [8, 1], ids=["gqa", "mqa"])
def test_gpu_grouped_prefill(cache_root, hkv):
    """GQA and MQA prefill at the demos' shapes (32 query heads over 8 and over 1 kv head,
    S_Q = S_KV = 4096): every plan loops over the keys with the max carried; the first
    four are held to every case."""
    _gpu()
    tolerance = tolerance_for("exact_attention")
    program = trace(gqa, dict(HQ=32, HKV=hkv, SQ=4096, SKV=4096, D=64), cl.f16)
    plans = P.schedule(P.superoptimize(program, P.SuperoptConfig(max_ops=11)),
                       P.ScheduleConfig(topk=4))
    assert len(plans) == 4 and all(_looped(p) for p in plans)
    _held(_build(plans, tolerance))


def test_gpu_hand_written_max_attention(cache_root):
    """``exp(s - max(s))`` written by hand is the graph ``cl.softmax`` records: carried
    over S_KV = 4096 and held to every adversarial case (by structure, not by request)."""
    _gpu()
    tolerance = tolerance_for("exact_attention")
    program = trace(X.flash_attention, dict(H=32, SQ=64, SKV=4096, D=64), cl.f16)
    plans = P.schedule(P.superoptimize(program, P.SuperoptConfig(max_ops=11)),
                       P.ScheduleConfig(topk=4))
    assert plans and all(_looped(p) for p in plans)
    _held(_build(plans, tolerance))


def test_gpu_logits_that_are_an_input(cache_root):
    """``softmax(x) @ v`` with ``x`` an fp16 input: the max reads storage, every plan
    carries it over C = 4096."""
    _gpu()
    tolerance = tolerance_for("flash_rows")
    program = trace(X.flash_rows, dict(R=64, C=4096, D=64), cl.f16)
    plans = P.schedule(P.superoptimize(program, P.SuperoptConfig()), P.ScheduleConfig(topk=4))
    assert plans and all(_looped(p) for p in plans)
    _held(_build(plans, tolerance))


def softmax_first_axis(x: cl.Tensor["H", "K", "N"], y: cl.Tensor["H", "K", "N"]):
    return cl.sum(cl.softmax(x, 1) * y, 1, keepdim=True)


def test_gpu_a_softmax_over_the_first_axis(cache_root):
    """``sum(softmax(x, 1) * y, 1)`` at 4x4096x64: the softmax runs over the first non-batch
    axis, and every plan of every graph is held to every case. The flash form
    (``sum(e * y) / sum(e)``) loops over K with the max carried; in the forms that divide
    before the second sum the max keeps its pin, and it may run a loop of its own ahead of
    what reads it (two passes) -- kernels the hand-derived pass never had."""
    _gpu()
    tolerance = tolerance_for("softmax_first_axis")
    program = trace(softmax_first_axis, dict(H=4, K=4096, N=64), cl.f16)
    graphs = P.superoptimize(program, P.SuperoptConfig())
    plans = P.schedule(graphs, P.ScheduleConfig(topk=1000))
    looped = [p for p in plans if _looped(p)]
    assert looped and len(looped) < len(plans)
    assert {p.graph_index for p in plans} == {g.index for g in graphs}
    report = _build(plans, tolerance)
    _held(report)
    assert len(report.correct) == len(plans)


def softmax(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1)


def softmax_mid(x: cl.Tensor["H", "C", "R"]):
    return cl.softmax(x, 1)


def test_gpu_a_returned_softmax(cache_root):
    """Its obstacle is refused (the exp is read next to its finished sum): it keeps its pin
    and is served while the row fits a block, by the flat graph's own max."""
    _gpu()
    kernel = P.compile(softmax, shapes=dict(R=64, C=1024), dtype=cl.f16, split=False,
                       tolerance=tolerance_for("softmax"), **FAST)
    _held(kernel.report)
    assert not any(_looped(r.plan) for r in kernel.report.results)


def test_gpu_a_returned_softmax_over_a_middle_axis(cache_root):
    _gpu()
    kernel = P.compile(softmax_mid, shapes=dict(H=4, C=256, R=64), dtype=cl.f16, split=False,
                       tolerance=tolerance_for("softmax_mid"), **FAST)
    _held(kernel.report)


def query_softmax(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                  v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-2) @ v


def test_gpu_a_softmax_over_the_queries(cache_root):
    """``dim=-2`` at a shape whose column fits a block: the max sits in the loop of the
    value matmul over the keys and takes each column whole; the key operand of the scores
    is the queries (``key_operand == 0``), and the cases are laid along them."""
    _gpu()
    kernel = P.compile(query_softmax, shapes=dict(H=32, SQ=64, SKV=1024, D=64), dtype=cl.f16,
                       split=False, superopt=P.SuperoptConfig(max_ops=11),
                       tolerance=tolerance_for("exact_attention"), **FAST)
    _held(kernel.report)


def softmax_plus_exp(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1) + cl.exp(x)


def test_gpu_softmax_next_to_an_exp_of_its_logits(cache_root):
    """The plain ``exp(x)`` next to the softmax makes no claim, and reads the same input:
    no adversarial case is run, and the report says why. On logits around -100, where
    ``exp`` is 0 in float32, only the exact softmax has anything to divide by."""
    _gpu()
    shapes = dict(R=64, C=256)
    kernel = P.compile(softmax_plus_exp, shapes=shapes, dtype=cl.f16, split=False,
                       tolerance=tolerance_for("softmax_plus_exp"), **FAST)
    _held(kernel.report, cases=("normal",), note="input ")
    assert all("an exp that is not stabilized" in r.cases_note for r in kernel.report.results)
    x = (torch.randn(64, 256) - 100.0).to("cuda", torch.float16)
    assert float(kernel(x).float().sum(-1).min()) > 0.99  # the rows are softmax rows


def dual_softmax(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1) * cl.softmax(x, 0)


def test_gpu_two_softmaxes_of_one_input(cache_root):
    """Two maxes, each over its own axis, both kept in the flat graph: the kernels hold the
    whole tensor and are held to every adversarial case."""
    _gpu()
    kernel = P.compile(dual_softmax, shapes=dict(R=16, C=32), dtype=cl.f16, split=False,
                       tolerance=tolerance_for("dual_softmax"), **FAST)
    _held(kernel.report)
