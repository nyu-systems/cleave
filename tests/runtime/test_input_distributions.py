"""The validation input distributions (CPU): ``normal`` and the adversarial softmax cases.

``large_logits`` and ``increasing`` are shared by the dense programs and the dynamic
operators. The dense programs have four more (``decreasing``, ``last_spike``, ``constant``,
``offset``), the dynamic operators two (``large_scale``, ``dominant_last``). A dense program
is held to its adversarial cases once its softmax is exact, that is when ``cl.softmax``
recorded the request (a fixed-shift reference is NaN on them); one dynamic operator's
softmax is exact so far. ``test_adversarial_inputs_are_switched_on_per_operator`` pins both.
"""

import inspect
import math

import pytest

torch = pytest.importorskip("torch")

import cleave.lang as cl
from cleave.core.errors import SpecError
from cleave.dynamic import OPERATORS, keyed_workloads
from cleave.dynamic import reference as dyn
from cleave.dynamic.workloads import ADVERSARIAL_WORKLOADS, MLA_SM_SCALE, WORKLOADS
from cleave.ir.graph import OutputNode, PlaceHolderNode
from cleave.ir.graph_utils import find_topo_sort
from cleave.ir.ops import Exp
from cleave.pipeline import EvalConfig
from cleave.runtime import inputs as rt_inputs
from cleave.runtime import reference as rt
from cleave.runtime.inputs import (
    ADVERSARIAL as DENSE_ADVERSARIAL, DISTRIBUTIONS, INCREASING_SPAN, LARGE_LOGIT_FACTOR,
    LOGIT_OFFSET, OFFSET, OFFSET_GRID, SHARED_DISTRIBUTIONS, adversarial_cases, logit_inputs,
    why_not_adversarial,
)

from tests import _graphs as g
from tests._softmax import exact_softmax, lowered
from tests.frontend import _programs as programs

DTYPE = "float16"

# name -> (graph, number of score matmuls). The softmax axis is the last axis of the scores.
ATTENTION = {
    "mha_bhsd": (lambda: g.mha_bhsd_graph(1, 2, 8, 96, 16), 1),
    "mha_bshd": (lambda: g.mha_bshd_graph(1, 8, 96, 2, 16), 1),  # PermuteView operands
    "gqa_bhsd": (lambda: g.gqa_bhsd_graph(1, 4, 2, 8, 96, 16), 1),  # KV heads broadcast
    "mqa_bhsd": (lambda: g.mqa_bhsd_graph(1, 4, 1, 8, 96, 16), 1),
    "mla_prefill_bhsd": (lambda: g.mla_prefill_bhsd_graph(1, 2, 8, 96, 24, 16), 1),
    "mla_decode_bhsd": (lambda: g.mla_decode_bhsd_graph(1, 4, 96, 32, 8), 2),  # two terms
    "gqa_split_partial": (lambda: g.gqa_split_partial_graph(1, 4, 2, 4, 8, 24, 16), 1),
}


def _scores(graph, inputs):
    """What every ``Exp`` of ``graph`` is fed (its logits), by the reference executor."""
    order = find_topo_sort(graph.nodes)
    placeholders = [n for n in order if isinstance(n, PlaceHolderNode)]
    env = {(p, 0): t for p, t in zip(placeholders, inputs)}
    found = []
    for node in order:
        if isinstance(node, (PlaceHolderNode, OutputNode)):
            continue
        args = [env[(e.src_node, e.src_id)] for e in sorted(node.inputs, key=lambda e: e.dst_id)]
        if isinstance(node, Exp):
            found.append(args[0].float())
        env[(node, 0)] = rt._eval_node(node, args)
    assert found
    return found


def _inputs(graph, distribution, seed=7):
    return rt.make_inputs(graph, DTYPE, seed=seed, device="cpu", distribution=distribution)


def test_normal_stream_is_the_one_make_inputs_always_sampled():
    # Snapshots and measured tolerances depend on these tensors: reproduce the sampling
    # by hand and require it bit for bit.
    graph = g.mha_split_combine_graph(4, 4, 8, 16)  # has a positive (rand + 0.5) placeholder
    seed = rt.fingerprint_seed(graph)
    torch.manual_seed(seed)
    positive = rt._positive_placeholders(graph)
    assert positive
    expected = []
    for placeholder in rt.placeholder_order(graph):
        shape = tuple(int(s) for s in placeholder.get_shape(0))
        sample = torch.rand(shape) + 0.5 if placeholder in positive else torch.randn(shape)
        expected.append(sample.to(torch.float16))
    for got in (rt.make_inputs(graph, DTYPE, device="cpu"),
                rt.make_inputs(graph, DTYPE, device="cpu", distribution="normal")):
        assert len(got) == len(expected)
        assert all(torch.equal(a, b) for a, b in zip(got, expected))


@pytest.mark.parametrize("name", sorted(ATTENTION))
def test_adversarial_inputs_rewrite_only_the_score_operands(name):
    build, n_scores = ATTENTION[name]
    graph = build()
    assert len(rt_inputs.score_matmuls(graph)) == n_scores
    normal = _inputs(graph, "normal")
    large = _inputs(graph, "large_logits")
    # x100 logits, by scaling the query operand(s) only: every other input is untouched,
    # and so is the normal stream (the rewrite happens after it is sampled).
    changed = [i for i, (a, b) in enumerate(zip(normal, large)) if not torch.equal(a, b)]
    assert len(changed) == n_scores
    for i in changed:
        want = (normal[i].float() * LARGE_LOGIT_FACTOR).to(normal[i].dtype)
        assert torch.equal(large[i], want)
        assert torch.isfinite(large[i]).all()
    for before, after in zip(_scores(graph, normal), _scores(graph, large)):
        # The fixtures' fixed shift is a constant and is not scaled; differences along the
        # keys cancel it, and those are 100 times larger.
        steps = before.diff(dim=-1)
        assert torch.allclose(after.diff(dim=-1), LARGE_LOGIT_FACTOR * steps,
                              rtol=2e-2, atol=2e-2 * LARGE_LOGIT_FACTOR * float(steps.abs().max()))
        assert float(steps.abs().max()) > 0.5
    assert all(a.dtype == b.dtype and a.shape == b.shape for a, b in zip(normal, large))
    assert all(torch.equal(a, b) for a, b in zip(normal, _inputs(graph, "normal")))


@pytest.mark.parametrize("name", sorted(ATTENTION))
def test_increasing_logits_grow_along_the_keys_for_every_row(name):
    build, _ = ATTENTION[name]
    graph = build()
    normal = _inputs(graph, "normal")
    rising = _inputs(graph, "increasing")
    assert all(a.dtype == b.dtype and a.shape == b.shape for a, b in zip(normal, rising))
    assert all(torch.isfinite(t).all() for t in rising)
    for scores in _scores(graph, rising):
        steps = scores.diff(dim=-1)
        assert (steps >= 0).all(), "logits must never fall along the key axis"
        # a tile is at least 16 keys wide: the running max moves at every tile
        assert (scores[..., 16:] > scores[..., :-16]).all()
        if name == "gqa_split_partial":
            continue  # each split sees its slice of the ramp; see the test below
        span = scores[..., -1] - scores[..., 0]
        # every row ramps over at least half the span, and past ln(fp32 max) = 88.7 for a
        # row of magnitude ~0.7 (the scales of these fixtures differ by up to sqrt(2))
        assert float(span.min()) > 0.4 * INCREASING_SPAN / math.sqrt(2)
        assert float(span.max()) > 88.7


def test_increasing_follows_the_whole_key_axis_through_a_split():
    # A split partial views the KV input as (split, chunk). The ramp runs along the input's
    # whole key axis, so each split holds the next slice of it: the logits rise within a
    # split and from one split to the next, and every split has a different max.
    graph = g.gqa_split_partial_graph(1, 4, 2, 4, 8, 24, 16)
    (scores,) = _scores(graph, _inputs(graph, "increasing"))
    assert tuple(scores.shape) == (4, 4, 8, 24)  # (heads, split, query rows, chunk)
    assert (scores.diff(dim=-1) >= 0).all()
    assert (scores[:, 1:, :, 0] > scores[:, :-1, :, -1]).all()
    whole = scores[:, -1, :, -1] - scores[:, 0, :, 0]
    assert float(whole.min()) > 0.4 * INCREASING_SPAN and float(whole.max()) > 88.7


@pytest.mark.parametrize("name", sorted(ATTENTION))
def test_decreasing_is_the_increasing_ramp_the_other_way_round(name):
    build, _ = ATTENTION[name]
    graph = build()
    falling, rising = _inputs(graph, "decreasing"), _inputs(graph, "increasing")
    assert all(torch.isfinite(t).all() for t in falling)
    for down, up in zip(_scores(graph, falling), _scores(graph, rising)):
        steps = down.diff(dim=-1)
        assert (steps <= 0).all(), "logits must never rise along the key axis"
        # the row max is the first key, and a later tile (16 keys or more) is strictly below
        assert (down[..., :-16] > down[..., 16:]).all()
        assert (down[..., :1] >= down).all()
        if name == "gqa_split_partial":
            # the ramp runs along the whole key axis: it falls from one split to the next
            assert (down[:, 1:, :, 0] < down[:, :-1, :, -1]).all()
            continue
        assert torch.equal(down, up.flip(-1))
        span = down[..., 0] - down[..., -1]
        assert float(span.min()) > 0.4 * INCREASING_SPAN / math.sqrt(2)
        assert float(span.max()) > 88.7


def _whole_key_axis(name, scores):
    """Scores with the keys of the whole key axis last (a split partial holds them as
    (split, rows, chunk))."""
    if name != "gqa_split_partial":
        return scores
    return scores.permute(0, 2, 1, 3).flatten(2)


@pytest.mark.parametrize("name", sorted(ATTENTION))
def test_last_spike_puts_one_large_logit_on_the_last_key_of_every_row(name):
    build, n_scores = ATTENTION[name]
    graph = build()
    normal, spiked = _inputs(graph, "normal"), _inputs(graph, "last_spike")
    assert all(a.dtype == b.dtype and a.shape == b.shape for a, b in zip(normal, spiked))
    assert all(torch.isfinite(t).all() for t in spiked)
    for scores in _scores(graph, spiked):
        scores = _whole_key_axis(name, scores)
        last, rest = scores[..., -1], scores[..., :-1]
        # a * 128 with a >= 0.5 (the fixtures' scales differ by up to sqrt(2)): for every
        # row far above every other key, for most rows past what fp32 exp holds
        assert float(last.min()) > 0.4 * INCREASING_SPAN / math.sqrt(2)
        assert float(last.max()) > 88.7
        assert float((last[..., None] - rest).min()) > 0.25 * INCREASING_SPAN
        # the other keys score as normal keys do (around the fixtures' constant shift)
        assert float(rest.max() - rest.min()) < 0.15 * INCREASING_SPAN
    # per score matmul one query input and one key input are rewritten, nothing else
    changed = [i for i, (a, b) in enumerate(zip(normal, spiked)) if not torch.equal(a, b)]
    assert len(changed) == 2 * n_scores
    assert all(torch.equal(a, b) for a, b in zip(normal, _inputs(graph, "normal")))


@pytest.mark.parametrize("name", sorted(ATTENTION))
def test_constant_makes_every_logit_of_a_row_the_same(name):
    build, n_scores = ATTENTION[name]
    graph = build()
    normal, constant = _inputs(graph, "normal"), _inputs(graph, "constant")
    for scores in _scores(graph, constant):
        scores = _whole_key_axis(name, scores)
        assert (scores == scores[..., :1]).all(), "one logit per row, bit for bit"
        assert scores[..., 0].unique().numel() > 1, "the rows still differ from each other"
    # only the keys are rewritten: every key is the first key
    changed = [i for i, (a, b) in enumerate(zip(normal, constant)) if not torch.equal(a, b)]
    assert len(changed) == n_scores


@pytest.mark.parametrize("name", sorted(ATTENTION))
def test_offset_moves_every_logit_far_below_zero_and_keeps_them_exact(name):
    build, n_scores = ATTENTION[name]
    graph = build()
    normal, shifted = _inputs(graph, "normal"), _inputs(graph, "offset")
    assert all(torch.isfinite(t).all() for t in shifted)
    # per score matmul the query input and the key input are put on the grid, nothing else
    changed = [i for i, (a, b) in enumerate(zip(normal, shifted)) if not torch.equal(a, b)]
    assert len(changed) == 2 * n_scores
    for i in changed:
        steps = shifted[i].float() / OFFSET_GRID
        assert (steps == steps.round()).all()
    for scores in _scores(graph, shifted):
        # about -1e4 per score matmul (again up to the fixtures' scales): exp of every logit
        # is 0, so a softmax that does not subtract the max divides 0 by 0
        assert float(scores.max()) < -0.5 * OFFSET
        assert (scores.exp() == 0).all()
        spread = scores.amax(dim=-1) - scores.amin(dim=-1)
        assert 1.0 < float(spread.min()) and float(spread.max()) < 40.0  # normal-sized
    # Every operand is on the grid, so each partial sum of a contraction is exact in fp32
    # and the logits do not depend on the order they are accumulated in: summed back to
    # front they are the same bits. (The normal inputs are not: there the order shows.)
    for inputs, same in ((shifted, True), (normal, False)):
        forward = _scores(graph, inputs)
        backward = _scores(graph, [t.flip(-1) for t in inputs])
        assert all(torch.equal(a, b) for a, b in zip(forward, backward)) is same


def test_adversarial_cases_are_what_the_inputs_of_a_graph_can_be_shaped_into():
    for build, _ in ATTENTION.values():
        assert adversarial_cases(build()) == DENSE_ADVERSARIAL
    assert DENSE_ADVERSARIAL == tuple(d for d in DISTRIBUTIONS if d != "normal")
    assert adversarial_cases(g.matmul_graph(8, 8, 16)) == ()  # no softmax, no logits
    # RMSNorm-ed operands: the logits are bounded whatever the inputs are
    assert adversarial_cases(g.qknorm_graph(1, 4, 2, 8, 32, 16)) == ()

    def self_attention(x: cl.Tensor["H", "S", "D"]):
        return exact_softmax((x @ x.mT) / x.shape[-1] ** 0.5) @ x

    from cleave.frontend import trace
    from cleave.lower import to_ir

    prog = trace(self_attention, dict(H=2, S=24, D=16), cl.f16)
    graph = to_ir(prog.to_graph_structure(), dtype=DTYPE, out_dtype=DTYPE)
    # one input is both the queries and the keys: it can be scaled, not ramped
    assert adversarial_cases(graph) == ("large_logits",)
    for distribution in DENSE_ADVERSARIAL[1:]:
        with pytest.raises(SpecError, match="both a query and a key"):
            _inputs(graph, distribution)

    # every case that is missing has a reason, and a graph that gets them all has none
    assert why_not_adversarial(g.mha_bhsd_graph(1, 2, 8, 96, 16)) == ""
    assert "both a query and a key" in why_not_adversarial(graph)
    assert "neither the result of a score matmul nor an input" in why_not_adversarial(
        g.matmul_graph(8, 8, 16))
    assert "not read from an input" in why_not_adversarial(g.qknorm_graph(1, 4, 2, 8, 32, 16))


def attention_plus_gate(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                        v: cl.Tensor["H", "SKV", "D"], a: cl.Tensor["H", "SQ", "E"],
                        b: cl.Tensor["H", "D", "E"]):
    """An exact softmax next to an ``exp`` written by hand, which has a matmul of its own."""
    return exact_softmax((q @ k.mT) / q.shape[-1] ** 0.5) @ v + cl.exp((a @ b.mT) / 8.0)


def attention_plus_exp_of_the_queries(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                                      v: cl.Tensor["H", "SKV", "D"], w: cl.Tensor["H", "D", "D"]):
    """... and one whose plain ``exp`` reads the queries of the softmax."""
    return exact_softmax((q @ k.mT) / q.shape[-1] ** 0.5) @ v + cl.exp((q @ w.mT) / 8.0)


def test_a_plain_exp_next_to_an_exact_softmax_is_not_shaped():
    """The cases shape the logits of the softmax, not the argument of an ``exp`` that makes
    no claim: its reference would overflow, and a kernel with a correct softmax would be
    failed for reproducing the program's own ``inf``."""
    shapes = dict(H=2, SQ=8, SKV=96, D=16, E=16)
    graph = lowered(attention_plus_gate, shapes)
    exps = [n for n in graph.nodes if type(n) is Exp]
    assert sorted(n.stabilize is not None for n in exps) == [False, True]
    (scores,) = rt_inputs.score_matmuls(graph)  # the softmax's own matmul only
    assert scores.K == 16 and scores.N == 96
    assert adversarial_cases(graph) == DENSE_ADVERSARIAL and why_not_adversarial(graph) == ""
    names = [p.name for p in rt.placeholder_order(graph)]
    normal = dict(zip(names, _inputs(graph, "normal")))
    for distribution in DENSE_ADVERSARIAL:
        shaped = dict(zip(names, _inputs(graph, distribution)))
        same = sorted(n for n in names if torch.equal(shaped[n], normal[n]))
        assert len(same) >= 3 and len(same) < len(names), (distribution, same)  # v, a, b
        _, reference, _ = rt.build_inputs_and_reference(
            graph, DTYPE, seed=7, device="cpu", distribution=distribution)
        assert all(torch.isfinite(r).all() for r in reference), distribution
    # the inputs of the plain exp's matmul are the ones left alone, in every case
    gate = next(n for n in graph.nodes if type(n) is Exp and n.stabilize is None)
    assert rt_inputs._plain_exp_fed_by(graph, rt.placeholder_order(graph))[0] is gate

    # the same graph without the record: every exp counts again, as for any graph that
    # makes no claim (an explicit request shapes what it can)
    for n in exps:
        n.stabilize = None
    assert len(rt_inputs.score_matmuls(graph)) == 2


def test_an_input_shared_with_a_plain_exp_gets_no_adversarial_case():
    graph = lowered(attention_plus_exp_of_the_queries, dict(H=2, SQ=8, SKV=96, D=16))
    assert len(rt_inputs.score_matmuls(graph)) == 1
    assert adversarial_cases(graph) == ()
    why = why_not_adversarial(graph)
    assert "also feeds" in why and "not stabilized" in why
    # asked for explicitly it is still shaped, and the reference shows why it is not held
    # to it: the plain exp overflows
    _, reference, _ = rt.build_inputs_and_reference(
        graph, DTYPE, seed=7, device="cpu", distribution="large_logits")
    assert not all(torch.isfinite(r).all() for r in reference)


def softmax_rows(x: cl.Tensor["R", "C"]):
    return exact_softmax(x, -1)


def softmax_columns(x: cl.Tensor["C", "R"]):
    return exact_softmax(x, 0)


def softmax_of_negated_rows(x: cl.Tensor["R", "C"]):
    return exact_softmax(x * -0.5, -1)


# name -> (program, shapes, softmax axis of the input, the constant between input and exp)
LOGITS = {
    "rows": (softmax_rows, dict(R=8, C=96), 1, 1.0),
    "columns": (softmax_columns, dict(R=8, C=96), 0, 1.0),
    "negated_rows": (softmax_of_negated_rows, dict(R=8, C=96), 1, -0.5),
}


@pytest.mark.parametrize("name", sorted(LOGITS))
def test_logits_that_are_an_input_are_shaped_along_their_softmax_axis(name):
    """A standalone ``cl.softmax(x)`` has no score matmul: its logits are the input (times a
    constant). The cases write the same rows into it, along the axis the Exp's record names."""
    fn, shapes, axis, scale = LOGITS[name]
    graph = lowered(fn, shapes)  # the flat graph the pipeline validates against: annotated
    ((placeholder, found_axis, sign),) = logit_inputs(graph)
    assert (found_axis, sign) == (axis, math.copysign(1.0, scale))
    assert adversarial_cases(graph) == DENSE_ADVERSARIAL
    span = INCREASING_SPAN * abs(scale)

    def logits(distribution):
        (inputs,) = [_inputs(graph, distribution)]
        assert len(inputs) == 1 and inputs[0].dtype == torch.float16
        assert torch.isfinite(inputs[0]).all()
        (scores,) = _scores(graph, inputs)
        return scores.movedim(axis, -1)  # the softmax axis last

    base = logits("normal")
    large = logits("large_logits")  # the input is rounded to fp16 again after scaling
    assert torch.allclose(large, LARGE_LOGIT_FACTOR * base, rtol=1e-3, atol=1e-3)

    up, down = logits("increasing"), logits("decreasing")
    assert (up.diff(dim=-1) >= 0).all() and (up[..., 16:] > up[..., :-16]).all()
    assert torch.equal(down, up.flip(-1))
    rise = up[..., -1] - up[..., 0]
    assert float(rise.min()) > 0.4 * span and float(rise.max()) > 88.7

    spiked = logits("last_spike")
    last, rest = spiked[..., -1], spiked[..., :-1]
    assert torch.equal(rest, base[..., :-1]), "every key but the last is the normal one"
    assert float(last.min()) > 0.4 * span and float(last.max()) > 88.7
    assert float((last[..., None] - rest).min()) > 0.25 * span

    constant = logits("constant")
    assert (constant == constant[..., :1]).all()
    assert constant[..., 0].unique().numel() > 1

    shifted = logits("offset")
    assert float(shifted.max()) < -0.9 * LOGIT_OFFSET * abs(scale)
    assert (shifted.exp() == 0).all()
    spread = shifted.amax(dim=-1) - shifted.amin(dim=-1)
    assert 1.0 * abs(scale) < float(spread.min()) and float(spread.max()) < 40.0


def test_a_plain_exp_of_an_input_names_no_softmax_axis():
    """Without the record nothing says what the exp is summed over: no logit input, and the
    adversarial cases are an error, as for any graph without logits to shape."""
    from cleave.lower import to_ir

    prog = programs.matmul.trace(M=8, N=8, K=16)
    assert logit_inputs(to_ir(prog.to_graph_structure(), dtype=DTYPE, out_dtype=DTYPE)) == []
    graph = lowered(softmax_rows, dict(R=8, C=96))
    for node in graph.nodes:
        if type(node) is Exp:
            node.stabilize = None
    assert logit_inputs(graph) == [] and adversarial_cases(graph) == ()
    with pytest.raises(SpecError, match="no score matmul"):
        _inputs(graph, "increasing")


def test_lowered_frontend_attention_is_recognized():
    from cleave.lower import to_ir

    prog = programs.attention.trace(HQ=4, HKV=2, SQ=8, SKV=96, D=16)  # GQA, as the demos lower it
    graph = to_ir(prog.to_graph_structure(), dtype=DTYPE, out_dtype=DTYPE)
    assert len(rt_inputs.score_matmuls(graph)) == 1
    (scores,) = _scores(graph, _inputs(graph, "increasing"))
    assert (scores.diff(dim=-1) >= 0).all() and float((scores[..., -1] - scores[..., 0]).max()) > 88.7
    normal, large = _inputs(graph, "normal"), _inputs(graph, "large_logits")
    assert sum(not torch.equal(a, b) for a, b in zip(normal, large)) == 1


def test_adversarial_inputs_need_softmax_scores_and_a_known_name():
    matmul = g.matmul_graph(8, 8, 16)
    for distribution in DENSE_ADVERSARIAL:
        with pytest.raises(SpecError, match="no score matmul"):
            _inputs(matmul, distribution)
    with pytest.raises(SpecError, match="must be drawn from"):
        _inputs(matmul, "uniform")
    # A score operand that is computed (RMSNorm of q and k) cannot be shaped through an input.
    for distribution in DENSE_ADVERSARIAL:
        with pytest.raises(SpecError, match="not read from an input"):
            _inputs(g.qknorm_graph(1, 4, 2, 8, 32, 16), distribution)
    assert rt_inputs.check_distributions("normal") == ("normal",)
    assert rt_inputs.check_distributions(list(DISTRIBUTIONS)) == DISTRIBUTIONS
    with pytest.raises(SpecError, match="must be drawn from"):
        rt_inputs.check_distributions(())


def test_adversarial_inputs_are_switched_on_per_operator():
    """The generators exist; switching one on is a deliberate, per-operator change.

    Nothing asks for them by default. A dynamic operator declares that its softmax is exact
    (``exact_softmax``), and from then on every validation of it includes its adversarial
    workloads; one operator does so far. A dense program says so by what it records: the
    request for the exact softmax (``Program._stable_exp``, what ``cl.softmax`` records),
    and the kernels of such a program are held to every adversarial case its inputs can be
    shaped into, whatever was asked for. A fixed-shift program, or an ``exp / sum`` written
    by hand, is validated on what was asked.
    """
    assert {name for name, op in OPERATORS.items() if op.exact_softmax} == {
        "gqa_paged_decode_h32_kv8_d128_ps1"}
    assert EvalConfig().distributions == ("normal",)
    assert EvalConfig(distributions="increasing").distributions == ("increasing",)
    assert inspect.signature(rt.make_inputs).parameters["distribution"].default == "normal"
    assert inspect.signature(keyed_workloads).parameters["adversarial"].default is False
    from cleave.pipeline.dynamic import compile_dynamic, resolve_workloads

    assert inspect.signature(compile_dynamic).parameters["adversarial"].default is False
    assert inspect.signature(resolve_workloads).parameters["adversarial"].default is False
    assert cl.workloads.paged_decode().distribution == "normal"

    from cleave.frontend import trace
    from cleave.lower import to_ir
    from cleave.pipeline.build import _reference_fn

    def exact(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
              v: cl.Tensor["H", "SKV", "D"]):
        return exact_softmax((q @ k.mT) / q.shape[-1] ** 0.5) @ v

    def by_hand(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                v: cl.Tensor["H", "SKV", "D"]):
        e = cl.exp((q @ k.mT) / q.shape[-1] ** 0.5)
        return (e / cl.sum(e, -1, keepdim=True)) @ v

    seen = []

    def publish(cache_dir, graph, dtype, distributions):
        seen.append(distributions)
        return "published"

    def held_to(program, asked, graph=None):
        if graph is None:
            graph = to_ir(program.to_graph_structure(), dtype=DTYPE, out_dtype=DTYPE)
        assert _reference_fn(asked, program, publish)("dir", graph, DTYPE) == "published"
        return seen[-1]

    shapes = dict(H=2, SQ=8, SKV=96, D=16)
    fixed = programs.attention.trace(HQ=2, HKV=2, SQ=8, SKV=96, D=16)
    assert trace(exact, shapes, cl.f16).stabilize_requests and not fixed.stabilize_requests
    assert held_to(trace(exact, shapes, cl.f16), ("normal",)) == ("normal",) + DENSE_ADVERSARIAL
    assert held_to(trace(exact, shapes, cl.f16), ("increasing", "normal")) == (
        "increasing", "normal", "large_logits", "decreasing", "last_spike", "constant", "offset")
    assert held_to(fixed, ("normal",)) == ("normal",)
    assert held_to(trace(by_hand, shapes, cl.f16), ("normal",)) == ("normal",)
    assert held_to(fixed, ("normal", "large_logits")) == ("normal", "large_logits")

    # a softmax of an input: the logits are the input, and the lowered graph says along
    # which axis (the Exp's record), so it is held to every case as well
    alone = trace(softmax_rows, dict(R=4, C=32), cl.f16)
    assert held_to(alone, ("normal",), lowered(softmax_rows, dict(R=4, C=32))) == (
        "normal",) + DENSE_ADVERSARIAL

    def exp_of_a_sum(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]):
        return exact_softmax(x + y)

    # logits that are neither a score matmul nor an input: nothing to shape, nothing added
    summed = trace(exp_of_a_sum, dict(R=4, C=32), cl.f16)
    assert held_to(summed, ("normal",), lowered(exp_of_a_sum, dict(R=4, C=32))) == ("normal",)


def _stubbed_build(monkeypatch, program, asked=("normal",), tolerance=5e-3):
    """``pipeline.build`` of ``program``'s identity graph with the workers and the GPU
    stubbed out: every candidate "passes". Returns ``(BuildSet, published)``, ``published``
    being the input cases ``build`` had the reference of each flat graph made for."""
    from cleave import pipeline as P
    from cleave.pipeline.superopt import GraphSet
    from cleave.runtime import evaluate, validate

    published = []

    def publish(reference_cache_dir, graph, dtype, distributions=("normal",)):
        published.append(tuple(distributions))
        return [-1], 0

    def run_candidates(candidates, dtype, *, reference_fn=None, **kw):
        seen = set()
        for cand in candidates:
            if id(cand.flat_graph) not in seen:
                seen.add(id(cand.flat_graph))
                reference_fn("dir", cand.flat_graph, dtype)
        return [evaluate.CandidateResult(cand, "ok") for cand in candidates]

    monkeypatch.setattr(validate, "publish_reference", publish)
    monkeypatch.setattr(evaluate, "run_candidates", run_candidates)
    plans = P.schedule(GraphSet.identity(program), P.ScheduleConfig(topk=2))
    assert len(plans) >= 1
    built = P.build(plans, config=EvalConfig(tolerance=cl.Tolerance(rtol=tolerance),
                                             distributions=asked))
    return built, published


def test_build_holds_an_exact_program_to_its_adversarial_cases(monkeypatch):
    """The switch where it takes effect: ``build()`` itself has the references made for the
    adversarial cases, and says on every result what its kernel was held to."""
    from cleave import pipeline as P
    from cleave.frontend import trace

    built, published = _stubbed_build(monkeypatch, trace(softmax_rows, dict(R=64, C=256), cl.f16))
    assert published == [("normal",) + DENSE_ADVERSARIAL]
    assert len(built) >= 1 and all(r.cases == ("normal",) + DENSE_ADVERSARIAL for r in built)
    assert not any(r.cases_note for r in built)
    report = P.evaluate(built)
    assert report.to_dict()["validation"]["inputs"] == [
        {"cases": ["normal", *DENSE_ADVERSARIAL], "kernels": len(built), "note": ""}]
    validate = dict((stage, text) for stage, text, _ in report.stage_lines())["validate"]
    assert validate.endswith("inputs: normal, " + ", ".join(DENSE_ADVERSARIAL))

    # what was asked for comes first; the cases are added behind it
    built, published = _stubbed_build(monkeypatch, trace(softmax_rows, dict(R=64, C=256), cl.f16),
                                      asked=("increasing", "normal"))
    assert published == [("increasing", "normal", "large_logits", "decreasing", "last_spike",
                          "constant", "offset")]

    # a fixed-shift program makes no claim: what was asked, and nothing said about it
    fixed = programs.attention.trace(HQ=32, HKV=32, SQ=64, SKV=256, D=64)
    built, published = _stubbed_build(monkeypatch, fixed)
    assert set(published) == {("normal",)} and all(r.cases == ("normal",) for r in built)
    report = P.evaluate(built)
    assert "inputs:" not in dict((s, t) for s, t, _ in report.stage_lines())["validate"]
    assert report.to_dict()["validation"]["inputs"] == [
        {"cases": ["normal"], "kernels": len(built), "note": ""}]


def test_build_says_when_an_exact_program_is_held_to_fewer_cases(monkeypatch):
    """A graph of an exact program whose logits cannot be shaped is validated on the normal
    inputs alone. That is a weaker validation than the program's claim, so the result and
    the report say so, with the reason."""
    from cleave import pipeline as P
    from cleave.frontend import trace

    def exp_of_a_sum(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]):
        return exact_softmax(x + y)

    built, published = _stubbed_build(monkeypatch, trace(exp_of_a_sum, dict(R=64, C=256), cl.f16))
    assert published == [("normal",)]
    assert all(r.cases == ("normal",) and "neither the result of a score matmul nor an input"
               in r.cases_note for r in built)
    report = P.evaluate(built)
    validate = dict((stage, text) for stage, text, _ in report.stage_lines())["validate"]
    assert "inputs: normal [exact softmax not held to every adversarial case: " in validate
    (entry,) = report.to_dict()["validation"]["inputs"]
    assert entry["cases"] == ["normal"] and entry["note"] == built[0].cases_note


# ----------------------------------------------------------------------------- dynamic
# logits x100 through the queries and through the scale; increasing along the keys; one
# dominant key at the end of every request
ADVERSARIAL = ("large_logits", "large_scale", "increasing", "dominant_last")


def test_dynamic_distributions_extend_the_shared_ones():
    assert SHARED_DISTRIBUTIONS == ("normal", "large_logits", "increasing")
    assert dyn.DISTRIBUTIONS == SHARED_DISTRIBUTIONS + ("large_scale", "dominant_last")
    assert DISTRIBUTIONS == SHARED_DISTRIBUTIONS + ("decreasing", "last_spike", "constant", "offset")
    assert set(ADVERSARIAL) == set(dyn.DISTRIBUTIONS) - {"normal"}
    # the dense programs know nothing of the two the dynamic operators add ...
    for name in ("large_scale", "dominant_last"):
        with pytest.raises(SpecError, match="must be drawn from"):
            rt_inputs.check_distributions(name)
    # ... and the dynamic operators nothing of the four the dense programs add
    for name in ("decreasing", "last_spike", "constant", "offset"):
        with pytest.raises(SpecError, match="must be one of"):
            dyn.apply_distribution("gqa_paged_decode", [], name)
    with pytest.raises(SpecError, match="must be one of"):
        dyn.apply_distribution("gqa_paged_decode", [], "uniform")


def test_the_captured_cases_are_followed_by_the_adversarial_ones_only_on_request():
    """``workloads.cases`` and ``keyed_workloads`` yield the adversarial table after the
    captured workloads, and only when asked: timing never sees it."""
    import importlib

    # the module: ``cleave.dynamic.workloads`` the attribute is the registry's function
    wl = importlib.import_module("cleave.dynamic.workloads")
    name = "gqa_paged_decode_h32_kv8_d128_ps1"
    op = OPERATORS[name]
    assert inspect.signature(wl.cases).parameters["adversarial"].default is False
    seen = []

    def fake(operator, axes, dims, page_size=1, scale=None, distribution="normal"):
        seen.append((distribution, axes))
        return axes, distribution

    original, wl.case = wl.case, fake
    try:
        plain = list(wl.cases(name, op.dims))
        assert seen == [("normal", axes) for axes in WORKLOADS[name]] and len(plain) == len(seen)
        del seen[:]
        both = list(wl.cases(name, op.dims, adversarial=True))
    finally:
        wl.case = original
    captured = [("normal", axes) for axes in WORKLOADS[name]]
    assert seen == captured + list(ADVERSARIAL_WORKLOADS[name])
    assert len(both) == len(WORKLOADS[name]) + 12
    # the registry's keyed factory is the same list, keyed: captured first, never mixed
    keys = [key for key, *_ in _keys(op, adversarial=True)]
    assert len(keys) == len(captured) + 12
    assert not any("@" in key for key in keys[: len(captured)])
    assert [key.rpartition("@")[2] for key in keys[len(captured):]] == [
        d for d, _ in ADVERSARIAL_WORKLOADS[name]]
    assert not any("@" in key for key, *_ in _keys(op, adversarial=False))


def _keys(op, adversarial):
    """``op.keyed_workloads`` with the tensors stubbed out: the keys alone, on the CPU."""
    import dataclasses

    stub = dataclasses.replace(op, reference=lambda axes, distribution="normal": ((), ()))
    original = OPERATORS[op.name]
    OPERATORS[op.name] = stub
    try:
        return list(keyed_workloads(op.name, adversarial=adversarial)())
    finally:
        OPERATORS[op.name] = original


def test_adversarial_workloads_are_a_table_of_their_own():
    assert set(ADVERSARIAL_WORKLOADS) == set(WORKLOADS) == set(OPERATORS)
    for name, cases in ADVERSARIAL_WORKLOADS.items():
        captured = WORKLOADS[name]
        assert {d for d, _ in cases} == set(ADVERSARIAL)
        assert len(cases) == len(set(cases))
        for distribution in ADVERSARIAL:
            axes = [a for d, a in cases if d == distribution]
            # the smallest, a middle and the largest captured shape (long KV included)
            assert axes[0] == captured[0] and axes[-1] == captured[-1] and len(axes) == 3
            assert set(axes) <= set(captured)
        assert OPERATORS[name].adversarial == cases
        assert not any(isinstance(w[0], str) for w in captured)  # WORKLOADS itself is unchanged


def _indptr(lengths):
    out = [0]
    for n in lengths:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32)


def _bf16(*shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen).to(torch.bfloat16)


def _family_inputs(family):
    """Hand-built CPU inputs in the layout of ``cleave.dynamic.reference.*_inputs``."""
    kv_lens, q_lens = [5, 0, 37], [3, 0, 9]
    heads, kv_heads, dim, pages = 4, 2, 16, sum(kv_lens) + 6
    table = torch.randperm(pages, generator=torch.Generator().manual_seed(3))[: sum(kv_lens)]
    table = table.to(torch.int32)
    if family == "gqa_paged_decode":
        return [_bf16(3, heads, dim, seed=0), _bf16(pages, 1, kv_heads, dim, seed=1),
                _bf16(pages, 1, kv_heads, dim, seed=2), _indptr(kv_lens), table, dim ** -0.5]
    if family == "gqa_paged_prefill":
        return [_bf16(sum(q_lens), heads, dim, seed=0), _bf16(pages, 1, kv_heads, dim, seed=1),
                _bf16(pages, 1, kv_heads, dim, seed=2), _indptr(q_lens), _indptr(kv_lens), table,
                dim ** -0.5]
    if family == "gqa_ragged_prefill":
        return [_bf16(sum(q_lens), heads, dim, seed=0), _bf16(sum(kv_lens), kv_heads, dim, seed=1),
                _bf16(sum(kv_lens), kv_heads, dim, seed=2), _indptr(q_lens), _indptr(kv_lens),
                dim ** -0.5]
    ckv, kpe = 32, 8
    queries = len(kv_lens) if family == "mla_paged_decode" else sum(q_lens)
    head = [_bf16(queries, heads, ckv, seed=0), _bf16(queries, heads, kpe, seed=1),
            _bf16(pages, 1, ckv, seed=2), _bf16(pages, 1, kpe, seed=3)]
    if family == "mla_paged_prefill":
        head.append(_indptr(q_lens))
    return head + [_indptr(kv_lens), table, MLA_SM_SCALE]


def _request_logits(family, inputs, request):
    """(queries, heads, kv_len) logits of one request, keys in page-table order."""
    terms, indptr_at, indices_at = dyn._SCORE_INPUTS[family]
    lo, hi = (int(v) for v in inputs[indptr_at][request : request + 2])
    rows = inputs[indices_at][lo:hi].long() if indices_at is not None else torch.arange(lo, hi)
    total = 0
    for query_at, key_at in terms:
        keys = inputs[key_at].float()
        keys = keys[rows, 0] if indices_at is not None else keys[rows]
        query = inputs[query_at].float()
        if keys.dim() == 3:  # (kv, kv_heads, dim): repeat for the query heads of each kv head
            keys = keys.repeat_interleave(query.shape[1] // keys.shape[1], dim=1)
            total = total + torch.einsum("qhd,khd->qhk", query, keys)
        else:
            total = total + torch.einsum("qhd,kd->qhk", query, keys)
    return total * float(inputs[-1])


FAMILIES = sorted(dyn._SCORE_INPUTS)


def test_every_registry_operator_has_an_adversarial_layout():
    for name in OPERATORS:
        assert sum(name.startswith(f) for f in FAMILIES) == 1, name
    with pytest.raises(SpecError, match="no adversarial inputs are defined"):
        dyn.apply_distribution("some_other_operator", [], "increasing")


@pytest.mark.parametrize("family", FAMILIES)
def test_dynamic_large_logits_scales_the_queries_only(family):
    inputs = _family_inputs(family)
    terms = dyn._SCORE_INPUTS[family][0]
    assert dyn.apply_distribution(family, inputs, "normal") is inputs
    large = dyn.apply_distribution(family, inputs, "large_logits")
    queries = {q for q, _ in terms}
    for i, (before, after) in enumerate(zip(inputs, large)):
        if i in queries:
            assert torch.equal(after, (before.float() * LARGE_LOGIT_FACTOR).to(torch.bfloat16))
            assert after.dtype == torch.bfloat16
        elif torch.is_tensor(before):
            assert after is before
        else:
            assert after == before
    for request in (0, 2):
        before = _request_logits(family, inputs, request)
        # bf16 rounds the scaled queries again: within a few ulp of the largest logit
        assert torch.allclose(_request_logits(family, large, request), LARGE_LOGIT_FACTOR * before,
                              rtol=0.0, atol=2e-2 * LARGE_LOGIT_FACTOR * float(before.abs().max()))


@pytest.mark.parametrize("family", FAMILIES)
def test_dynamic_large_scale_scales_the_softmax_scale_only(family):
    inputs = _family_inputs(family)
    large = dyn.apply_distribution(family, inputs, "large_scale")
    assert all(a is b for a, b in zip(large[:-1], inputs[:-1])), "no tensor is touched"
    assert large[-1] == pytest.approx(LARGE_LOGIT_FACTOR * inputs[-1], rel=1e-12)
    assert isinstance(large[-1], float) and inputs[-1] == _family_inputs(family)[-1]
    for request in (0, 2):
        # exactly the logits of large_logits, without rounding the queries a second time
        assert torch.allclose(_request_logits(family, large, request),
                              LARGE_LOGIT_FACTOR * _request_logits(family, inputs, request),
                              rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("family", FAMILIES)
def test_dynamic_dominant_last_puts_one_large_logit_at_the_end_of_each_request(family):
    inputs = _family_inputs(family)
    saved = [t.clone() if torch.is_tensor(t) else t for t in inputs]
    spiked = dyn.apply_distribution(family, inputs, "dominant_last")
    assert all(torch.equal(a, b) if torch.is_tensor(a) else a == b for a, b in zip(inputs, saved))
    terms = dyn._SCORE_INPUTS[family][0]
    replaced = {i for pair in terms for i in pair}
    for i, (before, after) in enumerate(zip(inputs, spiked)):
        if i not in replaced:
            assert after is before
        else:
            assert after.shape == before.shape and after.dtype == before.dtype
            assert torch.isfinite(after).all()
    for request, kv_len in ((0, 5), (2, 37)):
        logits = _request_logits(family, spiked, request)
        assert logits.shape[-1] == kv_len
        last, rest = logits[..., -1], logits[..., :-1]
        # a * 128 with a >= 0.5: past what fp32 exp holds for most rows, for every row far
        # above every other key of the request
        assert float(last.min()) > 0.48 * INCREASING_SPAN
        assert float(last.max()) > 88.7
        assert float((last[..., None] - rest).min()) > 0.4 * INCREASING_SPAN
        assert float(rest.abs().max()) < 0.1 * INCREASING_SPAN
    # every key but each request's last is the normal one
    for _, key_at in terms:
        changed = (spiked[key_at].float() != inputs[key_at].float()).flatten(1).any(dim=1)
        assert int(changed.sum()) == 2, "one key row per non-empty request"


def test_dynamic_dominant_last_is_the_last_live_token_of_a_partly_filled_page():
    """A slab's last page holds ``kv_last_page_len`` tokens: the dominant key is the last of
    those, not the last slot of the page."""
    page, kv_heads, heads, dim = 4, 2, 4, 16
    pages_per_request, last_len = [2, 0, 3], [3, 0, 1]
    pool = sum(pages_per_request) + 2
    table = torch.randperm(pool, generator=torch.Generator().manual_seed(5))[:5].to(torch.int32)
    inputs = [_bf16(3, heads, dim, seed=0), _bf16(pool, page, kv_heads, dim, seed=1),
              _bf16(pool, page, kv_heads, dim, seed=2), _indptr(pages_per_request), table,
              torch.tensor(last_len, dtype=torch.int32), dim ** -0.5]
    spiked = dyn.apply_distribution("gqa_paged_decode", inputs, "dominant_last")
    changed = (spiked[1].float() != inputs[1].float()).flatten(2).any(dim=2)  # (page, slot)
    want = torch.zeros_like(changed)
    want[int(table[1]), 3 - 1] = True  # request 0: second page, third token
    want[int(table[4]), 1 - 1] = True  # request 2: third page, first token
    assert torch.equal(changed, want)


@pytest.mark.parametrize("family", FAMILIES)
def test_dynamic_increasing_ramps_each_request_in_page_table_order(family):
    inputs = _family_inputs(family)
    saved = [t.clone() if torch.is_tensor(t) else t for t in inputs]
    rising = dyn.apply_distribution(family, inputs, "increasing")
    # the normal inputs are not modified in place, and only queries and keys are replaced
    assert all(torch.equal(a, b) if torch.is_tensor(a) else a == b for a, b in zip(inputs, saved))
    terms = dyn._SCORE_INPUTS[family][0]
    replaced = {i for pair in terms for i in pair}
    for i, (before, after) in enumerate(zip(inputs, rising)):
        if i not in replaced:
            assert after is before
        else:
            assert after.shape == before.shape and after.dtype == before.dtype
            assert torch.isfinite(after).all()
    for request, kv_len in ((0, 5), (2, 37)):
        logits = _request_logits(family, rising, request)
        assert logits.shape[-1] == kv_len
        assert (logits.diff(dim=-1) > 0).all(), "strictly increasing over the request's keys"
        span = logits[..., -1] - logits[..., 0]
        # bf16 keeps 8 bits: the span is a * 128 to within a percent or two, a >= 0.5
        assert float(span.min()) > 0.48 * INCREASING_SPAN
        assert torch.allclose(logits[..., 0], torch.zeros(()), atol=1e-3)
    # the empty request owns no keys; nothing of it is touched or needed
    lo, hi = (int(v) for v in rising[dyn._SCORE_INPUTS[family][1]][1:3])
    assert lo == hi
