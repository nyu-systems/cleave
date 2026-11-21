"""``cl.softmax(x)`` next to another ``exp`` of the same tensor (CPU; the search runs).

The search computes a value once, so the graphs it returns have one exp op where such a
program wrote two: the exp of the softmax and the other one are the same value to it. The
request is a record of the program, not a node of the searched graph, and the merged exp
does not cancel. Two things went wrong with that:

* the program had no kernel: every searched graph was refused ("softmax cannot be
  stabilized"), although its own graph, with the two exps apart, is served;
* next to an ``exp / sum`` written by hand, which does cancel, the graph was accepted with
  the requested softmax on a plain exp: the hand-written one filled the count of records,
  validation shaped only its logits, and the build reported every adversarial case.

Lowering now computes a merged exp again for the readers that need their own
(``cleave.lower.unmerge``), and ``annotate`` refuses a graph that leaves an exp plain that
may be a requested one.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import LoweringRejected
from cleave.frontend import trace
from cleave.frontend.program import Op
from cleave.lower.process import to_bundles
from cleave.lower.scale import ScaleError, annotate, exps, written_exps
from cleave.lower.unmerge import exp_ops, unmerged
from cleave.pipeline.plans import bind_inputs, lower
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.inputs import adversarial_cases, logit_inputs, why_not_adversarial
from cleave.runtime.reference import execute, placeholder_order
from cleave.runtime.validate import Tolerance, compare_outputs
from tests._lowering import run_lowered


def dual_softmax(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1) * cl.softmax(x, 0)


def softmax_plus_exp(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1) + cl.exp(x)


def softmax_times_exp(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1) * cl.exp(x)


def softmax_next_to_its_own_exp(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]):
    e = cl.exp(y)  # exp / sum written out by hand: no request
    return cl.softmax(x) + cl.exp(x), e / cl.sum(e, -1, keepdim=True)


SEARCH = P.SuperoptConfig(cache=False)


@pytest.fixture(autouse=True)
def _no_measurement(monkeypatch):
    """These tests compare references and stand-ins, not kernels: nothing they compare
    belongs in a measurement of kernel errors (``CLEAVE_VALIDATION_LOG``)."""
    monkeypatch.delenv("CLEAVE_VALIDATION_LOG", raising=False)
# name -> (program, shapes, the records of its exps, sorted: softmax axes, "plain")
MERGED = {
    "dual_softmax": (dual_softmax, dict(R=16, C=32), [(-1,), (-2,)]),
    "softmax_plus_exp": (softmax_plus_exp, dict(R=64, C=256), [(-1,), "plain"]),
    "softmax_times_exp": (softmax_times_exp, dict(R=64, C=256), [(-1,), "plain"]),
}


def _records(flat):
    return sorted((e.stabilize.axes if e.stabilize is not None else "plain" for e in exps(flat)),
                  key=str)


BOUND = Tolerance(rtol=5e-3)  # of the row max; an all-zero row has to be zeros


def _close(got, want, what=""):
    compare_outputs([got], [want], BOUND, what=what)


@pytest.mark.parametrize("name", sorted(MERGED))
def test_search_a_softmax_next_to_an_exp_of_its_logits_has_plans(name):
    """Every searched graph has one exp where the program wrote two. Each lowers with the
    exp computed twice, the softmax's copy marked, and computes the program; the program
    schedules. (It had 0 plans: every graph was refused.)"""
    fn, shapes, records = MERGED[name]
    prog = trace(fn, shapes, cl.f16)
    graphs = P.superoptimize(prog, SEARCH)
    assert len(graphs) >= 1 and written_exps(prog) == 2
    assert all(exp_ops(g.structure) == 1 for g in graphs)
    inputs = [torch.randn(*spec.shape).to(spec.dtype.torch) for spec in prog.inputs]
    (want,) = prog.reference(*inputs)
    for graph in graphs:
        (flat,) = lower(graph)
        assert _records(flat) == records, graph.summary()
        assert exp_ops(graph.structure) == 1  # the searched structure is left as it is
        (again,) = lower(graph)  # ... and lowering it again gives the same graph
        assert [n.name for n in again.nodes] == [n.name for n in flat.nodes]
        (got,) = run_lowered(flat, prog, inputs)
        _close(got, want, graph.summary())
    plans = P.schedule(graphs, P.ScheduleConfig(topk=2))
    assert len(plans) > 0, plans.summary()
    for plan in plans:
        assert _records(plan.tile.flat_graph) == records
        (got,) = run_lowered(plan.tile.flat_graph, prog, inputs)
        _close(got, want, plan.label)


def test_search_two_softmaxes_of_one_input_are_exact_at_large_logits():
    """Both softmaxes of the dual softmax are stabilized, each over its own axis: at x100
    logits the lowered graph is finite and is the program. The merged exp alone would need
    its factor constant along both axes."""
    fn, shapes, _ = MERGED["dual_softmax"]
    prog = trace(fn, shapes, cl.f16)
    torch.manual_seed(0)
    x = (torch.randn(16, 32) * 100).to(torch.float16)
    (want,) = prog.reference(x)
    assert torch.isfinite(want).all() and float(want.abs().max()) > 0
    for graph in P.superoptimize(prog, SEARCH):
        (flat,) = lower(graph)
        (got,) = run_lowered(flat, prog, [x])
        assert torch.isfinite(got).all()
        _close(got, want, graph.summary())
        assert adversarial_cases(flat) and not why_not_adversarial(flat)


def test_search_a_handwritten_softmax_does_not_stand_in_for_the_requested_one():
    """``cl.softmax(x) + cl.exp(x)`` next to ``exp(y) / sum(exp(y))`` written by hand. The
    searched graph has one exp of ``x``; the exp of ``y`` cancels and used to fill the count
    of records, so the graph was accepted with the requested softmax on a plain exp, and
    logits far below zero (where only the exact softmax is finite) gave NaN while the build
    reported every adversarial case. Every graph the search returns is now refused or
    computes the program there, with the requested logits among the shaped inputs or the
    reason why no case is run."""
    prog = trace(softmax_next_to_its_own_exp, dict(R=16, C=24), cl.f32)
    graphs = P.superoptimize(prog, SEARCH)
    assert len(graphs) >= 1
    torch.manual_seed(0)
    x, y = torch.randn(16, 24) - 512.0, torch.randn(16, 24)
    want = prog.reference(x, y)
    assert all(torch.isfinite(w).all() for w in want)
    accepted = 0
    for graph in graphs:
        try:
            (flat,) = lower(graph)
        except LoweringRejected as error:
            assert "softmax cannot be stabilized" in str(error)
            continue
        accepted += 1
        got = run_lowered(flat, prog, [x, y])
        assert all(torch.isfinite(t).all() for t in got), graph.summary()
        for g, w in zip(got, want):
            torch.testing.assert_close(g.float(), w.float(), rtol=1e-4, atol=1e-6)
        requested = dict(zip(bind_inputs(flat, prog), ("x", "y")))
        shaped = {requested[p.name] for p, _, _ in logit_inputs(flat)}
        assert "x" in shaped or (adversarial_cases(flat) == () and why_not_adversarial(flat))
    assert accepted >= 1  # the program compiles: the exp of x is computed twice

    # unit level: the merged graph as the search returns it, with the program's records
    class Records:
        stabilize_requests = ((0, 1),)
        ops = [Op("exp", (0,), (("stabilize", 1),), 2), Op("exp", (0,), (), 5),
               Op("exp", (1,), (), 6)]

    for graph in graphs:
        (bundle,) = to_bundles([graph.structure], symbol_map=graph.symbol_map,
                               split_factors=None, dtype=graph.dtype, out_dtype=graph.dtype)
        (merged,) = list(bundle)
        assert len(exps(merged)) == 2
        with pytest.raises(ScaleError, match="wrote 3 different exps and this graph has 2"):
            annotate(merged, Records)
        assert [e.stabilize for e in exps(merged)] == [None, None]  # nothing is half-marked
        # ... and with the program itself: the exp that reads x is the requested one
        with pytest.raises(ScaleError):
            annotate(merged, prog)


def test_a_plain_exp_of_the_requested_logits_is_refused_whatever_else_cancels():
    """The tie that does not count: an Exp reads the input a softmax was requested of, no
    reader of that input carries a record, and another exp of the graph cancels. Here the
    program's own graph is rewired by hand so that its two exps of ``x`` are one, while the
    program still says three exps were written and the graph still has three (a second,
    unread copy): counting exps finds nothing wrong."""
    prog = trace(softmax_next_to_its_own_exp, dict(R=16, C=24), cl.f32)
    structure = prog.to_graph_structure()
    x = structure[0]["output_tensors"][0]["guid"]
    first, second = [i for i, op in enumerate(structure) if op["op_type"] == "kn_exp_op"
                     and op["input_tensors"][0]["guid"] == x]
    softmax_exp = structure[first]["output_tensors"][0]["guid"]
    plain_exp = structure[second]["output_tensors"][0]["guid"]
    for op in structure:  # the softmax reads the plain exp's value; the other is read by one
        for tensor in op["input_tensors"]:
            if tensor["guid"] == softmax_exp:
                tensor["guid"] = plain_exp
    structure.insert(len(structure) - 2, {
        "op_type": "kn_output_op", "output_tensors": [],
        "input_tensors": [dict(structure[first]["output_tensors"][0])]})
    (bundle,) = to_bundles([structure], symbol_map=None, split_factors=None,
                           dtype="float32", out_dtype="float32")
    (graph,) = list(bundle)
    assert len(exps(graph)) == written_exps(prog) == 3
    with pytest.raises(ScaleError, match=r"exp_\d+ reads 'x', the logits of a softmax the "
                                         "program requested, and stays a plain exp"):
        annotate(graph, prog)


def test_unmerged_structures_compute_the_exp_again_for_a_part_of_its_readers():
    prog = trace(softmax_plus_exp, dict(R=4, C=8), cl.f32)
    (graph,) = GraphSet.identity(prog)
    structure = [op for op in graph.structure]
    assert unmerged(structure, 0) == [] and unmerged(structure, -1) == []
    # the program's own graph: the first exp is read by its sum and its div
    twins = unmerged(structure, 1)
    assert len(twins) == 1 and [exp_ops(t) for t in twins] == [3]
    (twin,) = twins
    exp_at = [i for i, op in enumerate(twin) if op["op_type"] == "kn_exp_op"]
    assert exp_at[1] == exp_at[0] + 1  # the copy stands right behind the exp it copies
    outs = [twin[i]["output_tensors"][0]["guid"] for i in exp_at[:2]]
    assert len(set(outs)) == 2 and twin[exp_at[0]]["input_tensors"] == twin[exp_at[1]]["input_tensors"]
    read = [t["guid"] for op in twin for t in op["input_tensors"]]
    assert read.count(outs[0]) == 1 and read.count(outs[1]) == 1
    assert structure == prog.to_graph_structure()  # the structure handed in is not edited
    x = torch.randn(4, 8)
    for candidate in twins:
        (bundle,) = to_bundles([candidate], symbol_map=None, split_factors=None,
                               dtype="float32", out_dtype="float32")
        (ir,) = list(bundle)
        (out,) = execute(ir, [x for _ in placeholder_order(ir)])
        torch.testing.assert_close(out, prog.reference(x)[0])


def test_a_program_with_one_exp_record_is_not_unmerged():
    """``softmax(x) * e`` with ``e`` the softmax's own exp is one record: nothing was
    merged, and the refusal stands as it is pinned in ``tests/lower``."""
    from tests._softmax import stable_exp

    def merged_with_exp(x: cl.Tensor["R", "C"]):
        e = stable_exp(x, -1)
        return (e / cl.sum(e, -1, keepdim=True)) * e

    prog = trace(merged_with_exp, dict(R=16, C=24), cl.f32)
    assert written_exps(prog) == 1
    (graph,) = GraphSet.identity(prog)
    with pytest.raises(LoweringRejected) as error:
        lower(graph)
    assert "the search merged" not in str(error.value)
