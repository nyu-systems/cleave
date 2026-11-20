"""``pipeline.plans.lower`` marks the softmax exps of a program that requested them (CPU).

The mark is plain data on the freshly lowered flat graph. The search space and the tiling
candidates are those of the shift-free graph; a graph that cannot honour the request is
refused with the analysis' reason, and a split graph is refused until the exact combine exists.
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import LoweringRejected
from cleave.frontend import trace
from cleave.ir.ops import Exp, Stabilize
from cleave.pipeline.plans import SPLIT_SOFTMAX, describe_tile, lower
from cleave.pipeline.superopt import GraphSet, SearchedGraph
from cleave.runtime.reference import fingerprint_seed
from tests._softmax import exact_softmax, stable_exp
from tests.pipeline import _pipeline as PL


def flash(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], v: cl.Tensor["H", "SKV", "D"]):
    e = stable_exp((q @ k.mT) / q.shape[-1] ** 0.5, -1)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def flash_twin(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], v: cl.Tensor["H", "SKV", "D"]):
    e = cl.exp((q @ k.mT) / q.shape[-1] ** 0.5)  # the same records without the request
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def traced(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], v: cl.Tensor["H", "SKV", "D"]):
    return exact_softmax((q @ k.mT) / q.shape[-1] ** 0.5, -1) @ v


def softmax(x: cl.Tensor["R", "C"]):
    return exact_softmax(x, -1)


def cube(x: cl.Tensor["R", "C"]):
    e = stable_exp(x, -1)
    return (e * e) / cl.sum(e, -1, keepdim=True)


SHAPES = dict(H=32, SQ=64, SKV=4096, D=64)


def _exps(ir):
    return [n for n in ir.nodes if type(n) is Exp]


def _lowered(fn, sizes, dtype=cl.f16):
    prog = trace(fn, sizes, dtype)
    (graph,) = GraphSet.identity(prog)
    (ir,) = lower(graph)
    return prog, ir


def test_identity_graph_is_annotated():
    _, ir = _lowered(flash, SHAPES)
    (exp,) = _exps(ir)
    assert isinstance(exp.stabilize, Stabilize)
    assert exp.stabilize.axes == (-1,) and exp.stabilize.rolling == "" and exp.stabilize.export is None
    assert [type(next(n for n in ir.nodes if n.name == name)).__name__
            for name, _ in exp.stabilize.scaled] == ["Matmul", "ReduceSum"]


def test_flat_graph_stays_shift_free():
    """Same nodes, names, shapes and dtypes as the plain twin: no max ahead of the sums, and
    the validation inputs (seeded by the flat graph) are the twin's."""
    _, ir = _lowered(flash, SHAPES)
    _, twin = _lowered(flash_twin, SHAPES)

    def nodes(graph):
        return [(type(n).__name__, n.name, [tuple(int(s) for s in n.get_shape(i))
                                           for i in range(max(1, n.num_outputs()))])
                for n in graph.nodes]

    assert nodes(ir) == nodes(twin)
    assert not any("Max" in kind for kind, *_ in nodes(ir))
    assert [e.stabilize for e in _exps(twin)] == [None]
    assert fingerprint_seed(ir) == fingerprint_seed(twin)


@pytest.mark.parametrize("fn, sizes", [(flash, SHAPES), (flash, dict(H=32, SQ=64, SKV=256, D=64)),
                                       (traced, dict(H=32, SQ=64, SKV=256, D=64)),
                                       (softmax, dict(R=64, C=256))],
                         ids=["flash_4096", "flash_256", "traced_256", "softmax_256"])
def test_tiling_candidates_are_those_of_the_unmarked_graph(fn, sizes):
    """The record is invisible to ``search_tiling``: same candidates, in the same order."""
    from cleave.schedule import search_tiling

    _, ir = _lowered(fn, sizes)
    marked = [e for e in _exps(ir) if e.stabilize is not None]
    assert marked
    with_record = [describe_tile(t) for t in search_tiling(ir, topk=16)]
    assert with_record
    for exp in marked:
        exp.stabilize = None
    without = [describe_tile(t) for t in search_tiling(ir, topk=16)]
    assert with_record == without


def test_schedule_keeps_the_plans_and_reports_no_refusal():
    prog = trace(flash, SHAPES, cl.f16)
    plans = P.schedule(GraphSet.identity(prog), P.ScheduleConfig(topk=4))
    twin = P.schedule(GraphSet.identity(trace(flash_twin, SHAPES, cl.f16)), P.ScheduleConfig(topk=4))
    assert len(plans) == len(twin) > 0 and not plans.refused
    assert [describe_tile(p.tile) for p in plans] == [describe_tile(p.tile) for p in twin]
    (exp,) = _exps(plans[0].tile.flat_graph)
    assert exp.stabilize is not None  # the candidate carries the marked flat graph


def test_refusal_reason_reaches_the_plan_summary():
    prog = trace(cube, dict(R=64, C=256), cl.f16)
    plans = P.schedule(GraphSet.identity(prog), P.ScheduleConfig(topk=4))
    assert not plans and plans.refused
    reason = ("LoweringRejected: graph 0: softmax cannot be stabilized: exp_1: output of div_4 "
              "keeps degree 1 in its scale")
    assert {r.reason for r in plans.refused} == {reason}
    assert reason in plans.summary()


def _split_graph(program):
    structure = json.loads((PL.STRUCTURES_DIR / "gqa" / "before_split_gqa_graph_structure_valid_1.graph")
                           .read_text())
    return SearchedGraph(0, structure, dict(PL.SYMBOL_MAPS["gqa"]), True, program, ("SKV",))


def test_split_graph_of_a_requesting_program_is_refused_for_now():
    """Until the partial exports its shift and the combine re-weights by it."""
    requesting, plain = trace(flash, SHAPES, cl.f16), trace(flash_twin, SHAPES, cl.f16)
    with pytest.raises(LoweringRejected) as e:
        lower(_split_graph(requesting), 4)
    assert str(e.value) == f"graph 0: softmax cannot be stabilized: {SPLIT_SOFTMAX}"
    assert SPLIT_SOFTMAX == "exact softmax across a split is not available yet"
    # the same structure of a program without the request lowers as it always did
    partial, combine = lower(_split_graph(plain), 4)
    assert [e.stabilize for g in (partial, combine) for e in _exps(g)] == [None]


def test_split_refusal_is_per_graph_in_a_schedule():
    """A requesting program keeps its solo plans; each split graph is refused with the reason."""
    prog = trace(flash, SHAPES, cl.f16)
    (solo,) = GraphSet.identity(prog)
    graphs = GraphSet([solo, _split_graph(prog)], prog)
    graphs[1].index = 1
    plans = P.schedule(graphs, P.ScheduleConfig(topk=2, split_k=(PL.SPLIT_FACTOR,)))
    assert plans and {p.variant for p in plans} == {"solo"}
    assert [(r.variant, r.role, str(r.error)) for r in plans.refused] == [
        ("split-k=4", "partial", f"graph 1: softmax cannot be stabilized: {SPLIT_SOFTMAX}")]
    assert f"1x LoweringRejected: graph 1: softmax cannot be stabilized: {SPLIT_SOFTMAX}" in plans.summary()


def test_search_graphs_are_annotated():
    """The request survives the search as a program-level fact; each graph gets its own mark."""
    prog = trace(softmax, dict(R=16, C=256), cl.f16)
    graphs = P.superoptimize(prog, P.SuperoptConfig(cache=False))
    assert len(graphs) >= 1 and all("max" not in g.ops for g in graphs)
    for graph in graphs:
        (ir,) = lower(graph)
        (exp,) = _exps(ir)
        assert exp.stabilize is not None and exp.stabilize.axes == (-1,)


def test_search_of_attention_marks_both_orders():
    """The flash-form graph carries the rolling form, the traced-order graph does not."""
    prog = trace(traced, dict(H=4, SQ=64, SKV=1024, D=64), cl.f16)
    graphs = P.superoptimize(prog, P.SuperoptConfig(max_ops=13, cache=False))
    rolling = {}
    for graph in graphs:
        assert not graph.is_split
        (ir,) = lower(graph)
        (exp,) = _exps(ir)
        assert exp.stabilize is not None and exp.stabilize.axes == (-1,)
        rolling[graph.summary()] = exp.stabilize.rolling
    assert sorted(bool(r) for r in rolling.values()) == [False, True], rolling
