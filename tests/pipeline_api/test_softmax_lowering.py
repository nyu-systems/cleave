"""``pipeline.plans.lower`` of a ``cl.softmax`` program: the max is in the flat graph (CPU).

``cl.softmax`` records ``exp(x - max(x)) / sum``, and the flat graph keeps it as written: a
``ReduceMax`` and the subtraction ahead of the exp, which ``runtime.inputs.softmax_axis``
recognizes as the exp of an exact softmax. Where a reduction wants to loop under the max (a
streaming obstacle), lowering writes the proven update rule on the max (a ``Streaming``
record) or the reason it has none (``ir.streaming_refusals``); a graph that has no tiling
says that reason in the plan summary.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.frontend import trace
from cleave.ir.ops import Exp, ReduceMax, Sub
from cleave.ir.update_rule import loop_key
from cleave.pipeline.plans import MAX_ROW_LIMIT, _lower_structure, describe_tile, lower
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.inputs import softmax_axis
from cleave.runtime.reference import fingerprint_seed
from tests._explicit_softmax import stable_exp

FACTOR = ("exp", ("sub", "m", "m'"))


def flash(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], v: cl.Tensor["H", "SKV", "D"]):
    e = stable_exp((q @ k.mT) / q.shape[-1] ** 0.5, -1)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def flash_twin(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], v: cl.Tensor["H", "SKV", "D"]):
    e = cl.exp((q @ k.mT) / q.shape[-1] ** 0.5)  # the same records without the max
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def traced(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, -1) @ v


def softmax(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1)


def cube(x: cl.Tensor["R", "C"]):
    e = stable_exp(x, -1)
    return (e * e) / cl.sum(e, -1, keepdim=True)


SHAPES = dict(H=32, SQ=64, SKV=4096, D=64)
SHORT = dict(H=32, SQ=64, SKV=256, D=64)


def _exps(ir):
    return [n for n in ir.nodes if type(n) is Exp]


def _lowered(fn, sizes, dtype=cl.f16):
    prog = trace(fn, sizes, dtype)
    (graph,) = GraphSet.identity(prog)
    (ir,) = lower(graph)
    return prog, ir


def _without_derivation(fn, sizes, dtype=cl.f16):
    (graph,) = GraphSet.identity(trace(fn, sizes, dtype))
    (ir,) = _lower_structure(graph, graph.structure)
    return ir


def test_identity_graph_carries_the_proven_rule_on_its_max():
    _, ir = _lowered(flash, SHAPES)
    (full,) = [n for n in ir.nodes if type(n) is ReduceMax]
    record = full.streaming
    assert [type(next(n for n in ir.nodes if n.name == name)).__name__
            for name in record.loops] == ["Matmul", "ReduceSum"]
    assert record.rules == tuple((acc, FACTOR) for acc in record.loops)
    assert record.key == loop_key(ir, full.name, record.loops)
    assert ir.streaming_refusals == [] and len(ir.derivations) == 1


def test_flat_graph_has_the_max_ahead_of_the_exp():
    """The plain twin's nodes plus the max and the subtraction, which the exp reads: the
    exp is the exp of an exact softmax over the last axis by structure. The validation
    inputs are seeded by the flat graph, so they are not the twin's."""
    _, ir = _lowered(flash, SHAPES)
    _, twin = _lowered(flash_twin, SHAPES)
    (exp,) = _exps(ir)
    sub = exp.inputs[0].src_node
    (full,) = [n for n in ir.nodes if type(n) is ReduceMax]
    assert type(sub) is Sub and {e.src_node.name for e in sub.inputs} == {
        full.name, full.inputs[0].src_node.name}
    assert softmax_axis(exp) == -1 and [softmax_axis(e) for e in _exps(twin)] == [None]
    kinds = sorted(type(n).__name__ for n in ir.nodes)
    assert kinds == sorted([type(n).__name__ for n in twin.nodes] + ["ReduceMax", "Sub"])
    assert full.get_dtype(0) == "float32"  # the statistic is float32 over float16 scores
    assert fingerprint_seed(ir) != fingerprint_seed(twin)


@pytest.mark.parametrize("fn, sizes, why", [
    (traced, SHORT, "a finished reduction over the dim"),
    (softmax, dict(R=64, C=256), "which is not summed over that dim"),
], ids=["traced_256", "softmax_256"])
def test_a_refused_obstacle_leaves_the_tiling_candidates_as_they_are(fn, sizes, why):
    """A graph whose obstacle is not served gets its reason, and ``search_tiling`` lists the
    candidates of the same graph lowered without any derivation, in the same order."""
    from cleave.schedule import search_tiling

    _, ir = _lowered(fn, sizes)
    (reason,) = ir.streaming_refusals
    assert why in reason and not [n for n in ir.nodes if getattr(n, "streaming", None)]
    plain = _without_derivation(fn, sizes)
    with_reason = [describe_tile(t) for t in search_tiling(ir, topk=16)]
    assert with_reason
    assert with_reason == [describe_tile(t) for t in search_tiling(plain, topk=16)]


def test_schedule_keeps_looped_plans_and_reports_no_refusal():
    """The flash form at S_KV = 4096: every plan loops over the keys with the max carried;
    the plain twin, without a max, has the same number of tilings."""
    from cleave.schedule import apply_tiling
    from cleave.schedule.streaming import sites

    prog = trace(flash, SHAPES, cl.f16)
    plans = P.schedule(GraphSet.identity(prog), P.ScheduleConfig(topk=4))
    twin = P.schedule(GraphSet.identity(trace(flash_twin, SHAPES, cl.f16)), P.ScheduleConfig(topk=4))
    assert len(plans) == len(twin) == 4 and not plans.refused
    for plan in plans:
        (site,) = sites(apply_tiling(plan.tile))
        assert site.extent == site.block == 4096 and 1 < site.step < 4096
        (full,) = [n for n in plan.tile.flat_graph.nodes if type(n) is ReduceMax]
        assert full.streaming is not None  # the candidate carries the recorded flat graph


def test_refusal_reason_reaches_the_plan_summary():
    """A row longer than a block whose exp is also read where it is not summed (the square
    of the softmax's numerator): no tiling, and the summary says why the max is not carried."""
    prog = trace(cube, dict(R=64, C=16384), cl.f16)
    plans = P.schedule(GraphSet.identity(prog), P.ScheduleConfig(topk=4))
    assert not plans and plans.refused  # (fusion="auto": once unforced, once forced)
    for refusal in plans.refused:
        text = str(refusal.error)
        assert text.startswith(f"{MAX_ROW_LIMIT}; it is not carried across a loop: reduce_max_")
        assert "is also read by mul_" in text and "which is not summed over that dim" in text
        assert refusal.reason in plans.summary()


def test_search_graphs_keep_the_max():
    """The max is an opaque reduction of the searched graph: every graph the search returns
    for ``cl.softmax`` has it, and its exp is an exact softmax by structure."""
    prog = trace(softmax, dict(R=16, C=256), cl.f16)
    graphs = P.superoptimize(prog, P.SuperoptConfig(cache=False))
    assert len(graphs) >= 1 and all({"max", "sub"} <= set(g.ops) for g in graphs)
    for graph in graphs:
        (ir,) = lower(graph)
        (exp,) = _exps(ir)
        assert softmax_axis(exp) == -1
        (reason,) = ir.streaming_refusals  # a returned softmax: T4
        assert "which is not summed over that dim" in reason


def test_search_of_attention_records_the_flash_form_only():
    """The flash form (row sum and value matmul siblings) gets the proven rule; the traced
    order, whose value matmul reads the finished row sum, keeps its pin with the reason."""
    prog = trace(traced, dict(H=4, SQ=64, SKV=1024, D=64), cl.f16)
    graphs = P.superoptimize(prog, P.SuperoptConfig(max_ops=13, cache=False))
    found = {}
    for graph in graphs:
        assert not graph.is_split
        (ir,) = lower(graph)
        (full,) = [n for n in ir.nodes if type(n) is ReduceMax]
        assert softmax_axis(_exps(ir)[0]) == -1
        found[graph.summary()] = (getattr(full, "streaming", None) is not None,
                                  ir.streaming_refusals)
    assert sorted(found) == ["solo matmul>div>max>sub>exp>sum>div>matmul",
                             "solo matmul>div>max>sub>exp>sum>matmul>div"]
    assert found["solo matmul>div>max>sub>exp>sum>matmul>div"] == (True, [])
    recorded, (why,) = found["solo matmul>div>max>sub>exp>sum>div>matmul"]
    assert not recorded and "a finished reduction over the dim" in why
