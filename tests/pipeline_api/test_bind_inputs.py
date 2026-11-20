"""Program inputs are bound to graph placeholders by declaration, not by graph order.

``Graph.nodes`` is a topological order: for ``b / a`` it lists ``b`` first. Two inputs of one
shape were therefore bound the other way round, and since validation feeds the graph in its
own order, the kernel validated and then computed ``a / b`` for the caller.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.frontend import trace
from cleave.ir.graph import PlaceHolderNode
from cleave.pipeline.kernel import _fit, stages_for
from cleave.pipeline.plans import Plan, bind_inputs, lower
from cleave.pipeline.superopt import GraphSet
from cleave.runtime.reference import execute, placeholder_order
from tests._lowering import run_lowered


def swapped(a: cl.Tensor["R", "C"], b: cl.Tensor["R", "C"]):
    return b / a


def late_first(a: cl.Tensor["R", "C"], b: cl.Tensor["R", "C"], c: cl.Tensor["R", "C"]):
    return (c * 2.0 + b) / a


def attention(q: cl.Tensor["H", "S", "D"], k: cl.Tensor["H", "S", "D"], v: cl.Tensor["H", "S", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1, shift=-1.0) @ v


CASES = [(swapped, dict(R=4, C=8)), (late_first, dict(R=4, C=8)), (attention, dict(H=2, S=16, D=16))]
IDS = [fn.__name__ for fn, _ in CASES]


def _inputs(prog):
    torch.manual_seed(0)
    return [torch.rand(*s.shape) + 0.5 for s in prog.inputs]


def _check(graph, prog):
    (ir,) = lower(graph)
    names = bind_inputs(ir, prog)
    phs = {n.name: n for n in ir.nodes if isinstance(n, PlaceHolderNode)}
    assert names is not None and sorted(names) == sorted(phs)
    assert [phs[n].input_index for n in names] == list(range(len(prog.inputs)))
    ins = _inputs(prog)
    (got,) = run_lowered(ir, prog, ins)
    (want,) = prog.reference(*ins)
    torch.testing.assert_close(got.reshape(want.shape), want, rtol=1e-5, atol=1e-6)
    return ir


@pytest.mark.parametrize("fn, sizes", CASES, ids=IDS)
def test_identity_graph_binds_in_program_order(fn, sizes):
    prog = trace(fn, sizes, cl.f32)
    (graph,) = GraphSet.identity(prog)
    _check(graph, prog)


def test_graph_order_differs_from_program_order():
    """The case is only a test if the lowered graph really lists the inputs out of order."""
    prog = trace(swapped, dict(R=4, C=8), cl.f32)
    (graph,) = GraphSet.identity(prog)
    ir = _check(graph, prog)
    in_graph_order = [n.input_index for n in ir.nodes if isinstance(n, PlaceHolderNode)]
    assert in_graph_order == [1, 0]


@pytest.mark.parametrize("fn, sizes", CASES[:2], ids=IDS[:2])
def test_search_graphs_bind_in_program_order(fn, sizes):
    prog = trace(fn, sizes, cl.f32)
    graphs = [g for g in P.superoptimize(prog, P.SuperoptConfig(cache=False))
              if not g.is_split and not g.invalid_split]
    assert graphs
    for graph in graphs:
        _check(graph, prog)


def chained(q: cl.Tensor["BH", "SQ", "D"], k: cl.Tensor["BH", "SKV", "D"], v: cl.Tensor["BH", "SKV", "D"]):
    return (q @ k.mT) @ v  # demo/matmul_matmul.py: k and v have one shape


class _Tile:
    def __init__(self, flat_graph):
        self.flat_graph = flat_graph


def _call(stages, graphs, inputs):
    """``Kernel.__call__`` with the graph executor in place of each compiled stage."""
    outs = []
    for stage, graph in zip(stages, graphs):
        feed = {p.name: _fit(inputs[p.index] if p.source == "input" else outs[p.index], p.shape)
                for p in stage.params}
        outs = execute(graph, [feed[n.name] for n in placeholder_order(graph)])
    return outs


def test_search_split_partials_bind_in_program_order():
    """A split partial is bound like a solo graph (``_schedule_once``), and most searched
    graphs whose binding the declaration order changed are partials: by graph order one
    partial of this search took ``v`` for ``k`` and two could not be bound at all."""
    prog = trace(chained, dict(BH=4, SQ=8, SKV=64, D=16), cl.f32)
    graphs = [g for g in P.superoptimize(prog, P.SuperoptConfig(max_ops=8, split=("SKV",), cache=False))
              if g.is_split and not g.invalid_split]
    assert graphs
    torch.manual_seed(0)
    ins = [torch.randn(*s.shape) / 4 for s in prog.inputs]
    (want,) = prog.reference(*ins)
    for graph in graphs:
        partial, combine = lower(graph, 4)
        names = bind_inputs(partial, prog)
        phs = {n.name: n for n in partial.nodes if isinstance(n, PlaceHolderNode)}
        assert names is not None and sorted(names) == sorted(phs), graph.summary()
        assert [phs[n].input_index for n in names] == list(range(len(prog.inputs)))
        plans = [Plan(graph.index, "split-k=4", role, _Tile(ir), f"g{graph.index} {role}", 0, 4, graph, bound)
                 for role, ir, bound in (("partial", partial, names), ("combine", combine, None))]
        stages, problem = stages_for([(plan, None) for plan in plans], prog)
        assert problem is None, graph.summary()
        (got,) = _call(stages, (partial, combine), ins)
        assert got.numel() == want.numel(), graph.summary()
        torch.testing.assert_close(got.reshape(want.shape), want, rtol=1e-5, atol=1e-5)


def test_kernel_stage_params_follow_the_binding():
    """``Kernel.__call__`` feeds placeholder ``p`` with ``inputs[p.index]``."""
    prog = trace(swapped, dict(R=4, C=8), cl.f32)
    (graph,) = GraphSet.identity(prog)
    (ir,) = lower(graph)

    class Tile:
        flat_graph = ir

    plan = Plan(0, "solo", "main", Tile(), "g0 solo main #0", 0, None, graph, bind_inputs(ir, prog))
    stages, problem = stages_for([(plan, None)], prog)
    assert problem is None
    (stage,) = stages
    index = {p.name: p.index for p in stage.params}
    declared = {n.name: n.input_index for n in ir.nodes if isinstance(n, PlaceHolderNode)}
    assert index == declared and sorted(index.values()) == [0, 1]


def test_graphs_without_the_record_bind_by_shape_or_not_at_all():
    """No ``input_index`` (a hand-built graph): unique shapes decide, equal shapes do not."""
    def distinct(a: cl.Tensor["R", "C"], w: cl.Tensor["C", "D"]):
        return w.mT @ a.mT  # the graph reaches ``w`` first

    for fn, sizes, bound in ((distinct, dict(R=4, C=8, D=16), True), (swapped, dict(R=4, C=8), False)):
        prog = trace(fn, sizes, cl.f32)
        (graph,) = GraphSet.identity(prog)
        (ir,) = lower(graph)
        want = bind_inputs(ir, prog)
        phs = [n for n in ir.nodes if isinstance(n, PlaceHolderNode)]
        assert [n.input_index for n in phs] == [1, 0]
        for n in phs:
            del n.input_index
        assert bind_inputs(ir, prog) == (want if bound else None)
