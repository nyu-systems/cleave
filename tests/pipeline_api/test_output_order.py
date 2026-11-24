"""CPU: a Kernel returns a program's outputs in the order the program returns them.

``to_ir`` rebuilt a lowered graph by a walk from its first output, so the generated
kernel's output parameters came in the order that walk found them. For three or more
outputs that need not be the declared order: ``return a, c, b`` of three products that
share an operand came out of the kernel as ``a, b, c``. Each kernel is validated against
the reference of its own graph, which is in the same order, so validation cannot see it;
``Kernel.__call__`` handed the tensors back as they came, exchanged.

``to_ir`` now lists the outputs in the program's order (``tests/lower/test_output_order.py``),
and the declared place of every output is on its ``OutputNode`` (``Boundary.index``), which
is what the Kernel follows for a graph listed in any other order: the order of computation
(``_walked``), as the walk gave it and as a rewrite may build it. The stages here are the
graphs' own torch references, in the order a kernel returns them.
"""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave.frontend import trace
from cleave.ir.graph import Boundary
from cleave.ir.graph_utils import find_topo_sort
from cleave.pipeline.kernel import declared_output_order, load
from cleave.pipeline.plans import bind_inputs, lower
from cleave.pipeline.report import Report, Result
from cleave.pipeline.superopt import GraphSet
from cleave.runtime import evaluate
from cleave.runtime.evaluate import CandidateResult
from cleave.runtime.reference import (declared_output_places, execute, output_order,
                                      placeholder_order)
from cleave.runtime.validate import Tolerance, compare_outputs


def three_products(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"], z: cl.Tensor["K", "N"],
                   u: cl.Tensor["K", "N"]):
    a, b, c = x @ w, x @ z, x @ u
    return a, c, b


def three_products_in_order(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"],
                            z: cl.Tensor["K", "N"], u: cl.Tensor["K", "N"]):
    return x @ w, x @ z, x @ u


def two_products(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"], z: cl.Tensor["K", "N"]):
    a, b = x @ w, x @ z
    return b, a


def sum_max_double(x: cl.Tensor["R", "C"]):
    """Three outputs of three shapes, returned in another order than they are computed."""
    s, e, m = cl.sum(x, -1, keepdim=True), x * 2.0, cl.max(x, 0, keepdim=True)
    return s, m, e


MM = dict(M=64, K=64, N=32)


@pytest.fixture(autouse=True)
def _no_measurement(monkeypatch):
    """These tests compare references and stand-ins, not kernels: nothing they compare
    belongs in a measurement of kernel errors (``CLEAVE_VALIDATION_LOG``)."""
    monkeypatch.delenv("CLEAVE_VALIDATION_LOG", raising=False)
# name -> (program, shapes, the kernel's outputs as places of the program's, where pinned)
PROGRAMS = {
    "three_products": (three_products, MM, [0, 2, 1]),
    "three_products_in_order": (three_products_in_order, MM, [0, 1, 2]),
    "two_products": (two_products, MM, None),
    "sum_max_double": (sum_max_double, dict(R=16, C=8), None),
}


class _Compiled:
    """Stands in for a compiled TileLang kernel: named input buffers, then the outputs, in
    the order the generated kernel declares them. It computes the graph's own reference."""

    def __init__(self, graph):
        names = [ph.name for ph in placeholder_order(graph)]
        n_out = len(output_order(graph))
        params = list(range(len(names) + n_out))
        buffers = {p: SimpleNamespace(name=n)
                   for p, n in zip(params, names + [f"out{i}" for i in range(n_out)])}
        self.adapter = SimpleNamespace(
            prim_func=SimpleNamespace(params=params, buffer_map=buffers),
            result_idx=params[len(names):])
        self.graph = graph

    def __call__(self, *args):
        return tuple(execute(self.graph, list(args)))


def _walked(ir):
    """``ir`` with its outputs listed in the order their producers are computed (the trace
    order of the structure's ops, which name them): the order ``to_ir``'s walk gave the
    three products, ``a, b, c``."""
    outputs = sorted(output_order(ir), key=lambda o: int(o.inputs[0].src_node.name.rsplit("_", 1)[1]))
    ir.nodes = find_topo_sort(outputs)
    return ir


def _kernel(name):
    """The Kernel ``Report.best()`` makes of the program's lowered identity graph (the three
    products listed in the order of computation, ``_walked``), with the graph's own
    reference in the place of the compiled kernel."""
    fn, shapes, _ = PROGRAMS[name]
    prog = trace(fn, shapes, cl.f32)
    (graph,) = GraphSet.identity(prog)
    (ir,) = lower(graph)
    # lowering lists the outputs in the program's order
    assert declared_output_places(ir) in (None, list(range(len(prog.outputs))))
    if name == "three_products":
        ir = _walked(ir)
    plan = SimpleNamespace(tile=SimpleNamespace(flat_graph=ir), role="main", variant="solo",
                           label="g0 solo main", inputs=bind_inputs(ir, prog), program=prog,
                           graph_index=0)
    run = CandidateResult(None, "ok", code="def main_kernel(): pass\n", latency_ms=1.0)
    kernel = Report([Result(plan, run)]).best()
    assert kernel.unrunnable is None
    kernel._compiled = [_Compiled(ir)]
    torch.manual_seed(0)
    inputs = [torch.randn(*spec.shape) for spec in prog.inputs]
    return prog, plan, kernel, inputs


def _assert_program_order(kernel, prog, inputs):
    got, want = kernel(*inputs), prog.reference(*inputs)
    assert [tuple(t.shape) for t in got] == [tuple(t.shape) for t in want]
    for k, (g, w) in enumerate(zip(got, want)):
        compare_outputs([g], [w], Tolerance(rtol=1e-5), what=f"output {k}")


@pytest.mark.parametrize("name", PROGRAMS)
def test_kernel_returns_the_outputs_in_the_declared_order(name, tmp_path):
    prog, plan, kernel, inputs = _kernel(name)
    places = declared_output_places(plan.tile.flat_graph)
    assert sorted(places) == list(range(len(prog.outputs)))
    if PROGRAMS[name][2] is not None:
        assert places == PROGRAMS[name][2]
    assert kernel.output_order == [places.index(k) for k in range(len(places))]
    _assert_program_order(kernel, prog, inputs)

    loaded = load(kernel.save(tmp_path / "k"))
    assert loaded.output_order == kernel.output_order
    loaded._compiled = kernel._compiled
    _assert_program_order(loaded, prog, inputs)


def test_the_kernel_order_of_three_products_is_not_the_declared_one():
    """The case that was wrong: returned positionally, outputs 1 and 2 are exchanged."""
    prog, _, kernel, inputs = _kernel("three_products")
    assert kernel.output_order == [0, 2, 1]
    kernel.output_order = None  # hand the tensors back as the kernel returns them
    got, want = kernel(*inputs), prog.reference(*inputs)
    assert torch.equal(got[0], want[0])
    assert torch.allclose(got[1], want[2]) and not torch.allclose(got[1], want[1])


def _saved_without_the_order(kernel, path):
    saved = json.loads((kernel.save(path) / "kernel.json").read_text())
    del saved["output_order"]
    (path / "kernel.json").write_text(json.dumps(saved))
    return path


def test_a_kernel_saved_without_the_order_of_three_outputs_is_not_loaded(tmp_path):
    _, _, kernel, _ = _kernel("three_products")
    with pytest.raises(ValueError, match="saved without the order of its 3 outputs"):
        load(_saved_without_the_order(kernel, tmp_path / "three"))
    _, _, two, _ = _kernel("two_products")  # the walk starts at output 0: one order
    assert load(_saved_without_the_order(two, tmp_path / "two")).output_order is None


def test_outputs_that_are_not_the_programs_make_the_kernel_unrunnable():
    """Records that do not name every output of the program once give no order to return
    them in: the Kernel says so instead of guessing. A graph without records (a hand-built
    one) is returned as its kernel returns it."""
    prog, plan, _, _ = _kernel("three_products")
    graph = plan.tile.flat_graph.clone()
    fake = SimpleNamespace(tile=SimpleNamespace(flat_graph=graph), label="g0 solo main")
    assert declared_output_order(fake, prog) == ([0, 2, 1], None)
    outs = output_order(graph)
    was = outs[1].boundary
    outs[1].boundary = Boundary(0)
    order, problem = declared_output_order(fake, prog)
    assert order is None and "returns the outputs [#0, #0, #1]" in problem
    outs[1].boundary = Boundary(was.index, "shift")
    order, problem = declared_output_order(fake, prog)
    assert order is None and f"#{was.index}:shift" in problem
    for out in outs:
        out.boundary = None
    assert declared_output_order(fake, prog) == (None, None)
    assert declared_output_places(graph) is None


def test_a_tolerance_per_output_bounds_the_declared_outputs():
    """``tolerance=[t0, t1, t2]`` is one bound per output of the program; a candidate is
    held to them in the order its kernel returns the outputs."""
    _, plan, _, _ = _kernel("three_products")
    graph = plan.tile.flat_graph
    bounds = [Tolerance(rtol=1e-3), Tolerance(rtol=2e-3), Tolerance(rtol=5e-3)]
    assert evaluate.declared_bounds(bounds, graph) == ((bounds[0], bounds[2], bounds[1]), None)
    one = Tolerance(rtol=1e-3)
    assert evaluate.declared_bounds(one, graph) == (one, None)
    # a kernel without its graph, and a graph without records: by position, as given
    assert evaluate.declared_bounds(bounds, None) == (bounds, None)
    bare = graph.clone()
    for out in output_order(bare):
        out.boundary = None
    assert evaluate.declared_bounds(bounds, bare) == (bounds, None)
