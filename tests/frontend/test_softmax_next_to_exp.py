"""``cl.softmax(x)`` next to another ``exp`` of the same tensor (CPU; the search runs).

``cl.softmax`` records its max, so the exp of the softmax reads ``x - max(x)`` and the
other one reads ``x``: two different values, which the search never merges. Every graph it
returns has both exps, the softmax's is exact by structure
(``cleave.runtime.inputs.softmax_axis``), and the program is served wherever a softmax
that is returned is served: while its row fits one tile (the exp of the softmax is read by
the division, which is not summed over the row, so its max is not carried across a loop).
Two softmaxes of ``x`` over different axes need a row and a column in one tile.

The sizes below pin which programs have plans with the explicit max.

``test_search_*`` run the superoptimizer (root conftest marks).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.frontend import trace
from cleave.ir.ops import Exp
from cleave.pipeline.plans import MAX_ROW_LIMIT, NO_TILING, lower
from cleave.runtime.inputs import adversarial_cases, softmax_axis, why_not_adversarial
from cleave.runtime.validate import Tolerance, compare_outputs
from tests._lowering import run_lowered


def dual_softmax(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1) * cl.softmax(x, 0)


def softmax_plus_exp(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1) + cl.exp(x)


def softmax_times_exp(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1) * cl.exp(x)


def softmax_alone(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1)


SEARCH = P.SuperoptConfig(cache=False)


@pytest.fixture(autouse=True)
def _no_measurement(monkeypatch):
    """These tests compare lowered graphs with the program, not kernels: nothing they
    compare belongs in a measurement of kernel errors (``CLEAVE_VALIDATION_LOG``)."""
    monkeypatch.delenv("CLEAVE_VALIDATION_LOG", raising=False)


# name -> (program, shapes, the softmax axes of its exps, sorted; None for a plain exp)
PROGRAMS = {
    "dual_softmax": (dual_softmax, dict(R=16, C=32), [-2, -1]),
    "softmax_plus_exp": (softmax_plus_exp, dict(R=64, C=256), [-1, None]),
    "softmax_times_exp": (softmax_times_exp, dict(R=64, C=256), [-1, None]),
}


def _axes(flat):
    return sorted((softmax_axis(n) for n in flat.nodes if type(n) is Exp),
                  key=lambda a: (a is None, a))


BOUND = Tolerance(rtol=5e-3)  # of the row max; an all-zero row has to be zeros


def _close(got, want, what=""):
    compare_outputs([got], [want], BOUND, what=what)


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_search_a_softmax_next_to_an_exp_of_its_logits_has_plans(name):
    """Every searched graph keeps the exps apart: the softmax's reads ``x - max(x)`` and is
    exact by structure, the other one stays plain. Each graph lowers and computes the
    program, and the program schedules."""
    fn, shapes, axes = PROGRAMS[name]
    prog = trace(fn, shapes, cl.f16)
    graphs = P.superoptimize(prog, SEARCH)
    assert len(graphs) >= 1
    inputs = [torch.randn(*spec.shape).to(spec.dtype.torch) for spec in prog.inputs]
    (want,) = prog.reference(*inputs)
    for graph in graphs:
        (flat,) = lower(graph)
        assert _axes(flat) == axes, graph.summary()
        (got,) = run_lowered(flat, prog, inputs)
        _close(got, want, graph.summary())
    plans = P.schedule(graphs, P.ScheduleConfig(topk=2))
    assert len(plans) > 0, plans.summary()
    for plan in plans:
        assert _axes(plan.tile.flat_graph) == axes
        (got,) = run_lowered(plan.tile.flat_graph, prog, inputs)
        _close(got, want, plan.label)


def _plans(fn, rows, cols):
    prog = trace(fn, dict(R=rows, C=cols), cl.f16)
    return P.schedule(P.superoptimize(prog, SEARCH), P.ScheduleConfig(topk=8))


def _output_tiles(plans):
    tiles = set()
    for plan in plans:
        (out,) = [n for n in plan.tile.flat_graph.nodes if n.is_output()]
        tiles.add(tuple(int(t) for t in plan.tile.tile_map[out.inputs[0].src_node]))
    return tiles


def _reasons(plans):
    assert plans.refused
    return [str(refusal.error) for refusal in plans.refused]


def test_search_where_a_softmax_next_to_an_exp_of_its_logits_has_plans():
    """Which sizes have plans (the ``cl.softmax`` docstring states the limit). These return their
    softmax, and a softmax that is returned is served while its row is in one tile: its
    exp is read by the division, which is not summed over the row, so the max is not
    carried across a loop and the reason says so."""
    two_passes = ("which is not summed over that dim: that needs two passes over the row")
    # softmax + exp: whole rows in every tile; 4096 columns have a plan, 8192 have none: the
    # one tiling that holds a row of 8192 in one block holds one row, which no kernel has
    plans = _plans(softmax_plus_exp, 64, 4096)
    assert len(plans) >= 1 and {tile[1] for tile in _output_tiles(plans)} == {4096}
    plans = _plans(softmax_plus_exp, 64, 8192)
    assert len(plans) == 0
    reasons = _reasons(plans)
    assert all(reason.startswith(f"{NO_TILING}; a max is not carried across a loop: ")
               and two_passes in reason and "a unit extent that is not a batch dim" in reason
               for reason in reasons), reasons
    # ... where the softmax alone still has one; it has none at 16384, where the only
    # tilings that hold a row in one block hold a single row
    plans = _plans(softmax_alone, 64, 8192)
    assert len(plans) >= 1 and {tile[1] for tile in _output_tiles(plans)} == {8192}
    plans = _plans(softmax_alone, 64, 16384)
    assert len(plans) == 0
    reasons = _reasons(plans)
    assert all(reason.startswith(f"{NO_TILING}; a max is not carried across a loop: ")
               and two_passes in reason and "a unit extent that is not a batch dim" in reason
               for reason in reasons), reasons
    # two softmaxes, over the rows and over the columns: the tile is the whole tensor
    for rows, cols in ((16, 32), (64, 128)):
        plans = _plans(dual_softmax, rows, cols)
        assert len(plans) >= 1 and _output_tiles(plans) == {(rows, cols)}, plans.summary()
    plans = _plans(dual_softmax, 64, 256)
    assert len(plans) == 0, plans.summary()
    reasons = _reasons(plans)
    assert all(MAX_ROW_LIMIT in reason and two_passes in reason for reason in reasons), reasons


def test_search_every_graph_of_two_softmaxes_has_its_whole_tile_plan():
    """The row max (over C) and the column max (over R) both read ``x`` and feed neither
    other, so the policy puts them, and both sums, into one group that steps in lockstep
    (``DefaultPolicy._get_compatible_reduction_groups``, rule 2). The walk used to step a
    group only while its first member could: where that was the row max, already at its
    extent, the column max never left its first step, its loop was refused (a max is not
    carried across a loop here), and 5 of the 12 searched graphs had no plan at all. Every
    graph has the tiling that holds the whole tensor, with every reduction at its extent."""
    prog = trace(dual_softmax, dict(R=16, C=32), cl.f16)
    graphs = P.superoptimize(prog, SEARCH)
    plans = P.schedule(graphs, P.ScheduleConfig(topk=100000))
    assert (len(graphs), len(plans), len(plans.refused)) == (12, 33, 0), plans.summary()

    def whole(plan):
        return list(plan.tile.output_tile) == [16, 32] and all(
            int(step) == int(ax.dom.extent)
            for node, steps in plan.tile.rstep_map.items() for ax in node.raxis
            for name, step in steps.items() if name == ax.var.name)

    assert {p.graph_index for p in plans if whole(p)} == set(range(len(graphs)))


def test_search_two_softmaxes_of_one_input_are_exact_at_large_logits():
    """Both softmaxes of the dual softmax carry their own max, each over its own axis: at
    x100 logits every lowered graph is finite and is the program, and its kernels are held
    to the adversarial cases."""
    fn, shapes, _ = PROGRAMS["dual_softmax"]
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
