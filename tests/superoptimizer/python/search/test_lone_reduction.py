"""A program that is only a reduction is searched and scheduled like any other, and a program
the search finds no graph for is refused with its reason.

The reduced dim of an input that nothing but a reduction reads was tied to no other dim by
the grouping rules of the symbolic graph, so it got no symbol, and building the symbolic
graph aborted the whole process (a C++ assert in ``DimensionSymbolizer::get_symbol``, core
dumped) for ``cl.max(x)``, ``cl.sum(x)``, ``cl.max(x) + 1`` and the like. Such a dim is now a
group of its own; a program whose input dims all have a group is grouped and numbered as
before (``test_search_reproducible.py`` and the pipeline snapshots hold the rest).

An input that no op reads aborted the same way (no rule mentions any of its dims). The native
search would find no graph for it: it counts every tensor that nothing reads as an output, so
no graph has the program's outputs; ``Program.to_tile_graph`` does not build the symbolic graph
of such an input (one declared ``equals=`` aside). ``superoptimize`` refuses such a program, and
one that returns an input, with a ``SpecError`` before the search, and any search that finds
no graph (a reduction over a dim of one element, a ``max_ops`` below the program's own size)
with a ``SuperoptError`` that says so. Before, an empty ``GraphSet`` came back, which
``schedule`` turned into no plan and no refusal, and a returned input was refused by lowering
with the text of an internal assert.

Every program runs in a fresh interpreter on the CPU, so an abort fails its case instead of
taking the test session with it.
"""

import json
import os
import subprocess
import sys

import pytest

PRELUDE = """
import json, sys

import cleave.lang as cl
import cleave.superoptimizer as csopt
from cleave.core.errors import CleaveError
from cleave.frontend import trace
from cleave.pipeline import ScheduleConfig, SuperoptConfig, schedule, superoptimize
"""

CHILD = PRELUDE + """

def f(x: cl.Tensor["R", "C"]):
    return {expr}


program = trace(f, dict(R=64, C=1024), cl.f16)
graphs = superoptimize(program, SuperoptConfig(cache=False))
plans = schedule(graphs, ScheduleConfig(topk=4))
print("RESULT " + json.dumps(dict(graphs=[g.summary() for g in graphs], plans=len(plans),
                                   refused=[str(r.reason) for r in plans.refused])))
"""

PROGRAMS = {
    "max": ("cl.max(x, -1, keepdim=True)", ["solo max"]),
    "sum": ("cl.sum(x, -1, keepdim=True)", ["solo sum"]),
    "max_first_axis": ("cl.max(x, 0, keepdim=True)", ["solo max"]),
    "sum_first_axis": ("cl.sum(x, 0, keepdim=True)", ["solo sum"]),
    "max_plus_one": ("cl.max(x, -1, keepdim=True) + 1.0", ["solo max>add"]),
    "max_times_two": ("cl.max(x, -1, keepdim=True) * 2.0", ["solo max>mul"]),
    "max_and_sum": ("(cl.max(x, -1, keepdim=True), cl.sum(x, -1, keepdim=True))", ["solo sum>max"]),
}


def _run(source, tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CLEAVE_CACHE_ROOT=str(tmp_path / "cache"))
    proc = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True,
                          timeout=900, env=env, cwd=tmp_path)
    assert proc.returncode == 0, (
        f"the child exited with {proc.returncode}"
        f"{' (aborted: SIGABRT)' if proc.returncode == -6 else ''}\n{proc.stderr[-3000:]}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert len(lines) == 1, proc.stdout[-3000:]
    return json.loads(lines[0][len("RESULT "):])


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_search_a_lone_reduction_is_served(name, tmp_path):
    expr, summaries = PROGRAMS[name]
    got = _run(CHILD.format(expr=expr), tmp_path)
    assert got["graphs"] == summaries
    assert got["plans"] == 4 and got["refused"] == []


REFUSED_CHILD = PRELUDE + """

def f({params}):
    return {expr}


program = trace(f, {sizes}, cl.f16)
try:
    graphs = superoptimize(program, SuperoptConfig(cache=False, {options}))
except CleaveError as e:
    print("RESULT " + json.dumps(dict(error=type(e).__name__, message=str(e))))
else:
    print("RESULT " + json.dumps(dict(graphs=[g.summary() for g in graphs])))
"""

X, XY = 'x: cl.Tensor["R", "C"]', 'x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]'
WIDE, UNIT = dict(R=64, C=1024), dict(R=64, C=1)
NO_GRAPH = "f: the search found no graph equal to the program within max_ops="
UNIT_DIM = "reduces one element, and the search builds no reduction over a dim of one element"
UNREAD = ("f: input 'y' is read by no op; the search counts a tensor that nothing reads as an "
          "output, so none of its graphs can equal the program (drop the input or read it)")
RETURNED = ("f: output 'out0' is input 'x'; no op writes such an output, so no kernel computes "
            "it (return a computed value)")

REFUSED = {
    "unread_input": (XY, "cl.exp(x)", WIDE, "", "SpecError", UNREAD),
    "unread_input_next_to_a_sum": (XY, "cl.sum(x, -1, keepdim=True)", WIDE, "", "SpecError",
                                   UNREAD),
    # returned as it is: its one graph has no op, and lowering refused it with an internal
    # assert ("OutputNode can only connect to ..."); next to its exp the search found none
    "returned_input": (X, "x", WIDE, "", "SpecError", RETURNED),
    "returned_input_next_to_its_exp": (X, "(x, cl.exp(x))", WIDE, "", "SpecError", RETURNED),
    "sum_over_a_unit_dim": (X, "cl.sum(x, -1, keepdim=True)", UNIT, "", "SuperoptError",
                            f"{NO_GRAPH}13; its sum over dim C {UNIT_DIM}"),
    "max_over_a_unit_dim": (X, "cl.max(x, -1, keepdim=True)", UNIT, "", "SuperoptError",
                            f"{NO_GRAPH}13; its max over dim C {UNIT_DIM}"),
    "softmax_within_five_ops": (
        X, "cl.softmax(x, -1)", WIDE, "max_ops=5", "SuperoptError",
        f"{NO_GRAPH}5; max_ops=5 is less than the program's own 6 ops (inputs and compute "
        "ops), so the search cannot reach its own graph"),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_search_a_program_without_a_graph_is_refused_with_its_reason(name, tmp_path):
    params, expr, sizes, options, error, message = REFUSED[name]
    got = _run(REFUSED_CHILD.format(params=params, expr=expr, sizes=sizes, options=options),
               tmp_path)
    assert got == dict(error=error, message=message)


NATIVE_CHILD = PRELUDE + """

def f(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]):
    return cl.exp(x)


try:
    trace(f, dict(R=64, C=1024), cl.f16).to_tile_graph()
    print("RESULT " + json.dumps(dict(error=None)))
except CleaveError as e:
    print("RESULT " + json.dumps(dict(error=type(e).__name__, message=str(e))))
"""


def test_search_the_native_search_is_not_handed_a_program_with_an_unread_input(tmp_path):
    """Under the refusal of ``superoptimize``: the tile graph is not built either, so the
    native search is never handed it."""
    assert _run(NATIVE_CHILD, tmp_path) == dict(
        error="SpecError", message="f: input 'y' is not read by any output")


NATIVE_UNREAD_CHILD = PRELUDE + """
from cleave.frontend.program import Program


def f(x: cl.Tensor["R", "C"], y: cl.Tensor["R", "C"]):
    return cl.exp(x)


Program._unread_inputs = lambda self: []  # past the frontend's refusal, on purpose
symbolic, _ = csopt.create_symbolic_tile_graph(trace(f, dict(R=64, C=1024), cl.f16).to_tile_graph())
config = symbolic.setup_search_config()
config.max_num_kernel_graph_op = 13
print("RESULT " + json.dumps(dict(graphs=len(symbolic.search_tile_graph("", False, False, config)))))
"""


def test_search_the_native_search_of_a_program_with_an_unread_input_finds_no_graph(tmp_path):
    """Under both refusals: handed such a tile graph anyway, the native search itself does
    not abort (the input's dims get symbols of their own) and finds no graph."""
    assert _run(NATIVE_UNREAD_CHILD, tmp_path) == dict(graphs=0)
