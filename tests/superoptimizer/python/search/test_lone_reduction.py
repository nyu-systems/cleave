"""A program that is only a reduction is searched and scheduled like any other.

The reduced dim of an input that nothing but a reduction reads was tied to no other dim by
the grouping rules of the symbolic graph, so it got no symbol, and building the symbolic
graph aborted the whole process (a C++ assert in ``DimensionSymbolizer::get_symbol``, core
dumped) for ``cl.max(x)``, ``cl.sum(x)``, ``cl.max(x) + 1`` and the like. Such a dim is now a
group of its own; a program whose input dims all have a group is grouped and numbered as
before (``test_search_reproducible.py`` and the pipeline snapshots hold the rest).

Every program runs in a fresh interpreter on the CPU, so an abort fails its case instead of
taking the test session with it.
"""

import json
import os
import subprocess
import sys

import pytest

CHILD = """
import json, sys

import cleave.lang as cl
from cleave.frontend import trace
from cleave.pipeline import ScheduleConfig, SuperoptConfig, schedule, superoptimize


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


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_search_a_lone_reduction_is_served(name, tmp_path):
    expr, summaries = PROGRAMS[name]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CLEAVE_CACHE_ROOT=str(tmp_path / "cache"))
    proc = subprocess.run([sys.executable, "-c", CHILD.format(expr=expr)], capture_output=True,
                          text=True, timeout=900, env=env, cwd=tmp_path)
    assert proc.returncode == 0, (
        f"{expr}: the child exited with {proc.returncode}"
        f"{' (aborted: SIGABRT)' if proc.returncode == -6 else ''}\n{proc.stderr[-3000:]}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert len(lines) == 1, proc.stdout[-3000:]
    got = json.loads(lines[0][len("RESULT "):])
    assert got["graphs"] == summaries
    assert got["plans"] == 4 and got["refused"] == []
