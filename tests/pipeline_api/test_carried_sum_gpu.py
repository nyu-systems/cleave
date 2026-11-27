"""A sum carried across a loop by a searched update rule, on a GPU.

``exp(y - sum(y)) @ v / sum(exp(y - sum(y)))`` for ``y = x / 16``: the softmax of ``y``,
shifted by the sum of its row instead of its max. The conflict's ancestor is a sum; the
update rule of the row sum and of the value matmul under it is the one a max gets,
``a * exp(m - m')`` (``tests/update/test_general_conflicts.py`` on CPU). Here every plan
the pipeline makes for it is compiled and validated: the ones whose loop carries the sum
from zero and repairs both accumulators every iteration, and the ones that hold the row
whole. (``y = x / 16`` keeps the sum of a row of 256 normal values near N(0, 1), where the
float16 exp of the flat graph itself is finite.)

``test_gpu_*`` compile kernels (root conftest marks).
"""

import pytest

torch = pytest.importorskip("torch")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.frontend import trace
from cleave.ir.ops import ReduceSum
from cleave.pipeline.report import validation_kind
from cleave.schedule import apply_tiling
from cleave.schedule.streaming import sites

from tests._tolerances import tolerance_for

TOLERANCE = tolerance_for("sum_shifted")
SHAPES = dict(R=64, C=256, D=64)


def sum_shifted(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    y = x * 0.0625
    e = cl.exp(y - cl.sum(y, -1, keepdim=True))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    return tmp_path


def test_gpu_a_sum_carried_across_its_loop_validates(cache_root):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    prog = trace(sum_shifted, SHAPES, cl.f16)
    graphs = P.superoptimize(prog, P.SuperoptConfig())
    plans = P.schedule(graphs, P.ScheduleConfig(topk=1000))
    report = P.evaluate(plans, P.EvalConfig(warmup=2, iters=10, max_parallel=4,
                                            tolerance=TOLERANCE))
    failed = [(r.label, r.error_type, validation_kind(r.error), str(r.error)[-300:])
              for r in report if not r.ok]
    assert not failed, failed
    looped, whole = [], []
    for r in report:
        found = sites(apply_tiling(r.plan.tile))
        if found:
            (site,) = found
            full = next(n for n in r.plan.tile.flat_graph.nodes if n.name == site.full)
            assert type(full) is ReduceSum
            looped.append(r)
        else:
            whole.append(r)
        for measured in r.measured:
            assert measured.rel <= TOLERANCE.rtol, (r.label, measured)
    assert looped and whole, ([r.label for r in looped], [r.label for r in whole])
