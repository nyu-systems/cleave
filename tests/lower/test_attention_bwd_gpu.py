"""GPU: the single-group attention-backward kernels against torch autograd of the forward.

Inputs are realistic: L = logsumexp(q k^T c) in fp32, o = softmax(q k^T c) v, D = rowsum(dO * o).
The reference is autograd in fp32 from the fp16 inputs, rounded to fp16; every kernel measured
<= 9.71e-4 row-normalized (one fp16 ulp of the row max), hence the bound.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

from cleave.pipeline import GraphSet, ScheduleConfig, schedule  # noqa: E402
from cleave.pipeline.build import BuiltKernel  # noqa: E402
from cleave.pipeline.kernel import Kernel, stages_for  # noqa: E402
from cleave.runtime.compile import build_candidate_code  # noqa: E402
from cleave.runtime.validate import Tolerance, compare_outputs  # noqa: E402
from tests.lower import _attention_bwd as A  # noqa: E402

TOL = Tolerance(rtol=1.5e-3)
OUTPUTS = {"dv": ("dv",), "dq": ("dq",), "dk": ("dk",), "dkdv": ("dk", "dv"), "dkdv_o": ("dk", "dv")}


def _inputs(program):
    shapes = {s.name: s.shape for s in program.inputs}
    H, SQ, D = shapes["q"]
    SKV = shapes["k"][1]
    g = torch.Generator(device="cuda").manual_seed(0)
    r = lambda *sh: torch.randn(*sh, generator=g, device="cuda", dtype=torch.float16)  # noqa: E731
    q, k, v, dO = r(H, SQ, D), r(H, SKV, D), r(H, SKV, D), r(H, SQ, D)
    s = (q.float() @ k.float().mT) * A.C
    o = (torch.softmax(s, dim=-1) @ v.float()).half()
    env = dict(q=q, k=k, v=v, dO=dO, L=torch.logsumexp(s, dim=-1, keepdim=True), o=o,
               D=(dO.float() * o.float()).sum(-1, keepdim=True).half())
    qf, kf, vf = (x.float().requires_grad_() for x in (q, k, v))
    (torch.softmax((qf @ kf.mT) * A.C, dim=-1) @ vf).backward(dO.float())
    ref = dict(dq=qf.grad.half(), dk=kf.grad.half(), dv=vf.grad.half())
    return [env[spec.name] for spec in program.inputs], ref


@pytest.mark.parametrize("name", sorted(OUTPUTS))
def test_matches_autograd(name):
    program = A.program(name)
    plan = list(schedule(GraphSet.identity(program), ScheduleConfig(topk=1)))[0]
    assert plan.n_threads == 128
    built = BuiltKernel.from_plan(plan, build_candidate_code(plan.tile))
    stages, problem = stages_for([(plan, None, built)], program)
    assert problem is None
    kernel = Kernel(stages, name=name, input_names=[s.name for s in program.inputs],
                    input_shapes=[tuple(s.shape) for s in program.inputs])
    args, ref = _inputs(program)
    out = kernel(*args)
    out = out if isinstance(out, tuple) else (out,)
    want = [ref[o] for o in OUTPUTS[name]]
    compare_outputs([a.reshape(w.shape) for a, w in zip(out, want)], want, TOL, what=name)
