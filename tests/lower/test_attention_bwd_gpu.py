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
OUTPUTS = {"dv": ("dv",), "dq": ("dq",), "dk": ("dk",), "dkdv": ("dk", "dv"), "dkdv_o": ("dk", "dv"),
           "dq_o": ("dq",)}
FUSED = {"fused_o": ("dq", "dk", "dv")}


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


def _check(name, program, plan):
    built = BuiltKernel.from_plan(plan, build_candidate_code(plan.tile))
    stages, problem = stages_for([(plan, None, built)], program)
    assert problem is None
    kernel = Kernel(stages, name=name, input_names=[s.name for s in program.inputs],
                    input_shapes=[tuple(s.shape) for s in program.inputs])
    args, ref = _inputs(program)
    out = kernel(*args)
    out = out if isinstance(out, tuple) else (out,)
    want = [ref[o] for o in {**OUTPUTS, **FUSED}[name]]
    compare_outputs([a.reshape(w.shape) for a, w in zip(out, want)], want, TOL, what=name)


@pytest.mark.parametrize("name", sorted(OUTPUTS))  # single-group programs
def test_matches_autograd(name):
    program = A.program(name)
    plan = list(schedule(GraphSet.identity(program), ScheduleConfig(topk=1)))[0]
    assert plan.n_threads == 128
    _check(name, program, plan)


def test_in_graph_d_dq_with_register_do_matches_autograd():
    # [1, 128, 64] x 128 threads: dO is loaded once per Q block and read by a gemm and the
    # row sum (failed TileLang's ReduceOp lowering while dO was held in registers)
    program = A.program("dq_o")
    plan = next(p for p in schedule(GraphSet.identity(program), ScheduleConfig(topk=40))
                if p.output_tile[1] == 128 and p.n_threads == 128)
    _check("dq_o", program, plan)


@pytest.mark.parametrize("threads", [128, 256])
def test_fused_with_hoisted_row_sum_matches_autograd(threads):
    # dQ intra, dK/dV inter: D = sum_D(dO * o) is evaluated on the first KV iteration only
    # (tests/codegen/test_loop_invariants.py); later iterations read it from shared memory.
    # Q block 128: TileLang 0.1.13's register-A gemm needs M = 128 at 256 threads (P8b).
    program = A.program("fused_o")
    plans = [p for p in schedule(GraphSet.identity(program), ScheduleConfig(topk=1000))
             if p.n_threads == threads and p.output_tile[1] == 128
             and "_hoisted" in build_candidate_code(p.tile)]
    assert plans
    _check("fused_o", program, plans[-1])
