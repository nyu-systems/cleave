"""Sibling outputs of one launch that read a shared value through their own gemms (CPU).

``a = e @ v`` and ``b = e @ u`` with ``e = exp(x @ w * c)`` (the shape of a per-output
merge that shares ``e``): the canonical output is ``a``; ``u`` is read only by ``b``.
"""

import re

import cleave.lang as cl
from cleave.codegen import passes
from cleave.frontend import Program
from cleave.ir.graph import OutputNode, PlaceHolderNode
from cleave.pipeline import GraphSet
from cleave.pipeline.plans import lower
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling


def sibling_gemms(M=128, K=64, N=256, D=64):
    p = Program("sibling_gemms", dtype=cl.f16)
    p.dims(M=M, K=K, N=N, D=D)
    x, w = p.input("x", ("M", "K")), p.input("w", ("K", "N"))
    v, u = p.input("v", ("N", "D")), p.input("u", ("N", "D"))
    e = cl.exp(x @ w * 0.125)
    p.output(e @ v, "a")
    p.output(e @ u, "b")
    return p


def _candidates():
    (graph,) = lower(GraphSet.identity(sibling_gemms())[0])
    return search_tiling(graph, topk=1000)


def test_sibling_operand_follows_the_canonical_launch_axes():
    # Tiling D (output tile [.., 32] of 64) launches CTAs along D; u's D dim had no
    # launch-axis dependency (only a's ancestors were mapped) and apply_tiling raised
    # "Expected exactly one launch-axis mapping for iter dim 1".
    cands = [c for c in _candidates() if list(c.output_tile)[-1] < 64]
    assert cands
    for cand in cands:
        code = build_candidate_code(cand)
        main = code[code.index("def main_kernel(") :]
        # v and u are sliced along D by the same launch coordinate as both outputs
        assert len(re.findall(r"32 \* (i\d) : 32 \* \1 \+ 32", main)) == 4, main


def test_outputs_are_the_trailing_kernel_parameters():
    # A gemm whose step spans its whole reduction is stored straight from the main graph,
    # so its OutputNode came before u in topological order: the kernel's parameter list
    # was (x, w, v, a, u, b) while the runtime takes the last two parameters as outputs.
    cands = _candidates()
    full = [c for c in cands if any(256 in r.values() for r in c.rstep_map.values())]
    assert full
    for cand in cands:
        graph = passes.annotate_args(apply_tiling(cand), is_main=True)
        kinds = [isinstance(n, OutputNode) for n in graph.args]
        assert kinds == [False] * 4 + [True] * 2, [n.name for n in graph.args]
        assert all(isinstance(n, PlaceHolderNode) for n in graph.args[:4])
