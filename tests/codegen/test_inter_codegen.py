"""Inter outputs (summed across CTAs) in the generated kernel text (CPU).

An inter output's loop accumulator is a fresh partial each iteration (cleared, fp32),
atomically added into an fp32 global buffer at the CTA's block + the loop offset; the
main graph does not copy it after the loop, and the module declares it in
``INTER_OUTPUTS`` for the runtime, which zeroes the buffer and casts it
(``tests/runtime/test_inter_outputs.py`` runs it).
"""

import re

import pytest

import cleave.lang as cl
from cleave.codegen import TilelangCodeGen, passes
from cleave.core.errors import CodegenError
from cleave.frontend import Program
from cleave.ir.graph import GraphNode, OutputNode
from cleave.pipeline import GraphSet
from cleave.pipeline.plans import lower
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling


def inter_pair(M=1024, N=1024, K=64):
    """S = x @ w^T; Y1 = S @ u sums over N; Y2 = S^T @ exp(x2) sums over M.

    The two outputs reduce over orthogonal axes, so one of them is summed across CTAs.
    Y2's gemm reads two computed operands, which the existing two-fragment rule stages
    in shared memory: no transposed register operand.
    """
    p = Program("inter_pair", dtype=cl.f16)
    p.dims(M=M, N=N, K=K)
    x, w = p.input("x", ("M", "K")), p.input("w", ("N", "K"))
    u, x2 = p.input("u", ("N", "K")), p.input("x2", ("M", "K"))
    S = x @ w.mT
    p.output(S @ u)
    p.output(S.mT @ cl.exp(x2))
    return p


def candidates(program, topk=1000):
    (graph,) = lower(GraphSet.identity(program)[0])
    return search_tiling(graph, topk=topk)


def contexts(cand):
    outputs = [n for n in cand.flat_graph.nodes if isinstance(n, OutputNode)]
    return tuple(cand.accumulation.get(o.inputs[0].src_node, "intra") for o in outputs)


def first_per_assignment(cands):
    firsts = {}
    for cand in cands:
        firsts.setdefault(contexts(cand), cand)
    return firsts


def annotated(cand):
    """The tiled graph as ``build_candidate_code`` hands it to the code generator."""
    g = apply_tiling(cand)
    g = passes.annotate_args(g, is_main=True)
    g = passes.annotate_tinfo(g, is_main=True)
    g = passes.annotate_tc_layout(g)
    return passes.annotate_fragment_layout_conflicts(g)


def _macro(code, name):
    """The text of ``def name(...)`` up to the next top-level definition."""
    start = code.index(f"def {name}(")
    end = code.find("\n@T.", start)
    return code[start : None if end < 0 else end]


@pytest.fixture(scope="module")
def pair_cands():
    return first_per_assignment(candidates(inter_pair()))


def test_both_assignments_are_offered(pair_cands):
    assert set(pair_cands) == {("intra", "inter"), ("inter", "intra")}


@pytest.mark.parametrize("assignment", [("intra", "inter"), ("inter", "intra")])
def test_inter_output_is_atomically_added_per_iteration(pair_cands, assignment):
    cand = pair_cands[assignment]
    g = annotated(cand)
    code = build_candidate_code(cand)
    (inter_out,) = [o for o, c in g.accumulation.items() if c == "inter"]
    (intra_out,) = [o for o, c in g.accumulation.items() if c == "intra"]
    (loop,) = [n for n in g.nodes if isinstance(n, GraphNode)]
    (acc,) = [n for n, c in loop.graph.accumulation.items() if c == "inter"]
    t = loop.graph.tinfo_map[acc][0]
    block = [int(e) for e in loop.graph.tmap[acc].shape]
    tile = [int(e) for e in loop.graph.tmap[acc].tile_size]

    # declared for the runtime, with the output's own dtype
    assert f'INTER_OUTPUTS = {{\n    "{inter_out.name}": "float16",\n}}' in code
    # the kernel parameter is the fp32 accumulation buffer; the intra output keeps fp16
    main = _macro(code, "main_kernel")
    assert f'{inter_out.name}: T.Tensor((1024, 64), "float32")' in main
    assert f'{intra_out.name}: T.Tensor((1024, 64), "float16")' in main

    # inside the iteration: a fresh fp32 partial (cleared gemm), then the atomic add
    body = _macro(code, f"{loop.name}_iter")
    assert f'{t}_global: T.Tensor(({tile[0]}, {tile[1]}), "float32")' in body
    lines = [l.strip() for l in body.splitlines()]
    (add,) = [i for i, l in enumerate(lines) if l.startswith("T.atomic_add(")]
    assert lines[add] == f"T.atomic_add({t}_global, {t})"
    assert re.match(rf"ukernel_mma_\w*_clear\(.*, {t}\)$", lines[add - 1])
    assert body.count(f"{t} = T.alloc_fragment(") == 1
    assert '"float32"' in body[body.index(f"{t} = T.alloc_fragment(") :].split(")", 2)[1]

    # the loop slices the CTA's block with the loop counter: tile rows at i0 * tile
    wrapper = _macro(code, loop.name)
    assert f'{t}_global: T.Tensor(({block[0]}, {block[1]}), "float32")' in wrapper
    step = tile[0]
    assert (
        f"{t}_global=_S({t}_global[{step} * i0 : {step} * i0 + {step}, 0 : 0 + {tile[1]}], "
        "squeeze_dim=[])"
    ) in wrapper

    # main hands its buffer to the loop and copies nothing into it afterwards
    main_iter = _macro(code, "main_kernel_iter")
    call = main_iter[main_iter.index(f"{loop.name}(") :].split(")", 1)[0]
    assert inter_out.name in call
    assert f"T.clear({t})" not in main_iter
    assert not re.search(rf"T\.copy\([^)]*, {inter_out.name}\)", main_iter)
    assert re.search(rf"T\.copy\([^)]*, {intra_out.name}\)", main_iter)


def test_programs_without_contexts_emit_no_inter_code():
    p = Program("mm", dtype=cl.f16)
    p.dims(M=256, N=256, K=128)
    a, b = p.input("a", ("M", "K")), p.input("b", ("K", "N"))
    p.output(a @ b)
    for cand in candidates(p, topk=4):
        assert not cand.accumulation
        code = build_candidate_code(cand)
        assert "INTER_OUTPUTS" not in code and "atomic_add" not in code


def test_refuses_unknown_context(pair_cands):
    g = annotated(pair_cands[("intra", "inter")])
    out = next(iter(g.accumulation))
    g.accumulation[out] = "bogus"
    with pytest.raises(CodegenError, match="unknown accumulation contexts"):
        TilelangCodeGen().gen(g)


def test_refuses_inter_output_its_loop_does_not_sum_across_ctas(pair_cands):
    g = annotated(pair_cands[("intra", "inter")])
    (intra_out,) = [o for o, c in g.accumulation.items() if c == "intra"]
    g.accumulation[intra_out] = "inter"
    with pytest.raises(CodegenError, match="does not sum across CTAs"):
        TilelangCodeGen().gen(g)


def test_refuses_inter_output_in_split_kernel(pair_cands):
    g = annotated(pair_cands[("intra", "inter")])
    with pytest.raises(CodegenError, match="split kernels"):
        TilelangCodeGen().gen([g, g])
