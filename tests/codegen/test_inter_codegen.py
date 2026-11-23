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


def inter_pair_batched(H=2, M=256, N=256, K=64):
    """``inter_pair`` with a batch dim H on every tensor."""
    p = Program("inter_pair_batched", dtype=cl.f16)
    p.dims(H=H, M=M, N=N, K=K)
    x, w = p.input("x", ("H", "M", "K")), p.input("w", ("H", "N", "K"))
    u, x2 = p.input("u", ("H", "N", "K")), p.input("x2", ("H", "M", "K"))
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

    # declared for the runtime, with the output's own dtype and the buffer's layout
    assert f'INTER_OUTPUTS = {{\n    "{inter_out.name}": "float16",\n}}' in code
    assert f'INTER_LAYOUTS = {{\n    "{inter_out.name}": "blocked8x8",\n}}' in code
    # the kernel parameter is the fp32 accumulation buffer; the intra output keeps fp16
    main = _macro(code, "main_kernel")
    assert f'{inter_out.name}: T.Tensor((1024, 64), "float32")' in main
    assert f'{intra_out.name}: T.Tensor((1024, 64), "float16")' in main

    # inside the iteration: a fresh fp32 partial (cleared gemm), then the atomic add of
    # each element into the 8x8-blocked buffer (64-wide rows: block row, column in block)
    body = _macro(code, f"{loop.name}_iter")
    assert f'{t}_global: T.Tensor(({tile[0]}, {tile[1]}), "float32")' in body
    assert tile[1] == 64
    lines = [l.strip() for l in body.splitlines()]
    (add,) = [i for i, l in enumerate(lines) if l == f"for ai0, ai1 in T.Parallel(*{t}.shape):"]
    flat = " ".join(" ".join(lines[add + 1 :]).replace("( ", "(").replace(" )", ")").split())
    assert flat.startswith(
        f"T.atomic_add({t}_global[ai0 // 8 * 8 + ai1 // 8, ai0 % 8 * 8 + ai1 % 8], {t}[ai0, ai1])"
    )
    assert body.count("T.atomic_add(") == 1
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


def test_batched_pair_builds_every_candidate():
    # exp(x2) is read only by the inter output: its batch dim used to get no launch
    # axis ("Expected exactly one launch-axis mapping for iter dim 0"), since the
    # canonical output's batch axis did not include the inter output's own tokens.
    cands = candidates(inter_pair_batched())
    assert set(first_per_assignment(cands)) == {("intra", "inter"), ("inter", "intra")}
    for cand in cands:
        code = build_candidate_code(cand)
        assert code.count("T.atomic_add(") == 1
        # the inter block is the CTA's head: sliced on the batch launch index only
        assert re.search(r"OutputNod\w+\[1 \* i0 : 1 \* i0 \+ 1, 0 : 0 \+ 256, 0 : 0 \+ 64\]", code)


# ---- 8x8-blocked inter buffers -------------------------------------------------------


def _tmap(shape, tile):
    from types import SimpleNamespace

    return SimpleNamespace(shape=shape, tile_size=tile)


@pytest.mark.parametrize(
    "shape, tile, width",
    [
        ((4096, 64), (64, 64), 64),  # dK/dV of the attention backward
        ((1, 4096, 64), (1, 128, 64), 64),  # a batch entry
        ((1024, 128), (8, 128), 128),
        ((1024, 128), (64, 64), None),  # the tile does not span the rows
        ((1024, 64), (4, 64), None),  # tile rows not a multiple of 8
        ((1000, 64), (16, 64), None),  # tile rows do not divide the block
        ((1024, 12), (64, 12), None),  # width not a multiple of 8
        ((1024, 64), (1, 64), None),  # a single row (squeezed to rank 1)
        ((64,), (64,), None),
    ],
)
def test_which_inter_buffers_are_blocked(shape, tile, width):
    from cleave.codegen.kernel import inter_blocked_width

    assert inter_blocked_width(_tmap(shape, tile)) == width


def test_symbolic_inter_buffers_are_not_blocked():
    import sympy

    from cleave.codegen.kernel import inter_blocked_width

    assert inter_blocked_width(_tmap((sympy.Symbol("M"), 64), (64, 64))) is None


def test_unblocked_atomic_add_keeps_the_region_form():
    from cleave.codegen.tensor_codegen import gen_atomic_add

    assert gen_atomic_add("t", "t_global") == "T.atomic_add(t_global, t)\n"


def _blocked_index(width):
    """The (row, col) the emitted blocked atomic add writes element (i, j) to, as a function."""
    from cleave.codegen.tensor_codegen import gen_atomic_add

    code = gen_atomic_add("t", "g", width, 2)
    assert code.startswith("for ai0, ai1 in T.Parallel(*t.shape):\n")
    index = code.split("g[", 1)[1].rsplit("], t[ai0, ai1])", 1)[0]
    return lambda i, j: eval(f"({index})", {"ai0": i, "ai1": j})


@pytest.mark.parametrize("width", [64, 128, 24])
def test_blocked_index_is_the_buffer_layout_relative_to_any_tile(width):
    """For a tile of whole rows starting at a multiple of 8 (row r0), the view index the
    kernel writes element (i, j) to is the blocked position of (r0 + i, j) in the plane."""
    index = _blocked_index(width)
    rows, tile_rows = 64, 16
    for r0 in range(0, rows, tile_rows):
        seen = set()
        for i in range(tile_rows):
            for j in range(width):
                a, b = index(i, j)
                assert 0 <= b < width and 0 <= a < tile_rows
                m, n = r0 + i, j
                blocked = (m // 8) * 8 * width + (n // 8) * 64 + (m % 8) * 8 + n % 8
                assert (r0 + a) * width + b == blocked
                seen.add((a, b))
        assert len(seen) == tile_rows * width


def test_output_layout_follows_its_loop_accumulator():
    from types import SimpleNamespace

    from cleave.codegen.kernel import inter_output_layout
    from cleave.ir.graph import AccumEdge

    acc = object()
    loop = SimpleNamespace(graph=SimpleNamespace(tmap={acc: _tmap((1024, 64), (4, 64))}))
    edge = AccumEdge.__new__(AccumEdge)
    edge.src_node, edge.accum_node = loop, acc
    out = SimpleNamespace(inputs=[edge])
    assert inter_output_layout(out) is None
    loop.graph.tmap[acc] = _tmap((1024, 64), (8, 64))
    assert inter_output_layout(out) == "blocked8x8"
