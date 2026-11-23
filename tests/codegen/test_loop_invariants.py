"""Loop-invariant ops of a kernel with accumulation contexts are evaluated once.

In the fused attention backward with ``D = sum_D(dO * o)`` in the graph, a CTA owning a Q block
loops over KV blocks (dQ intra, dK/dV inter); D depends only on the Q block's dO and o tiles,
so it is computed on the first iteration into a shared buffer allocated before the loop
(``Graph.invariant_nodes``, ``codegen.operands.hoisted_nodes``). Owning a KV block and looping
over Q blocks, D depends on the loop counter and stays in the loop body.
"""

import re

import pytest

from cleave.ir.graph import GraphNode
from cleave.ir.ops import ReduceSum
from cleave.pipeline import GraphSet, ScheduleConfig, schedule
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling
from tests.lower import _attention_bwd as A


@pytest.fixture(scope="module")
def plans():
    return list(schedule(GraphSet.identity(A.program("fused_o")), ScheduleConfig(topk=1000)))


def _contexts(plan):
    return plan.label.split("acc=")[1].split(" ")[0]


def _iter_body(code, loop):
    """Lines of ``def <loop>_iter(...)``'s body (after the signature)."""
    lines = code.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith(f"def {loop}_iter("))
    start = next(i for i in range(start, len(lines)) if lines[i].startswith("):")) + 1
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("@"))
    return lines[start:end]


def _loop_name(code):
    (name,) = set(re.findall(r"^def (GraphNode_\w+)_iter\(", code, re.M))
    return name


def test_kv_loop_computes_the_row_sum_once(plans):
    picked = [p for p in plans if _contexts(p) == "[intra,inter,inter]"]
    assert picked
    for plan in picked:
        code = build_candidate_code(plan.tile)
        loop = _loop_name(code)
        body = _iter_body(code, loop)
        guard = body.index("    if i0 == 0:")
        end = next(i for i in range(guard + 1, len(body)) if not body[i].startswith("        "))
        guarded = body[guard + 1:end]
        # D's product and row sum (and nothing else) run under the first-iteration guard
        calls = [l.split("(")[0].strip() for l in guarded if "ukernel_" in l]
        assert calls == ["ukernel_mul_2dims_ssr", "ukernel_reduction_axis2_rr_clear"], plan.label
        assert any(re.match(r"\s+T\.copy\(t_reduction_\w+, t_reduction_\w+_hoisted\)", l)
                   for l in guarded)
        # every gemm and the subtraction reading D are loop-dependent: unguarded, every iteration
        rest = body[:guard] + body[end:]
        assert sum("ukernel_mma" in l for l in rest) == 5
        (sub,) = [l for l in rest if "ukernel_sub_broadcast_0" in l and "t_reduction" in l]
        assert re.search(r"t_reduction_\w+_hoisted\b", sub)
        assert not any("_intermediate" in l for l in rest if "t_reduction" in l)
        # the buffer is allocated in the loop's wrapper, before the loop, and passed in
        wrapper = code.split(f"def {loop}(")[1]
        alloc = re.search(r"t_reduction_\w+_hoisted = T\.alloc_shared", wrapper)
        assert alloc and alloc.start() < wrapper.index("for i0 in T.Pipelined(")
        assert "i0=i0," in wrapper


def test_q_loop_keeps_the_row_sum_in_the_body(plans):
    picked = [p for p in plans if _contexts(p) == "[inter,intra,intra]"]
    assert picked
    for plan in picked:
        code = build_candidate_code(plan.tile)
        assert "_hoisted" not in code and "if i0 == 0:" not in code, plan.label
        body = _iter_body(code, _loop_name(code))
        assert any(l.startswith("    ukernel_mul_2dims_ssr(") for l in body)


def test_invariant_nodes_of_the_tiled_graph(plans):
    by_contexts = {}
    for plan in plans:
        by_contexts.setdefault(_contexts(plan), plan)
    for contexts, plan in by_contexts.items():
        (gnode,) = [n for n in apply_tiling(plan.tile).nodes if isinstance(n, GraphNode)]
        names = [type(n).__name__ for n in gnode.graph.invariant_nodes]
        if contexts == "[intra,inter,inter]":
            assert names == ["Mul", "ReduceSum"]
            mul, red = gnode.graph.invariant_nodes
            assert red.inputs[0].src_node is mul and isinstance(red, ReduceSum)
        else:
            assert names == []


def test_kernels_without_accumulation_contexts_are_unchanged():
    # dq_o alone: D is invariant in its KV loop too, but the hoist is limited to kernels with
    # accumulation contexts, so every existing kernel keeps its text.
    for plan in schedule(GraphSet.identity(A.program("dq_o")), ScheduleConfig(topk=40)):
        tiled = apply_tiling(plan.tile)
        assert all(not n.graph.invariant_nodes for n in tiled.nodes if isinstance(n, GraphNode))
        assert "_hoisted" not in build_candidate_code(plan.tile)
