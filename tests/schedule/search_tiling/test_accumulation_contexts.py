"""Accumulation contexts: outputs that reduce over orthogonal axes (CPU).

The attention backward sums dK and dV over SQ and dQ over SKV. A CTA owns a tile along one
axis and loops along the other, so only one of the two reduction groups accumulates in the
CTA ("intra"); the other is summed across CTAs ("inter"), its tile following the loop
variable. The policy searches every assignment and ranks them together; ``apply_tiling``
builds the shared loop; code generation of an inter output is not implemented and refuses.
Every other graph has no orthogonal groups and is scheduled as before.
"""

import pytest

import cleave.lang as cl
from cleave.core.errors import ScheduleRejected
from cleave.core.target import current_target
from cleave.frontend import Program
from cleave.ir.analysis import find_orthogonal_reduction_groups
from cleave.ir.graph import GraphNode, OutputNode
from cleave.ir.ops import Matmul
from cleave.pipeline import GraphSet, schedule
from cleave.pipeline.plans import describe_tile, lower
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling
from cleave.schedule.policy import default as policy_module
from cleave.schedule.policy.default import DefaultPolicy, TCPolicy

from tests.demo.test_demos import CASES as DEMO_CASES, run_demo
from tests.schedule.end2end.test_end2end import CASES as GRAPH_CASES

C = 64**-0.5


def attention_bwd(H, SQ, SKV, D=64, outputs=("dQ", "dK", "dV"), scale_first=False):
    """dQ, dK, dV of softmax attention with ``D = rowsum(dO * o)`` given (``Dl``)."""
    p = Program("attn_bwd", dtype=cl.f16)
    p.dims(H=H, SQ=SQ, SKV=SKV, D=D)
    q = p.input("q", ("H", "SQ", "D"))
    k = p.input("k", ("H", "SKV", "D"))
    v = p.input("v", ("H", "SKV", "D"))
    dO = p.input("dO", ("H", "SQ", "D"))
    L = p.input("L", ("H", "SQ", 1), dtype=cl.f32)
    Dl = p.input("Dl", ("H", "SQ", 1), dtype=cl.f32)
    P = cl.exp((q @ k.mT) * C - L)
    dS = P * ((dO @ v.mT) - Dl)
    if scale_first:  # `c * dS @ k`, as Python parses it
        outs = dict(dQ=(dS * C) @ k, dK=(dS * C).mT @ q, dV=P.mT @ dO)
    else:
        outs = dict(dQ=(dS @ k) * C, dK=(dS.mT @ q) * C, dV=P.mT @ dO)
    for name in outputs:
        p.output(outs[name])
    return p


def pair(M, N, K=64):
    """S = x @ w^T; Y1 = S @ u sums over N, Y2 = S^T @ x2 sums over M."""
    p = Program("pair", dtype=cl.f16)
    p.dims(M=M, N=N, K=K)
    x, w = p.input("x", ("M", "K")), p.input("w", ("N", "K"))
    u, x2 = p.input("u", ("N", "K")), p.input("x2", ("M", "K"))
    S = x @ w.mT
    p.output(S @ u)
    p.output(S.mT @ x2)
    return p


def pair_shared(M, K=64):
    """Like ``pair`` but square and fed by x, w only (Y1 = S @ w, Y2 = S^T @ x)."""
    p = Program("pair_shared", dtype=cl.f16)
    p.dims(M=M, K=K)
    x, w = p.input("x", ("M", "K")), p.input("w", ("M", "K"))
    S = x @ w.mT
    p.output(S @ w)
    p.output(S.mT @ x)
    return p


def sum_mm(M=256, N=256, K1=128, K2=256):
    """Y = a @ b + c @ d: sibling reductions over axes of different lengths."""
    p = Program("sum_mm", dtype=cl.f16)
    p.dims(M=M, N=N, K1=K1, K2=K2)
    a, b = p.input("a", ("M", "K1")), p.input("b", ("K1", "N"))
    c, d = p.input("c", ("M", "K2")), p.input("d", ("K2", "N"))
    p.output(a @ b + c @ d)
    return p


ORTHOGONAL = {
    "attn_bwd_square": lambda: attention_bwd(32, 4096, 4096),
    "attn_bwd_sq_lt_skv": lambda: attention_bwd(32, 2048, 4096),
    "attn_bwd_sq_gt_skv": lambda: attention_bwd(32, 4096, 2048),
    "attn_bwd_scale_first": lambda: attention_bwd(2, 256, 512, scale_first=True),
    "pair": lambda: pair(1024, 512),
    "pair_shared": lambda: pair_shared(512),
}


def _lower(program):
    (graph,) = lower(GraphSet.identity(program)[0])
    return graph


def _policy(graph, force_fusion=False):
    """The policy ``search_tiling`` builds for ``graph``."""
    outputs = [n for n in graph.nodes if isinstance(n, OutputNode)]
    cls = TCPolicy if any(isinstance(n, Matmul) for n in graph.nodes) else DefaultPolicy
    return cls(outputs, current_target().arch, force_fusion=force_fusion, flat_graph=graph)


def _groups(graph):
    """Each reduction group as the sorted names of the OutputNodes it feeds."""
    policy = _policy(graph)
    by_src = {n.inputs[0].src_node: n.name for n in graph.nodes if isinstance(n, OutputNode)}
    return sorted(sorted(by_src[o] for o in g.outputs) for g in policy.reduction_groups)


def _output_names(graph):
    return [n.name for n in graph.nodes if isinstance(n, OutputNode)]


def _contexts(cand):
    """Context per OutputNode name, in graph order."""
    outputs = [n for n in cand.flat_graph.nodes if isinstance(n, OutputNode)]
    return {o.name: cand.accumulation[o.inputs[0].src_node] for o in outputs}


# ---- detection ---------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GRAPH_CASES))
def test_detection_silent_on_test_graphs(name):
    build, dims, _, _ = GRAPH_CASES[name]
    policy = _policy(build(*dims))
    assert policy.reduction_groups == []
    assert policy.accumulation == {}


@pytest.mark.parametrize("case", DEMO_CASES, ids=[c[0] for c in DEMO_CASES])
def test_detection_silent_on_demos(monkeypatch, case):
    _, demo, argv = case
    program = run_demo(monkeypatch, demo, argv)["program"]
    graph = _lower(program)
    assert _policy(graph).reduction_groups == []
    for cand in search_tiling(graph, topk=4):
        assert cand.accumulation == {}
        assert "acc=" not in describe_tile(cand)


@pytest.mark.parametrize("name", sorted(ORTHOGONAL))
def test_detection_active_on_orthogonal_outputs(name):
    graph = _lower(ORTHOGONAL[name]())
    names = _output_names(graph)
    if name.startswith("attn_bwd"):
        dq, dk, dv = names
        assert _groups(graph) == sorted([[dq], sorted([dk, dv])])
    else:
        assert _groups(graph) == sorted([[n] for n in names])


@pytest.mark.parametrize("outputs", [("dQ",), ("dK", "dV"), ("dV",)])
def test_detection_silent_on_single_group_backward(outputs):
    assert _groups(_lower(attention_bwd(2, 256, 512, outputs=outputs))) == []


def test_detection_needs_orthogonal_axes():
    # two outputs reducing over unrelated axes are not orthogonal: neither keeps the
    # other's reduction axis
    p = Program("independent", dtype=cl.f16)
    p.dims(M=128, N=128, K1=64, K2=128)
    a, b = p.input("a", ("M", "K1")), p.input("b", ("K1", "N"))
    c, d = p.input("c", ("M", "K2")), p.input("d", ("K2", "N"))
    p.output(a @ b)
    p.output(c @ d)
    graph = _lower(p)
    policy = _policy(graph)
    assert find_orthogonal_reduction_groups(
        policy.ordered_nodes,
        policy.output_nodes,
        policy.output_axis_provenance,
        policy.reduction_axis_provenance,
    ) == []


# ---- policy ------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ORTHOGONAL))
def test_policy_enumerates_every_assignment(name):
    graph = _lower(ORTHOGONAL[name]())
    groups = _groups(graph)
    cands = search_tiling(graph, topk=10000)
    assert cands
    seen = set()
    for cand in cands:
        contexts = _contexts(cand)
        intra = sorted(n for n, c in contexts.items() if c == "intra")
        assert set(contexts.values()) == {"intra", "inter"}
        assert intra in groups  # exactly one group is intra
        seen.add(tuple(intra))
        acc = ",".join(contexts[n] for n in _output_names(graph))
        assert f" acc=[{acc}]" in describe_tile(cand)
    assert seen == {tuple(g) for g in groups}
    # ranked together, by the policy's priority
    prios = [(c.traffic + 1) * c.num_wave for c in cands]
    assert prios == sorted(prios)


def test_atomic_adds_are_counted(monkeypatch):
    H, SQ, SKV, D = 32, 2048, 4096, 64
    graph = _lower(attention_bwd(H, SQ, SKV, D))
    dq = _output_names(graph)[0]

    def traffic(weight):
        monkeypatch.setattr(policy_module, "ATOMIC_COST_WEIGHT", weight)
        return {
            (tuple(_contexts(c).values()), tuple(c.output_tile), describe_tile(c)): c.traffic
            for c in search_tiling(graph, topk=10000)
        }

    with_atomics, without = traffic(2), traffic(0)
    assert with_atomics.keys() == without.keys()
    for (contexts, _, label), t in with_atomics.items():
        # per CTA every element of an inter output's block (all of its non-batch dims)
        # is added once in fp32; the block is SQ x D for dQ, SKV x D for dK and dV
        dq_inter = contexts[0] == "inter"
        inter_elems = SQ * D if dq_inter else 2 * SKV * D
        assert t - without[(contexts, _, label)] == 2 * 4 * inter_elems, (dq, label)


def test_inter_output_launch_dims_stay_whole():
    # the D dim of the intra output is no inter reduction's axis: splitting it across
    # CTAs would add the same dQ partial twice
    graph = _lower(attention_bwd(32, 2048, 4096))
    for cand in search_tiling(graph, topk=10000):
        assert int(cand.output_tile[-1]) == 64


# ---- apply_tiling ------------------------------------------------------------------


def _first_per_assignment(cands):
    firsts = {}
    for cand in cands:
        firsts.setdefault(tuple(_contexts(cand).values()), cand)
    return list(firsts.values())


def _iterated(it):
    return [(d, int(s) // int(t)) for d, (s, t) in enumerate(zip(it.shape, it.tile_size)) if int(s) // int(t) > 1]


@pytest.mark.parametrize("name", sorted(ORTHOGONAL))
def test_inter_output_follows_loop(name):
    graph = _lower(ORTHOGONAL[name]())
    for cand in _first_per_assignment(search_tiling(graph, topk=10000)):
        contexts = _contexts(cand)
        inter_feeds = {
            o.inputs[0].src_node.name
            for o in cand.flat_graph.nodes
            if isinstance(o, OutputNode) and contexts[o.name] == "inter"
        }
        tiled = apply_tiling(cand)

        assert {o.name: c for o, c in tiled.accumulation.items()} == contexts
        (loop,) = [n for n in tiled.nodes if isinstance(n, GraphNode)]
        inner = loop.graph
        assert {n.name for n, c in inner.accumulation.items() if c == "inter"} == inter_feeds
        assert set(inner.accumulation) == set(inner.accum_nodes)

        for node in inner.accum_nodes:
            it = inner.tmap[node]
            if inner.accumulation[node] == "intra":
                assert _iterated(it) == []  # one tile, accumulated over the loop
                continue
            # the iteration tile is sliced with the loop counter (arg 0 of the loop)
            ((dim, count),) = _iterated(it)
            assert count == int(inner.extent)
            terms = it.offset.terms
            assert terms[dim] is not None and terms[dim].base == 0
            assert terms[dim].scale == int(it.tile_size[dim])
            assert all(t is None for d, t in enumerate(terms) if d != dim)

        for out, context in tiled.accumulation.items():
            if context != "inter":
                continue
            # the block of an inter output is the whole tensor but for the batch entry:
            # no launch-axis mapping on any other dim
            it = tiled.tmap[out]
            flags = list(out.is_batch_dim)
            for d, (s, t, term) in enumerate(zip(it.shape, it.tile_size, it.offset.terms)):
                if flags[d]:
                    assert int(t) == 1 and (term is not None) == (int(s) > 1)
                else:
                    assert int(t) == int(s) and term is None


@pytest.mark.parametrize("name", sorted(ORTHOGONAL))
def test_codegen_emits_inter_output(name):
    # Every assignment builds; its inter outputs are atomically added (P5b), and the
    # module declares them for the runtime (tests/codegen/test_inter_codegen.py: where).
    graph = _lower(ORTHOGONAL[name]())
    for cand in _first_per_assignment(search_tiling(graph, topk=10000)):
        code = build_candidate_code(cand)
        n_inter = list(_contexts(cand).values()).count("inter")
        assert code.count("T.atomic_add(") == n_inter
        assert "INTER_OUTPUTS = {" in code


def test_schedule_labels_show_assignment():
    plans = schedule(GraphSet.identity(attention_bwd(32, 2048, 4096)))
    assert not plans.refused
    labels = {p.label.split(" acc=")[1].split(" ")[0] for p in plans}
    assert labels == {"[inter,intra,intra]", "[intra,inter,inter]"}


# ---- loop merge compares axes ------------------------------------------------------


def test_merge_refuses_sibling_over_another_axis(monkeypatch):
    # Without accumulation contexts pair_shared is tiled as one output and Y2's
    # reduction (over M) shares S with Y1's loop (over N). It used to be merged into that
    # loop, its axis sliced with Y1's counter; TileLang then failed on the gemm shapes.
    monkeypatch.setattr(policy_module, "find_orthogonal_reduction_groups", lambda *a: [])
    graph = _lower(pair_shared(512))
    cands = search_tiling(graph, topk=8)
    assert cands and all(c.accumulation == {} for c in cands)
    for cand in cands:
        with pytest.raises(ScheduleRejected, match="runs over another axis"):
            apply_tiling(cand)


def test_merge_refuses_same_step_over_shorter_axis():
    # a @ b (K1 = 128) and c @ d (K2 = 256) share no node and no input. With the same
    # step they used to be merged into one loop of K2 / step iterations: a @ b read
    # past K1 (NaN in the output). Different trip counts are refused.
    graph = _lower(sum_mm())
    seen = 0
    for cand in search_tiling(graph, topk=10000):
        steps = [
            (int(next(iter(r.values()))), int(n.raxis[0].dom.extent))
            for n, r in cand.rstep_map.items()
            if r
        ]
        if len({s for s, _ in steps}) != 1 or any(s >= e for s, e in steps):
            continue  # different steps, or one of them does not loop
        seen += 1
        with pytest.raises(ScheduleRejected, match="steps of"):
            apply_tiling(cand)
    assert seen
