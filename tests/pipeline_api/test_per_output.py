"""Per-output search and merge (``SuperoptConfig(per_output=...)``), CPU.

A program with an ``equals=`` input and several outputs is searched one output at a time
(``Program.for_output``), and one graph per output is merged into a multi-output structure over
the parent's inputs with common sub-expressions shared. Every other program is searched whole.
"""

import json

import pytest

import cleave.lang as cl
from cleave import pipeline as P
from cleave.core.errors import SpecError
from cleave.frontend.program import Program
from cleave.lower import to_ir
from cleave.lower.split import _renumber_graph
from cleave.pipeline import per_output as po
from cleave.pipeline import superopt
from cleave.pipeline.plans import bind_inputs
from cleave.runtime import reference as rt
from tests.frontend.test_declared_inputs import builder as attention_bwd

torch = pytest.importorskip("torch")


def _names(p):
    return [s.name for s in p.inputs]


# ------------------------------------------------------------------ sub-programs


def test_for_output_keeps_what_the_output_reads():
    p = attention_bwd()
    dq, kept = p.for_output("dQ")
    assert kept == (0, 1, 2, 3, 4, 5) and _names(dq) == ["q", "k", "v", "dO", "L", "o"]
    assert dq.inputs[5].equals is p.inputs[5].equals and dq.inputs[5].over == (0, 1, 2, 4)
    assert dq.inputs[4].sample is p.inputs[4].sample and dq.inputs[4].over == (0, 1)
    assert [o.name for o in dq.outputs] == ["dQ"]
    # dV reads neither v nor o (o's source v is gone, so o is dropped); L's sample stays
    dv, kept = p.for_output(2)
    assert kept == (0, 1, 3, 4) and _names(dv) == ["q", "k", "dO", "L"]
    assert dv.inputs[3].sample is not None and dv.inputs[3].over == (0, 1)
    assert dv.op_count == 4 + 5  # S, * c, - L, exp, P^T @ dO
    # the auto budget is the sub-program's own
    assert P.SuperoptConfig().resolve_max_ops(dv) == 13
    assert P.SuperoptConfig().resolve_max_ops(dq) == 6 + 11 + 2


def test_for_output_is_the_program_of_that_output_alone():
    """Same fingerprint as the one-output program written directly (shared cache entries)."""
    import tests.frontend.test_declared_inputs as t

    p = Program("x", cl.f16)
    h, sq, skv, d = p.dims(**t.SHAPES)
    q, k = p.input("q", (h, sq, d)), p.input("k", (h, skv, d))
    dO = p.input("dO", (h, sq, d))
    L = p.input("L", (h, sq, 1), cl.f32, sample=t.log_sum_exp)
    p.output(cl.exp(q @ k.mT * t.C - L).mT @ dO, "dV")
    sub, _ = attention_bwd().for_output("dV")
    assert sub.fingerprint() == p.fingerprint()
    x = attention_bwd().random_inputs(seed=1, device="cpu")
    want = attention_bwd().reference(*x)[2]
    torch.testing.assert_close(sub.reference(x[0], x[1], x[3], x[4]), [want], rtol=0, atol=0)


def _strip_program():
    """y = a + b declared; out0 = y * a reads y but not b: y becomes a free input there."""
    p = Program("strip", cl.f16)
    m, n = p.dims(M=16, N=32)
    a, b = p.input("a", (m, n)), p.input("b", (m, n))
    y = p.input("y", (m, n), equals=lambda a, b: a + b)
    p.output(y * a, "out0")
    p.output(a - b, "out1")
    return p


def test_for_output_frees_a_declared_input_without_its_sources():
    p = _strip_program()
    s0, kept0 = p.for_output("out0")
    assert kept0 == (0, 2) and _names(s0) == ["a", "y"]
    assert s0.inputs[1].equals is None and s0.inputs[1].over == ()
    s1, kept1 = p.for_output("out1")
    assert kept1 == (0, 1, 2) and s1.inputs[2].equals is not None  # unread, sources kept
    s0.to_tile_graph(), s1.to_tile_graph()  # both searchable


# ------------------------------------------------------------------ the auto rule


def _two_outputs(declared, n_outputs=2, kind="equals"):
    p = Program("two", cl.f16)
    m, n = p.dims(M=16, N=32)
    a = p.input("a", (m, n))
    y = p.input("y", (m, n), **({kind: lambda a: a * 2.0} if declared else {}))
    p.output(cl.exp(a) * y, "o0")
    if n_outputs > 1:
        p.output(a * y, "o1")
    return p


def test_auto_rule():
    auto = P.SuperoptConfig()
    assert auto.per_output == "auto"
    assert auto.resolve_per_output(_two_outputs(True))
    assert not auto.resolve_per_output(_two_outputs(True, n_outputs=1))
    assert not auto.resolve_per_output(_two_outputs(False))
    assert not auto.resolve_per_output(_two_outputs(True, kind="sample"))
    assert P.SuperoptConfig(per_output=True).resolve_per_output(_two_outputs(False))
    assert not P.SuperoptConfig(per_output=False).resolve_per_output(_two_outputs(True))
    with pytest.raises(SpecError, match="per_output"):
        P.SuperoptConfig(per_output="yes")


def test_undeclared_programs_take_the_whole_program_path(monkeypatch, tmp_path):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))

    def refuse(*a, **k):
        raise AssertionError("per-output path taken")

    monkeypatch.setattr(superopt, "_superoptimize_per_output", refuse)
    monkeypatch.setattr(Program, "for_output", refuse)
    p = Program("mm", cl.f16)
    m, k, n = p.dims(M=64, K=64, N=64)
    x, w = p.input("x", (m, k)), p.input("w", (k, n))
    p.output(x @ w, "y")
    p.output(x @ w, "y2")
    gs = P.superoptimize(p, P.SuperoptConfig(threads=2, cache=False))
    assert gs.parts == () and all(g.parts == () for g in gs)


def test_per_output_refuses_split():
    with pytest.raises(SpecError, match="split"):
        P.superoptimize(_two_outputs(True), P.SuperoptConfig(split=("M",), cache=False))


# ------------------------------------------------------------------ merge / CSE


def test_combinations_by_index_sum():
    assert list(po.combinations([2, 3])) == [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (1, 2)]
    assert sorted(po.combinations([3, 1, 2])) == sorted(
        (a, 0, c) for a in range(3) for c in range(2))
    assert list(po.combinations([2, 0])) == []


def _identity_parts(p):
    """Every output's un-optimized structure, merged (dims are plain extents: no symbols)."""
    parts, kepts = [], []
    for o in p.outputs:
        sub, kept = p.for_output(o.name)
        parts.append(sub.to_graph_structure())
        kepts.append(kept)
    inputs = [op["output_tensors"][0] for op in p.to_graph_structure()
              if op["op_type"] == "kn_input_op"]
    return _renumber_graph(po.merge(parts, kepts, [{}] * len(parts), inputs))


def _signature(structure):
    """Ops by kind, attrs and operand positions (guid-free)."""
    pos, out = {}, []
    for i, op in enumerate(structure):
        args = [pos[t["guid"]] for t in op["input_tensors"]]
        attrs = {k: v for k, v in op.items() if k not in ("input_tensors", "output_tensors")}
        out.append(json.dumps([attrs, args], sort_keys=True))
        for t in op["output_tensors"]:
            pos[t["guid"]] = i
    return out


def test_merging_the_outputs_alone_recovers_the_shared_program():
    """CSE of the three per-output programs of the backward = the 3-output program: S, P, dP,
    delta and dS are computed once."""
    p = attention_bwd()
    merged = _identity_parts(p)
    assert _signature(merged) == _signature(_renumber_graph(p.to_graph_structure()))
    assert sum(op["op_type"] == "kn_exp_op" for op in merged) == 1
    assert sum(op["op_type"] == "kn_matmul_op" for op in merged) == 5  # S, dP, dQ, dK, dV


def test_cse_shares_commutative_operands_only():
    def part(first, second, trans=False):
        a = {"num_dims": 2, "dim": [8, 8], "guid": 1, "dtype": "float16"}
        b = {"num_dims": 2, "dim": [8, 8], "guid": 2, "dtype": "float16"}
        c = {"num_dims": 2, "dim": [8, 8], "guid": 3}
        x, y = (a, b) if first == "a" else (b, a)
        return [{"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [a]},
                {"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [b]},
                {"op_type": second, "input_tensors": [x, y], "output_tensors": [c],
                 **({"trans_A": trans, "trans_B": False} if second == "kn_matmul_op" else {})},
                {"op_type": "kn_output_op", "input_tensors": [c], "output_tensors": []}]

    inputs = [part("a", "kn_mul_op")[i]["output_tensors"][0] for i in (0, 1)]
    count = lambda parts: sum(  # noqa: E731
        op["op_type"] not in ("kn_input_op", "kn_output_op")
        for op in po.merge(parts, [(0, 1)] * len(parts), [{}] * len(parts), inputs))
    assert count([part("a", "kn_mul_op"), part("b", "kn_mul_op")]) == 1
    assert count([part("a", "kn_sub_op"), part("b", "kn_sub_op")]) == 2
    assert count([part("a", "kn_matmul_op"), part("b", "kn_matmul_op")]) == 2
    assert count([part("a", "kn_matmul_op"), part("a", "kn_matmul_op", trans=True)]) == 2
    assert count([part("a", "kn_matmul_op"), part("a", "kn_matmul_op")]) == 1


def test_symbols_are_translated_through_the_inputs():
    parent = [{"dim": [12, 20], "guid": 7}, {"dim": [20, 28], "guid": 8}]
    part = [{"dim": [5, 6]}]
    assert po.symbol_translation(part, parent, {5: 4, 6: 128}, (1,)) == {5: 20, 6: 28}
    with pytest.raises(Exception, match="maps to"):
        po.symbol_translation([{"dim": [5, 5]}], parent, {5: 4}, (0,))


def test_an_unread_input_stays_an_input():
    p = attention_bwd()
    p_dv, kept = p.for_output("dV")
    inputs = [op["output_tensors"][0] for op in p.to_graph_structure()
              if op["op_type"] == "kn_input_op"]
    merged = po.merge([p_dv.to_graph_structure()], [kept], [{}], inputs)
    assert [op["op_type"] for op in merged[:6]] == ["kn_input_op"] * 6
    read = {t["guid"] for op in merged for t in op["input_tensors"]}
    assert [t["guid"] in read for t in inputs] == [True, True, False, True, True, False]


# ------------------------------------------------------------------ evaluation of merged graphs


def evaluate(graph, inputs):
    """A merged structure, lowered and run by the runtime reference (fp32), on ``inputs``
    (the Program's, in its order); one tensor per output (the kernel's shape)."""
    ir = to_ir(graph.structure, symbol_map=graph.symbol_map, dtype="float32",
               out_dtype="float32")
    names = bind_inputs(ir, graph)
    assert names is not None
    by_name = {n: x.float() for n, x in zip(names, inputs) if n is not None}
    return rt.execute(ir, [by_name[ph.name].reshape(tuple(int(d) for d in ph.get_shape(0)))
                           for ph in rt.placeholder_order(ir)])


def test_identity_merge_evaluates_to_the_reference():
    p = attention_bwd()
    g = superopt.SearchedGraph(0, _identity_parts(p), None, False, p)
    x = p.random_inputs(seed=2, device="cpu")
    want = [y.float() for y in p._evaluate([t.float() for t in x])]
    for got, ref in zip(evaluate(g, x), want, strict=True):
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------ a search, end to end (toy)


def toy():
    """P2's toy (P an input, o = P @ v declared). x = dP - delta, delta = sum_SKV(P dP),
    dV = P^T dO. dV reads neither v nor o."""
    p = Program("toy", cl.f16)
    h, sq, skv, d = p.dims(H=4, SQ=128, SKV=256, D=64)
    pm, v, dO = p.input("P", (h, sq, skv)), p.input("v", (h, skv, d)), p.input("dO", (h, sq, d))
    p.input("o", (h, sq, d), equals=lambda P, v: P @ v)
    dp = dO @ v.mT
    delta = cl.sum(pm * dp, -1, keepdim=True)
    p.output(dp - delta, "x")
    p.output(delta, "delta")
    p.output(pm.mT @ dO, "dV")
    return p


def test_toy_search_merges_per_output_graphs(monkeypatch, tmp_path):
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path))
    p = toy()
    gs = P.superoptimize(p, P.SuperoptConfig(max_ops=8, threads=4))
    assert [len(part) for part in gs.parts] == [1, 2, 1]
    assert all(len(part.program.outputs) == 1 for part in gs.parts)
    assert [g.parts for g in gs] == [(0, 0, 0), (0, 1, 0)]
    # (0, 1, 0): delta's second graph is the one x computes: everything is shared
    assert [g.ops for g in gs] == [["matmul", "mul", "sum", "sub", "matmul", "mul", "sum", "matmul"],
                                   ["matmul", "mul", "sum", "sub", "matmul"]]
    for g in gs:
        assert sum(op["op_type"] == "kn_input_op" for op in g.structure) == len(p.inputs)
        assert g.program is p and not g.is_split and g.symbol_map
    x = p.random_inputs(seed=3, device="cpu")
    want = [y.float() for y in p._evaluate([t.float() for t in x])]
    for g in gs:
        for got, ref in zip(evaluate(g, x), want, strict=True):
            torch.testing.assert_close(got.reshape(ref.shape), ref, rtol=1e-4, atol=1e-4)
    again = P.superoptimize(p, P.SuperoptConfig(max_ops=8, threads=4))
    assert again.from_cache and [g.structure for g in again] == [g.structure for g in gs]


def test_an_unread_declared_input_is_not_derived():
    """The input graph of the backward reads no ``o``: validation still derives ``L`` (which the
    graph reads), and leaves ``o`` (no placeholder) alone."""
    p = attention_bwd()
    g = superopt.SearchedGraph(0, _identity_parts(p), None, False, p)
    ir = to_ir(g.structure, symbol_map=None)
    names = bind_inputs(ir, g)
    assert names[5] is None and None not in names[:5]
    hook = rt.declared_inputs_hook(p, names, ir)
    x = rt.make_inputs(ir, "float16", seed=0, device="cpu", derive=hook)
    by_name = dict(zip([ph.name for ph in rt.placeholder_order(ir)], x))
    q, k, L = (by_name[names[i]] for i in (0, 1, 4))
    import tests.frontend.test_declared_inputs as t
    want = torch.logsumexp(q.float() @ k.float().mT * t.C, -1, keepdim=True)
    torch.testing.assert_close(L.reshape(want.shape), want, rtol=1e-5, atol=1e-5)


def test_a_read_declared_input_needs_its_sources():
    p = _strip_program()  # out0 reads y = a + b but not b
    sub, kept = p.for_output("out0")
    inputs = [op["output_tensors"][0] for op in p.to_graph_structure()
              if op["op_type"] == "kn_input_op"]
    # a merged graph of out0 alone over the parent's inputs: y is read, b (its source) is not
    g = superopt.SearchedGraph(0, po.merge([sub.to_graph_structure()], [kept], [{}], inputs),
                               None, False, p)
    ir = to_ir(g.structure, symbol_map=None)
    names = bind_inputs(ir, g)
    assert names[1] is None and names[2] is not None
    with pytest.raises(SpecError, match="does not read"):
        rt.declared_inputs_hook(p, names, ir)
