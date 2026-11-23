"""Declared inputs in the search: an input that IS a function of earlier inputs.

The attention backward reads the forward's output ``o = exp(q @ k^T * c - L) @ v`` as an input.
Declared (``TileGraph.declare_input``), the search may use it wherever its definition occurs: the
row reduction ``delta = sum_SKV(P * dP)`` of the naive dQ / dK becomes ``D = sum_D(dO * o)``,
which a fused kernel computes without a second pass over SKV. The search sees ``o`` through its
definition: its pattern is the definition's over its sources' patterns, its fingerprint the
definition's over theirs. An unread declared input is no output (and no longer aborts the
symbolizer).

The graphs are built through the csopt API directly, at small shapes, and searched on the CPU.
"""

import re

import pytest

import cleave.superoptimizer as csopt
from cleave.superoptimizer import core

H, SQ, SKV, D = 4, 128, 256, 64
C = 64 ** -0.5
F16, F32 = core.float16, core.float32
THREADS = 4  # the counts do not depend on it (checked with 1, 4, 8)


def inp(g, dims, dtype):
    """An input that may not be split (as ``Program.to_tile_graph`` makes them without split
    dims): the default, inferred can_split would let the search try split ops."""
    return g.create_tiled_input(dims, (False,) * len(dims), dtype)


def probabilities(g, q, k, L):
    return g.exp(g.sub(g.mul(g.matmul(q, k, trans_B=True), C), L))


def definition(wrong=False):
    """o = P @ v over (q, k, v, L); ``wrong``: P @ k (same shape: SKV == 2 SQ, D == D)."""
    d = csopt.new_tile_graph()
    q = inp(d, (H, SQ, D), F16)
    k = inp(d, (H, SKV, D), F16)
    v = inp(d, (H, SKV, D), F16)
    L = inp(d, (H, SQ, 1), F32)
    d.create_tiled_output(d.matmul(probabilities(d, q, k, L), k if wrong else v))
    return d


def target(kind, *, declare=True, wrong=False):
    """Inputs q, k, v, dO, L, o (o declared over q, k, v, L; no dO for ``p_at_v``).

    p_at_v: P @ v (o unread); dq / dk: the naive gradients, with delta = sum_SKV(P * dP);
    dq_o: dQ with D = sum_D(dO * o) (reads o)."""
    g = csopt.new_tile_graph()
    q = inp(g, (H, SQ, D), F16)
    k = inp(g, (H, SKV, D), F16)
    v = inp(g, (H, SKV, D), F16)
    dO = None if kind == "p_at_v" else inp(g, (H, SQ, D), F16)
    L = inp(g, (H, SQ, 1), F32)
    o = inp(g, (H, SQ, D), F16)
    if declare:
        g.declare_input(o, definition(wrong), [q, k, v, L])
    p = probabilities(g, q, k, L)
    if kind == "p_at_v":
        g.create_tiled_output(g.matmul(p, v))
        return g
    dp = g.matmul(dO, v, trans_B=True)
    delta = g.reduce_sum(g.mul(dO, o) if kind == "dq_o" else g.mul(p, dp), 2)
    ds = g.mul(g.mul(p, g.sub(dp, delta)), C)
    g.create_tiled_output(g.matmul(ds, k) if kind.startswith("dq") else g.matmul(ds, q, trans_A=True))
    return g


def search(g, max_ops, capfd, checkpoint=""):
    symbolic, _ = csopt.create_symbolic_tile_graph(g)
    cfg = symbolic.setup_search_config()
    cfg.max_num_kernel_graph_op = max_ops
    cfg.search_thread = THREADS
    capfd.readouterr()
    found = symbolic.search_tile_graph(checkpoint, False, False, cfg)
    err = capfd.readouterr().err
    states = re.findall(r"\[Search\] Total states explored: (\d+)", err)
    return [f.get_graph_structure() for f in found], int(states[0]) if states else None


def named(structure, names=None):
    """Each op as (type, operand names, output names): the inputs are named ``names`` (q k v
    dO L o, or q k v L o), the others t<i> in definition order."""
    inputs = [op for op in structure if op["op_type"] == "kn_input_op"]
    if names is None:
        names = ["q", "k", "v", "dO", "L", "o"] if len(inputs) == 6 else ["q", "k", "v", "L", "o"]
    index = {}
    out = []
    for op in structure:
        args = [index[t["guid"]] for t in op["input_tensors"]]
        for t in op["output_tensors"]:
            index[t["guid"]] = names[len(index)] if op["op_type"] == "kn_input_op" else f"t{len(index)}"
        if op["op_type"] != "kn_input_op":
            out.append((op["op_type"], args, [index[t["guid"]] for t in op["output_tensors"]]))
    return out


def readers_of_o(structure, names=None):
    return [op for op in named(structure, names) if "o" in op[1]]


def uses_D(structure, names=None):
    """D = sum_D(dO * o): a mul of dO and o reduced over its last dim."""
    ops = named(structure, names)
    products = {outs[0] for kind, args, outs in ops if kind == "kn_mul_op" and sorted(args) == ["dO", "o"]}
    return any(kind == "kn_reduce_sum_2_op" and args[0] in products for kind, args, _ in ops)


# ---------------------------------------------------------------- the row reduction, small
# The full naive dQ / dK (6 inputs, 17 operators) is far beyond a test's budget: the search
# explores 33.6M states for the undeclared dS alone (5 inputs, 14 operators). The same rewrite
# at its smallest: the probabilities P are an input, o = P @ v is declared over (P, v), and the
# target is dS = P * (dP - sum_SKV(P * dP)), dP = dO @ v^T (4 inputs, 9 operators).

TOY = ["P", "v", "dO", "o"]


def toy_definition(wrong=False):
    """o = P @ v; ``wrong``: exp(P @ v) (same shape)."""
    d = csopt.new_tile_graph()
    p = inp(d, (H, SQ, SKV), F16)
    v = inp(d, (H, SKV, D), F16)
    pv = d.matmul(p, v)
    d.create_tiled_output(d.exp(pv) if wrong else pv)
    return d


def toy(*, reads_o=False, scale=False, wrong=False):
    g = csopt.new_tile_graph()
    p = inp(g, (H, SQ, SKV), F16)
    v = inp(g, (H, SKV, D), F16)
    dO = inp(g, (H, SQ, D), F16)
    o = inp(g, (H, SQ, D), F16)
    g.declare_input(o, toy_definition(wrong), [p, v])
    dp = g.matmul(dO, v, trans_B=True)
    ds = g.mul(p, g.sub(dp, g.reduce_sum(g.mul(dO, o) if reads_o else g.mul(p, dp), 2)))
    g.create_tiled_output(g.mul(ds, C) if scale else ds)
    return g


NAIVE = [("kn_matmul_op", ["dO", "v"], ["t4"]), ("kn_mul_op", ["P", "t4"], ["t5"]),
         ("kn_reduce_sum_2_op", ["t5"], ["t6"]), ("kn_sub_op", ["t4", "t6"], ["t7"]),
         ("kn_mul_op", ["P", "t7"], ["t8"]), ("kn_output_op", ["t8"], [])]
WITH_D = [("kn_matmul_op", ["dO", "v"], ["t4"]), ("kn_mul_op", ["dO", "o"], ["t5"]),
          ("kn_reduce_sum_2_op", ["t5"], ["t6"]), ("kn_sub_op", ["t4", "t6"], ["t7"]),
          ("kn_mul_op", ["P", "t7"], ["t8"]), ("kn_output_op", ["t8"], [])]


def test_the_row_reduction_becomes_a_reduction_of_dO_times_o(capfd):
    graphs, states = search(toy(), 9, capfd)
    assert [named(g, TOY) for g in graphs] == [NAIVE, WITH_D]
    assert [uses_D(g, TOY) for g in graphs] == [False, True]
    assert states == STATES["toy"]


def test_a_target_that_reads_o_is_still_found(capfd):
    graphs, states = search(toy(reads_o=True), 9, capfd)
    assert WITH_D in [named(g, TOY) for g in graphs]
    assert states == STATES["toy_reads_o"]


def test_a_wrong_declaration_is_not_used(capfd):
    # o declared as exp(P @ v): its fingerprint is not P @ v's, the search finds only the naive dS
    graphs, states = search(toy(wrong=True), 9, capfd)
    assert [named(g, TOY) for g in graphs] == [NAIVE]
    assert states == STATES["toy_wrong"]


def test_the_scale_applies_to_the_rewritten_dS(capfd):
    # The scalar op is tried only on the pattern the target scales: P * (dP - delta). Its
    # rewritten form P * (dP - D) is the same pattern up to commutativity and moving the
    # reductions, which the scalar gate compares for programs that declare inputs.
    graphs, states = search(toy(scale=True), 10, capfd)
    scaled = [named(g, TOY) for g in graphs]
    tail = [("kn_mul_op", ["t8"], ["t9"]), ("kn_output_op", ["t9"], [])]
    assert WITH_D[:-1] + tail in scaled and NAIVE[:-1] + tail in scaled
    assert states == STATES["toy_scale"]


def test_an_unread_declared_input_is_accepted_and_no_output(capfd):
    # It used to abort the symbolizer ("Dimension not found in symbol map"); and were it an
    # output, no candidate could match the one output of the target.
    graphs, states = search(target("p_at_v"), 10, capfd)
    assert len(graphs) == 1 and readers_of_o(graphs[0]) == []
    assert [k for k, _, _ in named(graphs[0])] == [
        "kn_matmul_op", "kn_mul_op", "kn_sub_op", "kn_exp_op", "kn_matmul_op", "kn_output_op"]
    assert states == STATES["p_at_v"]


def test_declare_input_refuses_bad_declarations():
    g = csopt.new_tile_graph()
    q = inp(g, (H, SQ, D), F16)
    k = inp(g, (H, SKV, D), F16)
    v = inp(g, (H, SKV, D), F16)
    L = inp(g, (H, SQ, 1), F32)
    o = inp(g, (H, SQ, D), F16)
    with pytest.raises(ValueError, match="does not come before"):
        g.declare_input(q, definition(), [q, k, v, L])
    with pytest.raises(ValueError, match="inputs for 3 sources"):
        g.declare_input(o, definition(), [q, k, v])
    with pytest.raises(ValueError, match="another shape"):
        g.declare_input(o, definition(), [k, q, v, L])
    assert not g.has_declared_inputs()
    g.declare_input(o, definition(), [q, k, v, L])
    assert g.has_declared_inputs()
    with pytest.raises(ValueError, match="already declared"):
        g.declare_input(o, definition(), [q, k, v, L])


def test_a_declaration_changes_the_checkpoint_key_and_only_then():
    keys = {}
    for declare, wrong in ((False, False), (True, False), (True, True)):
        # dq_o reads o: undeclared, it is a plain input (an unread one would abort the symbolizer)
        symbolic, _ = csopt.create_symbolic_tile_graph(target("dq_o", declare=declare, wrong=wrong))
        keys[declare, wrong] = symbolic.checkpoint_key()
    assert len(set(keys.values())) == 3
    # ... and is a function of the program alone (no guids)
    symbolic, _ = csopt.create_symbolic_tile_graph(target("dq_o"))
    assert symbolic.checkpoint_key() == keys[True, False]


# Minimal budgets (operators, inputs included) and the states the searches explore there.
STATES = {"p_at_v": 16, "toy": 31643, "toy_reads_o": 10927, "toy_wrong": 14067, "toy_scale": 157616}


def test_a_checkpoint_keeps_the_declaration(tmp_path, capfd):
    # The searched graphs go through JSON (checkpoint) with their declared input; a reload
    # gives the same graphs, and its key is the declared program's.
    ckpt = str(tmp_path / "toy.json")
    fresh, _ = search(toy(), 9, capfd, checkpoint=ckpt)
    with open(ckpt) as f:
        text = f.read()
    assert text.count('"declaration"') == len(fresh) == 2
    again, states = search(toy(), 9, capfd, checkpoint=ckpt)
    assert states is None  # loaded, not searched
    assert [named(g, TOY) for g in again] == [named(g, TOY) for g in fresh]


def test_superoptimize_a_program_with_an_equals_input():
    # Through the frontend: ``o`` declared with equals= reaches the search as a declaration.
    import cleave.lang as cl
    from cleave.frontend import Program
    from cleave.pipeline import SuperoptConfig, superoptimize

    p = Program("toy_bwd", cl.f16)
    h, sq, skv, d = p.dims(H=H, SQ=SQ, SKV=SKV, D=D)
    P = p.input("P", (h, sq, skv))
    v = p.input("v", (h, skv, d))
    dO = p.input("dO", (h, sq, d))
    p.input("o", (h, sq, d), equals=lambda P, v: P @ v)
    dP = dO @ v.mT
    p.output(P * (dP - cl.sum(P * dP, -1, keepdim=True)), "dS")
    graphs = superoptimize(p, SuperoptConfig(max_ops=9, threads=THREADS, cache=False))
    assert [named(g.structure, TOY) for g in graphs] == [NAIVE, WITH_D]
