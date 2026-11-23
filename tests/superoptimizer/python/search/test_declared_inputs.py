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
THREADS = 8


def probabilities(g, q, k, L):
    return g.exp(g.sub(g.mul(g.matmul(q, k, trans_B=True), C), L))


def definition(wrong=False):
    """o = P @ v over (q, k, v, L); ``wrong``: P @ k (same shape: SKV == 2 SQ, D == D)."""
    d = csopt.new_tile_graph()
    q = d.create_tiled_input((H, SQ, D), dtype=F16)
    k = d.create_tiled_input((H, SKV, D), dtype=F16)
    v = d.create_tiled_input((H, SKV, D), dtype=F16)
    L = d.create_tiled_input((H, SQ, 1), dtype=F32)
    d.create_tiled_output(d.matmul(probabilities(d, q, k, L), k if wrong else v))
    return d


def target(kind, *, declare=True, wrong=False):
    """Inputs q, k, v, dO, L, o (o declared over q, k, v, L; no dO for ``p_at_v``).

    p_at_v: P @ v (o unread); dq / dk: the naive gradients, with delta = sum_SKV(P * dP);
    dq_o: dQ with D = sum_D(dO * o) (reads o)."""
    g = csopt.new_tile_graph()
    q = g.create_tiled_input((H, SQ, D), dtype=F16)
    k = g.create_tiled_input((H, SKV, D), dtype=F16)
    v = g.create_tiled_input((H, SKV, D), dtype=F16)
    dO = None if kind == "p_at_v" else g.create_tiled_input((H, SQ, D), dtype=F16)
    L = g.create_tiled_input((H, SQ, 1), dtype=F32)
    o = g.create_tiled_input((H, SQ, D), dtype=F16)
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


def named(structure):
    """Each op as (type, operand names): inputs are q k v dO L o (or q k v L o), the others
    are numbered t<i> in definition order."""
    inputs = [op for op in structure if op["op_type"] == "kn_input_op"]
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


def readers_of_o(structure):
    return [op for op in named(structure) if "o" in op[1]]


def uses_D(structure):
    """D = sum_D(dO * o): a mul of dO and o reduced over its last dim."""
    ops = named(structure)
    products = {outs[0] for kind, args, outs in ops if kind == "kn_mul_op" and sorted(args) == ["dO", "o"]}
    return any(kind == "kn_reduce_sum_2_op" and args[0] in products for kind, args, _ in ops)


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
    q = g.create_tiled_input((H, SQ, D), dtype=F16)
    k = g.create_tiled_input((H, SKV, D), dtype=F16)
    v = g.create_tiled_input((H, SKV, D), dtype=F16)
    L = g.create_tiled_input((H, SQ, 1), dtype=F32)
    o = g.create_tiled_input((H, SQ, D), dtype=F16)
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
STATES = {"p_at_v": 3566}
