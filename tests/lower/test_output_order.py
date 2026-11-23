"""``to_ir`` keeps the outputs in the structure's (the Program's) order (CPU).

The kernel returns its outputs, and validation applies per-output tolerances, in the lowered
graph's output order. ``to_ir`` used to take that order from a traversal that also walks from
inputs to their consumers, so a later output reached through a shared input came first.
"""

import cleave.lang as cl
from cleave.frontend.program import Program
from cleave.lower import to_ir
from cleave.runtime import reference as rt


def test_outputs_keep_the_program_order():
    p = Program("order", cl.f16)
    m, n = p.dims(M=16, N=32)
    a, b = p.input("a", (m, n)), p.input("b", (m, n))
    t = a * b
    p.output(cl.exp(t), "o0")
    p.output(t * 2.0, "o1")
    p.output(cl.sum(b, -1, keepdim=True), "o2")  # reached from b before o1 is reached from t
    ir = to_ir(p.to_graph_structure())
    assert [tuple(o.get_shape(0)) for o in rt.output_order(ir)] == [(16, 32), (16, 32), (16,)]
    kinds = [type(o.inputs[0].src_node).__name__ for o in rt.output_order(ir)]
    assert kinds[2] != kinds[1]
