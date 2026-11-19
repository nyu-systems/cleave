"""Graph.clone: every field copied, no mutable state shared, codegen unchanged."""

from dataclasses import fields

import pytest

pytest.importorskip("tilelang")

from cleave.ir.graph import Graph, GraphNode, OutputNode, PlaceHolderNode
from cleave.ir.ops import Exp, Matmul, Mul, ReduceSum, Stabilize, wire_input
from cleave.ir.graph_utils import clone_graph, find_topo_sort, gather_all_nodes
from tests.graph._rolling_tier import build_rolling_tier, schedule_rolling_tier
from tests.graph.handwritten._cases import CASES
from tests.graph.handwritten._test_utils import gen_handwritten_code


def _all_nodes(graph):
    for node in graph.nodes + graph.else_nodes:
        yield node
        if isinstance(node, GraphNode):
            yield from _all_nodes(node.graph)


def _simple_graph():
    mm = Matmul((16, 16, 16), name="mm")
    scale = Mul((16, 16), const=2.0, name="scale")
    wire_input(mm, scale, 0)
    out = OutputNode(scale, dtype="float32")
    return Graph(nodes=find_topo_sort(gather_all_nodes(out)))


def test_clone_copies_every_field_and_owns_its_state():
    graph = _simple_graph()
    graph.sargs = [("s", "int32")]
    graph.smap = {"s": lambda i: i}
    graph.loop_kind, graph.rasterization = "serial", 1
    graph.args = [n for n in graph.nodes if isinstance(n, (PlaceHolderNode, OutputNode))]
    graph.tinfo_map = {n: (n.name, "g") for n in graph.nodes}
    mm = next(n for n in graph.nodes if isinstance(n, Matmul))
    space = list(mm.get_space_dim())  # memoized before the clone
    clone = clone_graph(graph)

    by_id = {id(n) for n in graph.nodes}
    assert not any(id(n) in by_id for n in clone.nodes)
    for f in fields(Graph):
        assert getattr(clone, f.name) == getattr(graph, f.name), f.name
    assert all(any(c is n for n in clone.nodes) for c in clone.args)
    assert all(any(k is n for n in clone.nodes) for k in clone.tinfo_map)
    assert clone.sargs is not graph.sargs and clone.smap is not graph.smap

    for old, new in zip(graph.nodes, clone.nodes):
        for attr in ("_tag", "_shapes", "_dtypes"):
            assert getattr(new, attr) == getattr(old, attr)
            assert getattr(new, attr) is not getattr(old, attr), (old.name, attr)
        assert all(e.src_node is new or e.src_node in clone.nodes for e in new.inputs)
        assert all(e.dst_node in clone.nodes for e in new.outputs)

    mm_clone = next(n for n in clone.nodes if isinstance(n, Matmul))
    assert mm_clone.get_space_dim() == space
    mm_clone.set_tag("probe", 1)
    mm_clone.get_space_dim().append(0)
    assert mm.get_tag("probe") is None and mm.get_space_dim() == space


def test_clone_remaps_lag_nodes_and_accum_scale():
    """The two rolling fields name nodes of their own tier; a clone's name the clone's."""
    t = build_rolling_tier()
    main = schedule_rolling_tier(t)
    inner = t.inner
    assert [n.name for n in inner.lag_nodes] == ["exp_0_max_prev"]
    assert {k.name: v.name for k, v in inner.accum_scale.items()} == {
        "rowsum_0": "exp_0_alpha", "psum_0": "exp_0_alpha"}

    clone = main.clone()
    loop = next(n for n in clone.nodes if isinstance(n, GraphNode))
    cloned = loop.graph
    assert cloned is not inner and cloned.lag_nodes is not inner.lag_nodes
    assert cloned.accum_scale is not inner.accum_scale

    def own(node):
        return any(node is n for n in cloned.nodes)

    old_ids = {id(n) for n in inner.nodes}
    assert [n.name for n in cloned.lag_nodes] == ["exp_0_max_prev"]
    assert all(own(n) and id(n) not in old_ids for n in cloned.lag_nodes)
    assert {k.name: v.name for k, v in cloned.accum_scale.items()} == {
        "rowsum_0": "exp_0_alpha", "psum_0": "exp_0_alpha"}
    assert all(own(k) and own(v) for k, v in cloned.accum_scale.items())
    assert all(id(k) not in old_ids and id(v) not in old_ids
               for k, v in cloned.accum_scale.items())
    # one factor node, shared by both accumulators, in the clone as in the original
    assert len({id(v) for v in cloned.accum_scale.values()}) == 1
    # the carried max stays the last accumulator, with its per-instance start value
    assert [n.name for n in cloned.accum_nodes] == ["rowsum_0", "psum_0", "exp_0_max"]
    assert cloned.accum_nodes[2].accum_init == t.m.accum_init != type(t.m).accum_init
    # the lag node of the clone snapshots the clone's max
    assert cloned.lag_nodes[0].inputs[0].src_node is cloned.accum_nodes[2]
    # the original is untouched
    assert inner.lag_nodes == [t.m_prev]
    assert inner.accum_scale == {t.rowsum: t.alpha, t.psum: t.alpha}


def test_clone_keeps_the_stabilize_record():
    """The annotation is written on the flat graph and read on its tiled clone."""
    exp = Exp((16, 32), name="exp_0")
    rowsum = ReduceSum((16, 32), name="rowsum_0")
    plain = Exp((16,), name="exp_1", n_batch_dims=0)
    wire_input(exp, rowsum, 0)
    wire_input(rowsum, plain, 0)
    graph = Graph(nodes=find_topo_sort(gather_all_nodes(OutputNode(plain, dtype="float32"))))
    record = Stabilize(axes=(-1,), scaled=(("rowsum_0", 1),), rolling="why not", export="max_0")
    exp.stabilize = record

    clone = graph.clone()
    by_name = {n.name: n for n in clone.nodes}
    assert by_name["exp_0"] is not exp
    assert by_name["exp_0"].stabilize == record
    assert by_name["exp_1"].stabilize is None and Exp.stabilize is None
    # Consuming the record on the clone (what the rewrite does) leaves the flat graph's.
    by_name["exp_0"].stabilize = None
    assert exp.stabilize == record


@pytest.mark.parametrize("name", sorted(CASES))
def test_clone_generates_the_same_kernel(name):
    graphs = CASES[name].graph()
    for graph in graphs if isinstance(graphs, list) else [graphs]:
        clone = graph.clone()
        old_ids = {id(n) for n in _all_nodes(graph)}
        assert not any(id(n) in old_ids for n in _all_nodes(clone))
        # Codegen mutates its graph, so the clone goes first: the original must not notice.
        assert gen_handwritten_code(clone) == gen_handwritten_code(graph)
