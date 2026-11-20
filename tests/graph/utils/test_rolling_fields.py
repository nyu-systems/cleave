"""``Graph.lag_nodes`` / ``Graph.accum_scale``, ``GraphNode.carry`` and ``rolling_nodes`` (CPU).

The fields declare a rolling softmax on a loop tier: a lag node snapshots a carried max
before the iteration updates it, and a rescaled accumulator is multiplied by a factor right
before it accumulates. These tests cover what the fields do to the *structure*; emission and
validation are covered next to the codegen.
"""

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.ir.graph import AccumEdge, OutputNode
from cleave.ir.ops import NEG_FLT_MAX, Copy, Exp, ReduceMax, Sub
from cleave.ir.rolling import STAT_DTYPE, rolling_nodes

from tests.graph._rolling_tier import ROWS, build_rolling_tier


def _names(nodes):
    return [n.name for n in nodes]


def test_fields_are_empty_by_default_and_appended():
    from dataclasses import fields

    from cleave.ir.graph import Graph

    graph = Graph(nodes=[])
    assert graph.lag_nodes == [] and graph.accum_scale == {}
    assert Graph(nodes=[]).lag_nodes is not graph.lag_nodes
    # Appended after every field the dataclass had: none of those moved, so positional
    # construction is what it was.
    names = [f.name for f in fields(Graph)]
    assert names[0] == "nodes" and names.index("tmap") == 1
    assert names.index("block_guard") < names.index("lag_nodes") < names.index("accum_scale")


def test_a_factor_and_a_lag_node_are_not_accumulators():
    t = build_rolling_tier(carry=False)
    # alpha has no dataflow consumer -- exactly what marks an accumulator -- but it is a
    # per-iteration value read through `accum_scale`.
    assert not t.alpha.outputs
    assert _names(t.inner.accum_nodes) == ["rowsum_0", "psum_0"]
    assert len(t.loop.outputs) == 2 and all(isinstance(e, AccumEdge) for e in t.loop.outputs)


def test_without_the_declaration_the_factor_would_be_carried():
    """The exclusion is what the fields add; with them empty, inference is what it was."""
    t = build_rolling_tier(declare=False, carry=False)
    assert _names(t.inner.accum_nodes) == ["exp_0_alpha", "rowsum_0", "psum_0"]


@pytest.mark.parametrize("declare, expected", [(True, []), (False, ["rowsum_0_prev"])])
def test_an_unused_lag_node_is_not_an_accumulator(declare, expected):
    from cleave.ir.graph import Graph, GraphNode
    from cleave.ir.ops import ReduceSum, wire_input

    rowsum = ReduceSum((ROWS, 64), name="rowsum_0")
    lag = Copy((ROWS,), name="rowsum_0_prev", n_batch_dims=0)
    wire_input(rowsum, lag, 0)
    OutputNode(rowsum, dtype="float32")
    inner = Graph(nodes=[rowsum.inputs[0].src_node, rowsum, lag])
    if declare:
        inner.lag_nodes = [lag]
    GraphNode(name="GraphNode_0", graph=inner)
    # The lag node has no consumer, which is what marks an accumulator; declared, it is
    # left alone. (rowsum is read by it inside the loop, so it is not inferred either way:
    # a node its own loop reads is carried only by declaration, `carry`.)
    assert _names(inner.accum_nodes) == expected


def test_carry_appends_the_max_and_keeps_in_loop_readers():
    t = build_rolling_tier(carry=False)
    before = list(t.inner.accum_nodes)
    in_loop = [(e.dst_node.name, e.dst_id) for e in t.m.outputs]

    assert t.loop.carry(t.m) == 2
    assert t.inner.accum_nodes[:2] == before and t.inner.accum_nodes[2] is t.m
    assert _names(t.inner.accum_nodes) == ["rowsum_0", "psum_0", "exp_0_max"]
    # the GraphNode's output slot of the new accumulator
    assert list(t.loop.get_shape(2)) == list(t.m.get_shape(0)) == [ROWS]
    assert str(t.loop.get_dtype(2)) == "float32"
    assert t.loop._resolve_output_binding(2)[0] is t.m
    # readers inside the loop keep their ordinary edges to the carried buffer
    assert [(e.dst_node.name, e.dst_id) for e in t.m.outputs] == in_loop
    assert sorted(in_loop) == [("exp_0_max_prev", 0), ("exp_0_max_step", 1), ("exp_0_shifted", 1)]
    assert not any(isinstance(e, AccumEdge) for e in t.m.outputs)
    assert len(t.loop.outputs) == 2  # nothing outside reads the max


def test_carry_is_idempotent():
    t = build_rolling_tier()
    assert t.loop.carry(t.m) == 2 and t.loop.carry(t.m) == 2
    assert _names(t.inner.accum_nodes) == ["rowsum_0", "psum_0", "exp_0_max"]
    # ... and a no-op for a node `_assign_accum_nodes` already took (an exported max).
    edges = list(t.loop.outputs)
    assert t.loop.carry(t.rowsum) == 0 and t.loop.carry(t.psum) == 1
    assert _names(t.inner.accum_nodes) == ["rowsum_0", "psum_0", "exp_0_max"]
    assert list(t.loop.outputs) == edges


def test_carry_rewires_readers_outside_the_loop():
    """A max that is read in the loop AND twice outside it: every reader outside gets an
    AccumEdge that names the max by its accumulator index.

    Two readers, because one cannot tell the index from an edge count: the loop has two
    edges when ``carry`` runs, which is also the index the max gets, and only the second
    reader's number runs ahead (to an output the loop does not have) when edges are counted.
    """
    t = build_rolling_tier(carry=False)
    # two plain edges across the loop boundary
    readers = [OutputNode(t.m, dtype="float32") for _ in range(2)]
    assert not any(isinstance(r.inputs[0], AccumEdge) for r in readers)
    edges_before = len(t.loop.outputs)

    index = t.loop.carry(t.m)
    assert index == next(i for i, n in enumerate(t.inner.accum_nodes) if n is t.m) == 2
    # Counting edges gives the first reader 2 as well; it is the second one that tells.
    assert edges_before == index

    rewired = [r.inputs[0] for r in readers]
    assert rewired[0] is not rewired[1]
    for edge in rewired:
        assert isinstance(edge, AccumEdge) and edge.src_node is t.loop and edge.accum_node is t.m
        assert edge.src_id == index and edge.dst_id == 0
        assert any(edge is e for e in t.loop.outputs) and not any(edge is e for e in t.m.outputs)
        bound, shape, _ = t.loop._resolve_output_binding(edge.src_id)
        assert bound is t.m and list(shape) == list(t.m.get_shape(0)) == [ROWS]
    assert len(t.loop.outputs) == edges_before + 2
    # only the readers inside the loop are left on the max itself
    assert sorted((e.dst_node.name, e.dst_id) for e in t.m.outputs) == [
        ("exp_0_max_prev", 0), ("exp_0_max_step", 1), ("exp_0_shifted", 1)]
    # the accumulators that were there keep their numbers
    assert sorted((e.accum_node.name, e.src_id) for e in t.loop.outputs[:edges_before]) == [
        ("psum_0", 1), ("rowsum_0", 0)]


def test_carry_refuses_what_is_not_a_compute_node_of_the_loop():
    t = build_rolling_tier(carry=False)
    with pytest.raises(AssertionError, match="not a node of GraphNode_0"):
        t.loop.carry(t.div)  # a node of the outer tier
    inner_in = t.inner.binding[t.outer_in.name]
    with pytest.raises(AssertionError, match="not a compute node"):
        t.loop.carry(inner_in)
    assert _names(t.inner.accum_nodes) == ["rowsum_0", "psum_0"]


class _OtherExp(Exp):
    kind = "other_exp"


@pytest.mark.parametrize("exp_cls", [Exp, _OtherExp])
@pytest.mark.parametrize("rows, n_batch_dims", [((ROWS,), 0), ((4, ROWS), 1)])
def test_rolling_nodes_names_dtypes_and_wiring(exp_cls, rows, n_batch_dims):
    m = ReduceMax((*rows, 64), axis=-1, name="exp_5_max", accum_init=NEG_FLT_MAX)
    m_prev, step, alpha = rolling_nodes(
        m, stem="exp_5", rows=rows, n_batch_dims=n_batch_dims, exp_cls=exp_cls)

    assert _names([m_prev, step, alpha]) == ["exp_5_max_prev", "exp_5_max_step", "exp_5_alpha"]
    assert type(m_prev) is Copy and type(step) is Sub and type(alpha) is exp_cls
    for node in (m_prev, step, alpha):
        assert node.dtype == node.out_dtype == STAT_DTYPE == "float32", node.name
        assert tuple(node.dims) == tuple(rows) and node.n_batch_dims == n_batch_dims, node.name
        assert [int(d) for d in node.get_shape(0)] == list(rows), node.name
    assert step.const is None and step.broadcast == -1  # m_prev - m, elementwise

    def sources(node):
        return [(e.src_node, e.dst_id) for e in node.inputs]

    assert sources(m_prev) == [(m, 0)]
    assert sources(step) == [(m_prev, 0), (m, 1)]  # m_prev - m: <= 0, so alpha <= 1
    assert sources(alpha) == [(step, 0)]
    assert all(e.src_node is n for n, e in ((m, m_prev.inputs[0]), (m, step.inputs[1])))
    assert sorted((e.dst_node.name, e.dst_id) for e in m.outputs) == [
        ("exp_5_max_prev", 0), ("exp_5_max_step", 1)]
    assert not alpha.outputs  # read through `accum_scale`, not through an edge


def test_rolling_nodes_need_a_float32_max():
    m = ReduceMax((ROWS, 64), name="exp_5_max", dtype="float16", out_dtype="float16")
    with pytest.raises(AssertionError, match="float32"):
        rolling_nodes(m, stem="exp_5", rows=(ROWS,), n_batch_dims=0, exp_cls=Exp)
