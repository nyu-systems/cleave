"""The exact softmax across a split (CPU): ``cleave.lower.scale.stabilize_split``.

A split partial cannot cancel the factor of its softmax: every output of a partial keeps
degree 1, each split relative to its own max. The pair is made exact on both sides of the
cut instead. The partial exports that max as one more output, in float32 and under the
boundary role ``"shift"``; the combine, which the scale analysis proves to cancel the factor
once its inputs are seeded with the partial's output scales, weights every partial by
``exp(m_i)`` before it sums them over the split axis: a softmax over the split axis, marked
by the solo rule and served by the rewrite like any other.

The pairs here are lowered from the stored GQA split structures (no search); no kernel is
involved. ``tests/integration/test_opaque_ops_lowering.py`` pins that a max a user wrote is
still refused in a split graph.
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave.core.errors import LoweringRejected
from cleave.frontend import trace
from cleave.ir.graph import Boundary, Graph, OutputNode, PlaceHolderNode
from cleave.ir.graph_utils import boundary_signature, find_topo_sort, gather_all_nodes
from cleave.ir.ops import (NEG_FLT_MAX, Div, Exp, Matmul, Mul, ReduceMax, ReduceSum, Stabilize,
                           wire_input)
from cleave.lower.process import to_bundles
from cleave.lower.scale import (SHIFT, ScaleError, analyse, analyse_split, exact_combine, exps,
                                export_shift, record_for, stabilize_split)
from cleave.pipeline.plans import bind_inputs, lower
from cleave.pipeline.superopt import SearchedGraph
from cleave.runtime.reference import execute, output_order, placeholder_order
from tests._softmax import chained_reference
from tests.pipeline import _pipeline as PL

FACTOR = PL.SPLIT_FACTOR
VALID = (1, 2, 3, 4, 5)  # the stored structures that cut into a partial and a combine
# the stored structures' symbols: 64 query heads over 16 key heads, 64 queries, 256 keys
SHAPES = dict(HQ=64, HKV=16, SQ=64, SKV=256, D=64)
ROWSUM, VALUE, MAX = (64, FACTOR, 64), (64, FACTOR, 64, 64), (64, FACTOR, 64)
CROSSING = ((0, "", ROWSUM, "float16"), (1, "", VALUE, "float16"), (2, SHIFT, MAX, "float32"))


def attention(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"],
              v: cl.Tensor["HKV", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


def attention_twin(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"],
                   v: cl.Tensor["HKV", "SKV", "D"]):
    e = cl.exp((q @ k.mT) / q.shape[-1] ** 0.5)  # the same ops without the request
    return (e / cl.sum(e, -1, keepdim=True)) @ v


def _structure(n):
    return json.loads((PL.STRUCTURES_DIR / "gqa" / f"before_split_gqa_graph_structure_valid_{n}"
                       ".graph").read_text())


def _program(fn=attention):
    return trace(fn, SHAPES, cl.f16)


def _exact_pair(n=1, program=None, factor=FACTOR):
    """The stored split structure ``n`` lowered for a program that requested a softmax."""
    graph = SearchedGraph(0, _structure(n), dict(PL.SYMBOL_MAPS["gqa"]), True,
                          program or _program(), ("SKV",))
    return lower(graph, factor)


def _plain_pair(n=1, factor=FACTOR):
    """The same structure as ``to_bundles`` hands it over: no request, no record."""
    ((partial, combine),) = to_bundles([_structure(n)], symbol_map=dict(PL.SYMBOL_MAPS["gqa"]),
                                       split_factors=[factor])
    return partial, combine


def _nodes(graph):
    return {n.name: n for n in graph.nodes}


def _described(graph):
    return {n.name: (type(n).__name__, tuple(int(s) for s in n.get_shape(0)))
            for n in graph.nodes}


def _reads(node):
    return [(e.src_node.name, e.src_id) for e in sorted(node.inputs, key=lambda e: e.dst_id)]


# ---------------------------------------------------------------------------- the partial

@pytest.mark.parametrize("n", VALID)
def test_the_partial_exports_its_shift_as_one_more_output(n):
    partial, _ = _exact_pair(n)
    twin, _ = _plain_pair(n)
    assert boundary_signature(output_order(partial)) == CROSSING

    # The outputs the cut produced keep their name, producer and index, and their order
    # among themselves; the kernel returns the shift in front of them.
    outs, twin_outs = output_order(partial), output_order(twin)
    assert [(o.name, _reads(o), o.boundary) for o in outs[1:]] == [
        (o.name, _reads(o), o.boundary) for o in twin_outs]
    assert [o.boundary for o in outs] == [Boundary(2, SHIFT), Boundary(0), Boundary(1)]
    assert [o.name for o in outs] == ["OutputNodg", "OutputNode", "OutputNodf"]

    (exp,) = exps(partial)
    (m,) = [node for node in partial.nodes if type(node) is ReduceMax]
    assert outs[0].inputs[0].src_node is m and str(outs[0].get_dtype(0)) == "float32"
    # ... so the max stands in front of the exp, where the kernel takes it: the tiling
    # policy frees the scores once the exp has read them, as in the two-output partial
    order = [node.name for node in partial.nodes
             if not node.is_placeholder() and not node.is_output()]
    assert order.index(m.name) == order.index(exp.name) - 1
    assert m.name == f"reduce_max_3_{exp.name}" and m.out_dtype == "float32"
    # a sibling of the Exp: it reads what the Exp reads, over the softmax axis, and feeds
    # nothing but its own output (no max is an ancestor of a sum)
    assert _reads(m) == _reads(exp) and m.flat_axis == 3 and m.dims == exp.dims
    assert [type(e.dst_node) for e in m.outputs] == [OutputNode]
    sums = [node for node in partial.nodes if type(node) in (ReduceSum, Matmul)
            and any(e.src_node is exp for e in node.inputs)]
    assert exp.stabilize == Stabilize(
        axes=(-1,), scaled=exp.stabilize.scaled, rolling="", export=m.name)
    assert sorted(exp.stabilize.scaled) == sorted((s.name, 1) for s in sums) and len(sums) == 2

    # Otherwise the partial is the one the cut produced: the search and the policy see
    # the max next to the sums and nothing else.
    added = {k: v for k, v in _described(partial).items() if k not in _described(twin)}
    assert added == {m.name: ("ReduceMax", MAX), outs[0].name: ("OutputNode", MAX)}
    assert {k: v for k, v in _described(partial).items() if k not in added} == _described(twin)
    assert [e.stabilize for e in exps(twin)] == [None]


def test_the_partial_binds_the_program_inputs_as_before():
    """The exported max adds an output, no input: the three placeholders are the program's
    inputs in declaration order."""
    program = _program()
    partial, _ = _exact_pair(1, program)
    twin, _ = _plain_pair(1)
    assert bind_inputs(partial, program) == bind_inputs(twin, program)
    assert len(bind_inputs(partial, program)) == 3


def test_the_max_reads_the_scores_as_they_are_stored():
    """Where the Exp reads an input, the max reads that input in its own dtype and still
    returns float32 (a max of stored values is exact there)."""
    partial, combine = _plain_pair(1)
    (exp,) = exps(partial)
    scores = PlaceHolderNode(name="scores")
    scores.set_shape(tuple(int(s) for s in exp.get_shape(0)))
    import tvm

    scores.set_dtype(tvm.DataType("float16"))
    wire_input(scores, exp, 0)
    stored = Graph(nodes=find_topo_sort(output_order(partial)))
    report = analyse(stored, exp)
    _, m = export_shift(stored, exp, report)
    assert (m.dtype, m.out_dtype) == ("float16", "float32")
    assert m.inputs[0].src_node is scores


# ---------------------------------------------------------------------------- the combine

@pytest.mark.parametrize("n", VALID)
def test_the_combine_is_a_softmax_over_the_split_axis(n):
    partial, combine = _exact_pair(n)
    _, twin = _plain_pair(n)
    # both sides of the cut carry the same three tensors
    assert boundary_signature(placeholder_order(combine)) == CROSSING

    (w,) = exps(combine)
    nodes = _nodes(combine)
    shift = w.inputs[0].src_node
    assert (w.name, shift.name, shift.boundary) == ("exp_shift", "PlaceHolder_shift",
                                                    Boundary(2, SHIFT))
    assert (w.dtype, w.out_dtype, w.n_batch_dims) == ("float32", "float32", 1)
    # The record is the solo rule's own: taken off, the analysis gives it back.
    record, w.stabilize = w.stabilize, None
    assert record_for(combine, w) == record
    w.stabilize = record
    sums = [node for node in combine.nodes if type(node) is ReduceSum]
    assert record.axes == (-2,) and record.rolling == "" and record.export is None
    assert sorted(record.scaled) == sorted((s.name, 1) for s in sums) and len(sums) == 2

    # Derived from the lowered combine: the sums keep their names, axis and consumers and
    # read a float32 product of a partial and its weight; everything behind them is the
    # twin's. No max: the flat graph stays shift-free.
    twin_nodes = _nodes(twin)
    for s in sums:
        old = twin_nodes[s.name]
        assert (s.flat_axis, s.out_dtype, s.dtype) == (old.flat_axis, old.out_dtype, "float32")
        (product,) = [e.src_node for e in s.inputs]
        assert type(product) is Mul and product.name == f"{s.name}_weighted"
        assert (product.dtype, product.out_dtype) == ("float32", "float32")
        assert _reads(product) == [(_reads(old)[0][0], 0), ("exp_shift", 0)]
        # the value partials have one more axis than the weights: B is A without it
        value = len(product.dims) == 4
        assert (product.broadcast, product._lead()) == ((0, True) if value else (-1, False))
        assert [e.dst_node.name for e in s.outputs] == [e.dst_node.name for e in old.outputs]
    assert not any(type(node) is ReduceMax for node in combine.nodes)
    added = {k for k in _described(combine) if k not in _described(twin)}
    assert added == {"exp_shift", "PlaceHolder_shift"} | {f"{s.name}_weighted" for s in sums}
    assert {k: v for k, v in _described(combine).items() if k not in added} == _described(twin)
    (div,) = [node for node in combine.nodes if type(node) is Div]
    assert _reads(div) == _reads(twin_nodes[div.name])


def test_lowering_twice_gives_equal_pairs_on_fresh_nodes():
    """``lower`` runs once per split factor and again under ``fusion="auto"``."""
    first, second = _exact_pair(1), _exact_pair(1)
    for a, b in zip(first, second):
        assert _described(a) == _described(b)
        assert [e.stabilize for e in exps(a)] == [e.stabilize for e in exps(b)]
        assert not {id(n) for n in a.nodes} & {id(n) for n in b.nodes}


# ------------------------------------------------------------------------------- pairing

def test_the_two_sides_are_paired_by_boundary_record_not_by_position():
    partial, combine = _plain_pair(1)
    # The lowered graphs list the two tensors the other way round: by position the value
    # partials would be seeded with the scale of the row sums.
    assert [o.boundary.index for o in partial.nodes if isinstance(o, OutputNode)] == [0, 1]
    assert [p.boundary.index for p in combine.nodes if isinstance(p, PlaceHolderNode)] == [1, 0]
    exp, report, combined = analyse_split(partial, combine)
    assert report.const == (3,) and combined.const == (1,)
    assert [s.deg for s in report.outputs.values()] == [1, 1]
    assert [s.deg for s in combined.outputs.values()] == [0]
    # ... and what is summed over the split axis are the two partials, each of degree 1
    assert sorted((r.kind, r.exp_axis, r.degree) for r in combined.reductions) == [
        ("ReduceSum", 1, 1), ("ReduceSum", 1, 1)]


def test_a_seed_is_the_scale_of_the_tensor_it_is_given_to():
    """What pairing by position would do to this pair: the value partials seeded with the
    scale of the row sums, a rank short. The analysis refuses a seed of another rank
    instead of reading axes off it that the input does not have."""
    partial, combine = _plain_pair(1)
    exp, report, _ = analyse_split(partial, combine)
    rowsum, value = report.outputs.values()
    assert (len(rowsum.ax), len(value.ax)) == (3, 4)
    inputs = [p for p in combine.nodes if isinstance(p, PlaceHolderNode)]
    by_record = {p.name: (rowsum, value)[p.boundary.index] for p in inputs}
    assert analyse(combine, None, by_record).const == (1,)
    by_position = dict(zip((p.name for p in inputs), (rowsum, value)))
    assert by_position != by_record
    with pytest.raises(ScaleError, match=r"^PlaceHolder_input_\d+: seeded with the scale of a "
                                         r"value of rank 3, the input has rank 4$"):
        analyse(combine, None, by_position)


def test_a_combine_that_reads_other_tensors_than_the_partial_writes_is_refused():
    partial, combine = _plain_pair(1)
    a, b = [p for p in combine.nodes if isinstance(p, PlaceHolderNode)]
    a.boundary, b.boundary = b.boundary, a.boundary  # the right indices on the wrong tensors
    with pytest.raises(ScaleError) as e:
        analyse_split(partial, combine)
    assert str(e.value) == (
        "split boundary: the combine reads [#0 (64, 4, 64, 64) float16, #1 (64, 4, 64) float16] "
        "but the partial writes [#0 (64, 4, 64) float16, #1 (64, 4, 64, 64) float16]")


@pytest.mark.parametrize("side", ["partial", "combine", "one input"])
def test_a_pair_without_its_boundary_records_is_refused(side):
    partial, combine = _plain_pair(1)
    nodes = {"partial": [n for n in partial.nodes if isinstance(n, OutputNode)],
             "combine": [n for n in combine.nodes if isinstance(n, PlaceHolderNode)]}
    for node in nodes.get(side, nodes["combine"][:1]):
        node.boundary = None
    text = {"partial": "the partial's outputs carry no boundary records",
            "combine": "the combine's inputs carry no boundary records",
            "one input": "boundary records on only some of"}[side]
    with pytest.raises(ScaleError, match=f"^split boundary: .*{text}"):
        analyse_split(partial, combine)


# ------------------------------------------------------------------------------ refusals

def test_a_partial_output_that_does_not_carry_the_factor_is_refused():
    """A partial is only exact relative to its own max when every output is: one that has
    lost the factor (here, the scores themselves) cannot be combined with the others."""
    partial, combine = _plain_pair(1)
    (exp,) = exps(partial)
    scores = exp.inputs[0].src_node
    extra = OutputNode(scores, dtype="float16")
    extra.boundary = Boundary(2)
    outs = [n for n in partial.nodes if isinstance(n, OutputNode)] + [extra]
    gather_all_nodes(*outs)  # unique names, as ``to_ir`` gives them
    partial = Graph(nodes=find_topo_sort(outs))
    with pytest.raises(ScaleError) as e:
        analyse_split(partial, combine)
    assert str(e.value) == (f"{exp.name}: partial output of {scores.name} has degree 0 in its "
                            "scale; every output of a split partial keeps degree 1")


def test_a_combine_that_does_not_cancel_the_factor_is_refused():
    """The row sums alone, summed over the split axis: the result still has degree 1."""
    partial, combine = _plain_pair(1)
    rowsum = next(n for n in combine.nodes if type(n) is ReduceSum and len(n.dims) == 3)
    value = next(n for n in combine.nodes if type(n) is ReduceSum and len(n.dims) == 4)
    out = OutputNode(rowsum, dtype="float16")
    out.boundary = Boundary(0)
    extra = OutputNode(value, dtype="float16")  # keeps the value partials an input
    extra.boundary = Boundary(1)
    gather_all_nodes(out, extra)
    broken = Graph(nodes=find_topo_sort([out, extra]))
    (exp,) = exps(partial)
    with pytest.raises(ScaleError) as e:
        analyse_split(partial, broken)
    assert str(e.value) == (f"combine: output of {rowsum.name} keeps degree 1 in the scale of "
                            f"{exp.name}")


def _hand_combine(*, projected=False, product=False, other_axis=False):
    """A combine built by hand: ``sum_i o_i / sum_i l_i`` over the split axis, optionally
    with an output projection, with the partials multiplied before they are summed, or
    with the row sums summed over the rows."""
    bh, f, sq, d, e = 32, 16, 64, 64, 48
    kw = dict(dtype="float16", out_dtype="float32")
    sum_l = ReduceSum((bh, f, sq), axis=2 if other_axis else 1, name="reduction_1_2", **kw)
    sum_o = ReduceSum((bh, f, sq, d), axis=1, name="reduction_1_3", **kw)
    l, o = sum_l.inputs[0].src_node, sum_o.inputs[0].src_node
    l.boundary, o.boundary = Boundary(0), Boundary(1)
    if product:
        both = Mul((bh, f, sq, d), broadcast=0, name="mul_1", n_batch_dims=1, **kw)
        wire_input(o, both, 0)
        wire_input(l, both, 1)
        wire_input(both, sum_o, 0)
    if other_axis:
        return Graph(nodes=find_topo_sort([OutputNode(sum_l, dtype="float16"),
                                           OutputNode(sum_o, dtype="float16")]))
    div = Div((bh, sq, d), broadcast=0, name="div_4", dtype="float32", out_dtype="float32",
              n_batch_dims=1)
    wire_input(sum_o, div, 0)
    wire_input(sum_l, div, 1)
    last = div
    if projected:
        last = Matmul((bh, sq, e, d), input_batch_dims_B=(1,), name="mma_5", **kw)
        wire_input(div, last, 0)
    return Graph(nodes=find_topo_sort([OutputNode(last, dtype="float16")]))


def test_exact_combine_keeps_what_follows_the_sums():
    combine = exact_combine(_hand_combine(), (32, 16, 64), 1)
    (w,) = exps(combine)
    assert w.stabilize == Stabilize(axes=(-2,), scaled=w.stabilize.scaled)
    assert sorted(name for name, _ in w.stabilize.scaled) == ["reduction_1_2", "reduction_1_3"]
    (div,) = [n for n in combine.nodes if type(n) is Div]
    assert _reads(div) == [("reduction_1_3", 0), ("reduction_1_2", 0)]


@pytest.mark.parametrize("build, reader", [
    (lambda: _hand_combine(projected=True), r"PlaceHolder_\w+ is read by Matmul mma_5"),
    (lambda: _hand_combine(product=True), r"PlaceHolder_\w+ is read by Mul mul_1"),
    (lambda: _hand_combine(other_axis=True), r"PlaceHolder_\w+ is read by ReduceSum reduction_1_2"),
], ids=["output_projection", "partials_multiplied", "summed_over_the_rows"])
def test_exact_combine_refuses_an_input_that_is_not_summed_over_the_split_axis(build, reader):
    """Every boundary input needs a place for its weight: a sum over the split axis that
    reads it. An input of an output projection behind the sums has none (it is not a
    partial), nor has a partial that something else reads; the combine is refused, loudly,
    instead of being rebuilt without what it computes."""
    with pytest.raises(ScaleError, match="^combine: not every boundary input is summed over the "
                                         rf"split axis \({reader}\)$"):
        exact_combine(build(), (32, 16, 64), 1)


def test_exact_combine_refuses_partials_of_another_shape_than_the_shift():
    with pytest.raises(ScaleError, match=r"^combine: PlaceHolder_\w+ \(32, 16, 64(, 64)?\) is "
                                         r"not the shift \(32, 16, 32\) with at most one "
                                         "trailing axis$"):
        exact_combine(_hand_combine(), (32, 16, 32), 1)


def test_exact_combine_holds_its_weights_to_the_solo_rule():
    """A split axis that is a batch axis of the weights: the solo rule's refusal, with its
    reason. (The exp has two non-batch axes, the last two of the shift.)"""
    bh, f, sq = 4, 8, 16
    kw = dict(dtype="float16", out_dtype="float32")
    sum_l = ReduceSum((f, bh, sq), axis=0, name="reduction_0_2", **kw)
    sum_l.inputs[0].src_node.boundary = Boundary(0)
    combine = Graph(nodes=find_topo_sort([OutputNode(sum_l, dtype="float16")]))
    with pytest.raises(ScaleError, match="exp_shift: output of reduction_0_2 keeps degree 1"):
        exact_combine(combine, (f, bh, sq), 0)


class _Requests:
    """What ``stabilize_split`` reads of a program."""

    def __init__(self, requests, plain_exp=False):
        from cleave.frontend.program import Op

        self.stabilize_requests = tuple((i, 1) for i in range(requests))
        self.ops = [Op("exp", (0,), (), 1)] if plain_exp else []


def test_a_split_carries_one_softmax():
    for requests in (0, 2):
        with pytest.raises(ScaleError, match=f"^{requests} softmax requests; a split carries "
                                             "one softmax$"):
            stabilize_split(*_plain_pair(1), _Requests(requests))


def test_an_exp_the_program_never_wrote_is_not_left_plain():
    """An Exp next to the softmax's stays a plain exp. That is only right when the
    program wrote one; otherwise it is a softmax the pair cannot serve."""
    def with_second_exp():
        partial, combine = _plain_pair(1)
        (div,) = [n for n in combine.nodes if type(n) is Div]
        (out,) = [n for n in combine.nodes if isinstance(n, OutputNode)]
        second = Exp(tuple(int(s) for s in div.get_shape(0)), name="exp_9", dtype="float32",
                     out_dtype="float32", n_batch_dims=1)
        wire_input(div, second, 0)
        wire_input(second, out, 0)
        return partial, Graph(nodes=find_topo_sort([out]))

    with pytest.raises(ScaleError, match=r"^\['exp_9'\] would stay plain exps, and the program "
                                         "wrote none$"):
        stabilize_split(*with_second_exp(), _Requests(1))
    partial, combine = stabilize_split(*with_second_exp(), _Requests(1, plain_exp=True))
    assert {e.name: e.stabilize is not None for e in exps(combine)} == {
        "exp_shift": True, "exp_9": False}


def test_a_refused_pair_is_refused_with_the_reason_at_lowering():
    graph = SearchedGraph(0, _structure(1), dict(PL.SYMBOL_MAPS["gqa"]), True,
                          _Requests(2), ("SKV",))
    with pytest.raises(LoweringRejected) as e:
        lower(graph, FACTOR)
    assert str(e.value) == ("graph 0: softmax cannot be stabilized: 2 softmax requests; a split "
                            "carries one softmax")


# ----------------------------------------------------------------------------- references

def _truth(q, k, v):
    """float32 torch attention with grouped key heads."""
    group = q.shape[0] // k.shape[0]
    k, v = (x.float().repeat_interleave(group, 0) for x in (k, v))
    return torch.softmax((q.float() @ k.mT) / q.shape[-1] ** 0.5, -1) @ v


def _row_error(out, truth):
    return float(((out.float() - truth).abs() / truth.abs().amax(-1, keepdim=True)).max())


@pytest.mark.parametrize("n", VALID)
@pytest.mark.parametrize("scale", [1.0, 100.0], ids=["x1", "x100"])
def test_the_chained_references_compute_the_program(n, scale):
    """Partial, then combine, handed over by boundary record: the program, at any logit
    scale. Without the records on the two exps the pair overflows at x100."""
    program = _program()
    partial, combine = _exact_pair(n, program)
    torch.manual_seed(0)
    q, k, v = (torch.randn(s.shape).to(torch.float16) for s in program.inputs)
    q = (q.float() * scale).to(torch.float16)
    crossing, (out,) = chained_reference(partial, combine, program, [q, k, v])
    assert [str(x.dtype) for x in crossing] == ["torch.float16", "torch.float16",
                                               "torch.float32"]
    assert all(torch.isfinite(x).all() for x in (*crossing, out))
    # the shift is the max of the scores of each split, exactly as the kernel has to return it
    assert _row_error(out, _truth(q, k, v)) <= 2e-3
    (reference,) = program.reference(q, k, v)
    assert _row_error(out, reference.float()) <= 2e-3

    marks = [(e, e.stabilize) for g in (partial, combine) for e in exps(g)]
    for e, _ in marks:
        e.stabilize = None
    _, (plain,) = chained_reference(partial, combine, program, [q, k, v])
    for e, record in marks:
        e.stabilize = record
    assert bool(torch.isfinite(plain).all()) == (scale == 1.0)


def _combine_inputs(combine, l, o, m):
    by_role = {0: l, 1: o, 2: m}
    return [by_role[p.boundary.index] for p in placeholder_order(combine)]


def test_a_split_that_exports_the_sentinel_takes_no_part():
    """A partial that saw no key returns ``l = o = 0`` and the start value of its carried
    max. Its weight is ``exp(-FLT_MAX - M) = 0``: the other splits give the answer, and
    nothing in the combine is NaN."""
    _, combine = _exact_pair(1)
    torch.manual_seed(1)
    l = (torch.rand(ROWSUM) + 0.5).to(torch.float16)
    o = torch.randn(VALUE).to(torch.float16)
    m = torch.randn(MAX) * 30
    l[:, 2], o[:, 2], m[:, 2] = 0, 0, NEG_FLT_MAX
    (out,) = execute(combine, _combine_inputs(combine, l, o, m))
    assert torch.isfinite(out).all()
    keep = [0, 1, 3]
    w = torch.exp(m[:, keep] - m[:, keep].amax(1, keepdim=True))
    want = (w[..., None] * o[:, keep].float()).sum(1) / (w * l[:, keep].float()).sum(1)[..., None]
    assert _row_error(out, want) <= 1e-3


@pytest.mark.parametrize("spread", [1.0, 100.0, 1.0e4])
def test_the_combine_reference_weights_by_the_exported_max(spread):
    """``sum_i exp(m_i - M) o_i / sum_i exp(m_i - M) l_i``: finite whatever the maxima are,
    where the plain ``exp(m_i)`` it is derived from overflows."""
    _, combine = _exact_pair(1)
    torch.manual_seed(2)
    l = (torch.rand(ROWSUM) + 0.5).to(torch.float16)
    o = torch.randn(VALUE).to(torch.float16)
    m = torch.randn(MAX) * spread
    (out,) = execute(combine, _combine_inputs(combine, l, o, m))
    w = torch.exp(m - m.amax(1, keepdim=True))
    want = (w[..., None] * o.float()).sum(1) / (w * l.float()).sum(1)[..., None]
    assert torch.isfinite(out).all() and _row_error(out, want) <= 1e-3
