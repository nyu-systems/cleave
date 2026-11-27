"""``cleave.lower.state_split``: a split pair built from the loop state (CPU only).

Nothing in the pipeline calls the module yet. These tests hold its two builders to the one
thing they promise: the partial of every split, handed to the combine, is the solo graph --
at the level of the torch reference, on normal and adversarial logits.
"""

import functools

import pytest

torch = pytest.importorskip("torch")

import cleave.lang as cl
from cleave import pipeline as P
from cleave.frontend import trace
from cleave.ir.graph_utils import boundary_signature
from cleave.ir.ops import Matmul, ReduceMax, ReduceSum, SplitView
from cleave.ir.update_rule import signature
from cleave.lower.state_split import (
    SplitRefused,
    combine_graph,
    lower_partial,
    op_indices,
    partial_structure,
)
from cleave.pipeline.plans import lower
from cleave.runtime.reference import execute, make_inputs, output_order, placeholder_order
from cleave.runtime.validate import compare_outputs
from tests._tolerances import tolerance_for

from ._attention import FACTOR, attention, flash

SHAPES = dict(H=4, SQ=8, SKV=256, D=16)


@functools.lru_cache(maxsize=None)
def solo():
    """``(searched flash graph, its lowered IR, full-extent node name, record)``."""
    program = trace(attention, SHAPES, cl.f16)
    graph = flash(P.superoptimize(program, P.SuperoptConfig(max_ops=11, cache=False)))
    (ir,) = lower(graph)
    (full,) = [n for n in ir.nodes if getattr(n, "streaming", None) is not None]
    return graph, ir, full.name, full.streaming


def test_search_the_state_is_found_by_structure_index():
    graph, ir, full, record = solo()
    full_op, acc_ops = op_indices(ir, full, record.loops)
    ops = [op["op_type"] for op in graph.structure]
    assert ops[full_op].startswith("kn_reduce_max_")
    by_name = {n.name: n for n in ir.nodes}
    kinds = {type(by_name[acc]): ops[index] for acc, index in zip(record.loops, acc_ops)}
    assert kinds[Matmul] == "kn_matmul_op" and kinds[ReduceSum].startswith("kn_reduce_sum_")


def test_search_partial_structure_replays_the_solo_graph_on_a_split_of_the_looped_dim():
    graph, ir, full, record = solo()
    full_op, acc_ops = op_indices(ir, full, record.loops)
    ops, factor, looped, inner = partial_structure(graph.structure, full_op, acc_ops)
    assert factor in (2, 3, 5, 7) and looped == factor * inner
    kinds = [op["op_type"] for op in ops]
    assert kinds == [
        "kn_input_op", "kn_input_op", "kn_input_op",
        "kn_split_1_op", "kn_split_1_op",  # K and V, along the keys; Q is fixed context
        "kn_matmul_op", "kn_div_op", "kn_reduce_max_3_op", "kn_sub_op", "kn_exp_op",
        "kn_reduce_sum_3_op", "kn_matmul_op",
        "kn_output_op", "kn_output_op", "kn_output_op"]
    # the divide behind the loop state is not part of it
    assert sum(kind == "kn_div_op" for kind in kinds) == 1
    # every state tensor leads with the split factor and has the looped dim cut
    outputs = [op["input_tensors"][0]["dim"][: op["input_tensors"][0]["num_dims"]]
               for op in ops if op["op_type"] == "kn_output_op"]
    assert all(dims[0] == factor and looped not in dims for dims in outputs)
    with pytest.raises(SplitRefused, match="is not a reduction of one axis"):
        partial_structure(graph.structure, acc_ops[0], acc_ops)


@pytest.mark.parametrize("split_factor", [4, 16])
def test_search_partial_then_combine_is_the_solo_graph(split_factor, monkeypatch):
    # (References on the CPU, no kernel: nothing here belongs in a measurement.)
    monkeypatch.delenv("CLEAVE_VALIDATION_LOG", raising=False)
    graph, ir, full, record = solo()
    full_op, acc_ops = op_indices(ir, full, record.loops)
    assert dict(record.rules) == {acc: FACTOR for acc in record.loops}

    partial = lower_partial(graph, full_op, acc_ops, split_factor)
    outs = output_order(partial)
    # the max first (role "shift"), then the accumulators in the record's order; all
    # float32 at the cut, each with the split axis behind the batch dim
    assert [(o.boundary.index, o.boundary.role, str(o.get_dtype(0))) for o in outs] == [
        (2, "shift", "float32"), (0, "", "float32"), (1, "", "float32")]
    by_index = {o.boundary.index: tuple(int(s) for s in o.get_shape(0)) for o in outs}
    by_name = {n.name: n for n in ir.nodes}
    for index, acc in enumerate(record.loops):
        whole = tuple(int(s) for s in by_name[acc].get_shape())
        assert by_index[index] == whole[:1] + (split_factor,) + whole[1:]
    assert by_index[2] == (4, split_factor, 8)
    assert len([n for n in partial.nodes if type(n) is SplitView]) == 2
    (own_max,) = [n for n in partial.nodes if type(n) is ReduceMax]
    assert own_max.get_shape()[1] == split_factor  # the max of its OWN split

    combine = combine_graph(ir, full, record.rules, split_factor, split_pos=1,
                            out_dtype=graph.dtype)
    # the two sides of the cut carry the same tensors
    placeholders = [n for n in combine.nodes if n.is_placeholder()]
    assert boundary_signature(placeholders) == boundary_signature(outs)
    names = {n.name for n in combine.nodes}
    assert {"state_max", "state_upd0", "state_upd1"} <= names
    assert {f"{acc}_repaired" for acc in record.loops} | {
        f"{acc}_combined" for acc in record.loops} <= names
    merged = next(n for n in combine.nodes if n.name == "state_max")
    assert type(merged) is ReduceMax and merged.flat_axis == 1
    # the solo graph is left as it was
    assert by_name[full].streaming is record
    assert "state_max" not in {n.name for n in ir.nodes}

    tolerance = tolerance_for("exact_split_attention")[""]
    for case in ("normal", "large_logits", "offset"):
        inputs = make_inputs(ir, "float16", seed=0, device="cpu", distribution=case)
        (want,) = execute(ir, inputs)
        given = {p.name: x for p, x in zip(placeholder_order(ir), inputs)}
        state = execute(partial, [given[p.name] for p in placeholder_order(partial)])
        handed = {o.boundary.index: x for o, x in zip(outs, state)}
        (got,) = execute(combine, [handed[p.boundary.index] for p in placeholder_order(combine)])
        assert torch.isfinite(got).all() and got.shape == want.shape
        compare_outputs([got], [want], tolerance, what=f"{case} inputs")


def test_search_the_partials_of_every_split_factor_are_one_loop_structure():
    """The number of splits is an extent: it is not in the signature a rule is keyed by."""
    graph, ir, full, record = solo()
    full_op, acc_ops = op_indices(ir, full, record.loops)
    four, sixteen = (lower_partial(graph, full_op, acc_ops, f) for f in (4, 16))
    assert signature(four, dtypes=True) == signature(sixteen, dtypes=True)
    assert signature(four, dtypes=True) != signature(ir, dtypes=True)


def test_search_combine_refuses_a_factor_it_cannot_broadcast():
    graph, ir, full, record = solo()
    backwards = tuple((acc, ("exp", ("sub", "m'", "m"))) for acc in record.loops)
    with pytest.raises(SplitRefused, match="on the left of 'sub' has no tile op"):
        combine_graph(ir, full, backwards, 4, split_pos=1, out_dtype=graph.dtype)
