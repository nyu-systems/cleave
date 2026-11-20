"""CPU: ``apply_tiling`` records how the kernel stores each output (``Graph.stored_layouts``).

The stored layout is decided inside ``apply_tiling`` and depends on the tiling and on the
extents, not on the role of the output: a split partial with several loop-fed outputs
stores them with the split axis first, unless the split extent is counted as the K/V head
axis, in which case it stores the logical layout. Nothing outside ``apply_tiling`` can
recover that from two shapes (a split extent equal to the batch gives equal shapes for
both layouts), so it is recorded where it is decided.
"""

import json

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.ir.graph import OutputNode
from cleave.schedule import apply_tiling, search_tiling

from tests import _graphs as G
from tests.pipeline import _pipeline as PL

IDENTITY3, IDENTITY4 = ((0, 1, 2), 0), ((0, 1, 2, 3), 0)
SPLIT_FIRST = {"OutputNode_0": ((1, 0, 2), 1), "OutputNode_1": ((1, 0, 2, 3), 0)}

# builder -> (stored layouts, stored shapes) of every tiling
PINNED = {
    "mha_partial": (
        lambda: G.mha_split_partial_graph(1, 32, 4, 4096, 1024, 64),
        SPLIT_FIRST, {"OutputNode_0": (4, 32, 4096, 1), "OutputNode_1": (4, 32, 4096, 64)},
    ),
    # the split extent equals B*H: stored split first, in the shape of the logical layout
    "mha_partial_split_equals_heads": (
        lambda: G.mha_split_partial_graph(1, 32, 32, 64, 128, 64),
        SPLIT_FIRST, {"OutputNode_0": (32, 32, 64, 1), "OutputNode_1": (32, 32, 64, 64)},
    ),
    "gqa_partial": (
        lambda: G.gqa_split_partial_graph(1, 32, 8, 4, 4096, 1024, 64),
        SPLIT_FIRST, {"OutputNode_0": (4, 32, 4096, 1), "OutputNode_1": (4, 32, 4096, 64)},
    ),
    # the split extent equals H_KV: the same operator stores the logical layout
    "gqa_partial_split_equals_kv_heads": (
        lambda: G.gqa_split_partial_graph(1, 32, 8, 8, 4096, 512, 64),
        {"OutputNode_0": ((0, 1, 2), 1), "OutputNode_1": IDENTITY4},
        {"OutputNode_0": (32, 8, 4096, 1), "OutputNode_1": (32, 8, 4096, 64)},
    ),
    "mha_combine": (
        lambda: G.mha_split_combine_graph(32, 4, 4096, 64),
        {"OutputNode": IDENTITY3}, {"OutputNode": (32, 4096, 64)},
    ),
    "matmul_splitk_partial": (
        lambda: G.matmul_splitk_partial_graph(4, 128, 128, 1024),
        {"OutputNode": IDENTITY3}, {"OutputNode": (4, 128, 128)},
    ),
    "mha_solo": (
        lambda: G.mha_bhsd_graph(1, 32, 4096, 4096, 64),
        {"OutputNode": IDENTITY3}, {"OutputNode": (32, 4096, 64)},
    ),
}


def _outputs(graph):
    return [n for n in graph.nodes if isinstance(n, OutputNode)]


def _stored_shape(logical, layout):
    axes, pad = layout
    return tuple(logical[a] for a in axes) + (1,) * pad


@pytest.mark.parametrize("case", sorted(PINNED))
def test_stored_layouts_of_hand_built_graphs(case):
    build, layouts, shapes = PINNED[case]
    graph = build()
    candidates = search_tiling(graph, topk=PL.TOPK)
    assert candidates
    for cand in candidates:
        scheduled = apply_tiling(cand)
        assert scheduled.stored_layouts == layouts
        assert {o.name: tuple(scheduled.tmap[o].shape) for o in _outputs(scheduled)} == shapes
        # per candidate: on the scheduled clone, never on the lowered graph they all share
        assert cand.flat_graph.stored_layouts == {} and graph.stored_layouts == {}
        clone = scheduled.clone()
        assert clone.stored_layouts == layouts and clone.stored_layouts is not scheduled.stored_layouts


def _t(guid, dims):
    return {"guid": guid, "num_dims": len(dims), "dim": list(dims), "dtype": "float16"}


def _op(op_type, ins, outs, **extra):
    return {"op_type": op_type, "input_tensors": ins, "output_tensors": outs, **extra}


def _rowsum_and_value(BH, SQ, SKV, D):
    """exp(Q K^T) reduced two ways, both outputs of one unsplit graph."""
    q, k, v = _t(1, (BH, SQ, D)), _t(2, (BH, SKV, D)), _t(3, (BH, SKV, D))
    scores, weights = _t(4, (BH, SQ, SKV)), _t(5, (BH, SQ, SKV))
    return [
        _op("kn_input_op", [], [q]),
        _op("kn_input_op", [], [k]),
        _op("kn_input_op", [], [v]),
        _op("kn_matmul_op", [q, k], [scores], trans_A=False, trans_B=True),
        _op("kn_exp_op", [scores], [weights]),
        _op("kn_reduce_sum_2_op", [weights], [_t(6, (BH, SQ, 1))]),
        _op("kn_matmul_op", [weights, v], [_t(7, (BH, SQ, D))], trans_A=False, trans_B=False),
        _op("kn_output_op", [_t(6, (BH, SQ, 1))], []),
        _op("kn_output_op", [_t(7, (BH, SQ, D))], []),
    ]


@pytest.mark.parametrize("dims", [(32, 4096, 4096, 64), (4, 64, 1024, 64)])
def test_a_padded_sibling_output_without_a_split_records_its_pad(dims):
    """Two loop-fed outputs and no split: nothing is reordered, but the row sum is still
    padded to the rank of the value output. The pad has to be in the record, or a correct
    kernel of such a program is refused for a shape it does have."""
    from cleave.ir.graph import GraphNode
    from cleave.lower import to_ir

    BH, SQ, _, D = dims
    graph = to_ir(_rowsum_and_value(*dims))
    candidates = search_tiling(graph, topk=PL.TOPK)
    assert candidates
    for cand in candidates:
        scheduled = apply_tiling(cand)
        rowsum, value = _outputs(scheduled)
        assert all(isinstance(o.inputs[0].src_node, GraphNode) for o in (rowsum, value))
        assert scheduled.stored_layouts == {rowsum.name: ((0, 1), 1), value.name: IDENTITY3}
        assert tuple(scheduled.tmap[rowsum].shape) == (BH, SQ, 1)
        assert tuple(scheduled.tmap[value].shape) == (BH, SQ, D)


@pytest.mark.parametrize("name", PL.fixture_names())
def test_stored_shape_formula_equals_tmap(name):
    """``stored_shape == tuple(logical[a] for a in axes) + (1,) * pad`` for every output of
    every tiling the snapshot fixtures schedule (the kernel declares ``tmap[output].shape``)."""
    scheduled_candidates = 0
    for bundle in PL._bundles(name):
        for graph in bundle:
            try:
                candidates = search_tiling(graph, topk=PL.TOPK)
            except Exception:  # noqa: BLE001 - a fixture with no tiling is in the snapshot too
                continue
            for cand in candidates:
                try:
                    scheduled = apply_tiling(cand)
                except Exception:  # noqa: BLE001 - a tiling apply_tiling refuses records nothing
                    continue
                scheduled_candidates += 1
                logical = _outputs(cand.flat_graph)
                stored = _outputs(scheduled)
                # same outputs, same names, same order as the lowered graph: the reference
                # (one per lowered graph) and the kernel's outputs line up by position
                assert [o.name for o in stored] == [o.name for o in logical]
                assert set(scheduled.stored_layouts) == {o.name for o in stored}
                for flat, out in zip(logical, stored):
                    shape = tuple(int(s) for s in flat.get_shape(0))
                    layout = scheduled.stored_layouts[out.name]
                    assert sorted(layout[0]) == list(range(len(shape))) and layout[1] >= 0
                    assert _stored_shape(shape, layout) == tuple(
                        int(s) for s in scheduled.tmap[out].shape), (name, out.name, layout)
    # Not vacuous: every tiling the checked-in snapshot has a source for was checked here.
    snapshot = json.loads(PL.snapshot_path(name).read_text())
    with_source = sum(not digest.startswith("error:") for stages in snapshot["bundles"]
                      for stage in stages for digest in stage.get("sources", ()))
    assert scheduled_candidates >= with_source
