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

from cleave.core.errors import ScheduleRejected
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


def _t(guid, dims, dtype="float16"):
    return {"guid": guid, "num_dims": len(dims), "dim": list(dims), "dtype": dtype}


def _op(op_type, ins, outs, **extra):
    return {"op_type": op_type, "input_tensors": ins, "output_tensors": outs, **extra}


def _rowsum_and_value(BH, SQ, SKV, D, weights="float32"):
    """exp(Q K^T) reduced two ways, both outputs of one unsplit graph.

    ``weights`` is the dtype the exp is stored in: float32 is what lowering makes of a
    traced program. Stored in float16, the row sum's own first step is twice the value
    matmul's (the policy sizes each reduction's step by what it reads); the two share a
    loop over ``S_KV``, so the policy starts them together.
    """
    q, k, v = _t(1, (BH, SQ, D)), _t(2, (BH, SKV, D)), _t(3, (BH, SKV, D))
    scores, weights = _t(4, (BH, SQ, SKV)), _t(5, (BH, SQ, SKV), weights)
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


@pytest.mark.parametrize("weights", ["float32", "float16"])
@pytest.mark.parametrize("dims", [(32, 4096, 4096, 64), (4, 64, 1024, 64)])
def test_a_padded_sibling_output_without_a_split_records_its_pad(dims, weights):
    """Two loop-fed outputs and no split: nothing is reordered, but the row sum is still
    padded to the rank of the value output. The pad has to be in the record, or a correct
    kernel of such a program is refused for a shape it does have."""
    from cleave.ir.graph import GraphNode
    from cleave.lower import to_ir

    BH, SQ, _, D = dims
    graph = to_ir(_rowsum_and_value(*dims, weights=weights))
    candidates = search_tiling(graph, topk=PL.TOPK)
    assert candidates
    for cand in candidates:
        steps = {n.name: next(iter(v.values())) for n, v in cand.rstep_map.items() if v}
        assert steps["reduction_2_5"] == steps["mma_6"]  # one loop, one step
        scheduled = apply_tiling(cand)
        rowsum, value = _outputs(scheduled)
        assert all(isinstance(o.inputs[0].src_node, GraphNode) for o in (rowsum, value))
        assert scheduled.stored_layouts == {rowsum.name: ((0, 1), 1), value.name: IDENTITY3}
        assert tuple(scheduled.tmap[rowsum].shape) == (BH, SQ, 1)
        assert tuple(scheduled.tmap[value].shape) == (BH, SQ, D)


@pytest.mark.parametrize("dims", [(32, 4096, 4096, 64), (4, 64, 1024, 64)])
def test_siblings_stepped_apart_are_refused_and_record_nothing(dims):
    """The same graph with the exp stored in float16, and the step maps the policy used to
    propose for it: the row sum stepping twice as far as the value matmul. Both are in one
    loop over ``S_KV``, which then runs twice as many iterations as the weights have
    tiles. The tiler refuses that (it used to build a kernel that read past the row, and
    this file recorded its layouts), so there is no stored layout to speak of."""
    from cleave.arch import get_arch
    from cleave.lower import to_ir
    from cleave.schedule.policy.default import TCPolicy

    SKV = dims[2]
    graph = to_ir(_rowsum_and_value(*dims, weights="float16"))
    policy = TCPolicy([n for n in graph.nodes if isinstance(n, OutputNode)], get_arch("cuda"),
                      flat_graph=graph)
    candidates = search_tiling(graph, topk=PL.TOPK)
    assert candidates
    for together in candidates:
        rstep = {n: {k: step // 2 for k, step in v.items()} if n.name == "mma_6" else v
                 for n, v in together.rstep_map.items()}
        cand = policy.compute_tile_dict(list(together.output_tile), rstep)
        cand.n_threads = together.n_threads
        steps = {n.name: next(iter(v.values())) for n, v in cand.rstep_map.items() if v}
        rowsum, value = steps["reduction_2_5"], steps["mma_6"]
        assert cand.valid and rowsum == 2 * value
        with pytest.raises(ScheduleRejected) as refusal:
            apply_tiling(cand)
        assert str(refusal.value) == (
            f"GraphNode_mma_6 runs {SKV // value} iterations but dim 1 of exp_4 is cut into "
            f"{SKV // rowsum} tiles ({rowsum} of {SKV})"
        )
        assert cand.flat_graph.stored_layouts == {} and graph.stored_layouts == {}


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
