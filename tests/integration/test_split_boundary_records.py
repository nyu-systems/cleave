"""CPU: ``to_ir`` records which tensor every placeholder and output is (``Boundary``), and
the combine of a split pair is aligned with its partial by that record.

A lowered graph lists its nodes in the order a walk from its first output discovers them,
so the declaration order of the structure cannot be read back from the graph; with two
outputs the two orders happen to agree, with three they do not. The cut defines the
boundary by declaration order: the partial's k-th output op is the combine's k-th input op.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("tilelang")

from cleave.ir.graph import Boundary, OutputNode, PlaceHolderNode
from cleave.ir.graph_utils import boundary_order, boundary_signature
from cleave.lower import process as gp
from cleave.lower import split as cut
from cleave.lower import to_ir
from cleave.lower.process import to_bundles
from tests.pipeline._pipeline import SYMBOL_MAPS

STRUCTURES = Path(__file__).parent / "graph_structures"
SPLIT_FIXTURES = sorted(
    f"{path.parent.name}/{path.stem}"
    for path in STRUCTURES.glob("*/before_*.graph")
    if cut.is_split_graph(json.loads(path.read_text()))
)


def _load(name):
    return json.loads((STRUCTURES / f"{name}.graph").read_text())


def _placeholders(graph):
    return [n for n in graph.nodes if isinstance(n, PlaceHolderNode)]


def _outputs(graph):
    return [n for n in graph.nodes if isinstance(n, OutputNode)]


def _t(guid, dims):
    return {"guid": guid, "num_dims": len(dims), "dim": list(dims), "dtype": "float32"}


def _op(op_type, ins, outs, **extra):
    return {"op_type": op_type, "input_tensors": ins, "output_tensors": outs, **extra}


def test_split_fixtures_exist():
    families = {name.split("/")[0] for name in SPLIT_FIXTURES}
    assert families == {"gqa", "matmul", "rmsnorm_matmul"}, SPLIT_FIXTURES


@pytest.mark.parametrize("name", SPLIT_FIXTURES)
@pytest.mark.parametrize("factor", [4, 8, 16, 32])
def test_split_pair_carries_the_cut_order(name, factor):
    structure = _load(name)
    family = name.split("/")[0]
    symbols = dict(SYMBOL_MAPS[family])
    if family == "gqa":  # 8 is both B*H_Q and the split extent of S_KV: S_KV = factor * B*H_Q
        symbols[16] = factor * symbols[8]
    ((partial, combine),) = to_bundles([structure], symbol_map=symbols, split_factors=[factor])
    (partial_ops, combine_ops), _ = cut.cut_split_graph(structure)

    # every tensor of both graphs is numbered, 0..n-1, with the plain role
    for graph in (partial, combine):
        for nodes in (_placeholders(graph), _outputs(graph)):
            assert sorted(n.boundary.index for n in nodes) == list(range(len(nodes)))
            assert {n.boundary.role for n in nodes} == {""}

    # the combine's k-th input op is its placeholder k ...
    input_guids = [op["output_tensors"][0]["guid"] for op in combine_ops
                   if op["op_type"] == "kn_input_op"]
    by_name = {ph.name: ph.boundary.index for ph in _placeholders(combine)}
    assert by_name == {f"PlaceHolder_input_{guid}": k for k, guid in enumerate(input_guids)}

    # ... and the partial's k-th output op is its output k (nodes are named by op index)
    producer = {t["guid"]: i for i, op in enumerate(partial_ops) for t in op["output_tensors"]}
    declared = [producer[op["input_tensors"][0]["guid"]] for op in partial_ops
                if op["op_type"] == "kn_output_op"]
    for out in _outputs(partial):
        assert out.inputs[0].src_node.name.endswith(f"_{declared[out.boundary.index]}")

    # so the two sides of the cut are the same tensors: index, role, shape and dtype
    crossing = boundary_signature(_outputs(partial))
    assert crossing == boundary_signature(_placeholders(combine))
    assert [index for index, *_ in crossing] == list(range(len(input_guids)))
    assert all(factor in shape for _, _, shape, _ in crossing)


def test_solo_graph_numbers_inputs_and_outputs_by_declaration():
    graph = to_ir([
        _op("kn_input_op", [], [_t(1, (4, 8))]),
        _op("kn_input_op", [], [_t(2, (8, 16))]),
        _op("kn_matmul_op", [_t(1, (4, 8)), _t(2, (8, 16))], [_t(3, (4, 16))]),
        _op("kn_output_op", [_t(3, (4, 16))], []),
    ])
    assert {ph.name: ph.boundary for ph in _placeholders(graph)} == {
        "PlaceHolder_input_1": Boundary(0), "PlaceHolder_input_2": Boundary(1)}
    assert [out.boundary for out in _outputs(graph)] == [Boundary(0, "")]
    # the record is plain data on the node: a clone keeps it
    clone = graph.clone()
    assert [n.boundary for n in clone.nodes if n.is_output() or n.is_placeholder()] == [
        n.boundary for n in graph.nodes if n.is_output() or n.is_placeholder()]


def test_an_input_no_output_reads_keeps_its_place_in_the_count():
    """``to_ir`` does not lower an op that no output reads, so such an input has no
    placeholder. It is still an input op of the structure: the inputs declared after it
    keep their index. Counting the inputs that were converted instead numbered the third
    input 1, the place of the one that is not there, while ``input_index`` (which binds
    the program's inputs) said 2."""
    graph = to_ir([
        _op("kn_input_op", [], [_t(1, (4, 8))]),
        _op("kn_input_op", [], [_t(9, (4, 8))]),  # declared, read by nothing
        _op("kn_input_op", [], [_t(2, (8, 16))]),
        _op("kn_matmul_op", [_t(1, (4, 8)), _t(2, (8, 16))], [_t(3, (4, 16))]),
        _op("kn_output_op", [_t(3, (4, 16))], []),
    ])
    assert {ph.name: ph.boundary for ph in _placeholders(graph)} == {
        "PlaceHolder_input_1": Boundary(0), "PlaceHolder_input_2": Boundary(2)}
    assert [ph.input_index for ph in boundary_order(_placeholders(graph))] == [0, 2]


def _three_outputs():
    x = _t(1, (4, 8))
    return [
        _op("kn_input_op", [], [x]),
        _op("kn_exp_op", [x], [_t(2, (4, 8))]),
        _op("kn_reduce_sum_1_op", [_t(2, (4, 8))], [_t(3, (4, 1))]),
        _op("kn_mul_op", [_t(2, (4, 8))], [_t(4, (4, 8))], scalar=2.0),
        _op("kn_reduce_max_1_op", [x], [_t(5, (4, 1))]),
        _op("kn_output_op", [_t(3, (4, 1))], []),  # 0: the row sum
        _op("kn_output_op", [_t(4, (4, 8))], []),  # 1: the scaled exp
        _op("kn_output_op", [_t(5, (4, 1))], []),  # 2: the row max
    ]


def test_three_outputs_are_ordered_by_record_not_by_the_node_list():
    graph = to_ir(_three_outputs())
    outputs = _outputs(graph)
    producers = lambda nodes: [type(n.inputs[0].src_node).__name__ for n in nodes]  # noqa: E731
    # The walk from the row sum reaches the row max (through the shared input) before the
    # scaled exp: the node list has the outputs in an order nobody declared ...
    assert producers(outputs) == ["ReduceSum", "ReduceMax", "Mul"]
    # ... and under names that follow that order, so a name is no identity either.
    assert [n.name for n in outputs] == ["OutputNode", "OutputNodf", "OutputNodg"]
    # The record is the declaration order.
    ordered = boundary_order(outputs)
    assert producers(ordered) == ["ReduceSum", "Mul", "ReduceMax"]
    assert [n.boundary for n in ordered] == [Boundary(0), Boundary(1), Boundary(2)]
    assert [shape for _, _, shape, _ in boundary_signature(outputs)] == [(4,), (4, 8), (4,)]


def test_boundary_order_refuses_to_guess():
    graph = to_ir(_three_outputs())
    outputs = _outputs(graph)
    hand_built = [n for n in graph.clone().nodes if n.is_output()]
    for node in hand_built:
        node.boundary = None
    # no records at all: a hand-built graph; the caller keeps its own rule
    assert boundary_order(hand_built) is None and boundary_signature(hand_built) is None
    assert boundary_order([]) is None
    # a tensor somebody added without saying which one it is
    hand_built[0].boundary = Boundary(0)
    with pytest.raises(ValueError, match="only some"):
        boundary_order(hand_built)
    # two tensors claiming one index
    outputs[1].boundary = Boundary(0, "shift")
    with pytest.raises(ValueError, match="used twice"):
        boundary_order(outputs)


def _three_output_bundle():
    """A normalised split bundle (split dim first, rank 3) whose partial declares a row sum,
    a scaled exp and a row max -- the outputs of an exact-softmax partial -- and whose
    combine declares its inputs in that order, as a cut does."""
    x = _t(1, (2, 4, 8))  # (split, rows, columns)
    partial = [
        _op("kn_input_op", [], [x]),
        _op("kn_exp_op", [x], [_t(2, (2, 4, 8))]),
        _op("kn_reduce_sum_2_op", [_t(2, (2, 4, 8))], [_t(3, (2, 4, 1))]),
        _op("kn_mul_op", [_t(2, (2, 4, 8))], [_t(4, (2, 4, 8))], scalar=2.0),
        _op("kn_reduce_max_2_op", [x], [_t(5, (2, 4, 1))]),
        _op("kn_output_op", [_t(3, (2, 4, 1))], []),  # 0: the row sum
        _op("kn_output_op", [_t(4, (2, 4, 8))], []),  # 1: the scaled exp
        _op("kn_output_op", [_t(5, (2, 4, 1))], []),  # 2: the row max
    ]
    combine = [
        _op("kn_input_op", [], [_t(11, (2, 4, 1))]),
        _op("kn_input_op", [], [_t(12, (2, 4, 8))]),
        _op("kn_input_op", [], [_t(13, (2, 4, 1))]),
        _op("kn_reduce_sum_0_op", [_t(11, (2, 4, 1))], [_t(14, (4, 1))]),
        _op("kn_reduce_sum_0_op", [_t(12, (2, 4, 8))], [_t(15, (4, 8))]),
        _op("kn_reduce_max_0_op", [_t(13, (2, 4, 1))], [_t(16, (4, 1))]),
        _op("kn_output_op", [_t(14, (4, 1))], []),
        _op("kn_output_op", [_t(15, (4, 8))], []),
        _op("kn_output_op", [_t(16, (4, 1))], []),
    ]
    return [partial, combine]


def test_combine_inputs_take_the_partial_outputs_in_boundary_order():
    """``_align_split_boundary_inputs`` gives the combine's k-th input the shape the partial
    stores its k-th output in. The k-th output is the one with boundary index k: the
    partial's node list has its three outputs as (row sum, row max, scaled exp), and read
    in that order the scaled exp would be given the shape of the row max."""
    bundle = _three_output_bundle()
    partial = to_ir(bundle[0])
    listed = [tuple(out.get_shape()) for out in _outputs(partial)]
    declared = [tuple(out.get_shape()) for out in boundary_order(_outputs(partial))]
    assert listed == [(2, 4), (2, 4), (2, 4, 8)] and declared == [(2, 4), (2, 4, 8), (2, 4)]

    aligned = gp._align_split_boundary_inputs(bundle)
    assert aligned[0] is bundle[0]
    inputs = [op["output_tensors"][0] for op in aligned[1] if op["op_type"] == "kn_input_op"]
    assert [tuple(t["dim"][: t["num_dims"]]) for t in inputs] == declared
    # the two sides of the cut are then the same tensors, index by index
    combine = to_ir(aligned[1])
    assert boundary_signature(_placeholders(combine)) == boundary_signature(_outputs(partial))
    assert [shape for _, _, shape, _ in boundary_signature(_placeholders(combine))] == declared


def test_a_combine_that_leaves_an_output_unread_still_names_the_ones_it_reads():
    """The combine's k-th input op is the partial's output k whether or not the combine
    reads every one of them. A combine that leaves the scaled exp unread reads outputs 0
    and 2: its second placeholder is the row max, and must not be numbered 1 -- which is
    the scaled exp -- because the input before it was not lowered."""
    partial_ops, combine_ops = _three_output_bundle()
    unread = {12, 15}  # the scaled exp as the combine declares it, and its sum
    combine_ops = [op for op in combine_ops
                   if not unread & {t["guid"] for t in op["input_tensors"]}]
    assert [op["op_type"] for op in combine_ops].count("kn_input_op") == 3  # still declared

    partial = to_ir(partial_ops)
    aligned = gp._align_split_boundary_inputs([partial_ops, combine_ops])
    combine = to_ir(aligned[1])
    written = {index: rest for index, *rest in boundary_signature(_outputs(partial))}
    read = boundary_signature(_placeholders(combine))
    assert [index for index, *_ in read] == [0, 2]
    assert all(written[index] == rest for index, *rest in read)
    assert [ph.name for ph in boundary_order(_placeholders(combine))] == [
        "PlaceHolder_input_11", "PlaceHolder_input_13"]


def _rename_symbols(bundle, mapping):
    renamed = json.loads(json.dumps(bundle))
    for ops in renamed:
        for op in ops:
            for tensor in op.get("input_tensors", []) + op.get("output_tensors", []):
                tensor["dim"] = [mapping.get(d, d) for d in tensor["dim"]]
    return renamed


def test_combine_split_position_is_known_not_matched_by_extent():
    """The combine sums over the split axis, which the layout normalisation put in front of
    the last two dims of the partial's outputs. Finding it by trying positions until the
    extents fit takes position 0 whenever the leading batch dim has the split's extent,
    and the combine then sums over the batch."""
    (bundle, _symbolic) = cut.cut_split_graph(
        _load("gqa/before_split_gqa_graph_structure_valid_1"))

    def reductions(b):
        aligned = gp._align_split_boundary_inputs(gp._normalize_split_graph_layouts(b))
        return [op["op_type"] for op in aligned[1] if "reduce" in op["op_type"]]

    assert reductions(bundle) == ["kn_reduce_sum_1_op", "kn_reduce_sum_1_op"]
    # In this fixture 8 is the B*H_Q symbol, 2 the split symbol and 16 S_KV. Give the split
    # the extent of B*H_Q (and S_KV the extent that keeps its split part where it was).
    coinciding = _rename_symbols(bundle, {2: 8, 16: 64})
    boundary = [op["output_tensors"][0]["dim"][:2] for op in coinciding[1]
                if op["op_type"] == "kn_input_op"]
    assert boundary == [[8, 8], [8, 8]]  # (split, B*H_Q, ...) as the cut emits it
    assert reductions(coinciding) == ["kn_reduce_sum_1_op", "kn_reduce_sum_1_op"]


def test_combine_alignment_fails_loudly_when_the_layouts_disagree():
    (bundle, _symbolic) = cut.cut_split_graph(
        _load("gqa/before_split_gqa_graph_structure_valid_1"))
    normalized = gp._normalize_split_graph_layouts(bundle)
    broken = json.loads(json.dumps(normalized))
    for op in broken[1]:
        if op["op_type"] == "kn_input_op":  # a combine input that is not a partial output
            op["output_tensors"][0]["dim"][2] = 999
    with pytest.raises(ValueError, match="Split boundary mismatch"):
        gp._align_split_boundary_inputs(broken)
