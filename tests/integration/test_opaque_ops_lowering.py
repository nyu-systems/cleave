"""kn_sub_op / kn_reduce_max_{d}_op -> Sub / ReduceMax, and their split-graph refusal."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave.frontend import trace
from cleave.lower.split import Rejected, _contains_split_ops, cut_split_graph
from cleave.lower.convert import to_ir
from cleave.ir.ops import ReduceMax, ReduceSum, Sub
from cleave.runtime.reference import execute


def _softmax(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1)


def _t(guid, dims):
    return {"num_dims": len(dims), "dim": list(dims), "guid": guid}


def test_softmax_structure_lowers_to_max_and_sub():
    prog = trace(_softmax, {"R": 8, "C": 64}, cl.f32)
    structure = prog.to_graph_structure()
    assert [o["op_type"] for o in structure][1:3] == ["kn_reduce_max_1_op", "kn_sub_op"]
    graph = to_ir(structure, dtype="float32", out_dtype="float32")
    kinds = {type(n) for n in graph.nodes}
    assert {ReduceMax, Sub, ReduceSum} <= kinds
    mx = next(n for n in graph.nodes if type(n) is ReduceMax)
    assert mx.axis == 1 and mx.name.startswith(f"{ReduceMax.kind}_1_")
    sub = next(n for n in graph.nodes if isinstance(n, Sub))
    assert sub.const is None and sub.broadcast == 0

    x = torch.randn(8, 64) * 10
    (out,) = execute(graph, [x])
    torch.testing.assert_close(out, torch.softmax(x, -1), rtol=1e-5, atol=1e-6)


def test_reduce_max_reference_is_max_not_sum():
    structure = [
        {"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [_t(1, (4, 16))]},
        {"op_type": "kn_reduce_max_0_op", "input_tensors": [_t(1, (4, 16))],
         "output_tensors": [_t(2, (1, 16))]},
        {"op_type": "kn_output_op", "input_tensors": [_t(2, (1, 16))], "output_tensors": []},
    ]
    graph = to_ir(structure, dtype="float32", out_dtype="float32")
    x = torch.randn(4, 16)
    (out,) = execute(graph, [x])
    torch.testing.assert_close(out.reshape(-1), x.amax(0))


def test_split_graph_with_max_is_rejected():
    # split C, max over the split chunks: a sum-combine would be wrong, so the cut refuses.
    structure = [
        {"op_type": "kn_input_op", "input_tensors": [],
         "output_tensors": [{**_t(1, (8, 64)), "can_split": [False, True]}]},
        {"op_type": "kn_split_1_op", "input_tensors": [_t(1, (8, 64))],
         "output_tensors": [_t(2, (4, 8, 16))]},
        {"op_type": "kn_reduce_max_2_op", "input_tensors": [_t(2, (4, 8, 16))],
         "output_tensors": [_t(3, (4, 8, 1))]},
        {"op_type": "kn_reduce_sum_0_op", "input_tensors": [_t(3, (4, 8, 1))],
         "output_tensors": [_t(4, (1, 8, 1))]},
        {"op_type": "kn_output_op", "input_tensors": [_t(4, (1, 8, 1))], "output_tensors": []},
    ]
    assert _contains_split_ops(structure)
    cut = cut_split_graph(structure)
    assert isinstance(cut, Rejected) and "no max-combine" in cut.reason, cut
