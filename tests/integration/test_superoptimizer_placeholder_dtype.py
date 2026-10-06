import pytest

pytest.importorskip("tilelang")
tvm = pytest.importorskip("tvm")

from cleave.graph import PlaceHolderNode
from cleave.integration import graph_structure_to_cleave
from cleave.ops import Exp, Reduction
from cleave.ops import wire_input


def test_wire_input_keeps_placeholder_dtype_on_consumer_mismatch():
    src = PlaceHolderNode(name="src")
    src.set_shape((4, 8))
    src.set_dtype(tvm.DataType("float16"))

    rowsum = Reduction((4, 8), dtype="float32", out_dtype="float32", name="rowsum")
    wire_input(src, rowsum, dst_id=0, src_id=0)

    assert src.get_dtype(0) == tvm.DataType("float16")


def test_graph_structure_to_cleave_uses_superoptimizer_dtype_metadata():
    graph_structure = [
        {
            "op_type": "kn_input_op",
            "input_tensors": [],
            "output_tensors": [
                {"guid": 1, "num_dims": 2, "dim": [4, 8], "dtype": "float32"}
            ],
        },
        {
            "op_type": "kn_exp_op",
            "input_tensors": [
                {"guid": 1, "num_dims": 2, "dim": [4, 8], "dtype": "float32"}
            ],
            "output_tensors": [
                {"guid": 2, "num_dims": 2, "dim": [4, 8], "dtype": "float32"}
            ],
        },
        {
            "op_type": "kn_output_op",
            "input_tensors": [
                {"guid": 2, "num_dims": 2, "dim": [4, 8], "dtype": "float32"}
            ],
            "output_tensors": [],
        },
    ]

    graph = graph_structure_to_cleave(graph_structure)

    placeholder = next(node for node in graph.nodes if isinstance(node, PlaceHolderNode))
    exp = next(node for node in graph.nodes if isinstance(node, Exp))

    assert placeholder.get_dtype(0) == tvm.DataType("float32")
    assert exp.dtype == "float32"
    assert exp.out_dtype == "float32"
