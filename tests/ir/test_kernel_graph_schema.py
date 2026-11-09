import json
from pathlib import Path

import pytest

from cleave.superoptimizer.schema import KernelGraph, KGOp, format_op_type, parse_op_type

STRUCTURES_DIR = Path(__file__).resolve().parents[1] / "integration" / "graph_structures"
STRUCTURES = sorted(STRUCTURES_DIR.glob("*/*.graph"))


def _graphs(path):
    data = json.loads(path.read_text())
    # after_split fixtures hold a list of graphs (partial + combine).
    return data if data and isinstance(data[0], list) else [data]


def _assert_round_trip(structure, *, ordered=False):
    kg = KernelGraph.from_dicts(structure)
    assert kg.to_dicts() == structure
    # Fixtures are stored key-sorted; live output must keep the emitted key order too.
    assert json.dumps(kg.to_dicts(), sort_keys=not ordered) == json.dumps(
        structure, sort_keys=not ordered
    )
    for op, raw in zip(kg.ops, structure):
        assert op.op_type == raw["op_type"]
        for t, rt in zip(op.inputs + op.outputs, raw["input_tensors"] + raw["output_tensors"]):
            assert t.rank == rt["num_dims"]
    return kg


def test_parse_op_type():
    assert parse_op_type("kn_reduce_sum_2_op") == ("reduce_sum", 2)
    assert parse_op_type("kn_split_1_op") == ("split", 1)
    assert parse_op_type("kn_matmul_op") == ("matmul", None)
    assert format_op_type("split", 0) == "kn_split_0_op"
    assert parse_op_type("kn_reduce_max_1_op") == ("reduce_max", 1)
    assert parse_op_type("kn_sub_op") == ("sub", None)
    for kind, axis in (("reduce_max", 0), ("reduce_max", 4), ("sub", None), ("reduce_sum", 3)):
        assert parse_op_type(format_op_type(kind, axis)) == (kind, axis)
    with pytest.raises(ValueError):
        parse_op_type("matmul")


def test_sub_max_round_trip():
    t = lambda g, d: {"num_dims": len(d), "dim": list(d), "guid": g}  # noqa: E731
    structure = [
        {"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [t(1, (8, 12))]},
        {"op_type": "kn_reduce_max_1_op", "input_tensors": [t(1, (8, 12))],
         "output_tensors": [t(2, (8, 1))]},
        {"op_type": "kn_sub_op", "input_tensors": [t(1, (8, 12)), t(2, (8, 1))],
         "output_tensors": [t(3, (8, 12))]},
        {"op_type": "kn_output_op", "input_tensors": [t(3, (8, 12))], "output_tensors": []},
    ]
    kg = _assert_round_trip(structure, ordered=True)
    assert [(op.kind, op.axis) for op in kg.ops][1:3] == [("reduce_max", 1), ("sub", None)]


@pytest.mark.parametrize("path", STRUCTURES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_fixture_round_trip(path):
    for structure in _graphs(path):
        kg = _assert_round_trip(structure)
        assert all(isinstance(op, KGOp) for op in kg.ops)


def test_fixture_axis_fields():
    path = STRUCTURES_DIR / "matmul" / "before_split_matmul_split_k_graph_structure.graph"
    kg = KernelGraph.from_dicts(json.loads(path.read_text()))
    assert [(op.kind, op.axis) for op in kg.ops if op.axis is not None] == [
        ("split", 1),
        ("split", 0),
        ("reduce_sum", 0),
    ]
    assert kg.has_split and len(kg.inputs) == 2
    assert kg.inputs[0].dims == (4, 12)


@pytest.mark.search
def test_live_search_round_trip(tmp_path):
    import cleave.superoptimizer as csopt

    graph = csopt.new_tile_graph(split_enabled=[True, True])
    x = graph.create_tiled_input((64, 256), (False, True))
    w = graph.create_tiled_input((256, 64), (True, False))
    graph.create_tiled_output(graph.matmul(x, w))
    symbolic, _ = csopt.create_symbolic_tile_graph(graph)
    config = symbolic.setup_search_config()
    config.search_thread = 4
    found = symbolic.search_tile_graph(str(tmp_path / "ckpt.json"), False, False, config)
    structures = [g.get_graph_structure() for g in found]
    assert structures
    kinds = set()
    for s in structures:
        kinds |= {op.kind for op in _assert_round_trip(s, ordered=True).ops}
    assert {"input", "output", "matmul", "split", "reduce_sum"} <= kinds


def test_live_kernel_graph_attrs_round_trip():
    import cleave.superoptimizer as csopt

    g = csopt.new_kernel_graph()
    a = g.new_input((4, 8), csopt.float32)
    b = g.new_input((8, 4), csopt.float32)
    c = g.matmul(a, b)
    g.add(g.exp(c), c)
    structure = g.cygraph.get_graph_structure()
    kg = _assert_round_trip(structure, ordered=True)
    mm = kg.ops_of("matmul")[0]
    assert mm.attrs == {"trans_A": False, "trans_B": False}
    assert mm.inputs[0].dtype == "float32"
