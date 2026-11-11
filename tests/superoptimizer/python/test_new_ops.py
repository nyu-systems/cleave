"""Op table parity, binding leftovers, and the sub / max reduction ops."""

import gc
import json

import pytest

import cleave.superoptimizer as csopt
from cleave.superoptimizer import core

# Checkpoint names: these must never change.
FROZEN_NAMES = {
    "kn_input_op": 1001, "kn_output_op": 1002, "kn_matmul_op": 1003,
    "kn_exp_op": 1100, "kn_square_op": 1101, "kn_sqrt_op": 1102, "kn_silu_op": 1104,
    "kn_sigmoid_op": 1105, "kn_gelu_op": 1106, "kn_relu_op": 1150, "kn_clamp_op": 1151,
    "kn_log_op": 1160, "kn_add_op": 1200, "kn_mul_op": 1201, "kn_div_op": 1202,
    "kn_sub_op": 1203, "kn_rms_norm_op": 1350,
    **{f"kn_reduce_sum_{d}_op": 1300 + d for d in range(5)},
    **{f"kn_reduce_max_{d}_op": 1310 + d for d in range(5)},
    **{f"kn_split_{d}_op": 1420 + d for d in range(5)},
}
EXP_ROWSUM_GRAPHS = 4


def test_python_names_match_cpp_table():
    table = core.op_type_table()
    assert len(set(table.values())) == len(table)
    for name, value in FROZEN_NAMES.items():
        assert table[name] == value
    for name, value in table.items():
        expected = "kn_unknown" if value == 1000 else name
        assert core.get_kn_operator_type_string(value) == expected
    assert core.get_kn_operator_type_string(4242) == "unknown_op_type4242"


def test_null_operator_raises():
    op = core.CyKNOperator(None)
    assert op.op_type is None
    with pytest.raises(ValueError):
        op.get_input_dtensors()
    with pytest.raises(ValueError):
        op.get_output_dtensors()


def test_kn_graph_view_keeps_owner_alive():
    t = csopt.new_tile_graph()
    x = t.create_tiled_input((4, 8), csopt.float32)
    t.create_tiled_output(t.exp(x))
    view = t.cygraph.kn_graph()
    expected = view.get_graph_structure()
    inp = view.get_input_dtensors()[0]
    del t, x, view
    gc.collect()
    assert inp.dim(1) == 8 and inp.dtype == csopt.float32
    view = csopt.new_tile_graph().cygraph.kn_graph()
    gc.collect()
    assert view.get_graph_structure() == []
    assert expected[0]["op_type"] == "kn_input_op"


def _exp_rowsum(**cfg):
    concrete = csopt.new_tile_graph()
    x = concrete.create_tiled_input((16, 64), csopt.float32)
    concrete.create_tiled_output(concrete.reduce_sum(concrete.exp(x), 1))
    symbolic, _ = csopt.create_symbolic_tile_graph(concrete)
    config = symbolic.setup_search_config()
    config.search_thread = 4
    for k, v in cfg.items():
        setattr(config, k, v)
    return symbolic, config


SUB_MAX = {FROZEN_NAMES["kn_sub_op"]} | {FROZEN_NAMES[f"kn_reduce_max_{d}_op"] for d in range(5)}


def test_graph_without_sub_max_search_unchanged():
    symbolic, config = _exp_rowsum()
    assert not SUB_MAX & set(config._key_fields()["knop_to_explore"])
    assert len(symbolic.search_tile_graph("", config=config)) == EXP_ROWSUM_GRAPHS


def _softmax():
    concrete = csopt.new_tile_graph()
    x = concrete.create_tiled_input((16, 64), csopt.float32)
    e = concrete.exp(concrete.sub(x, concrete.reduce_max(x, 1)))
    concrete.create_tiled_output(concrete.div(e, concrete.reduce_sum(e, 1)))
    symbolic, _ = csopt.create_symbolic_tile_graph(concrete)
    config = symbolic.setup_search_config()
    config.search_thread = 8
    return concrete, symbolic, config


def test_structure_fields():
    concrete, _, _ = _softmax()
    ops = concrete.get_graph_structure()
    assert [o["op_type"] for o in ops] == [
        "kn_input_op", "kn_reduce_max_1_op", "kn_sub_op", "kn_exp_op",
        "kn_reduce_sum_1_op", "kn_div_op", "kn_output_op",
    ]
    mx, sub = ops[1], ops[2]
    assert set(mx) == {"op_type", "input_tensors", "output_tensors"}
    assert mx["output_tensors"][0]["dim"][:2] == [16, 1]
    assert set(sub) == {"op_type", "input_tensors", "output_tensors"}
    assert [t["guid"] for t in sub["input_tensors"]] == [
        ops[0]["output_tensors"][0]["guid"], mx["output_tensors"][0]["guid"]]


def _canonical(structure):
    ids = {}
    out = []
    for op in structure:
        op = json.loads(json.dumps(op))
        for key in ("input_tensors", "output_tensors"):
            for t in op[key]:
                t["guid"] = ids.setdefault(t["guid"], len(ids))
        out.append(op)
    return out


def _structures(graphs):
    # Search threads append in nondeterministic order.
    return sorted((_canonical(g.get_graph_structure()) for g in graphs), key=json.dumps)


def test_sub_max_search_and_round_trip(tmp_path):
    _, symbolic, config = _softmax()
    assert SUB_MAX & set(config._key_fields()["knop_to_explore"])
    ckpt = str(tmp_path / "ckpt.json")
    graphs = symbolic.search_tile_graph(ckpt, config=config)
    assert graphs
    structures = _structures(graphs)
    for s in structures:
        kinds = [o["op_type"] for o in s]
        assert "kn_sub_op" in kinds and "kn_reduce_max_1_op" in kinds
    saved = json.load(open(ckpt))
    assert {o["op_type"] for g in saved for o in g} >= {"kn_sub_op", "kn_reduce_max_1_op"}
    # Reload through the C++ JSON reader (matching key => no re-search).
    reloaded = symbolic.search_tile_graph(ckpt, config=config)
    assert _structures(reloaded) == structures
