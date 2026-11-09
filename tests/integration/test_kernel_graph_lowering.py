"""Typed-schema conversion, the rsqrt rewrite pass, Rejected reasons and the cut cache."""

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("tilelang")

from cleave.core.schema import KernelGraph
from cleave.lower import process as gp
from cleave.lower import split as cut
from cleave.lower import to_ir
from cleave.lower.convert import fuse_rsqrt_ops

STRUCTURES = Path(__file__).parent / "graph_structures"
SYMBOLS = {4: 1024, 8: 6144, 12: 4096}


def _load(family, name):
    return json.loads((STRUCTURES / family / f"{name}.graph").read_text())


def _t(guid, dims):
    return {"guid": guid, "num_dims": len(dims), "dim": list(dims), "dtype": "float32"}


def _op(op_type, ins, outs, **extra):
    return {"op_type": op_type, "input_tensors": ins, "output_tensors": outs, **extra}


def _rsqrt_chain():
    return [
        _op("kn_input_op", [], [_t(1, (4, 8))]),
        _op("kn_exp_op", [_t(1, (4, 8))], [_t(2, (4, 8))]),
        _op("kn_sqrt_op", [_t(2, (4, 8))], [_t(3, (4, 8))]),
        _op("kn_div_op", [_t(3, (4, 8))], [_t(4, (4, 8))], scalar=1.0, scalar_on_left=True),
        _op("kn_output_op", [_t(4, (4, 8))], []),
    ]


def _summary(graph):
    return [(type(n).__name__, n.name, tuple(n.get_shape(0)) if n.is_output() else None)
            for n in graph.nodes]


def test_kernel_graph_and_dicts_convert_identically():
    gs = _load("rmsnorm_matmul", "after_split_rmsnorm_matmul_graph_structure_valid_1")
    kg = KernelGraph.from_dicts(gs)
    a = to_ir(gs, symbol_map=SYMBOLS)
    b = to_ir(kg, symbol_map=SYMBOLS)
    assert _summary(a) == _summary(b)
    assert kg.to_dicts() == gs  # conversion does not mutate the structure


def test_fuse_rsqrt_ops_rewrites_and_keeps_source_indices():
    ops = list(enumerate(KernelGraph.from_dicts(_rsqrt_chain()).ops))
    fused = fuse_rsqrt_ops(ops)
    assert [(i, op.kind) for i, op in fused] == [(0, "input"), (1, "exp"), (3, "rsqrt"), (4, "output")]
    assert fused[2][1].inputs[0].guid == 2 and fused[2][1].outputs[0].guid == 4
    names = [n.name for n in to_ir(_rsqrt_chain()).nodes]
    assert "rsqrt_3" in names and not any(n.startswith("sqrt") for n in names)


def test_unsupported_op_type_is_reported_with_its_index():
    gs = _rsqrt_chain()
    gs[1]["op_type"] = "kn_sigmoid_op"
    with pytest.raises(ValueError, match=r"Unsupported op_type 'kn_sigmoid_op' at index 1"):
        to_ir(gs)
    gs[1]["op_type"] = "not-an-op"
    with pytest.raises(ValueError, match=r"Unsupported op_type 'not-an-op' at index 1"):
        to_ir(gs)


def test_invalid_split_carries_a_reason():
    gs = _load("gqa", "before_split_gqa_graph_structure_invalid_1")
    result = cut.cut_split_graph(gs)
    assert isinstance(result, cut.Rejected) and result.reason
    assert cut.split_graph(gs) is None and not cut.is_split_graph(gs)
    structures, rejected = gp.lowerable([gs])
    assert structures == [] and rejected == [(0, result)]


def test_solo_structure_is_rejected_as_having_no_split():
    gs = _rsqrt_chain()
    assert cut.cut_split_graph(gs) == cut.Rejected("no split ops")
    (s,), rejected = gp.lowerable([gs])
    assert rejected == [] and s.converted == [gs] and s.symbolic_split_factor is None


def test_cut_runs_once_per_structure(monkeypatch):
    gs = _load("rmsnorm_matmul", "before_split_rmsnorm_matmul_graph_structure_valid_2")
    cut._CUT_CACHE.clear()
    gp._PREPARED.clear()
    runs = []
    real = cut._Cut.run
    monkeypatch.setattr(cut._Cut, "run", lambda self: runs.append(1) or real(self))
    assert cut.is_split_graph(gs) and cut.is_split_graph(copy.deepcopy(gs))
    for factor in (2, 4, 8):
        (bundle,) = gp.to_bundles([gs], symbol_map=SYMBOLS, split_factors=[factor])
        assert len(bundle) == 2
    assert len(runs) == 1
    first = cut.split_graph(gs)
    first[0].clear()  # split_graph hands out a copy; the cache is untouched
    assert len(cut.split_graph(gs)[0]) == 2
