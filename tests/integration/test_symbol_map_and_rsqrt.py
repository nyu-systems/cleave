import pytest

pytest.importorskip("tilelang")

from cleave.lower import to_ir
from cleave.ir.ops import RSqrt, Sqrt


def _t(guid, dims):
    return {"guid": guid, "num_dims": len(dims), "dim": list(dims), "dtype": "float32"}


def _op(op_type, ins, outs, **extra):
    return {"op_type": op_type, "input_tensors": ins, "output_tensors": outs, **extra}


def _sqrt_then_rsqrt_div(extra_sqrt_output: bool):
    ops = [
        _op("kn_input_op", [], [_t(1, (4, 8))]),
        _op("kn_sqrt_op", [_t(1, (4, 8))], [_t(2, (4, 8))]),
        _op(
            "kn_div_op",
            [_t(2, (4, 8))],
            [_t(3, (4, 8))],
            scalar=1.0,
            scalar_on_left=True,
        ),
        _op("kn_output_op", [_t(3, (4, 8))], []),
    ]
    if extra_sqrt_output:
        ops.append(_op("kn_output_op", [_t(2, (4, 8))], []))
    return ops


def test_rsqrt_fuses_sole_consumer_sqrt():
    graph = to_ir(_sqrt_then_rsqrt_div(False))
    assert any(isinstance(n, RSqrt) for n in graph.nodes)
    assert not any(isinstance(n, Sqrt) for n in graph.nodes)


def test_rsqrt_not_fused_when_sqrt_has_other_consumers():
    graph = to_ir(_sqrt_then_rsqrt_div(True))
    sqrt = [n for n in graph.nodes if isinstance(n, Sqrt)]
    assert len(sqrt) == 1 and sqrt[0].inputs
    assert not any(isinstance(n, RSqrt) for n in graph.nodes)


def _exp_chain(intermediate_dims):
    return [
        _op("kn_input_op", [], [_t(1, (4, 8))]),
        _op("kn_exp_op", [_t(1, (4, 8))], [_t(2, (4, 8))]),
        _op("kn_exp_op", [_t(2, intermediate_dims)], [_t(3, (4, 8))]),
        _op("kn_output_op", [_t(3, (4, 8))], []),
    ]


def test_symbol_map_not_mutated():
    symbol_map = {4: 64, 8: 128}
    to_ir(_exp_chain((4, 8)), symbol_map=symbol_map)
    assert symbol_map == {4: 64, 8: 128}


def test_known_shape_skips_symbol_lookup():
    # The second exp's input is already known, so its (unmapped) descriptor dims must
    # not be looked up in the symbol map.
    graph = to_ir(_exp_chain((4, 999)), symbol_map={4: 64, 8: 128})
    shapes = {tuple(n.get_shape(0)) for n in graph.nodes if n.num_outputs()}
    assert (64, 128) in shapes
