import json
from pathlib import Path

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from tests._graphs import (
    rmsnorm_matmul_split_combine_graph,
    rmsnorm_matmul_split_partial_graph,
)
from cleave.graph import Graph, OutputNode, PlaceHolderNode
from cleave.integration.graph_processing import process_graphs, split_graph
from cleave.ops import Div, Matmul, Mul, Reduction, SplitView, Sqrt


_CASES_WITH_SYMBOLIC_SPLIT_FACTOR = set()


def _load_graph_structure(name: str):
    graph_dir = Path(__file__).parent / "graph_structures" / "rmsnorm_matmul"
    return json.loads((graph_dir / name).read_text())


def _node_attrs(node):
    if isinstance(node, PlaceHolderNode):
        return ()
    if isinstance(node, SplitView):
        return (("axis", node.axis), ("factor", node.factor))
    if isinstance(node, Reduction):
        return (("axis", node.axis),)
    if isinstance(node, Matmul):
        return (
            ("transpose_A", node.transpose_A),
            ("transpose_B", node.transpose_B),
            ("warp_policy", node.warp_policy),
            ("input_batch_dims_A", tuple(node.input_batch_dims_A)),
            ("input_batch_dims_B", tuple(node.input_batch_dims_B)),
        )
    if isinstance(node, Mul):
        return (("broadcast", node.broadcast), ("const", node.const))
    if isinstance(node, Div):
        return (
            ("broadcast", node.broadcast),
            ("const", node.const),
            ("const_on_left", node.const_on_left),
        )
    if isinstance(node, (OutputNode, Sqrt)):
        return ()
    raise TypeError(f"Unsupported node type in signature: {type(node).__name__}")


def _node_signature(node, cache):
    cached = cache.get(id(node))
    if cached is not None:
        return cached

    shape = tuple(node.get_shape()) if hasattr(node, "get_shape") else None
    dtype_sig = tuple(str(dtype) for dtype in getattr(node, "_dtypes", []))

    if isinstance(node, OutputNode):
        edge = node.inputs[0]
        signature = (
            type(node).__name__,
            shape,
            dtype_sig,
            edge.src_id,
            _node_signature(edge.src_node, cache),
        )
    elif isinstance(node, PlaceHolderNode):
        signature = (type(node).__name__, shape, dtype_sig, _node_attrs(node))
    else:
        signature = (
            type(node).__name__,
            shape,
            dtype_sig,
            _node_attrs(node),
            tuple(
                (edge.src_id, edge.dst_id, _node_signature(edge.src_node, cache))
                for edge in node.inputs
            ),
        )

    cache[id(node)] = signature
    return signature


def _graph_signature(graph):
    assert isinstance(graph, Graph)
    cache = {}
    output_signatures = sorted(
        _node_signature(node, cache)
        for node in graph.nodes
        if isinstance(node, OutputNode)
    )
    return (len(graph.nodes), tuple(output_signatures))


def _renumber_graph(graph):
    guid_map = {}
    next_guid = 10000001

    def _renumber_tensor(tensor):
        nonlocal next_guid

        new_tensor = dict(tensor)
        guid = new_tensor.get("guid")
        if guid is not None:
            if guid not in guid_map:
                guid_map[guid] = next_guid
                next_guid += 1
            new_tensor["guid"] = guid_map[guid]
        return new_tensor

    return [
        {
            **op,
            "input_tensors": [
                _renumber_tensor(tensor) for tensor in op.get("input_tensors", [])
            ],
            "output_tensors": [
                _renumber_tensor(tensor) for tensor in op.get("output_tensors", [])
            ],
        }
        for op in graph
    ]


def _discover_cases():
    graph_dir = Path(__file__).parent / "graph_structures" / "rmsnorm_matmul"
    before_prefix = "before_split_rmsnorm_matmul_graph_structure_"
    after_prefix = "after_split_rmsnorm_matmul_graph_structure_"

    before_cases = {
        path.stem.removeprefix(before_prefix)
        for path in graph_dir.glob("before_split_rmsnorm_matmul_graph_structure_*.graph")
    }
    after_cases = {
        path.stem.removeprefix(after_prefix)
        for path in graph_dir.glob("after_split_rmsnorm_matmul_graph_structure_*.graph")
    }
    return sorted(before_cases & after_cases)


def _assert_rmsnorm_matmul_split_structure(case_name: str):
    before_graph = _load_graph_structure(
        f"before_split_rmsnorm_matmul_graph_structure_{case_name}.graph"
    )
    expected_after_graph = _load_graph_structure(
        f"after_split_rmsnorm_matmul_graph_structure_{case_name}.graph"
    )

    split_result = split_graph(before_graph)
    expects_symbolic_split_factor = case_name in _CASES_WITH_SYMBOLIC_SPLIT_FACTOR

    if expected_after_graph == []:
        assert split_result is None
        assert not expects_symbolic_split_factor
        return

    if expected_after_graph and isinstance(expected_after_graph[0], dict):
        assert split_result is None
        assert _renumber_graph(before_graph) == expected_after_graph
        assert not expects_symbolic_split_factor
        return

    assert split_result is not None
    split_graphs, symbolic_split_factor = split_result

    assert split_graphs == expected_after_graph
    if expects_symbolic_split_factor:
        assert symbolic_split_factor is not None
    else:
        assert symbolic_split_factor is None


@pytest.mark.parametrize("case_name", _discover_cases())
def test_rmsnorm_matmul_split_structure(case_name: str):
    _assert_rmsnorm_matmul_split_structure(case_name)


def test_rmsnorm_matmul_symbolic_split_k_cleave_graph_with_symbol_mapping():
    before_graph = _load_graph_structure(
        "before_split_rmsnorm_matmul_graph_structure_valid_2.graph"
    )

    symbol_map = {
        4: 256,
        8: 1536,
        12: 4096,
    }

    cleave_graphs = process_graphs(
        [before_graph],
        symbol_map=symbol_map,
        split_factors=[4],
    )

    assert len(cleave_graphs) == 1
    first_graph, second_graph = cleave_graphs[0]

    split_views = [node for node in first_graph.nodes if isinstance(node, SplitView)]
    assert sorted((tuple(node.get_shape()), node.factor) for node in split_views) == [
        ((4, 256, 1024), 4),
        ((4, 1024, 1536), 4),
    ]

    matmuls = [node for node in first_graph.nodes if isinstance(node, Matmul)]
    assert [tuple(node.get_shape()) for node in matmuls] == [(4, 256, 1536)]

    first_output_shapes = sorted(
        tuple(node.get_shape())
        for node in first_graph.nodes
        if isinstance(node, OutputNode)
    )
    assert first_output_shapes == [(4, 256), (4, 256, 1536)]

    second_output_shapes = [
        tuple(node.get_shape())
        for node in second_graph.nodes
        if isinstance(node, OutputNode)
    ]
    assert second_output_shapes == [(256, 1536)]


def test_rmsnorm_matmul_split_cleave_graph_matches_apply_tiling_references():
    before_graph = _load_graph_structure(
        "before_split_rmsnorm_matmul_graph_structure_valid_2.graph"
    )

    cleave_graphs = process_graphs(
        [before_graph],
        symbol_map={
            4: 1024,
            8: 6144,
            12: 4096,
        },
        split_factors=[4],
    )

    assert len(cleave_graphs) == 1
    actual_partial_graph, actual_combine_graph = cleave_graphs[0]

    expected_partial_graph = rmsnorm_matmul_split_partial_graph(4, 1024, 6144, 1024)
    expected_combine_graph = rmsnorm_matmul_split_combine_graph(4, 1024, 6144, 4096)

    assert _graph_signature(actual_partial_graph) == _graph_signature(
        expected_partial_graph
    )
    assert _graph_signature(actual_combine_graph) == _graph_signature(
        expected_combine_graph
    )
