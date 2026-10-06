import json
from pathlib import Path

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.graph import Graph, OutputNode, PlaceHolderNode
from cleave.integration.graph_processing import process_graphs, split_graph
from cleave.ops import Matmul, Reduction, SplitView


def _load_graph_structure(name: str):
    graph_dir = Path(__file__).parent / "graph_structures" / "matmul"
    return json.loads((graph_dir / name).read_text())


def _assert_split_k_cleave_graph_pair(
    first_graph,
    second_graph,
    *,
    m: int,
    k: int,
    n: int,
    split_factor: int,
):
    k_partial = k // split_factor

    assert isinstance(first_graph, Graph)
    assert isinstance(second_graph, Graph)

    assert [type(node) for node in first_graph.nodes] == [
        PlaceHolderNode,
        SplitView,
        PlaceHolderNode,
        SplitView,
        Matmul,
        OutputNode,
    ]
    assert [tuple(node.get_shape()) for node in first_graph.nodes] == [
        (m, k),
        (split_factor, m, k_partial),
        (k, n),
        (split_factor, k_partial, n),
        (split_factor, m, n),
        (split_factor, m, n),
    ]
    assert first_graph.nodes[1].factor == split_factor
    assert first_graph.nodes[3].factor == split_factor
    assert first_graph.nodes[4].inputs[0].src_node is first_graph.nodes[1]
    assert first_graph.nodes[4].inputs[1].src_node is first_graph.nodes[3]
    assert first_graph.nodes[5].inputs[0].src_node is first_graph.nodes[4]

    assert [type(node) for node in second_graph.nodes] == [
        PlaceHolderNode,
        Reduction,
        OutputNode,
    ]
    assert [tuple(node.get_shape()) for node in second_graph.nodes] == [
        (split_factor, m, n),
        (m, n),
        (m, n),
    ]
    assert second_graph.nodes[1].inputs[0].src_node is second_graph.nodes[0]
    assert second_graph.nodes[2].inputs[0].src_node is second_graph.nodes[1]


def test_matmul_split_k_structure():
    before_graph = _load_graph_structure(
        "before_split_matmul_split_k_graph_structure.graph"
    )
    expected_after_graph = _load_graph_structure(
        "after_split_matmul_split_k_graph_structure.graph"
    )

    split_graphs, symbolic_split_factor = split_graph(before_graph)

    assert split_graphs == expected_after_graph
    assert symbolic_split_factor == 2


def test_matmul_split_k_cleave_graph():
    before_graph = _load_graph_structure(
        "before_split_matmul_split_k_graph_structure.graph"
    )

    cleave_graphs = process_graphs([before_graph])

    assert isinstance(cleave_graphs, list)
    assert len(cleave_graphs) == 1
    assert isinstance(cleave_graphs[0], list)
    assert len(cleave_graphs[0]) == 2

    first_graph, second_graph = cleave_graphs[0]
    _assert_split_k_cleave_graph_pair(
        first_graph,
        second_graph,
        m=4,
        k=12,
        n=8,
        split_factor=2,
    )


def test_matmul_symbolic_split_k_cleave_graph_with_symbol_mapping():
    before_graph = _load_graph_structure(
        "before_split_matmul_split_k_graph_structure.graph"
    )

    symbol_map = {
        12: 2048,
        4: 256,
        8: 1536,
    }

    cleave_graphs = process_graphs(
        [before_graph],
        symbol_map=symbol_map,
        split_factors=[4],
    )

    assert len(cleave_graphs) == 1
    first_graph, second_graph = cleave_graphs[0]
    _assert_split_k_cleave_graph_pair(
        first_graph,
        second_graph,
        m=256,
        n=1536,
        k=2048,
        split_factor=4,
    )


def test_matmul_symbolic_split_k_cleave_graph_with_multiple_split_factors():
    before_graph = _load_graph_structure(
        "before_split_matmul_split_k_graph_structure.graph"
    )

    split_factors = [2, 4, 8, 16]
    symbol_map = {
        4: 256,
        8: 1536,
        12: 4096,
    }

    cleave_graphs = process_graphs(
        [before_graph],
        symbol_map=symbol_map,
        split_factors=split_factors,
    )

    assert len(cleave_graphs) == len(split_factors)

    for graph_pair, split_factor in zip(cleave_graphs, split_factors):
        assert isinstance(graph_pair, list)
        assert len(graph_pair) == 2
        first_graph, second_graph = graph_pair
        _assert_split_k_cleave_graph_pair(
            first_graph,
            second_graph,
            m=256,
            n=1536,
            k=4096,
            split_factor=split_factor,
        )
