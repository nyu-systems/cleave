import json
from pathlib import Path

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.integration.graph_processing import split_graph


_CASES_WITH_SYMBOLIC_SPLIT_FACTOR = {f"valid_{i}" for i in range(1, 6)}


def _load_graph_structure(name: str):
    graph_dir = Path(__file__).parent / "graph_structures" / "gqa"
    return json.loads((graph_dir / name).read_text())


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
    graph_dir = Path(__file__).parent / "graph_structures" / "gqa"
    before_prefix = "before_split_gqa_graph_structure_"
    after_prefix = "after_split_gqa_graph_structure_"

    before_cases = {
        path.stem.removeprefix(before_prefix)
        for path in graph_dir.glob("before_split_gqa_graph_structure_*.graph")
    }
    after_cases = {
        path.stem.removeprefix(after_prefix)
        for path in graph_dir.glob("after_split_gqa_graph_structure_*.graph")
    }
    return sorted(before_cases & after_cases)


def _assert_gqa_split_structure(case_name: str):
    before_graph = _load_graph_structure(
        f"before_split_gqa_graph_structure_{case_name}.graph"
    )
    expected_after_graph = _load_graph_structure(
        f"after_split_gqa_graph_structure_{case_name}.graph"
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
def test_gqa_split_structure(case_name: str):
    _assert_gqa_split_structure(case_name)
