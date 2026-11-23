"""The policy charges a graph output's write traffic for the tile the CTA writes (CPU).

``PrimFuncNode.propagate_outputs`` clipped output ``i`` with input ``i``'s shape: a split-K
combine (input ``[S, M, N]``, output ``[M, N]``) wrote tile ``[8, 4]`` as ``[min(8, S), 4]``.
And a batched matmul whose stage fuses the batch dims read the output-space tile in its stage
space: mha_split_partial's ``[1, 1, 128, 64]`` partial-O tile was written as ``[1, 1, 1, 128]``.
Both skewed the traffic that ranks candidates.
"""

import pytest

from cleave.core.target import current_target
from cleave.ir.graph import OutputNode, PrimFuncNode
from cleave.ir.ops import Matmul
from cleave.schedule.policy.default import DefaultPolicy, TCPolicy

from tests import _graphs as g

# end2end cases (tests/schedule/end2end) whose write tiles were wrong, with their dims
CASES = {
    "matmul_splitk_combine": (g.matmul_splitk_combine_graph, (4, 128, 128)),
    "batched_matmul_splitk_combine": (g.batched_matmul_splitk_combine_graph, (2, 4, 128, 128)),
    "mha_split_partial": (g.mha_split_partial_graph, (1, 32, 4, 4096, 1024, 64)),
}


def _policy(graph):
    """The policy ``search_tiling`` builds for ``graph``."""
    outputs = [n for n in graph.nodes if isinstance(n, OutputNode)]
    cls = TCPolicy if any(isinstance(n, Matmul) for n in graph.nodes) else DefaultPolicy
    return cls(outputs, current_target().arch, flat_graph=graph)


@pytest.mark.parametrize("name", sorted(CASES))
def test_output_write_tile_is_the_written_block(name, monkeypatch):
    build, dims = CASES[name]
    policy = _policy(build(*dims))
    candidates = policy.emit_config(1000)
    assert candidates

    written = {}
    real = PrimFuncNode.propagate_outputs

    def spy(node, tile, rstep=None):
        shapes = real(node, tile, rstep)
        if any(e.dst_node.is_output() for e in node.outputs):
            written[node] = shapes
        return shapes

    monkeypatch.setattr(PrimFuncNode, "propagate_outputs", spy)
    for cand in candidates:
        written.clear()
        td = policy.compute_tile_dict(cand.output_tile, cand.rstep_map)
        assert written, "no output write was costed"
        for node, shapes in written.items():
            shape = [int(s) for s in node.get_shape(0)]
            block = [min(int(t), s) for t, s in zip(td.tile_map[node], shape)]
            assert [list(map(int, s)) for s in shapes] == [block], (
                f"{node.name}: tile {list(cand.output_tile)} writes {block}, costed {shapes}"
            )
