"""An absorbed reduction keeps its whole axis (CPU: tiling search, apply_tiling and codegen).

A reduction that iterates (step < extent) while one of its descendant reductions also iterates
is absorbed into that descendant's loop: in attention, the ``Q K^T`` contraction over ``D`` ends
up inside the ``S_KV`` loop. The loop has one counter, so it cannot advance a second axis.
``apply_tiling`` used to slice the absorbed axis with that counter anyway; the kernel then
contracted one ``D`` block per ``S_KV`` step and read past the tile.
"""

import re

import pytest

from cleave import codegen
from cleave.core.errors import ScheduleRejected
from cleave.core.target import current_target
from cleave.dynamic.dense import flash_attention_sd_graph
from cleave.ir.graph import GraphNode, OutputNode
from cleave.ir.graph_utils import get_op_ancestors
from cleave.ir.ops import Matmul
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling
from cleave.schedule.policy.default import DefaultPolicy, TCPolicy

from tests import _graphs as g

# name -> (graph builder, dims, search options): every tests/schedule/end2end case in which
# the policy emits candidates with an absorbed reduction, with that case's dims and options.
CASES = {
    "flash_attention_sd": (flash_attention_sd_graph, (4096, 4096, 64), {}),
    "gqa_bhsd": (g.gqa_bhsd_graph, (1, 32, 8, 4096, 4096, 64), {}),
    "matmul_matmul_bhsd": (g.matmul_matmul_bhsd_graph, (1, 32, 4096, 4096, 64), {}),
    "matmul_matmul_small_k": (g.matmul_matmul_graph, (4096, 4096, 64), {}),
    "matmul_matmul_small_n": (g.matmul_matmul_graph, (4096, 64, 64), {}),
    "matmul_rowsum_small_k": (g.matmul_rowsum_graph, (4096, 4096, 64), {}),
    "mha_bhsd": (g.mha_bhsd_graph, (1, 32, 4096, 4096, 64), {}),
    "mha_split_partial": (g.mha_split_partial_graph, (1, 32, 4, 4096, 1024, 64), {}),
    "mla_decode_bhsd": (g.mla_decode_bhsd_graph, (1, 128, 4096, 512, 64), {}),
    "mla_prefill_bhsd": (g.mla_prefill_bhsd_graph, (1, 128, 4096, 4096, 192, 128), {}),
    "mqa_bhsd": (g.mqa_bhsd_graph, (1, 32, 1, 4096, 4096, 64), {}),
    "qknorm": (g.qknorm_graph, (1, 64, 8, 4096, 4096, 128), {"topk": 32, "force_fusion": True}),
}


def _policy(graph, force_fusion):
    """The policy ``search_tiling`` builds for ``graph``."""
    outputs = [n for n in graph.nodes if isinstance(n, OutputNode)]
    cls = TCPolicy if any(isinstance(n, Matmul) for n in graph.nodes) else DefaultPolicy
    return cls(outputs, current_target().arch, force_fusion=force_fusion, flat_graph=graph)


def _step(node, cand):
    (step,) = cand.rstep_map[node].values()
    return int(step)


def _absorbed(cand):
    """Reductions that iterate while one of their descendant reductions iterates."""
    active = [
        n
        for n, rstep in cand.rstep_map.items()
        if rstep and _step(n, cand) < int(n.raxis[0].dom.extent)
    ]
    return [a for a in active if any(r is not a and a in get_op_ancestors(r) for r in active)]


def _mis_sliced(scheduled):
    """``(loop, tensor, block tile, iteration tile)`` of every tensor a loop slices off its axis.

    Inside a loop every dim whose block tile is larger than its iteration tile is sliced with
    the loop counter. That is right for at most one dim per tensor, and only when the dim has
    as many iteration tiles as the loop has iterations.
    """
    bad = []
    for loop in (n for n in scheduled.nodes if isinstance(n, GraphNode)):
        for node, it in loop.graph.tmap.items():
            counts = [int(s) // int(t) for s, t in zip(it.shape, it.tile_size)]
            sliced = [c for c in counts if c > 1]
            if len(sliced) > 1 or any(c != int(loop.graph.extent) for c in sliced):
                bad.append((loop.name, node.name, list(it.shape), list(it.tile_size)))
    return bad


@pytest.fixture
def unformatted(monkeypatch):
    """Kernel sources as codegen emits them, without the ``black`` pass over the result.

    Formatting is nine tenths of the time it takes to build a source and no part of what is
    compared: two sources that are equal before it are equal after it.
    """
    monkeypatch.setattr(codegen, "format_source", lambda debug, *sections: "".join(sections))


@pytest.mark.parametrize("name", sorted(CASES))
def test_absorbed_reduction_keeps_its_whole_axis(name, unformatted):
    build, dims, options = CASES[name]
    policy = _policy(build(*dims), options.get("force_fusion", False))
    candidates = policy.emit_config(options.get("topk", 1000))

    n_absorbed = 0
    siblings = {}  # (tile, threads, steps) -> kernel source; several candidates share one
    for cand in candidates:
        label = f"tile={list(cand.output_tile)} rstep={cand.rstep_map}"
        assert not _mis_sliced(apply_tiling(cand)), label

        absorbed = _absorbed(cand)
        if not absorbed:
            continue
        n_absorbed += 1
        # The absorbed step is dropped, not half applied: the kernel is exactly the one
        # scheduled with those reductions at their full extent.
        whole = dict(cand.rstep_map)
        for node in absorbed:
            whole[node] = {node.raxis[0].var.name: int(node.raxis[0].dom.extent)}
        key = (
            tuple(cand.output_tile),
            cand.n_threads,
            tuple((node.name, tuple(rstep.items())) for node, rstep in whole.items()),
        )
        if key not in siblings:
            sibling = policy.compute_tile_dict(list(cand.output_tile), whole)
            sibling.n_threads = cand.n_threads
            siblings[key] = build_candidate_code(sibling)
        assert build_candidate_code(cand) == siblings[key], label

    assert n_absorbed, f"{name}: no candidate with an absorbed reduction; the case tests nothing"


def test_refuses_to_step_an_axis_that_an_absorbed_reduction_holds_whole():
    """LoRA ``X @ W + (X @ A) @ B`` with all three matmuls stepped below their extents.

    ``X @ A`` is absorbed into the loop of ``(X @ A) @ B``, which walks the rank, so it needs
    the whole ``K`` of ``X`` in every iteration. ``X @ W`` steps that same ``K`` of the same
    ``X``. No tile of ``X`` serves both, so the tiling is refused -- with a typed error, which
    ``python -O`` cannot strip -- rather than scheduled into a kernel that slices ``X`` wrong.
    """
    # Rank 128: at the end2end case's rank of 16 the up projection never iterates.
    candidates = _policy(g.lora_graph(16, 4096, 4096, 128), False).emit_config(1000)

    n_refused = n_built = 0
    for cand in candidates:
        label = f"tile={list(cand.output_tile)} rstep={cand.rstep_map}"
        absorbed = [node.name for node in _absorbed(cand)]
        if not absorbed:
            # Nothing is held whole, so this rule refuses nothing. The tiling may still be
            # refused for the other reason two matmuls cannot share ``X``: ``X @ W`` and
            # ``X @ A`` stepped differently give a loop whose trip count is not the number
            # of tiles of ``X``.
            try:
                apply_tiling(cand)
                n_built += 1
            except ScheduleRejected as e:
                assert re.fullmatch(
                    r"GraphNode_mma_down runs \d+ iterations but dim 1 of PlaceHolder_AA is "
                    r"cut into \d+ tiles \(\d+ of 4096\)",
                    str(e),
                ), f"{label}: {e}"
                steps = {n.name: next(iter(v.values())) for n, v in cand.rstep_map.items() if v}
                assert steps["mma_base"] != steps["mma_down"], label
            continue
        assert absorbed == ["mma_down"], label
        with pytest.raises(ScheduleRejected, match="mma_base is shared with an absorbed"):
            apply_tiling(cand)
        n_refused += 1

    assert n_refused, "no candidate steps all three matmuls; the case tests nothing"
    assert n_built, "no candidate without an absorbed reduction is built; the case tests nothing"
