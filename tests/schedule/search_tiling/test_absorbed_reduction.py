"""An absorbed reduction keeps its whole axis (CPU: tiling search, apply_tiling and codegen).

A reduction that iterates (step < extent) while one of its descendant reductions also iterates
is absorbed into that descendant's loop: in attention, the ``Q K^T`` contraction over ``D`` ends
up inside the ``S_KV`` loop. The loop has one counter, so it cannot advance a second axis.
``apply_tiling`` used to slice the absorbed axis with that counter anyway; the kernel then
contracted one ``D`` block per ``S_KV`` step and read past the tile.

The policy rates such a candidate as the kernel that is generated for it -- the absorbed
reduction at its whole extent -- not as the steps it names: it used to emit candidates as
valid whose kernel it rates over the shared-memory cap, and to emit step maps that
``apply_tiling`` refuses.
"""

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


def _verdict(cand):
    """What the policy decides and ranks a candidate on."""
    return (cand.valid, cand.smem_cost, cand.traffic, cand.grid_size, cand.block_per_SM, cand.num_wave)


def _stepped(policy, **steps):
    """An rstep map for ``policy``'s graph: ``steps`` by reduction name, ``{}`` elsewhere."""
    by_name = {node.name: node for node in policy.ordered_nodes}
    assert set(steps) <= set(by_name), f"no such node: {set(steps) - set(by_name)}"
    return {
        node: {node.raxis[0].var.name: steps[name]} if name in steps else {}
        for name, node in by_name.items()
    }


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
            siblings[key] = (sibling, build_candidate_code(sibling))
        sibling, sibling_code = siblings[key]
        assert build_candidate_code(cand) == sibling_code, label
        # ... and it is rated as that kernel: a candidate is only proposed if its kernel
        # fits, at the shared memory and the wave count (the rank) its kernel has.
        assert cand.valid and sibling.valid, label
        assert _verdict(cand) == _verdict(sibling), label

    assert n_absorbed, f"{name}: no candidate with an absorbed reduction; the case tests nothing"


@pytest.mark.parametrize(
    "K",
    [
        pytest.param(4096, id="K4096_kernel_over_the_cap"),
        pytest.param(1024, id="K1024_kernel_fits"),
    ],
)
def test_refuses_to_step_an_axis_that_an_absorbed_reduction_holds_whole(K):
    """LoRA ``X @ W + (X @ A) @ B`` with all three matmuls stepped below their extents.

    ``X @ A`` is absorbed into the loop of ``(X @ A) @ B``, which walks the rank, so it needs
    the whole ``K`` of ``X`` in every iteration. ``X @ W`` steps that same ``K`` of the same
    ``X``. No tile of ``X`` serves both, so the tiling is refused -- with a typed error, which
    ``python -O`` cannot strip -- rather than scheduled into a kernel that slices ``X`` wrong.

    The policy agrees: a step map with no kernel is not a valid candidate, and none is
    proposed. (It used to propose all of these; ``apply_tiling`` refused them one by one,
    126 of 256 candidates at ``K = 4096``.) At ``K = 4096`` the kernel that holds ``K``
    whole would not fit the shared-memory cap either; at ``K = 1024`` it would, and the
    shared axis is the only reason.
    """
    # Rank 128: at the end2end case's rank of 16 the up projection never iterates.
    policy = _policy(g.lora_graph(16, 4096, K, 128), False)
    candidates = policy.emit_config(1000)
    assert candidates, "nothing is proposed; the case tests nothing"
    n_built = 0
    for cand in candidates:
        label = f"tile={list(cand.output_tile)} rstep={cand.rstep_map}"
        assert not _absorbed(cand), label
        # Nothing is held whole, so this rule refuses nothing. The tiling may still be
        # refused for the other reason two matmuls cannot share ``X``: ``X @ W`` and
        # ``X @ A`` stepped differently give a loop whose trip count is not the number
        # of tiles of ``X``.
        try:
            apply_tiling(cand)
            n_built += 1
        except ScheduleRejected as e:
            steps = {n.name: next(iter(v.values())) for n, v in cand.rstep_map.items() if v}
            # The loop is the down projection's while it steps ``K``; once it holds ``K``
            # whole, ``X @ W`` rides the up projection's walk of the rank instead.
            loop, runs = (
                ("mma_down", K // steps["mma_down"])
                if steps["mma_down"] < K
                else ("mma_up", 128 // steps["mma_up"])
            )
            tile = steps["mma_base"]
            assert str(e) == (
                f"GraphNode_{loop} runs {runs} iterations but dim 1 of PlaceHolder_AA is "
                f"cut into {K // tile} tiles ({tile} of {K})"
            ), label
            assert runs != K // tile, label
            assert steps["mma_base"] != steps["mma_down"], label
    assert n_built, "no candidate without an absorbed reduction is built; the case tests nothing"

    # The step maps the search walks through on its way to those: up projection and base
    # matmul stepped together (they are one group), the down projection at any step.
    n_refused = 0
    for rows in (32, 64, 128, 256, 512, 1024):
        for up in (32, 64):
            for down in (32, 64, 128, 256, 512):
                steps = _stepped(policy, mma_base=up, mma_down=down, mma_up=up)
                cand = policy.compute_tile_dict([16, rows], steps)
                label = f"tile={[16, rows]} base=up={up} down={down}"
                assert [node.name for node in _absorbed(cand)] == ["mma_down"], label
                assert not cand.valid, label
                with pytest.raises(ScheduleRejected, match="mma_base is shared with an absorbed"):
                    apply_tiling(cand)
                n_refused += 1
    assert n_refused == 60

    if K == 1024:
        # The shared axis alone: the kernel with ``K`` held whole is within the caps.
        steps = _stepped(policy, mma_base=32, mma_down=32, mma_up=32)
        whole = dict(steps)
        (down,) = [node for node in steps if node.name == "mma_down"]
        whole[down] = {down.raxis[0].var.name: K}
        assert policy.compute_tile_dict([16, 32], whole).smem_cost <= current_target().arch.smem_cap
        assert not policy.compute_tile_dict([16, 32], steps).valid


def test_a_candidate_whose_kernel_does_not_fit_is_not_proposed():
    """MLA decode at batch 2: the kernel of 192 of 444 proposed candidates did not fit.

    ``Q K^T`` over the latent dim (and over the positional one) is absorbed into the
    ``S_KV`` loop, so the kernel holds the whole 512-wide latent tile in every iteration.
    Rated on the steps the candidate names (32 of 512) that is 73728 bytes of shared
    memory; the kernel that is generated needs 196608, against a cap of 167936. Such
    candidates were emitted as valid, ranked on the smaller figure, and failed to compile.
    """
    policy = _policy(g.mla_decode_bhsd_graph(2, 128, 4096, 512, 64), False)
    cap = current_target().arch.smem_cap

    # The candidate of the report, by hand: named steps fit, the kernel does not.
    steps = _stepped(policy, mma_0=32, mma_pe=32, mma_1=128, rowsum_0=128)
    named = policy.compute_tile_dict([1, 64, 32], steps)
    assert sorted(node.name for node in _absorbed(named)) == ["mma_0", "mma_pe"]
    kernel = policy.compute_tile_dict(
        [1, 64, 32], _stepped(policy, mma_0=512, mma_pe=64, mma_1=128, rowsum_0=128)
    )
    assert not _absorbed(kernel) and not kernel.valid and kernel.smem_cost > cap
    assert not named.valid and named.smem_cost == kernel.smem_cost

    candidates = policy.emit_config(100000)
    n_absorbed = 0
    for cand in candidates:
        label = f"tile={list(cand.output_tile)} rstep={cand.rstep_map}"
        assert cand.valid and cand.smem_cost <= cap, label
        absorbed = _absorbed(cand)
        if not absorbed:
            continue
        n_absorbed += 1
        whole = dict(cand.rstep_map)
        for node in absorbed:
            whole[node] = {node.raxis[0].var.name: int(node.raxis[0].dom.extent)}
        kernel = policy.compute_tile_dict(list(cand.output_tile), whole)
        assert _verdict(cand) == _verdict(kernel), label
    # Absorbed candidates whose kernel fits are still proposed: they are not de-duplicated.
    assert n_absorbed, "no candidate with an absorbed reduction is left; the case tests nothing"


@pytest.mark.parametrize(
    ("build", "dims", "expected"),
    [
        # (mma_0 iterates, mma_1 iterates) -> candidates. (True, False) is the family that
        # pinning every ancestor reduction to its extent loses: mma_0 in a loop of its own.
        pytest.param(
            g.matmul_matmul_graph, (4096, 64, 64),
            {(True, True): 16, (True, False): 12, (False, True): 16, (False, False): 12},
            id="matmul_matmul_small_n",
        ),
    ],
)
def test_rating_the_kernel_leaves_the_walk_alone(build, dims, expected):
    """The search still walks through every step map, absorbed or not.

    A step map with an absorbed reduction is the way to the ones behind it: ``mma_0``
    stops being absorbed once ``mma_1`` reaches its extent. Rating a state as its kernel
    must not prune the walk there, or those candidates are never reached.
    """
    policy = _policy(build(*dims), False)
    found = {}
    for cand in policy.emit_config(1000):
        iterates = tuple(
            _step(node, cand) < int(node.raxis[0].dom.extent)
            for node in cand.rstep_map
            if node.name in ("mma_0", "mma_1")
        )
        found[iterates] = found.get(iterates, 0) + 1
    assert found == expected


def test_lora_proposes_what_it_did():
    """The end2end LoRA case has no absorbed reduction: nothing about it may move."""
    candidates = _policy(g.lora_graph(16, 4096, 4096, 16), False).emit_config(1000)
    assert len(candidates) == 720
    assert not any(_absorbed(cand) for cand in candidates)
