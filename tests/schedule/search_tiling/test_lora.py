from cleave.schedule import search_tiling

from tests._graphs import lora_graph


def test_lora():
    # LoRA (O = X@W + (X@A)@B) fuses matmuls with heterogeneous reduction
    # extents -- the base/down projections reduce over K while the up projection
    # reduces over the rank R. The tiling policy grouped the base (K) and up (R)
    # reductions together (shared X, no descendant reduction) and expanded their
    # reduction steps in lockstep, which raised "ValueError: <K-step> is not in
    # list" when a K-scale step was written into the R-extent axis. This exercises
    # that path: search_tiling must enumerate candidates without crashing.
    M, N, K, R = 16, 4096, 4096, 16
    main_graph = lora_graph(M, N, K, R)

    candidates = search_tiling(main_graph, topk=1000)

    assert (
        len(candidates) > 0
    ), "Expected to find at least one tiling candidate for LoRA, but found none."


def _policy_and_candidates(graph):
    from cleave.core.target import current_target
    from cleave.schedule.policy.default import TCPolicy

    outputs = [n for n in graph.nodes if n.is_output()]
    policy = TCPolicy(outputs, current_target().arch, flat_graph=graph)
    return policy, search_tiling(graph, topk=1000)


def test_lora_base_and_down_projections_step_together():
    # X@W and X@A share X, so apply_tiling runs them in one loop, which has one counter.
    # The policy used to step them independently: 616 of 720 candidates had different
    # K steps, sliced the shared X with one of them and all failed in TileLang with
    # "T.gemm K shape check failed" (wasted candidates).
    from cleave.schedule import apply_tiling

    candidates = search_tiling(lora_graph(16, 4096, 4096, 16), topk=1000)
    assert len(candidates) == 104
    for cand in candidates:
        steps = {n.name: s for n, s in cand.rstep_map.items()}
        assert steps["mma_base"] == steps["mma_down"], steps
        apply_tiling(cand)


def test_apply_tiling_refuses_shared_input_merge_with_other_trip_count():
    import pytest

    from cleave.core.errors import ScheduleRejected
    from cleave.schedule import apply_tiling

    policy, candidates = _policy_and_candidates(lora_graph(16, 4096, 4096, 16))
    cand = candidates[0]
    rstep_map = {n: dict(s) for n, s in cand.rstep_map.items()}
    base = next(n for n in rstep_map if n.name == "mma_base")
    (axis,) = rstep_map[base]
    rstep_map[base][axis] *= 2  # half the trips of the loop it would join
    tile = policy.compute_tile_dict(list(cand.output_tile), rstep_map)
    with pytest.raises(ScheduleRejected, match="mma_base takes"):
        apply_tiling(tile)
