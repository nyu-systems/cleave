"""A softmax in traced order over a score matmul, returned itself (attention weights).

``matmul_softmax_graph`` is ``exp(scale * (Q @ K^T) - shift)``, its row sum, then
``exp / sum``: the divide reads the exp next to the finished row sum. A tiling that loops
the row sum computes the exp per iteration of that loop, and the divide outside the loop
would need every iteration's exp again. No schedule here recomputes it (that is a second
pass over the keys), so ``apply_tiling`` refuses such a tiling, with the reason. The tiling
pinned below is one: the row sum over 4096 keys in steps of 32.

A tiling that holds the whole row in one tile has no such loop. At 4096 keys no tile
holds the row. At 256 keys the row sum can take the row in one step, and the tilings
whose output tile holds the whole row of weights as well are built and validated on a
GPU. The others write a part of the row per block (32, 64 or 128 of the 256 columns)
while the block holds every ``exp`` of the row for the sum: their divide read the first
columns of the row in every block. They were built, and 12 of 12 were wrong on a GPU (off
by the row maximum); the build refuses them now (``refuse_parts_of_computed_tiles``).
"""

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.core.errors import ScheduleRejected
from cleave.ir.graph import GraphNode
from cleave.ir.graph_utils import graph_tiers
from cleave.runtime.evaluate import run_candidates
from cleave.schedule import apply_tiling, search_tiling
from cleave.schedule.tiling import schedule_kernel

from tests._graphs import matmul_softmax_graph
from tests._tolerances import tolerance_for
from tests.graph.apply_tiling._test_utils import DTYPE, build_tiled_candidate

REFUSAL = "exp_0 is computed per iteration of GraphNode_rowsum_0 but also read by div_0 outside it"


def _step(cand, name):
    (steps,) = [steps for node, steps in cand.rstep_map.items() if node.name == name]
    (step,) = steps.values()
    return int(step)


def _outcomes(cands):
    """``(built, refused)``: the tilings the default build schedules (``schedule_kernel``),
    and the reasons of the others. Anything but a ``ScheduleRejected`` propagates."""
    built, refused = [], []
    for cand in cands:
        try:
            built.append((cand, schedule_kernel(cand)))
        except ScheduleRejected as refusal:
            refused.append((cand, str(refusal)))
    return built, refused


@pytest.mark.parametrize(
    ("target_output_tile", "target_rstep"),
    [
        (
            [1, 128, 128],
            {
                "mma_0": {"v3": 64},
                "mul_0": {},
                "sub_0": {},
                "exp_0": {},
                "rowsum_0": {"v2": 32},
                "div_0": {},
            },
        ),
    ],
)
def test_apply_tiling_matmul_softmax(target_output_tile, target_rstep):
    """The pinned tiling loops the row sum while the divide reads the exp outside the
    loop: a refusal of its own type, naming the three nodes. (It was a ``KeyError`` out of
    ``apply_tiling`` before the refusal existed.)"""
    main_graph = matmul_softmax_graph(1, 32, 4096, 4096, 64)
    cand = build_tiled_candidate(main_graph, target_output_tile, target_rstep)
    with pytest.raises(ScheduleRejected) as refusal:
        apply_tiling(cand)
    assert type(refusal.value) is ScheduleRejected
    assert str(refusal.value) == REFUSAL


def test_every_tiling_of_a_long_row_loops_the_row_sum_and_is_refused():
    """4096 keys: no tile holds a row of weights, so every tiling the search proposes
    loops the row sum, and every one is refused for the same reason (by ``apply_tiling``
    itself, before the build's own checks)."""
    cands = search_tiling(matmul_softmax_graph(1, 32, 4096, 4096, 64), topk=1000)
    assert cands and all(_step(c, "rowsum_0") < 4096 for c in cands)
    built, refused = _outcomes(cands)
    assert not built and len(refused) == len(cands)
    assert {reason for _, reason in refused} == {REFUSAL}


SHORT = (1, 4, 64, 256, 64)  # B, H, S_Q, S_KV, D: a row of 256 weights fits one tile


def _whole_row_tilings():
    cands = search_tiling(matmul_softmax_graph(*SHORT), topk=1000)
    built, refused = _outcomes(cands)
    return cands, built, refused


def _part(columns):
    return (f"div_0 writes a tile (1, 64, {columns}) and reads exp_0, which its block holds as "
            "(1, 64, 256): it would read the first elements of that tile in every block, "
            "wherever the block is launched")


def test_the_tilings_that_hold_the_row_in_one_tile_are_built():
    cands, built, refused = _whole_row_tilings()
    one_step = [c for c in cands if _step(c, "rowsum_0") == 256]
    assert one_step and len(one_step) < len(cands), "the case needs both kinds of tiling"
    whole = [c for c in one_step if list(c.output_tile)[-1] == 256]
    parts = [c for c in one_step if list(c.output_tile)[-1] < 256]
    assert len(whole) >= 4 and len(parts) >= 4
    # Exactly the tilings that hold the row whole, in the sum and in the output tile, are
    # built. A row sum that is looped is refused by ``apply_tiling`` ...
    assert [c for c, _ in built] == whole
    reasons = {id(c): reason for c, reason in refused}
    assert {reasons[id(c)] for c in cands if _step(c, "rowsum_0") < 256} == {REFUSAL}
    # ... and an output tile that is a part of the row by the build, which would have
    # handed every block the first columns of the row
    assert {reasons[id(c)] for c in parts} == {_part(n) for n in (32, 64, 128)}
    for cand in parts:
        apply_tiling(cand)  # the check is the default build's, like the unit-extent one
    # none of the built ones has a loop that contains the exp
    for _, tiled in built:
        loops = [n for n in tiled.nodes if isinstance(n, GraphNode)]
        assert "GraphNode_rowsum_0" not in {n.name for n in loops}
        for tier, owner in graph_tiers(tiled):
            assert owner is None or "exp_0" not in {n.name for n in tier.nodes}


def test_gpu_the_whole_row_tilings_validate():
    """GPU: every tiling that holds the row whole, against the reference, at the bound
    ``matmul_softmax_graph`` declares."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to compile and run candidates.")
    _, built, _ = _whole_row_tilings()
    cands = [c for c, _ in built]
    assert len(cands) >= 4
    results = run_candidates(cands, DTYPE, max_parallel=4,
                             tolerance=tolerance_for(matmul_softmax_graph))
    wrong = [(r.label, type(r.error).__name__, str(r.error)[-300:]) for r in results
             if not r.ok and r.stage in ("validate", "timeout", "crash")]
    assert not wrong, wrong
    ok = [r for r in results if r.ok]
    assert len(ok) == len(cands), [(r.label, r.stage, str(r.error)[-200:])
                                   for r in results if not r.ok]
    for r in ok:
        (measured,) = r.measured
        assert measured.rel <= tolerance_for(matmul_softmax_graph).rtol
