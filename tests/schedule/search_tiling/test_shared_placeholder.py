"""A placeholder two reductions share has one tile in the loop they share (CPU).

LoRA is ``X @ W + (X @ A) @ B``. ``apply_tiling`` builds the loop of ``X @ A``, keys the loop's
tensor map by the loop's copy of ``X``, and then merges ``X @ W`` into the loop by giving that
copy a second consumer. ``PlaceHolderNode.__hash__`` used to cover the consumers, so the key's
hash moved while the key was in the map. The next assignment then added a second entry for the
same node -- or, when the old and the new hash happened to probe the same slot, which depends
on ``PYTHONHASHSEED``, updated the entry in place and left it unreachable once the map grew,
and codegen died on ``KeyError`` (every regrouped LoRA plan of ``demo/lora.py`` under
``PYTHONHASHSEED=1``).
"""

from cleave.ir.graph import GraphNode
from cleave.runtime.compile import build_candidate_code
from cleave.schedule import apply_tiling, search_tiling
from cleave.schedule import tiling

from tests import _graphs as g

DIMS = (16, 4096, 4096, 16)  # the end2end LoRA case


def test_placeholder_hash_outlives_a_change_of_consumers():
    graph = g.lora_graph(*DIMS)
    x = next(n for n in graph.nodes if n.name == "PlaceHolder_A")
    assert [e.dst_node.name for e in x.outputs] == ["mma_base", "mma_down"]

    tiles = {x: "tile"}
    before = hash(x)
    x._out_edges.pop()
    assert hash(x) == before
    assert tiles[x] == "tile"


def test_shared_placeholder_has_one_reachable_tile(monkeypatch):
    candidates = list(search_tiling(g.lora_graph(*DIMS), topk=1000))[::8]

    # The map is assigned while ``X @ W`` is merged into the loop, whatever the two steps
    # are. Most of these tilings step ``X @ W`` and ``X @ A`` differently, and such a loop
    # is refused once it is complete (its trip count is not the number of tiles of ``X``).
    # That refusal is lifted here, so that every one of them still goes through the merge.
    n_shared = 0
    with monkeypatch.context() as patch:
        patch.setattr(tiling, "_refuse_loops_that_miscount_their_tiles", lambda main: None)
        for cand in candidates:
            label = f"tile={list(cand.output_tile)} rstep={cand.rstep_map}"
            loops = [n for n in apply_tiling(cand).nodes if isinstance(n, GraphNode)]
            for loop in loops:
                keys = list(loop.graph.tmap)
                assert len({id(k) for k in keys}) == len(keys), f"{label}: a node has two tiles"
                assert all(k in loop.graph.tmap for k in keys), (
                    f"{label}: a tile cannot be looked up"
                )
                n_shared += any(k.is_placeholder() and len(k.outputs) > 1 for k in keys)
    assert n_shared, "no loop shares a placeholder between two consumers; the case tests nothing"

    # ... and codegen, which looks every placeholder's tile up, builds the kernel.
    assert "main_kernel" in build_candidate_code(candidates[0])
