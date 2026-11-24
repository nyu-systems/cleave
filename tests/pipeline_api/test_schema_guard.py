"""The kernel-cache schema follows the generated text of ``cl.softmax`` kernels (CPU).

A kernel-cache hit returns the saved kernel without building or validating anything, and the
key holds the program's fingerprint, the configs and ``pipeline.cache.SCHEMA``. A change to
the tiler, the update search, the assembly of a carried max or codegen changes what the same
program builds without changing any of the first two, so it has to come with a bump of the
third; nothing else tells a saved kernel that it is stale.

This pins the text (by digest) of a few kernels of ``cl.softmax`` programs (the max in the
flat graph), next to the ``SCHEMA`` they were generated under. They are identity graphs
(lowered as traced, with the update rules ``plans.lower`` derives; no search), one tiling
each: a tile that holds the row whole (the flat graph's own max), a loop that carries the
max, both of one graph, and a carried max over accumulators wide enough that the lazy form of
the hand-derived pass rescaled them only when the max moved (here: every iteration).
When a digest moves, the test fails until ``SCHEMA`` is above the pinned value and the new
digests are pinned with it.

    python -m tests.pipeline_api.test_schema_guard        # print the table to pin
"""

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("tilelang")

import cleave.lang as cl
from cleave.pipeline import cache
from cleave.runtime.compile import build_candidate_code
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet
from cleave.frontend import trace
from cleave.schedule import apply_tiling, search_tiling
from cleave.schedule.streaming import sites, streaming_accept

from tests import _explicit_softmax as X


def softmax_rows(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1)


def softmax_mid(x: cl.Tensor["H", "C", "R"]):
    return cl.softmax(x, 1)


def attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
              v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


def attention_over_queries(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                           v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-2) @ v


def flash_wide_values(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                      v: cl.Tensor["H", "SKV", "DV"]):
    """Flash form with values wider than the keys: a 64 x 512 accumulator of ``P @ V``."""
    e = X.stable_exp((q @ k.mT) / q.shape[-1] ** 0.5)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


SHORT, LONG = dict(H=32, SQ=64, SKV=256, D=64), dict(H=32, SQ=4096, SKV=4096, D=64)
WIDE = dict(H=8, SQ=64, SKV=1024, D=64, DV=512)
# name -> (program, shapes, how the pinned tiling serves the softmax). The flash form is the
# one the search finds for attention (the row sum next to the value matmul); written out
# here with the max ``cl.softmax`` records.
KERNELS = {
    "softmax_64x1024/one-tile": (softmax_rows, dict(R=64, C=1024), "one-tile"),
    # a softmax axis that is not the last: behind a squeezed batch dim, and inside the loop
    # of the value matmul
    "softmax_mid_4x256x64/one-tile": (softmax_mid, dict(H=4, C=256, R=64), "one-tile"),
    "attention_over_queries_skv1024/one-tile": (
        attention_over_queries, dict(H=32, SQ=64, SKV=1024, D=64), "one-tile"),
    "attention_skv256/one-tile": (attention, SHORT, "one-tile"),
    "flash_skv256/one-tile": (X.flash_attention, SHORT, "one-tile"),
    "flash_skv256/looped": (X.flash_attention, SHORT, "looped"),
    "flash_skv4096/looped": (X.flash_attention, LONG, "looped"),
    "flash_wide_values/looped": (flash_wide_values, WIDE, "looped"),
}
# name -> the output tile of the pinned tiling, where it is not the first one of its kind:
# the whole 64 x 512 value accumulator in one block (256 elements a thread), which the
# hand-derived pass's lazy form rescaled only when the max moved; here it is rescaled every
# iteration.
OUTPUT_TILE = {"flash_wide_values/looped": [1, 64, 512]}

# The SCHEMA these digests were generated under, and the digests. Re-pin both together.
# (A bump that changes none of these eight texts leaves both as they are.) Pinned again at
# the move to the searched update rules, where every softmax program gained its max, and at
# 43, where a gemm stopped rebinding its shared and register operands to ``A_ptr`` /
# ``B_ptr`` (tests/codegen/test_operand_names.py).
PINNED_SCHEMA = 43
PINNED = {
    'softmax_64x1024/one-tile': ('tile=[2, 1024] rstep=[64, 1024]', 'f9936a0218828a65'),
    'softmax_mid_4x256x64/one-tile': ('tile=[1, 256, 2] rstep=[2, 256]', '86aa9f6b4d2251bd'),
    'attention_over_queries_skv1024/one-tile': ('tile=[1, 64, 32] rstep=[32, 1, 64, 32]', '5afdf6676aeee861'),
    'attention_skv256/one-tile': ('tile=[1, 64, 32] rstep=[32, 256, 256, 256]', '3cf8d3f796b1497d'),
    'flash_skv256/one-tile': ('tile=[1, 64, 32] rstep=[32, 256, 256, 256]', '478b000ea10efe33'),
    'flash_skv256/looped': ('tile=[1, 64, 32] rstep=[32, 32, 32, 32]', 'fccc101cff42d630'),
    'flash_skv4096/looped': ('tile=[1, 128, 64] rstep=[32, 32, 32, 32]', '143b00f3a86ae06d'),
    'flash_wide_values/looped': ('tile=[1, 64, 512] rstep=[32, 32, 32, 32]', 'c1111b823b651c2b'),
}


def _kind(tiled) -> str:
    """``looped`` where a loop carries the max (one site), ``one-tile`` where the tile holds
    the row whole and the flat graph's own max stands (no site)."""
    found = sites(tiled)
    assert len(found) <= 1, found
    return "looped" if found else "one-tile"


def _lowered(fn, shapes, dtype=cl.f16):
    (graph,) = list(GraphSet.identity(trace(fn, shapes, dtype)))
    (ir,) = lower(graph)
    return ir


def digests() -> dict:
    """``name -> (tiling label, digest of the generated text)``: for each kernel the first
    128-thread tiling the accept filter keeps that serves the softmax the named way (with
    the output tile of ``OUTPUT_TILE``, where one is named)."""
    out = {}
    for name, (fn, shapes, case) in KERNELS.items():
        ir = _lowered(fn, shapes)
        accept = streaming_accept(ir)
        assert ir.nodes and accept is not None, f"{name}: no streaming obstacle"
        for cand in search_tiling(ir, topk=1000, accept=accept):
            tile = OUTPUT_TILE.get(name, list(cand.output_tile))
            if (cand.n_threads == 128 and list(cand.output_tile) == tile
                    and _kind(apply_tiling(cand)) == case):
                break
        else:
            raise AssertionError(f"{name}: no {case} tiling among the accepted candidates")
        steps = [int(x) for v in cand.rstep_map.values() for x in (v.values() if v else ())]
        label = f"tile={list(cand.output_tile)} rstep={steps}"
        text = build_candidate_code(cand)
        out[name] = (label, hashlib.sha256(text.encode()).hexdigest()[:16])
    return out


def stale(got: dict, schema: int) -> str | None:
    """What is wrong with the pins for the digests ``got`` under ``schema``, or None."""
    moved = sorted(name for name in set(got) | set(PINNED) if got.get(name) != PINNED.get(name))
    if not moved:
        # The text is what was pinned. SCHEMA may have moved on for another reason.
        if schema < PINNED_SCHEMA:
            return f"pipeline.cache.SCHEMA went back from {PINNED_SCHEMA} to {schema}"
        return None
    table = "\n".join(f"    {name!r}: {got[name]!r}," for name in got)
    if schema <= PINNED_SCHEMA:
        return (
            f"the generated text of {moved} changed, but pipeline.cache.SCHEMA is still "
            f"{schema}: a kernel saved before this change would be returned as it is. Bump "
            "SCHEMA (say what changed in the commit message), then pin it and these digests in "
            f"tests/pipeline_api/test_schema_guard.py:\n{table}")
    return (
        f"the generated text of {moved} changed and pipeline.cache.SCHEMA was bumped to "
        f"{schema}: pin PINNED_SCHEMA = {schema} and these digests in "
        f"tests/pipeline_api/test_schema_guard.py:\n{table}")


def test_generated_text_of_softmax_kernels_is_pinned_with_the_schema():
    problem = stale(digests(), cache.SCHEMA)
    assert problem is None, problem


def test_a_text_change_fails_until_the_schema_is_bumped_and_pinned():
    assert stale(dict(PINNED), PINNED_SCHEMA) is None
    assert stale(dict(PINNED), PINNED_SCHEMA + 3) is None  # bumped for another reason
    assert "went back" in stale(dict(PINNED), PINNED_SCHEMA - 1)
    name = "flash_skv256/looped"
    label, digest = PINNED[name]
    changed = {**PINNED, name: (label, digest[::-1])}
    no_bump = stale(changed, PINNED_SCHEMA)
    assert "is still" in no_bump and name in no_bump and digest[::-1] in no_bump
    bumped = stale(changed, PINNED_SCHEMA + 1)
    assert "was bumped" in bumped and f"PINNED_SCHEMA = {PINNED_SCHEMA + 1}" in bumped
    # a pinned kernel that is no longer generated, or a new one, is a change as well
    assert name in stale({k: v for k, v in PINNED.items() if k != name}, PINNED_SCHEMA)
    assert "'new'" in stale({**PINNED, "new": ("tile=[1]", "0" * 16)}, PINNED_SCHEMA)


def test_the_guard_pins_both_kinds():
    """Each pinned kernel is the kind its name says: the row whole in one tile (the flat
    graph's own max), or the max carried by a loop."""
    assert {name.rpartition("/")[2] for name in PINNED} == {"one-tile", "looped"}
    assert set(PINNED) == set(KERNELS)
    for name, (label, digest) in PINNED.items():
        assert label.startswith("tile=[") and len(digest) == 16, name


if __name__ == "__main__":
    print(f"PINNED_SCHEMA = {cache.SCHEMA}\nPINNED = {{")
    for name, value in digests().items():
        print(f"    {name!r}: {value!r},")
    print("}")
