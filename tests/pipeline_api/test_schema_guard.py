"""The kernel-cache schema follows the generated text of ``cl.softmax`` kernels (CPU).

A kernel-cache hit returns the saved kernel without building or validating anything, and the
key holds the program's fingerprint, the configs and ``pipeline.cache.SCHEMA``. A change to
the tiler, the stabilization rewrite or codegen changes what the same program builds without
changing any of the first two, so it has to come with a bump of the third; nothing else
tells a saved kernel that it is stale.

This pins the text (by digest) of a few kernels of programs that record the exact softmax,
next to the ``SCHEMA`` they were generated under. They are identity graphs (lowered as
traced, no search), one tiling each: an in-tile max, a running max, both of one graph, and
a running max over accumulators wide enough for the lazily rescaled form.
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
from cleave.schedule import apply_tiling, search_tiling
from cleave.schedule.stabilize import sites, stabilize_accept

from tests import _softmax as S


def softmax_rows(x: cl.Tensor["R", "C"]):
    return cl.softmax(x, -1)


def attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
              v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


def flash_wide_values(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                      v: cl.Tensor["H", "SKV", "DV"]):
    """Flash form with values wider than the keys: a 64 x 512 accumulator of ``P @ V``."""
    e = S.stable_exp((q @ k.mT) / q.shape[-1] ** 0.5)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


SHORT, LONG = dict(H=32, SQ=64, SKV=256, D=64), dict(H=32, SQ=4096, SKV=4096, D=64)
WIDE = dict(H=8, SQ=64, SKV=1024, D=64, DV=512)
# name -> (program, shapes, how the pinned tiling serves the softmax). The flash form is the
# one the search finds for attention (the row sum next to the value matmul); written out
# here, it makes the request the way ``cl.softmax`` does.
KERNELS = {
    "softmax_64x1024/in-tile": (softmax_rows, dict(R=64, C=1024), "in-tile"),
    "attention_skv256/in-tile": (attention, SHORT, "in-tile"),
    "flash_skv256/in-tile": (S.flash_attention, SHORT, "in-tile"),
    "flash_skv256/rolling": (S.flash_attention, SHORT, "rolling"),
    "flash_skv4096/rolling": (S.flash_attention, LONG, "rolling"),
    "flash_wide_values/rolling-lazy": (flash_wide_values, WIDE, "rolling-lazy"),
}

# The SCHEMA these digests were generated under, and the digests. Re-pin both together.
PINNED_SCHEMA = 12
PINNED = {
    'softmax_64x1024/in-tile': ('tile=[2, 1024] rstep=[1024]', '8211bb6438573d68'),
    'attention_skv256/in-tile': ('tile=[1, 64, 32] rstep=[32, 256, 256]', 'd5982c362d1466dd'),
    'flash_skv256/in-tile': ('tile=[1, 64, 32] rstep=[32, 256, 256]', '9464e85013b3084e'),
    'flash_skv256/rolling': ('tile=[1, 64, 32] rstep=[32, 32, 32]', '963ffee491434feb'),
    'flash_skv4096/rolling': ('tile=[1, 128, 64] rstep=[32, 32, 32]', '6f82f9c294167554'),
    'flash_wide_values/rolling-lazy': ('tile=[1, 64, 512] rstep=[32, 32, 32]', 'cf186bc349c18508'),
}


def _kind(site) -> str:
    return site.case + ("-lazy" if site.lazy else "")


def digests() -> dict:
    """``name -> (tiling label, digest of the generated text)``: for each kernel the first
    128-thread tiling the accept filter keeps that serves the softmax the named way."""
    out = {}
    for name, (fn, shapes, case) in KERNELS.items():
        ir = S.lowered(fn, shapes)
        assert ir.nodes and stabilize_accept(ir) is not None, f"{name}: no softmax request"
        for cand in search_tiling(ir, topk=1000, accept=stabilize_accept(ir)):
            if cand.n_threads == 128 and [_kind(s) for s in sites(apply_tiling(cand))] == [case]:
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
            "SCHEMA (say what changed in its comment), then pin it and these digests in "
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
    name = "flash_skv256/rolling"
    label, digest = PINNED[name]
    changed = {**PINNED, name: (label, digest[::-1])}
    no_bump = stale(changed, PINNED_SCHEMA)
    assert "is still" in no_bump and name in no_bump and digest[::-1] in no_bump
    bumped = stale(changed, PINNED_SCHEMA + 1)
    assert "was bumped" in bumped and f"PINNED_SCHEMA = {PINNED_SCHEMA + 1}" in bumped
    # a pinned kernel that is no longer generated, or a new one, is a change as well
    assert name in stale({k: v for k, v in PINNED.items() if k != name}, PINNED_SCHEMA)
    assert "'new'" in stale({**PINNED, "new": ("tile=[1]", "0" * 16)}, PINNED_SCHEMA)


def test_the_guard_pins_both_rewrites():
    """Each pinned kernel is the kind its name says: an in-tile max, a running max, and a
    running max that is rescaled lazily."""
    assert {name.rpartition("/")[2] for name in PINNED} == {"in-tile", "rolling", "rolling-lazy"}
    assert set(PINNED) == set(KERNELS)
    for name, (label, digest) in PINNED.items():
        assert label.startswith("tile=[") and len(digest) == 16, name


if __name__ == "__main__":
    print(f"PINNED_SCHEMA = {cache.SCHEMA}\nPINNED = {{")
    for name, value in digests().items():
        print(f"    {name!r}: {value!r},")
    print("}")
