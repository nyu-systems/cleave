"""Content-addressed cache paths under ``current_session().cache_root / "persistent"``."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

# In every key. Bump it when the layout changes, or when the same program and configs would
# now build a different kernel: a kernel-cache hit returns the saved one as it is.
# 2: a reduction absorbed into another's loop is scheduled over its whole axis.
# 3: lowering takes broadcast and reduction axes from the structure (not from extents that
#    happen to agree), and a kernel's inputs are bound in the program's declaration order.
# 4: an op that no output reads is not lowered (it used to hand its shape to the inputs it
#    read), and a divisor over two unit axes is broadcast along the one the program names.
# 5: a reduction that is not looped reduces its own axis of the stored tile.
# 6: a candidate with an absorbed reduction is rated (and cut at top-k) as its kernel.
# 7: a loop tier carries the dims its block stores (single-row loop kernels now build).
# 8: a split combine builds wherever its two sums are placed; work no loop reads is
#    emitted behind the loops.
# 9: split partials validate (in their stored layout) and chain into the combine, so a
#    program's best kernel can now be a split variant; saved stages carry that layout.
# 10: gqa_paged_decode_h32_kv8_d128_ps1 builds the exact (running-max) softmax: the same
#     paged program now emits a different kernel. (Paged kernels are not kernel-cached, so
#     no stale one can be returned today.)
# 11: cl.softmax records the exact softmax as a request on its exp (no max in the program),
#     every schedule of such a program gets its own shift (in-tile max or running max), and
#     its kernels are validated on the adversarial softmax inputs as well.
#     tests/pipeline_api/test_schema_guard.py pins the text of a few such kernels next to
#     this number: text that changes without a bump here fails there.
# 12: a running max whose accumulators are too wide for a thread's registers is rescaled
#     lazily (shift on a grid, guarded rescale): the exact MLA decode builds another kernel.
# 13: a softmax over an axis that is not the last one takes its max in place wherever a
#     tile holds that axis (in a loop that does not tile it, behind a squeezed unit dim):
#     tilings that were refused now build, so a program's top-k and best kernel can change.
# 14: every paged layer builds the exact softmax, and a split-KV pair hands the partials'
#     running max to its combine as a third partial: every paged program emits a different
#     kernel, and every split pair a different parameter list.
# 15: an input the graph declares without its unit axes is bound to its program input (the
#     Kernel saved for such a program could not be called).
# 16: the outputs next to the canonical one are launched on its axes: a split partial whose sums
#     take the whole per-split extent in one step now builds (it was refused), so a program's
#     best partial can be one of those.
# 17: sibling reductions over one axis start at one step, so the loop they share counts its
#     tiles: a float32 flash-form softmax now has tilings (every one was refused).
SCHEMA = 17
# What "validated" means for a saved kernel. A kernel-cache hit returns without validating,
# so bump this whenever the comparison or its inputs change: saved kernels are then rebuilt.
# (The inputs include the adversarial cases a program with an exact softmax is held to,
# cleave.runtime.inputs.ADVERSARIAL, which are not in the key by name.)
# 3: the adversarial cases shape the logits of the exact softmax only; the operands of a
#    plain exp written next to it are left alone (its reference overflowed on them).
VALIDATION = 3


@functools.lru_cache(maxsize=None)
def build_id() -> str:
    """Cleave version + a digest of the compiled superoptimizer extension."""
    from .. import superoptimizer as csopt

    so = Path(csopt.core.__file__)
    h = hashlib.sha256()
    with open(so, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{csopt.__version__}+{h.hexdigest()[:12]}"


def digest(*parts) -> str:
    blob = json.dumps([SCHEMA, build_id(), *parts], sort_keys=True, default=str,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def root() -> Path:
    from ..core.session import current_session

    return current_session().cache_root / "persistent"


def superopt_key(program, max_ops: int, split: tuple, concat_matmul: bool) -> str:
    # threads and cache only change how the search runs, not what it finds.
    return digest("superopt", program.fingerprint(), int(max_ops), sorted(split), bool(concat_matmul))


def checkpoint_path(key: str) -> Path:
    d = root() / "superopt" / key[:2] / key
    d.mkdir(parents=True, exist_ok=True)
    return d / "checkpoint.json"


def kernel_dir(name: str, key: str) -> Path:
    return root() / "kernels" / f"{name}-{key[:16]}"
