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
SCHEMA = 5
# What "validated" means for a saved kernel. A kernel-cache hit returns without validating,
# so bump this whenever the comparison or its inputs change: saved kernels are then rebuilt.
VALIDATION = 2


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
