"""Content-addressed cache paths under ``current_session().cache_root / "persistent"``."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

SCHEMA = 1


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
