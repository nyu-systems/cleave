"""``Session``: the target, cache root, parallelism and settings the pipeline stages use.

``with cleave.Session(cache_root=..., max_parallel=4): cleave.compile(...)``. Outside a
``with`` block, ``current_session()`` is built from ``Settings.from_env()`` and
``current_target()``.
"""

from __future__ import annotations

import dataclasses
from contextvars import ContextVar
from pathlib import Path
from typing import Optional, Tuple, Union

from .settings import Settings
from .target import Target, default_target

# The ``with`` stack of this context (per thread / task), innermost last.
_ACTIVE: ContextVar[Tuple["Session", ...]] = ContextVar("cleave_sessions", default=())


class Session:
    def __init__(self, target: Union[Target, str, None] = None, cache_root=None,
                 max_parallel: Optional[int] = None, settings: Optional[Settings] = None):
        settings = settings or Settings.from_env()
        if cache_root is not None:
            settings = dataclasses.replace(settings, cache_root=Path(cache_root).expanduser())
        if max_parallel is not None and int(max_parallel) < 1:
            raise ValueError(f"max_parallel must be >= 1, got {max_parallel}")
        self.settings = settings
        self.explicit_target = Target(target) if isinstance(target, str) else target
        self.max_parallel = None if max_parallel is None else int(max_parallel)

    @property
    def target(self) -> Target:
        return self.explicit_target or default_target()

    @property
    def cache_root(self) -> Path:
        return Path(self.settings.cache_root)

    def __enter__(self) -> "Session":
        _ACTIVE.set(_ACTIVE.get() + (self,))
        return self

    def __exit__(self, *exc) -> None:
        stack = _ACTIVE.get()
        if not stack or stack[-1] is not self:
            raise RuntimeError("Session exited out of order")
        _ACTIVE.set(stack[:-1])

    def __repr__(self):
        return (f"Session(target={self.target!r}, cache_root={str(self.cache_root)!r}, "
                f"max_parallel={self.max_parallel})")


def active_session() -> Optional[Session]:
    """The innermost ``with Session(...)``, or None."""
    stack = _ACTIVE.get()
    return stack[-1] if stack else None


def current_session() -> Session:
    return active_session() or Session()
