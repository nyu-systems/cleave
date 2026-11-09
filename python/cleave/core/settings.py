from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from .._cache_env import get_cleave_cache_root


def _positive_float(env: Mapping[str, str], name: str, default: float) -> Optional[float]:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else None


def _bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    return default


@dataclass(frozen=True)
class Settings:
    compile_timeout_secs: Optional[float] = 300.0
    heartbeat_secs: Optional[float] = 15.0
    progress_log: bool = False
    test_seed: Optional[int] = None
    cache_root: Optional[Path] = None

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        env = os.environ if env is None else env
        seed = env.get("CLEAVE_TEST_SEED")
        return cls(
            compile_timeout_secs=_positive_float(env, "CLEAVE_COMPILE_AND_RUN_TIMEOUT_SECS", 300.0),
            heartbeat_secs=_positive_float(env, "CLEAVE_COMPILE_PROGRESS_HEARTBEAT_SECS", 15.0),
            progress_log=_bool(env, "COMPILATION_PROGRESS_LOG"),
            test_seed=None if seed is None else int(seed) & 0xFFFFFFFF,
            cache_root=get_cleave_cache_root(env),
        )
