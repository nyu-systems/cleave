from __future__ import annotations

import os
from pathlib import Path
import tempfile

_CLEAVE_CACHE_ROOT_ENV = "CLEAVE_CACHE_ROOT"


def _set_env_if_unset(key: str, value: str) -> None:
    if os.environ.get(key) is None:
        os.environ[key] = value


def _default_scratch_root() -> Path:
    scratch_root = os.environ.get("SCRATCH")
    if scratch_root:
        return Path(scratch_root).expanduser()

    user = os.environ.get("USER")
    if user:
        user_scratch = Path("/scratch") / user
        if user_scratch.exists():
            return user_scratch

    return Path(tempfile.gettempdir())


def get_cleave_cache_root() -> Path:
    configured_root = os.environ.get(_CLEAVE_CACHE_ROOT_ENV)
    if configured_root:
        return Path(configured_root).expanduser()

    return _default_scratch_root() / "cleave_cache"


def get_cleave_runtime_root() -> Path:
    return get_cleave_cache_root() / "runtime"


def get_cleave_pid_root(pid: int | None = None) -> Path:
    return get_cleave_runtime_root() / f"pid_{pid or os.getpid()}"


def get_cleave_persistent_cache_root() -> Path:
    return get_cleave_cache_root() / "persistent"


def configure_cache_env() -> Path:
    cache_root = get_cleave_cache_root()
    pid_root = get_cleave_pid_root()
    persistent_root = get_cleave_persistent_cache_root()

    _set_env_if_unset(_CLEAVE_CACHE_ROOT_ENV, str(cache_root))
    _set_env_if_unset("TILELANG_CACHE_DIR", str(persistent_root / "tilelang"))
    _set_env_if_unset("CUDA_CACHE_PATH", str(pid_root / "cuda_cache"))
    _set_env_if_unset("TORCH_EXTENSIONS_DIR", str(pid_root / "torch_extensions"))
    _set_env_if_unset("XDG_CACHE_HOME", str(pid_root / "xdg_cache"))
    _set_env_if_unset("TMPDIR", str(pid_root / "tmp"))

    for key in [
        "TILELANG_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "TORCH_EXTENSIONS_DIR",
        "XDG_CACHE_HOME",
        "TMPDIR",
    ]:
        path = os.environ.get(key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)

    return cache_root
