from __future__ import annotations

import atexit
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Optional

_CLEAVE_CACHE_ROOT_ENV = "CLEAVE_CACHE_ROOT"
_PID_SCOPED_ENV = {
    "CUDA_CACHE_PATH": "cuda_cache",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "XDG_CACHE_HOME": "xdg_cache",
    "TMPDIR": "tmp",
}
_CONFIGURED_PID: Optional[int] = None


def _default_scratch_root(env: Mapping[str, str]) -> Path:
    scratch_root = env.get("SCRATCH")
    if scratch_root:
        return Path(scratch_root).expanduser()

    user = env.get("USER")
    if user:
        user_scratch = Path("/scratch") / user
        if user_scratch.exists():
            return user_scratch

    return Path(tempfile.gettempdir())


def get_cleave_cache_root(env: Optional[Mapping[str, str]] = None) -> Path:
    env = os.environ if env is None else env
    configured_root = env.get(_CLEAVE_CACHE_ROOT_ENV)
    if configured_root:
        return Path(configured_root).expanduser()

    return _default_scratch_root(env) / "cleave_cache"


def get_cleave_runtime_root() -> Path:
    return get_cleave_cache_root() / "runtime"


def get_cleave_pid_root(pid: int | None = None) -> Path:
    return get_cleave_runtime_root() / f"pid_{pid or os.getpid()}"


def get_cleave_persistent_cache_root() -> Path:
    return get_cleave_cache_root() / "persistent"


def _remove_pid_root(pid_root: Path, owner_pid: int) -> None:
    # Bound at registration: a forked child that reconfigures must not remove this.
    if owner_pid == os.getpid():
        shutil.rmtree(pid_root, ignore_errors=True)


def configure_cache_env() -> Path:
    """Point TileLang/CUDA/torch caches under the cleave cache root (once per process).

    Variables already set (e.g. inherited from a parent cleave process) are left alone;
    the per-pid directory this process creates is removed at exit.
    """
    global _CONFIGURED_PID
    cache_root = get_cleave_cache_root()
    if _CONFIGURED_PID == os.getpid():
        return cache_root

    pid_root = get_cleave_pid_root()
    os.environ.setdefault(_CLEAVE_CACHE_ROOT_ENV, str(cache_root))
    os.environ.setdefault(
        "TILELANG_CACHE_DIR", str(get_cleave_persistent_cache_root() / "tilelang")
    )
    owns_pid_root = False
    for key, sub in _PID_SCOPED_ENV.items():
        if os.environ.get(key) is None:
            os.environ[key] = str(pid_root / sub)
            owns_pid_root = True

    for key in ["TILELANG_CACHE_DIR", *_PID_SCOPED_ENV]:
        path = os.environ.get(key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)

    _CONFIGURED_PID = os.getpid()
    if owns_pid_root:
        atexit.register(_remove_pid_root, pid_root, _CONFIGURED_PID)
    return cache_root
