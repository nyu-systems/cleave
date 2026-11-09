import json
import os
import subprocess
import sys

_MODULES = [
    "cleave",
    "cleave.core",
    "cleave.ir.graph",
    "cleave.ir.ops",
    "cleave.lower",
    "cleave.schedule",
    "cleave.codegen",
    "cleave.codegen.passes",
    "cleave.runtime.evaluate",
]

_SCRIPT = """
import importlib, json, os, sys
before = dict(os.environ)
for m in sys.argv[1:]:
    importlib.import_module(m)
from cleave.ir.graph_utils import clone_graph
from cleave.runtime.compile import check_code
from cleave.runtime.evaluate import run_candidates
changed = sorted(k for k in set(before) | set(os.environ) if before.get(k) != os.environ.get(k))
print(json.dumps(changed))
"""

_CLEAVE_ENV = {
    "CLEAVE_CACHE_ROOT",
    "CUDA_CACHE_PATH",
    "TORCH_EXTENSIONS_DIR",
    "XDG_CACHE_HOME",
    "TMPDIR",
    "TILELANG_CACHE_DIR",
}


def test_import_without_gpu_does_not_touch_env(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in _CLEAVE_ENV}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CLEAVE_CACHE_ROOT"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, *_MODULES],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    changed = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert not changed & _CLEAVE_ENV, changed
    assert not (tmp_path / "runtime").exists()


def test_configured_pid_dir_removed_at_exit(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in _CLEAVE_ENV}
    env["CLEAVE_CACHE_ROOT"] = str(tmp_path)
    script = (
        "import os; from cleave._cache_env import configure_cache_env, get_cleave_pid_root;"
        "configure_cache_env(); assert os.path.isdir(os.environ['TMPDIR']);"
        "print(get_cleave_pid_root())"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    pid_root = proc.stdout.strip().splitlines()[-1]
    assert pid_root.startswith(str(tmp_path))
    assert not os.path.exists(pid_root)
    assert (tmp_path / "persistent" / "tilelang").is_dir()
