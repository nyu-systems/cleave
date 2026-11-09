"""Import-direction contract: importing a module must not load a higher layer.

    core <- ir <- superopt <- lower <- schedule <- codegen <- runtime <- dynamic <- api

Every module under ``cleave`` is imported first thing in a fresh interpreter. Lazy imports
inside functions are not checked. ``torch`` is only checked for the GPU-free modules:
TileLang imports it, so every tvm-based layer has it.
"""

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

LAYERS = [  # lowest first
    ("core", ("cleave.core", "cleave.arch", "cleave._cache_env")),
    ("ir", ("cleave.ir", "cleave.visualizer")),
    ("superopt", ("cleave.superoptimizer",)),
    ("lower", ("cleave.lower",)),
    ("schedule", ("cleave.schedule",)),
    ("codegen", ("cleave.codegen",)),
    ("runtime", ("cleave.runtime",)),
    ("dynamic", ("cleave.dynamic",)),
    ("api", ("cleave.frontend", "cleave.pipeline", "cleave.lang", "cleave.examples")),
]
OLD_PATHS = (
    "cleave.graph", "cleave.ops", "cleave.ops.elementwise", "cleave.integration",
    "cleave.integration.superoptimizer", "cleave.integration.graph_processing", "cleave.policy",
    "cleave.policy.default", "cleave.passes", "cleave.paging", "cleave.paging.specs", "cleave.util",
    "cleave.util.analysis", "cleave.util.graph_utils", "cleave.util.tensor_codegen", "cleave.util.runtime",
    "cleave.util.env", "cleave.util.timer", "cleave.util.graph_executor", "cleave.util.autotune",
    "cleave.util.splitk", "cleave._compat",
)
GPU_FREE = ("cleave.core", "cleave.core.errors", "cleave.superoptimizer", "cleave.superoptimizer.schema")
GPU_STACK = ("tvm", "tilelang", "torch")
EXTRA_FORBIDDEN = {"cleave.ir.ops.registry": ("cleave.ir.analysis",)}


def layer_of(module):
    for rank, (_, roots) in enumerate(LAYERS):
        if any(module == r or module.startswith(r + ".") for r in roots):
            return rank
    return None


def _modules():
    root = Path(importlib.util.find_spec("cleave").submodule_search_locations[0])
    suffixes = (".py", *importlib.machinery.EXTENSION_SUFFIXES)
    found = set()
    for p in root.rglob("*"):
        if "__pycache__" in p.parts or not p.name.endswith(suffixes):
            continue
        rel = p.relative_to(root.parent)
        name = ".".join(rel.parts[:-1] + (p.name.split(".")[0],))
        found.add(name.removesuffix(".__init__"))
    return sorted(found - {"cleave"})


MODULES = _modules()

_PROBE = """
import importlib, json, multiprocessing, sys
err = None
try:
    importlib.import_module(sys.argv[1])
except Exception as e:
    err = f"{type(e).__name__}: {e}"
print(json.dumps({
    "error": err,
    "loaded": sorted(m for m in sys.modules if m.split(".")[0] in ("cleave", "tvm", "tilelang", "torch")),
    "start_method": multiprocessing.get_start_method(allow_none=True),
}))
"""


def _probe(module):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, "-c", _PROBE, module], capture_output=True, text=True,
                       env=env, timeout=600)
    lines = p.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else {"error": p.stderr[-2000:]}


@pytest.fixture(scope="module")
def probes():
    with ThreadPoolExecutor(min(32, os.cpu_count() or 4)) as pool:
        return dict(zip(MODULES, pool.map(_probe, MODULES)))


def test_every_module_has_a_layer():
    assert [m for m in MODULES if layer_of(m) is None] == []


@pytest.mark.parametrize("module", MODULES)
def test_import_direction(probes, module):
    r = probes[module]
    assert r["error"] is None, r["error"]
    own = layer_of(module)
    above = [m for m in r["loaded"] if (layer_of(m) or 0) > own]
    assert above == [], f"{module} ({LAYERS[own][0]}) loaded higher layers: {above}"
    bad = [m for m in r["loaded"] if m in EXTRA_FORBIDDEN.get(module, ())]
    if module in GPU_FREE:
        bad += [m for m in r["loaded"] if m.split(".")[0] in GPU_STACK]
    assert bad == [], f"importing {module} loaded {sorted(set(bad))}"
    assert r["start_method"] is None, "multiprocessing start method set at import time"


_OLD_PROBE = """
import importlib, json, sys
import cleave
out = {}
for m in json.loads(sys.argv[1]):
    try:
        importlib.import_module(m)
        out[m] = "imported"
    except ModuleNotFoundError:
        out[m] = None
    except Exception as e:
        out[m] = type(e).__name__
print(json.dumps(out))
"""


def test_old_paths_are_gone():
    import cleave

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, "-c", _OLD_PROBE, json.dumps(OLD_PATHS)], capture_output=True,
                       text=True, env=env, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    got = json.loads(p.stdout.strip().splitlines()[-1])
    assert {m: v for m, v in got.items() if v is not None} == {}
    tops = {m.split(".")[1] for m in OLD_PATHS}
    assert [t for t in tops if hasattr(cleave, t)] == []
