"""demo/*.py emit their recorded search: csopt builder calls, max_ops and split factors
(``snapshots/builder_calls.json``; ``--update-snapshots`` rewrites it).

Each demo runs as ``__main__`` with ``cleave.examples``' compile stubbed (no search, no GPU).
"""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import cleave.examples as examples

from ..frontend.test_trace_equivalence import normalized, spy
from ._snapshot import SNAPSHOT, record, updating

DEMO = Path(__file__).resolve().parents[2] / "demo"
HELPERS = {"attention"}
DEMOS = sorted(p.stem for p in DEMO.glob("*.py") if p.stem not in HELPERS)


class _Stop(Exception):
    pass


def run_demo(monkeypatch, name, argv):
    seen = {}

    def fake_compile(program, *, superopt, schedule, eval, **kw):
        seen.update(program=program, so=superopt, sc=schedule, ev=eval)
        raise _Stop

    monkeypatch.setattr(examples, "_compile", fake_compile)
    monkeypatch.setattr(examples, "_target", lambda: "cpu")
    monkeypatch.setattr(sys, "argv", [str(DEMO / f"{name}.py"), *argv])
    monkeypatch.syspath_prepend(str(DEMO))
    with pytest.raises(_Stop):
        runpy.run_path(str(DEMO / f"{name}.py"), run_name="__main__")
    return seen


CASES = [
    # id, demo, argv
    ("matmul", "matmul", []),
    ("matmul-split", "matmul", ["--split", "K:4,8,16"]),
    ("matmul-split-K", "matmul", ["--split", "K"]),
    ("rmsnorm_matmul", "rmsnorm_matmul", []),
    ("rmsnorm_matmul-split-k", "rmsnorm_matmul", ["--preset", "split-k"]),
    ("swiglu", "swiglu", []),
    ("lora", "lora", []),
    ("matmul_rowsum", "matmul_rowsum", []),
    ("matmul_matmul", "matmul_matmul", []),
    ("matmul_matmul-split", "matmul_matmul", ["--preset", "split"]),
    ("matmul_matmul-split-off", "matmul_matmul", ["--preset", "split", "--split", "none", "--max-ops", "5"]),
    *[(f"{v}_decode{s}", f"{v}_decode", argv)
      for v in ("mha", "gqa", "mqa") for s, argv in (("", []), ("-nosplit", ["--split", "none"]))],
    *[(f"{v}_prefill", f"{v}_prefill", []) for v in ("mha", "gqa", "mqa", "mla")],
    ("mla_decode", "mla_decode", []),
    ("qknorm", "qknorm", []),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_demo_emits_recorded_search(monkeypatch, request, case):
    cid, name, argv = case
    seen = run_demo(monkeypatch, name, argv)
    program, so, sc = seen["program"], seen["so"], seen["sc"]
    with spy(monkeypatch) as log:
        program.to_tile_graph(split=so.split)
    if updating(request):
        record("demos", cid, {"calls": normalized(log), "max_ops": so.resolve_max_ops(program),
                              "split_k": list(sc.split_k)})
    rec = SNAPSHOT["demos"][cid]
    assert normalized(log) == rec["calls"]
    assert so.resolve_max_ops(program) == rec["max_ops"]
    assert sc.split_k == tuple(rec["split_k"])
    assert sc.topk == 1000
    assert sc.fusion == ("force" if name == "qknorm" else "auto")


def test_every_demo_has_a_case():
    covered = {c[1] for c in CASES}
    assert set(DEMOS) - covered == {"staged", "mha_decode_exact"}
    assert {c[0] for c in CASES} == set(SNAPSHOT["demos"])


def test_mha_decode_exact_config(monkeypatch):
    seen = run_demo(monkeypatch, "mha_decode_exact", [])
    program, so = seen["program"], seen["so"]
    assert [r.kind for r in program.ops if r.kind != "input"] == [
        "matmul", "div", "reduce_max", "sub", "exp", "reduce_sum", "div", "matmul"]
    assert not so.split and so.resolve_max_ops(program) == 14
    assert program.sizes == {"H": 32, "SQ": 64, "D": 64, "SKV": 256}


@pytest.mark.parametrize("name", DEMOS)
def test_demo_imports_without_gpu_work(name):
    # fresh interpreter: another test in this worker may already have initialized CUDA
    code = (
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(DEMO)!r})\n"
        f"ns = runpy.run_path({str(DEMO / (name + '.py'))!r}, run_name='_demo')\n"
        f"assert {name == 'staged'} or callable(ns.get({name!r}))\n"
        "torch = sys.modules.get('torch')\n"
        "assert torch is None or not torch.cuda.is_initialized()\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]


@pytest.mark.parametrize("name", DEMOS)
def test_demo_cli_help(monkeypatch, capsys, name):
    monkeypatch.setattr(sys, "argv", [str(DEMO / f"{name}.py"), "--help"])
    monkeypatch.syspath_prepend(str(DEMO))
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(DEMO / f"{name}.py"), run_name="__main__")
    assert e.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_demo_cli_overrides(monkeypatch):
    seen = run_demo(monkeypatch, "mha_decode", ["--SKV", "2048", "--preset", "fast", "--topk", "20",
                                                "--split", "SKV:8", "--max-ops", "15"])
    assert seen["program"].sizes["SKV"] == 2048
    assert (seen["sc"].topk, seen["sc"].split_k, seen["so"].split, seen["so"].max_ops) == (20, (8,), ("SKV",), 15)
    assert seen["ev"].iters == 100  # the fast preset
