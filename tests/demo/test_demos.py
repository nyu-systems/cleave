"""demo/*.py emit their recorded search: csopt builder calls, max_ops and split factors
(``snapshots/builder_calls.json``; ``--update-snapshots`` rewrites it).

Each demo runs as ``__main__`` with ``cleave.examples``' compile stubbed (no search, no GPU).
"""

import json
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

import cleave.examples as examples
from cleave.runtime.validate import Tolerance

from .._measure_tolerances import MEASURED, expected_tolerance, expected_tolerances_by_role
from .._tolerances import CEILING, FP32_CEILING
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
    assert set(DEMOS) - covered == {"staged"}
    assert {c[0] for c in CASES} == set(SNAPSHOT["demos"])


# The demos whose softmax is ``cl.softmax``, the exact one: every attention demo.
ATTENTION = ["mha_decode", "gqa_decode", "mqa_decode", "mha_prefill", "gqa_prefill",
             "mqa_prefill", "mla_prefill", "mla_decode", "qknorm"]
# The demos whose kernels can return an output with a boundary role: an exact softmax,
# split over the keys (by default for the decode demos, with the ``--split SKV`` their
# docstrings name for the MHA / GQA / MQA prefill demos). Its partials return their float32
# shift, so the demo declares one bound per role. mla_decode, mla_prefill and qknorm offer
# no split: no split graph of mla_decode has a partial / combine cut, the split graphs of
# mla_prefill need about 190 GB of references, and qknorm's split search is no demo run.
BY_ROLE = set(ATTENTION) - {"mla_decode", "mla_prefill", "qknorm"}


def _assert_declared(config, name):
    """One measured bound for the operator, within the ceiling (one per boundary role for a
    demo of ``BY_ROLE``); it asks for normal inputs."""
    tolerance = config.tolerance
    measured = json.loads(MEASURED.read_text())[f"demo:{name}"]
    if name in BY_ROLE:
        assert isinstance(tolerance, dict) and set(tolerance) == {"", "shift"}
        expected = expected_tolerances_by_role(measured)
        shift, tolerance = tolerance["shift"], tolerance[""]
        assert isinstance(shift, Tolerance) and 0 < shift.rtol <= FP32_CEILING
        assert shift.atol == 0 and shift == expected["shift"]
        expected = expected[""]
    else:
        expected = expected_tolerance(measured)
    assert isinstance(tolerance, Tolerance), "the demo declares no tolerance"
    assert 0 < tolerance.rtol <= CEILING and tolerance.atol == 0
    # ... and exactly the one the rule gives for the pinned measurement of this demo
    assert tolerance == expected
    # No demo asks for more than the normal inputs. (The build holds a program whose softmax
    # is exact to the adversarial cases as well, whatever is asked: pipeline.build.)
    assert config.distributions == ("normal",)


@pytest.mark.parametrize("name", [d for d in DEMOS if d != "staged"])
def test_demo_declares_its_own_tolerance(monkeypatch, name):
    # There is no default bound to fall back to: a demo without one cannot validate.
    _assert_declared(run_demo(monkeypatch, name, [])["ev"], name)


def test_staged_demo_declares_its_own_tolerance(monkeypatch):
    import cleave.pipeline as pipeline

    seen = {}

    def fake_build(plans, workers=None, *, config=None):
        seen["config"] = config
        raise _Stop

    monkeypatch.setattr(pipeline, "superoptimize", lambda program, config=None: "graphs")
    monkeypatch.setattr(pipeline, "schedule",
                        lambda graphs, config=None: types.SimpleNamespace(summary=lambda: ""))
    monkeypatch.setattr(pipeline, "build", fake_build)
    monkeypatch.setattr(sys, "argv", [str(DEMO / "staged.py")])
    monkeypatch.syspath_prepend(str(DEMO))
    with pytest.raises(_Stop):
        runpy.run_path(str(DEMO / "staged.py"), run_name="__main__")
    _assert_declared(seen["config"], "staged")
    assert (seen["config"].warmup, seen["config"].iters) == (16, 1000)


def test_mha_decode_config(monkeypatch):
    """The decode demo: no max in the program (the request stands for it), the split over
    the keys, and the search budget the auto rule gives."""
    seen = run_demo(monkeypatch, "mha_decode", [])
    program, so, sc = seen["program"], seen["so"], seen["sc"]
    assert [r.kind for r in program.ops if r.kind != "input"] == [
        "matmul", "div", "exp", "reduce_sum", "div", "matmul"]
    assert len(program.stabilize_requests) == 1
    assert so.split == ("SKV",) and sc.split_k == (16, 32)
    assert so.max_ops == "auto" and so.resolve_max_ops(program) == 13
    assert program.sizes == {"H": 32, "SQ": 64, "D": 64, "SKV": 4096}
    # --split none: the solo kernels only, on the same budget
    solo = run_demo(monkeypatch, "mha_decode", ["--split", "none"])
    assert not solo["so"].split and not solo["sc"].split_k
    assert solo["so"].resolve_max_ops(solo["program"]) == 13


@pytest.mark.parametrize("name", ATTENTION)
def test_attention_demo_softmax_is_exact(monkeypatch, name):
    """No attention demo shifts its scores by a constant: each records the one request of
    ``cl.softmax`` on an exp that reads the scaled scores themselves, and no max."""
    program = run_demo(monkeypatch, name, [])["program"]
    assert len(program.stabilize_requests) == 1
    ops = {r.out: r for r in program.ops}
    (exp,) = [r for r in program.ops if r.kind == "exp"]
    assert exp.attr("stabilize") == 2  # the keys: the last of three axes
    scores = ops[exp.args[0]]
    # the exp's operand is the division by sqrt(d), not an add of a constant behind it
    assert scores.kind == "div" and isinstance(scores.args[1], float)
    assert not [r for r in program.ops if r.kind in ("reduce_max", "sub")]
    # an add in the program adds two tensors (MLA's two score terms) or qknorm's eps to
    # a mean of squares; none of them feeds the exp
    for add in (r for r in program.ops if r.kind == "add"):
        assert add.out != exp.args[0]
        assert all(isinstance(a, int) for a in add.args) or name == "qknorm"


def test_no_demo_passes_a_shift():
    """The parameter is gone, and so is every mention of it in a demo's source."""
    for path in sorted(DEMO.glob("*.py")) + sorted((DEMO / "dynamic").glob("*.py")):
        assert "shift=" not in path.read_text(), path.name
    assert not (DEMO / "mha_decode_exact.py").exists()  # merged into mha_decode.py


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
