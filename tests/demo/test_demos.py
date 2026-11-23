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
from cleave.core.errors import SpecError
from cleave.runtime.validate import Tolerance

from .._measure_tolerances import MEASURED, expected_declaration, expected_tolerances_by_role
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
    ("mha_backward", "mha_backward", []),
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
# The demos that declare one bound per boundary role: an exact softmax split over the keys
# (by default for the decode demos, with ``--split SKV`` for the MHA / GQA / MQA prefill
# demos), whose partials return their float32 max. That split is built from the loop state
# of the solo graph (``pipeline.plans.state_pair``). mla_decode,
# mla_prefill and qknorm offer no split: mla_decode asks for none, the split of mla_prefill
# would need about 190 GB of references, and qknorm declares a single bound.
BY_ROLE = set(ATTENTION) - {"mla_decode", "mla_prefill", "qknorm"}


def test_the_readme_says_which_attention_demos_have_no_split_run():
    """Every demo parses ``--split``, and the README lists it among the flags the demos
    share. Three attention demos have no split run; the README names them where it lists
    the flag."""
    readme = (DEMO.parent / "README.md").read_text()
    (paragraph,) = [p for p in readme.split("\n\n") if p.startswith("`--split` is parsed by")]
    assert sorted(set(ATTENTION) - BY_ROLE) == ["mla_decode", "mla_prefill", "qknorm"]
    for name in ATTENTION:
        assert (f"`{name}.py`" in paragraph) == (name not in BY_ROLE), name


def _assert_declared(config, name):
    """One measured bound for the operator (or one per output), within the ceiling (one per
    boundary role for a demo of ``BY_ROLE``); it asks for normal inputs."""
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
        expected = expected_declaration(tolerance, measured)
    per_output = (tolerance,) if isinstance(tolerance, Tolerance) else tolerance
    assert isinstance(per_output, tuple) and per_output, "the demo declares no tolerance"
    for t in per_output:
        assert isinstance(t, Tolerance) and 0 < t.rtol <= CEILING and t.atol == 0
    # ... and exactly the one(s) the rule gives for the pinned measurement of this demo
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
    """The decode demo: the max of ``cl.softmax`` in the program, the split over the keys,
    and the search budget the auto rule gives (15 with the split; the search itself runs
    unsplit, at 13, and the pair is built from the loop state:
    ``tests/pipeline_api/test_softmax_split.py``)."""
    seen = run_demo(monkeypatch, "mha_decode", [])
    program, so, sc = seen["program"], seen["so"], seen["sc"]
    assert [r.kind for r in program.ops if r.kind != "input"] == [
        "matmul", "div", "reduce_max", "sub", "exp", "reduce_sum", "div", "matmul"]
    (m,) = [r for r in program.ops if r.kind == "reduce_max"]
    assert m.attr("dim") == 2  # the keys
    assert so.split == ("SKV",) and sc.split_k == (16, 32)
    assert so.max_ops == "auto" and so.resolve_max_ops(program) == 15
    assert program.sizes == {"H": 32, "SQ": 64, "D": 64, "SKV": 4096}
    # --split none: the solo kernels only, on the same budget
    solo = run_demo(monkeypatch, "mha_decode", ["--split", "none"])
    assert not solo["so"].split and not solo["sc"].split_k
    assert solo["so"].resolve_max_ops(solo["program"]) == 13


@pytest.mark.parametrize("name", ATTENTION)
def test_attention_demo_softmax_is_exact(monkeypatch, name):
    """No attention demo shifts its scores by a constant: each records the one max of
    ``cl.softmax``, over the keys, and its exp reads the scaled scores minus that max."""
    program = run_demo(monkeypatch, name, [])["program"]
    ops = {r.out: r for r in program.ops}
    (m,) = [r for r in program.ops if r.kind == "reduce_max"]
    assert m.attr("dim") == 2  # the keys: the last of three axes
    (exp,) = [r for r in program.ops if r.kind == "exp"]
    (sub,) = [r for r in program.ops if r.kind == "sub"]
    assert exp.args == (sub.out,) and sub.args == (m.args[0], m.out)  # exp(s - max(s))
    scores = ops[m.args[0]]
    # the softmax's logits are the division by sqrt(d), not an add of a constant behind it
    assert scores.kind == "div" and isinstance(scores.args[1], float)
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


def _run_to_exit(monkeypatch, name, argv):
    """Run the demo as ``__main__`` to the ``SystemExit`` it ends on; its message."""
    monkeypatch.setattr(examples, "_target", lambda: "cpu")
    monkeypatch.setattr(sys, "argv", [str(DEMO / f"{name}.py"), *argv])
    monkeypatch.syspath_prepend(str(DEMO))
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(DEMO / f"{name}.py"), run_name="__main__")
    return str(e.value.code)


def test_a_declaration_that_does_not_fit_ends_a_demo_in_one_line(monkeypatch):
    """A flag can ask for plans the demo's declaration does not cover: ``--split`` on a
    demo that declares one ``Tolerance``, whose split partials return a shift. That is a
    ``SpecError`` of the pipeline, and the demo ends on its message, behind the demo's name:
    it says what to declare, and a traceback through the pipeline says nothing more.
    ``-v`` keeps the traceback."""
    reason = ("build(): plan 'g4 split-k=4 partial #0' (partial): one Tolerance covers the "
              "outputs a program or its cut defines, and this kernel also returns outputs of "
              "role ['shift']")

    def refuse(program, **kw):
        raise SpecError(reason)

    monkeypatch.setattr(examples, "_compile", refuse)
    args = ["--split", "SKV", "--max-ops", "13"]
    assert _run_to_exit(monkeypatch, "mla_prefill", args) == f"mla_prefill: {reason}"
    monkeypatch.setattr(sys, "argv", [str(DEMO / "mla_prefill.py"), *args, "-v"])
    with pytest.raises(SpecError, match="one Tolerance covers"):
        runpy.run_path(str(DEMO / "mla_prefill.py"), run_name="__main__")

    # any other error without a report is not the demo's to reword
    def broken(program, **kw):
        raise examples.CleaveError("something else")

    monkeypatch.setattr(examples, "_compile", broken)
    monkeypatch.setattr(sys, "argv", [str(DEMO / "mla_prefill.py")])
    with pytest.raises(examples.CleaveError, match="something else"):
        runpy.run_path(str(DEMO / "mla_prefill.py"), run_name="__main__")


def test_qknorm_says_which_inputs_its_kernels_are_validated_on():
    """The adversarial softmax cases shape inputs. qknorm's logits are computed from the
    normalized q and k, so its kernels are held to the normal inputs only (the build's
    report gives that reason): the demo and the README say so and do not claim the
    adversarial cases for it."""
    from cleave.lower import to_ir
    from cleave.pipeline.build import _reference_fn

    doc = " ".join((DEMO / "qknorm.py").read_text().split('"""')[1].split())
    assert "this demo's kernels are validated on normal inputs" in doc
    readme = " ".join((DEMO.parent / "README.md").read_text().split())
    assert ("wherever the logits can be shaped through an input: every attention demo but "
            "`qknorm.py`") in readme
    # ... which is what the build does with the demo's program
    sys.path.insert(0, str(DEMO))
    try:
        qknorm = runpy.run_path(str(DEMO / "qknorm.py"), run_name="_demo")["qknorm"]
    finally:
        sys.path.remove(str(DEMO))
    program = qknorm.trace(HQ=4, HKV=2, SQ=16, SKV=64, D=32)
    held = []
    reference = _reference_fn(("normal",),
                              lambda cache_dir, graph, dtype, distributions:
                              held.append(distributions) or "published")
    graph = to_ir(program.to_graph_structure(), dtype="float16", out_dtype="float16")
    assert reference("dir", graph, "float16") == "published"
    assert held == [("normal",)]
    cases, note = reference.held[id(graph)]
    assert tuple(cases) == ("normal",) and "not read from an input" in note


def test_demo_cli_overrides(monkeypatch):
    seen = run_demo(monkeypatch, "mha_decode", ["--SKV", "2048", "--preset", "fast", "--topk", "20",
                                                "--split", "SKV:8", "--max-ops", "15"])
    assert seen["program"].sizes["SKV"] == 2048
    assert (seen["sc"].topk, seen["sc"].split_k, seen["so"].split, seen["so"].max_ops) == (20, (8,), ("SKV",), 15)
    assert seen["ev"].iters == 100  # the fast preset
