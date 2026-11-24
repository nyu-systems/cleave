"""A validation tolerance declared by boundary role (CPU).

One ``Tolerance`` is the bound of the outputs a program or its cut defines. A rewrite can
add a tensor to the cut: the split partial of an exact softmax returns its shift, a float32
max, next to its sums. Such an output carries a boundary role, and its bound is
declared under that role: ``{"": Tolerance(...), "shift": Tolerance(...)}``. Nothing falls
back to another output's bound, in either direction.
"""

import multiprocessing as mp
from types import SimpleNamespace

import pytest

from cleave.core.errors import SpecError, ValidationError
from cleave.ir.graph import Boundary
from cleave.pipeline.report import bounds_of, format_measured, tolerance_dicts, worst_measured
from cleave.runtime import evaluate as evaluate_mod
from cleave.runtime.evaluate import run_candidates
from cleave.runtime.reference import output_roles
from cleave.runtime.validate import MeasuredError, Tolerance, tolerances_of
from cleave.runtime.workers import WorkerPool

OUT, SHIFT = Tolerance(rtol=5e-3), Tolerance(rtol=2e-6)
GIB = 2**30
BY_ROLE = {"": OUT, "shift": SHIFT}
# A split partial returns its max first (lower.state_split puts the shift ahead of the sums)
SOLO, PARTIAL = ("",), ("shift", "", "")


def test_each_output_takes_the_bound_of_its_role():
    assert tolerances_of(BY_ROLE, PARTIAL) == (SHIFT, OUT, OUT)
    assert tolerances_of(BY_ROLE, SOLO) == (OUT,)
    # the order is the kernel's, whatever it is
    assert tolerances_of(BY_ROLE, ("", "", "shift")) == (OUT, OUT, SHIFT)
    # an entry no output of this kernel has is not this kernel's business
    assert tolerances_of({**BY_ROLE, "other": SHIFT}, SOLO) == (OUT,)


def test_one_tolerance_covers_the_outputs_without_a_role_only():
    assert tolerances_of(OUT, SOLO) is OUT and tolerances_of(OUT, ("", "")) is OUT
    with pytest.raises(SpecError) as e:
        tolerances_of(OUT, PARTIAL)
    assert str(e.value) == (
        "one Tolerance covers the outputs a program or its cut defines, and this kernel also "
        "returns outputs of role ['shift'] (a split partial returns its float32 shift): "
        "declare the tolerance by role, tolerance={'': Tolerance(...), 'shift': Tolerance(...)}")


def test_a_role_without_an_entry_has_no_bound_to_fall_back_to():
    with pytest.raises(SpecError, match=r"^no validation tolerance declared for the outputs of "
                                        r"role \['shift'\] \(the declaration has \[''\]\); "
                                        "there is no default bound$"):
        tolerances_of({"": OUT}, PARTIAL)
    # ... nor do the program's own outputs fall back to the shift's
    with pytest.raises(SpecError, match=r"outputs of role \[''\] \(the declaration has "
                                        r"\['shift'\]\)"):
        tolerances_of({"shift": SHIFT}, PARTIAL)
    with pytest.raises(SpecError, match=r"outputs of role \[''\]"):
        tolerances_of({"shift": SHIFT}, SOLO)


@pytest.mark.parametrize("declared", [{"": 5e-3}, {"": (OUT, OUT)}, {0: OUT}, {"": None}])
def test_a_declaration_by_role_maps_role_names_to_single_tolerances(declared):
    with pytest.raises(SpecError, match="maps each boundary role"):
        tolerances_of(declared, SOLO)


def test_other_declarations_pass_through():
    """One per output is positional and is checked against the kernel's outputs where they
    are compared; no declaration is the comparator's error to raise."""
    per_output = (SHIFT, OUT, OUT)
    assert tolerances_of(per_output, PARTIAL) is per_output
    assert tolerances_of([OUT], PARTIAL) == [OUT]
    assert tolerances_of(None, PARTIAL) is None
    # a kernel without a graph: its outputs are not known
    assert tolerances_of(OUT, None) is OUT and tolerances_of(per_output, None) is per_output
    with pytest.raises(SpecError, match="needs the roles of the kernel's outputs"):
        tolerances_of(BY_ROLE, None)


def test_output_roles_reads_the_boundary_records():
    from cleave.ir.graph import Graph, OutputNode
    from cleave.ir.graph_utils import find_topo_sort
    from cleave.ir.ops import ReduceMax, ReduceSum

    a, b, c = ReduceSum((4, 8), name="a"), ReduceMax((4, 8), name="b"), ReduceSum((4, 8), name="c")
    outs = [OutputNode(a, dtype="float16"), OutputNode(b, dtype="float32"),
            OutputNode(c, dtype="float16")]
    outs[1].name, outs[2].name = "OutputNodf", "OutputNodg"
    outs[0].boundary, outs[1].boundary = Boundary(0), Boundary(1, "shift")  # the third: none
    graph = Graph(nodes=find_topo_sort(outs))
    assert output_roles(graph) == ("", "shift", "")


# ------------------------------------------------------------------------- run_candidates

def _check(code_str, *, out_idx, reference_cache_dir, tolerance=None, benchmark=None,
           roles=None):
    """Worker-side fake: reports the declaration and the roles it was handed."""
    return repr((tolerance, roles)), 0.0


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="needs fork")
def test_every_candidate_is_checked_at_the_bounds_of_its_own_outputs(monkeypatch):
    """One declaration for the plans of a program; each kernel gets what it comes to for
    the outputs it returns. A kernel left without a bound fails at validate, with the
    reason, and the others are not affected.

    The reference of a graph whose kernels have no bound is not built either (it used to
    be, for candidates that then all failed), and such a graph has no say in how many
    workers the run gets: that follows the largest reference that was built. This is what
    a caller of ``run_candidates`` gets; ``pipeline.build`` refuses a declaration that
    leaves a plan without a bound before it starts (``tests/pipeline_api/test_config.py``)."""
    roles = {"solo": SOLO, "partial": PARTIAL, "combine": SOLO}
    footprint = {"solo": 1 * GIB, "partial": 3 * GIB, "combine": 2 * GIB}
    monkeypatch.setattr(evaluate_mod, "output_roles", lambda graph: roles[graph.name])
    cands = [SimpleNamespace(code=name, rstep_map={}, output_tile=(i,),
                             flat_graph=SimpleNamespace(name=name, nodes=[]))
             for i, name in enumerate(roles)]

    built, sized_for = [], []

    def reference(cache_dir, graph, dtype):
        built.append(graph.name)
        return [-1], footprint[graph.name]

    def cap(max_parallel, io_bytes):
        sized_for.append(io_bytes)
        return 1

    monkeypatch.setattr(evaluate_mod, "_effective_compile_and_run_parallelism", cap)

    def run(tolerance):
        built.clear()
        sized_for.clear()
        with WorkerPool(mp.get_context("fork")) as pool:
            return run_candidates(cands, "float16", build_fn=lambda c: c.code, check_fn=_check,
                                  reference_fn=reference, pool=pool,
                                  max_parallel=1, tolerance=tolerance)

    solo, partial, combine = run(BY_ROLE)
    assert (solo.ok, partial.ok, combine.ok) == (True, True, True)
    assert built == ["solo", "partial", "combine"]
    assert sized_for == [footprint["partial"]]  # the largest, not the first
    # the roles go with the task (for the validation log) only where an output has one
    assert solo.cuda_src == combine.cuda_src == repr(((OUT,), None))
    assert partial.cuda_src == repr(((SHIFT, OUT, OUT), PARTIAL))

    solo, partial, combine = run(OUT)
    assert (solo.ok, partial.ok, combine.ok) == (True, False, True)
    assert solo.cuda_src == repr((OUT, None))
    assert partial.stage == "validate" and isinstance(partial.error, ValidationError)
    assert "returns outputs of role ['shift']" in str(partial.error)
    assert partial.code == "partial"  # it is still built: the failure names its source
    assert built == ["solo", "combine"]  # ... and no reference was made for it
    assert sized_for == [footprint["combine"]]  # the largest of those that were

    solo, partial, combine = run({"": OUT})
    assert (solo.ok, partial.ok, combine.ok) == (True, False, True)
    assert "no validation tolerance declared for the outputs of role ['shift']" in str(
        partial.error)
    assert built == ["solo", "combine"]
    assert sized_for == [footprint["combine"]]

    # a run whose first graph is the one without a bound is sized like any other (the
    # first candidate's graph used to size it: 0 bytes here, which switched the cap off)
    cands.insert(0, cands.pop(1))
    partial, solo, combine = run(OUT)
    assert (partial.ok, solo.ok, combine.ok) == (False, True, True)
    assert built == ["solo", "combine"]
    assert sized_for == [footprint["combine"]]


# ------------------------------------------------------------------------------------ log

def test_the_validation_log_names_the_role_of_an_output(tmp_path, monkeypatch):
    """The measuring tool tells a partial's shift from its sums by the role in the record,
    not by the place the kernel returns it in."""
    import json

    import torch

    from cleave.runtime.validate import compare_outputs

    log = tmp_path / "validation.jsonl"
    monkeypatch.setenv("CLEAVE_VALIDATION_LOG", str(log))
    shift = torch.randn(4, 8)
    sums = torch.rand(4, 8).to(torch.float16)
    compare_outputs([shift, sums], [shift, sums], (SHIFT, OUT), what="a partial",
                    roles=("shift", ""))
    compare_outputs([sums], [sums], OUT, what="a combine")
    partial, combine = [json.loads(line) for line in log.read_text().splitlines()]
    assert [o.get("role") for o in partial["outputs"]] == ["shift", None]
    assert [o["dtype"] for o in partial["outputs"]] == ["float32", "float16"]
    assert [o["rtol"] for o in partial["outputs"]] == [SHIFT.rtol, OUT.rtol]
    assert "role" not in combine["outputs"][0]


# --------------------------------------------------------------------------------- report

def test_the_report_names_the_bound_of_each_output():
    """mha_decode's mix: solo kernels (one output), split partials (their float32 max first,
    role "shift", then two sums) and combines (one output). By position the partial's shift
    was every solo kernel's output 0: one row, no bound beside it, and the shift's own error
    (and its margin to its 5e-6) nowhere. By role the shift has its row and its bound, and
    the outputs without a role are numbered among themselves, as the tolerance measurement
    numbers them."""
    solo = (MeasuredError(9.77e-4, 2.0e-3),)
    partial = (MeasuredError(9.72e-7, 3.1e-5), MeasuredError(9.32e-4, 1.5e-3),
               MeasuredError(7.82e-5, 1.0e-3))
    combine = (MeasuredError(8.18e-4, 1.2e-3),)
    measured = [solo, partial, combine, partial]
    roles = [SOLO, PARTIAL, SOLO, PARTIAL]
    worst = worst_measured(measured, roles)
    # out0: the solo and combine outputs and the partial's first sum; out1: its second sum
    assert worst == [{"rel": 9.77e-4, "abs": 2.0e-3}, {"rel": 7.82e-5, "abs": 1.0e-3},
                     {"rel": 9.72e-7, "abs": 3.1e-5, "role": "shift"}]
    assert bounds_of(worst, BY_ROLE) == [OUT, OUT, SHIFT]
    assert format_measured(worst, BY_ROLE) == (
        "out0 rel 9.8e-04 of rtol 0.005; out1 rel 7.8e-05 of rtol 0.005; "
        "shift rel 9.7e-07 of rtol 2e-06")
    # by position (no roles) the shift is folded into output 0 and its error is gone
    assert worst_measured(measured) == [{"rel": 9.77e-4, "abs": 2.0e-3},
                                        {"rel": 9.32e-4, "abs": 1.5e-3},
                                        {"rel": 7.82e-5, "abs": 1.0e-3}]
    assert tolerance_dicts(BY_ROLE) == {"": {"rtol": 5e-3, "atol": 0.0},
                                        "shift": {"rtol": 2e-6, "atol": 0.0}}
    # other declarations are what they were
    assert bounds_of(worst, OUT) is OUT
    assert format_measured(worst[:1], OUT) == "rel 9.8e-04 of rtol 0.005"
    assert format_measured(worst[:1]) == "rel 9.8e-04 abs 2.0e-03"
    assert format_measured([dict(rel=m.rel, abs=m.abs) for m in (MeasuredError(1e-3, 2e-3),)],
                           OUT) == "rel 1.0e-03 of rtol 0.005"


def test_a_report_of_solo_partial_and_combine_kernels_gives_the_shift_its_row(monkeypatch):
    """``Report.measured_error`` and its validate line, on mha_decode's mix: the roles come
    from each plan's flat graph, the shift gets its row and its own bound, and ``to_dict``
    labels the row with the role the ``tolerance`` entry is keyed by."""
    import json

    from cleave.pipeline.report import Report, Result
    from cleave.runtime import reference
    from cleave.runtime.evaluate import CandidateResult

    roles = {"solo": SOLO, "partial": PARTIAL, "combine": SOLO}
    monkeypatch.setattr(reference, "output_roles", lambda graph: roles[graph.name])

    def result(kind, *measured):
        plan = SimpleNamespace(label=kind, variant="solo", role=kind,
                               tile=SimpleNamespace(flat_graph=SimpleNamespace(name=kind)))
        return Result(plan, CandidateResult(None, "ok", None, None, latency_ms=None,
                                            measured=tuple(MeasuredError(*m) for m in measured)))

    report = Report([result("solo", (9.77e-4, 2.0e-3)),
                     result("partial", (9.72e-7, 3.1e-5), (9.32e-4, 1.5e-3), (7.82e-5, 1e-3)),
                     result("combine", (8.18e-4, 1.2e-3))], tolerance=BY_ROLE)
    assert report.measured_error() == [{"rel": 9.77e-4, "abs": 2.0e-3},
                                       {"rel": 7.82e-5, "abs": 1e-3},
                                       {"rel": 9.72e-7, "abs": 3.1e-5, "role": "shift"}]
    validate = dict((stage, text) for stage, text, _ in report.stage_lines())["validate"]
    assert validate.endswith("max error out0 rel 9.8e-04 of rtol 0.005; out1 rel 7.8e-05 of "
                             "rtol 0.005; shift rel 9.7e-07 of rtol 2e-06"), validate
    validation = json.loads(json.dumps(report.to_dict()))["validation"]
    assert validation["max_error"][-1] == {"rel": 9.72e-7, "abs": 3.1e-5, "role": "shift"}
    assert set(validation["tolerance"]) == {"", "shift"}