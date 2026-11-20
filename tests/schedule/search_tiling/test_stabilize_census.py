"""The census tool reports a job ``ok`` only for checks that ran (CPU).

``tests/_stabilize_census.py`` runs the checks unit tests cannot afford over whole candidate
lists. A tool like that is only worth its green line if it cannot be green for having had
nothing to look at: a graph without a record, a graph refused at lowering, a job none of
whose candidates is accepted, a device compile that found no kernel to compile. Each of
those ends ``NOT CHECKED`` (exit status 2), a failed check ends ``FAILED`` (1).
"""

import json

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")


from tests import _softmax as S
from tests import _stabilize_census as tool

ROWS = (S.softmax_rows, dict(R=64, C=1024),
        dict(rolling="div reads the exp next to the finished reduction"))


def _run(tmp_path, capsys, job, *flags):
    status = tool.main(["census", str(tmp_path), job, "--lower", "0", *flags])
    lines = capsys.readouterr().out.rstrip().splitlines()
    report = json.loads((tmp_path / f"{job.replace(':', '_')}.json").read_text())
    return status, lines, report


def _job(monkeypatch, graphs, nvcc=None):
    monkeypatch.setitem(tool.FIXTURES, "job", (graphs, {}, None, nvcc))


def test_a_job_whose_checks_ran_is_ok(monkeypatch, tmp_path, capsys):
    _job(monkeypatch, lambda: [S.softmax_ir(*ROWS)])
    status, lines, report = _run(tmp_path, capsys, "job")
    assert status == 0 and lines[-1] == "job: ok"
    assert lines[0].startswith("job: 35 raw candidates, 4 accepted, 4 build")
    assert not report["problems"] and not report["unchecked"]
    assert sum(r["accepted"] for r in report["rows"]) == 4


def test_a_failed_check_is_exit_status_1(monkeypatch, tmp_path, capsys):
    """A filter that keeps what the build refuses: check (a), accepted == builds."""

    def keeps_everything(ir):
        return lambda cand: True

    _job(monkeypatch, lambda: [S.softmax_ir(*ROWS)])
    monkeypatch.setattr(tool, "stabilize_accept", keeps_everything)
    status, lines, report = _run(tmp_path, capsys, "job")
    assert status == 1 and lines[-1] == "job: FAILED, 31 problem(s)"
    assert all(p.startswith("(a) ") for p in report["problems"])
    assert sum(line.lstrip().startswith("PROBLEM") for line in lines) == 20  # capped print


def test_a_graph_without_a_record_is_not_checked(monkeypatch, tmp_path, capsys):
    def unmarked():
        ir = S.lowered(ROWS[0], ROWS[1])
        for exp in S.exps(ir):
            exp.stabilize = None
        return [ir]

    _job(monkeypatch, unmarked)
    status, lines, report = _run(tmp_path, capsys, "job")
    assert status == 2 and lines[-1] == "job: NOT CHECKED, 2 item(s)"
    assert report["unchecked"] == ["graph job: no record on any exp",
                                   "no candidate of any graph was censused"]
    assert not report["rows"]


def test_a_job_that_accepts_no_candidate_is_not_checked(monkeypatch, tmp_path, capsys):
    """A row too long for one block and no rolling form: every tiling is refused, so
    lowering parity, device compile and staging had nothing to look at."""
    _job(monkeypatch, lambda: [S.softmax_ir(ROWS[0], dict(R=64, C=16384), ROWS[2])])
    status, lines, report = _run(tmp_path, capsys, "job")
    assert status == 2 and lines[-1] == "job: NOT CHECKED, 1 item(s)"
    assert report["rows"] and not any(r["accepted"] for r in report["rows"])
    assert not report["problems"]
    (item,) = report["unchecked"]
    assert item.startswith(f"none of the {len(report['rows'])} candidates was accepted")


def test_nvcc_without_the_named_kernel_is_not_checked(monkeypatch, tmp_path, capsys):
    """The job names a tiling the filter refuses (a single-row tile): nothing reaches the
    device compile, and that is not a pass."""
    _job(monkeypatch, lambda: [S.softmax_ir(*ROWS)],
         nvcc=lambda c: list(c.output_tile) == [1, 1024])
    status, lines, report = _run(tmp_path, capsys, "job", "--nvcc")
    assert status == 2 and lines[-1] == "job: NOT CHECKED, 1 item(s)"
    (item,) = report["unchecked"]
    assert item.startswith("(d) --nvcc: the kernel this job names is not among the accepted")
    assert not any("nvcc" in r for r in report["rows"])
    # without --nvcc the same job has nothing left unchecked
    assert _run(tmp_path, capsys, "job")[0] == 0


def test_nvcc_on_a_job_that_names_no_kernel_says_so(monkeypatch, tmp_path, capsys):
    _job(monkeypatch, lambda: [S.softmax_ir(*ROWS)])
    status, lines, _ = _run(tmp_path, capsys, "job", "--nvcc")
    assert status == 0
    assert any("this job names no kernel" in line for line in lines)


@pytest.mark.parametrize(
    ("graphs", "notes", "last"),
    [
        ([], ["graph g0: refused at lowering: LoweringRejected: nope"],
         "demo:mha_prefill: NOT CHECKED, 2 item(s)"),
        ([], [], "demo:mha_prefill: NOT CHECKED, 1 item(s)"),
    ],
    ids=["every graph refused at lowering", "no graph at all"],
)
def test_a_demo_job_without_graphs_is_not_checked(monkeypatch, tmp_path, capsys, graphs,
                                                  notes, last):
    monkeypatch.setattr(tool, "_demo_graphs", lambda name: (graphs, {}, notes))
    status, lines, report = _run(tmp_path, capsys, "demo:mha_prefill")
    assert status == 2 and lines[-1] == last
    assert report["unchecked"] == notes + ["no candidate of any graph was censused"]


def test_one_refused_graph_keeps_a_demo_job_from_ok(monkeypatch, tmp_path, capsys):
    """The other graph is censused and clean; the job still does not end ``ok``."""
    note = "graph g1: refused at lowering: LoweringRejected: nope"
    monkeypatch.setattr(
        tool, "_demo_graphs",
        lambda name: ([("g0", S.softmax_ir(*ROWS), "by hand")], {}, [note]),
    )
    status, lines, report = _run(tmp_path, capsys, "demo:mha_prefill")
    assert status == 2 and lines[-1] == "demo:mha_prefill: NOT CHECKED, 1 item(s)"
    assert report["unchecked"] == [note] and not report["problems"]
    assert lines[0].startswith("demo:mha_prefill[g0]: 35 raw candidates, 4 accepted, 4 build")
    assert "    NOT CHECKED " + note in lines


def test_a_demo_job_censuses_the_searched_graphs(monkeypatch, tmp_path, capsys):
    """The real thing, on the smallest demo: its program traced with the softmax the
    frontend will record, searched, lowered, and every tiling of both searched graphs
    censused. Flash form: every tiling is served; traced order: only the ones that hold
    the row whole."""
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path / "cache"))
    status, lines, report = _run(tmp_path, capsys, "demo:mha_decode_exact")
    assert status == 0 and lines[-1] == "demo:mha_decode_exact: ok", lines[-6:]
    assert not report["unchecked"] and not report["problems"]

    by_graph = {}
    for row in report["rows"]:
        by_graph.setdefault(row["graph"], []).append(row)
    assert len(by_graph) == 2
    counts = sorted(
        (len(rows), sum(r["accepted"] for r in rows),
         sum(r["tiling"] == "rolling" for r in rows))
        for rows in by_graph.values()
    )
    assert counts == [(32, 8, 0), (32, 32, 24)]
    # the search saw no max: the records are lowering's, or set by hand where this tree's
    # frontend cannot make the request
    heads = [line for line in lines if line.startswith("demo:mha_decode_exact[")]
    assert len(heads) == 2 and all(
        line.endswith(("[records: lowering]", "[records: by hand]")) for line in heads
    )


def test_importing_the_tool_leaves_the_process_alone():
    """The tool is importable (this module does it): the environment and ``sys.path`` are
    only touched when it runs as a script, and a demo job puts back what it borrowed."""
    import sys

    import cleave.examples as examples
    import cleave.lang as cl
    from cleave.frontend import paged
    from cleave.pipeline import compile as real_compile

    assert str(tool.ROOT / "demo") not in sys.path
    assert cl.softmax is paged.softmax and examples._compile is real_compile
    source = (tool.ROOT / "tests" / "_stabilize_census.py").read_text()
    head, guard, rest = source.partition('if __name__ == "__main__":  # run as a script')
    assert guard and "os.environ" not in head
    assert rest.count("os.environ") == 1 and rest.index("os.environ") < rest.index("\n\n")


@pytest.mark.parametrize(
    ("fn", "shapes", "obstacle"),
    [
        (S.flash_attention, dict(H=32, SQ=64, SKV=256, D=64), None),
        (S.flash_rows, dict(R=64, C=4096, D=64), None),
        (S.traced_attention, dict(H=32, SQ=64, SKV=256, D=64), "div"),
        (S.softmax_rows, dict(R=64, C=1024), "div"),
        (S.sum_next_to_exp, dict(R=256, C=4096), "add"),
    ],
    ids=["flash attention", "flash rows", "traced attention", "softmax", "sum next to exp"],
)
def test_record_set_by_hand_on_a_searched_graph(fn, shapes, obstacle):
    """``hand_record`` reads off a graph what the tests of the written-out programs spell
    out: the last axis, the reductions that sum the exp, and whether the exp is read next
    to one of them. Where lowering wrote a record itself, the two agree."""
    ir = S.lowered(fn, shapes)
    (exp,) = S.exps(ir)
    record = S.hand_record(exp)
    assert record.axes == (-1,) and record.export is None
    assert record.scaled == S.scaled_reductions(exp) and record.scaled
    if obstacle is None:
        assert record.rolling == ""
    else:
        reader, _, finished = record.rolling.partition(
            " reads the exp next to the finished reduction ")
        assert reader.startswith(obstacle) and finished in dict(record.scaled)
    if exp.stabilize is not None:  # written by the scale analysis
        assert exp.stabilize.axes == record.axes
        assert dict(exp.stabilize.scaled) == dict(record.scaled)
        assert bool(exp.stabilize.rolling) == bool(record.rolling)


@pytest.mark.parametrize(
    ("fn", "shapes"),
    [
        (S.column_weighted, dict(M=128, K=4096, N=256)),
        (S.column_average, dict(H=4, K=4096, N=256)),
    ],
    ids=["column weighted", "column average"],
)
def test_no_record_by_hand_off_the_last_axis(fn, shapes):
    """The by-hand rule only knows a softmax over the last axis; anywhere else it gives no
    record, and the census reports the graph as not checked."""
    (exp,) = S.exps(S.lowered(fn, shapes))
    assert S.hand_record(exp) is None
