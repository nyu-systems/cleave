"""The streaming census tool reports a job ``ok`` only for checks that ran (CPU).

``tests/_streaming_census.py`` runs the checks unit tests cannot afford over whole candidate
lists: accepted == builds, typed refusals only, lowering parity with the plain twin, and,
for a searched program, the number of plans against the hand-derived pass. A tool like that
is only worth its green line if it cannot be green for having had nothing to look at: a
graph without an obstacle, a graph refused at lowering, a job none of whose candidates is
accepted, a device compile that found no kernel to compile. Each of those ends
``NOT CHECKED`` (exit status 2), a failed check ends ``FAILED`` (1).
"""

import json
from collections import Counter

import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from tests import _explicit_softmax as X
from tests import _streaming_census as tool

ROWS = (X.softmax_rows, dict(R=64, C=1024), "is also read by")
FLASH = (X.flash_attention, dict(H=32, SQ=64, SKV=256, D=64))


def _run(tmp_path, capsys, job, *flags):
    status = tool.main(["census", str(tmp_path), job, "--lower", "0", *flags])
    lines = capsys.readouterr().out.rstrip().splitlines()
    report = json.loads((tmp_path / f"{job.replace(':', '_')}.json").read_text())
    return status, lines, report


def _job(monkeypatch, graphs, nvcc=None):
    monkeypatch.setitem(tool.FIXTURES, "job", (graphs, {}, None, nvcc))


def test_a_job_whose_checks_ran_is_ok(monkeypatch, tmp_path, capsys):
    _job(monkeypatch, lambda: [X.streamed_ir(*FLASH)])
    status, lines, report = _run(tmp_path, capsys, "job")
    assert status == 0 and lines[-1] == "job: ok"
    assert lines[0].startswith("job: 32 raw candidates, 32 accepted, 32 build")
    assert not report["problems"] and not report["unchecked"]
    assert sorted({r["tiling"] for r in report["rows"]}) == ["looped", "one tile"]


def test_a_refused_obstacle_is_censused_too(monkeypatch, tmp_path, capsys):
    """A returned softmax: its obstacle is refused, and the filter keeps what builds."""
    _job(monkeypatch, lambda: [X.streamed_ir(*ROWS)])
    status, lines, report = _run(tmp_path, capsys, "job")
    assert status == 0 and lines[-1] == "job: ok"
    assert lines[0].startswith("job: 25 raw candidates, 20 accepted, 20 build")
    assert {r["tiling"] for r in report["rows"] if r["accepted"]} == {"one tile"}


def test_a_failed_check_is_exit_status_1(monkeypatch, tmp_path, capsys):
    """A filter that keeps what the build refuses: check (a), accepted == builds."""

    def keeps_everything(ir):
        return lambda cand: True

    _job(monkeypatch, lambda: [X.streamed_ir(*ROWS)])
    monkeypatch.setattr(tool, "streaming_accept", keeps_everything)
    status, lines, report = _run(tmp_path, capsys, "job")
    assert status == 1 and lines[-1] == "job: FAILED, 5 problem(s)"
    assert all(p.startswith("(a) ") for p in report["problems"])


def test_a_lowering_that_differs_from_the_plain_twin_is_a_problem(monkeypatch, tmp_path,
                                                                  capsys):
    """Check (c): the assembled kernel lowers where its plain twin lowers."""
    _job(monkeypatch, lambda: [X.streamed_ir(*FLASH)])
    real = tool._lower

    def lowers(code, device_compile=False):
        return "Error: no" if "ukernel_rescale" in code else real(code, device_compile)

    monkeypatch.setattr(tool, "_lower", lowers)
    status = tool.main(["census", str(tmp_path), "job", "--lower", "2"])
    lines = capsys.readouterr().out.rstrip().splitlines()
    assert status == 1 and lines[-1].startswith("job: FAILED")
    assert any("(c) " in line and "assembled Error: no | plain ok" in line for line in lines)


def test_the_plain_twin_is_the_same_loop_without_the_carried_state():
    """What check (c) holds the assembled kernel against: ``apply_tiling`` builds its loop,
    the record comes off the tiled copy instead of being assembled; the flat graph keeps
    it, and the assembly is back when the ``with`` ends."""
    from cleave.runtime.compile import build_candidate_code
    from cleave.schedule import search_tiling, streaming
    from cleave.schedule.tiling import schedule_kernel

    ir = X.streamed_ir(*FLASH)
    cand = next(c for c in search_tiling(ir, topk=1000) if streaming.sites(schedule_kernel(c)))
    assembled = build_candidate_code(cand)
    real = streaming.assemble
    with tool._Plain():
        plain = build_candidate_code(cand)
    assert streaming.assemble is real and streaming.recorded(ir)
    m = X.maxes(ir)[0].name
    assert f"t_{m}_prev" in assembled and "ukernel_rescale" in assembled
    assert f"t_{m}_prev" not in plain and "ukernel_rescale" not in plain
    assert "ukernel_reduce_max_axis2_rr(" in plain  # the max still accumulates in the loop


def test_a_graph_without_an_obstacle_is_not_checked(monkeypatch, tmp_path, capsys):
    _job(monkeypatch, lambda: [X.lowered(*FLASH)])
    status, lines, report = _run(tmp_path, capsys, "job")
    assert status == 2 and lines[-1] == "job: NOT CHECKED, 2 item(s)"
    assert report["unchecked"] == ["graph job: no streaming obstacle",
                                   "no candidate of any graph was censused"]
    assert not report["rows"]


def test_a_job_that_accepts_no_candidate_is_not_checked(monkeypatch, tmp_path, capsys):
    """A row too long for one block and no carried form: every tiling is refused, so
    lowering parity, device compile and staging had nothing to look at."""
    _job(monkeypatch, lambda: [X.streamed_ir(ROWS[0], dict(R=64, C=16384), ROWS[2])])
    status, lines, report = _run(tmp_path, capsys, "job")
    assert status == 2 and lines[-1] == "job: NOT CHECKED, 1 item(s)"
    assert report["rows"] and not any(r["accepted"] for r in report["rows"])
    assert not report["problems"]
    (item,) = report["unchecked"]
    assert item.startswith(f"none of the {len(report['rows'])} candidates was accepted")


def test_nvcc_without_the_named_kernel_is_not_checked(monkeypatch, tmp_path, capsys):
    """The job names a tiling the filter refuses: nothing reaches the device compile, and
    that is not a pass."""
    _job(monkeypatch, lambda: [X.streamed_ir(*ROWS)],
         nvcc=lambda c: list(c.output_tile) == [1, 1024])
    status, lines, report = _run(tmp_path, capsys, "job", "--nvcc")
    assert status == 2 and lines[-1] == "job: NOT CHECKED, 1 item(s)"
    (item,) = report["unchecked"]
    assert item.startswith("(d) --nvcc: the kernel this job names is not among the accepted")
    assert not any("nvcc" in r for r in report["rows"])
    # without --nvcc the same job has nothing left unchecked
    assert _run(tmp_path, capsys, "job")[0] == 0


def test_nvcc_on_a_job_that_names_no_kernel_says_so(monkeypatch, tmp_path, capsys):
    _job(monkeypatch, lambda: [X.streamed_ir(*ROWS)])
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
    monkeypatch.setattr(tool, "_searched_graphs", lambda job: (graphs, {}, notes, (0, 0)))
    status, lines, report = _run(tmp_path, capsys, "demo:mha_prefill")
    assert status == 2 and lines[-1] == last
    assert report["unchecked"] == notes + ["no candidate of any graph was censused"]


def test_one_refused_graph_keeps_a_demo_job_from_ok(monkeypatch, tmp_path, capsys):
    """The other graph is censused and clean; the job still does not end ``ok``."""
    note = "graph g1: refused at lowering: LoweringRejected: nope"
    monkeypatch.setattr(
        tool, "_searched_graphs",
        lambda job: ([("g0", X.streamed_ir(*ROWS), "engine")], {}, [note], (20, 20)),
    )
    status, lines, report = _run(tmp_path, capsys, "demo:mha_prefill")
    assert status == 2 and lines[-1] == "demo:mha_prefill: NOT CHECKED, 1 item(s)"
    assert report["unchecked"] == [note] and not report["problems"]
    assert lines[0].startswith("demo:mha_prefill[g0]: 25 raw candidates, 20 accepted, 20 build")
    assert "    NOT CHECKED " + note in lines


@pytest.mark.parametrize(("counts", "problems"), [
    ((20, 20), []),
    ((20, 24), ["(f) 20 plans where the hand-derived pass keeps 24"]),
    ((24, 24), ["(f) 24 plans but 20 candidates accepted"]),
], ids=["equal", "fewer than the hand-derived pass", "more than the census accepted"])
def test_the_plan_counts_are_held_to_the_hand_derived_pass(monkeypatch, tmp_path, capsys,
                                                           counts, problems):
    """Check (f): the searched program keeps as many plans as the hand-derived pass, and
    they are the candidates the census accepted."""
    monkeypatch.setattr(
        tool, "_searched_graphs",
        lambda job: ([("g0", X.streamed_ir(*ROWS), "engine")], {}, [], counts),
    )
    status, lines, report = _run(tmp_path, capsys, "demo:mha_prefill")
    assert report["problems"] == problems and status == (1 if problems else 0)
    assert f"    (f) plans: {counts[0]} searched, {counts[1]} with the hand-derived pass, " \
           "20 accepted by the census" in lines


def test_a_count_that_differs_by_design_says_why(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        tool, "_searched_graphs",
        lambda job: ([("g0", X.streamed_ir(*ROWS), "engine")], {}, [], (20, 24)),
    )
    monkeypatch.setitem(tool.COUNTS_DIFFER, "demo:mha_prefill", (20, 24, "a reason"))
    status, lines, report = _run(tmp_path, capsys, "demo:mha_prefill")
    assert status == 0 and not report["problems"] and "    (f) by design: a reason" in lines
    monkeypatch.setitem(tool.COUNTS_DIFFER, "demo:mha_prefill", (20, 16, "a reason"))
    status, _, report = _run(tmp_path, capsys, "demo:mha_prefill")
    assert status == 1 and report["problems"] == [
        "(f) 20 and 24 plans where 20 and 16 are expected"]


def test_search_a_demo_job_censuses_the_searched_graphs(monkeypatch, tmp_path, capsys):
    """The real thing, on the smallest demo: its own program (``cl.softmax`` records the
    explicit max), searched, lowered with the engine's records, and every tiling of both
    searched graphs censused. Flash form: every tiling is served, 24 of them looped; traced
    order: only the ones that hold the row whole. The plans are the hand-derived pass's 40."""
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path / "cache"))
    status, lines, report = _run(tmp_path, capsys, "demo:mha_decode@256")
    assert status == 0 and lines[-1] == "demo:mha_decode@256: ok", lines[-6:]
    assert not report["unchecked"] and not report["problems"]
    assert report["counts"] == [40, 40]

    by_graph = {}
    for row in report["rows"]:
        by_graph.setdefault(row["graph"], []).append(row)
    assert len(by_graph) == 2
    counts = sorted(
        (len(rows), sum(r["accepted"] for r in rows), sum(r["tiling"] == "looped" for r in rows))
        for rows in by_graph.values()
    )
    assert counts == [(32, 8, 0), (32, 32, 24)]
    heads = [line for line in lines if line.startswith("demo:mha_decode@256[")]
    assert len(heads) == 2 and sum(line.endswith("[records: engine]") for line in heads) == 1


def test_search_the_demo_at_its_own_shape_is_all_looped(monkeypatch, tmp_path, capsys):
    """The demo's default row (S_KV = 4096) is longer than a block: the flash form's tilings
    all carry the max, and the traced order has no tiling to census."""
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path / "cache"))
    status, lines, report = _run(tmp_path, capsys, "demo:mha_decode")
    assert status == 0 and lines[-1] == "demo:mha_decode: ok", lines[-6:]
    assert not report["unchecked"] and not report["problems"]
    assert report["counts"] == [32, 32]
    rows = [r for r in report["rows"] if r["accepted"]]
    assert len(rows) == 32 and all(r["tiling"] == "looped" for r in rows)
    heads = [line for line in lines if line.startswith("demo:mha_decode[")]
    assert len(heads) == 2 and sum(" 0 raw candidates" in line for line in heads) == 1


# ------------------------------------------------------------------------------ split jobs


def test_search_a_split_job_censuses_the_partial_and_the_combine(monkeypatch, tmp_path,
                                                                 capsys):
    """MHA decode, searched with the demo's own split, its pair built from the loop state
    at 32 splits of 128 keys. Every tiling of the partial is served with the max it
    returns (the looped ones carry it, the ones that take a split in one tile keep the
    flat graph's own), every combine that loops over the splits is a plain loop; the
    traced graph has no pair and says why."""
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path / "cache"))
    status, lines, report = _run(tmp_path, capsys, "split:mha_decode@32")
    assert status == 0 and lines[-1] == "split:mha_decode@32: ok", lines[-8:]
    assert not report["unchecked"] and not report["problems"]
    by_graph = {}
    for row in report["rows"]:
        by_graph.setdefault(row["graph"], []).append(row)
    (partial,) = [rows for g, rows in by_graph.items() if g.endswith(".partial")]
    (combine,) = [rows for g, rows in by_graph.items() if g.endswith(".combine")]
    assert Counter(r["tiling"] for r in partial if r["accepted"]) == {"looped": 16, "one tile": 8}
    assert Counter(r["tiling"] for r in combine if r["accepted"]) == {"looped": 21, "one tile": 3}
    # (raw candidates, accepted, looped, one tile): the hand-derived pass's counts. The
    # 12 combines that are not accepted step one partial at a time or hold a single row
    assert _split_counts(partial) == (24, 24, 16, 8)
    assert _split_counts(combine) == (36, 24, 21, 3)
    assert tool.SPLITS["mha_decode@32"] == ("mha_decode", 32)
    assert any("no pair: LoweringRejected" in line
               and "carries no proven update over that dim: reduce_max_2_5" in line
               for line in lines)


def _split_counts(rows):
    """(raw candidates, accepted, accepted looped, accepted in one tile) of a split stage."""
    accepted = [r for r in rows if r["accepted"]]
    return (len(rows), len(accepted), sum(r["tiling"] == "looped" for r in accepted),
            sum(r["tiling"] == "one tile" for r in accepted))


def test_search_a_split_job_of_a_softmax_over_an_input(monkeypatch, tmp_path, capsys):
    """``softmax(x) @ v`` split over the softmax axis, 4 splits of 256 keys: the partial
    reads the logits through the view of the split. All 12 of its tilings are served: 8
    loop over the keys of a split with its returned max carried, 4 take a split in one
    tile. (The 8 looped ones were once refused, by the hand-derived pass and on this path:
    the tile of the input was asked for a dim of the split tensor.)"""
    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path / "cache"))
    status, lines, report = _run(tmp_path, capsys, "split:rows@4")
    assert status == 0 and lines[-1] == "split:rows@4: ok", lines[-8:]
    assert not report["unchecked"] and not report["problems"]
    partial = [r for r in report["rows"] if r["graph"].endswith(".partial")]
    combine = [r for r in report["rows"] if r["graph"].endswith(".combine")]
    assert Counter(r["tiling"] for r in partial) == {"looped": 8, "one tile": 4}
    assert all(r["accepted"] for r in partial)
    assert _split_counts(partial) == (12, 12, 8, 4)
    assert _split_counts(combine) == (21, 12, 6, 6)  # the hand-derived pass's counts, as above
    assert sorted(tool.PROGRAM_SPLITS) == ["rows@4", "rows@8"]


def test_the_split_check_fails_a_pipelined_loop_over_the_splits(monkeypatch, tmp_path,
                                                                capsys):
    """Check (g) has teeth: a combine whose loop over the splits stays pipelined."""
    real = tool._split_check

    def pipelined(role):
        check = real(role)
        if role != "combine":
            return check

        def loose(tiled):
            for tier, owner in tool.graph_tiers(tiled):
                if owner is not None:
                    tier.loop_kind = "pipelined"
            return check(tiled)

        return loose

    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(tool, "_split_check", pipelined)
    status, lines, report = _run(tmp_path, capsys, "split:rows@4")
    assert status == 1 and lines[-1].startswith("split:rows@4: FAILED")
    assert report["problems"] and all(
        p.startswith("(g) ") and "the loop over the splits is" in p for p in report["problems"])


def test_the_split_check_fails_a_partial_that_does_not_return_its_carried_max(monkeypatch,
                                                                              tmp_path):
    """Check (g) has teeth on the partial too. On the pair of ``split:rows@4`` it passes a
    looped and a one-tile tiling as they are built, and names what a looped partial loses:
    the carried max as the value it returns, its start value, its place among the loop's
    accumulators, its lag copy, a repair; on either kind, the order and roles of what the
    partial returns; and a one-tile partial whose output reads a carried value."""
    from cleave.ir.graph import AccumEdge
    from cleave.runtime.reference import output_order

    monkeypatch.setenv("CLEAVE_CACHE_ROOT", str(tmp_path / "cache"))
    graphs, _, notes, _ = tool._split_graphs("split:rows@4")
    assert not notes
    ((_, partial, _, check),) = [g for g in graphs if g[0].endswith(".partial")]
    tilings = {}
    for cand in tool.search_tiling(partial, topk=1000):
        tilings.setdefault(bool(tool.sites(tool.apply_tiling(cand))), cand)
    assert set(tilings) == {True, False}

    def tiled(looped):
        """A fresh tiling, and for a looped one the loop that carries the max, and the max."""
        graph = tool.apply_tiling(tilings[looped])
        if not looped:
            return graph, None, None
        (site,) = tool.sites(graph)
        (tier,) = [t for t, o in tool.graph_tiers(graph) if o is not None and o.name == site.tier]
        (full,) = [n for n in tier.nodes if n.name == site.full]
        return graph, tier, full

    assert check(tiled(True)[0]) is None and check(tiled(False)[0]) is None

    graph, tier, full = tiled(True)
    (edge,) = output_order(graph)[0].inputs
    edge.accum_node = next(n for n in tier.accum_nodes if n is not full)
    assert check(graph) == "the returned max is not the carried one"

    graph, tier, full = tiled(True)
    full.accum_init = float("-inf")
    assert check(graph) == "the max is not carried from the most negative float32"

    graph, tier, full = tiled(True)
    tier.accum_nodes = [n for n in tier.accum_nodes if n is not full]
    assert check(graph) == "the max is not carried from the most negative float32"

    graph, tier, full = tiled(True)
    tier.lag_nodes = []
    assert check(graph) == "lag copies []"

    graph, tier, full = tiled(True)
    kept = sorted(tier.accum_scale, key=lambda a: a.name)[0]
    tier.accum_scale = {kept: tier.accum_scale[kept]}
    assert check(graph) == f"repairs {[kept.name]}"

    for looped in (True, False):
        graph = tiled(looped)[0]
        shift = output_order(graph)[0]
        shift.boundary = shift.boundary._replace(role="")
        assert check(graph) == (
            "the partial does not return its float32 max first, then two float32 sums")

    graph = tiled(False)[0]
    (edge,) = output_order(graph)[0].inputs
    edge.__class__ = type("Carried", (AccumEdge,), {})  # what the check tells a carried max by
    assert check(graph) == "the returned max is carried"


def test_importing_the_tool_leaves_the_process_alone():
    """The tool is importable (this module does it): the environment and ``sys.path`` are
    only touched when it runs as a script, and a demo job puts back what it borrowed."""
    import sys

    import cleave.examples as examples
    from cleave.pipeline import compile as real_compile

    assert str(tool.ROOT / "demo") not in sys.path
    assert examples._compile is real_compile
    source = (tool.ROOT / "tests" / "_streaming_census.py").read_text()
    head, guard, rest = source.partition('if __name__ == "__main__":  # run as a script')
    assert guard and "os.environ" not in head
    assert rest.count("os.environ") == 1 and rest.index("os.environ") < rest.index("\n\n")


def test_every_census_job_is_listed():
    """The census jobs: MHA / GQA / MQA prefill and decode, ``softmax(x) @ v``, the
    first-axis program, MLA decode at 14 (and MLA prefill)."""
    jobs = set(tool.jobs())
    assert {"demo:mha_prefill", "demo:gqa_prefill", "demo:mqa_prefill", "demo:mha_decode",
            "demo:gqa_decode", "demo:mqa_decode", "demo:mla_decode", "demo:mla_prefill",
            "program:rows_v", "program:first_axis"} <= jobs
    # ... and the split jobs: every decode demo at every factor it asks for, and
    # softmax(x) @ v over its rows
    assert {j for j in jobs if j.startswith("split:")} == {
        "split:mha_decode@16", "split:mha_decode@32", "split:gqa_decode@4",
        "split:gqa_decode@8", "split:gqa_decode@16", "split:mqa_decode@4",
        "split:mqa_decode@8", "split:mqa_decode@16", "split:rows@4", "split:rows@8"}
    assert set(tool.HAND_DERIVED_PLANS) == {j for j in jobs if j.startswith(("demo:", "program:"))}
    # MLA decode is censused at the budget its demo pins, the one its graphs need with the
    # max in them
    program, so, _ = tool._configured("demo:mla_decode")
    assert so.resolve_max_ops(program) == 14
