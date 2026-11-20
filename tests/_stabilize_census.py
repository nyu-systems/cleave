"""Census of the softmax stabilization rewrite over whole candidate lists (CPU only).

    python tests/_stabilize_census.py <outdir> list
    python tests/_stabilize_census.py <outdir> <job> [--lower N|all] [--nvcc]

For every tiling candidate of a marked graph (``search_tiling`` with the job's options, no
top-k cut) it checks what unit tests cannot afford to:

  (a) accepted == builds   the accept filter keeps a candidate iff the default build
                           (``build_candidate_code``) produces a kernel for it;
  (b) typed refusals       ``apply_tiling`` raises nothing but ``ScheduleRejected``;
  (c) lowering parity      for accepted candidates (``--lower N`` evenly spread ones plus
                           the ones a job singles out, or ``all``): ``tilelang.lower`` for
                           an offline sm_90a target gives the same verdict for the
                           stabilized kernel and for its plain twin, the same tiling with
                           the record taken off;
  (d) device compile       ``--nvcc``: the job's named kernel also passes the device
                           compile (set ``TILELANG_CACHE_DIR`` to an empty directory, or a
                           cached binary is returned without running nvcc);
  (e) staging              with ``mla_decode_pass`` the value matmul still stages the exp
                           through shared memory, as in the plain twin.

Jobs come from two sources.

``fixture`` jobs: the written-out programs of ``tests/_softmax.py`` and the attention
fixtures of ``tests/_graphs.py`` (the end2end cases), with the ``Stabilize`` record put on
the Exp by hand (checked against lowering's own where the tree's lowering writes one).

``demo:`` jobs: a demo's own program and configs through ``superoptimize`` and
``plans.lower``, i.e. the *searched* graphs. The demo is traced with the ``shift=`` of its
``cl.softmax`` dropped, so that it records the exact softmax (the requested exp, its sum,
the division) and the search sees the shift-free program. The records are the ones lowering
writes. On a tree whose frontend cannot make the request (no ``Program._stable_exp``)
lowering writes none, and they are set by hand instead (``tests/_softmax.py::hand_record``);
the summary says which.

Exit status, also the last line printed:

  0  ``ok``           every check the job is there for ran, and passed;
  1  ``FAILED``       a check failed (lines marked ``PROBLEM``);
  2  ``NOT CHECKED``  nothing failed, but something was not checked (lines marked
                      ``NOT CHECKED``): a graph without a record, a graph refused at
                      lowering, a job that censused no graph or accepted no candidate, or
                      ``--nvcc`` on a job whose named kernel is not among the accepted
                      candidates. A job is never reported ``ok`` for having had nothing
                      to look at.

Writes ``<outdir>/<job>.json``.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":  # run as a script: no GPU, and this tree first on the path
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import tilelang  # noqa: E402,F401  (puts its bundled tvm on sys.path)

from cleave.codegen import TilelangCodeGen, passes  # noqa: E402
from cleave.core.errors import ScheduleRejected  # noqa: E402
from cleave.ir.ops import Exp, Matmul, Stabilize  # noqa: E402
from cleave.ir.graph_utils import graph_tiers  # noqa: E402
from cleave.runtime.compile import CompilerPassOptions, build_candidate_code  # noqa: E402
from cleave.schedule import apply_tiling, search_tiling  # noqa: E402
from cleave.schedule.stabilize import (  # noqa: E402
    _reason_key, annotated, overhead_bytes, sites, stabilize_accept,
)

from tests import _graphs as G  # noqa: E402
from tests import _softmax as S  # noqa: E402
from tests._lowering import lower_to_cuda  # noqa: E402

NEXT_TO_SUM = "div reads the exp next to the finished reduction"
MLA = CompilerPassOptions(mla_decode_pass=True)


def _marked_fixture(build, dims):
    """An attention fixture with the record put on its Exp by hand."""
    ir = build(*dims)
    (exp,) = S.exps(ir)
    exp.stabilize = Stabilize(axes=(-1,), scaled=S.scaled_reductions(exp))
    return ir


def _rsteps(cand):
    return [int(x) for v in cand.rstep_map.values() for x in (v.values() if v else ())]


# name -> (graphs(), search options, compiler pass options, selector of the +nvcc kernel
#          [, selector of candidates that are always lowered, whatever --lower samples])
FIXTURES = {
    "mha_4096_flash": (
        lambda: [S.softmax_ir(S.flash_attention, dict(H=32, SQ=4096, SKV=4096, D=64))],
        {}, None, lambda c: list(c.output_tile) == [1, 128, 64] and c.n_threads == 128),
    "mha_256_flash": (
        lambda: [S.softmax_ir(S.flash_attention, dict(H=32, SQ=64, SKV=256, D=64))],
        {}, None, lambda c: _rsteps(c)[1:] == [256, 256] and c.n_threads == 128),
    "mha_256_traced": (
        lambda: [S.softmax_ir(S.traced_attention, dict(H=32, SQ=64, SKV=256, D=64),
                              dict(rolling=NEXT_TO_SUM))],
        {}, None, None),
    "softmax": (
        lambda: [S.softmax_ir(S.softmax_rows, dict(R=64, C=1024), dict(rolling=NEXT_TO_SUM))],
        {}, None, lambda c: True),
    "gqa": (lambda: [_marked_fixture(G.gqa_bhsd_graph, (1, 32, 8, 4096, 4096, 64))],
            {}, None, None),
    "mqa": (lambda: [_marked_fixture(G.mqa_bhsd_graph, (1, 32, 1, 4096, 4096, 64))],
            {}, None, None),
    "mla_prefill": (
        lambda: [_marked_fixture(G.mla_prefill_bhsd_graph, (1, 128, 4096, 4096, 192, 128))],
        {}, None, None),
    # The only tilings of this operator that validate on a GPU hold the whole head dim in
    # the block: output tile [1, 64, 512] with the softmax loop stepped 32 (20 of the 600).
    "mla_decode": (
        lambda: [_marked_fixture(G.mla_decode_bhsd_graph, (1, 128, 4096, 512, 64))],
        {}, MLA,
        lambda c: (list(c.output_tile) == [1, 64, 512] and _rsteps(c) == [512, 64, 32, 32]
                   and c.n_threads == 128),
        lambda c: list(c.output_tile) == [1, 64, 512] and _rsteps(c)[2:] == [32, 32]),
    "qknorm": (
        lambda: [_marked_fixture(G.qknorm_graph, (1, 64, 8, 4096, 4096, 128))],
        {"topk": 32, "force_fusion": True}, None, lambda c: True),
    "raw_statistics": (lambda: [S.raw_statistics_graph(32, 64, 4096, 64)], {}, None,
                       lambda c: c.n_threads == 128),
}


def _whole_head_dim(c):
    """mla_decode: the block holds all 512 head dims and the softmax loop steps 32 keys
    (the shape of the only tilings of this operator that validate on a GPU)."""
    return list(c.output_tile) == [1, 64, 512] and _rsteps(c)[-2:] == [32, 32]


# name -> (compiler pass options, selector of the +nvcc kernel,
#          selector of candidates that are always lowered, whatever --lower samples)
DEMOS = {
    "mha_prefill": (None, None, None),
    "gqa_prefill": (None, None, None),
    "mqa_prefill": (None, None, None),
    "mla_prefill": (None, None, None),
    "mla_decode": (MLA, lambda c: _whole_head_dim(c) and c.n_threads == 128, _whole_head_dim),
    "qknorm": (None, lambda c: c.n_threads == 128, None),
    "mha_decode_exact": (None, None, None),
    # the same demo at a row one block holds: one graph has in-tile and rolling tilings
    "mha_decode_exact@256": (None, None, None),
}
# job -> (demo file, its command line), where they differ from (the job's name, none)
DEMO_ARGV = {"mha_decode_exact@256": ("mha_decode_exact", ["--SKV", "256"])}


def _demo_graphs(name):
    """``(graphs, search options, notes)`` for one demo.

    ``graphs`` are the solo graphs the demo's own program lowers to, each as ``(label, ir,
    where its records come from)``; ``notes`` say why a graph of the search is not among
    them. The program is the demo's, traced with ``cl.softmax`` as the frontend will
    record it (see the module docstring).
    """
    import runpy

    import cleave.examples as examples
    import cleave.lang as cl
    from cleave.frontend.tensor import Tensor
    from cleave.pipeline import superoptimize
    from cleave.pipeline.plans import lower

    seen = {}

    class Stop(Exception):
        pass

    def fake_compile(program, *, superopt, schedule, eval, **kw):
        seen.update(program=program, so=superopt, sc=schedule)
        raise Stop

    real_softmax = cl.softmax

    def requested_softmax(x, dim=-1, shift=None, **paged):
        if paged or not isinstance(x, Tensor):
            return real_softmax(x, dim, shift, **paged)
        return real_softmax(x, dim)  # the demo's ``shift=`` dropped: the exact softmax

    name, flags = DEMO_ARGV.get(name, (name, []))
    real_compile, real_target, argv = examples._compile, examples._target, sys.argv
    examples._compile, examples._target = fake_compile, lambda: "cpu"
    cl.softmax = requested_softmax
    sys.argv = [str(ROOT / "demo" / f"{name}.py"), *flags]
    sys.path.insert(0, str(ROOT / "demo"))  # the demos import each other by bare name
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except Stop:
        pass
    finally:
        examples._compile, examples._target = real_compile, real_target
        cl.softmax, sys.argv = real_softmax, argv
        sys.path.remove(str(ROOT / "demo"))
    program = seen["program"]
    requests = getattr(program, "stabilize_requests", None)
    graphs, notes = [], []
    for g in superoptimize(program, seen["so"].replace(split=())):
        if g.is_split or g.invalid_split:
            continue
        try:
            irs = lower(g)
        except Exception as e:  # noqa: BLE001 - a refused graph is reported, not hidden
            notes.append(f"graph g{g.index}: refused at lowering: {_first(e)}")
            continue
        for k, ir in enumerate(irs):
            label = f"g{g.index}" if len(irs) == 1 else f"g{g.index}.{k}"
            if annotated(ir):
                graphs.append((label, ir, "lowering"))
            elif requests is not None:
                notes.append(f"graph {label}: no record (lowering marked no exp of it; the "
                             f"program makes {len(requests)} request(s))")
            else:  # this tree's frontend cannot make the request: records by hand
                found = [(exp, S.hand_record(exp)) for exp in S.exps(ir)]
                if not found or any(record is None for _, record in found):
                    notes.append(f"graph {label}: no record (no request in this tree, and "
                                 "the by-hand rule does not cover this graph)")
                    continue
                for exp, record in found:
                    exp.stabilize = record
                graphs.append((label, ir, "by hand"))
    options = {"force_fusion": True} if seen["sc"].fusion == "force" else {}
    return graphs, options, notes


def _first(e):
    return f"{type(e).__name__}: {(str(e).splitlines() or [''])[0][:160]}"


def _passes(tiled, options):
    g = passes.annotate_args(tiled, is_main=True)
    g = passes.annotate_tinfo(g, is_main=True)
    if options.tc_awareness_pass:
        g = passes.annotate_tc_layout(g)
    g = passes.annotate_fragment_layout_conflicts(g)
    if options.mla_decode_pass:
        g = passes.annotate_mla_accumulation_layout_conflict(g)
    return g


def _staged(graph):
    """Value matmuls (lhs an Exp) that stage operand 0 through shared memory."""
    return sum(
        1 for tier, _ in graph_tiers(graph) for n in tier.nodes
        if isinstance(n, Matmul) and isinstance(n.inputs[0].src_node, Exp)
        and tuple(getattr(n, "stage_fragment_operands_in_shared", ())) == (0,)
    )


class _Plain:
    """The flat graph with its records taken off, for the duration of a ``with``."""

    def __init__(self, ir):
        self.marks = [(exp, exp.stabilize) for _, _, exp in annotated(ir)]

    def __enter__(self):
        for exp, _ in self.marks:
            exp.stabilize = None

    def __exit__(self, *exc):
        for exp, record in self.marks:
            exp.stabilize = record


def _lower(code, device_compile=False):
    try:
        lower_to_cuda(code, device_compile=device_compile)
        return "ok"
    except Exception as e:  # noqa: BLE001 - the verdict is the result
        return _first(e)


def census(ir, search, options, n_lower, nvcc, focus=None):
    options = options or CompilerPassOptions()
    problems, rows = [], []
    force = search.get("force_fusion", False)
    cands = list(search_tiling(ir, **{"topk": 1000, **search}))
    if not cands and not force:  # fusion="auto": retry forced, as pipeline.schedule does
        search = {**search, "force_fusion": True}
        cands = list(search_tiling(ir, **{"topk": 1000, **search}))
    accept = stabilize_accept(ir)
    assert accept is not None, "census of a graph without a marked exp"
    for i, cand in enumerate(cands):
        label = (f"#{i} tile={list(cand.output_tile)} rstep={_rsteps(cand)} "
                 f"threads={cand.n_threads}")
        row = dict(label=label, accepted=bool(accept(cand)))
        try:  # (b)
            tiled = apply_tiling(cand)
            row["tiling"] = "+".join(sorted(s.case for s in sites(tiled)))
            row["overhead_bytes"] = overhead_bytes(tiled)
        except ScheduleRejected as e:
            row["tiling"] = "refused: " + _reason_key(e)
        except Exception as e:  # noqa: BLE001
            row["tiling"] = "BUG " + _first(e)
            problems.append(f"(b) {label}: apply_tiling raised {_first(e)}")
        try:  # (a)
            row["code"] = build_candidate_code(cand, options)
            row["build"] = "ok"
        except Exception as e:  # noqa: BLE001
            row["code"], row["build"] = None, _first(e)
        if row["accepted"] != (row["build"] == "ok"):
            problems.append(f"(a) {label}: accepted={row['accepted']} build={row['build']}")
        if not row["accepted"] and not row["build"].startswith(
                ("ScheduleRejected", "StabilizeRefused")):
            problems.append(f"(a) {label}: refused candidate failed as {row['build']}")
        rows.append((cand, row))

    kept = [(c, r) for c, r in rows if r["build"] == "ok"]
    for cand, row in kept:  # (e), and the plain twin's text
        row["staged"] = _staged(_passes(apply_tiling(cand), options))
        with _Plain(ir):
            try:
                row["plain_code"] = build_candidate_code(cand, options)
                row["plain_staged"] = _staged(_passes(apply_tiling(cand), options))
            except Exception as e:  # noqa: BLE001
                row["plain_code"], row["plain_staged"] = None, None
                row["plain_build"] = _first(e)
        if options.mla_decode_pass and row["staged"] != row["plain_staged"]:
            problems.append(f"(e) {row['label']}: staged {row['staged']} value matmul(s), "
                            f"plain twin {row['plain_staged']}")
    if options.mla_decode_pass and kept and not all(r["staged"] for _, r in kept):
        problems.append("(e) mla_decode_pass stages no value matmul in some kernel")

    if n_lower == "all" or len(kept) <= int(n_lower):
        chosen = kept
    elif int(n_lower) == 0:
        chosen = []
    else:
        step = len(kept) / int(n_lower)
        chosen = [kept[int(k * step)] for k in range(int(n_lower))]
    named = [(c, r) for c, r in kept if nvcc is not None and nvcc(c)][:1]
    extra = named + [(c, r) for c, r in kept if focus is not None and focus(c)]
    for cand, row in chosen + extra:
        if "lower" in row:
            continue
        row["lower"] = _lower(row["code"])  # (c)
        row["plain_lower"] = "no plain twin" if row["plain_code"] is None else _lower(
            row["plain_code"])
        if row["lower"].split(":")[0] != row["plain_lower"].split(":")[0]:
            problems.append(f"(c) {row['label']}: stabilized {row['lower']} | plain "
                            f"{row['plain_lower']}")
    for cand, row in named:  # (d)
        row["nvcc"] = _lower(row["code"], device_compile=True)
        if row["nvcc"] != "ok":
            problems.append(f"(d) {row['label']}: device compile: {row['nvcc']}")
    return [r for _, r in rows], problems


def summarize(name, rows, problems, records=None):
    tiling = Counter(r["tiling"] for r in rows)
    lowered = [r for r in rows if "lower" in r]
    verdicts = Counter((r["lower"].split(":")[0], r["plain_lower"].split(":")[0]) for r in lowered)
    over = sorted(r["overhead_bytes"] for r in rows if "overhead_bytes" in r)
    print(f"{name}: {len(rows)} raw candidates, {sum(r['accepted'] for r in rows)} accepted, "
          f"{sum(r['build'] == 'ok' for r in rows)} build"
          + (f"  [records: {records}]" if records else ""))
    for text, n in tiling.most_common():
        print(f"    {n:4d} x {text[:150]}")
    if over:
        print(f"    row-vector overhead per block: {over[0]}..{over[-1]} bytes (diagnostic)")
    if lowered:
        print(f"    lowering (stabilized, plain) over {len(lowered)} kernels: "
              + ", ".join(f"{n}x {a}/{b}" for (a, b), n in verdicts.most_common()))
    staged = [r["staged"] for r in rows if "staged" in r]
    if staged and any(staged):
        print(f"    value matmul staged through shared memory in {sum(bool(s) for s in staged)} "
              f"of {len(staged)} kernels")
    for r in rows:
        if "nvcc" in r:
            print(f"    +nvcc {r['label']}: {r['nvcc']}")
    for line in problems[:20]:
        print("    PROBLEM", line[:300])


def run(job, n_lower="8", with_nvcc=False):
    """Census one job: ``(rows, problems, unchecked)``.

    ``problems`` are checks that failed; ``unchecked`` is what the job was there to check
    and could not: neither list empty means the job is not ``ok``.
    """
    unchecked = []
    if job.startswith("demo:"):
        graphs, search, notes = _demo_graphs(job[5:])
        options, nvcc, focus = DEMOS[job[5:]]
        unchecked += notes
    else:
        build, search, options, nvcc, focus = (FIXTURES[job] + (None,))[:5]
        irs = build()
        graphs = [("" if len(irs) == 1 else str(k), ir, "by hand, or lowering's checked "
                   "against the expectation") for k, ir in enumerate(irs)]
    all_rows, all_problems = [], []
    for label, ir, records in graphs:
        name = f"{job}[{label}]" if label else job
        if not annotated(ir):
            unchecked.append(f"graph {name}: no record on any exp")
            continue
        rows, problems = census(ir, dict(search), options, n_lower,
                                nvcc if with_nvcc else None, focus)
        summarize(name, rows, problems, records)
        all_rows += [dict(r, graph=label) for r in rows]
        all_problems += problems
    if not all_rows:
        unchecked.append("no candidate of any graph was censused")
    elif not any(r["accepted"] for r in all_rows):
        unchecked.append(f"none of the {len(all_rows)} candidates was accepted: lowering "
                         "parity, device compile and staging had nothing to look at")
    if with_nvcc:
        if nvcc is None:
            print("    +nvcc: this job names no kernel (the device compile is not part of it)")
        elif not any("nvcc" in r for r in all_rows):
            unchecked.append("(d) --nvcc: the kernel this job names is not among the "
                             "accepted candidates; nothing went through the device compile")
    return all_rows, all_problems, unchecked


def main(argv):
    out, job = Path(argv[1]), argv[2]
    if job == "list":
        print("\n".join(list(FIXTURES) + [f"demo:{d}" for d in DEMOS]))
        return 0
    n_lower = argv[argv.index("--lower") + 1] if "--lower" in argv else "8"
    rows, problems, unchecked = run(job, n_lower, "--nvcc" in argv)
    for line in unchecked:
        print("    NOT CHECKED", line[:300])
    for r in rows:
        r.pop("code", None), r.pop("plain_code", None)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{job.replace(':', '_')}.json").write_text(
        json.dumps(dict(rows=rows, problems=problems, unchecked=unchecked), indent=1) + "\n")
    if problems:
        print(f"{job}: FAILED, {len(problems)} problem(s)"
              + (f", {len(unchecked)} item(s) not checked" if unchecked else ""))
        return 1
    if unchecked:
        print(f"{job}: NOT CHECKED, {len(unchecked)} item(s)")
        return 2
    print(f"{job}: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
