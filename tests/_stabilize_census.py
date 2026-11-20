"""Census of the softmax stabilization rewrite over whole candidate lists (CPU only).

    python tests/_stabilize_census.py <outdir> list
    python tests/_stabilize_census.py <outdir> <job> [--lower N|all] [--nvcc]

For every tiling candidate of a marked graph (``search_tiling`` with the job's options, no
top-k cut) it checks what unit tests cannot afford to:

  (a) accepted == builds   the accept filter keeps a candidate iff the default build
                           (``build_candidate_code``) produces a kernel for it;
  (b) typed refusals       ``apply_tiling`` raises nothing but ``ScheduleRejected``;
  (c) lowering parity      for accepted candidates (``--lower N`` evenly spread ones plus
                           the ones a job singles out, or
                           ``all``): ``tilelang.lower`` for an offline sm_90a target gives
                           the same verdict for the stabilized kernel and for its plain
                           twin, the same tiling with the record taken off;
  (d) device compile       ``--nvcc``: the job's named kernel also passes the device
                           compile (set ``TILELANG_CACHE_DIR`` to an empty directory, or a
                           cached binary is returned without running nvcc);
  (e) staging              with ``mla_decode_pass`` the value matmul still stages the exp
                           through shared memory, as in the plain twin.

Jobs come from two sources. ``fixture`` jobs run today: the written-out programs of
``tests/_softmax.py`` and the attention fixtures of ``tests/_graphs.py`` (the end2end
cases), with the ``Stabilize`` record put on the Exp by hand. ``demo`` jobs take a demo's
own program and configs through ``superoptimize`` and ``plans.lower`` and need the record
lowering writes: they report ``no record`` on a tree whose frontend does not request
stabilization yet.

Writes ``<outdir>/<job>.json`` and prints a summary; exits 1 if any check failed.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python"), str(ROOT / "demo")]

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
DEMOS = ("mha_prefill", "gqa_prefill", "mqa_prefill", "mla_prefill", "mla_decode", "qknorm",
         "mha_decode_exact")


def _demo_graphs(name):
    """The solo graphs a demo's own program lowers to (``plans.lower``, records included)."""
    import runpy

    import cleave.examples as examples
    from cleave.pipeline import superoptimize
    from cleave.pipeline.plans import lower

    seen = {}

    class Stop(Exception):
        pass

    def fake_compile(program, *, superopt, schedule, eval, **kw):
        seen.update(program=program, so=superopt, sc=schedule)
        raise Stop

    examples._compile, examples._target = fake_compile, lambda: "cpu"
    sys.argv = [str(ROOT / "demo" / f"{name}.py")]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except Stop:
        pass
    graphs = superoptimize(seen["program"], seen["so"].replace(split=()))
    irs = []
    for g in graphs:
        if not g.is_split and not g.invalid_split:
            try:
                irs.extend(lower(g))
            except Exception as e:  # noqa: BLE001 - a refused graph is reported, not hidden
                print(f"  graph {g.index}: {type(e).__name__}: {str(e)[:200]}")
    options = {"force_fusion": True} if seen["sc"].fusion == "force" else {}
    return irs, options


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


def summarize(name, rows, problems, with_nvcc):
    tiling = Counter(r["tiling"] for r in rows)
    lowered = [r for r in rows if "lower" in r]
    verdicts = Counter((r["lower"].split(":")[0], r["plain_lower"].split(":")[0]) for r in lowered)
    over = sorted(r["overhead_bytes"] for r in rows if "overhead_bytes" in r)
    print(f"{name}: {len(rows)} raw candidates, {sum(r['accepted'] for r in rows)} accepted, "
          f"{sum(r['build'] == 'ok' for r in rows)} build")
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
    if with_nvcc and not any("nvcc" in r for r in rows):
        print("    +nvcc: the job names no kernel, or it was not accepted")
    for line in problems[:20]:
        print("    PROBLEM", line[:300])


def main(argv):
    out, job = Path(argv[1]), argv[2]
    if job == "list":
        print("\n".join(list(FIXTURES) + [f"demo:{d}" for d in DEMOS]))
        return 0
    n_lower = argv[argv.index("--lower") + 1] if "--lower" in argv else "8"
    with_nvcc = "--nvcc" in argv
    out.mkdir(parents=True, exist_ok=True)
    if job.startswith("demo:"):
        irs, search = _demo_graphs(job[5:])
        options, nvcc = (MLA if job == "demo:mla_decode" else None), None
        focus = None
    else:
        graphs, search, options, nvcc, focus = (FIXTURES[job] + (None,))[:5]
        irs = graphs()
    all_rows, all_problems = [], []
    for k, ir in enumerate(irs):
        name = job if len(irs) == 1 else f"{job}[{k}]"
        if not annotated(ir):
            print(f"{name}: no record (lowering marked no exp of this graph)")
            continue
        rows, problems = census(ir, dict(search), options, n_lower,
                                nvcc if with_nvcc else None, focus)
        summarize(name, rows, problems, with_nvcc)
        all_rows += [dict(r, graph=k) for r in rows]
        all_problems += problems
    for r in all_rows:
        r.pop("code", None), r.pop("plain_code", None)
    (out / f"{job.replace(':', '_')}.json").write_text(
        json.dumps(dict(rows=all_rows, problems=all_problems), indent=1) + "\n")
    print(f"{job}: {'FAILED, ' + str(len(all_problems)) + ' problem(s)' if all_problems else 'ok'}")
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
