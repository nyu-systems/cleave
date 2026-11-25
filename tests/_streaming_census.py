"""Census of the carried-max assembly over whole candidate lists (CPU only).

    python tests/_streaming_census.py <outdir> list
    python tests/_streaming_census.py <outdir> <job> [--lower N|all] [--nvcc]

For every tiling candidate of a graph with a streaming obstacle (``search_tiling`` with the
job's options, no top-k cut) it checks what unit tests cannot afford to:

  (a) accepted == builds   the accept filter (``streaming_accept``) keeps a candidate iff
                           the default build (``build_candidate_code``) produces a kernel;
  (b) typed refusals       ``apply_tiling`` raises nothing but ``ScheduleRejected``;
  (c) lowering parity      for accepted candidates (``--lower N`` evenly spread ones plus
                           the ones a job singles out, or ``all``): ``tilelang.lower`` for
                           an offline sm_90a target gives the same verdict for the
                           assembled kernel and for its plain twin, the same tiled loop
                           without the snapshot and the repairs (the record taken off the
                           tiled copy, so the max is emitted as a plain accumulator);
  (d) device compile       ``--nvcc``: the job's named kernel also passes the device
                           compile (set ``TILELANG_CACHE_DIR`` to an empty directory, or a
                           cached binary is returned without running nvcc);
  (e) staging              with ``mla_decode_pass`` the value matmul still stages the exp
                           through shared memory, as in the plain twin;
  (f) counts               a ``demo:`` or ``program:`` job: the plans ``pipeline.schedule``
                           keeps for the program equal the plans the hand-derived pass
                           kept for the same program (``HAND_DERIVED_PLANS``, recorded
                           while that pass was in the tree; ``COUNTS_DIFFER`` where they
                           differ by design), and are the candidates the census accepted;
  (g) split                a ``split:`` job: where the partial loops over the keys of its
                           split, the max it returns is the carried max (an accumulator of
                           that loop, started at the most negative float32, one lag copy,
                           both sums repaired, read by the output through the loop), else
                           the flat graph's own; it returns that max first and its two sums,
                           all float32. The combine loops over the splits in a plain loop.

Jobs come from four sources.

``fixture`` jobs: the programs of ``tests/_explicit_softmax.py`` (the max written out),
lowered as identity graphs; a classified obstacle gets a record by hand (the rule the
engine proves for each of them), a refused one its reason.

``demo:`` jobs: a demo's own program and configs (``cl.softmax`` records the explicit max),
through ``superoptimize`` and ``plans.lower``: the searched graphs, with the records the
engine derives.

``program:`` jobs: the same for a program that is no demo (``softmax(x) @ v`` with an fp16
logits input, the first-axis program ``sum(softmax(x, 1) * y, 1)``).

``split:`` jobs: a decode demo's program, searched with the demo's own split, and the pair
built from the loop state of each solo graph at one factor (``split:mha_decode@16``): the
partial and the combine, each with the records the engine derives for it.
``split:rows@4`` is the same for ``softmax(x) @ v`` split over the softmax axis, whose
partial reads its logits from storage. A solo graph without a pair (the traced order) is
printed with its reason.

Exit status, also the last line printed:

  0  ``ok``           every check the job is there for ran, and passed;
  1  ``FAILED``       a check failed (lines marked ``PROBLEM``);
  2  ``NOT CHECKED``  nothing failed, but something was not checked (lines marked
                      ``NOT CHECKED``): a graph with no obstacle, a graph refused at
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

import cleave.lang as cl  # noqa: E402

from cleave.codegen import passes  # noqa: E402
from cleave.core.errors import ScheduleRejected  # noqa: E402
from cleave.ir.graph_utils import graph_tiers  # noqa: E402
from cleave.ir.ops import Exp, Matmul  # noqa: E402
from cleave.runtime.compile import CompilerPassOptions, build_candidate_code  # noqa: E402
from cleave.schedule import apply_tiling, search_tiling  # noqa: E402
from cleave.schedule import streaming  # noqa: E402
from cleave.schedule.streaming import _reason_key, recorded, sites, streaming_accept  # noqa: E402

from tests import _explicit_softmax as X  # noqa: E402
from tests._lowering import lower_to_cuda  # noqa: E402

MLA = CompilerPassOptions(mla_decode_pass=True)
TWO_PASSES = "a finished reduction over the dim"


def _rsteps(cand):
    return [int(x) for v in cand.rstep_map.values() for x in (v.values() if v else ())]


# name -> (graphs(), search options, compiler pass options, selector of the +nvcc kernel
#          [, selector of candidates that are always lowered, whatever --lower samples])
FIXTURES = {
    "mha_4096_flash": (
        lambda: [X.streamed_ir(X.flash_attention, dict(H=32, SQ=4096, SKV=4096, D=64))],
        {}, None, lambda c: list(c.output_tile) == [1, 128, 64] and c.n_threads == 128),
    "mha_256_flash": (
        lambda: [X.streamed_ir(X.flash_attention, dict(H=32, SQ=64, SKV=256, D=64))],
        {}, None, lambda c: _rsteps(c)[2:] == [256, 256] and c.n_threads == 128),
    "mha_256_traced": (
        lambda: [X.streamed_ir(X.traced_attention, dict(H=32, SQ=64, SKV=256, D=64),
                               TWO_PASSES)],
        {}, None, None),
    "returned_max": (
        lambda: [X.streamed_ir(X.flash_and_its_max, dict(H=32, SQ=64, SKV=4096, D=64))],
        {}, None, None),
    "softmax": (
        lambda: [X.streamed_ir(X.softmax_rows, dict(R=64, C=1024), "is also read by")],
        {}, None, lambda c: True),
    "rows_v": (
        lambda: [X.streamed_ir(X.flash_rows, dict(R=64, C=4096, D=64))], {}, None, None),
    "column_average": (
        lambda: [X.streamed_ir(X.column_average, dict(H=4, K=4096, N=64))], {}, None, None),
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
    "mha_decode": (None, None, None),
    "gqa_decode": (None, None, None),
    "mqa_decode": (None, None, None),
    # the same demo at a row one block holds: one graph has one-tile and looped tilings
    "mha_decode@256": (None, None, None),
}
# job -> (demo file, its command line), where they differ from (the job's name, none)
DEMO_ARGV = {"mha_decode@256": ("mha_decode", ["--SKV", "256"])}


def rows_v(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """``softmax(x) @ v``: the logits are an fp16 input."""
    return cl.softmax(x) @ v


def first_axis(x: cl.Tensor["H", "K", "N"], y: cl.Tensor["H", "K", "N"]):
    """``sum(softmax(x, 1) * y, 1)``: the softmax over the first non-batch axis."""
    return cl.sum(cl.softmax(x, 1) * y, 1, keepdim=True)


# job -> the plans ``pipeline.schedule`` kept for the program with the hand-derived pass
# (no top-k cut), recorded by this census at the integration of the searched path
# (a60f5ed, where both paths were in the tree), before that pass was deleted; less the
# tilings the policy has proposed to neither path since (the attention backward merge): a
# gemm that reads its first operand from another gemm's accumulator with fewer than 16
# rows a warp, or two warp groups at another M than 128, unless codegen stages that operand
# (``TCPolicy._register_operand_layout_conflict``; the 16 CompileErrors of MHA prefill), and
# an input two readers would need with two tiles (``DefaultPolicy._reads_placeholder_
# unevenly``; MLA decode's latent cache under tiles narrower than 512). At a60f5ed: 68 for
# the prefill and the GQA / MQA decode jobs, 138 for MLA prefill, 600 for MLA decode, 32 for
# MHA decode and 40 for MHA decode at 256 keys.
HAND_DERIVED_PLANS = {
    "demo:mha_prefill": 52, "demo:gqa_prefill": 52, "demo:mqa_prefill": 52,
    "demo:mla_prefill": 102, "demo:mla_decode": 60, "demo:mha_decode": 16,
    "demo:gqa_decode": 52, "demo:mqa_decode": 52, "demo:mha_decode@256": 20,
    "program:rows_v": 16, "program:first_axis": 62,
}
# job -> (plans of the searched program, plans of the hand-derived pass, why they differ):
# where the two paths do not keep the same number of tilings, by design. Anything else
# must be equal.
COUNTS_DIFFER = {
    "program:rows_v": (12, 16, "the policy starts a max of 16-bit logits read from storage "
                       "at 64 keys, where its read fills a transaction, and the sums it is "
                       "carried for step with it: the 4 tilings of the hand-derived pass "
                       "that step 32 keys have no counterpart"),
    "program:first_axis": (117, 62, "the streamed family (max>sub>exp>mul>sum>sum>div) has "
                           "57 plans on both paths; in the three families that divide "
                           "before the second sum the max keeps its pin and may run a loop "
                           "of its own ahead of what reads it, at every step the policy "
                           "offers (two passes): 12 + 24 + 24 plans where the hand-derived "
                           "pass, which had no max in the graph, had 1 + 2 + 2"),
}

# name -> (program, shapes, max_ops or None for the auto rule): programs that are no demo
PROGRAM_JOBS = {
    "rows_v": (rows_v, dict(R=64, C=4096, D=64), None),
    "first_axis": (first_axis, dict(H=4, K=4096, N=64), None),
}

# ``split:`` jobs: the pair built from the loop state of a program's flash graph at one
# factor (``pipeline.plans.state_pair``), both stages censused.
# job -> (demo, split factor): the decode demos at the factors they split their keys by
SPLITS = {
    f"{demo}@{factor}": (demo, factor)
    for demo, factors in (("mha_decode", (16, 32)), ("gqa_decode", (4, 8, 16)),
                          ("mqa_decode", (4, 8, 16)))
    for factor in factors
}
# job -> (program, shapes, split dim, split factor): a softmax whose logits are an input,
# split over the softmax axis (its partial reads them through the view of the split)
PROGRAM_SPLITS = {
    f"rows@{factor}": (rows_v, dict(R=64, C=1024, D=64), "C", factor) for factor in (4, 8)
}


def _traced_demo(name, flags=()):
    """``(program, superopt config, schedule config)`` of a demo, as the demo itself traces
    and configures it."""
    import runpy

    import cleave.examples as examples

    seen = {}

    class Stop(Exception):
        pass

    def fake_compile(program, *, superopt, schedule, eval, **kw):
        seen.update(program=program, so=superopt, sc=schedule)
        raise Stop

    real_compile, real_target, argv = examples._compile, examples._target, sys.argv
    examples._compile, examples._target = fake_compile, lambda: "cpu"
    sys.argv = [str(ROOT / "demo" / f"{name}.py"), *flags]
    sys.path.insert(0, str(ROOT / "demo"))  # the demos import each other by bare name
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except Stop:
        pass
    finally:
        examples._compile, examples._target = real_compile, real_target
        sys.argv = argv
        sys.path.remove(str(ROOT / "demo"))
    return seen["program"], seen["so"], seen["sc"]


def _configured(job):
    """``(program, superopt config, schedule config)`` of a demo or program job."""
    from cleave.frontend import trace
    from cleave.pipeline import ScheduleConfig, resolve_configs

    if job.startswith("program:"):
        fn, shapes, max_ops = PROGRAM_JOBS[job[8:]]
        program = trace(fn, shapes, cl.f16)
        so, sc, _ = resolve_configs(program, split=False)
        if max_ops:
            so = so.replace(max_ops=max_ops)
        return program, so, sc if sc is not None else ScheduleConfig()
    name = job[5:]
    program, so, sc = _traced_demo(*DEMO_ARGV.get(name, (name, [])))
    return program, so.replace(split=()), sc


def _searched_graphs(job):
    """``(graphs, search options, notes, counts)`` for a demo or program job.

    ``graphs`` are the solo graphs the program lowers to, each as ``(label, ir, where its
    records come from)``; ``notes`` say why a graph of the search is not among them;
    ``counts`` is ``(plans of the program, plans the hand-derived pass kept for it)``, both
    of ``pipeline.schedule`` with no top-k cut (the second as recorded,
    ``HAND_DERIVED_PLANS``).
    """
    from cleave.pipeline import schedule, superoptimize
    from cleave.pipeline.plans import lower

    program, so, sc = _configured(job)
    graphs, notes = [], []
    found = superoptimize(program, so)
    for g in found:
        if g.is_split or g.invalid_split:
            continue
        try:
            irs = lower(g)
        except Exception as e:  # noqa: BLE001 - a refused graph is reported, not hidden
            notes.append(f"graph g{g.index}: refused at lowering: {_first(e)}")
            continue
        for k, ir in enumerate(irs):
            label = f"g{g.index}" if len(irs) == 1 else f"g{g.index}.{k}"
            if recorded(ir) or getattr(ir, "streaming_refusals", None):
                where = "engine" if recorded(ir) else "refused: " + "; ".join(
                    ir.streaming_refusals)[:200]
                graphs.append((label, ir, where))
            else:
                notes.append(f"graph {label}: no streaming obstacle")
    new = len(schedule(found, sc.replace(topk=100000)))
    old = HAND_DERIVED_PLANS[job]
    options = {"force_fusion": True} if sc.fusion == "force" else {}
    return graphs, options, notes, (new, old)


def _split_graphs(job):
    """``(graphs, search options, notes, info)`` for a split job: the partial and the
    combine of every solo graph whose pair is built from its loop state (``lower(g, f)``),
    each as ``(label, ir, where its records come from, check)``; ``notes`` is what the job
    could not check (no pair at all), ``info`` why the other solo graphs have none (the
    traced order: P1 with its two-pass reason)."""
    from cleave.core.errors import LoweringRejected
    from cleave.frontend import trace
    from cleave.pipeline import SuperoptConfig, superoptimize
    from cleave.pipeline.plans import lower

    name = job[6:]
    if name in SPLITS:
        demo, factor = SPLITS[name]
        program, so, sc = _traced_demo(demo)
    else:
        fn, shapes, dim, factor = PROGRAM_SPLITS[name]
        program, so, sc = trace(fn, shapes, cl.f16), SuperoptConfig(split=(dim,)), None
    graphs, notes, info = [], [], []
    for g in superoptimize(program, so):
        try:
            partial, combine = lower(g, factor)
        except LoweringRejected as e:
            info.append(f"graph g{g.index}: no pair: {_first(e)}")
            continue
        for role, ir in (("partial", partial), ("combine", combine)):
            where = "engine" if recorded(ir) else "refused: " + "; ".join(
                getattr(ir, "streaming_refusals", None) or ["no obstacle"])[:200]
            graphs.append((f"g{g.index}.{role}", ir, where, _split_check(role)))
    if not graphs:
        notes.append("no solo graph of the program has a pair built from its loop state")
    options = {"force_fusion": True} if sc is not None and sc.fusion == "force" else {}
    return graphs, options, notes, info


def _split_check(role):
    """Check (g) of a split stage, on each tiling that builds: ``(tiled) -> problem | None``.

    partial  where the keys of a split are looped, the max it returns is the carried max:
             an accumulator of that loop behind the sums it repairs, started at the most
             negative float32, snapshotted by one lag copy, and read by its output through
             the loop; where a tile holds the split, the flat graph's own max. Either way
             it returns the max first, then the two sums, all float32.
    combine  where it loops over the splits, the loop is plain (it weights the partials it
             loads: ``streaming._weighted_slices``).
    """
    from cleave.ir.graph import AccumEdge
    from cleave.ir.ops import NEG_FLT_MAX
    from cleave.runtime.reference import output_order

    def check(tiled):
        found = sites(tiled)
        tiers = {o.name: t for t, o in graph_tiers(tiled) if o is not None}
        if role == "combine":
            kinds = {s.tier: tiers[s.tier].loop_kind for s in found}
            loose = {k: v for k, v in kinds.items() if v != "serial"}
            return f"the loop over the splits is {loose}" if loose else None
        outs = output_order(tiled)
        if [(o.boundary.role, str(o.get_dtype(0))) for o in outs] != [
                ("shift", "float32"), ("", "float32"), ("", "float32")]:
            return "the partial does not return its float32 max first, then two float32 sums"
        (edge,) = outs[0].inputs
        if not found:
            return "the returned max is carried" if isinstance(edge, AccumEdge) else None
        (site,) = found
        tier = tiers[site.tier]
        (full,) = [n for n in tier.nodes if n.name == site.full]
        if not (isinstance(edge, AccumEdge) and edge.accum_node is full):
            return "the returned max is not the carried one"
        if full not in tier.accum_nodes or full.accum_init != NEG_FLT_MAX:
            return "the max is not carried from the most negative float32"
        if [n.name for n in tier.lag_nodes] != [f"{full.name}_prev"]:
            return f"lag copies {[n.name for n in tier.lag_nodes]}"
        if sorted(a.name for a in tier.accum_scale) != sorted(site.repaired) \
                or len(site.repaired) != 2:
            return f"repairs {sorted(a.name for a in tier.accum_scale)}"
        return None

    return check


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
    """For the duration of a ``with``: ``apply_tiling`` builds the loop it always builds,
    and the records are taken off the TILED copy instead of being assembled -- the plain
    twin of a kernel, the same loop without the snapshot and the repairs.

    The twin reads its running value inside the loop with no snapshot and nothing
    repaired, which the codegen refuses in a kernel (``codegen.validate``: such a value is
    meaningful only as carried state). The twin is only lowered, to compare with the
    assembled kernel, never run: the rolling-state checks are off inside the ``with``."""

    def __enter__(self):
        from cleave.codegen.validate import GraphValidator

        self.assemble = streaming.assemble
        self.validator = GraphValidator
        self.rolling = GraphValidator._validate_rolling_state

        def plain(tiled, flat_graph=None):
            for _, _, node in recorded(tiled):
                node.streaming = None
            return tiled

        streaming.assemble = plain
        GraphValidator._validate_rolling_state = lambda self, graph, where, is_main: None

    def __exit__(self, *exc):
        streaming.assemble = self.assemble
        self.validator._validate_rolling_state = self.rolling


def _lower(code, device_compile=False):
    try:
        lower_to_cuda(code, device_compile=device_compile)
        return "ok"
    except Exception as e:  # noqa: BLE001 - the verdict is the result
        return _first(e)


def census(ir, search, options, n_lower, nvcc, focus=None, check=None):
    options = options or CompilerPassOptions()
    problems, rows = [], []
    force = search.get("force_fusion", False)
    cands = list(search_tiling(ir, **{"topk": 1000, **search}))
    if not cands and not force:  # fusion="auto": retry forced, as pipeline.schedule does
        search = {**search, "force_fusion": True}
        cands = list(search_tiling(ir, **{"topk": 1000, **search}))
    accept = streaming_accept(ir)
    assert accept is not None, "census of a graph without a streaming obstacle"
    for i, cand in enumerate(cands):
        label = (f"#{i} tile={list(cand.output_tile)} rstep={_rsteps(cand)} "
                 f"threads={cand.n_threads}")
        row = dict(label=label, accepted=bool(accept(cand)))
        try:  # (b)
            tiled = apply_tiling(cand)
            row["tiling"] = "looped" if sites(tiled) else "one tile"
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
                ("ScheduleRejected", "StreamingRefused")):
            problems.append(f"(a) {label}: refused candidate failed as {row['build']}")
        rows.append((cand, row))

    kept = [(c, r) for c, r in rows if r["build"] == "ok"]
    for cand, row in kept:  # (g): a split stage
        problem = None if check is None else check(apply_tiling(cand))
        if problem is not None:
            problems.append(f"(g) {row['label']}: {problem}")
    for cand, row in kept:  # (e), and the plain twin's text
        row["staged"] = _staged(_passes(apply_tiling(cand), options))
        with _Plain():
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
            problems.append(f"(c) {row['label']}: assembled {row['lower']} | plain "
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
    print(f"{name}: {len(rows)} raw candidates, {sum(r['accepted'] for r in rows)} accepted, "
          f"{sum(r['build'] == 'ok' for r in rows)} build"
          + (f"  [records: {records}]" if records else ""))
    for text, n in tiling.most_common():
        print(f"    {n:4d} x {text[:150]}")
    if lowered:
        print(f"    lowering (assembled, plain) over {len(lowered)} kernels: "
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
    """Census one job: ``(rows, problems, unchecked, counts)``.

    ``problems`` are checks that failed; ``unchecked`` is what the job was there to check
    and could not: neither list empty means the job is not ``ok``.
    """
    unchecked, counts = [], None
    if job.startswith("split:"):
        graphs, search, notes, info = _split_graphs(job)
        options, nvcc, focus = None, None, None
        unchecked += notes
        for line in info:
            print(f"    {line[:300]}")
    elif job.startswith(("demo:", "program:")):
        graphs, search, notes, counts = _searched_graphs(job)
        graphs = [g + (None,) for g in graphs]
        options, nvcc, focus = DEMOS[job[5:]] if job.startswith("demo:") else (None,) * 3
        unchecked += notes
    else:
        build, search, options, nvcc, focus = (FIXTURES[job] + (None,))[:5]
        irs = build()
        graphs = [("" if len(irs) == 1 else str(k), ir, "by hand: the engine's rule", None)
                  for k, ir in enumerate(irs)]
    all_rows, all_problems = [], []
    for label, ir, records, check in graphs:
        name = f"{job}[{label}]" if label else job
        if not recorded(ir) and not getattr(ir, "streaming_refusals", None):
            unchecked.append(f"graph {name}: no streaming obstacle")
            continue
        rows, problems = census(ir, dict(search), options, n_lower,
                                nvcc if with_nvcc else None, focus, check)
        summarize(name, rows, problems, records)
        all_rows += [dict(r, graph=label) for r in rows]
        all_problems += problems
    if counts is not None:  # (f)
        new, old = counts
        accepted = sum(r["accepted"] for r in all_rows)
        print(f"    (f) plans: {new} searched, {old} with the hand-derived pass, "
              f"{accepted} accepted by the census")
        if job in COUNTS_DIFFER:
            *expected, why = COUNTS_DIFFER[job]
            print(f"    (f) by design: {why}")
            if [new, old] != expected:
                all_problems.append(f"(f) {new} and {old} plans where {expected[0]} and "
                                    f"{expected[1]} are expected")
        elif new != old:
            all_problems.append(f"(f) {new} plans where the hand-derived pass keeps {old}")
        if new != accepted:
            all_problems.append(f"(f) {new} plans but {accepted} candidates accepted")
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
    return all_rows, all_problems, unchecked, counts


def jobs() -> list:
    return (list(FIXTURES) + [f"demo:{d}" for d in DEMOS]
            + [f"program:{p}" for p in PROGRAM_JOBS]
            + [f"split:{s}" for s in list(SPLITS) + list(PROGRAM_SPLITS)])


def main(argv):
    out, job = Path(argv[1]), argv[2]
    if job == "list":
        print("\n".join(jobs()))
        return 0
    n_lower = argv[argv.index("--lower") + 1] if "--lower" in argv else "8"
    rows, problems, unchecked, counts = run(job, n_lower, "--nvcc" in argv)
    for line in unchecked:
        print("    NOT CHECKED", line[:300])
    for r in rows:
        r.pop("code", None), r.pop("plain_code", None)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{job.replace(':', '_')}.json").write_text(json.dumps(
        dict(rows=rows, problems=problems, unchecked=unchecked, counts=counts), indent=1) + "\n")
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
