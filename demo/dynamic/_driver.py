"""Shared driver for the dynamic (paged/ragged/MLA) attention demos.

Each demo in this directory names one operator and drives it through

    dense attention graph -> superoptimizer search -> tiling search
      -> paging pass (the dynamic access layer) -> codegen -> compile -> run

including the two MLA operators, whose latent body -- two score matmuls summed, the
compressed cache reused as V -- is searched from the same collapsed single-block form the
plain body uses.

The dense stages never see a page table or a per-request length; they tile plain
attention against a proxy KV extent. The paging pass then replaces that proxy with a
runtime trip count and symbolic grid extents, so one compiled kernel serves every request
shape. See ``cleave.paging`` for why the proxy is sound.

With ``--split FACTORS`` a decode operator is searched a second time with its KV axis
splittable, and the split-K pairs the superoptimizer proposes -- a partial body over one KV
slice, a combine summing the slices -- are wrapped by ``cleave.paging.apply_paging_split``
into a two-kernel split-KV module at the tiling the search chose for each half. Where the
superoptimizer proposes no split-K form (it never does for the MLA body, whose two score
matmuls are summed before the softmax) the pairs are derived from the single-kernel tilings
instead and labelled as such. Every pair is one more candidate: it is validated like the
rest and wins only if it is faster.

Every candidate is then validated against the operator's captured workloads and timed, and
the fastest correct one wins -- the same validate/benchmark/pick shape the static demos use.
The operator's checked-in expected kernel is entered as one more candidate, because it is a
**codegen** oracle and not a claim about speed: measured, a searched tiling beats it on
``gqa_paged_decode_h32_kv4_d128_ps1``, and on the two ``gqa_paged_prefill_causal_*``
operators it is not reachable by the search at all.

Each demo is its own process, which is the natural shape for a demo but also sidesteps a
real defect: cleave memoises its shape analysis on ``PrimFuncNode`` methods with
``lru_cache`` while hashing those nodes on name and type alone, so two dense bodies with
identically named nodes share cache entries and the second search reads the first one's
tile space. ``_searched_candidates`` clears those caches so that running several operators in one
interpreter gives the same answers as running them apart; see its docstring.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import cleave.superoptimizer as csopt
from tests.graph.handwritten._test_utils import EXPECTED_KERNELS_DIR
from cleave.codegen import TilelangCodeGen
from cleave.integration import graph_structure_to_cleave
from cleave.paging import apply_paging
from cleave.schedule import apply_tiling, search_tiling

DTYPE = "bfloat16"


def _dense_decode_graph(query_rows, head_dim, kv_proxy, split=False):
    """The dense body: one query tile of a head group attending over a KV window.

    ``kv_proxy`` stands in for a length only known per request. The search is extent
    agnostic (its own "symbolic" graph shrinks every dim to a small integer for
    verification), and the paging pass overwrites the one place the number survives
    scheduling, so any multiple of the KV tile does.

    With ``split`` the KV axis of K and V is declared splittable, and the search may also
    return split-K pairs: a partial body over one KV slice plus a combine summing the
    slices -- the dense form of split-KV decode.
    """
    graph = csopt.new_tile_graph(split_enabled=[False, split, split])
    q = graph.create_tiled_input((query_rows, head_dim), (False, False))
    k = graph.create_tiled_input((kv_proxy, head_dim), (split, False))
    v = graph.create_tiled_input((kv_proxy, head_dim), (split, False))
    scores = graph.matmul(q, k, trans_B=True)
    scores = graph.div(scores, head_dim**0.5)
    weights = graph.exp(scores)
    denom = graph.reduction(weights, 1)
    graph.create_tiled_output(graph.matmul(graph.div(weights, denom), v))
    return graph


def _dense_mla_graph(query_rows, ckv, kpe, kv_proxy, split=False):
    """The dense latent body: one token's block of query heads over a KV window.

    The MLA analogue of ``_dense_decode_graph``. What differs is the algebra, not the shape:
    two score matmuls -- against the compressed cache and the RoPE cache -- summed, and a
    value matmul that reads the compressed cache again, because MLA has no separate V.
    Collapsed to a single block for the same reason the plain body is, that the paging pass
    writes the grid itself.

    The fold is ``1/sqrt(ckv)``, the conventional scale for the width this body reduces
    over, which is what ``read_dense_mla`` requires. No constant here could be the
    operator's real scale -- that comes from the pre-absorption qk width and a YaRN
    correction and arrives as a runtime input -- and measured, the constant changes nothing
    about what the search finds.
    """
    graph = csopt.new_tile_graph(split_enabled=[False, split, False, split])
    q_nope = graph.create_tiled_input((query_rows, ckv), (False, False))
    ckv_cache = graph.create_tiled_input((kv_proxy, ckv), (split, False))
    q_pe = graph.create_tiled_input((query_rows, kpe), (False, False))
    kpe_cache = graph.create_tiled_input((kv_proxy, kpe), (split, False))
    scores = graph.add(
        graph.matmul(q_nope, ckv_cache, trans_B=True),
        graph.matmul(q_pe, kpe_cache, trans_B=True),
    )
    weights = graph.exp(graph.div(scores, ckv**0.5))
    denom = graph.reduction(weights, 1)
    graph.create_tiled_output(graph.matmul(graph.div(weights, denom), ckv_cache))
    return graph


def _dense_graph(spec, tiling, kv_proxy: int = 4096, split: bool = False):
    """The dense body whose tiling the access layer for ``spec`` will read off."""
    from cleave.paging import MLASpec

    if isinstance(spec, MLASpec):
        return _dense_mla_graph(
            tiling["query_rows"], tiling["ckv"], tiling["kpe"], kv_proxy, split=split
        )
    return _dense_decode_graph(
        tiling["query_rows"], tiling["head_dim"], kv_proxy, split=split
    )


def _reason(error: BaseException) -> str:
    """A stable one-line label for an exception, for grouping a search's refusals.

    The type prefix is what makes the grouping honest -- an `AssertionError` from the access
    layer is a refusal, anything else is a bug that happens to have been caught. Keyed on the
    FIRST line: tilelang's messages put the varying shape on the last one, which splits a
    single root cause into a bucket per candidate.
    """
    lines = str(error).strip().splitlines()
    return f"{type(error).__name__}: {lines[0][:100]}" if lines else type(error).__name__


def _histogram(counts, label: str, *, top: int = 3) -> None:
    """Print the commonest reasons, and account for the rest rather than dropping them."""
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    for reason, count in ranked[:top]:
        print(f"  {label} x{count:<4} {reason}", flush=True)
    if len(ranked) > top:
        rest = ranked[top:]
        others = f"{len(rest)} further reasons" if len(rest) > 1 else "1 further reason"
        print(f"  {label} x{sum(count for _, count in rest):<4} ... across {others}",
              flush=True)


def run_registered(
    operator: str,
    *,
    topk: int = 400,
    checkpoint: str | None = None,
    workloads: bool = True,
    max_workloads: int = 6,
    dump_best: str | None = None,
    split_factors=(),
) -> None:
    """Search, wrap, validate, benchmark, and name the fastest kernel for one operator.

    The same shape as the static demos: two library calls, one to validate every candidate
    and one to time the survivors, then the winner. ``cleave.util.autotune`` is the dynamic
    counterpart of ``compile_and_run_candidates_parallel``/``benchmark_graphs`` -- a paged
    operator's reference cannot come from its dense graph, so it comes from the operator's
    captured workloads instead.

    The operator's checked-in expected kernel is entered as one more candidate rather than
    used as the answer. It is a **codegen** oracle -- that the pass still emits exactly the
    text `tests/graph/apply_paging/test_expected_kernels.py` asserts -- and it is not
    necessarily the fastest kernel here. Two of the nine confirm that: on
    ``gqa_paged_decode_h32_kv4_d128_ps1`` a searched tiling beats it, and on the two
    ``gqa_paged_prefill_causal_*`` operators the tuned KV tile of 16 is outside the search
    space entirely (the policy seeds its reduction step at 32 and only enlarges it), so it
    can only enter as the tuned row.
    """
    import torch

    from tests.graph.apply_paging._test_utils import (
        OPERATOR_CASES,
        operator_kernel_source,
    )
    from tests.graph.handwritten._paged_workloads import keyed_workloads
    from tests.graph.handwritten._test_utils import GATHER_TOLERANCE
    from cleave.util.autotune import (
        WorkloadCandidate,
        benchmark_over_workloads,
        compile_and_validate_over_workloads,
        format_ranking,
        pick_best,
        rank,
    )

    from cleave.paging import SPLIT_KINDS

    started = time.perf_counter()
    spec, tiling = OPERATOR_CASES[operator]
    print(f"=== {operator}", flush=True)
    print(f"access layer: kind={spec.kind!r}, tuned tiling {tiling}", flush=True)
    split_factors = tuple(int(s) for s in split_factors)
    if split_factors and spec.kind not in SPLIT_KINDS:
        # Not a refusal to record per candidate: the whole transformation is a decode one.
        print(
            f"split-KV: not applicable to the {spec.kind!r} layer (a prefill block already "
            "owns one query tile of one head; splitting its KV window is not built)",
            flush=True,
        )
        split_factors = ()

    # --- 1. what the search proposes ----------------------------------------------
    candidates = _searched_candidates(
        operator, spec, tiling, topk=topk, checkpoint=checkpoint
    )
    if split_factors:
        split_candidates = _searched_split_candidates(
            operator, spec, tiling, topk=topk, checkpoint=checkpoint,
            split_factors=split_factors,
        )
        if not split_candidates:
            # The superoptimizer proposes no split-K form of this body (measured: it never
            # does for the MLA body, whose two score matmuls are summed before the softmax).
            # The transformation is still exact, so derive the pairs from the single-kernel
            # tilings the layer accepted: the partial keeps that tiling, the combine takes
            # the tiling the GQA searches rank first. Labelled as derived so the report
            # says which half the search actually chose.
            split_candidates = _derived_split_candidates(spec, candidates, split_factors)
        candidates += split_candidates

    # --- 2. plus the tuned point, as a candidate and not as the answer -------------
    tuned_source = operator_kernel_source(operator)
    golden = EXPECTED_KERNELS_DIR / f"{operator}.py"
    reproduces = golden.is_file() and tuned_source == golden.read_text(encoding="utf-8")
    print(
        f"codegen oracle: the pass reproduces {golden.name} byte for byte: {reproduces}"
        + ("" if reproduces else "  <-- test_expected_kernels.py would fail"),
        flush=True,
    )
    searched_it = next((c for c in candidates if c.source == tuned_source), None)
    if searched_it is not None:
        searched_it.origin = "searched == tuned"
        searched_it.tuned = True
    else:
        candidates.append(
            WorkloadCandidate(
                label=_label(tiling),
                source=tuned_source,
                # Reference only: the search never proposed this kernel, so letting it win
                # would put a kernel the pipeline cannot produce under a heading that says
                # "best found". It is measured and ranked so the report can state the gap.
                origin="tuned (reference)",
                tuned=True,
                eligible=False,
            )
        )

    if not workloads:
        print(f"  ({len(candidates)} candidates; validation and timing skipped)")
        print(f"elapsed {time.perf_counter() - started:.1f}s\n")
        return

    # --- 3. validate every candidate against every captured workload --------------
    success, errors = [], []
    compile_and_validate_over_workloads(
        candidates,
        keyed_workloads(operator),
        success,
        errors,
        tolerance=GATHER_TOLERANCE,
    )
    checked = max((c.checked for c in candidates), default=0)
    print(
        f"validated: {len(success)} of {len(candidates)} candidates correct over "
        f"{checked} captured workloads at 1e-2",
        flush=True,
    )
    rejected = [c for c in candidates if not c.correct]
    # Print why, once per distinct reason: tilelang's compile failures otherwise reach the
    # terminal only as unattached `Fatal:` lines from its C++ logger.
    _histogram(Counter(_reason(error) for _, error in errors), "error  ")
    if not success:
        raise SystemExit(
            f"no candidate for {operator} is correct over its captured workloads: "
            + "; ".join(f"{c.label}: {c.failure}" for c in rejected)
        )
    tuned = next(c for c in candidates if c.tuned)
    assert tuned.correct, (
        f"the tuned kernel for {operator} fails its own captured references "
        f"({tuned.failure}); that is a bug in the operator, not a benchmark result"
    )

    # --- 4. time the survivors and name the winner --------------------------------
    best, best_us, _ = benchmark_over_workloads(
        success, keyed_workloads(operator, keep=max_workloads)
    )
    print(format_ranking(success, rejected), flush=True)
    print(f"BEST {operator}: {best.label} at {best_us:.2f}us geomean", flush=True)
    if dump_best:
        # Named for the operator so a directory of winners can be handed straight to
        # `bench/bench_dynamic.py --cleave-dir`, which measures them against the
        # FlashInfer backends -- the claim this harness deliberately cannot make.
        out = Path(dump_best)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{operator}.py").write_text(best.source, encoding="utf-8")
        (out / f"{operator}.label").write_text(
            f"{best.label}\t{best.origin}\t{best_us:.3f}us\n", encoding="utf-8"
        )
        print(f"  wrote {out / (operator + '.py')}", flush=True)
        if split_factors:
            # The best single kernel as well, so the benchmark can put the split-KV
            # winner next to what the same search finds without splitting.
            solo = [c for c in success if not c.split]
            pairs = [c for c in success if c.split]
            best_solo = pick_best(rank(solo)) if solo else None
            best_pair = pick_best(rank(pairs)) if pairs else None
            for tag, cand in (("nosplit", best_solo), ("split", best_pair)):
                if cand is None:
                    continue
                (out / f"{operator}.{tag}.py").write_text(cand.source, encoding="utf-8")
                (out / f"{operator}.{tag}.label").write_text(
                    f"{cand.label}\t{cand.origin}\t{cand.geomean_us:.3f}us\n",
                    encoding="utf-8",
                )
                print(f"  wrote {out / (operator + '.' + tag + '.py')}", flush=True)
            if best_solo is not None and best_pair is not None:
                # --- 5. and the per-workload choice between the two ------------------
                plan = _calibrate_dispatch(operator, spec, best_pair, best_solo)
                (out / f"{operator}.hybrid.json").write_text(
                    json.dumps(plan, indent=1) + "\n", encoding="utf-8"
                )
                print(f"  wrote {out / (operator + '.hybrid.json')}", flush=True)

    for candidate in candidates:
        candidate.kernel = None
    torch.cuda.empty_cache()
    print(f"elapsed {time.perf_counter() - started:.1f}s\n")


def _calibrate_dispatch(operator, spec, best_pair, best_solo) -> dict:
    """Choose, per workload, between the best split pair and the best single kernel.

    A fixed split factor loses on the requests that have nothing to split -- one or two KV
    tiles, where the pair only adds its combine launch -- and those are most of the captured
    batch-1 workloads. FlashInfer's decode makes the same choice in its untimed ``plan()``.
    Here the rule is the longest request's tile count: split at or above ``min_tiles``, else
    run the single kernel. The threshold is calibrated on the two finalists' measured
    latencies over *every* captured workload (both are timed once more, on all of them), by
    picking the threshold with the best geomean; the request lengths come from the workload
    table, which synthesises them deterministically, so no tensor has to be rebuilt.

    Returns the plan the benchmark's wrapper applies: which kernel files, the threshold,
    where the indptr sits in the argument list and the tile it is measured in. The rule is
    a heuristic fitted to the captured distribution -- stated in the plan so the table can
    say so -- and it can never be wrong, only slower: both kernels compute the operator.
    """
    from tests.graph.handwritten._paged_workloads import (
        WORKLOADS,
        _decode_lengths,
        keyed_workloads,
    )
    from cleave.paging import MLASpec
    from cleave.util.autotune import (
        benchmark_over_workloads,
        geomean,
        param_index,
        workload_key,
    )

    benchmark_over_workloads([best_pair, best_solo], keyed_workloads(operator))
    kv_tile = int(best_pair.tiling.kv_tile)
    page_size = 1 if isinstance(spec, MLASpec) else int(spec.page_size)
    tiles = {}
    for axes in WORKLOADS[operator]:
        _batch, _pool, _n_idx, kv_lens = _decode_lengths(operator, axes, page_size)
        tiles[workload_key(axes)] = -(-max(kv_lens) // kv_tile)
    keys = [k for k in tiles if k in best_pair.latency_us and k in best_solo.latency_us]

    def hybrid(min_tiles):
        return geomean(
            [
                best_pair.latency_us[k] if tiles[k] >= min_tiles else best_solo.latency_us[k]
                for k in keys
            ]
        )

    # Smallest threshold among the best. Thresholds between two observed tile counts are
    # indistinguishable on this data, and the benchmark of record runs the *captured*
    # request lengths where these are synthesised, so the boundary goes just above the
    # last workload the single kernel won rather than just below the first the pair won:
    # an unseen length in between is more like the pair's side of the data than not.
    candidates = sorted(set(tiles[k] for k in keys) | {1, max(tiles.values()) + 1})
    scored = [(hybrid(t), t) for t in candidates]
    best_us, min_tiles = min(scored)
    split_us, solo_us = geomean([best_pair.latency_us[k] for k in keys]), geomean(
        [best_solo.latency_us[k] for k in keys]
    )
    print(
        f"dispatch: split when the longest request has >= {min_tiles} KV tiles of "
        f"{kv_tile}: hybrid {best_us:.2f}us vs split {split_us:.2f}us vs single "
        f"{solo_us:.2f}us over {len(keys)} workloads "
        f"(the pair runs {sum(tiles[k] >= min_tiles for k in keys)} of them)",
        flush=True,
    )
    return {
        "operator": operator,
        "split": f"{operator}.split.py",
        "single": f"{operator}.nosplit.py",
        "split_label": best_pair.label,
        "single_label": best_solo.label,
        "min_tiles": int(min_tiles),
        "kv_tile": kv_tile,
        "page_size": page_size,
        "indptr_index": param_index(best_pair.kernel, f"PlaceHolder_{spec.indptr_input}"),
        "geomean_us": {"hybrid": best_us, "split": split_us, "single": solo_us},
        "n_workloads": len(keys),
        "rule": "run the split pair when the longest request has at least min_tiles KV "
                "tiles (of kv_tile tokens; page_size pages per indptr unit), else the "
                "single kernel; calibrated on the demo's CUDA-graph timings",
    }


def _label(tiling) -> str:
    """A candidate's name: the tiling axes that actually vary between candidates."""
    bits = [f"kv={tiling['kv_tile']}", f"q={tiling['query_rows']}"]
    if tiling.get("n_threads"):
        bits.append(f"thr={tiling['n_threads']}")
    return " ".join(bits)


def _searched_candidates(operator, spec, tiling, *, topk, checkpoint) -> list:
    """Every distinct kernel the search proposes that the access layer can build.

    Raises rather than returning nothing. A run that wraps or compiles nothing has not shown
    the thing this half of the demo exists to show, and reporting ``0 wrappable`` next to a
    winner invites exactly the wrong conclusion.

    The cache clear is what makes that guard an invariant rather than a warning. cleave's
    shape and tensor-core analysis is memoised with ``functools.lru_cache`` on
    ``PrimFuncNode`` *methods*, and ``PrimFuncNode`` hashes on its name and type only -- so
    two different dense bodies whose nodes happen to share names (searched bodies all name
    theirs ``mma_3``/``mma_7``/``reduction_1_6``) collide, and the second search silently
    reads the first one's tile space. Measured: a decode search followed by a ragged one
    reports the decode's 88 tilings for the ragged body and wraps none of them.
    ``tests/conftest.py`` clears the same two caches between tests for the same reason.
    """
    from cleave.graph import PrimFuncNode
    from cleave.paging import MLASpec, read_dense_attention, read_dense_mla
    from cleave.util.autotune import WorkloadCandidate

    PrimFuncNode.get_space_dim.cache_clear()
    PrimFuncNode.infer_tensorcore_axis.cache_clear()

    graph = _dense_graph(spec, tiling)
    symbolic, symbol_map = csopt.create_symbolic_tile_graph(graph)
    config = symbolic.setup_search_config()
    config.search_thread = 16
    searched = symbolic.search_tile_graph(
        checkpoint or f"checkpoint_{operator}.json", False, False, config
    )

    candidates = []
    for optimized in searched:
        cleave_graph = graph_structure_to_cleave(
            optimized.get_graph_structure(),
            symbol_map=symbol_map,
            dtype=DTYPE,
            out_dtype=DTYPE,
        )
        candidates.extend(search_tiling(cleave_graph, topk=topk))

    read_dense = read_dense_mla if isinstance(spec, MLASpec) else read_dense_attention
    wrapped, refused, seen = [], Counter(), {}
    for candidate in candidates:
        try:
            tiled = apply_tiling(candidate)
            # The label comes from the recognizer the pass itself uses, so it describes what
            # was actually built rather than what was asked for.
            dense = read_dense(tiled)
            source = TilelangCodeGen().gen(apply_paging(tiled, spec))
        # Not just AssertionError: `apply_tiling` itself rejects some candidates with a
        # KeyError, and a demo that dies on a raw traceback mid-search reports nothing.
        except Exception as reason:  # noqa: BLE001 - a search reports, it does not stop
            refused[_reason(reason)] += 1
            continue
        # Distinct tilings can emit one kernel; rank the kernels, not the tilings.
        if source in seen:
            continue
        seen[source] = True
        wrapped.append(
            WorkloadCandidate(
                label=_label(
                    dict(
                        kv_tile=dense.kv_tile,
                        query_rows=dense.query_rows,
                        n_threads=dense.n_threads,
                    )
                ),
                source=source,
                tiling=dense,
            )
        )
    print(
        f"search: {len(searched)} dense graphs -> {len(candidates)} tilings "
        f"-> {len(wrapped)} distinct kernels the layer can build",
        flush=True,
    )
    _histogram(refused, "refused")
    if not wrapped:
        raise SystemExit(
            f"no searched tiling could be wrapped for {operator}; expected some to be. "
            "Either topk is too low for this body -- the policy ranks head-dim splits "
            "first and the layer refuses every one of them, so a 512-wide latent body "
            "needs several hundred candidates before an unsplit one appears -- or "
            "several operators were searched in one process (see the module docstring)."
        )
    return wrapped


def _derived_split_candidates(spec, solo, split_factors) -> list:
    """Split-KV pairs built from the single-kernel candidates' own tilings.

    The fallback for a body the superoptimizer cannot split: the same partial tiling the
    search found for the whole window, one split factor per candidate, and a combine
    tiling fixed at what the GQA split searches rank first -- one partial per step, a
    128-wide column tile, 128 threads -- which measured within 1% of every alternative.
    """
    from cleave.paging import CombineTiling, build_split_decode
    from cleave.util.autotune import WorkloadCandidate

    derived, seen = [], set()
    for cand in solo:
        dense = cand.tiling
        if dense is None:
            continue
        head_dim = getattr(dense, "head_dim", None) or getattr(dense, "head_dim_ckv")
        for split in split_factors:
            comb = CombineTiling(
                split=split, split_step=1, cols=min(128, head_dim), n_threads=128
            )
            try:
                source = TilelangCodeGen().gen(build_split_decode(spec, dense, split, comb))
            except Exception:  # noqa: BLE001 - a refusal, reported by the plain search
                continue
            if source in seen:
                continue
            seen.add(source)
            derived.append(
                WorkloadCandidate(
                    label=(
                        f"S={split} kv={dense.kv_tile} q={dense.query_rows} "
                        f"thr={dense.n_threads} comb={comb.split_step}x{comb.cols}@"
                        f"{comb.n_threads}"
                    ),
                    source=source,
                    origin="derived split-K",
                    split=split,
                    tiling=dense,
                )
            )
    print(
        f"split search proposed no split-K body; derived {len(derived)} split-KV pairs "
        f"from the {len(solo)} single-kernel tilings x {len(split_factors)} factors",
        flush=True,
    )
    return derived


def _searched_split_candidates(
    operator, spec, tiling, *, topk, checkpoint, split_factors
) -> list:
    """Every distinct split-KV kernel pair the search proposes that the layer can build.

    A second search of the same body with its KV axis declared splittable, kept apart from
    the plain one so the single-kernel candidates are exactly what they are without
    split-KV. Of what it returns, only the split-K pairs are taken -- a partial body over
    one KV slice and a combine summing the slices -- and each is tiled twice, once per half.
    The access layer reads the partial's tiling (query rows, KV tile, threads) and the
    combine's (split step, column tile, threads) and rebuilds both around the runtime page
    table, so the same tiling the search chose for the dense split pair is what runs.

    One candidate per distinct partial tiling and split factor, paired with the combine
    tiling the policy ranks first among those the layer accepts: measured, the combine's
    staging and step move the total by under 1%, so ranking every pairing would multiply
    the timing work for nothing.
    """
    from cleave.graph import PrimFuncNode
    from cleave.integration.graph_processing import is_split_graph, process_graphs
    from cleave.paging import (
        MLASpec,
        apply_paging_split,
        read_dense_combine,
        read_dense_mla_partial,
        read_dense_partial,
    )
    from cleave.util.autotune import WorkloadCandidate

    PrimFuncNode.get_space_dim.cache_clear()
    PrimFuncNode.infer_tensorcore_axis.cache_clear()

    graph = _dense_graph(spec, tiling, split=True)
    symbolic, symbol_map = csopt.create_symbolic_tile_graph(graph)
    config = symbolic.setup_search_config()
    config.search_thread = 16
    base = checkpoint or f"checkpoint_{operator}.json"
    searched = symbolic.search_tile_graph(
        str(Path(base).with_suffix("")) + "_split.json", False, False, config
    )
    structures = [g.get_graph_structure() for g in searched]
    split_structures = [gs for gs in structures if is_split_graph(gs)]
    read_partial = read_dense_mla_partial if isinstance(spec, MLASpec) else read_dense_partial

    wrapped, refused, seen = [], Counter(), {}
    n_tilings = 0
    for split in split_factors:
        bundles = process_graphs(
            split_structures,
            symbol_map=symbol_map,
            split_factors=[split],
            dtype=DTYPE,
            out_dtype=DTYPE,
            expand_broadcast_inputs=True,
        )
        for partial_graph, combine_graph in bundles:
            partials, combines = {}, []
            for cand in search_tiling(partial_graph, topk=topk):
                n_tilings += 1
                try:
                    tiled = apply_tiling(cand)
                    dense, found = read_partial(tiled)
                except Exception as reason:  # noqa: BLE001 - a search reports
                    refused[_reason(reason)] += 1
                    continue
                key = (dense.query_rows, dense.kv_tile, dense.n_threads, found)
                partials.setdefault(key, (dense, tiled))
            for cand in search_tiling(combine_graph, topk=topk):
                try:
                    tiled = apply_tiling(cand)
                    combines.append((read_dense_combine(tiled), tiled))
                except Exception as reason:  # noqa: BLE001
                    refused[_reason(reason)] += 1
            if not partials or not combines:
                continue
            comb, combine_tiled = combines[0]
            for (mq, bn, thr, found), (dense, partial_tiled) in partials.items():
                try:
                    pair = apply_paging_split([partial_tiled, combine_tiled], spec)
                    source = TilelangCodeGen().gen(pair)
                except Exception as reason:  # noqa: BLE001
                    refused[_reason(reason)] += 1
                    continue
                if source in seen:
                    continue
                seen[source] = True
                wrapped.append(
                    WorkloadCandidate(
                        label=(
                            f"S={found} kv={bn} q={mq} thr={thr} "
                            f"comb={comb.split_step}x{comb.cols}@{comb.n_threads}"
                        ),
                        source=source,
                        origin="searched split-K",
                        split=found,
                        tiling=dense,
                    )
                )
    print(
        f"split search: {len(searched)} dense graphs, {len(split_structures)} split-K "
        f"structures -> {n_tilings} partial tilings x {len(split_factors)} factors -> "
        f"{len(wrapped)} distinct split-KV kernel pairs the layer can build",
        flush=True,
    )
    _histogram(refused, "refused")
    if split_structures and not wrapped:
        raise SystemExit(
            f"the search proposed split-K structures for {operator} but no tiling of them "
            "could be wrapped; expected some to be (see the plain search's message)"
        )
    return wrapped
