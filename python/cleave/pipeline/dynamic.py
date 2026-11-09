"""``cleave.compile`` for paged programs: dense body search -> tilings -> ``apply_paging`` ->
validate and benchmark over workloads -> ``PagedKernel``."""

from __future__ import annotations

import dataclasses
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from ..core.errors import CleaveError, SpecError
from .compile import resolve_configs
from .kernel import Kernel, Stage
from .plans import Plan, PlanSet, Refusal
from .superopt import superoptimize


@dataclass(eq=False)
class PagedPlan(Plan):
    source: str = field(default="", repr=False)
    dense: object = None  # the DenseAttention the access layer read off the tiled graph
    origin: str = "searched"  # | "searched split-K" | "derived split-K"


def _label(dense, split=None, comb=None) -> str:
    text = f"kv={dense.kv_tile} q={dense.query_rows} thr={dense.n_threads}"
    if split:
        text = f"S={split} {text} comb={comb.split_step}x{comb.cols}@{comb.n_threads}"
    return text


def _codegen(graph) -> str:
    from ..codegen import TilelangCodeGen

    return TilelangCodeGen().gen(graph)


def _tilings(graph, topk, accept=None):
    from ..schedule import search_tiling

    return list(search_tiling(graph, topk=topk, accept=accept))


def schedule_paged(graphs, spec, config, *, split_graphs=(), filter_before_topk=False) -> PlanSet:
    """Tile every searched dense graph, page each tiling, keep one plan per distinct kernel.

    With ``config.split_k`` and no split pair from ``split_graphs`` (the search proposes no
    split-K MLA body), the pairs are derived from the single-kernel tilings."""
    from ..lower import to_ir
    from ..dynamic.paging import (SPLIT_KINDS, MLASpec, apply_paging, paging_accept, read_dense_attention,
                          read_dense_mla)
    from ..schedule import apply_tiling

    t0 = time.perf_counter()
    plans, refused, seen = [], [], set()

    def add(g, variant, tile, dense, source, rank, factor=None, comb=None, origin="searched"):
        if source in seen:
            return
        seen.add(source)
        label = f"g{g.index} {variant} {_label(dense, factor, comb)}"
        plans.append(PagedPlan(g.index, variant, "main", tile, label, rank, factor, g,
                               source=source, dense=dense, origin=origin))

    read = read_dense_mla if isinstance(spec, MLASpec) else read_dense_attention
    accept = paging_accept(spec) if filter_before_topk else None
    for g in graphs:
        ir = to_ir(g.structure, symbol_map=g.symbol_map, dtype=g.dtype, out_dtype=g.dtype)
        for rank, tile in enumerate(_tilings(ir, config.topk, accept)):
            try:
                tiled = apply_tiling(tile)
                dense = read(tiled)
                source = _codegen(apply_paging(tiled, spec))
            except Exception as e:  # noqa: BLE001 - a refusal is a result of the search
                refused.append(Refusal(g.index, "solo", "main", e))
                continue
            add(g, "solo", tile, dense, source, rank)
    split_k = tuple(config.split_k) if spec.kind in SPLIT_KINDS else ()
    if split_k:
        for g in split_graphs:
            _schedule_split(g, spec, config, add, refused)
        if not any(p.split_factor for p in plans):
            _derive_split(spec, [p for p in plans if not p.split_factor], split_k, add, refused)
    return PlanSet(plans, graphs, config, refused, seconds=time.perf_counter() - t0)


def _schedule_split(g, spec, config, add, refused):
    """Split-KV pairs: the partial tilings the layer reads, each with the first combine tiling."""
    from ..lower.process import to_bundles
    from ..dynamic.paging import (MLASpec, apply_paging_split, read_dense_combine, read_dense_mla_partial,
                          read_dense_partial)
    from ..schedule import apply_tiling

    read_partial = read_dense_mla_partial if isinstance(spec, MLASpec) else read_dense_partial

    for factor in config.split_k:
        variant = f"split-kv={factor}"
        try:
            bundles = to_bundles([g.structure], symbol_map=g.symbol_map, split_factors=[factor],
                                 dtype=g.dtype, out_dtype=g.dtype, expand_broadcast_inputs=True)
        except Exception as e:  # noqa: BLE001
            refused.append(Refusal(g.index, variant, "partial", e))
            continue
        for partial_graph, combine_graph in bundles:
            partials, combine = {}, None
            for tile in _tilings(partial_graph, config.topk):
                try:
                    tiled = apply_tiling(tile)
                    dense, found = read_partial(tiled)
                except Exception as e:  # noqa: BLE001
                    refused.append(Refusal(g.index, variant, "partial", e))
                    continue
                partials.setdefault((dense.query_rows, dense.kv_tile, dense.n_threads, found),
                                    (tile, dense, tiled))
            for tile in _tilings(combine_graph, config.topk):
                try:
                    tiled = apply_tiling(tile)
                    combine = (read_dense_combine(tiled), tiled)
                    break
                except Exception as e:  # noqa: BLE001
                    refused.append(Refusal(g.index, variant, "combine", e))
            if combine is None:
                continue
            for rank, ((*_, found), (tile, dense, tiled)) in enumerate(partials.items()):
                try:
                    source = _codegen(apply_paging_split([tiled, combine[1]], spec))
                except Exception as e:  # noqa: BLE001
                    refused.append(Refusal(g.index, variant, "main", e))
                    continue
                add(g, variant, tile, dense, source, rank, found, combine[0], "searched split-K")


def _derive_split(spec, solo, factors, add, refused):
    """Split-KV pairs on each single-kernel tiling, combine fixed at 1 step x 128 cols @ 128."""
    from ..dynamic.paging import CombineTiling, build_split_decode

    for p in solo:
        width = getattr(p.dense, "head_dim", None) or p.dense.head_dim_ckv
        for factor in factors:
            comb = CombineTiling(split=factor, split_step=1, cols=min(128, width), n_threads=128)
            try:
                source = _codegen(build_split_decode(spec, p.dense, factor, comb))
            except Exception as e:  # noqa: BLE001
                refused.append(Refusal(p.graph_index, f"split-kv={factor}", "main", e))
                continue
            add(p.graph, f"split-kv={factor}", p.tile, p.dense, source, p.rank, factor, comb,
                "derived split-K")


# ----------------------------------------------------------------------------- workloads
def resolve_workloads(attention, workloads=None, tolerance=None):
    """``(factory(keep), tolerance)``. ``workloads``: None (the registry's, else
    ``cl.workloads.paged_decode()``), a ``cl.workloads`` spec, or a zero-arg factory."""
    from ..dynamic import GATHER_TOLERANCE
    from ..frontend.paged import workloads as wl

    tol = tolerance or GATHER_TOLERANCE
    if workloads is None:
        op = _registry_op(attention)
        if op is not None:
            return (lambda keep=0: op.keyed_workloads(keep=keep)), tolerance or op.tolerance
        if attention.is_mla or attention.is_prefill:
            raise SpecError(f"{attention.program.name}: no captured {attention.kind} workloads; "
                            "pass workloads=")
        workloads = wl.paged_decode()
    if hasattr(workloads, "factory"):
        return (lambda keep=0: workloads.factory(attention, keep=keep)), tol
    if callable(workloads):
        return (lambda keep=0: workloads), tol
    raise TypeError(f"workloads must be a cl.workloads spec or a factory, got {workloads!r}")


def _registry_op(attention):
    """The registry operator of the same name and shape, if any."""
    from ..dynamic import OPERATORS

    a, op = attention, OPERATORS.get(attention.program.name)
    shape = (a.num_qo_heads, a.head_dim, a.pe_dim) if a.is_mla else (
        a.num_qo_heads, a.num_kv_heads, a.head_dim)
    if op is not None and op.is_mla == a.is_mla and tuple(op.dims[:3]) == shape \
            and op.page_size == a.page_size and op.spec.kind == a.kind:
        return op
    return None


def _tuned_reference(attention, spec, cands):
    """The registry's tuned kernel for ``spec``: a searched candidate if one has its source,
    else a reference-only (ineligible) one appended to ``cands``."""
    from ..runtime.autotune import WorkloadCandidate

    op = _registry_op(attention)
    if op is None or op.spec != spec:
        return None
    source = op.kernel_source()
    for c in cands:
        if c.source == source:
            c.origin, c.tuned = "searched == tuned", True
            return c
    t = op.tuned_tiling
    label = " ".join([f"tuned kv={t['kv_tile']} q={t['query_rows']}"]
                     + ([f"thr={t['n_threads']}"] if t.get("n_threads") else []))
    cands.append(WorkloadCandidate(label=label, source=source, origin="tuned (reference)",
                                   tuned=True, eligible=False))
    return cands[-1]


def calibrate_dispatch(pair, solo, factory, spec) -> dict:  # factory(keep) -> workloads_fn
    """Per workload, run the split pair or the single kernel by the longest request's KV tile
    count: the smallest ``min_tiles`` threshold with the best hybrid geomean over every workload."""
    from ..runtime.autotune import benchmark_over_workloads, geomean, param_index

    pair, solo = (dataclasses.replace(c, latency_us={}) for c in (pair, solo))
    index = param_index(solo.kernel, f"PlaceHolder_{spec.indptr_input}")
    unit, kv_tile, tiles = int(getattr(spec, "page_size", 1)), int(pair.tiling.kv_tile), {}

    def gen():
        for key, inputs, ref in factory(0)():
            ptr = inputs[index].tolist()
            tiles[key] = -(-max(b - a for a, b in zip(ptr, ptr[1:])) * unit // kv_tile)
            yield key, inputs, ref

    benchmark_over_workloads([pair, solo], gen)
    keys = [k for k in tiles if k in pair.latency_us and k in solo.latency_us]

    def hybrid(t):
        return geomean([(pair if tiles[k] >= t else solo).latency_us[k] for k in keys])

    # ties go to the smallest threshold: an unseen length between two observed tile counts
    # is more like the pair's side of the data
    best_us, min_tiles = min((hybrid(t), t) for t in
                             sorted({tiles[k] for k in keys} | {1, max(tiles.values()) + 1}))
    return {
        "split_label": pair.label, "single_label": solo.label, "min_tiles": int(min_tiles),
        "kv_tile": kv_tile, "page_size": unit, "indptr_index": index,
        "geomean_us": {"hybrid": best_us, "split": geomean([pair.latency_us[k] for k in keys]),
                       "single": geomean([solo.latency_us[k] for k in keys])},
        "n_workloads": len(keys), "split_runs": sum(tiles[k] >= min_tiles for k in keys),
        "rule": "run the split pair when the longest request has at least min_tiles KV tiles "
                "(indptr units x page_size tokens / kv_tile), else the single kernel",
    }


# ----------------------------------------------------------------------------- report
class DynamicReport:
    def __init__(self, plans: PlanSet, candidates, errors, *, workloads: int, timings: dict,
                 expected: str | None = None, tuned=None, dispatch: dict | None = None):
        self.plans, self.candidates, self.errors = plans, list(candidates), list(errors)
        self.n_workloads, self.timings, self.expected = workloads, dict(timings), expected
        self.tuned, self.dispatch = tuned, dispatch

    @property
    def correct(self):
        return [c for c in self.candidates if c.correct]

    @property
    def timed(self):
        return [c for c in self.correct if c.latency_us]

    @property
    def failures(self):
        return [c for c in self.candidates if not c.correct]

    def best(self, split: bool | None = None):
        """The fastest eligible timed candidate (``split``: only pairs / only single kernels)."""
        from ..runtime.autotune import pick_best, rank

        pool = [c for c in self.timed if c.eligible and (split is None or bool(c.split) == split)]
        return pick_best(rank(pool)) if pool else None

    def histogram(self) -> dict:
        return dict(Counter(f"{type(e).__name__}: {(str(e).strip().splitlines() or [''])[0][:100]}"
                            for _, e in self.errors).most_common())

    def summary(self) -> str:
        graphs, t = self.plans.graphs, self.timings
        rows = [("superoptimize", f"{len(graphs)} dense graph(s), "
                 f"{'cache hit' if graphs.from_cache else 'searched'}", t.get("superoptimize")),
                ("schedule", self.plans.summary().replace("\n  ", "  "), t.get("schedule")),
                ("validate", f"{len(self.correct)} of {len(self.candidates)} correct over "
                 f"{self.n_workloads} workloads", t.get("validate"))]
        rows += [("", f"  {n}x {k}", None) for k, n in self.histogram().items()]
        if self.expected:
            rows.append(("", f"  reproduces the expected kernel: {self.expected}", None))
        if self.tuned is not None:
            rows.append(("", f"  tuned kernel: {self.tuned.origin}"
                         + ("" if self.tuned.correct else f", FAILS: {self.tuned.failure}"), None))
        best = self.best()
        if best is not None:
            rows.append(("benchmark", f"best {best.geomean_us:.2f} us geomean  {best.label}",
                         t.get("benchmark")))
        if self.dispatch:
            d, g = self.dispatch, self.dispatch["geomean_us"]
            rows.append(("dispatch", f"split when >= {d['min_tiles']} KV tiles of {d['kv_tile']}: "
                         f"hybrid {g['hybrid']:.2f} us vs split {g['split']:.2f} vs single "
                         f"{g['single']:.2f} over {d['n_workloads']} workloads",
                         t.get("dispatch")))
        return "\n".join(f"{stage:<14}{text}" + ("" if s is None else f"  {s:.2f} s")
                         for stage, text, s in rows)

    def table(self, top: int = 10) -> str:
        from ..runtime.autotune import format_ranking

        return format_ranking(self.timed, self.failures)

    def to_dict(self) -> dict:
        return {"candidates": len(self.candidates), "correct": len(self.correct),
                "failures": self.histogram(), "timings_s": self.timings,
                "refused": len(self.plans.refused),
                "latency_us": {c.label: c.geomean_us for c in self.timed},
                "tuned": None if self.tuned is None else self.tuned.origin,
                "dispatch": self.dispatch}

    def __str__(self):
        return self.summary()


# ----------------------------------------------------------------------------- kernel
class PagedKernel(Kernel):
    """``k(*inputs)`` in the traced signature's order -> ``(out, lse)``; outputs are
    allocated per call. ``dispatch``: the split/single calibration (with ``dispatch_sources``)."""

    def __init__(self, attention, spec, source: str, compiled=None, *, dispatch=None,
                 dispatch_sources=None, **kw):
        self.attention, self.spec = attention, spec
        self.dispatch, self.dispatch_sources = dispatch, dict(dispatch_sources or {})
        inputs = attention.inputs
        super().__init__([Stage("main", source, [], [], kw.get("label", ""), kw.get("latency_us"))],
                         name=attention.program.name, input_names=[p.name for p in inputs],
                         input_shapes=[], **kw)
        self._compiled, self._slots = compiled, None

    def save(self, path):
        import pickle

        p = super().save(path)
        (p / "program.pkl").write_bytes(pickle.dumps(self.attention.program))
        meta = json.loads((p / "kernel.json").read_text())
        meta["paged"] = "program.pkl"
        if self.dispatch:
            files = {}
            for role, src in self.dispatch_sources.items():
                files[role] = f"dispatch.{role}.py"
                (p / files[role]).write_text(src, encoding="utf-8")
            (p / "dispatch.json").write_text(json.dumps(dict(self.dispatch, files=files), indent=1))
            meta["dispatch"] = "dispatch.json"
        (p / "kernel.json").write_text(json.dumps(meta, indent=1, default=str))
        return p

    @classmethod
    def _load(cls, p, m) -> "PagedKernel":
        import pickle

        attention = pickle.loads((p / m["paged"]).read_bytes()).attention()
        source = (p / m["stages"][0]["file"]).read_text()
        dispatch, sources = None, {}
        if m.get("dispatch"):
            dispatch = json.loads((p / m["dispatch"]).read_text())
            sources = {r: (p / f).read_text() for r, f in dispatch.pop("files", {}).items()}
        return cls(attention, attention.paging_spec(), source, label=m["label"],
                   variant=m["variant"], latency_us=m["latency_us"], dispatch=dispatch,
                   dispatch_sources=sources, meta=dict(m.get("meta") or {}, loaded_from=str(p)))

    def compile(self) -> "PagedKernel":
        if self._compiled is None:
            from ..runtime.autotune import _compiled

            self._compiled = _compiled(self.source)
        return self

    def _kernel_slots(self) -> list:
        """The traced inputs in the compiled kernel's argument order."""
        from ..runtime.autotune import param_index

        inputs = self.attention.inputs
        slots = [None] * len(inputs)
        for p in inputs:
            if not p.is_scalar:
                slots[param_index(self._compiled, f"PlaceHolder_{p.kernel_name}")] = p
        scalars = iter(p for p in inputs if p.is_scalar)
        return [s if s is not None else next(scalars) for s in slots]

    def __call__(self, *args, **named):
        import torch

        params = self.attention.program.params
        kept = self.attention.inputs
        if args and named:
            raise TypeError("pass inputs either positionally or by name, not both")
        if named:
            unknown = sorted(set(named) - {p.name for p in params})
            if unknown:
                raise TypeError(f"{self.name}: unexpected inputs {unknown}")
            bound = {p.name: named.get(p.name) for p in params}
        elif len(args) in (len(params), len(kept)):
            bound = dict(zip([p.name for p in (params if len(args) == len(params) else kept)], args))
        else:
            raise TypeError(f"{self.name}: expected inputs ({', '.join(p.name for p in params)})")
        for p in params:
            if p not in kept and bound.get(p.name) is not None:
                raise ValueError(f"{self.name}: page_size=1 reads no {p.name}; pass None")
            if p in kept and bound.get(p.name) is None:
                raise TypeError(f"{self.name}: missing input {p.name!r}")
        self.compile()
        if self._slots is None:
            self._slots = self._kernel_slots()
        sizes = {}
        for p in kept:
            if not p.is_scalar:
                for d, n in zip(p.spec.dims, bound[p.name].shape):
                    if hasattr(d, "name"):
                        sizes.setdefault(d.name, int(n))
        q = bound[self.attention.query.name]
        outs = [torch.empty([sizes[d.name] if hasattr(d, "name") else int(d) for d in dims],
                            dtype=dt.torch, device=q.device)
                for dims, dt in zip(self.attention.output_dims(), self.attention.output_dtypes())]
        self._compiled(*[bound[p.name] for p in self._slots], *outs)
        return outs[0] if len(outs) == 1 else tuple(outs)


# ----------------------------------------------------------------------------- compile
def compile_dynamic(fn, *, dtype=None, preset="default", superopt=None, schedule=None, eval=None,
                    split=None, hints=None, workloads=None, tolerance=None, max_workloads: int = 6,
                    filter_before_topk: bool = False, reference: bool = True, dispatch: bool = True,
                    progress: Callable[[str, object], None] | None = None, **unused) -> PagedKernel:
    """``split``: the KV dim (``cl.Split("kv_len", (4, 8))``) to also search split-KV pairs
    (decode kinds only); candidates are validated on every workload and timed on
    ``max_workloads`` of them. ``reference``: also measure the registry's tuned kernel (it
    cannot win). ``dispatch``: with split pairs, calibrate the split/single choice per workload."""
    from ..frontend.paged import trace_paged
    from ..dynamic.paging import SPLIT_KINDS
    from ..runtime.autotune import (benchmark_over_workloads, compile_and_validate_over_workloads,
                                 WorkloadCandidate)

    unused.pop("cache_kernels", None)  # paged kernels are not cached yet (searches are)
    if unused:
        raise TypeError(f"compile(): unexpected options for a paged program: {sorted(unused)}")
    note = progress or (lambda stage, value: None)
    program = trace_paged(fn, dtype, hints=hints)
    attention = program.attention()
    spec = attention.paging_spec()
    note("spec", spec)
    body = attention.dense_body()
    if split is None:
        split = program.split or None
    if split and spec.kind not in SPLIT_KINDS:
        note("split", f"split-KV is not built for the {spec.kind!r} layer; searching without it")
        split = False
    so, sc, ev = resolve_configs(body, preset=preset, superopt=superopt, schedule=schedule,
                                 eval=eval, split=split)
    if split is False:
        sc = sc.replace(split_k=())
    t0 = time.perf_counter()
    graphs = superoptimize(body, so.replace(split=()))
    split_graphs = superoptimize(body, so).filter(lambda g: g.is_split) if so.split else ()
    note("superoptimize", graphs)
    t1 = time.perf_counter()
    plans = schedule_paged(graphs, spec, sc, split_graphs=split_graphs,
                           filter_before_topk=filter_before_topk)
    note("schedule", plans)
    if not plans:
        raise CleaveError(f"{program.name}: the access layer refused every tiling\n{plans.summary()}")
    t2 = time.perf_counter()
    factory, tol = resolve_workloads(attention, workloads, tolerance or ev.tolerance)
    by_cand = {}
    for p in plans:
        c = WorkloadCandidate(label=p.label, source=p.source, origin=p.origin, tiling=p.dense,
                              split=p.split_factor)
        by_cand[id(c)] = (c, p)
    cands = [c for c, _ in by_cand.values()]
    tuned = _tuned_reference(attention, spec, cands) if reference else None
    success, errors = [], []
    compile_and_validate_over_workloads(cands, factory(0), success, errors, tolerance=tol)
    t3 = time.perf_counter()
    if any(c.eligible for c in success):
        benchmark_over_workloads(success, factory(max_workloads))
    t4 = time.perf_counter()
    expected = _expected_match(program.name, plans)
    report = DynamicReport(plans, cands, errors, workloads=max((c.checked for c in cands), default=0),
                           timings={"superoptimize": t1 - t0, "schedule": t2 - t1,
                                    "validate": t3 - t2, "benchmark": t4 - t3},
                           expected=expected, tuned=tuned)
    best = report.best()
    if best is None:
        note("evaluate", report)
        err = CleaveError(f"{program.name}: no searched candidate is correct over its workloads")
        err.report = report
        raise err
    pair, solo = report.best(split=True), report.best(split=False)
    sources = {}
    if dispatch and pair is not None and solo is not None:
        report.dispatch = calibrate_dispatch(pair, solo, factory, spec)
        report.timings["dispatch"] = time.perf_counter() - t4
        sources = {"split": pair.source, "single": solo.source}
    note("evaluate", report)
    plan = by_cand[id(best)][1]
    for c in cands:
        if c is not best:
            c.kernel = None
    return PagedKernel(attention, spec, best.source, best.kernel, label=best.label,
                       variant=plan.variant, latency_us=best.geomean_us, report=report,
                       plans=[plan], meta={"expected": expected}, dispatch=report.dispatch,
                       dispatch_sources=sources)


def _expected_match(name, plans) -> str | None:
    """The label of the plan whose source is the operator's checked-in expected kernel."""
    from ..dynamic import EXPECTED_KERNELS_DIR

    path = EXPECTED_KERNELS_DIR / f"{name}.py"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return next((p.label for p in plans if p.source == text), "none")
