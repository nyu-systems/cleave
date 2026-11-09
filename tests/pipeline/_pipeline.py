"""lower -> search_tiling -> apply_tiling -> passes -> codegen, hashed per candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests import _graphs as G

TOPK = 8
SPLIT_FACTOR = 4
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
STRUCTURES_DIR = Path(__file__).resolve().parents[1] / "integration" / "graph_structures"

# Concrete extents for the symbolic dims the stored structures use.
SYMBOL_MAPS = {
    "matmul": {4: 256, 8: 1536, 12: 2048},
    "rmsnorm_matmul": {4: 1024, 8: 6144, 12: 4096},
    # gqa: 8 is both B*H_Q and the split S_KV partial, so S_KV = 4 * B*H_Q.
    "gqa": {4: 16, 8: 64, 12: 64, 16: 256, 20: 64},
}

BUILDERS = {
    "matmul": lambda: G.matmul_graph(128, 128, 4096),
    "matmul_matmul_small_n": lambda: G.matmul_matmul_graph(4096, 64, 64),
    "matmul_rowsum_small_k": lambda: G.matmul_rowsum_graph(4096, 4096, 64),
    "rmsnorm_matmul": lambda: G.rmsnorm_matmul_graph(4096, 6144, 4096),
    "swiglu": lambda: G.swiglu_graph(4096, 14336, 4096),
    "lora": lambda: G.lora_graph(16, 4096, 4096, 16),
    "rowsum": lambda: G.rowsum_graph(4096, 4096),
    "mha_bhsd": lambda: G.mha_bhsd_graph(1, 32, 4096, 4096, 64),
    "gqa_bhsd": lambda: G.gqa_bhsd_graph(1, 32, 8, 4096, 4096, 64),
    "mqa_bhsd": lambda: G.mqa_bhsd_graph(1, 32, 1, 4096, 4096, 64),
    "matmul_splitk_partial": lambda: G.matmul_splitk_partial_graph(4, 128, 128, 1024),
    "matmul_splitk_combine": lambda: G.matmul_splitk_combine_graph(4, 128, 128),
    "mha_split_partial": lambda: G.mha_split_partial_graph(1, 32, 4, 4096, 1024, 64),
    "mha_split_combine": lambda: G.mha_split_combine_graph(32, 4, 4096, 64),
    "rmsnorm_matmul_split_partial": lambda: G.rmsnorm_matmul_split_partial_graph(
        4, 1024, 6144, 1024
    ),
    "rmsnorm_matmul_split_combine": lambda: G.rmsnorm_matmul_split_combine_graph(
        4, 1024, 6144, 4096
    ),
    "mla_decode_bhsd": lambda: G.mla_decode_bhsd_graph(1, 128, 4096, 512, 64),
}

MLA_DECODE = {"builder/mla_decode_bhsd"}


def fixture_names():
    paths = sorted(STRUCTURES_DIR.glob("*/before_*.graph"))
    return [f"structure/{p.parent.name}/{p.stem}" for p in paths] + [
        f"builder/{n}" for n in BUILDERS
    ]


def snapshot_path(name):
    return SNAPSHOT_DIR / (name.replace("/", "__") + ".json")


def _bundles(name):
    kind, _, rest = name.partition("/")
    if kind == "builder":
        return [[BUILDERS[rest]()]]
    from cleave.lower.process import to_bundles

    family, stem = rest.split("/")
    gs = json.loads((STRUCTURES_DIR / family / f"{stem}.graph").read_text())
    return to_bundles(
        [gs], symbol_map=SYMBOL_MAPS[family], split_factors=[SPLIT_FACTOR]
    )


def _build_code(cand, mla_decode_pass):
    # Mirrors cleave.runtime.compile.build_candidate_code.
    from cleave.codegen import passes
    from cleave.codegen import TilelangCodeGen
    from cleave.schedule import apply_tiling

    g = apply_tiling(cand)
    g = passes.annotate_args(g, is_main=True)
    g = passes.annotate_tinfo(g, is_main=True)
    g = passes.annotate_tc_layout(g)
    g = passes.annotate_fragment_layout_conflicts(g)
    if mla_decode_pass:
        g = passes.annotate_mla_accumulation_layout_conflict(g)
    return TilelangCodeGen().gen(g)


def _digest(cand, mla_decode_pass):
    try:
        code = _build_code(cand, mla_decode_pass)
    except Exception as e:  # a failing candidate is part of the snapshot
        return f"error:{type(e).__name__}"
    return hashlib.sha256(code.encode()).hexdigest()


def snapshot(name):
    from cleave.ir.graph import PrimFuncNode
    from cleave.schedule import search_tiling

    # Class-level lru_caches (pristine code) leak shape analysis across graphs.
    for fn in (PrimFuncNode.get_space_dim, PrimFuncNode.infer_tensorcore_axis):
        getattr(fn, "cache_clear", lambda: None)()
    mla = name in MLA_DECODE
    bundles = []
    for bundle in _bundles(name):
        stages = []
        for graph in bundle:
            try:
                cands = search_tiling(graph, topk=TOPK)
            except Exception as e:
                stages.append({"error": type(e).__name__})
                continue
            stages.append(
                {"count": len(cands), "sources": [_digest(c, mla) for c in cands]}
            )
        bundles.append(stages)
    return {"topk": TOPK, "bundles": bundles}


def dumps(snap):
    return json.dumps(snap, indent=1) + "\n"


if __name__ == "__main__":
    import sys
    import time

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else SNAPSHOT_DIR
    out.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[2:] or fixture_names():
        t = time.perf_counter()
        snap = snapshot(name)
        (out / snapshot_path(name).name).write_text(dumps(snap))
        n = [s.get("count", s.get("error")) for b in snap["bundles"] for s in b]
        print(f"{name}: {n} candidates, {time.perf_counter() - t:.1f}s", flush=True)
