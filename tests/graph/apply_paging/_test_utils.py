"""Shared helpers for the apply_paging tests.

Each test drives the real pipeline as far as codegen -- dense source graph, pinned
tiling, ``apply_tiling``, ``apply_paging`` -- and then asserts something about the
result. The tiling is pinned rather than searched because these tests are about the
access layer, and pinning is also what the apply_tiling tests do; which tilings the
policy *ranks* first is a separate question, tested under tests/schedule/.
"""

from __future__ import annotations

from tests._graphs import flash_attention_sd_graph, mla_attention_sd_graph
from tests.graph.apply_tiling._test_utils import build_tiled_candidate
from cleave.codegen import TilelangCodeGen
from cleave.paging import (
    MLA_PAGED_PREFILL_CAUSAL,
    PAGED_DECODE_SLAB,
    PAGED_PREFILL_CAUSAL,
    RAGGED_PREFILL_CAUSAL,
    MLASpec,
    PagingSpec,
    apply_paging,
)
from cleave.schedule import apply_tiling

DTYPE = "bfloat16"
ACCUM_DTYPE = "float32"

# The KV extent the dense stages tile against. It is a proxy for a length that is only
# known per request at runtime, and the paging pass replaces it with a runtime trip
# count, so its only lasting effect is which KV tile the policy considers legal. Any
# multiple of the tile works; 4096 is what the demos under demo/dynamic/ use.
KV_PROXY = 4096


def tiled_dense_attention(
    dense, query_rows, head_dim, kv_tile, *, steps=None, n_threads=None
):
    """``dense`` scheduled at a pinned tiling, ready for the paging pass.

    The reduction steps are keyed by node name and every other op gets an empty one, so a
    body carrying extra nodes still reaches the paging pass -- which is where a body the
    access layer cannot represent is supposed to be rejected, with an explanation.
    ``steps`` overrides the map for a body with extra *reductions* (MLA's positional score
    matmul), which do need one.

    ``n_threads`` pins the block size. The real pipeline searches it -- the policy expands
    every candidate into a 128- and a 256-thread variant -- rather than deriving it, so
    pinning it here is the same kind of pinning as the output tile and the reduction steps.
    """
    steps = steps or {
        # The score matmul reduces over the whole head dim: splitting it would make the
        # block recompute the scores, which is never what these operators want.
        "mma_0": {"v2": head_dim},
        # Both the value matmul and the row sum reduce over KV, one tile per iteration --
        # this is the number that becomes the loop's trip count.
        "mma_1": {"v2": kv_tile},
        "rowsum_0": {"v1": kv_tile},
    }
    rstep_by_name = {
        node.name: steps.get(node.name, {})
        for node in dense.nodes
        if not node.is_placeholder() and not node.is_output()
    }
    cand = build_tiled_candidate(dense, [query_rows, head_dim], rstep_by_name)
    if n_threads is not None:
        cand.n_threads = n_threads
    return apply_tiling(cand)


def build_paged_graph(
    spec, *, query_rows=16, head_dim=128, kv_tile=64, kv_proxy=KV_PROXY, n_threads=None
):
    """The scheduled, paged graph for ``spec`` at a pinned dense tiling."""
    dense = flash_attention_sd_graph(
        query_rows, kv_proxy, head_dim, dtype=DTYPE, out_dtype=DTYPE
    )
    return apply_paging(
        tiled_dense_attention(
            dense, query_rows, head_dim, kv_tile, n_threads=n_threads
        ),
        spec,
    )


def build_mla_paged_graph(
    spec,
    *,
    query_rows=16,
    ckv=512,
    kpe=64,
    kv_tile=64,
    kv_proxy=KV_PROXY,
    n_threads=None,
):
    """The scheduled, paged graph for an ``MLASpec`` at a pinned dense tiling."""
    dense = mla_attention_sd_graph(
        query_rows, kv_proxy, ckv, kpe, dtype=DTYPE, out_dtype=DTYPE
    )
    tiled = tiled_dense_attention(
        dense,
        query_rows,
        ckv,
        kv_tile,
        n_threads=n_threads,
        steps={
            # Both score matmuls reduce over the whole width they are given; only the KV
            # axis is stepped, and that step is what becomes the loop's trip count.
            "mma_0": {"v2": ckv},
            "mma_pe": {"v2": kpe},
            "mma_1": {"v2": kv_tile},
            "rowsum_0": {"v1": kv_tile},
        },
    )
    return apply_paging(tiled, spec)


def paged_kernel_source(spec, **kwargs) -> str:
    return TilelangCodeGen().gen(build_paged_graph(spec, **kwargs))


def mla_paged_kernel_source(spec, **kwargs) -> str:
    return TilelangCodeGen().gen(build_mla_paged_graph(spec, **kwargs))


# Every operator the access layer can build, with the spec that names its dynamic inputs
# and the dense tiling its tuned schedule was measured at. Each entry is verified to emit
# the operator's checked-in expected kernel byte for byte, which is what carries the tuned
# performance over to the searched pipeline -- see test_expected_kernels.py. The demos
# under demo/dynamic/ read the same table, so a demo and its test cannot drift apart.
OPERATOR_CASES: dict = {
    "gqa_paged_decode_h32_kv8_d128_ps1": (
        PagingSpec(num_qo_heads=32, num_kv_heads=8),
        dict(query_rows=16, kv_tile=64, head_dim=128),
    ),
    "gqa_paged_decode_h32_kv4_d128_ps1": (
        # kv4 puts the kv head on the leading grid axis; kv8 measured better the other way.
        PagingSpec(num_qo_heads=32, num_kv_heads=4, grid_order=("head", "batch")),
        dict(query_rows=16, kv_tile=64, head_dim=128),
    ),
    "gqa_paged_decode_h32_kv8_d128_ps64": (
        # A 64-token page holds a whole KV tile, so the tile is a contiguous slab and is
        # copied as a BufferRegion (which TMA can lower) rather than gathered row by row.
        # Bounds predication stays ON here: it is what predicates those slab copies.
        PagingSpec(
            num_qo_heads=32,
            num_kv_heads=8,
            kind=PAGED_DECODE_SLAB,
            page_size=64,
            grid_order=("head", "batch"),
            pass_configs={},
        ),
        dict(query_rows=16, kv_tile=64, head_dim=128),
    ),
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1": (
        PagingSpec(
            num_qo_heads=32, num_kv_heads=8, kind=PAGED_PREFILL_CAUSAL, causal=True
        ),
        # A prefill block's M axis fuses (token, head slot), and the narrow KV tile is
        # measured: the captured requests' causal windows are a handful of tokens.
        dict(query_rows=64, kv_tile=16, head_dim=128),
    ),
    "gqa_paged_prefill_causal_h32_kv4_d128_ps1": (
        PagingSpec(
            num_qo_heads=32, num_kv_heads=4, kind=PAGED_PREFILL_CAUSAL, causal=True
        ),
        dict(query_rows=64, kv_tile=16, head_dim=128),
    ),
    "gqa_ragged_prefill_causal_h32_kv8_d128": (
        # No page table: a ragged operator's keys are packed contiguously, so the KV tile
        # is an affine slice. That is what lets the loop pipeline, and the L2 swizzle pays
        # here (unlike the paged operators) because neighbouring blocks share keys.
        PagingSpec(
            num_qo_heads=32,
            num_kv_heads=8,
            kind=RAGGED_PREFILL_CAUSAL,
            causal=True,
            rasterization=None,
            cache_inputs=("K", "V"),
        ),
        dict(query_rows=64, kv_tile=64, head_dim=128),
    ),
    "gqa_ragged_prefill_causal_h32_kv4_d128": (
        PagingSpec(
            num_qo_heads=32,
            num_kv_heads=4,
            kind=RAGGED_PREFILL_CAUSAL,
            causal=True,
            rasterization=None,
            cache_inputs=("K", "V"),
        ),
        dict(query_rows=64, kv_tile=64, head_dim=128),
    ),
    "mla_paged_decode_h16_ckv512_kpe64_ps1": (
        # 256 threads and the look-ahead read are one decision, not two: the BN x 512
        # compressed tile spills as a fragment at 128 threads.
        MLASpec(num_qo_heads=16),
        dict(query_rows=16, kv_tile=64, n_threads=256, ckv=512, kpe=64),
    ),
    "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": (
        # A prefill tile is one query block of 16 rows -- too little work to keep 8 warps
        # busy -- so this one takes neither the wider block nor the prefetch.
        MLASpec(
            num_qo_heads=16,
            kind=MLA_PAGED_PREFILL_CAUSAL,
            causal=True,
            index_window_tiles=4,
            prefetch=False,
            rasterization=1,
            query_inputs=("Q_NOPE", "Q_PE"),
            cache_inputs=("CKV_CACHE", "KPE_CACHE"),
            index_input="KV_INDICES",
        ),
        dict(query_rows=16, kv_tile=64, n_threads=128, ckv=512, kpe=64),
    ),
}


def operator_kernel_source(operator: str) -> str:
    """The TileLang the pipeline emits for ``operator``, at its tuned tiling."""
    spec, tiling = OPERATOR_CASES[operator]
    build = mla_paged_kernel_source if isinstance(spec, MLASpec) else paged_kernel_source
    return build(spec, **tiling)


def build_operator_graph(operator: str):
    spec, tiling = OPERATOR_CASES[operator]
    build = build_mla_paged_graph if isinstance(spec, MLASpec) else build_paged_graph
    return build(spec, **tiling)
