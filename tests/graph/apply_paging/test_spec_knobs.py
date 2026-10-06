"""Every name a spec declares has to reach the emitted kernel, for every access layer.

The spec's input and extent names are the operator's declared signature -- the runtime binds
tensors to them by position and infers the symbolic extents from their shapes -- so a
builder that ignores a rename produces a kernel whose parameters mean something other than
what the caller asked for. Six builders write those names independently, and the shipped
specs happen to use the defaults everywhere, so nothing but this pins the plumbing.

Each layer is renamed only in the fields ``PagingSpec._FIELDS_READ`` says it reads; a field
the chosen layer does not read is refused by ``validate`` rather than silently dropped, and
that is tested in ``test_paging_spec.py``. The negative half matters as much as the positive
one: no default extent name may survive in the source, or a builder could be hardcoding it
alongside the renamed one.

The three knobs that are not names go the same way -- the pass configs, the block size and
the log-sum-exp output are each written by all six builders independently, and each of them
was pinned for the paged decode layer alone.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("tilelang")

from tests.graph.apply_paging._test_utils import build_mla_paged_graph, build_paged_graph
from cleave.codegen import TilelangCodeGen
from cleave.graph import OutputNode, PlaceHolderNode
from cleave.ops import Log2, MaskedLog2
from cleave.paging import (
    MLA_PAGED_DECODE,
    MLA_PAGED_PREFILL_CAUSAL,
    PAGED_DECODE,
    PAGED_DECODE_SLAB,
    PAGED_PREFILL_CAUSAL,
    RAGGED_PREFILL_CAUSAL,
    MLASpec,
    PagingSpec,
)

# Read by every layer, so renamed in every case. The pass configs are here too: they are read
# by every layer and every shipped spec but one leaves them at the default, so the value has
# to differ from that default (and from the empty dict the slab layer ships) for a builder
# that writes its own to be distinguishable from one that copies the spec's.
COMMON = dict(
    scale_input="softmax_scale",
    indptr_input="OFFSETS",
    indptr_dim="noff",
    rasterization=8,
    warp_policy="rows",
    pass_configs={"tl.disable_safe_memory_legalize": False},
)
# What ``build_tiled_candidate`` leaves the block size at when a case does not pin it. The
# tiling policy expands every candidate into a 128- and a 256-thread variant, so both values
# are real and a builder cannot assume either.
DEFAULT_N_THREADS = 128
# The MLA prefill layer adds two configs of its own on top of whatever the spec asked for: its
# query tile is a BufferRegion, which makes tilelang reach for TMA, and neither TMA's mbarrier
# protocol nor a warp-specialised producer/consumer split is something its nested serial
# macros on a 128-thread block can supply.
FORCED_PASS_CONFIGS = {
    MLA_PAGED_PREFILL_CAUSAL: {
        "tl.disable_tma_lower": True,
        "tl.disable_warp_specialized": True,
    },
}
# The extent names the dataclass defaults to. None of them may appear in a kernel built from
# a spec that renamed every extent the layer reads.
DEFAULT_DIMS = ("batch", "total_q", "total_kv", "num_pages", "len_indptr", "n_idx")

# One renamed spec per access layer, with the inputs the layer declares (in signature order)
# and the symbolic extents it puts on the grid.
CASES = {
    PAGED_DECODE: (
        PagingSpec(
            num_qo_heads=32,
            num_kv_heads=8,
            query_input="QUERY",
            cache_inputs=("KC", "VC"),
            index_input="PAGES",
            batch_dim="nreq",
            pages_dim="npage",
            index_dim="nidx",
            **COMMON,
        ),
        dict(query_rows=16, kv_tile=64),
        ["QUERY", "KC", "VC", "OFFSETS", "PAGES"],
        ["nreq", "npage", "noff", "nidx"],
    ),
    # The slab layer's sixth input is what recovers a token length from a page-unit indptr.
    PAGED_DECODE_SLAB: (
        PagingSpec(
            num_qo_heads=32,
            num_kv_heads=8,
            kind=PAGED_DECODE_SLAB,
            page_size=64,
            query_input="QUERY",
            cache_inputs=("KC", "VC"),
            index_input="PAGES",
            last_page_len_input="LASTLEN",
            batch_dim="nreq",
            pages_dim="npage",
            index_dim="nidx",
            **COMMON,
        ),
        dict(query_rows=16, kv_tile=64),
        ["QUERY", "KC", "VC", "OFFSETS", "PAGES", "LASTLEN"],
        ["nreq", "npage", "noff", "nidx"],
    ),
    # A prefill request has its own variable query length, hence a second CSR indptr, and its
    # leading extent is a token count rather than a request count.
    PAGED_PREFILL_CAUSAL: (
        PagingSpec(
            num_qo_heads=32,
            num_kv_heads=8,
            kind=PAGED_PREFILL_CAUSAL,
            causal=True,
            query_input="QUERY",
            cache_inputs=("KC", "VC"),
            index_input="PAGES",
            query_indptr_input="QOFFS",
            tokens_dim="ntok",
            pages_dim="npage",
            index_dim="nidx",
            **COMMON,
        ),
        dict(query_rows=64, kv_tile=16),
        ["QUERY", "KC", "VC", "QOFFS", "OFFSETS", "PAGES"],
        ["ntok", "npage", "noff", "nidx"],
    ),
    # Ragged keys are packed like the queries, so there is no page table and no page pool --
    # a second token axis takes their place.
    RAGGED_PREFILL_CAUSAL: (
        PagingSpec(
            num_qo_heads=32,
            num_kv_heads=8,
            kind=RAGGED_PREFILL_CAUSAL,
            causal=True,
            query_input="QUERY",
            cache_inputs=("KC", "VC"),
            query_indptr_input="QOFFS",
            tokens_dim="ntok",
            kv_tokens_dim="nkvtok",
            **COMMON,
        ),
        dict(query_rows=64, kv_tile=64),
        ["QUERY", "KC", "VC", "QOFFS", "OFFSETS"],
        ["ntok", "nkvtok", "noff"],
    ),
}
# MLA declares two query tensors and two caches instead of one and a pair.
MLA_CASES = {
    MLA_PAGED_DECODE: (
        MLASpec(
            num_qo_heads=16,
            query_inputs=("QNOPE", "QPE"),
            cache_inputs=("CKVC", "KPEC"),
            index_input="PAGES",
            batch_dim="nreq",
            pages_dim="npage",
            index_dim="nidx",
            **COMMON,
        ),
        dict(query_rows=16, kv_tile=64, n_threads=256),
        ["QNOPE", "QPE", "CKVC", "KPEC", "OFFSETS", "PAGES"],
        ["nreq", "npage", "noff", "nidx"],
    ),
    MLA_PAGED_PREFILL_CAUSAL: (
        MLASpec(
            num_qo_heads=16,
            kind=MLA_PAGED_PREFILL_CAUSAL,
            causal=True,
            index_window_tiles=4,
            prefetch=False,
            query_inputs=("QNOPE", "QPE"),
            cache_inputs=("CKVC", "KPEC"),
            index_input="PAGES",
            query_indptr_input="QOFFS",
            tokens_dim="ntok",
            pages_dim="npage",
            index_dim="nidx",
            **COMMON,
        ),
        dict(query_rows=16, kv_tile=64, n_threads=128),
        ["QNOPE", "QPE", "CKVC", "KPEC", "QOFFS", "OFFSETS", "PAGES"],
        ["ntok", "npage", "noff", "nidx"],
    ),
}
ALL_CASES = {**CASES, **MLA_CASES}


def _build(spec, tiling):
    build = build_mla_paged_graph if isinstance(spec, MLASpec) else build_paged_graph
    return build(spec, **tiling)


@pytest.mark.parametrize("kind", sorted(ALL_CASES))
def test_spec_knobs_reach_the_kernel(kind):
    spec, tiling, inputs, dims = ALL_CASES[kind]
    graph = _build(spec, tiling)
    # The source of the graph the assertions below read, not of a second build of it:
    # `paged_kernel_source` would run the whole pass again, and the two copies could then
    # disagree with nothing noticing.
    source = TilelangCodeGen().gen(graph)

    assert [n.name for n in graph.args if isinstance(n, PlaceHolderNode)] == [
        f"PlaceHolder_{name}" for name in inputs
    ]
    assert graph.dyn_dims == dims
    assert graph.scalar_params == [("softmax_scale", "float32")]
    assert graph.rasterization == 8
    # The pass configs are the spec's, plus whatever the layer itself has to force. A builder
    # that writes the shipped operator's dict instead sends tilelang a different compilation.
    assert graph.pass_configs == {
        **spec.pass_configs,
        **FORCED_PASS_CONFIGS.get(kind, {}),
    }
    # The block size comes off the searched schedule, not out of the spec or the builder.
    assert graph.n_threads == tiling.get("n_threads", DEFAULT_N_THREADS)

    for name in inputs:
        assert f"PlaceHolder_{name}: T.Tensor(" in source, (
            f"{name} is a declared parameter of the {kind!r} layer but the emitted kernel "
            "does not take it"
        )
    for dim in dims:
        assert f'{dim} = T.symbolic("{dim}")' in source, (
            f"{dim} is a declared extent of the {kind!r} layer but the emitted kernel does "
            "not bind it"
        )
    for dim in DEFAULT_DIMS:
        assert f'T.symbolic("{dim}")' not in source, (
            f"the {kind!r} layer emitted the default extent name {dim!r} even though the "
            "spec renamed every extent it reads"
        )
    # The two tuning knobs that are not names: an L2 swizzle and the warp split.
    assert "T.use_swizzle(8)" in source
    assert "GemmWarpPolicy.FullRow" in source


@pytest.mark.parametrize("kind", sorted(ALL_CASES))
def test_emit_lse_false_drops_the_second_output(kind):
    """A caller who does not want the log-sum-exp must not be given it, by any layer.

    The lse is a second output tensor and a second epilogue node, so emitting it unasked is
    not a harmless extra: the runtime binds outputs by position and would hand the kernel a
    buffer it did not allocate. All six builders write the flag's two consequences
    independently, and only the paged decode layer ever had them pinned.
    """
    spec, tiling, _, _ = ALL_CASES[kind]
    graph = _build(replace(spec, emit_lse=False), tiling)

    assert [n.name for n in graph.args if isinstance(n, OutputNode)] == ["OutputNode_0"]
    # Two node types, because the causal layers emit the row-predicated flavour of the same
    # epilogue node -- looking for only one of them would miss three of the six builders.
    assert not [n for n in graph.nodes if isinstance(n, (Log2, MaskedLog2))]
    # ... and it still emits: dropping an output must not leave the epilogue half-wired.
    assert TilelangCodeGen().gen(graph)
