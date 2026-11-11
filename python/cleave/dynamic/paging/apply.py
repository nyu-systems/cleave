"""The entry points: recognise the dense graph, check the tiling, build the layer."""

from __future__ import annotations

from ...ir.graph import Graph
from .builders import (
    _build_decode_combine,
    _build_mla_paged_decode,
    _build_mla_paged_prefill_causal,
    _build_paged_decode,
    _build_paged_decode_slab,
    _build_paged_prefill_causal,
    _build_ragged_prefill_causal,
)
from .constraints import _check_tiling
from .errors import NotDenseAttention, UnsupportedTiling
from .kinds import MLA_PAGED_PREFILL_CAUSAL, PAGED_DECODE_SLAB, PAGED_PREFILL_CAUSAL, RAGGED_PREFILL_CAUSAL
from .recognize import (
    read_dense_attention,
    read_dense_combine,
    read_dense_mla,
    read_dense_mla_partial,
    read_dense_partial,
)
from .specs import CombineTiling, MLASpec, PagingSpec


def apply_paging_split(graphs, spec) -> list:
    """Wrap a scheduled split-K pair in a paged decode access layer: ``[partial, combine]``.

    ``graphs`` is ``[partial, combine]`` as the dense stages tile a split-K decode body --
    the partial's KV operands sliced one split per block, the combine summing the partials
    and dividing. The result is what ``apply_paging`` is for a single graph: the same
    tiling decisions, wrapped in the runtime page table, per-request lengths and symbolic
    grid, emitted by ``TilelangCodeGen().gen`` as one prim_func of two kernels.
    """
    if len(graphs) != 2:
        raise NotDenseAttention(f"expected a [partial, combine] pair, got {len(graphs)} graphs")
    partial, combine = graphs
    spec.validate()
    if isinstance(spec, MLASpec):
        dense, split = read_dense_mla_partial(partial)
    else:
        dense, split = read_dense_partial(partial)
    comb = read_dense_combine(combine)
    if comb.split != split:
        raise NotDenseAttention(
            f"the partial makes {split} splits but the combine sums {comb.split}"
        )
    return build_split_decode(spec, dense, split, comb)


def apply_paging(graph: Graph, spec: PagingSpec) -> Graph:
    """Return a paged attention graph built around ``graph``'s tiling.

    ``graph`` must be a scheduled dense attention graph -- the output of
    ``apply_tiling`` on a Q@K^T/softmax/P@V source graph. An ``MLASpec`` selects the latent
    variant instead, whose dense body sums a compressed and a positional score matmul and
    reuses the compressed cache as V.
    """
    spec.validate()
    if isinstance(spec, MLASpec):
        mla = read_dense_mla(graph)
        _check_tiling(spec, mla)
        if spec.kind == MLA_PAGED_PREFILL_CAUSAL:
            return _build_mla_paged_prefill_causal(spec, mla)
        return _build_mla_paged_decode(spec, mla)
    dense = read_dense_attention(graph)
    _check_tiling(spec, dense)
    if spec.kind == RAGGED_PREFILL_CAUSAL:
        return _build_ragged_prefill_causal(spec, dense)
    if spec.kind == PAGED_PREFILL_CAUSAL:
        return _build_paged_prefill_causal(spec, dense)
    if spec.kind == PAGED_DECODE_SLAB:
        return _build_paged_decode_slab(spec, dense)
    return _build_paged_decode(spec, dense)


def build_split_decode(
    spec, dense, split: int, comb: CombineTiling
) -> list:
    """The split-KV pair ``[partial, combine]`` for a decode layer, from an explicit tiling.

    ``dense`` is the partial body's tiling (a ``DenseAttention``), ``split`` the number of
    KV splits and ``comb`` the combine kernel's tiling. Hand this list to
    ``TilelangCodeGen().gen`` and it emits one prim_func with both kernels, sharing the
    partial tensors. ``apply_paging_split`` is the entry point that reads all three off
    searched graphs; this one exists so the layer can be built and tested from numbers.
    """
    spec.validate()
    if comb.split != split:
        raise UnsupportedTiling(
            f"the combine tiling is for {comb.split} splits but the partial makes {split}"
        )
    _check_tiling(spec, dense, split=True)
    if isinstance(spec, MLASpec):
        partial = _build_mla_paged_decode(spec, dense, split=split)
        combine = _build_decode_combine(
            spec,
            heads=spec.num_qo_heads,
            group=spec.num_qo_heads,
            head_dim=dense.head_dim_ckv,
            out_dtype=dense.out_dtype,
            accum_dtype=dense.accum_dtype,
            comb=comb,
            grid_order=("batch", "head"),
            batch_dim=spec.batch_dim,
            rasterization=spec.rasterization,
        )
        return [partial, combine]
    if spec.kind == PAGED_DECODE_SLAB:
        partial = _build_paged_decode_slab(spec, dense, split=split)
    else:
        partial = _build_paged_decode(spec, dense, split=split)
    combine = _build_decode_combine(
        spec,
        heads=spec.num_qo_heads,
        group=spec.group,
        head_dim=dense.head_dim,
        out_dtype=dense.out_dtype,
        accum_dtype=dense.accum_dtype,
        comb=comb,
        grid_order=spec.grid_order,
        batch_dim=spec.batch_dim,
        rasterization=spec.rasterization,
    )
    return [partial, combine]
