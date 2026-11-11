"""Which tilings each access layer can be built at, checked before any builder runs."""

from __future__ import annotations

from ...core.errors import PagingRefusal
from ...ir.graph import Graph
from .errors import UnsupportedTiling
from .kinds import (
    MLA_PAGED_DECODE,
    MLA_PAGED_PREFILL_CAUSAL,
    PAGED_DECODE,
    PAGED_DECODE_SLAB,
    PAGED_PREFILL_CAUSAL,
    RAGGED_PREFILL_CAUSAL,
)
from .recognize import read_dense_attention, read_dense_mla
from .specs import MLASpec


def _check_tiling(spec, dense, split: bool = False) -> None:
    """Raise ``UnsupportedTiling`` unless ``spec``'s layer can be built at ``dense``'s tiling.

    The one copy of the tiling-vs-spec gates, shared by ``apply_paging``,
    ``build_split_decode`` (``split=True``) and ``can_page``.
    """
    if isinstance(spec, MLASpec):
        if split and spec.kind != MLA_PAGED_DECODE:
            raise UnsupportedTiling(
                f"split-KV is a decode transformation; got the {spec.kind!r} layer"
            )
        # Both MLA layers put one token's whole block of query heads in a single tile: the
        # row gather stages `num_qo_heads` rows and the epilogue writes that many, neither
        # of them predicated. A tile of any other height is not a slower kernel, it is an
        # out-of-bounds one -- measured, a 32-row tile over 16 heads makes 37 invalid
        # global reads per launch under compute-sanitizer, and 16 rows under 32 heads emits
        # an epilogue reading rows 16..31 of a 16-row fragment.
        if dense.query_rows != spec.num_qo_heads:
            raise UnsupportedTiling(
                f"the query tile is {dense.query_rows} rows but the operator has "
                f"{spec.num_qo_heads} heads; an MLA tile is one token's whole block of heads"
                + ("" if split else ", which it can only be when they are equal")
            )
        if spec.kind == MLA_PAGED_PREFILL_CAUSAL:
            # No prefetch, unlike the decode layer: a prefetch needs the wider block that
            # keeps the BN x ckv tile out of spill, and one query block of `mq` rows is too
            # little work to keep that many warps busy.
            if spec.prefetch:
                raise UnsupportedTiling(
                    "the MLA prefill layer's tiles are one query block, too little work for the wide "
                    "block a prefetch needs; set prefetch=False"
                )
        elif spec.prefetch and dense.n_threads < 256:
            raise UnsupportedTiling(
                f"the MLA prefetch needs at least 256 threads to keep the "
                f"{dense.kv_tile}x{dense.head_dim_ckv} cache tile out of spill, but the "
                f"schedule runs {dense.n_threads}; either re-tile or set prefetch=False"
            )
        return
    if split and spec.kind not in (PAGED_DECODE, PAGED_DECODE_SLAB):
        raise UnsupportedTiling(
            f"split-KV is a decode transformation; got the {spec.kind!r} layer"
        )
    if spec.kind == RAGGED_PREFILL_CAUSAL:
        # No `query_rows >= group` gate here: a ragged tile's rows are tokens of one query
        # head, not (token, head-slot) pairs, so the group never bounds them -- it reaches
        # only `kv_head = h // group`.
        #
        # Keeping P in a fragment (see the value matmul in `_build_ragged_prefill_causal`)
        # makes the P tile serve two masters: the mask/exp `T.Parallel` loop's layout and
        # the second gemm's operand-A layout. Measured at bn=64, those agree only for a
        # 128-thread block whose query tile is a multiple of 64 rows; everything else fails
        # deep inside tilelang with "Layout infer conflict between A_ptr and A_ptr" or "No
        # valid warp partition for T.gemm", neither of which names anything here. The
        # tiling policy really does propose the rejected points -- it expands every
        # candidate into a 128- and a 256-thread variant -- so this is the difference
        # between a searchable constraint and an unattributable crash.
        bm = dense.query_rows
        if not (bm >= 64 and bm % 64 == 0):
            raise UnsupportedTiling(
                f"the ragged layer needs a query tile of 64 rows or a multiple of it, got {bm}: it "
                "keeps P in a fragment for wgmma_rs, and any other height makes the mask loop's "
                "fragment layout and the value matmul's operand layout irreconcilable"
            )
        if dense.n_threads != 128:
            raise UnsupportedTiling(
                f"the ragged layer needs a 128-thread block, got {dense.n_threads}: with P in a "
                "fragment, a wider block cannot reconcile the mask loop's layout with the value "
                "matmul's operand layout at any query tile (measured at kv_tile 64)"
            )
        return
    # The epilogue writes `group` rows out of a `query_rows`-row accumulator, and the q
    # gather only stages `query_rows` of them. A group that does not fit reads past both.
    if dense.query_rows < spec.group:
        raise UnsupportedTiling(
            f"a block's query tile holds {dense.query_rows} rows but the head group needs "
            f"{spec.group}; re-tile with at least {spec.group} query rows"
        )
    # A prefill block's query tile fuses (token, head slot), so its rows have to divide
    # into whole head groups.
    if spec.kind == PAGED_PREFILL_CAUSAL and dense.query_rows % spec.group != 0:
        raise UnsupportedTiling(
            f"a prefill query tile of {dense.query_rows} rows does not divide into "
            f"head groups of {spec.group}; the M axis fuses token and head slot"
        )
    # One KV tile has to lie inside one page, or the slab would straddle two page-table
    # entries and a single scalar page id could not address it.
    if spec.kind == PAGED_DECODE_SLAB and spec.page_size % dense.kv_tile != 0:
        raise UnsupportedTiling(
            f"a page of {spec.page_size} tokens does not hold a whole number of "
            f"{dense.kv_tile}-token KV tiles, so a slab would straddle two pages"
        )


def can_page(tiled_graph: Graph, spec) -> bool:
    """Whether ``apply_paging(tiled_graph, spec)`` would build rather than refuse.

    Runs the same spec check, recognizer and tiling gates, but none of the builders.
    """
    try:
        spec.validate()
        read = read_dense_mla if isinstance(spec, MLASpec) else read_dense_attention
        _check_tiling(spec, read(tiled_graph))
    except PagingRefusal:
        return False
    return True


def paging_accept(spec):
    """A ``search_tiling(accept=...)`` filter: tile the candidate, then ``can_page`` it.

    Anything ``apply_tiling`` raises counts as a refusal too, as it would downstream.
    """
    from ...schedule import apply_tiling

    def accept(cand) -> bool:
        try:
            tiled = apply_tiling(cand)
        except Exception:  # noqa: BLE001 - a filter answers, it does not stop the search
            return False
        return can_page(tiled, spec)

    return accept
