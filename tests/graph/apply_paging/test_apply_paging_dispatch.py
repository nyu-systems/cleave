"""``apply_paging`` refuses a tiling its chosen layer cannot address.

The spec is checked by ``validate`` and the dense body by the recognizers; what is left is
the pairing of the two -- how many query rows a block owns against how many heads or tokens
the layer expects to find in them. That is the one thing neither side can check alone, and
every gate here guards a kernel that compiled and then read out of bounds or computed a
different operator.

The last two cases go the other way: the ragged layer's rows are tokens of a single query
head rather than (token, head-slot) pairs, so two tilings the paged layers must reject are
legitimate there, and this pins that they build and emit rather than merely validate.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tilelang")

from tests.graph.apply_paging._test_utils import (
    build_mla_paged_graph,
    build_paged_graph,
    paged_kernel_source,
)
from cleave.paging import (
    MLA_PAGED_DECODE,
    MLA_PAGED_PREFILL_CAUSAL,
    PAGED_DECODE_SLAB,
    PAGED_PREFILL_CAUSAL,
    RAGGED_PREFILL_CAUSAL,
    MLASpec,
    PagingSpec,
)

RAGGED = dict(kind=RAGGED_PREFILL_CAUSAL, causal=True, rasterization=None,
              cache_inputs=("K", "V"))


def test_rejects_a_prefill_tile_that_is_not_whole_head_groups():
    """A prefill block's M axis fuses (token, head slot), so its rows must divide by group.

    With 3 query heads per kv head a 16-row tile holds five tokens and a third of a sixth;
    the epilogue's row predicate counts tokens, so that last slot would be written from
    another token's accumulator.
    """
    spec = PagingSpec(
        num_qo_heads=24, num_kv_heads=8, kind=PAGED_PREFILL_CAUSAL, causal=True
    )
    with pytest.raises(AssertionError, match="does not divide into"):
        build_paged_graph(spec, query_rows=16, kv_tile=16)


def test_rejects_a_slab_that_straddles_two_pages():
    """A slab is addressed by ONE page-table entry plus an offset inside that page.

    At a 32-token page and a 64-token KV tile the tile spans two pages, whose ids need not
    be adjacent -- the emitted address would read 32 tokens of the right page and 32 of
    whatever happens to follow it in the pool.
    """
    spec = PagingSpec(
        num_qo_heads=32,
        num_kv_heads=8,
        kind=PAGED_DECODE_SLAB,
        page_size=32,
        pass_configs={},
    )
    with pytest.raises(AssertionError, match="straddle two pages"):
        build_paged_graph(spec, query_rows=16, kv_tile=64)


@pytest.mark.parametrize(
    "spec, n_threads",
    [
        (MLASpec(num_qo_heads=16), 256),
        (
            MLASpec(
                num_qo_heads=16,
                kind=MLA_PAGED_PREFILL_CAUSAL,
                causal=True,
                index_window_tiles=4,
                prefetch=False,
            ),
            128,
        ),
    ],
    ids=[MLA_PAGED_DECODE, MLA_PAGED_PREFILL_CAUSAL],
)
def test_rejects_an_mla_tile_that_is_not_one_tokens_heads(spec, n_threads):
    """Both MLA layers put one token's whole block of query heads in a single tile.

    The row gather stages ``num_qo_heads`` rows and the epilogue writes that many, neither
    of them predicated, so a taller tile is not a slower kernel but an out-of-bounds one:
    32 rows over 16 heads measured 37 invalid global reads per launch under
    compute-sanitizer. The check lived in the prefill builder only, which is how the decode
    layer went without it.
    """
    with pytest.raises(AssertionError, match="one token's whole block of heads"):
        build_mla_paged_graph(spec, query_rows=32, kv_tile=64, n_threads=n_threads)


def test_rejects_an_mla_prefetch_on_a_narrow_block():
    """The look-ahead read and the 256-thread block are one measured decision, not two.

    The BN x 512 compressed tile spills as a fragment at 128 threads, so a prefetch there
    buys a second live tile the block cannot hold. The thread count is searched -- the
    policy expands every candidate into a 128- and a 256-thread variant -- while the
    prefetch comes from the spec, so this pairing is a point the search really reaches and
    neither side can check alone. The parametrization above pairs 128 threads with the
    prefill layer, which does not prefetch, so only this reaches the decode layer's gate.
    """
    with pytest.raises(AssertionError, match="at least 256 threads"):
        build_mla_paged_graph(
            MLASpec(num_qo_heads=16), query_rows=16, kv_tile=64, n_threads=128
        )


def test_rejects_a_prefetch_on_the_mla_prefill_layer():
    """The prefill layer cannot prefetch at any thread count, so the spec cannot ask for it.

    Its tiles are one query block of ``num_qo_heads`` rows -- too little work to keep the 8
    warps the prefetch needs busy -- so the pairing the decode layer merely constrains is
    unavailable here outright. ``validate`` passes this spec (a wide window with a prefetch
    is a legal combination in general), which is exactly why the layer has to refuse it.
    """
    spec = MLASpec(
        num_qo_heads=16,
        kind=MLA_PAGED_PREFILL_CAUSAL,
        causal=True,
        index_window_tiles=4,
        prefetch=True,
    )
    spec.validate()
    with pytest.raises(AssertionError, match="set prefetch=False"):
        build_mla_paged_graph(spec, query_rows=16, kv_tile=64, n_threads=128)


@pytest.mark.parametrize(
    "query_rows,n_threads,expected",
    [
        # Below 64 rows, and at any height that is not a multiple of 64, the fragment does
        # not lay out: tilelang aborts in LayoutInference with "Layout infer conflict
        # between A_ptr and A_ptr" or in warp partitioning with "No valid warp partition",
        # neither of which names anything in the paging pass.
        (32, None, "64 rows or a multiple of it"),
        (16, None, "64 rows or a multiple of it"),
        (80, None, "64 rows or a multiple of it"),
        (96, None, "64 rows or a multiple of it"),
        # And a wider block cannot reconcile the two layouts at ANY height -- the tiling
        # policy expands every candidate into a 128- and a 256-thread variant, so this is a
        # point the search really reaches, at the shipped query tile.
        (64, 256, "needs a 128-thread block"),
        (128, 256, "needs a 128-thread block"),
    ],
)
def test_rejects_a_ragged_tiling_whose_fragment_cannot_lay_out(
    query_rows, n_threads, expected
):
    """The ragged layer keeps P in a fragment, which constrains the tile AND the block.

    Rejecting here is the difference between a searchable constraint and an unattributable
    crash inside tilelang.
    """
    spec = PagingSpec(num_qo_heads=32, num_kv_heads=8, **RAGGED)
    with pytest.raises(AssertionError, match=expected):
        build_paged_graph(spec, query_rows=query_rows, kv_tile=64, n_threads=n_threads)


def test_ragged_accepts_plain_mha():
    """A group of 1 is plain MHA, which only the ragged layer can express.

    The decode layers write a (group, head_dim) output tile, which squeezes to a rank the
    epilogue cannot handle at group 1; a ragged CTA owns a single query head and writes
    (rows, 1, head_dim), where the group reaches only its ``kv_head = h // group``.
    """
    spec = PagingSpec(num_qo_heads=8, num_kv_heads=8, **RAGGED)
    spec.validate()
    source = paged_kernel_source(spec, query_rows=64, kv_tile=64)
    # That divide is the only place the group reaches, so it is what "accepts" means here:
    # every query head is its own kv head, and the grid axis passes straight through.
    assert "kv_head=i1 // 1," in source


def test_ragged_accepts_a_query_tile_smaller_than_the_head_group():
    """A ragged tile's rows are tokens of one query head, so the group never bounds them.

    The paged layers reject this because their tile rows *are* head slots; here 64 rows
    under a group of 128 is simply 64 tokens of one head.
    """
    spec = PagingSpec(num_qo_heads=256, num_kv_heads=2, **RAGGED)
    spec.validate()
    source = paged_kernel_source(spec, query_rows=64, kv_tile=64)
    # 256 query heads over 2 kv heads: the group bounds nothing but this divide, and a tile
    # of 64 rows sits under it untouched.
    assert "kv_head=i1 // 128," in source
