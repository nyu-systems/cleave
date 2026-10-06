"""``PagingSpec`` rejects the specs it cannot honour, rather than emitting a wrong kernel.

Several of these are specs that used to validate and then produce a kernel that compiled
and returned garbage -- a page size the layer reads as tokens, a head group that does not
fit the query tile -- so each case names the wrong answer it prevents.

Two families of check are per-*kind* rather than global, and both need both directions
covered. A field the chosen layer does not read is refused rather than quietly ignored, so
that a caller who renames an input cannot get a kernel that ignored the rename; and the name
and extent namespaces contain only what the chosen layer actually emits, so the collisions
differ by kind and a decode spec is not rejected over an input it never declares.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tilelang")

from tests.graph.apply_paging._test_utils import OPERATOR_CASES
from cleave.paging import (
    PAGED_DECODE_SLAB,
    PAGED_PREFILL_CAUSAL,
    RAGGED_PREFILL_CAUSAL,
    MLASpec,
    PagingSpec,
)


def test_group_follows_from_the_head_counts():
    assert PagingSpec(num_qo_heads=32, num_kv_heads=8).group == 4
    assert PagingSpec(num_qo_heads=32, num_kv_heads=4).group == 8


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        # A block's query tile is one whole head group, so the counts must divide.
        (dict(num_qo_heads=32, num_kv_heads=5), "do not divide"),
        (dict(num_qo_heads=0, num_kv_heads=1), "must be positive"),
        # A single kv head leaves the cache with no head axis for the gather to index.
        (dict(num_qo_heads=32, num_kv_heads=1), "single kv head"),
        # A group of one squeezes the output tile to a rank the epilogue cannot write.
        (dict(num_qo_heads=16, num_kv_heads=16), "group of 1"),
        # page_size > 1 is a different access layer: this one would read a page count as
        # a token count and attend to one token per page.
        (dict(num_qo_heads=32, num_kv_heads=8, page_size=64), "slab access layer"),
        (dict(num_qo_heads=32, num_kv_heads=8, page_size=0), "only page_size=1"),
        # ... and the slab layer is the other side of that: it splits an address into a page
        # id and an offset inside the page, which one token per page leaves degenerate.
        (
            dict(num_qo_heads=32, num_kv_heads=8, kind=PAGED_DECODE_SLAB, page_size=1),
            "address split degenerate",
        ),
        # Two inputs sharing a name would collapse into one kernel parameter, and an
        # extent sharing the scale's name would make a float32 the grid bound.
        (dict(num_qo_heads=32, num_kv_heads=8, index_input="Q"), "input names collide"),
        (
            dict(num_qo_heads=32, num_kv_heads=8, cache_inputs=("K", "K")),
            "input names collide",
        ),
        (
            dict(num_qo_heads=32, num_kv_heads=8, scale_input="Q"),
            "input names collide",
        ),
        (
            dict(num_qo_heads=32, num_kv_heads=8, cache_inputs=("K",)),
            "expected a .K, V. cache pair",
        ),
        (
            dict(num_qo_heads=32, num_kv_heads=8, pages_dim="batch"),
            "symbolic extent names collide",
        ),
        (
            dict(num_qo_heads=32, num_kv_heads=8, scale_input="batch"),
            "symbolic extent names collide",
        ),
        # Values the ops or codegen would reject far from here, or not at all.
        (dict(num_qo_heads=32, num_kv_heads=8, warp_policy="bogus"), "warp policy"),
        (dict(num_qo_heads=32, num_kv_heads=8, rasterization="yes"), "must be an int"),
        (dict(num_qo_heads=32.0, num_kv_heads=8), "must be an int"),
        (
            dict(num_qo_heads=32, num_kv_heads=8, grid_order=("head", "head")),
            "grid_order must permute",
        ),
        # Causal masking is a property of the layer, not an independent flag: only the
        # causal prefill layer builds the diagonal bound and the row predicate.
        (
            dict(num_qo_heads=32, num_kv_heads=8, causal=True),
            "does not match the 'paged_decode' layer",
        ),
        (
            dict(num_qo_heads=32, num_kv_heads=8, kind="paged_prefill_causal"),
            "does not match the 'paged_prefill_causal' layer",
        ),
        (dict(num_qo_heads=32, num_kv_heads=8, kind="nonsense"), "unknown access layer"),
        # A field the chosen layer never reads, set to something other than its default. The
        # caller renamed an input or an extent and would otherwise see no error and get a
        # kernel that ignored the rename entirely.
        (
            dict(
                num_qo_heads=32,
                num_kv_heads=8,
                kind=RAGGED_PREFILL_CAUSAL,
                causal=True,
                grid_order=("head", "batch"),
            ),
            "does not read grid_order",
        ),
        (
            dict(
                num_qo_heads=32,
                num_kv_heads=8,
                kind=PAGED_PREFILL_CAUSAL,
                causal=True,
                batch_dim="nreq",
            ),
            "does not read batch_dim",
        ),
        (dict(num_qo_heads=32, num_kv_heads=8, tokens_dim="ntok"), "does not read tokens_dim"),
        (
            dict(num_qo_heads=32, num_kv_heads=8, query_indptr_input="QOFFS"),
            "does not read query_indptr_input",
        ),
        (
            dict(
                num_qo_heads=32,
                num_kv_heads=8,
                kind=RAGGED_PREFILL_CAUSAL,
                causal=True,
                index_input="PAGES",
            ),
            "does not read index_input",
        ),
        # The collision checks are per-kind too, and the prefill layers are the only ones
        # that emit a query indptr. Colliding it with the kv indptr used to produce a kernel
        # that silently lost a parameter and read the wrong one of the two.
        (
            dict(
                num_qo_heads=32,
                num_kv_heads=8,
                kind=PAGED_PREFILL_CAUSAL,
                causal=True,
                query_indptr_input="KV_INDPTR",
            ),
            "input names collide",
        ),
        # Likewise the extent namespace: the ragged layer is the only one with two token
        # axes, so these two names only take part in its grid.
        (
            dict(
                num_qo_heads=32,
                num_kv_heads=8,
                kind=RAGGED_PREFILL_CAUSAL,
                causal=True,
                tokens_dim="len_indptr",
            ),
            "symbolic extent names collide",
        ),
        (
            dict(
                num_qo_heads=32,
                num_kv_heads=8,
                kind=RAGGED_PREFILL_CAUSAL,
                causal=True,
                kv_tokens_dim="total_q",
            ),
            "symbolic extent names collide",
        ),
    ],
)
def test_rejects_unsupported_spec(kwargs, expected):
    with pytest.raises(AssertionError, match=expected):
        PagingSpec(**kwargs).validate()


def test_a_default_is_never_an_unread_field():
    """The unread-field gate fires on the value, not on the field being unread.

    Every layer leaves several optional fields at their defaults -- the ragged layer reads
    neither the page pool's extent nor the grid order -- so a spec that touches only what it
    needs has to pass however many fields its kind ignores.
    """
    PagingSpec(num_qo_heads=32, num_kv_heads=8, kind=RAGGED_PREFILL_CAUSAL, causal=True).validate()


@pytest.mark.parametrize(
    "operator",
    sorted(
        name for name, (spec, _) in OPERATOR_CASES.items() if not isinstance(spec, MLASpec)
    ),
)
def test_accepts_the_shipped_specs(operator):
    """Every tuned GQA operator's spec must pass the gate it ships behind.

    Keyed off the operator registry rather than a second copy of the specs, so a new assert
    that a shipped spec would trip fails here rather than only in the golden test.
    """
    OPERATOR_CASES[operator][0].validate()


def test_only_the_layers_that_emit_a_name_share_its_namespace():
    """The input namespace holds what the chosen layer emits, and nothing else.

    The collision cases in the table above are the negative half -- a name the layer *does*
    emit, set to another of its own names. This is the positive half, and it is what keeps
    the namespace from being quietly widened into a global one: ``indptr_input`` is read by
    every layer, ``KV_LAST_PAGE_LEN`` is the default of an input only the slab layer
    declares, and a decode kernel that has no page-length parameter cannot collide with it.

    It has to be reached by moving the always-read name onto the unread field's *default*:
    setting ``last_page_len_input`` itself is refused first, by the unread-field gate.
    """
    PagingSpec(num_qo_heads=32, num_kv_heads=8, indptr_input="KV_LAST_PAGE_LEN").validate()


def test_only_the_extents_the_layer_puts_on_the_grid_share_their_namespace():
    """And the same for the symbolic extents, whose leading one follows the kind.

    A decode grid is over requests and a prefill grid over query tokens, so ``batch_dim``
    takes part in one layer's extent namespace and ``tokens_dim`` in the other's. A decode
    spec must not be refused over ``total_q``, which its kernel never binds.
    """
    PagingSpec(num_qo_heads=32, num_kv_heads=8, indptr_dim="total_q").validate()
