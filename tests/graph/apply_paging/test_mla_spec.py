"""``MLASpec`` rejects the specs it cannot honour, rather than emitting a wrong kernel.

The companion of ``test_paging_spec.py`` for the latent-attention spec, which shares none
of ``PagingSpec``'s asserts: MLA has one kv head and therefore no head group to divide, two
caches of different widths, and a shared page-table window whose size is a tuning knob the
other layers do not have. Each case here names the wrong kernel it prevents.

Two of the checks are per-*kind* rather than global, and both directions are covered: the
name and extent namespaces only include what the chosen layer actually emits, so a decode
spec is not rejected over the query indptr it never declares, and a prefill spec that
collides on it is.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tilelang")

from tests.graph.apply_paging._test_utils import OPERATOR_CASES
from cleave.paging import MLA_PAGED_PREFILL_CAUSAL, MLASpec

# The shipped prefill tuning, as a base for the cases that need the prefill kind: its tiles
# are one query block, too little work for the wide block a prefetch needs.
PREFILL = dict(
    kind=MLA_PAGED_PREFILL_CAUSAL, causal=True, index_window_tiles=4, prefetch=False
)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        # A page holding more than one token would give the compressed cache a slot axis,
        # and the gather macro is written against (page, dim) with no slot to index.
        (dict(num_qo_heads=16, page_size=64), "only page_size=1 is expressible here"),
        # MLA puts every query head in one tile and `apply_paging` requires the tile to be
        # exactly that block, so the head count IS the matmul's M extent: it inherits the
        # tensor-core granularity. Measured, 2 heads fails on the fragment stride and 4 and
        # 8 on "M must be divisible by 16", so anything but a positive multiple of 16 emits
        # a kernel that cannot compile.
        (dict(num_qo_heads=1), "not a positive multiple of 16"),
        (dict(num_qo_heads=0), "not a positive multiple of 16"),
        (dict(num_qo_heads=8), "not a positive multiple of 16"),
        (dict(num_qo_heads=24), "not a positive multiple of 16"),
        (dict(num_qo_heads=16, index_window_tiles=0), "leaves no window to stage"),
        # The prefetch reads the NEXT tile's entries out of the current window, so at one
        # tile per window its guard (`win_off + bn < bn`, win_off always 0) is never true:
        # dead code that still forces the 256-thread block the prefetch is paired with.
        (dict(num_qo_heads=16, index_window_tiles=1), "no next tile"),
        # Values the ops or codegen would reject far from here, or not at all.
        (dict(num_qo_heads=16.0), "must be an int"),
        (dict(num_qo_heads=16, index_window_tiles=True), "must be an int"),
        (dict(num_qo_heads=16, rasterization="yes"), "must be an int"),
        (dict(num_qo_heads=16, query_inputs=("QN",)), "expected a .nope, pe. query pair"),
        (dict(num_qo_heads=16, cache_inputs=("CKV",)), "expected a .ckv, kpe. cache pair"),
        # Two inputs sharing a name would collapse into one kernel parameter. The query
        # indptr's collision is per-kind, so it lives in the named test below next to the
        # positive half it only means something beside.
        (dict(num_qo_heads=16, query_inputs=("QN", "QN")), "input names collide"),
        (dict(num_qo_heads=16, index_input="QN"), "input names collide"),
        (dict(num_qo_heads=16, scale_input="CKV"), "input names collide"),
        # An extent colliding with the scale makes a float32 the grid bound. The two
        # per-kind collisions -- the leading extent is a request count for decode and a
        # token count for prefill -- live in `test_the_leading_extent_name_follows_the_kind`
        # alongside the positive half they only mean something next to.
        (dict(num_qo_heads=16, scale_input="num_pages"), "symbolic extent names collide"),
        (dict(num_qo_heads=16, warp_policy="bogus"), "unsupported warp policy"),
        (dict(num_qo_heads=16, kind="nonsense"), "unknown MLA access layer"),
        # Causal masking is a property of the layer, not an independent flag: only the
        # prefill layer builds the diagonal bound and the row predicate.
        (dict(num_qo_heads=16, causal=True), "does not match the 'mla_paged_decode' layer"),
        (
            dict(num_qo_heads=16, kind=MLA_PAGED_PREFILL_CAUSAL, index_window_tiles=4,
                 prefetch=False),
            "does not match the 'mla_paged_prefill_causal' layer",
        ),
    ],
)
def test_rejects_unsupported_spec(kwargs, expected):
    with pytest.raises(AssertionError, match=expected):
        MLASpec(**kwargs).validate()


def test_a_one_tile_window_is_fine_without_the_prefetch():
    """The window/prefetch gate is about the pairing, not about the window on its own.

    One tile per window is a legal (if pointless) staging choice; what it cannot carry is
    the look-ahead read, which has no next tile inside the window to reach for.
    """
    MLASpec(num_qo_heads=16, index_window_tiles=1, prefetch=False).validate()


def test_only_the_prefill_layer_declares_a_query_indptr():
    """The name namespace is per-kind, and the decode layer's does not include the indptr.

    A decode spec is not refused for *colliding* on a name it never emits -- the collision
    check skips it -- but it is refused for setting it at all, because the decode builder
    would drop it. Both halves matter: the first keeps the namespace from being tightened
    into a global one, the second is the promise that an ignored field is an error.
    """
    with pytest.raises(AssertionError, match="does not read query_indptr_input"):
        MLASpec(num_qo_heads=16, query_indptr_input="Q_INDPTR").validate()
    MLASpec(num_qo_heads=16, query_indptr_input="Q_INDPTR", **PREFILL).validate()
    # And on the layer that does emit it, it shares the namespace: this is the rename that
    # used to pass and produce a kernel one parameter short, reading the kv indptr for both.
    with pytest.raises(AssertionError, match="input names collide"):
        MLASpec(num_qo_heads=16, query_indptr_input="KV_INDPTR", **PREFILL).validate()
    # The other way round, and the only way to reach it: an *always-read* name moved onto
    # the unread field's default. Setting `query_indptr_input` itself is refused by the gate
    # above before the collision check runs, so nothing else pins that a decode spec is free
    # to use the name of a parameter its kernel does not take.
    MLASpec(num_qo_heads=16, indptr_input="QO_INDPTR").validate()


def test_the_leading_extent_name_follows_the_kind():
    """A decode grid is over requests and a prefill grid over query tokens.

    So ``batch_dim`` takes part in the extent namespace for one kind and ``tokens_dim`` for
    the other -- and the one its layer does not read is refused rather than ignored, so
    neither can be quietly set to a colliding value.
    """
    with pytest.raises(AssertionError, match="does not read tokens_dim"):
        MLASpec(num_qo_heads=16, tokens_dim="num_pages").validate()
    with pytest.raises(AssertionError, match="does not read batch_dim"):
        MLASpec(num_qo_heads=16, batch_dim="num_pages", **PREFILL).validate()
    # The one each layer does read still takes part in its own namespace.
    with pytest.raises(AssertionError, match="symbolic extent names collide"):
        MLASpec(num_qo_heads=16, batch_dim="num_pages").validate()
    with pytest.raises(AssertionError, match="symbolic extent names collide"):
        MLASpec(num_qo_heads=16, tokens_dim="num_pages", **PREFILL).validate()
    # ... and the one it does not read is not in that namespace at all. Reached, as above, by
    # moving an always-read extent onto the unread one's default -- setting the unread field
    # is refused first -- so this is the only thing that keeps either namespace from being
    # widened to hold both names for both kinds.
    MLASpec(num_qo_heads=16, indptr_dim="total_q").validate()
    MLASpec(num_qo_heads=16, indptr_dim="batch", **PREFILL).validate()


@pytest.mark.parametrize(
    "operator",
    sorted(name for name, (spec, _) in OPERATOR_CASES.items() if isinstance(spec, MLASpec)),
)
def test_accepts_the_shipped_specs(operator):
    """Both tuned MLA operators' specs must pass the gate they ship behind.

    Keyed off the operator registry rather than a second copy of the specs, so a tuning
    change that trips a new assert fails here rather than in the golden test alone.
    """
    OPERATOR_CASES[operator][0].validate()
