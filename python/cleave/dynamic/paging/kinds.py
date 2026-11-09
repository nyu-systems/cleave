"""The access layers this package builds, and which spec fields each one reads."""

PAGED_DECODE = "paged_decode"
# Same operator as PAGED_DECODE, different access layer: at page_size > 1 a KV tile lies
# inside a single page, so the tile is a contiguous slab addressed by one page-table entry
# rather than BN gathered rows. That changes what the page table means (pages, not tokens),
# adds an input for the token length of the last page, and swaps the gathers for
# BufferRegions -- so it is its own kind.
PAGED_DECODE_SLAB = "paged_decode_slab"
PAGED_PREFILL_CAUSAL = "paged_prefill_causal"
RAGGED_PREFILL_CAUSAL = "ragged_prefill_causal"
KINDS = (
    PAGED_DECODE,
    PAGED_DECODE_SLAB,
    PAGED_PREFILL_CAUSAL,
    RAGGED_PREFILL_CAUSAL,
)
# The layers that mask against the diagonal.
CAUSAL_KINDS = (PAGED_PREFILL_CAUSAL, RAGGED_PREFILL_CAUSAL)
# The layers that read a page table; ragged keys are contiguous per request.
PAGED_KINDS = (PAGED_DECODE, PAGED_DECODE_SLAB, PAGED_PREFILL_CAUSAL)
# The layers whose page table is indexed in pages rather than tokens.
SLAB_KINDS = (PAGED_DECODE_SLAB,)

MLA_PAGED_DECODE = "mla_paged_decode"
MLA_PAGED_PREFILL_CAUSAL = "mla_paged_prefill_causal"
MLA_KINDS = (MLA_PAGED_DECODE, MLA_PAGED_PREFILL_CAUSAL)
SPLIT_KINDS = (PAGED_DECODE, PAGED_DECODE_SLAB, MLA_PAGED_DECODE)

# Which `PagingSpec` fields each layer reads, for the fields only some of them do. `validate`
# refuses a field the chosen layer would silently drop; everything not listed is read by
# every layer.
_OPTIONAL_FIELDS = (
    "batch_dim",
    "tokens_dim",
    "kv_tokens_dim",
    "pages_dim",
    "index_dim",
    "index_input",
    "last_page_len_input",
    "query_indptr_input",
    "grid_order",
)
_FIELDS_READ = {
    # A decode block is one (request, kv head) pair, so it has both grid axes to order.
    PAGED_DECODE: {"batch_dim", "pages_dim", "index_dim", "index_input", "grid_order"},
    PAGED_DECODE_SLAB: {
        "batch_dim",
        "pages_dim",
        "index_dim",
        "index_input",
        "grid_order",
        "last_page_len_input",
    },
    # A prefill block is one work-list slot over a token axis, found by scanning the
    # query indptr on the device.
    PAGED_PREFILL_CAUSAL: {
        "tokens_dim",
        "pages_dim",
        "index_dim",
        "index_input",
        "query_indptr_input",
    },
    # Ragged keys are packed like the queries: no page table, no page-table extents.
    RAGGED_PREFILL_CAUSAL: {"tokens_dim", "kv_tokens_dim", "query_indptr_input"},
}

_MLA_OPTIONAL_FIELDS = ("batch_dim", "tokens_dim", "query_indptr_input")
_MLA_FIELDS_READ = {
    MLA_PAGED_DECODE: {"batch_dim"},
    MLA_PAGED_PREFILL_CAUSAL: {"tokens_dim", "query_indptr_input"},
}

# The two partial tensors a split-KV pair hands from its partial kernel to its combine. The
# fused emitter turns them into the parameters `Partial_<name>_split` / `_combine`.
PARTIAL_ROWSUM = "ROWSUM"
PARTIAL_VALUE = "VALUE"
