"""Wrap a scheduled dense attention graph in a paged/ragged access layer.

The dense stages pick the tiling; this pass makes it dynamic (page-table gathers, masked
softmax, runtime trip counts, symbolic grid). The KV proxy extent does not survive it.
"""

from .apply import apply_paging, apply_paging_split, build_split_decode
from .constraints import can_page, paging_accept
from .errors import InvalidSpec, NotDenseAttention, PagingRefusal, UnsupportedTiling
from .kinds import (
    CAUSAL_KINDS,
    KINDS,
    MLA_KINDS,
    MLA_PAGED_DECODE,
    MLA_PAGED_PREFILL_CAUSAL,
    PAGED_DECODE,
    PAGED_DECODE_SLAB,
    PAGED_KINDS,
    PAGED_PREFILL_CAUSAL,
    PARTIAL_ROWSUM,
    PARTIAL_VALUE,
    RAGGED_PREFILL_CAUSAL,
    SLAB_KINDS,
    SPLIT_KINDS,
)
from .recognize import (
    DenseAttention,
    DenseMLA,
    graph_tile,
    read_dense_attention,
    read_dense_combine,
    read_dense_mla,
    read_dense_mla_partial,
    read_dense_partial,
)
from .specs import CombineTiling, MLASpec, PagingSpec
from .stages import DENOM_FLOOR, LOG2E

__all__ = [
    "CAUSAL_KINDS",
    "DENOM_FLOOR",
    "KINDS",
    "LOG2E",
    "MLA_KINDS",
    "MLA_PAGED_DECODE",
    "MLA_PAGED_PREFILL_CAUSAL",
    "PAGED_DECODE",
    "PAGED_DECODE_SLAB",
    "PAGED_KINDS",
    "PAGED_PREFILL_CAUSAL",
    "PARTIAL_ROWSUM",
    "PARTIAL_VALUE",
    "RAGGED_PREFILL_CAUSAL",
    "SLAB_KINDS",
    "SPLIT_KINDS",
    "CombineTiling",
    "DenseAttention",
    "DenseMLA",
    "InvalidSpec",
    "MLASpec",
    "NotDenseAttention",
    "PagingRefusal",
    "PagingSpec",
    "UnsupportedTiling",
    "apply_paging",
    "apply_paging_split",
    "build_split_decode",
    "can_page",
    "graph_tile",
    "paging_accept",
    "read_dense_attention",
    "read_dense_combine",
    "read_dense_mla",
    "read_dense_mla_partial",
    "read_dense_partial",
]
