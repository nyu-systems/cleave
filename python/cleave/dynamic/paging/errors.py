"""Typed refusals. Each is a PagingRefusal, hence an AssertionError that survives ``-O``."""

from ...core.errors import PagingRefusal


class InvalidSpec(PagingRefusal):
    """The spec contradicts itself or names something no layer reads."""


class NotDenseAttention(PagingRefusal):
    """The graph is not the dense attention body the access layer rebuilds."""


class UnsupportedTiling(PagingRefusal):
    """Dense attention, but at a tiling this access layer cannot page."""
