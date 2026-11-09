"""Lowering (internal pipeline step): searched kernel-graph structures -> Cleave IR."""

from .convert import to_ir
from .process import to_bundles

__all__ = ["to_bundles", "to_ir"]
