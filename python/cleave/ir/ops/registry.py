"""OpSpec registry: one row per op class in ``cleave.ir.ops`` (data only).

``kn_types``  superoptimizer op types lowered to this class by
              ``lower.convert.to_ir``.
``superopt``  "native" (a kn op maps straight to it), "decompose" (built from several
              kn ops), "opaque" (searched as a black box) or None (never produced
              by the superoptimizer path).
``reference`` ``runtime.reference`` evaluates it in torch.
``axis_map``  kind in ``ir.analysis.AXIS_MAP`` (type-exact).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from . import (
    Accumulate,
    Add,
    CausalMaskedExp,
    ConstFill,
    Copy,
    Div,
    Exp,
    IndexGather,
    IndptrTileScan,
    Log2,
    MaskedDiv,
    MaskedExp,
    MaskedLog2,
    Matmul,
    ReduceMax,
    Mul,
    PagedGather,
    PermuteView,
    RangeGather,
    ReduceSum,
    RowGather,
    RSqrt,
    Silu,
    SplitView,
    Sqrt,
    Sub,
)

SUPEROPT_KINDS = ("native", "decompose", "opaque")


@dataclass(frozen=True)
class OpSpec:
    cls: type
    kn_types: Tuple[str, ...] = ()
    superopt: Optional[str] = None
    reference: bool = True
    axis_map: Optional[str] = None

    @property
    def name(self) -> str:
        return self.cls.__name__


def _axes(prefix: str) -> Tuple[str, ...]:
    return tuple(f"kn_{prefix}_{i}_op" for i in range(5))


_SCHED_ONLY = "semantics live in gen_macro and depend on schedule scalars a source graph lacks"

# Ops graph_executor refuses, with the reason (asserted against graph_executor).
NO_REFERENCE: Dict[str, str] = {
    "PagedGather": _SCHED_ONLY,
    "IndexGather": _SCHED_ONLY,
    "RangeGather": _SCHED_ONLY,
    "RowGather": _SCHED_ONLY,
    "MaskedExp": _SCHED_ONLY,
    "CausalMaskedExp": _SCHED_ONLY,
    "MaskedDiv": _SCHED_ONLY,
    "MaskedLog2": _SCHED_ONLY,
    "ConstFill": "alternative producer in Graph.else_nodes, never in a source graph",
    "IndptrTileScan": "kernel-scope prologue, never in graph.nodes",
}

_UNARY = "unary elementwise; the identity fallback is exact"
_POST_TILING = "only built by paging/handwritten schedules after axis analysis"

# Ops without an AXIS_MAP entry, with the reason.
AXIS_MAP_EXEMPT: Dict[str, str] = {
    "Log2": _UNARY,
    "MaskedLog2": _UNARY,
    "Copy": _UNARY,
    "Accumulate": _UNARY,
    "MaskedExp": _UNARY,
    "CausalMaskedExp": _UNARY,
    "PagedGather": _POST_TILING,
    "IndexGather": _POST_TILING,
    "RangeGather": _POST_TILING,
    "RowGather": _POST_TILING,
    # Identity fallback is wrong for a non-trivial perm; PermuteView is only used by
    # hand-built test graphs today, so no lowering path hits it.
    "PermuteView": "not produced by lowering; identity fallback is inexact for perms",
    "ConstFill": "not a dataflow consumer (else_nodes)",
    "IndptrTileScan": "not a dataflow consumer (prologue)",
}


def _spec(cls, kn_types=(), superopt=None, axis_map=None) -> OpSpec:
    return OpSpec(cls, tuple(kn_types), superopt, cls.__name__ not in NO_REFERENCE, axis_map)


_SPECS = (
    _spec(Matmul, ["kn_matmul_op"], "native", "matmul"),
    _spec(SplitView, _axes("split"), "native", "split"),
    _spec(Add, ["kn_add_op"], "native", "binary"),
    _spec(Sub, ["kn_sub_op"], "native", "binary"),
    _spec(Mul, ["kn_mul_op"], "native", "binary"),
    _spec(Div, ["kn_div_op"], "native", "binary"),
    _spec(Exp, ["kn_exp_op"], "native", "identity"),
    _spec(Silu, ["kn_silu_op"], "native", "identity"),
    _spec(Sqrt, ["kn_sqrt_op"], "native", "identity"),
    # kn_sqrt_op followed by kn_div_op(1.0 / x), fused at lowering.
    _spec(RSqrt, ["kn_sqrt_op", "kn_div_op"], "decompose", "identity"),
    _spec(ReduceSum, _axes("reduce_sum"), "native", "reduction"),
    _spec(ReduceMax, _axes("reduce_max"), "opaque", "reduction"),
    _spec(Log2),
    _spec(Copy),
    _spec(Accumulate),
    _spec(PermuteView),
    _spec(MaskedExp),
    _spec(CausalMaskedExp),
    _spec(MaskedDiv, axis_map="binary"),
    _spec(MaskedLog2),
    _spec(PagedGather),
    _spec(IndexGather),
    _spec(RangeGather),
    _spec(RowGather),
    _spec(ConstFill),
    _spec(IndptrTileScan),
)

OPS: Dict[str, OpSpec] = {s.name: s for s in _SPECS}


def spec_for(cls_or_node) -> OpSpec:
    cls = cls_or_node if isinstance(cls_or_node, type) else type(cls_or_node)
    return OPS[cls.__name__]


def by_kn_type(op_type: str) -> Tuple[OpSpec, ...]:
    return tuple(s for s in _SPECS if op_type in s.kn_types)
