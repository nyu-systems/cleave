"""Typed view of ``KNGraph.get_graph_structure()`` output (list of op dicts).

``KernelGraph.from_dicts(d).to_dicts() == d`` for every structure the search emits
and every checked-in fixture: unknown keys are kept in ``attrs``/``extra`` and the
dim padding past ``num_dims`` is kept in ``dim_padding``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_AXIS_OP = re.compile(r"^kn_(split|reduce_sum|reduce_max)_(\d+)_op$")
_PLAIN_OP = re.compile(r"^kn_(\w+?)_op$")
_TENSOR_KEYS = ("num_dims", "dim", "guid", "dtype", "can_split", "can_reduce")


def parse_op_type(op_type: str) -> Tuple[str, Optional[int]]:
    """``"kn_reduce_sum_2_op" -> ("reduce_sum", 2)``, ``"kn_add_op" -> ("add", None)``."""
    m = _AXIS_OP.match(op_type)
    if m:
        return m.group(1), int(m.group(2))
    m = _PLAIN_OP.match(op_type)
    if m:
        return m.group(1), None
    raise ValueError(f"unrecognized op_type {op_type!r}")


def format_op_type(kind: str, axis: Optional[int] = None) -> str:
    return f"kn_{kind}_op" if axis is None else f"kn_{kind}_{axis}_op"


@dataclass
class KGTensor:
    guid: int
    dims: Tuple[int, ...]
    dtype: Optional[str] = None
    can_split: Optional[Tuple[bool, ...]] = None
    can_reduce: Optional[Tuple[bool, ...]] = None
    dim_padding: Tuple[int, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return len(self.dims)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KGTensor":
        dim = [int(v) for v in d.get("dim", [])]
        n = int(d.get("num_dims", len(dim)))
        opt = lambda k: None if d.get(k) is None else tuple(d[k])  # noqa: E731
        return cls(
            guid=int(d["guid"]),
            dims=tuple(dim[:n]),
            dtype=d.get("dtype"),
            can_split=opt("can_split"),
            can_reduce=opt("can_reduce"),
            dim_padding=tuple(dim[n:]),
            extra={k: v for k, v in d.items() if k not in _TENSOR_KEYS},
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "num_dims": self.rank,
            "dim": list(self.dims) + list(self.dim_padding),
            "guid": self.guid,
        }
        if self.dtype is not None:
            d["dtype"] = self.dtype
        if self.can_split is not None:
            d["can_split"] = list(self.can_split)
        if self.can_reduce is not None:
            d["can_reduce"] = list(self.can_reduce)
        d.update(self.extra)
        return d


@dataclass
class KGOp:
    kind: str
    axis: Optional[int] = None
    inputs: List[KGTensor] = field(default_factory=list)
    outputs: List[KGTensor] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def op_type(self) -> str:
        return format_op_type(self.kind, self.axis)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KGOp":
        kind, axis = parse_op_type(d["op_type"])
        return cls(
            kind,
            axis,
            [KGTensor.from_dict(t) for t in d.get("input_tensors", [])],
            [KGTensor.from_dict(t) for t in d.get("output_tensors", [])],
            {k: v for k, v in d.items() if k not in ("op_type", "input_tensors", "output_tensors")},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "input_tensors": [t.to_dict() for t in self.inputs],
            "output_tensors": [t.to_dict() for t in self.outputs],
            **self.attrs,
        }


@dataclass
class KernelGraph:
    ops: List[KGOp] = field(default_factory=list)

    @classmethod
    def from_dicts(cls, structure: List[Dict[str, Any]]) -> "KernelGraph":
        return cls([KGOp.from_dict(op) for op in structure])

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [op.to_dict() for op in self.ops]

    def ops_of(self, kind: str) -> List[KGOp]:
        return [op for op in self.ops if op.kind == kind]

    @property
    def inputs(self) -> List[KGTensor]:
        return [t for op in self.ops_of("input") for t in op.outputs]

    @property
    def has_split(self) -> bool:
        return any(op.kind == "split" for op in self.ops)
