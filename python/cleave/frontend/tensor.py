"""Dims, dtypes, annotations and the proxy ``Tensor`` of the ``cleave.lang`` frontend."""

from __future__ import annotations

import dataclasses
from typing import Sequence

from ..core.errors import SpecError


@dataclasses.dataclass(frozen=True)
class DType:
    name: str

    def __repr__(self):
        return f"cl.{_SHORT.get(self.name, self.name)}"

    def __str__(self):
        return self.name

    @property
    def torch(self):
        import torch

        return getattr(torch, self.name)


_SHORT = {"float16": "f16", "bfloat16": "bf16", "float32": "f32", "int32": "i32"}
f16, bf16, f32, i32 = DType("float16"), DType("bfloat16"), DType("float32"), DType("int32")
_BY_NAME = {d.name: d for d in (f16, bf16, f32, i32)}
_BY_NAME.update({v: _BY_NAME[k] for k, v in _SHORT.items()})


def as_dtype(dt) -> DType | None:
    if dt is None or isinstance(dt, DType):
        return dt
    name = str(dt).removeprefix("torch.")
    if name not in _BY_NAME:
        raise SpecError(f"unsupported dtype {dt!r}")
    return _BY_NAME[name]


@dataclasses.dataclass(frozen=True)
class Dim:
    """A named extent. ``dynamic=True`` dims are searched/scheduled at ``hint``."""

    name: str
    size: int | None = None
    dynamic: bool = False
    hint: int | None = None

    @property
    def extent(self) -> int:
        n = self.size if self.size is not None else self.hint
        if n is None:
            raise SpecError(f"dim {self.name!r} has no size; bind it (shapes={{{self.name!r}: ...}})")
        return int(n)

    def bind(self, size) -> "Dim":
        return dataclasses.replace(self, size=None if size is None else int(size))

    def __add__(self, k):  # ``batch + 1``: a derived extent (a CSR indptr's length)
        if not isinstance(k, int) or isinstance(k, bool):
            return NotImplemented
        grow = lambda n: None if n is None else n + k  # noqa: E731
        return Dim(f"{self.name}{k:+d}", grow(self.size), self.dynamic, grow(self.hint))

    __radd__ = __add__

    def __repr__(self):
        extra = "".join(f", {k}={v!r}" for k, v in (("size", self.size), ("hint", self.hint)) if v is not None)
        return f"Dim({self.name!r}{extra}{', dynamic=True' if self.dynamic else ''})"


def dims(names: str, dynamic: bool = False) -> tuple[Dim, ...]:
    """``M, K = cl.dims("M K")``: unsized named dims for annotations."""
    return tuple(Dim(n, dynamic=dynamic) for n in names.replace(",", " ").split())


def extent(d) -> int:
    return d.extent if isinstance(d, Dim) else int(d)


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    dims: tuple  # Dim | int
    dtype: DType | None = None
    optional: bool = False  # ``cl.Tensor[...] | None``

    def __or__(self, other):
        if other is not None:
            return NotImplemented
        return dataclasses.replace(self, optional=True)

    __ror__ = __or__


@dataclasses.dataclass(frozen=True)
class ScalarSpec:
    dtype: DType


class Scalar:
    """``cl.Scalar[cl.f32]``: a runtime scalar argument (paged programs only)."""

    def __class_getitem__(cls, dt):
        return ScalarSpec(as_dtype(dt))


@dataclasses.dataclass(frozen=True)
class Split:
    """A splittable dim (split-K / split-KV) and the factors to try."""

    dim: str
    factors: tuple = (4, 8, 16)

    def __init__(self, dim, factors: Sequence[int] = (4, 8, 16)):
        object.__setattr__(self, "dim", dim.name if isinstance(dim, Dim) else str(dim))
        object.__setattr__(self, "factors", tuple(int(f) for f in factors))


def as_splits(split) -> tuple[Split, ...]:
    if split is None or split is False:
        return ()
    if isinstance(split, (str, Dim, Split)):
        split = (split,)
    return tuple(s if isinstance(s, Split) else Split(s) for s in split)


def split_names(split) -> tuple[str, ...]:
    """Normalize ``"K" | Dim | Split | iterable of those`` to a tuple of dim names."""
    if split is None:
        return ()
    if isinstance(split, (str, Dim, Split)):
        split = (split,)
    out = []
    for s in split:
        name = s.dim if isinstance(s, Split) else s.name if isinstance(s, Dim) else str(s)
        if name not in out:
            out.append(name)
    return tuple(out)


def _scalar(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class Tensor:
    """Proxy for a value of a ``Program``. Also the annotation ``cl.Tensor["M", "K", cl.f16]``."""

    __array_priority__ = 1000

    def __class_getitem__(cls, items):
        items = items if isinstance(items, tuple) else (items,)
        dtype, ds = None, []
        for it in items:
            if isinstance(it, DType):
                dtype = it
            elif isinstance(it, str):
                ds.append(Dim(it))
            elif isinstance(it, Dim) or (isinstance(it, int) and not isinstance(it, bool)):
                ds.append(it)
            else:
                raise TypeError(f"bad cl.Tensor annotation item {it!r}")
        return TensorSpec(tuple(ds), dtype)

    def __init__(self, program, id: int, dims: tuple, dtype: DType, trans: bool = False):
        self.program, self.id, self.dims, self.dtype, self._trans = program, id, tuple(dims), dtype, trans

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(extent(d) for d in self.dims)

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def __repr__(self):
        names = ", ".join(d.name if isinstance(d, Dim) else str(d) for d in self.dims)
        return f"Tensor(%{self.id}{'.mT' if self._trans else ''}[{names}] {self.dtype!r})"

    # views folded into matmul operands
    @property
    def mT(self) -> "Tensor":
        if self.ndim < 2:
            raise SpecError(".mT needs a tensor of rank >= 2")
        d = self.dims
        return Tensor(self.program, self.id, d[:-2] + (d[-1], d[-2]), self.dtype, not self._trans)

    @property
    def T(self) -> "Tensor":
        if self.ndim != 2:
            raise SpecError(".T is only defined for 2-D tensors; use .mT")
        return self.mT

    def view(self, *dims) -> "Tensor":
        view = getattr(self.program, "_view", None)
        if view is None:
            raise SpecError(".view is only lowerable inside a paged (dynamic) program")
        return view(self, dims[0] if len(dims) == 1 and isinstance(dims[0], tuple) else dims)

    # arithmetic
    def __matmul__(self, o):
        return self.program._matmul(self, o)

    def __rmatmul__(self, o):
        return self.program._matmul(o, self)

    def __add__(self, o):
        return self.program._binary("add", self, o)

    __radd__ = __add__

    def __mul__(self, o):
        return self.program._binary("mul", self, o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self.program._binary("div", self, o)

    def __rtruediv__(self, o):
        return self.program._binary("div", o, self)

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        if _scalar(o):
            return self + (-float(o))
        return self.program._binary("sub", self, o)

    def __rsub__(self, o):
        if _scalar(o):
            return -self + float(o)
        return self.program._binary("sub", o, self)

    def __pow__(self, p):
        if p == 2:
            return self * self
        if p == 0.5:
            return self.sqrt()
        if p == -0.5:
            return self.rsqrt()
        raise SpecError(f"x ** {p!r}: only 2, 0.5 and -0.5 are lowerable")

    def exp(self):
        return self.program._unary("exp", self)

    def sqrt(self):
        return self.program._unary("sqrt", self)

    def silu(self):
        return self.program._unary("silu", self)

    def rsqrt(self):
        return 1.0 / self.sqrt()

    def sum(self, dim: int = -1, keepdim: bool = False):
        if not keepdim:
            raise SpecError("tile-IR reductions keep the reduced axis; pass keepdim=True")
        return self.program._reduction(self, dim)

    def max(self, dim: int = -1, keepdim: bool = False):
        """Max reduction (values only)."""
        if not keepdim:
            raise SpecError("tile-IR reductions keep the reduced axis; pass keepdim=True")
        reduce = getattr(self.program, "_max", None)
        if reduce is None:
            raise SpecError(".max is only lowerable in a dense program")
        return reduce(self, dim)

    def mean(self, dim: int = -1, keepdim: bool = False):
        return self.sum(dim, keepdim) / float(self.shape[dim])

    def __bool__(self):
        raise SpecError("data-dependent control flow on a traced cl.Tensor")
