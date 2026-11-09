"""``cleave.Program``: the frontend IR, replayable onto the csopt TileGraph builder."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import operator

from ..core.errors import SpecError
from .tensor import Dim, DType, Tensor, _scalar, as_dtype, extent, f16, split_names

_UNARY = ("exp", "sqrt", "silu")
_BINARY = {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": operator.truediv}
_REDUCE = ("reduce_sum", "reduce_max")  # record kinds named like their csopt builder methods


@dataclasses.dataclass(frozen=True)
class InputSpec:
    name: str
    dims: tuple  # Dim | int (literal extent)
    dtype: DType

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(extent(d) for d in self.dims)


@dataclasses.dataclass(frozen=True)
class OutputSpec:
    name: str
    tensor: Tensor


@dataclasses.dataclass(frozen=True)
class Op:
    """One builder call. ``args`` are value ids (int) or scalars (float)."""

    kind: str  # input | matmul | add | sub | mul | div | exp | sqrt | silu | reduction | max | output
    args: tuple
    attrs: tuple = ()  # sorted (key, value) pairs
    out: int | None = None

    def attr(self, key, default=None):
        return dict(self.attrs).get(key, default)


class Program:
    def __init__(self, name: str = "program", dtype=f16):
        self.name, self.dtype = name, as_dtype(dtype)
        self.split = ()  # declared Split hints (from @jit / trace); not used by to_tile_graph
        self._dims: dict[str, Dim] = {}
        self._records: list[Op] = []
        self._values: list[Tensor] = []
        self.inputs: list[InputSpec] = []
        self.outputs: list[OutputSpec] = []

    # ---------------------------------------------------------------- builder
    def dim(self, name: str, size=None, dynamic: bool = False, hint=None) -> Dim:
        return self._register(Dim(name, None if size is None else int(size), dynamic, hint))

    def dims(self, **sizes) -> tuple[Dim, ...]:
        return tuple(self.dim(n, s) for n, s in sizes.items())

    def _register(self, d: Dim) -> Dim:
        old = self._dims.get(d.name)
        if old is None or old == d:
            self._dims[d.name] = d
            d.extent  # must be sized (or hinted)
            return d
        if d.size is None and d.hint is None and not d.dynamic:
            return old  # bare name reference
        raise SpecError(f"dim {d.name!r} redeclared: {old} vs {d}")

    def input(self, name: str, dims, dtype=None) -> Tensor:
        if any(i.name == name for i in self.inputs):
            raise SpecError(f"duplicate input name {name!r}")
        ds = []
        for d in dims:
            if isinstance(d, str):
                if d not in self._dims:
                    raise SpecError(f"input {name!r}: unknown dim {d!r}; declare it with .dim()/.dims()")
                d = self._dims[d]
            elif isinstance(d, Dim):
                d = self._register(d)
            elif not isinstance(d, int) or isinstance(d, bool) or d <= 0:
                raise SpecError(f"input {name!r}: bad dim {d!r}")
            ds.append(d)
        spec = InputSpec(name, tuple(ds), as_dtype(dtype) or self.dtype)
        self.inputs.append(spec)
        return self._emit("input", (), spec.dims, spec.dtype, name=name)

    def output(self, t: Tensor, name: str | None = None) -> None:
        self._check(t, "output")
        name = name or f"out{len(self.outputs)}"
        if any(o.name == name for o in self.outputs):
            raise SpecError(f"duplicate output name {name!r}")
        self.outputs.append(OutputSpec(name, t))
        self._records.append(Op("output", (t.id,), (("name", name),)))

    # ---------------------------------------------------------------- views
    @property
    def ops(self) -> list[Op]:
        """Input and compute records in builder order (outputs are in ``.outputs``)."""
        return [r for r in self._records if r.kind != "output"]

    @property
    def op_count(self) -> int:
        """Input + compute ops; the search budget adds ``len(outputs)`` on top."""
        return len(self.ops)

    @property
    def sizes(self) -> dict[str, int]:
        return {n: d.extent for n, d in self._dims.items()}

    @property
    def dynamic_dims(self) -> tuple[Dim, ...]:
        return tuple(d for d in self._dims.values() if d.dynamic)

    def fingerprint(self) -> str:
        """sha256 over ops, dims and dtypes; independent of names and Python ids."""
        def dim(d):
            return [d.name, d.extent, d.dynamic] if isinstance(d, Dim) else int(d)

        payload = {
            "dtype": self.dtype.name,
            "inputs": [[[dim(d) for d in s.dims], s.dtype.name] for s in self.inputs],
            "ops": [[r.kind, [a if isinstance(a, int) else repr(float(a)) for a in r.args],
                     [[k, v] for k, v in r.attrs if k != "name"]] for r in self._records],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def __repr__(self):
        lines = [f"cleave.Program {self.name!r} dtype={self.dtype.name} dims={self.sizes} "
                 f"fingerprint={self.fingerprint()[:12]}"]
        for r in self._records:
            args = ", ".join(f"%{a}" if isinstance(a, int) else repr(a) for a in r.args)
            attrs = "".join(f" {k}={v!r}" for k, v in r.attrs if v is not False)
            lhs = f"%{r.out} = " if r.out is not None else ""
            shape = f"  : {list(self._values[r.out].shape)}" if r.out is not None else ""
            lines.append(f"  {lhs}{r.kind}({args}){attrs}{shape}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- emission
    def _emit(self, kind, args, dims, dtype, **attrs) -> Tensor:
        t = Tensor(self, len(self._values), dims, dtype)
        self._values.append(t)
        self._records.append(Op(kind, tuple(args), tuple(sorted(attrs.items())), t.id))
        return t

    def _check(self, t, what):
        if not isinstance(t, Tensor):
            raise SpecError(f"{what}: expected a cl.Tensor, got {type(t).__name__}")
        if t.program is not self:
            raise SpecError(f"{what}: tensor belongs to another Program")
        if t._trans:
            raise SpecError(f"{what} on a transposed view: .mT/.T only fold into matmul operands")
        return t

    def _matmul(self, a, b) -> Tensor:
        for x in (a, b):
            if not isinstance(x, Tensor) or x.program is not self:
                raise SpecError("matmul operands must be cl.Tensors of the same Program")
        if a.ndim != b.ndim or a.ndim < 2:
            raise SpecError(f"matmul rank mismatch: {a.shape} @ {b.shape}")
        if extent(a.dims[-1]) != extent(b.dims[-2]):
            raise SpecError(f"matmul inner dims differ: {a.shape} @ {b.shape}")
        batch = []
        for da, db in zip(a.dims[:-2], b.dims[:-2]):
            ea, eb = extent(da), extent(db)
            if ea != eb and max(ea, eb) % min(ea, eb):
                raise SpecError(f"matmul batch dims not grouped-broadcastable: {a.shape} @ {b.shape}")
            batch.append(da if ea >= eb else db)
        dims = tuple(batch) + (a.dims[-2], b.dims[-1])
        return self._emit("matmul", (a.id, b.id), dims, a.dtype, trans_A=a._trans, trans_B=b._trans)

    def _binary(self, op, a, b) -> Tensor:
        if not isinstance(a, Tensor):
            if not _scalar(a):
                raise SpecError(f"{op}: unsupported operand {type(a).__name__}")
            if op in ("add", "mul"):
                a, b = b, a  # the tile builder takes add/mul scalars on the right only
            else:
                self._check(b, op)
                return self._emit(op, (float(a), b.id), b.dims, b.dtype)
        self._check(a, op)
        if not isinstance(b, Tensor):
            if not _scalar(b):
                raise SpecError(f"{op}: unsupported operand {type(b).__name__}")
            return self._emit(op, (a.id, float(b)), a.dims, a.dtype)
        self._check(b, op)
        if a.ndim != b.ndim:
            raise SpecError(f"{op}: rank mismatch {a.shape} vs {b.shape}")
        dims = []
        for da, db in zip(a.dims, b.dims):
            ea, eb = extent(da), extent(db)
            if ea != eb and 1 not in (ea, eb):
                raise SpecError(f"{op}: shapes not broadcastable {a.shape} vs {b.shape}")
            dims.append(db if ea == 1 and eb != 1 else da)
        return self._emit(op, (a.id, b.id), dims, a.dtype)

    def _unary(self, op, a) -> Tensor:
        self._check(a, op)
        return self._emit(op, (a.id,), a.dims, a.dtype)

    def _reduction(self, a, dim, kind="reduce_sum") -> Tensor:
        what = kind.removeprefix("reduce_")
        self._check(a, what)
        if not -a.ndim <= dim < a.ndim:
            raise SpecError(f"{what}: dim {dim} out of range for rank {a.ndim}")
        dim %= a.ndim
        dims = a.dims[:dim] + (1,) + a.dims[dim + 1:]
        return self._emit(kind, (a.id,), dims, a.dtype, dim=dim)

    def _max(self, a, dim) -> Tensor:
        return self._reduction(a, dim, "reduce_max")

    # ---------------------------------------------------------------- lowering
    def can_split(self, split=()) -> list[tuple[bool, ...]]:
        """Per-input can_split flags: every axis whose dim name is in ``split``."""
        names = split_names(split)
        known = {d.name for s in self.inputs for d in s.dims if isinstance(d, Dim)}
        unknown = [n for n in names if n not in known]
        if unknown:
            raise SpecError(f"split dims {unknown} are not dims of any input (have {sorted(known)})")
        return [tuple(isinstance(d, Dim) and d.name in names for d in s.dims) for s in self.inputs]

    def to_tile_graph(self, split=()):
        """Replay onto ``csopt.new_tile_graph`` (the superoptimizer's input graph)."""
        import cleave.superoptimizer as csopt
        from cleave.superoptimizer import core

        flags = self.can_split(split)
        graph = csopt.new_tile_graph(split_enabled=any(map(any, flags)))
        flags = iter(flags)
        env = {}
        val = lambda a: env[a] if isinstance(a, int) else a  # noqa: E731
        for r in self._records:
            if r.kind == "input":
                t = self._values[r.out]
                env[r.out] = graph.create_tiled_input(t.shape, next(flags), core.dtype(t.dtype.name))
            elif r.kind == "output":
                graph.create_tiled_output(env[r.args[0]])
            elif r.kind == "matmul":
                env[r.out] = graph.matmul(env[r.args[0]], env[r.args[1]],
                                          trans_A=r.attr("trans_A"), trans_B=r.attr("trans_B"))
            elif r.kind in _REDUCE:
                env[r.out] = getattr(graph, r.kind)(env[r.args[0]], r.attr("dim"))
            else:
                env[r.out] = getattr(graph, r.kind)(*map(val, r.args))
        return graph

    def to_graph_structure(self) -> list[dict]:
        """The un-optimized program as a ``get_graph_structure()`` list (for to_ir)."""
        input_ids = {r.out for r in self._records if r.kind == "input"}

        def desc(i):
            t = self._values[i]
            d = {"num_dims": t.ndim, "dim": list(t.shape), "guid": i + 1}
            if i in input_ids:
                d["dtype"] = t.dtype.name
            return d

        out = []
        for r in self._records:
            kind = f"{r.kind}_{r.attr('dim')}" if r.kind in _REDUCE else r.kind
            info = {"op_type": f"kn_{kind}_op",
                    "input_tensors": [desc(a) for a in r.args if isinstance(a, int)],
                    "output_tensors": [desc(r.out)] if r.out is not None else []}
            if r.kind == "matmul":
                info.update(trans_A=bool(r.attr("trans_A")), trans_B=bool(r.attr("trans_B")))
            if r.kind in ("add", "mul", "div") and not all(isinstance(a, int) for a in r.args):
                info.update(scalar=float(next(a for a in r.args if not isinstance(a, int))),
                            scalar_on_left=not isinstance(r.args[0], int))
            out.append(info)
        return out

    # ---------------------------------------------------------------- torch
    def _bind(self, args, named) -> list:
        if args and named:
            raise TypeError("pass inputs either positionally or by name, not both")
        if named:
            missing = [s.name for s in self.inputs if s.name not in named]
            extra = set(named) - {s.name for s in self.inputs}
            if missing or extra:
                raise TypeError(f"inputs: missing {missing}, unexpected {sorted(extra)}")
            args = [named[s.name] for s in self.inputs]
        if len(args) != len(self.inputs):
            raise TypeError(f"{self.name}: expected {len(self.inputs)} inputs, got {len(args)}")
        for s, x in zip(self.inputs, args):
            if tuple(x.shape) != s.shape:
                raise SpecError(f"input {s.name!r}: shape {tuple(x.shape)} != {s.shape}")
        return list(args)

    def reference(self, *inputs, **named) -> list:
        """Run the program eagerly in torch (fp32 compute); one tensor per output."""
        import torch

        env = {}
        it = iter(self._bind(inputs, named))
        val = lambda a: env[a] if isinstance(a, int) else a  # noqa: E731
        for r in self._records:
            if r.kind == "input":
                env[r.out] = next(it).float()
            elif r.kind == "output":
                continue
            elif r.kind == "matmul":
                env[r.out] = grouped_matmul(env[r.args[0]], env[r.args[1]],
                                            r.attr("trans_A"), r.attr("trans_B"))
            elif r.kind == "reduce_sum":
                env[r.out] = env[r.args[0]].sum(r.attr("dim"), keepdim=True)
            elif r.kind == "reduce_max":
                env[r.out] = env[r.args[0]].amax(r.attr("dim"), keepdim=True)
            elif r.kind == "silu":
                env[r.out] = torch.nn.functional.silu(env[r.args[0]])
            elif r.kind in _UNARY:
                env[r.out] = getattr(torch, r.kind)(env[r.args[0]])
            else:
                env[r.out] = _BINARY[r.kind](*map(val, r.args))
        return [env[o.tensor.id].to(o.tensor.dtype.torch) for o in self.outputs]

    def random_inputs(self, seed: int = 0, device: str = "cuda") -> list:
        import torch

        gen = torch.Generator(device="cpu").manual_seed(seed)
        positive = self._positive_inputs()
        out = []
        for s in self.inputs:
            if s.dtype.name == "int32":
                x = torch.randint(0, 16, s.shape, generator=gen, dtype=torch.int32)
            elif s.name in positive:
                x = torch.rand(s.shape, generator=gen) + 0.5
            else:
                x = torch.randn(s.shape, generator=gen)
            out.append(x.to(dtype=s.dtype.torch, device=device))
        return out

    def _positive_inputs(self) -> set:
        """Inputs read directly by sqrt or as a tensor divisor (sampled strictly positive)."""
        names = {r.out: r.attr("name") for r in self._records if r.kind == "input"}
        pos = set()
        for r in self._records:
            if r.kind == "sqrt" or (r.kind == "div" and isinstance(r.args[1], int)):
                a = r.args[0] if r.kind == "sqrt" else r.args[1]
                if a in names:
                    pos.add(names[a])
        return pos


def grouped_matmul(a, b, trans_a=False, trans_b=False):
    """``a @ b`` with the tile IR's grouped batch broadcast (== repeat_interleave)."""
    if trans_a:
        a = a.mT
    if trans_b:
        b = b.mT
    for i in range(a.ndim - 2):
        ea, eb = a.shape[i], b.shape[i]
        if ea > eb:
            b = b.repeat_interleave(ea // eb, dim=i)
        elif eb > ea:
            a = a.repeat_interleave(eb // ea, dim=i)
    return a @ b
