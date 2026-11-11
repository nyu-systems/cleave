"""Paged attention: ``cl.PageTable`` / ``cl.paged_gather`` / ``cl.Scalar`` trace into a
``PagedProgram``; ``.attention()`` derives its ``PagingSpec`` and dense block body."""

from __future__ import annotations

import dataclasses
import inspect
import math

from ..core.errors import SpecError
from . import composites
from .program import Program
from .tensor import Dim, ScalarSpec, Tensor, TensorSpec, as_dtype, as_splits, bf16, f32

DEFAULT_KV_HINT = 4096  # cleave.dynamic.dense.KV_PROXY
MIN_QUERY_ROWS = 16  # one MMA's M extent
PREFILL_QUERY_ROWS = 64  # a GQA prefill block's (token, head slot) rows


# ----------------------------------------------------------------------------- user API
class ScheduleHints:
    """Schedule knobs of a paged kernel (tunables, not operator semantics)."""

    KNOBS = ("grid_order", "rasterization", "warp_policy", "pass_configs",
             "index_window_tiles", "prefetch")

    def __init__(self, **knobs):
        unknown = sorted(set(knobs) - set(self.KNOBS))
        if unknown:
            raise SpecError(f"unknown schedule hints {unknown}; expected some of {self.KNOBS}")
        if "grid_order" in knobs:
            knobs["grid_order"] = tuple(knobs["grid_order"])
        self.knobs = knobs

    def __eq__(self, other):
        return isinstance(other, ScheduleHints) and self.knobs == other.knobs

    def __repr__(self):
        return f"cl.ScheduleHints({', '.join(f'{k}={v!r}' for k, v in self.knobs.items())})"


class PageTable:
    """CSR page table: request ``r`` owns ``indices[indptr[r]:indptr[r+1]]``; ``last_page_len``
    is required iff a page holds more than one token. Without ``indices`` the rows are ragged:
    request ``r`` owns rows ``indptr[r]:indptr[r+1]`` of a packed tensor (queries, ragged KV)."""

    def __init__(self, indptr, indices=None, last_page_len=None, length: Dim | None = None):
        if length is not None and not isinstance(length, Dim):
            raise SpecError(f"PageTable(length=...) must be a cl.Dim, got {length!r}")
        if indices is None and last_page_len is not None:
            raise SpecError("PageTable: last_page_len needs indices (ragged rows have no pages)")
        self.indptr, self.indices, self.last_page_len = indptr, indices, last_page_len
        self.length = length or Dim("kv_len", dynamic=True, hint=DEFAULT_KV_HINT)

    @property
    def ragged(self) -> bool:
        return self.indices is None

    @property
    def requests(self) -> Dim:
        d = self.indptr.dims[0] if isinstance(self.indptr, Tensor) else None
        if isinstance(d, Dim) and d.name.endswith("+1"):
            return Dim(d.name[:-2], dynamic=True)
        return Dim("batch", dynamic=True)


def paged_gather(cache, pages: PageTable):
    """``cache[(num_pages, page_size, heads, d)]`` (ragged table: ``(total, heads, d)``)
    gathered per request through ``pages``: a logical ``(batch, heads, length, d)``."""
    if not isinstance(pages, PageTable):
        raise SpecError(f"paged_gather(cache, pages): pages must be a cl.PageTable, got {pages!r}")
    if not isinstance(cache, Tensor):
        raise NotImplementedError("eager paged_gather: validate a paged operator over workloads")
    if not isinstance(cache.program, PagedProgram):
        raise SpecError("paged_gather needs a paged program (annotate the runtime scale "
                        "as cl.Scalar)")
    return cache.program._gather(cache, pages)


def softmax(x, dim: int = -1, shift: float | None = None, return_lse: bool = False,
            causal: bool = False):
    """Softmax over ``dim``; ``return_lse`` also returns the base-2 log-sum-exp. ``causal``
    masks bottom-right aligned scores (query ``i`` sees keys ``j <= i + kv_len - q_len``)."""
    if isinstance(x, Tensor) and isinstance(x.program, PagedProgram):
        return x.program._softmax(x, dim, return_lse, causal)
    if causal:
        raise SpecError("softmax(causal=True) is only lowerable inside a paged program")
    if not return_lse:
        return composites.softmax(x, dim, shift)
    if isinstance(x, Tensor):
        raise SpecError("softmax(return_lse=True) is only lowerable inside a paged program")
    xf = x.float()
    return xf.softmax(dim).to(x.dtype), xf.logsumexp(dim) / math.log(2.0)


# ----------------------------------------------------------------------------- program
@dataclasses.dataclass(frozen=True)
class Rec:
    kind: str  # input | scalar | view | paged_gather | matmul | add | mul | div | exp | ...
    args: tuple  # value ids (int) or Python scalars
    outs: tuple  # value ids
    attrs: tuple = ()

    def attr(self, key, default=None):
        return dict(self.attrs).get(key, default)


@dataclasses.dataclass(frozen=True)
class Param:
    name: str
    spec: object  # TensorSpec | ScalarSpec
    value: int  # value id

    @property
    def is_scalar(self) -> bool:
        return isinstance(self.spec, ScalarSpec)

    @property
    def kernel_name(self) -> str:
        """Kernel parameter name: tensors upper-cased (``PlaceHolder_<NAME>``), scalars verbatim."""
        return self.name if self.is_scalar else self.name.upper()


class PagedProgram:
    def __init__(self, name: str = "paged", dtype=bf16):
        self.name, self.dtype = name, as_dtype(dtype)
        self.params: list[Param] = []
        self.records: list[Rec] = []
        self.outputs: list[tuple[str, int]] = []
        self.hints = ScheduleHints()
        self.split = ()  # declared cl.Split hints (split-KV), as on Program
        self._values: list[Tensor] = []
        self._producer: dict[int, Rec] = {}
        self.tables: list[PageTable] = []

    def __repr__(self):
        lines = [f"cleave.PagedProgram {self.name!r} dtype={self.dtype.name}"]
        for r in self.records:
            args = ", ".join(f"%{a}" if isinstance(a, int) else repr(a) for a in r.args)
            attrs = "".join(f" {k}={v!r}" for k, v in r.attrs if v is not False)
            outs = ", ".join(f"%{o}" for o in r.outs)
            dims = "; ".join(str([getattr(d, "name", d) for d in self._values[o].dims]) for o in r.outs)
            lines.append(f"  {outs} = {r.kind}({args}){attrs}  : {dims}")
        lines += [f"  output {n} = %{v}" for n, v in self.outputs]
        return "\n".join(lines)

    def value(self, i: int) -> Tensor:
        return self._values[i]

    def producer(self, i: int) -> Rec | None:
        return self._producer.get(i)

    def param(self, i: int) -> Param | None:
        return next((p for p in self.params if p.value == i), None)

    # ---------------------------------------------------------------- recording
    def _rec(self, kind, args, out_dims, dtype=None, **attrs):
        outs = []
        dtypes = dtype if isinstance(dtype, list) else [dtype] * len(out_dims)
        for dims, dt in zip(out_dims, dtypes):
            t = Tensor(self, len(self._values), tuple(dims), dt or self.dtype)
            self._values.append(t)
            outs.append(t)
        r = Rec(kind, tuple(args), tuple(t.id for t in outs), tuple(sorted(attrs.items())))
        self.records.append(r)
        for t in outs:
            self._producer[t.id] = r
        return outs[0] if len(outs) == 1 else tuple(outs)

    def _own(self, t, what) -> Tensor:
        if not isinstance(t, Tensor) or t.program is not self:
            raise SpecError(f"{what}: expected a cl.Tensor of this program, got {t!r}")
        return t

    def input(self, name: str, spec) -> Tensor:
        if isinstance(spec, ScalarSpec):
            t = self._rec("scalar", (), [()], spec.dtype or f32, name=name)
        else:
            t = self._rec("input", (), [spec.dims], spec.dtype or self.dtype, name=name)
        self.params.append(Param(name, spec, t.id))
        return t

    def output(self, t: Tensor, name: str | None = None) -> None:
        self._own(t, "output")
        self.outputs.append((name or f"out{len(self.outputs)}", t.id))

    def _gather(self, cache, pages: PageTable) -> Tensor:
        self._own(cache, "paged_gather")
        for t in (pages.indptr, pages.indices, pages.last_page_len):
            if t is not None:
                self._own(t, "PageTable")
        if pages.ragged and cache.ndim not in (2, 3):
            raise SpecError(f"paged_gather: ragged rows must be (total[, heads], d), got {cache!r}")
        if not pages.ragged and cache.ndim not in (3, 4):
            raise SpecError("paged_gather: cache must be (num_pages, page_size[, heads], d), "
                            f"got {cache!r}")
        if not any(t is pages for t in self.tables):
            self.tables.append(pages)
        heads = cache.dims[1 if pages.ragged else 2:-1]  # () for a headless (MLA latent) cache
        return self._rec("paged_gather", (cache.id,),
                         [(pages.requests, *heads, pages.length, cache.dims[-1])], cache.dtype,
                         pages=next(i for i, t in enumerate(self.tables) if t is pages))

    def _view(self, a, dims) -> Tensor:
        self._own(a, "view")
        return self._rec("view", (a.id,), [tuple(dims)], a.dtype, trans=a._trans)

    def _matmul(self, a, b) -> Tensor:
        self._own(a, "matmul"), self._own(b, "matmul")
        if a.ndim != b.ndim or a.ndim < 2:
            raise SpecError(f"matmul rank mismatch: {a!r} @ {b!r}")
        return self._rec("matmul", (a.id, b.id), [a.dims[:-2] + (a.dims[-2], b.dims[-1])], a.dtype,
                         trans_A=a._trans, trans_B=b._trans)

    def _binary(self, op, a, b) -> Tensor:
        if op == "sub":
            raise SpecError("tensor - tensor is only lowerable in a dense program")
        ts = [x for x in (a, b) if isinstance(x, Tensor)]
        for t in ts:
            self._own(t, op)
        big = max(ts, key=lambda t: t.ndim)
        args = tuple(x.id if isinstance(x, Tensor) else float(x) for x in (a, b))
        return self._rec(op, args, [big.dims], big.dtype)

    def _unary(self, op, a) -> Tensor:
        self._own(a, op)
        return self._rec(op, (a.id,), [a.dims], a.dtype)

    def _reduction(self, a, dim) -> Tensor:
        self._own(a, "sum")
        dim %= a.ndim
        return self._rec("reduce_sum", (a.id,), [a.dims[:dim] + (1,) + a.dims[dim + 1:]], a.dtype, dim=dim)

    def _softmax(self, x, dim, return_lse, causal=False) -> Tensor | tuple:
        self._own(x, "softmax")
        dim %= x.ndim
        outs = [x.dims] + ([x.dims[:dim] + x.dims[dim + 1:]] if return_lse else [])
        return self._rec("softmax", (x.id,), outs, [x.dtype, f32], dim=dim, causal=bool(causal))

    # ---------------------------------------------------------------- analysis
    def attention(self) -> "PagedAttention":
        return PagedAttention.recognize(self)


def _is_paged_spec(spec) -> bool:
    return isinstance(spec, ScalarSpec) or (isinstance(spec, TensorSpec) and spec.optional)


def _fn_of(fn):
    from .jit import JitFunction

    return fn.fn if isinstance(fn, JitFunction) else fn


def is_paged(obj) -> bool:
    """A ``PagedProgram``, or a function with a ``cl.Scalar`` or optional (``| None``) input."""
    if isinstance(obj, PagedProgram):
        return True
    if isinstance(obj, Program) or not callable(obj):
        return False
    try:
        params = inspect.signature(_fn_of(obj), eval_str=True).parameters.values()
    except (TypeError, ValueError, NameError):
        return False
    return any(_is_paged_spec(p.annotation) for p in params)


def trace_paged(fn, dtype=None, *, name=None, hints: ScheduleHints | None = None,
                split=None) -> PagedProgram:
    from .jit import JitFunction

    if isinstance(fn, PagedProgram):
        if hints is not None:
            fn.hints = hints
        return fn
    if isinstance(fn, JitFunction):
        dtype = dtype or fn.dtype
        name = name or fn.__name__
        hints = hints or fn.options.get("hints")
        split = fn.split if split is None else split
    f = _fn_of(fn)
    prog = PagedProgram(name or f.__name__, as_dtype(dtype) or bf16)
    prog.split = as_splits(split)
    if hints is not None:
        prog.hints = hints
    args = []
    for p in inspect.signature(f, eval_str=True).parameters.values():
        spec = p.annotation
        if isinstance(spec, (TensorSpec, ScalarSpec)):
            args.append(prog.input(p.name, spec))
        elif p.default is not inspect.Parameter.empty:
            args.append(p.default)
        else:
            raise SpecError(f"{f.__name__}: annotate {p.name!r} with cl.Tensor[...] / cl.Scalar[...]")
    out = f(*args)
    items = list(out.items()) if isinstance(out, dict) else [
        (None, o) for o in (out if isinstance(out, (tuple, list)) else (out,))]
    for n, o in items:
        prog.output(o, n)
    return prog


# ----------------------------------------------------------------------------- recognizer
@dataclasses.dataclass(frozen=True)
class PagedAttention:
    """GQA ``softmax((q @ gather(K).mT) * scale) @ gather(V)`` or MLA
    ``softmax((qn @ gather(CKV).mT + qp @ gather(KPE).mT) * scale) @ gather(CKV)``, [+ lse].
    Prefill: the queries are gathered through ragged rows and the softmax is causal."""

    program: PagedProgram
    queries: tuple  # (q,) | (q_nope, q_pe)
    caches: tuple  # (K, V) | (CKV, KPE)
    pages: PageTable
    scale: Param
    out: int  # value id of output 0
    lse: int | None  # value id of output 1
    rows: PageTable | None = None  # the ragged query rows of a prefill
    causal: bool = False

    @classmethod
    def recognize(cls, prog: PagedProgram) -> "PagedAttention":
        def fail(msg):
            raise SpecError(f"{prog.name}: not a paged attention program: {msg}")

        def unview(i):
            while (r := prog.producer(i)) is not None and r.kind == "view":
                i = r.args[0]
            return i

        def rec(i, kind, what):
            r = prog.producer(unview(i))
            if r is None or r.kind != kind:
                fail(f"{what} should be a {kind}, got {r.kind if r else 'an input'}")
            return r

        def param(i, what, scalar=False):
            p = prog.param(i)
            if p is None or p.is_scalar != scalar:
                fail(f"{what} should be a {'cl.Scalar' if scalar else 'cl.Tensor'} argument")
            return p

        def query(i):  # a query argument, or one gathered through ragged rows
            r = prog.producer(unview(i))
            if r is not None and r.kind == "paged_gather":
                if not prog.tables[r.attr("pages")].ragged:
                    fail("queries are gathered through ragged rows (a cl.PageTable without indices)")
                return param(r.args[0], "the query"), r.attr("pages")
            return param(unview(i), "the query"), None

        def score(i):  # q @ gather(cache).mT -> (query, rows, gather rec)
            r = rec(i, "matmul", "a score")
            if r.attr("trans_A") or not r.attr("trans_B"):
                fail("a score matmul must be q @ k.mT")
            return (*query(r.args[0]), rec(r.args[1], "paged_gather", "a key"))

        if len(prog.outputs) not in (1, 2):
            fail(f"expected (out[, lse]) outputs, got {len(prog.outputs)}")
        out = prog.outputs[0][1]
        value = rec(out, "matmul", "the output")
        if value.attr("trans_A") or value.attr("trans_B"):
            fail("the value matmul is transposed")
        sm = rec(value.args[0], "softmax", "the value matmul's lhs")
        v_gather = rec(value.args[1], "paged_gather", "the value matmul's rhs")
        if sm.attr("dim") != prog.value(sm.args[0]).ndim - 1:
            fail("softmax must reduce the last (KV) axis")
        scaled = rec(sm.args[0], "mul", "the softmax input (scores * cl.Scalar)")
        ids = [a for a in scaled.args if isinstance(a, int)]
        scale_ids = [a for a in ids if prog.producer(a) and prog.producer(a).kind == "scalar"]
        if len(scale_ids) != 1 or len(ids) != 2:
            fail("the scores must be scaled by a cl.Scalar argument")
        s = unview(next(a for a in ids if a != scale_ids[0]))
        r = prog.producer(s)
        scores = [score(a) for a in r.args] if r is not None and r.kind == "add" else [score(s)]
        gathers = [g for *_, g in scores] + [v_gather]
        if len({g.attr("pages") for g in gathers}) != 1:
            fail("every cache must be gathered through the same cl.PageTable")
        if len({rows for _, rows, _ in scores}) != 1:
            fail("every query must be gathered through the same ragged rows")
        rows = scores[0][1]
        if rows == v_gather.attr("pages"):
            fail("the queries and the caches need their own tables")
        caches = [param(g.args[0], "a cache") for g in gathers]
        if len(scores) == 2 and caches[2] != caches[0]:
            fail("the latent (MLA) form reads the compressed cache as its values")
        lse = None
        if len(prog.outputs) == 2:
            lse = prog.outputs[1][1]
            if len(sm.outs) != 2 or unview(lse) != sm.outs[1]:
                fail("the second output must be the softmax's lse")
        return cls(prog, tuple(q for q, *_ in scores), tuple(caches[:2]),
                   prog.tables[v_gather.attr("pages")], param(scale_ids[0], "the scale", scalar=True),
                   out, lse, None if rows is None else prog.tables[rows], bool(sm.attr("causal")))

    # ---------------------------------------------------------------- operator facts
    @property
    def is_mla(self) -> bool:
        return len(self.queries) == 2

    @property
    def is_prefill(self) -> bool:
        return self.rows is not None

    @property
    def kind(self) -> str:
        from ..dynamic.paging import (MLA_PAGED_DECODE, MLA_PAGED_PREFILL_CAUSAL, PAGED_DECODE,
                              PAGED_DECODE_SLAB, PAGED_PREFILL_CAUSAL, RAGGED_PREFILL_CAUSAL)

        name = self.program.name
        if self.is_prefill != self.causal:
            raise SpecError(f"{name}: only causal prefill is built: ragged query rows go with "
                            "cl.softmax(..., causal=True)")
        if self.pages.ragged and not self.is_prefill:
            raise SpecError(f"{name}: ragged (unpaged) KV is only built for prefill")
        if self.is_mla:
            if self.pages.ragged:
                raise SpecError(f"{name}: MLA reads a paged cache; ragged MLA is not built")
            return MLA_PAGED_PREFILL_CAUSAL if self.is_prefill else MLA_PAGED_DECODE
        if self.is_prefill:
            return RAGGED_PREFILL_CAUSAL if self.pages.ragged else PAGED_PREFILL_CAUSAL
        return PAGED_DECODE_SLAB if self.page_size > 1 else PAGED_DECODE

    @property
    def query(self) -> Param:
        return self.queries[0]

    @property
    def num_qo_heads(self) -> int:
        return int(self.query.spec.dims[1])

    @property
    def num_kv_heads(self) -> int:
        if self.is_mla:
            return 1
        return int(self.caches[0].spec.dims[1 if self.pages.ragged else 2])

    @property
    def page_size(self) -> int:
        return 1 if self.pages.ragged else int(self.caches[0].spec.dims[1])

    @property
    def head_dim(self) -> int:
        return int(self.query.spec.dims[-1])

    @property
    def pe_dim(self) -> int | None:
        return int(self.queries[1].spec.dims[-1]) if self.is_mla else None

    @property
    def kv_len(self) -> Dim:
        return self.pages.length

    def _table_param(self, t) -> Param | None:
        return None if t is None else self.program.param(t.id)

    @property
    def inputs(self) -> list[Param]:
        """The kernel's inputs, in the traced signature's order (an unread optional dropped)."""
        last = self._table_param(self.pages.last_page_len)
        drop = last if (last is not None and self.page_size == 1) else None
        return [p for p in self.program.params if p is not drop]

    def paging_spec(self, hints: ScheduleHints | None = None):
        from ..dynamic.paging import PAGED_KINDS, MLASpec, PagingSpec

        hints = hints or self.program.hints
        kind = self.kind
        indptr = self._table_param(self.pages.indptr)
        indices = self._table_param(self.pages.indices)
        last = self._table_param(self.pages.last_page_len)
        q, c = self.query, self.caches
        if not self.is_mla and c[0].spec.dims != c[1].spec.dims:
            raise SpecError(f"K and V caches differ: {c[0].spec.dims} vs {c[1].spec.dims}")
        for qq, cc in zip(self.queries, c):
            if qq.spec.dims[-1] != cc.spec.dims[-1]:
                raise SpecError(f"{qq.name} width {qq.spec.dims[-1]} != {cc.name} width {cc.spec.dims[-1]}")
        name = lambda d: d.name if isinstance(d, Dim) else None  # noqa: E731
        fields = dict(
            kind=kind, num_qo_heads=self.num_qo_heads, page_size=self.page_size,
            indptr_input=indptr.kernel_name, cache_inputs=tuple(p.kernel_name for p in c),
            scale_input=self.scale.kernel_name, emit_lse=self.lse is not None,
            **{"tokens_dim" if self.is_prefill else "batch_dim": name(q.spec.dims[0])},
        )
        if self.is_mla or kind in PAGED_KINDS:
            fields.update(index_input=indices.kernel_name, pages_dim=name(c[0].spec.dims[0]),
                          index_dim=name(indices.spec.dims[0]))
        else:
            fields["kv_tokens_dim"] = name(c[0].spec.dims[0])
        if self.is_prefill:
            qo = self._table_param(self.rows.indptr)
            if name(qo.spec.dims[0]) != name(indptr.spec.dims[0]):
                raise SpecError(f"{qo.name} and {indptr.name} must share one length")
            fields.update(query_indptr_input=qo.kernel_name, causal=True)
        indptr_dim = name(indptr.spec.dims[0])
        if indptr_dim and indptr_dim.isidentifier():
            fields["indptr_dim"] = indptr_dim
        if self.is_mla:
            cls, fields["query_inputs"] = MLASpec, tuple(p.kernel_name for p in self.queries)
        else:
            cls = PagingSpec
            fields.update(num_kv_heads=self.num_kv_heads, query_input=q.kernel_name)
        if self.page_size > 1 and not self.is_mla:
            if last is None:
                raise SpecError(f"page_size={self.page_size}: the PageTable needs last_page_len")
            fields["last_page_len_input"] = last.kernel_name
        elif last is not None and (self.is_mla or not last.spec.optional):
            raise SpecError(f"{self.program.name} reads no last_page_len; drop it or make it optional")
        missing = [k for k, val in fields.items() if val is None]
        if missing:
            raise SpecError(f"{self.program.name}: {missing} must be named cl.Dims")
        try:
            spec = cls(**fields, **hints.knobs)
        except TypeError as e:
            raise SpecError(f"{cls.__name__} has no such schedule knob: {e}") from None
        spec.validate()
        return spec

    @property
    def query_rows(self) -> int:
        """The dense body's M extent: a head group (decode, MLA) or a token tile (GQA prefill)."""
        group = self.num_qo_heads // self.num_kv_heads
        rows = max(MIN_QUERY_ROWS, -(-group // MIN_QUERY_ROWS) * MIN_QUERY_ROWS)
        return max(rows, PREFILL_QUERY_ROWS) if self.is_prefill and not self.is_mla else rows

    def dense_body(self) -> "DenseBody":
        """One block's dense body (KV at its hint, scale folded as ``1/sqrt(d)``; paging
        replaces both)."""
        d, rows = self.head_dim, self.query_rows
        kv = Dim(self.kv_len.name, self.kv_len.hint or DEFAULT_KV_HINT, dynamic=True)
        body = DenseBody(f"{self.program.name}.dense", self.program.dtype)
        if self.is_mla:
            qn, ckv = body.input("QN", (rows, d)), body.input("CKV", (kv, d))
            qp, kpe = body.input("QP", (rows, self.pe_dim)), body.input("KPE", (kv, self.pe_dim))
            s, v = (qn @ ckv.mT) + (qp @ kpe.mT), ckv
        else:
            q, k, v = body.input("Q", (rows, d)), body.input("K", (kv, d)), body.input("V", (kv, d))
            s = q @ k.mT
        w = (s / d**0.5).exp()
        body.output((w / w.sum(-1, keepdim=True)) @ v)
        return body

    def output_dims(self) -> list[tuple]:
        return [self.program.value(i).dims for i in (self.out, self.lse) if i is not None]

    def output_dtypes(self) -> list:
        return [self.program.dtype] + ([f32] if self.lse is not None else [])


class DenseBody(Program):
    """A dense ``Program`` replayed onto csopt with the exact calls the demo driver makes."""

    def to_tile_graph(self, split=()):
        import cleave.superoptimizer as csopt

        flags = self.can_split(split)
        graph = csopt.new_tile_graph(split_enabled=[any(f) for f in flags])
        flags, env = iter(flags), {}
        for r in self._records:
            args = [env[a] if isinstance(a, int) else a for a in r.args]
            if r.kind == "input":
                env[r.out] = graph.create_tiled_input(self._values[r.out].shape, next(flags),
                                                      dtype=self.dtype.name)
            elif r.kind == "output":
                graph.create_tiled_output(*args)
            elif r.kind == "matmul":
                env[r.out] = graph.matmul(*args, **{k: True for k in ("trans_A", "trans_B") if r.attr(k)})
            elif r.kind == "reduce_sum":
                env[r.out] = graph.reduce_sum(*args, r.attr("dim"))
            else:
                env[r.out] = getattr(graph, r.kind)(*args)
        return graph



# ----------------------------------------------------------------------------- workloads
@dataclasses.dataclass(frozen=True)
class PagedDecodeWorkloads:
    """Every ``batch`` x ``samples`` log-spaced KV lengths in ``kv_len`` (ragged below)."""

    batch: tuple = (1, 7, 64)
    kv_len: tuple = (1, 8192)
    samples: int = 3
    seed: int = 0

    def axes(self) -> list[tuple[int, int]]:
        lo, hi = self.kv_len
        n = max(1, self.samples)
        lens = sorted({round(lo * (hi / lo) ** (i / max(1, n - 1))) for i in range(n)})
        return [(int(b), int(n)) for b in self.batch for n in lens]

    def lengths(self, batch: int, kv_len: int) -> list[int]:
        import random

        rng = random.Random(hash((self.seed, batch, kv_len)) & 0xFFFFFFFF)
        lo = max(self.kv_len[0], kv_len // 2)
        return [kv_len] + [rng.randint(lo, kv_len) for _ in range(batch - 1)]

    def factory(self, attention: PagedAttention, keep: int = 0):
        from ..dynamic import reference
        from ..runtime.autotune import sampled, workload_key

        page = attention.page_size
        heads = (attention.num_qo_heads, attention.num_kv_heads, attention.head_dim)
        wanted = sampled(self.axes(), keep)

        def gen():
            for b, n in wanted:
                lens = self.lengths(b, n)
                pages = sum(-(-x // page) for x in lens)
                dims = (b, *heads, pages, pages)
                inputs = reference.gqa_paged_decode_inputs(dims, lens, page_size=page)
                yield workload_key((b, n)), inputs, reference.gqa_paged_decode_reference(
                    dims, lens, page_size=page)(*inputs)

        return gen


class workloads:  # noqa: N801 - ``cl.workloads.paged_decode(...)``
    paged_decode = PagedDecodeWorkloads
