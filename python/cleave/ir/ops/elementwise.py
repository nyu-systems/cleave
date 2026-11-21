from __future__ import annotations

import operator
from dataclasses import dataclass

import tvm
from tvm import te

from ..graph import ComputeOpNode
from .base import NEG_FLT_MAX, const_literal, const_token, mgroup_row, mgroup_slot


def _resolve_binary_operand_layout(
    resolve_operand_layout,
    mem_scope,
    *,
    lhs_uses_a: bool = True,
    rhs_uses_b: bool = True,
    rhs_uses_a: bool = False,
):
    assert resolve_operand_layout in (
        None,
        "lhs",
        "rhs",
    ), f"Unsupported resolve_operand_layout value: {resolve_operand_layout}"

    move_a_to_smem = False
    move_b_to_smem = False

    if resolve_operand_layout == "lhs" and lhs_uses_a:
        move_a_to_smem = len(mem_scope) > 0 and mem_scope[0] == "r"
    elif resolve_operand_layout == "rhs":
        if rhs_uses_b:
            move_b_to_smem = len(mem_scope) > 1 and mem_scope[1] == "r"
        elif rhs_uses_a:
            move_a_to_smem = len(mem_scope) > 0 and mem_scope[0] == "r"

    return move_a_to_smem, move_b_to_smem


def _binary_operand_prelude(dtype, move_a_to_smem: bool, move_b_to_smem: bool) -> str:
    lines = []
    if move_a_to_smem:
        lines.append(f"A_ptr = T.alloc_shared(A.shape, '{dtype}')")
        lines.append("T.copy(A, A_ptr)")
    if move_b_to_smem:
        lines.append(f"B_ptr = T.alloc_shared(B.shape, '{dtype}')")
        lines.append("T.copy(B, B_ptr)")
    if not lines:
        return ""
    # The macro body is nested inside a dedented triple-quoted string, so these
    # lines need the function-body indentation before dedent runs.
    return "".join(f"            {line}\n" for line in lines)


def _init_elementwise(op, dims, name, dtype, out_dtype, n_batch_dims):
    op.name = name
    op.dims = dims
    op.dtype = dtype
    op.out_dtype = out_dtype
    if n_batch_dims is None:
        op.batch_dims = dims[:-2]
        op.n_batch_dims = len(op.batch_dims)
    else:
        op.n_batch_dims = n_batch_dims
        op.batch_dims = tuple(dims[:n_batch_dims])
    prim_func = op.create_prim_func(op.dims, dtype, out_dtype)
    is_batch_dim = [True] * op.n_batch_dims + [False] * (len(dims) - op.n_batch_dims)
    ComputeOpNode.__init__(op, prim_func, name=name, is_batch_dim=is_batch_dim)


class BinaryElementwise(ComputeOpNode):
    """``C = A <token> B`` (or ``A <token> const``), with B optionally broadcast on one axis.

    ``broadcast == 0`` with exactly two non-batch dims: B is ``(*batch, M)``, indexed by the
    first non-batch index. With MORE than two non-batch dims B is A without its last axis
    (``B = A.shape[:-1]``), indexed by all but the last index -- the weight of a split
    combine, ``o[s, r, d] * w[s, r]``, where the split axis must stay a loop index of the
    tile. For two non-batch dims the two readings coincide, so no existing node changes.
    """

    kind = ""  # macro-name stem and default node name
    token = ""  # infix operator in the macro body
    py_op = None  # the same operator as a Python callable (te exprs and torch tensors)
    const_on_left = False
    denom_floor = None
    _macro_attrs = (
        "dims",
        "n_batch_dims",
        "const",
        "const_on_left",
        "broadcast",
        "denom_floor",
        "resolve_operand_layout",
        "dtype",
    )

    def __init__(
        self,
        dims: tuple[int],
        const: float = None,
        broadcast: int = -1,
        name: str = None,
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        self.const = const
        self.broadcast = broadcast
        self.resolve_operand_layout = None
        _init_elementwise(
            self, dims, self.kind if name is None else name, dtype, out_dtype, n_batch_dims
        )
        assert self.broadcast in (-1, 0, 1), "Only support broadcast -1, 0, or 1."
        assert (
            self.const is None or self.broadcast == -1
        ), "Cannot broadcast with const."

    def _te_const(self, a, c):
        return type(self).py_op(a, c)

    def _lead(self) -> bool:
        """``broadcast == 0`` over more than two non-batch dims: B lacks only A's last axis."""
        return self.broadcast == 0 and len(self.dims) - self.n_batch_dims > 2

    def create_prim_func(self, dims, in_dtype, out_dtype):
        nb = self.n_batch_dims
        lead = self._lead()
        b_shape = (*self.batch_dims, dims[self.broadcast + nb]) if self.broadcast != -1 else dims
        if lead:
            b_shape = dims[:-1]
        A = te.placeholder(dims, name="A", dtype=in_dtype)
        B = te.placeholder(b_shape, name="B", dtype=in_dtype) if self.const is None else None

        def gen_lambda(*args):
            a = A[args].astype(out_dtype)
            if B is None:
                return self._te_const(a, tvm.tirx.const(self.const, dtype=out_dtype))
            b_idx = (*args[:nb], args[self.broadcast + nb]) if self.broadcast != -1 else args
            if lead:
                b_idx = args[:-1]
            return type(self).py_op(a, B[b_idx].astype(out_dtype))

        C = te.compute(dims, gen_lambda, name="C")
        return te.create_prim_func([A, B, C] if B is not None else [A, C])

    def _body_lines(self, idxs: str, a: str, b: str) -> list:
        tok, c = self.token, self.const
        return [
            f"C[{idxs}] = {a} {tok} {b}" if c is None else "",
            f"C[{idxs}] = {a} {tok} {const_literal(c)}" if c is not None else "",
        ]

    def gen_macro(self, **kwargs) -> str:
        _idxs = [f"i{d}" for d in range(len(self.dims) - self.n_batch_dims)]
        _idxs_str = ", ".join(_idxs)
        b_sub = _idxs_str if self.broadcast == -1 else _idxs[self.broadcast]
        if self._lead():
            b_sub = ", ".join(_idxs[:-1])
        tensor_rhs = self.const is None
        move_a_to_smem, move_b_to_smem = _resolve_binary_operand_layout(
            self.resolve_operand_layout,
            kwargs["mem_scope"],
            lhs_uses_a=tensor_rhs or not self.const_on_left,
            rhs_uses_b=tensor_rhs,
            rhs_uses_a=not tensor_rhs and self.const_on_left,
        )
        prelude = _binary_operand_prelude(self.dtype, move_a_to_smem, move_b_to_smem)
        a_expr = f"{'A_ptr' if move_a_to_smem else 'A'}[{_idxs_str}]"
        b_expr = f"{'B_ptr' if move_b_to_smem else 'B'}[{b_sub}]"
        body = "\n".join(
            f"                {line}" for line in self._body_lines(_idxs_str, a_expr, b_expr)
        )
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}({"A, B, C" if tensor_rhs else "A, C"}):
{prelude}\
            for {_idxs_str} in T.Parallel(*C.shape):
{body}"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = f"ukernel_{self.kind}"
        if self.const is not None:
            prefix += f"_const_{const_token(self.const)}"
            prefix += "_const_on_left" if self.const_on_left else ""
        prefix += f"_broadcast_{self.broadcast}" if self.broadcast != -1 else ""
        if self.denom_floor is not None:
            prefix += f"_dfloor_{const_token(self.denom_floor)}"
        if self.resolve_operand_layout is not None:
            prefix += f"_resolve_{self.resolve_operand_layout}_layout"
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"


class Add(BinaryElementwise):
    kind, token, py_op = "add", "+", operator.add


class Sub(BinaryElementwise):
    kind, token, py_op = "sub", "-", operator.sub


class Mul(BinaryElementwise):
    kind, token, py_op = "mul", "*", operator.mul


class Div(BinaryElementwise):
    kind, token, py_op = "div", "/", operator.truediv

    def __init__(
        self,
        dims: tuple[int],
        const: float = None,
        const_on_left: bool = False,
        broadcast: int = -1,
        denom_floor: float | None = None,
        name: str = "div",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        self.const_on_left = const_on_left
        self.denom_floor = denom_floor
        super().__init__(dims, const, broadcast, name, dtype, out_dtype, n_batch_dims)
        assert (
            self.const is not None or not self.const_on_left
        ), "const_on_left requires const."
        assert self.denom_floor is None or self.const is None or self.const_on_left, (
            "denom_floor bounds a tensor divisor; with the constant on the right the "
            "divisor is that constant"
        )

    def _te_const(self, a, c):
        return c / a if self.const_on_left else a / c

    def _floor(self, e: str) -> str:
        if self.denom_floor is None:
            return e
        return f"T.max({e}, {const_literal(self.denom_floor)})"

    def _body_lines(self, idxs: str, a: str, b: str) -> list:
        c, left = self.const, self.const_on_left
        return [
            f"C[{idxs}] = {a} / {self._floor(b)}" if c is None else "",
            f"C[{idxs}] = {const_literal(c)} / {self._floor(a)}" if c is not None and left else "",
            f"C[{idxs}] = {a} / {const_literal(c)}" if c is not None and not left else "",
        ]


class UnaryElementwise(ComputeOpNode):
    """``B = f(A)``: ``_te`` builds the te expression, ``_tl`` the TileLang one (``{a}``)."""

    kind = ""
    _te = None
    _tl = ""
    _te_out = "B"  # name of the te output tensor
    _macro_attrs = ("dims", "n_batch_dims")

    def __init__(
        self,
        dims: tuple[int],
        name: str = None,
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        _init_elementwise(
            self, dims, self.kind if name is None else name, dtype, out_dtype, n_batch_dims
        )

    def create_prim_func(self, dims, in_dtype, out_dtype):
        A = te.placeholder(dims, name="A", dtype=in_dtype)
        fn = type(self)._te
        out = te.compute(dims, lambda *i: fn(A[i].astype(out_dtype)), name=self._te_out)
        return te.create_prim_func([A, out])

    def gen_macro(self, **kwargs) -> str:
        _idxs_str = ", ".join(f"i{d}" for d in range(len(self.dims) - self.n_batch_dims))
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, B):
            for {_idxs_str} in T.Parallel(*B.shape):
                B[{_idxs_str}] = {self._tl.format(a=f"A[{_idxs_str}]")}"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        return f"ukernel_{self.kind}_{len(self.dims) - self.n_batch_dims}dims_{kwargs['mem_scope']}"


class Silu(UnaryElementwise):
    kind, _tl, _te_out = "silu", "{a} * T.sigmoid({a})", "C"
    _te = lambda x: x * te.sigmoid(x)  # noqa: E731


@dataclass(frozen=True)
class Stabilize:
    """Annotation of an ``Exp`` whose unknown positive scale provably cancels (a softmax).

    The exp may then be evaluated as ``exp(x - max(x))`` without changing the graph's
    outputs. Plain data only: it lives on the node, and ``Graph.clone`` copies node
    attributes shallowly, so the record reaches the tiled clone as it is.

    ``axes``     the softmax axis of the Exp -- exactly one -- counted from the END
                 (``-1`` = last): a loop tier drops leading unit dims, so a front-relative
                 index would not survive tiling. The shift is the max over it.
    ``scaled``   ``(node name, degree)`` of every reduction / contraction that sums a
                 scaled value over the softmax axis: the nodes whose accumulators carry
                 the scale across loop iterations and are rescaled when the max moves.
                 The degree is always 1.
    ``rolling``  ``""`` when the softmax axis may be tiled by a loop (a rolling max);
                 otherwise the reason it may not, and only tilings that hold the whole
                 axis in one tile are served.
    ``export``   name of the ``ReduceMax`` sibling that already computes the shift (a
                 split partial exports it), to be reused instead of adding another.
    """

    axes: tuple
    scaled: tuple = ()
    rolling: str = ""
    export: str | None = None


class Exp(UnaryElementwise):
    kind, _te, _tl = "exp", te.exp, "T.exp({a})"
    # None: a plain exp. Not a macro attribute: the macro body is `T.exp` either way, the
    # shift is a separate node in front of it.
    stabilize: Stabilize | None = None


class RSqrt(UnaryElementwise):
    kind, _te, _tl = "rsqrt", te.rsqrt, "T.rsqrt({a})"


class Sqrt(UnaryElementwise):
    kind, _te, _tl = "sqrt", te.sqrt, "T.sqrt({a})"


class Log2(UnaryElementwise):
    kind, _te, _tl = "log2", te.log2, "T.log2({a})"


class Exp2(UnaryElementwise):
    """Base-2 exponential: the base the paged softmax works in (``exp2`` is one instruction
    where ``exp`` is a call). Its rescale factor ``alpha = exp2(m_prev - m)`` is built from
    this class so that it uses the same base as ``P = exp2(s - m)``."""

    kind, _te, _tl = "exp2", tvm.tirx.exp2, "T.exp2({a})"


class Lse2(ComputeOpNode):
    """``B = log2(A) + M``: the base-2 log-sum-exp of a softmax whose exp was shifted.

    ``A`` is the row sum ``l = sum_j exp2(s_j - m)`` and ``M`` the shift ``m`` (the row max
    of the scores, in the log2 domain), so ``log2(sum_j exp2(s_j)) = log2(l) + m``. ``l`` is
    read as it is: a row with no live key has ``l = 0`` and ``m`` at its finite start value,
    which gives the ``-inf`` the operator defines.
    """

    kind = "lse2"
    _macro_attrs = ("dims", "n_batch_dims")

    def __init__(
        self,
        dims: tuple[int],
        name: str = None,
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        _init_elementwise(
            self, dims, self.kind if name is None else name, dtype, out_dtype, n_batch_dims
        )

    def create_prim_func(self, dims, in_dtype, out_dtype):
        A = te.placeholder(dims, name="A", dtype=in_dtype)
        M = te.placeholder(dims, name="M", dtype=in_dtype)
        B = te.compute(
            dims,
            lambda *i: te.log2(A[i].astype(out_dtype)) + M[i].astype(out_dtype),
            name="B",
        )
        return te.create_prim_func([A, M, B])

    def gen_macro(self, **kwargs) -> str:
        _idxs_str = ", ".join(f"i{d}" for d in range(len(self.dims) - self.n_batch_dims))
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, M, B):
            for {_idxs_str} in T.Parallel(*B.shape):
                B[{_idxs_str}] = T.log2(A[{_idxs_str}]) + M[{_idxs_str}]"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        return f"ukernel_{self.kind}_{len(self.dims) - self.n_batch_dims}dims_{kwargs['mem_scope']}"


class Copy(UnaryElementwise):
    """Store a tile as it is, casting to the destination dtype.

    The epilogue of a split-KV *partial* kernel: the block's accumulators -- the
    unnormalised value matmul and the exp row sums -- leave the block unchanged, for a
    combine kernel to sum over the splits. It is an op rather than a bare ``T.copy``
    because the tile it writes is not the accumulator's tile: a grouped-query decode block
    accumulates ``query_rows`` rows of which only the head group's ``group`` are real, and
    the macro iterates the *destination's* extent, so it stores exactly those rows the way
    the single-kernel epilogue's ``Div`` does. ``dims`` is that destination tile.
    """

    kind, _tl = "copy", "T.cast({a}, B.dtype)"
    _te = lambda x: x  # noqa: E731


class GridFloor(UnaryElementwise):
    """``B = GRID * floor(A / GRID)``: ``A`` rounded down to a multiple of ``GRID``.

    The shift of a lazily rescaled rolling softmax (``cleave.ir.rolling.lazy_rolling_nodes``):
    ``P = exp(s - g(m))`` in place of ``exp(s - m)``, ``m`` the running max. ``g(m)`` moves
    only when ``m`` crosses a multiple of the grid, so the rescale factor
    ``exp(g(m_prev) - g(m))`` is exactly 1 on every other iteration. ``GRID`` is a power of
    two: the rounding is exact in float32 however it is compiled, the two places that take
    it of one value get the same bits, and ``P < e^GRID = 2981`` is far inside float16.
    """

    GRID = 8.0
    kind = "grid_floor"
    _tl = f"T.floor({{a}} * {1.0 / GRID!r}) * {GRID!r}"
    _te = lambda x: te.floor(x * (1.0 / 8.0)) * 8.0  # noqa: E731


def _kv_mask(scalar_args) -> str:
    """The columns of a KV tile inside the request's window (macro indices m0, m1)."""
    tile_base, valid_len = scalar_args[:2]
    return f"{tile_base} + m1 < {valid_len}"


def _causal_mask(scalar_args, row_group: int) -> str:
    """``_kv_mask`` and the end-aligned causal bound of the row.

    ``diag`` is the bound for this query tile's first row: (kv_len - q_len) + q_tile_base,
    so row m0 may attend up to column diag + m0. When the row axis fuses a head group,
    every head of one token shares a row.
    """
    tile_base, valid_len, diag = scalar_args[:3]
    row = mgroup_row("m0", row_group)
    return f"({tile_base} + m1 < {valid_len}) & ({tile_base} + m1 <= {diag} + {row})"


class MaskedExp(ComputeOpNode):
    # A fused ``M = tokens * row_group`` row axis only changes a body that reads the row
    # index, which the non-causal mask does not; the causal subclass overrides this.
    _uses_row_group = False
    # Number of leading `scalar_args` the body binds by position, before the optional
    # runtime scale.
    _n_mask_args = 2
    _macro_attrs = ("dims", "n_batch_dims", "scale", "scalar_args", "row_group", "shift")

    def __init__(
        self,
        dims: tuple[int],
        scale: float | None,
        scalar_args: tuple[str, ...] = ("tile_base", "valid_len"),
        row_group: int = 1,
        shift: bool = False,
        name: str = "maskedexp",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        self.name = name
        self.dims = dims
        # ``scale=None`` takes the scale as a scalar instead of baking it in, which is
        # what lets the kernel accept the operator's declared ``sm_scale`` input. The
        # value then arrives in the log2 domain and the exponential is base-2.
        self.scale = None if scale is None else float(scale)
        # ``shift=True`` is the exact softmax: the operand is a ``MaskedScore`` -- already
        # scaled, in the log2 domain -- and a second input ``M`` holds its running row max,
        # so the body is ``exp2(A - M[row])`` and takes no scale of its own. A masked
        # entry stays the literal 0 through the same predicate either way.
        self.shift = bool(shift)
        assert not self.shift or self.scale is None, (
            f"{type(self).__name__}(shift=True) reads a score that is already scaled, in "
            f"the log2 domain; a baked-in scale={scale} would be applied twice"
        )
        self.row_group = int(row_group)
        assert self.row_group >= 1, f"row_group must be positive, got {row_group}"
        assert self.row_group == 1 or self._uses_row_group, (
            f"{type(self).__name__}'s mask does not read the row index, so a "
            "row_group would name an attribute its body cannot depend on"
        )
        runtime_scale = self.scale is None and not self.shift
        assert len(scalar_args) == self._n_mask_args + (1 if runtime_scale else 0), (
            f"{type(self).__name__} takes {self._n_mask_args} mask scalars"
            + (" plus the runtime scale" if runtime_scale else "")
            + f", got {scalar_args}"
        )
        self.scalar_args = tuple(scalar_args)
        self.dtype = dtype
        self.out_dtype = out_dtype
        if n_batch_dims is None:
            self.batch_dims = dims[:-2]
            self.n_batch_dims = len(self.batch_dims)
        else:
            self.n_batch_dims = n_batch_dims
            self.batch_dims = tuple(dims[:n_batch_dims])
        prim_func = self.create_prim_func(self.dims, dtype, out_dtype)
        is_batch_dim = [True] * self.n_batch_dims + [False] * (
            len(dims) - self.n_batch_dims
        )
        super().__init__(prim_func, name=name, is_batch_dim=is_batch_dim)

    def create_prim_func(self, dims, in_dtype, out_dtype):
        A = te.placeholder(dims, name="A", dtype=in_dtype)
        if self.shift:
            # The row max: the operand's shape without its last (softmax) axis.
            M = te.placeholder(dims[:-1], name="M", dtype=in_dtype)
            B = te.compute(
                dims,
                lambda *i: te.exp(A[i].astype(out_dtype) - M[i[:-1]].astype(out_dtype)),
                name="B",
            )
            return te.create_prim_func([A, M, B])
        # The te function is a rank/dtype stand-in only; a runtime scale has no value
        # here, and 1.0 keeps the expression well formed.
        scale = 1.0 if self.scale is None else self.scale
        B = te.compute(
            dims,
            lambda *i: te.exp(
                A[i].astype(out_dtype) * tvm.tirx.const(scale, dtype=out_dtype)
            ),
            name="B",
        )
        return te.create_prim_func([A, B])

    def _exp_expr(self, access: str) -> str:
        if self.shift:
            return f"T.exp2({access} - M[m0])"
        if self.scale is None:
            return f"T.exp2({access} * {self.scalar_args[-1]})"
        return f"T.exp({access} * {self.scale})"

    def _operands(self) -> str:
        return "A, M" if self.shift else "A"

    def _name_prefix(self, kind: str) -> str:
        if self.shift:
            # shift2: the operand is shifted by its row max, base 2; there is no scale.
            prefix = f"ukernel_{kind}_shift2"
        else:
            scale_name = "rt2" if self.scale is None else const_token(self.scale)
            prefix = f"ukernel_{kind}_scale_{scale_name}"
        prefix += f"_mgrp{self.row_group}" if self.row_group != 1 else ""
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return prefix

    def gen_macro(self, **kwargs) -> str:
        params = ", ".join(self.scalar_args)
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}({self._operands()}, {params}, B):
            for m0, m1 in T.Parallel(*B.shape):
                B[m0, m1] = T.if_then_else(
                    {_kv_mask(self.scalar_args)}, {self._exp_expr("A[m0, m1]")}, 0.0
                )"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        return f"{self._name_prefix('maskedexp')}_{kwargs['mem_scope']}"


class CausalMaskedExp(MaskedExp):
    _uses_row_group = True
    _n_mask_args = 3

    def __init__(
        self,
        dims: tuple[int],
        scale: float | None,
        scalar_args: tuple[str, ...] = ("tile_base", "valid_len", "diag"),
        row_group: int = 1,
        shift: bool = False,
        name: str = "causal_maskedexp",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        super().__init__(
            dims,
            scale,
            scalar_args=scalar_args,
            row_group=row_group,
            shift=shift,
            name=name,
            dtype=dtype,
            out_dtype=out_dtype,
            n_batch_dims=n_batch_dims,
        )

    def gen_macro(self, **kwargs) -> str:
        params = ", ".join(self.scalar_args)
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}({self._operands()}, {params}, B):
            for m0, m1 in T.Parallel(*B.shape):
                B[m0, m1] = T.if_then_else(
                    {_causal_mask(self.scalar_args, self.row_group)},
                    {self._exp_expr("A[m0, m1]")},
                    0.0,
                )"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        return f"{self._name_prefix('causal_maskedexp')}_{kwargs['mem_scope']}"


class MaskedScore(ComputeOpNode):
    """The operand of a shifted ``MaskedExp``: ``A * scale`` where the mask keeps the
    column, the most negative finite float32 where it does not.

    The exact paged softmax takes its running max over this tile, so the max is taken
    *after* scaling: the scale is a runtime scalar of unknown sign, and
    ``max(s) * scale == max(s * scale)`` only for a positive one. A masked entry is the
    finite ``NEG_FLT_MAX`` rather than ``-inf``: it is also where the carried max starts, so
    a row with nothing live keeps a finite max and ``exp2(m_prev - m)`` never sees
    ``(-inf) - (-inf)``. The predicate is the one the exp uses, so a column is masked in
    both or in neither. ``scalar_args`` are the mask scalars and then the scale.
    """

    _uses_row_group = False
    _n_mask_args = 2
    _macro_attrs = ("dims", "n_batch_dims", "scalar_args", "row_group")
    kind = "maskedscore"

    def __init__(
        self,
        dims: tuple[int],
        scalar_args: tuple[str, ...] = ("tile_base", "valid_len", "scale2"),
        row_group: int = 1,
        name: str = None,
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        self.row_group = int(row_group)
        assert self.row_group >= 1, f"row_group must be positive, got {row_group}"
        assert self.row_group == 1 or self._uses_row_group, (
            f"{type(self).__name__}'s mask does not read the row index, so a "
            "row_group would name an attribute its body cannot depend on"
        )
        assert len(scalar_args) == self._n_mask_args + 1, (
            f"{type(self).__name__} takes {self._n_mask_args} mask scalars plus the "
            f"runtime scale, got {scalar_args}"
        )
        assert out_dtype == "float32", (
            f"a masked score is {NEG_FLT_MAX}, a float32 value; {out_dtype} cannot hold it"
        )
        self.scalar_args = tuple(scalar_args)
        _init_elementwise(
            self, dims, self.kind if name is None else name, dtype, out_dtype, n_batch_dims
        )

    def create_prim_func(self, dims, in_dtype, out_dtype):
        # A rank/dtype stand-in, as for MaskedExp: the runtime scale has no value here.
        A = te.placeholder(dims, name="A", dtype=in_dtype)
        B = te.compute(dims, lambda *i: A[i].astype(out_dtype), name="B")
        return te.create_prim_func([A, B])

    def _mask(self) -> str:
        return _kv_mask(self.scalar_args)

    def gen_macro(self, **kwargs) -> str:
        params = ", ".join(self.scalar_args)
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, {params}, B):
            for m0, m1 in T.Parallel(*B.shape):
                B[m0, m1] = T.if_then_else(
                    {self._mask()},
                    A[m0, m1] * {self.scalar_args[-1]},
                    {const_literal(NEG_FLT_MAX)},
                )"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = f"ukernel_{self.kind}_scale_rt2"
        prefix += f"_mgrp{self.row_group}" if self.row_group != 1 else ""
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"


class CausalMaskedScore(MaskedScore):
    """``MaskedScore`` under the causal mask of ``CausalMaskedExp``."""

    _uses_row_group = True
    _n_mask_args = 3
    kind = "causal_maskedscore"

    def __init__(
        self,
        dims: tuple[int],
        scalar_args: tuple[str, ...] = ("tile_base", "valid_len", "diag", "scale2"),
        row_group: int = 1,
        name: str = None,
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        super().__init__(
            dims,
            scalar_args=scalar_args,
            row_group=row_group,
            name=name,
            dtype=dtype,
            out_dtype=out_dtype,
            n_batch_dims=n_batch_dims,
        )

    def _mask(self) -> str:
        return _causal_mask(self.scalar_args, self.row_group)


class MaskedDiv(Div):
    _macro_attrs = (
        "dims",
        "n_batch_dims",
        "broadcast",
        "denom_floor",
        "scalar_args",
        "row_group",
        "fuse_dst",
    )

    def __init__(
        self,
        dims: tuple[int],
        broadcast: int = -1,
        scalar_args: tuple[str, ...] = ("n_rows",),
        denom_floor: float | None = None,
        row_group: int = 1,
        fuse_dst: bool = True,
        name: str = "maskeddiv",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        assert (
            len(scalar_args) == 1
        ), f"MaskedDiv takes a single row-count scalar, got {scalar_args}"
        super().__init__(
            dims,
            broadcast=broadcast,
            denom_floor=denom_floor,
            name=name,
            dtype=dtype,
            out_dtype=out_dtype,
            n_batch_dims=n_batch_dims,
        )
        # The source's row axis is a fused ``M = tokens * row_group``, while the
        # destination keeps the operator's own (token, head, dim) axes -- so the store
        # splits the row index back apart.
        self.row_group = int(row_group)
        # ...unless the destination tile spans a single token, in which case codegen has
        # already squeezed that axis away and only the head slot is left to index.
        self.fuse_dst = bool(fuse_dst)
        assert self.row_group >= 1, f"row_group must be positive, got {row_group}"
        assert self.fuse_dst or self.row_group > 1, (
            "fuse_dst only means something for a fused row axis"
        )
        self.scalar_args = tuple(scalar_args)

    def gen_macro(self, **kwargs) -> str:
        _idxs = [f"i{d}" for d in range(len(self.dims) - self.n_batch_dims)]
        _idxs_str = ", ".join(_idxs)
        b_access = (
            f"B[{_idxs_str}]" if self.broadcast == -1 else f"B[{_idxs[self.broadcast]}]"
        )
        if self.denom_floor is not None:
            b_access = f"T.max({b_access}, {const_literal(self.denom_floor)})"
        n_rows = self.scalar_args[0]
        if self.row_group == 1:
            loop_src, row, dst = "C", _idxs[0], f"C[{_idxs_str}]"
        else:
            # The loop has to walk the fused source rows, not the destination's tokens.
            loop_src = "A"
            row = mgroup_row(_idxs[0], self.row_group)
            slot = mgroup_slot(_idxs[0], self.row_group)
            head = [row, slot] if self.fuse_dst else [slot]
            dst = f"C[{', '.join(head + _idxs[1:])}]"
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, B, {n_rows}, C):
            for {_idxs_str} in T.Parallel(*{loop_src}.shape):
                if {row} < {n_rows}:
                    {dst} = A[{_idxs_str}] / {b_access}"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = "ukernel_masked_div"
        prefix += f"_broadcast_{self.broadcast}" if self.broadcast != -1 else ""
        prefix += (
            f"_dfloor_{const_token(self.denom_floor)}"
            if self.denom_floor is not None
            else ""
        )
        prefix += f"_mgrp{self.row_group}" if self.row_group != 1 else ""
        prefix += "" if self.fuse_dst else "_dst2"
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"


class MaskedLog2(Log2):
    _macro_attrs = ("dims", "n_batch_dims", "scalar_args", "row_group", "fuse_dst")

    def __init__(
        self,
        dims: tuple[int],
        scalar_args: tuple[str, ...] = ("n_rows",),
        row_group: int = 1,
        fuse_dst: bool = True,
        name: str = "maskedlog2",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        assert (
            len(scalar_args) == 1
        ), f"MaskedLog2 takes a single row-count scalar, got {scalar_args}"
        super().__init__(
            dims,
            name=name,
            dtype=dtype,
            out_dtype=out_dtype,
            n_batch_dims=n_batch_dims,
        )
        # See MaskedDiv: the source row axis is fused, the destination is not.
        self.row_group = int(row_group)
        self.fuse_dst = bool(fuse_dst)
        assert self.row_group >= 1, f"row_group must be positive, got {row_group}"
        assert self.fuse_dst or self.row_group > 1, (
            "fuse_dst only means something for a fused row axis"
        )
        self.scalar_args = tuple(scalar_args)

    def gen_macro(self, **kwargs) -> str:
        _idxs = [f"i{d}" for d in range(len(self.dims) - self.n_batch_dims)]
        _idxs_str = ", ".join(_idxs)
        n_rows = self.scalar_args[0]
        if self.row_group == 1:
            loop_src, row, dst = "B", _idxs[0], f"B[{_idxs_str}]"
        else:
            loop_src = "A"
            row = mgroup_row(_idxs[0], self.row_group)
            slot = mgroup_slot(_idxs[0], self.row_group)
            head = [row, slot] if self.fuse_dst else [slot]
            dst = f"B[{', '.join(head + _idxs[1:])}]"
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, {n_rows}, B):
            for {_idxs_str} in T.Parallel(*{loop_src}.shape):
                if {row} < {n_rows}:
                    {dst} = T.log2(A[{_idxs_str}])"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = "ukernel_masked_log2"
        prefix += f"_mgrp{self.row_group}" if self.row_group != 1 else ""
        prefix += "" if self.fuse_dst else "_dst2"
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"


class MaskedLse2(Lse2):
    """``Lse2`` written the way ``MaskedLog2`` writes: only the rows below ``n_rows``,
    through the same destination indexing (``row_group``, ``fuse_dst``)."""

    _macro_attrs = ("dims", "n_batch_dims", "scalar_args", "row_group", "fuse_dst")

    def __init__(
        self,
        dims: tuple[int],
        scalar_args: tuple[str, ...] = ("n_rows",),
        row_group: int = 1,
        fuse_dst: bool = True,
        name: str = "maskedlse2",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        assert (
            len(scalar_args) == 1
        ), f"MaskedLse2 takes a single row-count scalar, got {scalar_args}"
        super().__init__(
            dims,
            name=name,
            dtype=dtype,
            out_dtype=out_dtype,
            n_batch_dims=n_batch_dims,
        )
        # See MaskedDiv: the source row axis is fused, the destination is not.
        self.row_group = int(row_group)
        self.fuse_dst = bool(fuse_dst)
        assert self.row_group >= 1, f"row_group must be positive, got {row_group}"
        assert self.fuse_dst or self.row_group > 1, (
            "fuse_dst only means something for a fused row axis"
        )
        self.scalar_args = tuple(scalar_args)

    def gen_macro(self, **kwargs) -> str:
        _idxs = [f"i{d}" for d in range(len(self.dims) - self.n_batch_dims)]
        _idxs_str = ", ".join(_idxs)
        n_rows = self.scalar_args[0]
        if self.row_group == 1:
            loop_src, row, dst = "B", _idxs[0], f"B[{_idxs_str}]"
        else:
            loop_src = "A"
            row = mgroup_row(_idxs[0], self.row_group)
            slot = mgroup_slot(_idxs[0], self.row_group)
            head = [row, slot] if self.fuse_dst else [slot]
            dst = f"B[{', '.join(head + _idxs[1:])}]"
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, M, {n_rows}, B):
            for {_idxs_str} in T.Parallel(*{loop_src}.shape):
                if {row} < {n_rows}:
                    {dst} = T.log2(A[{_idxs_str}]) + M[{_idxs_str}]"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = "ukernel_masked_lse2"
        prefix += f"_mgrp{self.row_group}" if self.row_group != 1 else ""
        prefix += "" if self.fuse_dst else "_dst2"
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"
