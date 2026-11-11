from __future__ import annotations

import operator

import tvm
from tvm import te

from ..graph import ComputeOpNode
from .base import const_literal, const_token, mgroup_row, mgroup_slot


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
    """``C = A <token> B`` (or ``A <token> const``), with B optionally broadcast on one axis."""

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

    def create_prim_func(self, dims, in_dtype, out_dtype):
        nb = self.n_batch_dims
        b_shape = (*self.batch_dims, dims[self.broadcast + nb]) if self.broadcast != -1 else dims
        A = te.placeholder(dims, name="A", dtype=in_dtype)
        B = te.placeholder(b_shape, name="B", dtype=in_dtype) if self.const is None else None

        def gen_lambda(*args):
            a = A[args].astype(out_dtype)
            if B is None:
                return self._te_const(a, tvm.tirx.const(self.const, dtype=out_dtype))
            b_idx = (*args[:nb], args[self.broadcast + nb]) if self.broadcast != -1 else args
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


class Exp(UnaryElementwise):
    kind, _te, _tl = "exp", te.exp, "T.exp({a})"


class RSqrt(UnaryElementwise):
    kind, _te, _tl = "rsqrt", te.rsqrt, "T.rsqrt({a})"


class Sqrt(UnaryElementwise):
    kind, _te, _tl = "sqrt", te.sqrt, "T.sqrt({a})"


class Log2(UnaryElementwise):
    kind, _te, _tl = "log2", te.log2, "T.log2({a})"


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


class MaskedExp(ComputeOpNode):
    # A fused ``M = tokens * row_group`` row axis only changes a body that reads the row
    # index, which the non-causal mask does not; the causal subclass overrides this.
    _uses_row_group = False
    # Number of leading `scalar_args` the body binds by position, before the optional
    # runtime scale.
    _n_mask_args = 2
    _macro_attrs = ("dims", "n_batch_dims", "scale", "scalar_args", "row_group")

    def __init__(
        self,
        dims: tuple[int],
        scale: float | None,
        scalar_args: tuple[str, ...] = ("tile_base", "valid_len"),
        row_group: int = 1,
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
        self.row_group = int(row_group)
        assert self.row_group >= 1, f"row_group must be positive, got {row_group}"
        assert self.row_group == 1 or self._uses_row_group, (
            f"{type(self).__name__}'s mask does not read the row index, so a "
            "row_group would name an attribute its body cannot depend on"
        )
        assert len(scalar_args) == self._n_mask_args + (0 if self.scale is not None else 1), (
            f"{type(self).__name__} takes {self._n_mask_args} mask scalars"
            + ("" if self.scale is not None else " plus the runtime scale")
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
        if self.scale is None:
            return f"T.exp2({access} * {self.scalar_args[-1]})"
        return f"T.exp({access} * {self.scale})"

    def _name_prefix(self, kind: str) -> str:
        scale_name = "rt2" if self.scale is None else const_token(self.scale)
        prefix = f"ukernel_{kind}_scale_{scale_name}"
        prefix += f"_mgrp{self.row_group}" if self.row_group != 1 else ""
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return prefix

    def gen_macro(self, **kwargs) -> str:
        tile_base, valid_len = self.scalar_args[:2]
        params = ", ".join(self.scalar_args)
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, {params}, B):
            for m0, m1 in T.Parallel(*B.shape):
                B[m0, m1] = T.if_then_else(
                    {tile_base} + m1 < {valid_len}, {self._exp_expr("A[m0, m1]")}, 0.0
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
            name=name,
            dtype=dtype,
            out_dtype=out_dtype,
            n_batch_dims=n_batch_dims,
        )

    def gen_macro(self, **kwargs) -> str:
        tile_base, valid_len, diag = self.scalar_args[:3]
        params = ", ".join(self.scalar_args)
        # ``diag`` is the end-aligned causal bound for this query tile's first row:
        # (kv_len - q_len) + q_tile_base, so row m0 may attend up to column diag + m0.
        # When the row axis fuses a head group, every head of one token shares a row.
        row = mgroup_row("m0", self.row_group)
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, {params}, B):
            for m0, m1 in T.Parallel(*B.shape):
                B[m0, m1] = T.if_then_else(
                    ({tile_base} + m1 < {valid_len})
                    & ({tile_base} + m1 <= {diag} + {row}),
                    {self._exp_expr("A[m0, m1]")},
                    0.0,
                )"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        return f"{self._name_prefix('causal_maskedexp')}_{kwargs['mem_scope']}"


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

