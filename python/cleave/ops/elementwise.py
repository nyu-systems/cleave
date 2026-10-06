from __future__ import annotations

import tvm
from tvm import te

from ..graph import ComputeOpNode
from .base import const_token, mgroup_row, mgroup_slot


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


class Add(ComputeOpNode):
    def __init__(
        self,
        dims: tuple[int],
        const: float = None,
        broadcast: int = -1,
        name: str = "add",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        self.name = name
        self.dims = dims
        self.const = const
        self.broadcast = broadcast
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.resolve_operand_layout = None
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

        assert self.broadcast in (-1, 0, 1), "Only support broadcast -1, 0, or 1."
        assert (
            self.const is None or self.broadcast == -1
        ), "Cannot broadcast with const."

    def create_prim_func(self, dims, in_dtype, out_dtype):
        input_shape = dims
        b_shape = (
            (*self.batch_dims, dims[self.broadcast + self.n_batch_dims])
            if self.broadcast != -1
            else dims
        )
        output_shape = dims
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)
        B = (
            te.placeholder(b_shape, name="B", dtype=in_dtype)
            if self.const is None
            else None
        )

        def gen_lambda(*args):
            a_indices = args
            if B is not None:
                if self.broadcast != -1:
                    b_indices = (
                        *args[: self.n_batch_dims],
                        args[self.broadcast + self.n_batch_dims],
                    )
                else:
                    b_indices = args
                return A[a_indices].astype(out_dtype) + B[b_indices].astype(out_dtype)
            else:
                return A[a_indices].astype(out_dtype) + tvm.tirx.const(
                    self.const, dtype=out_dtype
                )

        C = te.compute(output_shape, gen_lambda, name="C")

        args = [A, B, C] if B is not None else [A, C]
        return te.create_prim_func(args)

    def gen_macro(self, **kwargs) -> str:
        _idxs = [f"i{d}" for d in range(len(self.dims) - self.n_batch_dims)]
        _idxs_str = ", ".join(_idxs)
        a_access = f"A[{_idxs_str}]"
        a_ptr_access = f"A_ptr[{_idxs_str}]"
        b_access = (
            f"B[{_idxs_str}]" if self.broadcast == -1 else f"B[{_idxs[self.broadcast]}]"
        )
        b_ptr_access = (
            f"B_ptr[{_idxs_str}]"
            if self.broadcast == -1
            else f"B_ptr[{_idxs[self.broadcast]}]"
        )
        move_a_to_smem, move_b_to_smem = _resolve_binary_operand_layout(
            self.resolve_operand_layout,
            kwargs["mem_scope"],
            rhs_uses_b=self.const is None,
        )
        prelude = _binary_operand_prelude(self.dtype, move_a_to_smem, move_b_to_smem)
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}({f"A, B, C" if self.const is None else f"A, C"}):
{prelude}\
            for {_idxs_str} in T.Parallel(*C.shape):
                {f"C[{_idxs_str}] = {a_ptr_access if move_a_to_smem else a_access} + {b_ptr_access if move_b_to_smem else b_access}" if self.const is None else ""}
                {f"C[{_idxs_str}] = {a_ptr_access if move_a_to_smem else a_access} + {self.const}" if self.const is not None else ""}"""

        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = "ukernel_add"
        if self.const is not None:
            const_name = str(self.const).replace(".", "_").replace("-", "minus_")
            prefix += f"_const_{const_name}"
        prefix += f"_broadcast_{self.broadcast}" if self.broadcast != -1 else ""
        if self.resolve_operand_layout is not None:
            prefix += f"_resolve_{self.resolve_operand_layout}_layout"
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"


class Sub(ComputeOpNode):
    def __init__(
        self,
        dims: tuple[int],
        const: float = None,
        broadcast: int = -1,
        name: str = "sub",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        self.name = name
        self.dims = dims
        self.const = const
        self.broadcast = broadcast
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.resolve_operand_layout = None
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

        assert self.broadcast in (-1, 0, 1), "Only support broadcast -1, 0, or 1."
        assert (
            self.const is None or self.broadcast == -1
        ), "Cannot broadcast with const."

    def create_prim_func(self, dims, in_dtype, out_dtype):
        input_shape = dims
        b_shape = (
            (*self.batch_dims, dims[self.broadcast + self.n_batch_dims])
            if self.broadcast != -1
            else dims
        )
        output_shape = dims
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)
        B = (
            te.placeholder(b_shape, name="B", dtype=in_dtype)
            if self.const is None
            else None
        )

        def gen_lambda(*args):
            a_indices = args
            if B is not None:
                if self.broadcast != -1:
                    b_indices = (
                        *args[: self.n_batch_dims],
                        args[self.broadcast + self.n_batch_dims],
                    )
                else:
                    b_indices = args
                return A[a_indices].astype(out_dtype) - B[b_indices].astype(out_dtype)
            else:
                return A[a_indices].astype(out_dtype) - tvm.tirx.const(
                    self.const, dtype=out_dtype
                )

        C = te.compute(output_shape, gen_lambda, name="C")

        args = [A, B, C] if B is not None else [A, C]
        return te.create_prim_func(args)

    def gen_macro(self, **kwargs) -> str:
        _idxs = [f"i{d}" for d in range(len(self.dims) - self.n_batch_dims)]
        _idxs_str = ", ".join(_idxs)
        a_access = f"A[{_idxs_str}]"
        a_ptr_access = f"A_ptr[{_idxs_str}]"
        b_access = (
            f"B[{_idxs_str}]" if self.broadcast == -1 else f"B[{_idxs[self.broadcast]}]"
        )
        b_ptr_access = (
            f"B_ptr[{_idxs_str}]"
            if self.broadcast == -1
            else f"B_ptr[{_idxs[self.broadcast]}]"
        )
        move_a_to_smem, move_b_to_smem = _resolve_binary_operand_layout(
            self.resolve_operand_layout,
            kwargs["mem_scope"],
            rhs_uses_b=self.const is None,
        )
        prelude = _binary_operand_prelude(self.dtype, move_a_to_smem, move_b_to_smem)
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}({f"A, B, C" if self.const is None else f"A, C"}):
{prelude}\
            for {_idxs_str} in T.Parallel(*C.shape):
                {f"C[{_idxs_str}] = {a_ptr_access if move_a_to_smem else a_access} - {b_ptr_access if move_b_to_smem else b_access}" if self.const is None else ""}
                {f"C[{_idxs_str}] = {a_ptr_access if move_a_to_smem else a_access} - {self.const}" if self.const is not None else ""}"""

        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = "ukernel_sub"
        if self.const is not None:
            const_name = str(self.const).replace(".", "_").replace("-", "minus_")
            prefix += f"_const_{const_name}"
        prefix += f"_broadcast_{self.broadcast}" if self.broadcast != -1 else ""
        if self.resolve_operand_layout is not None:
            prefix += f"_resolve_{self.resolve_operand_layout}_layout"
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"


class Mul(ComputeOpNode):
    def __init__(
        self,
        dims: tuple[int],
        const: float = None,
        broadcast: int = -1,
        name: str = "mul",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        self.name = name
        self.dims = dims
        self.const = const
        self.broadcast = broadcast
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.resolve_operand_layout = None
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

        assert self.broadcast in (-1, 0, 1), "Only support broadcast -1, 0, or 1."
        assert (
            self.const is None or self.broadcast == -1
        ), "Cannot broadcast with const."

    def create_prim_func(self, dims, in_dtype, out_dtype):
        input_shape = dims
        b_shape = (
            (*self.batch_dims, dims[self.broadcast + self.n_batch_dims])
            if self.broadcast != -1
            else dims
        )
        output_shape = dims
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)
        B = (
            te.placeholder(b_shape, name="B", dtype=in_dtype)
            if self.const is None
            else None
        )

        def gen_lambda(*args):
            a_indices = args
            if B is not None:
                if self.broadcast != -1:
                    b_indices = (
                        *args[: self.n_batch_dims],
                        args[self.broadcast + self.n_batch_dims],
                    )
                else:
                    b_indices = args
                return A[a_indices].astype(out_dtype) * B[b_indices].astype(out_dtype)
            else:
                return A[a_indices].astype(out_dtype) * tvm.tirx.const(
                    self.const, dtype=out_dtype
                )

        C = te.compute(output_shape, gen_lambda, name="C")

        args = [A, B, C] if B is not None else [A, C]
        return te.create_prim_func(args)

    def gen_macro(self, **kwargs) -> str:
        _idxs = [f"i{d}" for d in range(len(self.dims) - self.n_batch_dims)]
        _idxs_str = ", ".join(_idxs)
        a_access = f"A[{_idxs_str}]"
        a_ptr_access = f"A_ptr[{_idxs_str}]"
        b_access = (
            f"B[{_idxs_str}]" if self.broadcast == -1 else f"B[{_idxs[self.broadcast]}]"
        )
        b_ptr_access = (
            f"B_ptr[{_idxs_str}]"
            if self.broadcast == -1
            else f"B_ptr[{_idxs[self.broadcast]}]"
        )
        move_a_to_smem, move_b_to_smem = _resolve_binary_operand_layout(
            self.resolve_operand_layout,
            kwargs["mem_scope"],
            rhs_uses_b=self.const is None,
        )
        prelude = _binary_operand_prelude(self.dtype, move_a_to_smem, move_b_to_smem)
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}({f"A, B, C" if self.const is None else f"A, C"}):
{prelude}\
            for {_idxs_str} in T.Parallel(*C.shape):
                {f"C[{_idxs_str}] = {a_ptr_access if move_a_to_smem else a_access} * {b_ptr_access if move_b_to_smem else b_access}" if self.const is None else ""}
                {f"C[{_idxs_str}] = {a_ptr_access if move_a_to_smem else a_access} * {self.const}" if self.const is not None else ""}"""

        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = "ukernel_mul"
        if self.const is not None:
            const_name = str(self.const).replace(".", "_").replace("-", "minus_")
            prefix += f"_const_{const_name}"
        prefix += f"_broadcast_{self.broadcast}" if self.broadcast != -1 else ""
        if self.resolve_operand_layout is not None:
            prefix += f"_resolve_{self.resolve_operand_layout}_layout"
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"


class Div(ComputeOpNode):
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
        self.name = name
        self.dims = dims
        self.const = const
        self.const_on_left = const_on_left
        self.broadcast = broadcast
        self.denom_floor = denom_floor
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.resolve_operand_layout = None
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

        assert self.broadcast in (-1, 0, 1), "Only support broadcast -1, 0, or 1."
        assert (
            self.const is None or self.broadcast == -1
        ), "Cannot broadcast with const."
        assert (
            self.const is not None or not self.const_on_left
        ), "const_on_left requires const."
        assert self.denom_floor is None or self.const is None or self.const_on_left, (
            "denom_floor bounds a tensor divisor; with the constant on the right the "
            "divisor is that constant"
        )

    def create_prim_func(self, dims, in_dtype, out_dtype):
        input_shape = dims
        b_shape = (
            (*self.batch_dims, dims[self.broadcast + self.n_batch_dims])
            if self.broadcast != -1
            else dims
        )
        output_shape = dims
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)
        B = (
            te.placeholder(b_shape, name="B", dtype=in_dtype)
            if self.const is None
            else None
        )

        def gen_lambda(*args):
            a_indices = args
            if B is not None:
                if self.broadcast != -1:
                    b_indices = (
                        *args[: self.n_batch_dims],
                        args[self.broadcast + self.n_batch_dims],
                    )
                else:
                    b_indices = args
                return A[a_indices].astype(out_dtype) / B[b_indices].astype(out_dtype)
            else:
                const_val = tvm.tirx.const(self.const, dtype=out_dtype)
                if self.const_on_left:
                    return const_val / A[a_indices].astype(out_dtype)
                return A[a_indices].astype(out_dtype) / const_val

        C = te.compute(output_shape, gen_lambda, name="C")

        args = [A, B, C] if B is not None else [A, C]
        return te.create_prim_func(args)

    def gen_macro(self, **kwargs) -> str:
        _idxs = [f"i{d}" for d in range(len(self.dims) - self.n_batch_dims)]
        _idxs_str = ", ".join(_idxs)
        a_access = f"A[{_idxs_str}]"
        a_ptr_access = f"A_ptr[{_idxs_str}]"
        b_access = (
            f"B[{_idxs_str}]" if self.broadcast == -1 else f"B[{_idxs[self.broadcast]}]"
        )
        b_ptr_access = (
            f"B_ptr[{_idxs_str}]"
            if self.broadcast == -1
            else f"B_ptr[{_idxs[self.broadcast]}]"
        )

        lhs_is_tensor_a = self.const is None or not self.const_on_left
        rhs_is_tensor_b = self.const is None
        rhs_is_tensor_a = self.const is not None and self.const_on_left
        move_a_to_smem, move_b_to_smem = _resolve_binary_operand_layout(
            self.resolve_operand_layout,
            kwargs["mem_scope"],
            lhs_uses_a=lhs_is_tensor_a,
            rhs_uses_b=rhs_is_tensor_b,
            rhs_uses_a=rhs_is_tensor_a,
        )
        prelude = _binary_operand_prelude(self.dtype, move_a_to_smem, move_b_to_smem)
        a_expr = a_ptr_access if move_a_to_smem else a_access
        b_expr = b_ptr_access if move_b_to_smem else b_access
        floor = (
            (lambda e: e)
            if self.denom_floor is None
            else (lambda e: f"T.max({e}, {self.denom_floor})")
        )
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}({f"A, B, C" if self.const is None else f"A, C"}):
{prelude}\
            for {_idxs_str} in T.Parallel(*C.shape):
                {f"C[{_idxs_str}] = {a_expr} / {floor(b_expr)}" if self.const is None else ""}
                {f"C[{_idxs_str}] = {self.const} / {floor(a_expr)}" if self.const is not None and self.const_on_left else ""}
                {f"C[{_idxs_str}] = {a_expr} / {self.const}" if self.const is not None and not self.const_on_left else ""}"""

        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = "ukernel_div"
        if self.const is not None:
            const_name = str(self.const).replace(".", "_").replace("-", "minus_")
            prefix += f"_const_{const_name}"
        prefix += (
            "_const_on_left" if self.const is not None and self.const_on_left else ""
        )
        prefix += f"_broadcast_{self.broadcast}" if self.broadcast != -1 else ""
        prefix += (
            f"_dfloor_{const_token(self.denom_floor)}"
            if self.denom_floor is not None
            else ""
        )
        if self.resolve_operand_layout is not None:
            prefix += f"_resolve_{self.resolve_operand_layout}_layout"
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"


class Silu(ComputeOpNode):
    def __init__(
        self,
        dims: tuple[int],
        name: str = "silu",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        self.name = name
        self.dims = dims
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
        input_shape = dims
        output_shape = dims
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)

        C = te.compute(
            output_shape,
            lambda *i: A[i].astype(out_dtype) * te.sigmoid(A[i].astype(out_dtype)),
            name="C",
        )

        args = [A, C]
        return te.create_prim_func(args)

    def gen_macro(self, **kwargs) -> str:
        _idxs = [f"i{d}" for d in range(len(self.dims) - self.n_batch_dims)]
        _idxs_str = ", ".join(_idxs)
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, B):
            for {_idxs_str} in T.Parallel(*B.shape):
                B[{_idxs_str}] = A[{_idxs_str}] * T.sigmoid(A[{_idxs_str}])"""

        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = f"ukernel_silu"
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"


class Exp(ComputeOpNode):
    def __init__(
        self,
        dims: tuple[int],
        name: str = "exp",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        self.name = name
        self.dims = dims
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
        input_shape = dims
        output_shape = dims
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)

        B = te.compute(
            output_shape,
            lambda *i: te.exp(A[i].astype(out_dtype)),
            name="B",
        )

        args = [A, B]
        return te.create_prim_func(args)

    def gen_macro(self, **kwargs) -> str:
        _idxs = [f"i{d}" for d in range(len(self.dims) - self.n_batch_dims)]
        _idxs_str = ", ".join(_idxs)
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, B):
            for {_idxs_str} in T.Parallel(*B.shape):
                B[{_idxs_str}] = T.exp(A[{_idxs_str}])"""

        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = f"ukernel_exp"
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"


class RSqrt(ComputeOpNode):
    def __init__(
        self,
        dims: tuple[int],
        name: str = "rsqrt",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        self.name = name
        self.dims = dims
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
        input_shape = dims
        output_shape = dims
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)

        B = te.compute(
            output_shape,
            lambda *i: te.rsqrt(A[i].astype(out_dtype)),
            name="B",
        )

        args = [A, B]
        return te.create_prim_func(args)

    def gen_macro(self, **kwargs) -> str:
        _idxs = [f"i{d}" for d in range(len(self.dims) - self.n_batch_dims)]
        _idxs_str = ", ".join(_idxs)
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, B):
            for {_idxs_str} in T.Parallel(*B.shape):
                B[{_idxs_str}] = T.rsqrt(A[{_idxs_str}])"""

        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = "ukernel_rsqrt"
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"


class Sqrt(ComputeOpNode):
    def __init__(
        self,
        dims: tuple[int],
        name: str = "sqrt",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        self.name = name
        self.dims = dims
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
        input_shape = dims
        output_shape = dims
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)

        B = te.compute(
            output_shape,
            lambda *i: te.sqrt(A[i].astype(out_dtype)),
            name="B",
        )

        args = [A, B]
        return te.create_prim_func(args)

    def gen_macro(self, **kwargs) -> str:
        _idxs = [f"i{d}" for d in range(len(self.dims) - self.n_batch_dims)]
        _idxs_str = ", ".join(_idxs)
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, B):
            for {_idxs_str} in T.Parallel(*B.shape):
                B[{_idxs_str}] = T.sqrt(A[{_idxs_str}])"""

        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = "ukernel_sqrt"
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"


class MaskedExp(ComputeOpNode):
    # A fused ``M = tokens * row_group`` row axis only changes a body that reads the row
    # index, which the non-causal mask does not; the causal subclass overrides this.
    _uses_row_group = False
    # Number of leading `scalar_args` the body binds by position, before the optional
    # runtime scale.
    _n_mask_args = 2

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


class Log2(ComputeOpNode):
    def __init__(
        self,
        dims: tuple[int],
        name: str = "log2",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        self.name = name
        self.dims = dims
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
        B = te.compute(
            dims,
            lambda *i: te.log2(A[i].astype(out_dtype)),
            name="B",
        )
        return te.create_prim_func([A, B])

    def gen_macro(self, **kwargs) -> str:
        _idxs = [f"i{d}" for d in range(len(self.dims) - self.n_batch_dims)]
        _idxs_str = ", ".join(_idxs)
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, B):
            for {_idxs_str} in T.Parallel(*B.shape):
                B[{_idxs_str}] = T.log2(A[{_idxs_str}])"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = "ukernel_log2"
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"


class Copy(ComputeOpNode):
    """Store a tile as it is, casting to the destination dtype.

    The epilogue of a split-KV *partial* kernel: the block's accumulators -- the
    unnormalised value matmul and the exp row sums -- leave the block unchanged, for a
    combine kernel to sum over the splits. It is an op rather than a bare ``T.copy``
    because the tile it writes is not the accumulator's tile: a grouped-query decode block
    accumulates ``query_rows`` rows of which only the head group's ``group`` are real, and
    the macro iterates the *destination's* extent, so it stores exactly those rows the way
    the single-kernel epilogue's ``Div`` does. ``dims`` is that destination tile.
    """

    def __init__(
        self,
        dims: tuple[int],
        name: str = "copy",
        dtype: str = "float32",
        out_dtype: str = "float32",
        n_batch_dims: int = None,
    ):
        self.name = name
        self.dims = dims
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
        B = te.compute(dims, lambda *i: A[i].astype(out_dtype), name="B")
        return te.create_prim_func([A, B])

    def gen_macro(self, **kwargs) -> str:
        _idxs = [f"i{d}" for d in range(len(self.dims) - self.n_batch_dims)]
        _idxs_str = ", ".join(_idxs)
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, B):
            for {_idxs_str} in T.Parallel(*B.shape):
                B[{_idxs_str}] = T.cast(A[{_idxs_str}], B.dtype)"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        prefix = "ukernel_copy"
        prefix += f"_{len(self.dims) - self.n_batch_dims}dims"
        return f"{prefix}_{kwargs['mem_scope']}"


class MaskedDiv(Div):
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
            b_access = f"T.max({b_access}, {self.denom_floor})"
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

