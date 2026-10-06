from __future__ import annotations

import tvm
from tvm import te

from ..graph import ComputeOpNode


class Matmul(ComputeOpNode):
    def __init__(
        self,
        dims: tuple[int],  # (*batch, M, N, K)
        input_batch_dims_A: tuple[int] | None = None,
        input_batch_dims_B: tuple[int] | None = None,
        name: str = "mma",
        transpose_A: bool = False,
        transpose_B: bool = False,
        warp_policy: str = "even",
        dtype: str = "float32",
        out_dtype: str = "float32",
    ):
        assert warp_policy in (
            "rows",
            "cols",
            "even",
        ), f'Unsupported warp policy: "{warp_policy}"'
        assert len(dims) >= 3

        self.name = name
        self.dims = dims
        self.M, self.N, self.K = dims[-3:]
        self.batch_dims = dims[:-3]
        self.n_batch_dims = len(self.batch_dims)
        self.input_batch_dims_A = tuple(
            self.batch_dims if input_batch_dims_A is None else input_batch_dims_A
        )
        self.input_batch_dims_B = tuple(
            self.batch_dims if input_batch_dims_B is None else input_batch_dims_B
        )
        self._validate_input_batch_dims(self.input_batch_dims_A, "A")
        self._validate_input_batch_dims(self.input_batch_dims_B, "B")
        self.transpose_A, self.transpose_B = transpose_A, transpose_B
        self.warp_policy = warp_policy
        self.dtype = dtype
        self.out_dtype = out_dtype
        prim_func = self.create_prim_func(self.dims, dtype, out_dtype)
        is_batch_dim = [True] * self.n_batch_dims + [
            False,
        ] * 2
        super().__init__(prim_func, name=name, is_batch_dim=is_batch_dim)

    def _validate_input_batch_dims(self, input_batch_dims: tuple[int], input_name: str):
        if len(input_batch_dims) != self.n_batch_dims:
            raise ValueError(
                f"Matmul input {input_name} batch rank mismatch: "
                f"expected {self.n_batch_dims}, got {len(input_batch_dims)}"
            )
        for out_dim, in_dim in zip(self.batch_dims, input_batch_dims):
            if in_dim <= 0:
                raise ValueError(
                    f"Matmul input {input_name} has non-positive batch dim: {in_dim}"
                )
            if out_dim % in_dim != 0:
                raise ValueError(
                    f"Matmul input {input_name} batch dim {in_dim} does not divide output "
                    f"batch dim {out_dim}."
                )

    def _map_batch_indices(
        self, batch_indices: tuple, input_batch_dims: tuple[int]
    ) -> tuple:
        mapped = []
        for idx, out_dim, in_dim in zip(
            batch_indices, self.batch_dims, input_batch_dims
        ):
            if out_dim == in_dim:
                mapped.append(idx)
            else:
                mapped.append(tvm.tirx.floordiv(idx, out_dim // in_dim))
        return tuple(mapped)

    def create_prim_func(self, dims, in_dtype, out_dtype):
        M, N, K = dims[-3:]
        batch_dims = dims[:-3]
        A_shape = (
            (*self.input_batch_dims_A, M, K)
            if not self.transpose_A
            else (*self.input_batch_dims_A, K, M)
        )
        B_shape = (
            (*self.input_batch_dims_B, K, N)
            if not self.transpose_B
            else (*self.input_batch_dims_B, N, K)
        )
        output_shape = (*batch_dims, M, N)

        # Create TVM placeholders for input tensors
        A = te.placeholder(A_shape, name="A", dtype=in_dtype)
        B = te.placeholder(B_shape, name="B", dtype=in_dtype)

        # Define a reduction axis for matrix multiplication
        k = te.reduce_axis((0, K), name="k")

        def _compute_matmul(*args):
            """
            Compute function for matrix multiplication.
            """
            i, j = args[-2:]
            batch_indices = args[:-2]
            A_batch_indices = self._map_batch_indices(
                batch_indices, self.input_batch_dims_A
            )
            B_batch_indices = self._map_batch_indices(
                batch_indices, self.input_batch_dims_B
            )
            A_indices = (
                (*A_batch_indices, i, k)
                if not self.transpose_A
                else (*A_batch_indices, k, i)
            )
            B_indices = (
                (*B_batch_indices, k, j)
                if not self.transpose_B
                else (*B_batch_indices, j, k)
            )
            return te.sum(
                A[A_indices].astype(out_dtype) * B[B_indices].astype(out_dtype),
                axis=k,
            )

        # Compute matrix multiplication result
        C = te.compute(
            output_shape,
            fcompute=_compute_matmul,
            name="C",
        )

        args = [A, B, C]
        return te.create_prim_func(args)

    def gen_macro(self, **kwargs) -> str:
        # TODO: improve memory flow (to support smem or gmem output)
        a_scope, b_scope, c_scope = kwargs["mem_scope"]
        assert c_scope == "r", "Matmul only support register accumulation."

        warp_policy = {
            "even": "T.GemmWarpPolicy.Square",
            "rows": "T.GemmWarpPolicy.FullRow",
            "cols": "T.GemmWarpPolicy.FullCol",
        }[self.warp_policy]

        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, B, C):
            {f"A_ptr = T.alloc_shared(A.shape, '{self.dtype}')" if a_scope == "g" else ""}
            {"T.copy(A, A_ptr)" if a_scope == "g" else ""}
            {"A_ptr = A" if a_scope in ("s", "r") else ""}

            {f"B_ptr = T.alloc_shared(B.shape, '{self.dtype}')" if b_scope == "g" else ""}
            {"T.copy(B, B_ptr)" if b_scope == "g" else ""}
            {"B_ptr = B" if b_scope in ("s", "r") else ""}

            C_ptr = C

            {"T.clear(C_ptr)" if kwargs["clear"] else ""}

            T.gemm(
                A_ptr,
                B_ptr,
                C_ptr,
                transpose_A={self.transpose_A},
                transpose_B={self.transpose_B},
                policy={warp_policy},
            )"""

        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        func_name = f"ukernel_mma_{kwargs['mem_scope']}"
        func_name += "_transA" if self.transpose_A else ""
        func_name += "_transB" if self.transpose_B else ""
        func_name += "_clear" if kwargs["clear"] else ""
        return func_name

class Reduction(ComputeOpNode):
    def __init__(
        self,
        dims: tuple[int],
        axis: int = -1,
        name: str = "reduction",
        dtype: str = "float32",
        out_dtype: str = "float32",
    ):
        assert len(dims) >= 1
        self.name = name
        self.dims = dims
        self.axis_arg = axis
        self.axis = axis if axis >= 0 else len(dims) + axis
        assert 0 <= self.axis < len(dims), f"Invalid reduction axis {axis} for {dims}"
        self.output_dims = dims[: self.axis] + dims[self.axis + 1 :]
        input_batch_flags = [False] * len(dims)
        for i in range(max(0, len(dims) - 2)):
            input_batch_flags[i] = True
        output_batch_flags = [
            is_batch for i, is_batch in enumerate(input_batch_flags) if i != self.axis
        ]
        self.batch_dims = tuple(
            dim for dim, is_batch in zip(self.output_dims, output_batch_flags) if is_batch
        )
        self.n_batch_dims = len(self.batch_dims)
        self.dtype = dtype
        self.out_dtype = out_dtype
        prim_func = self.create_prim_func(self.dims, dtype, out_dtype)
        is_batch_dim = output_batch_flags
        input_is_batch_dim = [input_batch_flags]
        super().__init__(
            prim_func,
            name=name,
            is_batch_dim=is_batch_dim,
            input_is_batch_dim=input_is_batch_dim,
        )

    def create_prim_func(self, dims, in_dtype, out_dtype):
        input_shape = dims
        output_shape = dims[: self.axis] + dims[self.axis + 1 :]
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)
        k = te.reduce_axis((0, dims[self.axis]), name="k")

        def _compute_reduction(*args):
            a_indices = []
            out_idx = 0
            for dim_idx in range(len(dims)):
                if dim_idx == self.axis:
                    a_indices.append(k)
                else:
                    a_indices.append(args[out_idx])
                    out_idx += 1
            return te.sum(A[tuple(a_indices)].astype(out_dtype), axis=k)

        C = te.compute(
            output_shape,
            fcompute=_compute_reduction,
            name="C",
        )

        args = [A, C]
        return te.create_prim_func(args)

    def gen_macro(self, **kwargs) -> str:
        # TODO: improve memory flow (to support smem or gmem output)
        a_scope, b_scope = kwargs["mem_scope"]
        assert b_scope == "r", "Reduction only support register accumulation."

        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, B):
            {f"A_ptr = T.alloc_fragment(A.shape, '{self.dtype}')" if a_scope in ("g", "s") else ""}
            {"T.copy(A, A_ptr)" if a_scope in ("g", "s") else ""}
            {"A_ptr = A" if a_scope == "r" else ""}

            B_ptr = B

            T.reduce_sum(A_ptr, B_ptr, dim={self.axis_arg}, clear={kwargs["clear"]})"""

        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        func_name = f"ukernel_reduction_axis{self.axis}_{kwargs['mem_scope']}"
        func_name += "_clear" if kwargs["clear"] else ""
        return func_name


class Accumulate(ComputeOpNode):
    """Adds its input into its own tile, iteration after iteration.

    The elementwise counterpart of a ``Reduction`` left unclear: where that folds a tile
    into a vector every iteration, this keeps the tile whole and leaves the fold to a single
    reduction after the loop. Folding a row is cross-lane work -- the lanes holding a row's
    columns have to exchange partial sums -- so reducing per iteration pays that exchange
    once per tile, while accumulating elementwise is a register add and pays the exchange
    once per block. Only worth it where the loop is long and the tile is small enough that
    keeping it whole costs no more registers than the reduction's temporaries.
    """

    def __init__(
        self,
        dims: tuple[int],
        n_batch_dims: int = 0,
        name: str = "accumulate",
        dtype: str = "float32",
        out_dtype: str = "float32",
    ):
        assert len(dims) >= 1, f"Accumulate needs at least a 1-D tile, got {dims}"
        self.name = name
        self.dims = dims
        self.n_batch_dims = n_batch_dims
        self.batch_dims = tuple(dims[:n_batch_dims])
        self.dtype = dtype
        self.out_dtype = out_dtype
        is_batch_dim = [i < n_batch_dims for i in range(len(dims))]
        prim_func = self.create_prim_func(self.dims, dtype, out_dtype)
        super().__init__(
            prim_func,
            name=name,
            is_batch_dim=is_batch_dim,
            input_is_batch_dim=[is_batch_dim],
        )

    def create_prim_func(self, dims, in_dtype, out_dtype):
        A = te.placeholder(dims, name="A", dtype=in_dtype)
        C = te.compute(dims, lambda *i: A[i].astype(out_dtype), name="C")
        return te.create_prim_func([A, C])

    def gen_macro(self, **kwargs) -> str:
        a_scope, b_scope = kwargs["mem_scope"]
        assert b_scope == "r", "Accumulate only supports register accumulation."
        idx = ", ".join(f"i{d}" for d in range(len(self.dims)))
        op = "=" if kwargs["clear"] else "+="
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, B):
            for {idx} in T.Parallel(*B.shape):
                B[{idx}] {op} A[{idx}]"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        func_name = f"ukernel_accumulate_{len(self.dims)}dims_{kwargs['mem_scope']}"
        func_name += "_clear" if kwargs["clear"] else ""
        return func_name
