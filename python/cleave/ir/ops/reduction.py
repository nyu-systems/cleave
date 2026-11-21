from __future__ import annotations

import tvm
from tvm import te

from ..graph import ComputeOpNode, PlaceHolderNode
from .base import MacroOp


class Matmul(ComputeOpNode):
    _macro_attrs = ("transpose_A", "transpose_B", "warp_policy", "dtype")

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
        # The tiling policy keys tensor-core tiling on this tag; declaring it here means
        # search_tiling's own set_tag is a no-op instead of a mutation of its input.
        self.set_tag("tensorcore_config", True)

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

class ReduceSum(ComputeOpNode):
    """Sum over ``axis``; ``ReduceMax`` swaps the reducer.

    The reduced axis is held twice, in two index spaces:

    * ``flat_axis`` / ``flat_axis_arg``: the axis the node was built with, over its full
      ``dims`` (``flat_axis`` counted from the front, ``flat_axis_arg`` as given, so
      possibly from the end). They never change. Anything that asks *which axis* the node
      reduces reads these; on a scheduled graph, the place of that axis in the tier tile
      of the value the node reads is ``node.position_in(tile)`` (the tile numbers the dims
      of the value it is a tile of, which are not always ``dims``: ``operand_dims``).
    * ``axis`` / ``axis_arg``: what is emitted. ``axis_arg`` is the ``dim`` given to
      ``T.reduce_*`` and ``axis`` is in the macro's name. On a flat graph they are the
      built axis. On a scheduled graph ``axis_arg`` indexes the buffer the input is
      *stored* in, which has every unit extent of its tile squeezed out
      (``codegen.operands.kept_extents``): neither ``dims`` nor a tier's
      ``Iterator.tile_size``, which keeps the unit extents of the dims it carries. An axis
      built from the front is counted from the front of that buffer, and ``axis`` is the
      same number; one built from the end is counted from its end, and ``axis`` stays the
      built axis. ``apply_tiling`` sets both for every reduction it schedules; a reduction
      added to a scheduled graph afterwards is given them by
      ``cleave.schedule.tiling.rebase_reduction_axis``.
    * ``stored_axis``: the reduced axis counted from the front of that stored buffer,
      however the axis was named. It is set together with the emitted pair and never
      emitted. Anything that asks *where in the tile* a scheduled reduction reduces ("the
      leading stored axis") reads this: ``axis`` answers that for an axis built from the
      front only, since for one built from the end it stays the built axis.
    """

    _macro_attrs = ("axis", "axis_arg", "dtype")
    kind = "reduction"  # macro-name stem, pinned: generated kernels keep it
    _tl_reduce = "T.reduce_sum"
    # Fill value of a loop accumulator before its first iteration (None: T.clear).
    accum_init = None

    @staticmethod
    def _te_reduce(expr, axis):
        return te.sum(expr, axis=axis)

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
        # The axis as built (see the class docstring): the tiler derives ``axis`` /
        # ``axis_arg`` from these two, which never change.
        self.flat_axis, self.flat_axis_arg = self.axis, axis
        self.stored_axis = self.axis  # a flat graph stores a tensor as it is
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

    def operand_dims(self) -> tuple:
        """``(dims, rank)``: for each dim of the tensor this node reduces (``self.dims``),
        which dim of the value it reads that is, in that value's own numbering (``None``
        where the value has none), and the rank of that value.

        They are the same dims one for one, with one exception. ``apply_tiling`` takes the
        split views out of a graph, so a reduction that read a ``SplitView`` reads the
        input behind it: a tensor of one dim less, which has no split dim (the grid walks
        the splits, a block holds one of them). A ``SplitView`` puts the split dim in
        front of the last two dims of its output, so the dims in front of it keep their
        place and the two behind it are one place further back. A tile of that input -- a
        tier's ``Iterator``, an entry of a candidate's tile maps -- is over the input's
        dims: asked where dim 2 of ``(f, R, C / f)`` is, it has to be asked for dim 1 of
        ``(R, C)``.

        A value whose rank differs in any other way is not something ``apply_tiling``
        builds; its dims are matched from the end, as ``Iterator.position`` matches a
        tile that records no ``axes``.
        """
        rank = len(self.dims)
        edge = self.inputs[0]
        src = edge.src_node
        own = len(src.get_shape(edge.src_id)) if isinstance(src, PlaceHolderNode) else rank
        if own == rank - 1:
            split = max(rank - 3, 0)  # ``SplitView.split_pos`` for an output of this rank
            dims = [d if d < split else None if d == split else d - 1 for d in range(rank)]
        else:
            dims = [d - (rank - own) if d >= rank - own else None for d in range(rank)]
        return dims, own

    def position_in(self, it, dim=None):
        """Where dim ``dim`` of the tensor this node reduces (the reduced axis, by
        default) sits in ``it``, the ``Iterator`` of the value it reads in the tier that
        stores it; ``None`` if the tile does not carry it: it then holds one element.
        """
        dims, own = self.operand_dims()
        dim = dims[self.flat_axis if dim is None else dim]
        return None if dim is None else it.position(dim, own)

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
            return self._te_reduce(A[tuple(a_indices)].astype(out_dtype), k)

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
        assert b_scope == "r", "ReduceSum only support register accumulation."

        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, B):
            {f"A_ptr = T.alloc_fragment(A.shape, '{self.dtype}')" if a_scope in ("g", "s") else ""}
            {"T.copy(A, A_ptr)" if a_scope in ("g", "s") else ""}
            {"A_ptr = A" if a_scope == "r" else ""}

            B_ptr = B

            {self._tl_reduce}(A_ptr, B_ptr, dim={self.axis_arg}, clear={kwargs["clear"]})"""

        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        func_name = f"ukernel_{self.kind}_axis{self.axis}_{kwargs['mem_scope']}"
        func_name += "_clear" if kwargs["clear"] else ""
        return func_name


class ReduceMax(ReduceSum):
    """Max over ``axis``. Partial maxima combine by max, never by sum."""

    kind = "reduce_max"
    _tl_reduce = "T.reduce_max"
    accum_init = -float("inf")
    running_max = True

    def __init__(self, dims, axis=-1, name=kind, dtype="float32", out_dtype="float32",
                 accum_init=None):
        super().__init__(dims, axis=axis, name=name, dtype=dtype, out_dtype=out_dtype)
        # Per instance, so the class default (the identity of max) is untouched. Not a
        # macro attribute: the fill is emitted where the accumulator is allocated.
        if accum_init is not None:
            self.accum_init = accum_init

    @staticmethod
    def _te_reduce(expr, axis):
        return te.max(expr, axis=axis)


class Accumulate(ComputeOpNode):
    """Adds its input into its own tile, iteration after iteration.

    The elementwise counterpart of a ``ReduceSum`` left unclear: where that folds a tile
    into a vector every iteration, this keeps the tile whole and leaves the fold to a single
    reduction after the loop. Folding a row is cross-lane work -- the lanes holding a row's
    columns have to exchange partial sums -- so reducing per iteration pays that exchange
    once per tile, while accumulating elementwise is a register add and pays the exchange
    once per block. Only worth it where the loop is long and the tile is small enough that
    keeping it whole costs no more registers than the reduction's temporaries.
    """

    _macro_attrs = ("dims",)

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


class AccumulateMax(Accumulate):
    """Keeps the elementwise maximum of its input, iteration after iteration.

    The running max of a loop whose softmax axis is one element per iteration: a split-KV
    combine at ``split_step == 1`` reads one split's max per trip, the split axis of that
    slice is a unit extent the codegen squeezes away, and so there is no axis left for a
    ``ReduceMax`` to fold. What ``Accumulate`` is to ``ReduceSum``, this is to ``ReduceMax``:
    the same carried state, with the loop itself as the fold.

    Like every running max it may be read inside the loop that carries it
    (``running_max``), and it starts at ``accum_init``: the identity of max by default, a
    finite value per instance where the loop also forms ``exp(m_prev - m)``.
    """

    accum_init = -float("inf")
    running_max = True

    def __init__(
        self,
        dims: tuple[int],
        n_batch_dims: int = 0,
        name: str = "accumulate_max",
        dtype: str = "float32",
        out_dtype: str = "float32",
        accum_init: float | None = None,
    ):
        super().__init__(
            dims, n_batch_dims=n_batch_dims, name=name, dtype=dtype, out_dtype=out_dtype
        )
        # Per instance, as for ReduceMax: the class default is untouched, and the start
        # value is no part of the macro (it is the fill of the accumulator's buffer).
        if accum_init is not None:
            self.accum_init = accum_init

    def gen_macro(self, **kwargs) -> str:
        a_scope, b_scope = kwargs["mem_scope"]
        assert b_scope == "r", "AccumulateMax only supports register accumulation."
        idx = ", ".join(f"i{d}" for d in range(len(self.dims)))
        value = f"A[{idx}]" if kwargs["clear"] else f"T.max(B[{idx}], A[{idx}])"
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(A, B):
            for {idx} in T.Parallel(*B.shape):
                B[{idx}] = {value}"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        func_name = f"ukernel_accumulate_max_{len(self.dims)}dims_{kwargs['mem_scope']}"
        func_name += "_clear" if kwargs["clear"] else ""
        return func_name


class Rescale(MacroOp):
    """``ACC *= S`` in place, ``S`` indexed by the leading ``lead`` axes of ``ACC``.

    The in-place half of a rescaled accumulation (``Graph.accum_scale``): the codegen emits
    it on the accumulator's own buffer, right before the accumulator's own macro. It is not
    a node of the dataflow -- it produces no tile of its own -- and one body serves every
    accumulator class (``ReduceSum``, ``Matmul``, ``Accumulate``), which is why the scale is
    declared on the graph rather than being a variant of each of them. ``rank`` and ``lead``
    are tile ranks of accumulator and factor (unit extents already squeezed).

    ``guard``: a thread skips the multiply when every entry of ``S`` it holds is exactly 1
    (``x * 1.0`` is ``x``, so the result is the same bits; any other entry, a NaN included,
    takes the multiply). For a factor that is 1 on most iterations
    (``cleave.ir.rolling.lazy_rolling_nodes``).
    """

    _macro_attrs = ("rank", "lead", "guard")

    def __init__(self, rank: int, lead: int, guard: bool = False):
        assert 1 <= lead <= rank, f"Rescale: factor rank {lead} vs accumulator rank {rank}"
        self.rank, self.lead, self.guard = int(rank), int(lead), bool(guard)

    def gen_macro(self, **kwargs) -> str:
        s_scope, acc_scope = kwargs["mem_scope"]
        assert s_scope == "r" and acc_scope == "r", "Rescale works on register fragments."
        idx = [f"i{d}" for d in range(self.rank)]
        if not self.guard:
            macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(S, ACC):
            for {", ".join(idx)} in T.Parallel(*ACC.shape):
                ACC[{", ".join(idx)}] *= S[{", ".join(idx[: self.lead])}]"""
            return self._format_code(macro_str)
        own = [f"j{d}" for d in range(self.lead)]
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(S, ACC):
            low = T.alloc_var("float32")
            low = 1.0
            for {", ".join(own)} in T.Parallel(*S.shape):
                low = T.min(low, T.if_then_else(S[{", ".join(own)}] == 1.0, 1.0, 0.0))
            if low < 1.0:
                for {", ".join(idx)} in T.Parallel(*ACC.shape):
                    ACC[{", ".join(idx)}] *= S[{", ".join(idx[: self.lead])}]"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        lead = "" if self.lead == self.rank else f"_lead{self.lead}"
        guard = "_if_moved" if self.guard else ""
        return f"ukernel_rescale{guard}{lead}_{self.rank}dims_{kwargs['mem_scope']}"
