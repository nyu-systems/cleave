from __future__ import annotations

from tvm import te

from ..graph import ComputeOpNode
from .base import mgroup_row, mgroup_slot

# Sentinel for a clamp whose bound is the receiving buffer's own leading extent. Needed
# when that extent is symbolic, since no integer literal can express it.
BUFFER_BOUND = "buffer"


def _bound_expr(bound, buffer_name):
    """Resolve a clamp bound to an expression string, or None for no clamp."""
    if bound is None:
        return None
    if bound == BUFFER_BOUND:
        return f"{buffer_name}.shape[0] - 1"
    return str(bound)


def _static_extent(value, fallback):
    """A concrete extent for the te placeholder behind a possibly-symbolic dimension.

    The gather prim_funcs are identity stand-ins used only for rank/dtype inference -- the
    emitted tensor extent comes from the graph's tensor map -- so a symbolic dimension can
    be represented here by any value large enough to keep the placeholder well formed.
    """
    return value if isinstance(value, int) else fallback


def _bound_tag(bound):
    """Macro-name fragment for a clamp bound, so two bounds never share a name."""
    return "buf" if bound == BUFFER_BOUND else str(bound)


def _macro_suffix(*parts):
    """Name fragments for the attributes a gather body is built from.

    A gather body is extent-generic -- every loop runs over ``*DST.shape`` -- so the tile
    extents are deliberately absent and two call sites with different tile shapes share
    one macro, exactly as the rank-tagged elementwise ops do. What the name must carry is
    everything that changes the body, since a name missing one would let the second op
    silently shadow the first (asserted in ``codegen.emit_op_node``).
    """
    return "".join(f"_{p}" for p in parts if p)


def _cache_row_expr(idx_expr, intra_pos, page_size, row_stride, head, clamp_row):
    # A page id addresses ``page_size * row_stride`` cache rows; a token's own row is
    # its intra-page offset scaled by the per-token stride, plus the head slot.
    if page_size == 1:
        row = idx_expr if row_stride == 1 else f"{idx_expr} * {row_stride}"
    else:
        row = f"{idx_expr} * {page_size * row_stride}"
        offset = f"({intra_pos}) % {page_size}"
        row = (
            f"{row} + {offset}"
            if row_stride == 1
            else f"{row} + ({offset}) * {row_stride}"
        )
    if head is not None:
        row = f"{row} + {head}"
    if clamp_row is not None:
        row = f"T.min({row}, {clamp_row})"
    return row


class PagedGather(ComputeOpNode):
    def __init__(
        self,
        dims: tuple[int],
        source_rows: int,
        index_extent: int,
        row_stride: int = 1,
        page_size: int = 1,
        clamp_index: int | None = None,
        clamp_row: int | None = None,
        paged_axes: bool = False,
        source_heads: int = 1,
        head_scalar: bool | None = None,
        scalar_args: tuple[str, ...] = ("ibase",),
        name: str = "paged_gather",
        dtype: str = "float32",
        out_dtype: str = "float32",
    ):
        assert len(dims) == 2, f"PagedGather emits a 2-D tile, got {dims}"
        assert page_size >= 1, f"page_size must be positive, got {page_size}"
        self.name = name
        self.dims = dims
        self.source_rows = source_rows
        self.index_extent = index_extent
        self.row_stride = int(row_stride)
        self.page_size = int(page_size)
        self.clamp_index = clamp_index
        self.clamp_row = clamp_row
        # Address the cache on the operator's own axes -- page, slot within the page,
        # kv head, dim -- instead of flattening it to (rows, dim) with a row stride, so
        # the kernel can take the tensor exactly as the operator declares it.
        #
        # The *rank the macro sees* is not the declared rank: codegen squeezes unit
        # extents out of every tier's view, so a page_size of 1 leaves no slot axis and
        # a single kv head leaves no head axis. The axes are therefore derived, never
        # given as a number.
        self.paged_axes = bool(paged_axes)
        self.source_heads = int(source_heads)
        # A head scalar is independent of an index base: a resolved row tile carries no
        # base, but the cache still has a head slot to pick.
        self.head_scalar = len(scalar_args) > 1 if head_scalar is None else bool(head_scalar)
        self.index_base = len(scalar_args) > (1 if self.head_scalar else 0)
        assert len(scalar_args) == int(self.index_base) + int(self.head_scalar), (
            f"scalar_args must name exactly the base and/or head this op binds, "
            f"got {scalar_args} for base={self.index_base} head={self.head_scalar}"
        )
        assert not (self.paged_axes and self.row_stride != 1), (
            "a multi-axis cache addresses the head directly, so it needs no row stride"
        )
        assert not (self.head_scalar and not self.paged_axes), (
            "a flat (rows, dim) cache folds the head into the row, so a head scalar "
            "here would offset the row twice"
        )
        self.scalar_args = tuple(scalar_args)
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.batch_dims = ()
        self.n_batch_dims = 0
        self.index_inputs = (1,)
        prim_func = self.create_prim_func(self.dims, dtype, out_dtype)
        super().__init__(prim_func, name=name, is_batch_dim=[False, False])

    def _mid_axes(self) -> tuple[int, ...]:
        """Extents between the page axis and the dim axis, as the macro sees them."""
        if not self.paged_axes:
            return ()
        heads = _static_extent(self.source_heads, 1) if self.head_scalar else 1
        return tuple(e for e in (self.page_size, heads) if e > 1)

    def create_prim_func(self, dims, in_dtype, out_dtype):
        rows = _static_extent(self.source_rows, dims[0])
        mid = self._mid_axes()
        A = te.placeholder((rows, *mid, dims[1]), name="A", dtype=in_dtype)
        slots = _static_extent(self.index_extent, dims[0])
        I = te.placeholder((slots,), name="I", dtype="int32")
        zeros = (0,) * len(mid)

        C = te.compute(
            dims,
            lambda i, j: te.if_then_else(
                I[i] >= 0,
                A[(i, *zeros, j)].astype(out_dtype),
                A[(i, *zeros, j)].astype(out_dtype),
            ),
            name="C",
        )
        return te.create_prim_func([A, I, C])

    def gen_macro(self, **kwargs) -> str:
        params = ("base",) if self.index_base else ()
        params += ("head",) if self.head_scalar else ()
        pos = "base + g0" if self.index_base else "g0"
        slot = pos if self.page_size == 1 else f"({pos}) // {self.page_size}"
        idx = f"IDX[{slot}]"
        clamp_index = _bound_expr(self.clamp_index, "IDX")
        if clamp_index is not None:
            idx = f"IDX[T.min({slot}, {clamp_index})]"
        head = "head" if self.head_scalar else None
        if not self.paged_axes:
            row = _cache_row_expr(
                idx,
                pos,
                self.page_size,
                self.row_stride,
                head,
                _bound_expr(self.clamp_row, "SRC"),
            )
            subscript = f"{row}, g1"
        else:
            if self.index_base:
                # Reading the page table directly: the entry IS a page id.
                page, intra = idx, f"({pos}) % {self.page_size}"
            else:
                # Reading a tile an IndexGather already resolved to flat token rows, so
                # the page and the slot within it have to be split back apart.
                page = idx if self.page_size == 1 else f"({idx}) // {self.page_size}"
                intra = f"({idx}) % {self.page_size}"
            clamp_row = _bound_expr(self.clamp_row, "SRC")
            if clamp_row is not None:
                page = f"T.min({page}, {clamp_row})"
            axes = [page]
            if self.page_size > 1:
                axes.append(intra)
            if head is not None:
                axes.append(head)
            subscript = ", ".join(axes + ["g1"])
        signature = ", ".join(("SRC", "IDX", *params, "DST"))
        # The register stage exists to decouple the shape of the global read from the shape
        # of the shared write, so each can take the layout it wants. A destination that is
        # already a fragment (a prefetch buffer) needs no such decoupling, and staging it
        # through a second fragment would just be a register-to-register copy.
        if kwargs["mem_scope"][-1] == "r":
            body = f"""\
            for g0, g1 in T.Parallel(*DST.shape):
                DST[g0, g1] = SRC[{subscript}]"""
        else:
            body = f"""\
            DST_reg = T.alloc_fragment(DST.shape, DST.dtype)
            for g0, g1 in T.Parallel(*DST.shape):
                DST_reg[g0, g1] = SRC[{subscript}]
            T.copy(DST_reg, DST)"""
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}({signature}):
{body}"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        return "ukernel_paged_gather" + _macro_suffix(
            f"ps{self.page_size}" if self.page_size != 1 else None,
            f"stride{self.row_stride}" if self.row_stride != 1 else None,
            "paged" if self.paged_axes else None,
            "base" if self.index_base else None,
            "head" if self.head_scalar else None,
            (
                f"ci{_bound_tag(self.clamp_index)}"
                if self.clamp_index is not None
                else None
            ),
            f"cr{_bound_tag(self.clamp_row)}" if self.clamp_row is not None else None,
            kwargs["mem_scope"],
        )


class IndexGather(ComputeOpNode):
    def __init__(
        self,
        dims: tuple[int],
        index_extent: int,
        row_stride: int,
        clamp_index: int,
        clamp_row: int | None = None,
        page_size: int = 1,
        scalar_args: tuple[str, ...] = ("base", "tile_base", "valid_len"),
        name: str = "index_gather",
        dtype: str = "int32",
        out_dtype: str = "int32",
    ):
        assert len(dims) == 1, f"IndexGather emits a 1-D tile, got {dims}"
        assert page_size >= 1, f"page_size must be positive, got {page_size}"
        assert len(scalar_args) in (
            3,
            4,
        ), (
            "IndexGather takes (base, tile_base, valid_len) and an optional head, "
            f"got {scalar_args}"
        )
        # This macro only ever receives the index table, so a buffer-derived row bound here
        # would read the table's extent where the cache's is meant. Clamp the row where the
        # cache is in scope instead: PagedGather(clamp_row=BUFFER_BOUND).
        assert clamp_row != BUFFER_BOUND, (
            "IndexGather cannot resolve a buffer row bound; pass clamp_row=BUFFER_BOUND "
            "to the PagedGather that reads the cache"
        )
        self.name = name
        self.dims = dims
        self.index_extent = index_extent
        self.row_stride = int(row_stride)
        self.page_size = int(page_size)
        self.clamp_index = clamp_index
        self.clamp_row = clamp_row
        self.scalar_args = tuple(scalar_args)
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.batch_dims = ()
        self.n_batch_dims = 0
        prim_func = self.create_prim_func(self.dims, dtype, out_dtype)
        super().__init__(prim_func, name=name, is_batch_dim=[False])

    def create_prim_func(self, dims, in_dtype, out_dtype):
        slots = _static_extent(self.index_extent, dims[0])
        I = te.placeholder((slots,), name="I", dtype=in_dtype)
        C = te.compute(dims, lambda i: I[i].astype(out_dtype), name="C")
        return te.create_prim_func([I, C])

    def gen_macro(self, **kwargs) -> str:
        base, tile_base, valid_len = "base", "tile_base", "valid_len"
        head = "head" if len(self.scalar_args) > 3 else None
        params = (base, tile_base, valid_len) + ((head,) if head else ())
        pos = f"T.max(T.min({tile_base} + x0, {valid_len} - 1), 0)"
        slot = (
            f"{base} + {pos}"
            if self.page_size == 1
            else f"{base} + ({pos}) // {self.page_size}"
        )
        row = _cache_row_expr(
            f"IDX[T.min({slot}, {_bound_expr(self.clamp_index, 'IDX')})]",
            pos,
            self.page_size,
            self.row_stride,
            head,
            None,
        )
        clamp_row = _bound_expr(self.clamp_row, "IDX")
        row = (
            row
            if clamp_row is None
            else f"T.min(\n                    {row},\n                    {clamp_row},\n                )"
        )
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(IDX, {", ".join(params)}, DST):
            DST_reg = T.alloc_fragment(DST.shape, "int32")
            for x0 in T.Parallel(*DST.shape):
                DST_reg[x0] = {row}
            T.copy(DST_reg, DST)"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        return "ukernel_index_gather" + _macro_suffix(
            f"ps{self.page_size}" if self.page_size != 1 else None,
            f"stride{self.row_stride}" if self.row_stride != 1 else None,
            "head" if len(self.scalar_args) > 3 else None,
            f"ci{_bound_tag(self.clamp_index)}",
            f"cr{_bound_tag(self.clamp_row)}" if self.clamp_row is not None else None,
            kwargs["mem_scope"],
        )


class RowGather(ComputeOpNode):
    def __init__(
        self,
        dims: tuple[int],
        source_rows: int,
        n_valid: int,
        src_dims: int = 2,
        source_heads: int = 1,
        scalar_args: tuple[str, ...] = ("row0",),
        name: str = "row_gather",
        dtype: str = "float32",
        out_dtype: str = "float32",
    ):
        assert len(dims) == 2, f"RowGather emits a 2-D tile, got {dims}"
        self.name = name
        self.dims = dims
        self.source_rows = source_rows
        self.n_valid = int(n_valid)
        # A 3-D source (token, head, dim) is the operator's own q layout: the row axis of
        # the tile then walks the head axis of the source, from a head base scalar.
        self.src_dims = int(src_dims)
        self.source_heads = int(source_heads)
        assert self.src_dims in (2, 3), f"RowGather source is 2-D or 3-D, got {src_dims}"
        assert len(scalar_args) == (1 if self.src_dims == 2 else 2), (
            "RowGather takes (row0), plus a head base when the source is 3-D, "
            f"got {scalar_args}"
        )
        self.scalar_args = tuple(scalar_args)
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.batch_dims = ()
        self.n_batch_dims = 0
        prim_func = self.create_prim_func(self.dims, dtype, out_dtype)
        super().__init__(prim_func, name=name, is_batch_dim=[False, False])

    def create_prim_func(self, dims, in_dtype, out_dtype):
        rows = _static_extent(self.source_rows, dims[0])
        if self.src_dims == 2:
            A = te.placeholder((rows, dims[1]), name="A", dtype=in_dtype)
            C = te.compute(dims, lambda i, j: A[i, j].astype(out_dtype), name="C")
        else:
            A = te.placeholder(
                (rows, _static_extent(self.source_heads, 1), dims[1]),
                name="A",
                dtype=in_dtype,
            )
            C = te.compute(dims, lambda i, j: A[i, 0, j].astype(out_dtype), name="C")
        return te.create_prim_func([A, C])

    def gen_macro(self, **kwargs) -> str:
        row0 = "row0"
        params = (row0,) if self.src_dims == 2 else (row0, "head0")
        clamped = f"T.min(r0, {self.n_valid} - 1)"
        subscript = (
            f"{row0} + {clamped}, r1"
            if self.src_dims == 2
            else f"{row0}, head0 + {clamped}, r1"
        )
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(SRC, {", ".join(params)}, DST):
            DST_reg = T.alloc_fragment(DST.shape, DST.dtype)
            for r0, r1 in T.Parallel(*DST.shape):
                DST_reg[r0, r1] = T.if_then_else(
                    r0 < {self.n_valid},
                    SRC[{subscript}],
                    T.cast(0, DST.dtype),
                )
            T.copy(DST_reg, DST)"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        return "ukernel_row_gather" + _macro_suffix(
            f"valid{self.n_valid}",
            "r3" if self.src_dims == 3 else None,
            kwargs["mem_scope"],
        )


class RangeGather(ComputeOpNode):
    def __init__(
        self,
        dims: tuple[int],
        source_rows: int,
        row_stride: int = 1,
        clamp_row: int | None = None,
        column_offset: bool = False,
        src_dims: int = 2,
        row_group: int = 1,
        source_heads: int = 1,
        scalar_args: tuple[str, ...] = ("row_base", "tile_base", "valid_len", "head"),
        name: str = "range_gather",
        dtype: str = "float32",
        out_dtype: str = "float32",
    ):
        assert len(dims) == 2, f"RangeGather emits a 2-D tile, got {dims}"
        assert len(scalar_args) in (3, 4), (
            "RangeGather takes (row_base, tile_base, valid_len) and an optional head, "
            f"got {scalar_args}"
        )
        self.name = name
        self.dims = dims
        self.source_rows = source_rows
        self.row_stride = int(row_stride)
        self.clamp_row = clamp_row
        self.column_offset = bool(column_offset)
        self.src_dims = int(src_dims)
        # The destination's row axis is a fused ``M = tokens * row_group``: row g0 belongs
        # to token ``g0 // row_group`` and head slot ``g0 % row_group``. That is how one
        # staged KV tile comes to serve a whole query head group.
        self.row_group = int(row_group)
        self.source_heads = int(source_heads)
        assert self.src_dims in (2, 3), f"RangeGather source is 2-D or 3-D, got {src_dims}"
        assert self.row_group >= 1, f"row_group must be positive, got {row_group}"
        assert not (self.row_group > 1 and self.src_dims != 3), (
            "a fused row axis needs the head as its own source axis (src_dims=3), "
            "otherwise the head slot would have to be folded into the column offset"
        )
        assert not (
            self.column_offset and self.row_stride != 1
        ), "column_offset addresses a (row, head * dim) tensor, so the row stride is 1"
        assert not (self.src_dims == 3 and (self.column_offset or self.row_stride != 1)), (
            "a 3-D source addresses the head directly, so it needs neither a column "
            "offset nor a row stride"
        )
        self.scalar_args = tuple(scalar_args)
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.batch_dims = ()
        self.n_batch_dims = 0
        prim_func = self.create_prim_func(self.dims, dtype, out_dtype)
        super().__init__(prim_func, name=name, is_batch_dim=[False, False])

    def create_prim_func(self, dims, in_dtype, out_dtype):
        rows = _static_extent(self.source_rows, dims[0])
        src_shape = (
            (rows, dims[1])
            if self.src_dims == 2
            else (rows, _static_extent(self.source_heads, 1), dims[1])
        )
        A = te.placeholder(src_shape, name="A", dtype=in_dtype)
        if self.src_dims == 2:
            C = te.compute(dims, lambda i, j: A[i, j].astype(out_dtype), name="C")
        else:
            C = te.compute(dims, lambda i, j: A[i, 0, j].astype(out_dtype), name="C")
        return te.create_prim_func([A, C])

    def gen_macro(self, **kwargs) -> str:
        # One kernel gathers both Q and K/V ranges through this op, so the macro names its
        # own parameters instead of the caller's: callers differing only in which enclosing
        # scalars they bind then share a single macro. ``scalar_args`` still drives the call.
        row_base, tile_base, valid_len = "row_base", "tile_base", "valid_len"
        head = "head" if len(self.scalar_args) > 3 else None
        params = (row_base, tile_base, valid_len) + ((head,) if head else ())
        # Rows past the request's last token clamp onto it; the score for those
        # columns is masked to zero downstream, so the read only has to stay in range.
        token = (
            f"{row_base} + T.min({tile_base} + "
            f"{mgroup_row('g0', self.row_group)}, {valid_len} - 1)"
        )
        if self.src_dims == 3:
            # The source is (token, head, dim), exactly as the operator declares it, so
            # the head slot of a fused row axis indexes its own axis.
            slot = mgroup_slot("g0", self.row_group)
            if head is None:
                head_expr = slot
            elif self.row_group == 1:
                head_expr = head
            else:
                head_expr = f"{head} + {slot}"
            row, trailing = token, f"{head_expr}, g1"
        elif self.column_offset:
            # The source is (row, head * dim): the token indexes the row and the head
            # selects a column block, so no row stride is needed.
            row = token
            trailing = "g1" if head is None else f"{head} + g1"
        else:
            row = token if self.row_stride == 1 else f"({token}) * {self.row_stride}"
            if head is not None:
                row = f"{row} + {head}"
            trailing = "g1"
        clamp_row = _bound_expr(self.clamp_row, "SRC")
        if clamp_row is not None:
            row = f"T.min({row}, {clamp_row})"
        subscript = f"{row}, {trailing}"
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(SRC, {", ".join(params)}, DST):
            DST_reg = T.alloc_fragment(DST.shape, DST.dtype)
            for g0, g1 in T.Parallel(*DST.shape):
                DST_reg[g0, g1] = SRC[{subscript}]
            T.copy(DST_reg, DST)"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        return "ukernel_range_gather" + _macro_suffix(
            f"stride{self.row_stride}" if self.row_stride != 1 else None,
            "col" if self.column_offset else None,
            "r3" if self.src_dims == 3 else None,
            f"mgrp{self.row_group}" if self.row_group != 1 else None,
            "head" if len(self.scalar_args) > 3 else None,
            f"cr{_bound_tag(self.clamp_row)}" if self.clamp_row is not None else None,
            kwargs["mem_scope"],
        )
