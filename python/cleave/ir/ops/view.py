from __future__ import annotations

from typing import List, Sequence

from tvm import te

from ..graph import ViewOpNode


class SplitView(ViewOpNode):
    def __init__(
        self,
        input_shape: Sequence[int],
        *,
        axis: int,
        factor: int,
        name: str = "split_view",
        dtype: str = "float32",
        split_dim_is_batch: bool = True,
        input_is_batch_dim: List[bool] | None = None,
    ) -> None:
        src_shape = tuple(int(x) for x in input_shape)
        ndim = len(src_shape)
        if ndim < 1:
            raise ValueError("SplitView requires rank >= 1")
        axis = axis if axis >= 0 else ndim + axis
        if axis < 0 or axis >= ndim:
            raise ValueError(f"Invalid split axis {axis} for shape {src_shape}")
        if factor <= 0:
            raise ValueError(f"Split factor must be positive, got {factor}")
        if src_shape[axis] % factor != 0:
            raise ValueError(
                f"Input extent {src_shape[axis]} is not divisible by split factor {factor}"
            )

        inner_extent = src_shape[axis] // factor
        base_out = list(src_shape)
        base_out[axis] = inner_extent
        position = ndim + 1 - 3

        out_shape = base_out.copy()
        out_shape.insert(position, factor)

        if input_is_batch_dim is None:
            input_flags = [True] * max(0, ndim - 2) + [False] * min(2, ndim)
        else:
            input_flags = list(input_is_batch_dim)
            if len(input_flags) != ndim:
                raise ValueError(
                    f"input_is_batch_dim rank mismatch: expected {ndim}, got {len(input_flags)}"
                )

        output_flags = input_flags.copy()
        output_flags.insert(position, bool(split_dim_is_batch))

        self.input_shape = src_shape
        self.output_shape = tuple(out_shape)
        self.axis = axis
        self.factor = factor
        self.inner_extent = inner_extent
        self.split_pos = position
        self.inner_pos = axis + (1 if position <= axis else 0)
        self.split_dim_is_batch = bool(split_dim_is_batch)
        self.dtype = dtype
        prim_func = self.create_prim_func(
            self.input_shape, self.output_shape, dtype, dtype
        )

        super().__init__(
            prim_func,
            name=name,
            is_batch_dim=output_flags,
            input_is_batch_dim=[input_flags],
        )

    def create_prim_func(self, input_shape, output_shape, in_dtype, out_dtype):
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)

        def _compute_split(*out_indices):
            src_indices = self.map_output_index_to_input_index(out_indices)
            return A[src_indices].astype(out_dtype)

        B = te.compute(output_shape, _compute_split, name="B")
        return te.create_prim_func([A, B])

    def _propagate_input_tile(self, output_tile: tuple[int, ...]) -> List[int]:
        if len(output_tile) != len(self.output_shape):
            raise ValueError(
                f"SplitView tile rank mismatch: expected {len(self.output_shape)}, got {len(output_tile)}"
            )
        out_tile = list(map(int, output_tile))
        split_tile = out_tile[self.split_pos]
        inner_tile = out_tile[self.inner_pos]

        src_tile = []
        src_idx = 0
        for out_idx, tile_v in enumerate(out_tile):
            if out_idx == self.split_pos:
                continue
            if src_idx == self.axis:
                # Conservative contiguous span upper bound across split chunks.
                span = max(0, split_tile - 1) * self.inner_extent + inner_tile
                src_tile.append(int(span))
            else:
                src_tile.append(int(tile_v))
            src_idx += 1
        return src_tile

    def map_output_index_to_input_index(self, output_indices: Sequence) -> tuple:
        if len(output_indices) != len(self.output_shape):
            raise ValueError(
                f"SplitView index rank mismatch: expected {len(self.output_shape)}, got {len(output_indices)}"
            )
        split_idx = output_indices[self.split_pos]
        inner_idx = output_indices[self.inner_pos]
        src_indices = []
        src_dim = 0
        for out_dim, out_idx in enumerate(output_indices):
            if out_dim == self.split_pos:
                continue
            if src_dim == self.axis:
                src_indices.append(split_idx * self.inner_extent + inner_idx)
            else:
                src_indices.append(out_idx)
            src_dim += 1
        return tuple(src_indices)

    def map_output_tile_to_input_slices(
        self, offsets, tile_sizes, *, source_rank: int | None = None
    ):
        offsets = list(offsets)
        tile_sizes = [int(v) for v in tile_sizes]
        if len(offsets) != len(tile_sizes):
            raise ValueError("offset and tile rank mismatch")

        if len(tile_sizes) == len(self.output_shape):
            out_keep = list(range(len(self.output_shape)))
            in_keep = list(range(len(self.input_shape)))
        else:
            out_keep = [
                i for i, is_batch in enumerate(self.is_batch_dim or []) if not is_batch
            ]
            in_keep = [
                i
                for i, is_batch in enumerate(self.input_is_batch_dim[0])
                if not is_batch
            ]
            if len(out_keep) != len(tile_sizes):
                raise ValueError(
                    "SplitView codegen rank mismatch: cannot reconcile visible output rank"
                )
        if source_rank is not None and len(in_keep) != source_rank:
            raise ValueError(
                f"SplitView source rank mismatch: expected visible rank {len(in_keep)}, got {source_rank}"
            )

        full_out_offsets = ["0"] * len(self.output_shape)
        full_out_tiles = [1] * len(self.output_shape)
        for vis_i, full_i in enumerate(out_keep):
            full_out_offsets[full_i] = str(offsets[vis_i])
            full_out_tiles[full_i] = int(tile_sizes[vis_i])

        split_present = self.split_pos in out_keep
        split_tile = full_out_tiles[self.split_pos] if split_present else 1
        inner_tile = full_out_tiles[self.inner_pos]
        split_off = full_out_offsets[self.split_pos] if split_present else "0"
        inner_off = full_out_offsets[self.inner_pos]

        if split_tile != 1:
            if not (
                split_tile > 1 and inner_tile == self.inner_extent and inner_off == "0"
            ):
                raise NotImplementedError(
                    "SplitView only supports split-dim tile > 1 when the inner split dim is fully covered."
                )

        src_slices_full = []
        src_dim = 0
        for out_dim in range(len(self.output_shape)):
            if out_dim == self.split_pos:
                continue
            off = full_out_offsets[out_dim]
            ts = full_out_tiles[out_dim]
            if src_dim == self.axis:
                start = f"({split_off}) * {self.inner_extent} + ({inner_off})"
                length = (
                    split_tile * self.inner_extent if split_tile != 1 else inner_tile
                )
                src_slices_full.append((start, int(length)))
            else:
                src_slices_full.append((off, int(ts)))
            src_dim += 1

        src_slices = [src_slices_full[i] for i in in_keep]
        squeeze_dims = [i for i, (_, ts) in enumerate(src_slices) if ts == 1]
        return src_slices, squeeze_dims


class PermuteView(ViewOpNode):
    """A pure axis permutation, mirroring ``torch.Tensor.permute(*perm)``.

    Output axis ``i`` is input axis ``perm[i]`` (so ``output_shape[i] ==
    input_shape[perm[i]]``). Like :class:`SplitView` this is a *view*: the emitted
    prim func is only ever walked in torch by the graph executor to build a
    reference -- it is never scheduled or codegened -- so PermuteView carries no
    tiling/codegen machinery (no ``_propagate_input_tile``,
    ``map_output_tile_to_input_slices`` or ``gen_macro``), only the shape/dtype/
    batch-dim metadata and the index mapping that describes the permute.
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        perm: Sequence[int],
        *,
        name: str = "permute_view",
        dtype: str = "float32",
        input_is_batch_dim: List[bool] | None = None,
    ) -> None:
        src_shape = tuple(int(x) for x in input_shape)
        ndim = len(src_shape)
        if ndim < 1:
            raise ValueError("PermuteView requires rank >= 1")

        perm = tuple((p if p >= 0 else ndim + p) for p in (int(v) for v in perm))
        if sorted(perm) != list(range(ndim)):
            raise ValueError(
                f"perm {perm} is not a permutation of range({ndim}) for shape {src_shape}"
            )

        out_shape = tuple(src_shape[perm[i]] for i in range(ndim))

        if input_is_batch_dim is None:
            input_flags = [True] * max(0, ndim - 2) + [False] * min(2, ndim)
        else:
            input_flags = list(input_is_batch_dim)
            if len(input_flags) != ndim:
                raise ValueError(
                    f"input_is_batch_dim rank mismatch: expected {ndim}, got {len(input_flags)}"
                )

        output_flags = [input_flags[perm[i]] for i in range(ndim)]

        self.input_shape = src_shape
        self.output_shape = out_shape
        self.perm = perm
        self.dtype = dtype
        prim_func = self.create_prim_func(
            self.input_shape, self.output_shape, dtype, dtype
        )

        super().__init__(
            prim_func,
            name=name,
            is_batch_dim=output_flags,
            input_is_batch_dim=[input_flags],
        )

    def create_prim_func(self, input_shape, output_shape, in_dtype, out_dtype):
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)

        def _compute_permute(*out_indices):
            src_indices = self.map_output_index_to_input_index(out_indices)
            return A[src_indices].astype(out_dtype)

        B = te.compute(output_shape, _compute_permute, name="B")
        return te.create_prim_func([A, B])

    def map_output_index_to_input_index(self, output_indices: Sequence) -> tuple:
        if len(output_indices) != len(self.output_shape):
            raise ValueError(
                f"PermuteView index rank mismatch: expected {len(self.output_shape)}, got {len(output_indices)}"
            )
        # Output axis i is input axis perm[i], so the input index at axis perm[i]
        # is exactly the output index at axis i (== torch.permute(*perm)).
        src_indices: List = [None] * len(self.perm)
        for out_dim, in_dim in enumerate(self.perm):
            src_indices[in_dim] = output_indices[out_dim]
        return tuple(src_indices)
