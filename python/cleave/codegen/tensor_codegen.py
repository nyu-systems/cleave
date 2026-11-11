from __future__ import annotations

from typing import Optional, Tuple


def gen_tensor_type(shape, dtype):
    return f"T.Tensor(({', '.join(map(str, shape))}), '{dtype}')"


def gen_copy(
    src: str,
    dst: str,
    same_level: bool,
    is_tc_layout: bool = False,
    movement: Optional[Tuple] = None,
    tile_shape: Optional[Tuple[int]] = None,
    dtype: Optional[str] = None,
):
    copy_code = ""
    intermediate_scope = {
        ("g", "r"): "s",
        (
            "r",
            "g",
        ): "s",  # TODO: this is only worth it if fragment is not vectorized (e.g., tensor core output layout)
    }.get(movement, None)
    if intermediate_scope is not None and is_tc_layout:
        assert intermediate_scope in [
            "s",
        ], f"Not supported intermediate copy scope: {intermediate_scope}"
        assert not same_level, "same level copy with intermediate copy is not supported"
        assert (
            tile_shape is not None
        ), "tile_shape must be provided for intermediate copy"
        assert dtype is not None, "dtype must be provided for intermediate copy"

        copy_code += gen_tensor_allocation_str(
            tname=src + "_intermediate",
            tile_shape=tile_shape,
            dtype=dtype,
            scope=intermediate_scope,
            is_tc_layout=is_tc_layout,
        )
        copy_code += f"T.copy({src}, {src + '_intermediate'})\n"
        src = src + "_intermediate"
    copy_code += f"T.copy({src}, {dst})\n" if not same_level else f"{dst} = {src}\n"
    return copy_code


def gen_tensor_allocation_str(
    tname: str, tile_shape: tuple, dtype: str, scope: str, is_tc_layout: bool = False
):
    scope_name = {"s": "shared", "g": "global", "r": "fragment", "l": "local"}[scope]
    alloc_code = f"{tname} = T.alloc_{scope_name}(("
    for ts in tile_shape:
        assert isinstance(
            ts, int
        ), f"{tname}: {scope_name} extents must be compile-time constants, got {ts!r}"
        alloc_code += f"{ts}, "
    alloc_code += f"), '{dtype}')\n"

    if scope == "s" and is_tc_layout:
        alloc_code += (
            f"T.annotate_layout({{{tname}: tl.layout.make_swizzled_layout({tname})}})\n"
        )

    return alloc_code
