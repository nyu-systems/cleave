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


def gen_atomic_add(src: str, dst: str, blocked_width=None, rank: int = 2) -> str:
    """Add the tile ``src`` (a fragment) into ``dst`` (a global view of the same extents)
    atomically, so partial sums from every CTA land in one fp32 buffer.

    The tile-region form, not a ``T.Parallel`` of scalar atomics: TileLang vectorises it
    (``AtomicAddx2``/``x4`` for fp32) along the fragment's contiguous elements.

    With ``blocked_width`` (the buffer's last extent N; see ``kernel.inter_blocked_width``)
    the buffer is stored 8x8-blocked: element ``(m, n)`` of each ``[M, N]`` plane lives at
    ``(m // 8) * 8N + (n // 8) * 64 + (m % 8) * 8 + n % 8``. Each warp's vectorised add then
    covers 256 contiguous bytes instead of 8 rows of 32 B (the mma fragment's 8x8 pieces).
    ``dst`` is this iteration's tile of whole rows starting at a multiple of 8, so the same
    arithmetic holds relative to it; TileLang still vectorises the element form.
    """
    if blocked_width is None:
        return f"T.atomic_add({dst}, {src})\n"
    ivs = [f"ai{d}" for d in range(rank)]
    a, b = ivs[-2:]
    if blocked_width == INTER_BLOCK * INTER_BLOCK:
        row, col = f"{a} // 8 * 8 + {b} // 8", f"{a} % 8 * 8 + {b} % 8"
    else:
        q = f"{b} // 8 * 64 + {a} % 8 * 8 + {b} % 8"
        row, col = f"{a} // 8 * 8 + ({q}) // {blocked_width}", f"({q}) % {blocked_width}"
    index = ", ".join(ivs[:-2] + [row, col])
    return (
        f"for {', '.join(ivs)} in T.Parallel(*{src}.shape):\n"
        f"    T.atomic_add({dst}[{index}], {src}[{', '.join(ivs)}])\n"
    )


# Side of the square blocks an inter output's fp32 buffer is stored in (gen_atomic_add).
INTER_BLOCK = 8


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

    # TileLang's swizzled layout needs a rank >= 2 buffer (a 1-D broadcast row of a
    # tensor-core chain is read element-wise, never by the gemm).
    if scope == "s" and is_tc_layout and len(tile_shape) >= 2:
        alloc_code += (
            f"T.annotate_layout({{{tname}: tl.layout.make_swizzled_layout({tname})}})\n"
        )

    return alloc_code
