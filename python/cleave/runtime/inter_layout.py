"""Layouts of inter-output fp32 buffers (``INTER_LAYOUTS``) and the casts that undo them.

No ``from __future__ import annotations`` here: TileLang evaluates the prim func's
parameter annotations, which read this module's closure variables.
"""

import functools

import tilelang as tl
import tilelang.language as T


# ``INTER_LAYOUTS`` value of a buffer stored in 8x8 blocks (codegen.kernel.BLOCKED_LAYOUT).
BLOCKED_LAYOUT = "blocked8x8"


@functools.lru_cache(maxsize=None)
def unblock_kernel(shape, acc, dtype):
    """Cast a ``blocked8x8`` buffer to ``dtype`` in row-major order, in one pass.

    Element ``(m, n)`` of each ``[M, N]`` plane is at ``(m // 8) * 8N + (n // 8) * 64 +
    (m % 8) * 8 + n % 8`` (``codegen.tensor_codegen.gen_atomic_add``); the kernel takes the
    buffer as ``[rows * N / 64, 8, 8]`` and returns ``[rows, N]`` (rows = all planes' M).
    """
    width = int(shape[-1])
    rows = 1
    for d in shape[:-1]:
        rows *= int(d)
    block = next(b for b in (128, 64, 32, 16, 8) if rows % b == 0)
    src_t, dst_t = str(acc).removeprefix("torch."), str(dtype).removeprefix("torch.")

    @T.prim_func
    def unblock(
        src: T.Tensor((rows * width // 64, 8, 8), src_t),
        dst: T.Tensor((rows, width), dst_t),
    ):
        with T.Kernel(rows // block, threads=128) as bx:
            for i, j in T.Parallel(block, width):
                r = bx * block + i
                dst[r, j] = T.cast(src[r // 8 * (width // 8) + j // 8, r % 8, j % 8], dst_t)

    return tl.compile(unblock, out_idx=[1])
