from math import gcd
from typing import Optional
import tilelang as tl
import tilelang.language as T
import tvm

VEC_WIDTH = 4
DEBUG = False


def _S(
    A: tvm.tirx.stmt.BufferRegion, new_offset_factor: Optional[int] = None, squeeze_dim=None
):  # convert a sliced bufferregion to a tensor
    ranges = A.region
    shape = [r.extent for r in ranges]
    strides = A.buffer.strides
    if len(strides) == 0:  # not specified
        # reversed cumprod on shapes. e.g., shape=[4, 2], strides=[2, 1]
        strides = A.buffer.shape[1:] + [1]
        for i in reversed(range(len(strides) - 1)):
            strides[i] *= strides[i + 1]
    offset = sum([r.min * s for s, r in zip(strides, ranges)])
    # TODO: infer buffer_type from original tensor
    # elem_offset: the starting offset of the first element.
    # offset_factor: a factor that divides elem_offset (though not sure where it's used in TVM?)
    if new_offset_factor is not None:
        offset_factor = gcd(A.buffer.offset_factor, new_offset_factor)
    else:
        offset_factor = A.buffer.offset_factor

    if squeeze_dim is not None:
        for i in squeeze_dim:
            assert (
                shape[i] == 1
            ), f"Can only squeeze dimension of size 1, but got shape {shape} and squeeze_dim {squeeze_dim}"
        shape = [s for i, s in enumerate(shape) if i not in squeeze_dim]
        strides = [s for i, s in enumerate(strides) if i not in squeeze_dim]
        if DEBUG:
            print(f"Squeezed shape: {shape}, strides: {strides}")

    return T.Buffer(
        shape,
        dtype=A.buffer.dtype,
        strides=strides,
        elem_offset=A.buffer.elem_offset + offset,
        offset_factor=offset_factor,
        buffer_type="default",
        align=A.buffer.data_alignment,
        data=A.buffer.data,
        scope=A.buffer.scope(),
    )


PASS_CONFIGS = {
    "tl.enable_fast_math": True,
    "tl.ptxas_register_usage_level": 10,
}


@T.macro
def ukernel_mma_ssr_transB_clear(A, B, C):

    A_ptr = A

    B_ptr = B

    C_ptr = C

    T.clear(C_ptr)

    T.gemm(
        A_ptr,
        B_ptr,
        C_ptr,
        transpose_A=False,
        transpose_B=True,
        policy=T.GemmWarpPolicy.FullRow,
    )


@T.macro
def ukernel_mma_rsr(A, B, C):

    A_ptr = A

    B_ptr = B

    C_ptr = C

    T.gemm(
        A_ptr,
        B_ptr,
        C_ptr,
        transpose_A=False,
        transpose_B=False,
        policy=T.GemmWarpPolicy.FullRow,
    )


@T.macro
def GraphNode_0_iter(
    PlaceHolder_AA: T.Tensor((64, 64), "float16"),
    PlaceHolder_B: T.Tensor((64, 64), "float16"),
    PlaceHolder_C: T.Tensor((64, 64), "float16"),
    t_mma_1: T.Tensor((64, 64), "float32"),
):
    PlaceHolder_BB = T.alloc_shared(
        (
            64,
            64,
        ),
        "float16",
    )
    T.annotate_layout({PlaceHolder_BB: tl.layout.make_swizzled_layout(PlaceHolder_BB)})
    T.copy(PlaceHolder_B, PlaceHolder_BB)
    t_mma_0 = T.alloc_fragment(
        (
            64,
            64,
        ),
        "float32",
    )
    ukernel_mma_ssr_transB_clear(PlaceHolder_AA, PlaceHolder_BB, t_mma_0)
    PlaceHolder_CC = T.alloc_shared(
        (
            64,
            64,
        ),
        "float16",
    )
    T.annotate_layout({PlaceHolder_CC: tl.layout.make_swizzled_layout(PlaceHolder_CC)})
    T.copy(PlaceHolder_C, PlaceHolder_CC)
    t_mma_0_cast = T.alloc_fragment(
        (
            64,
            64,
        ),
        "float16",
    )
    T.copy(t_mma_0, t_mma_0_cast)
    ukernel_mma_rsr(t_mma_0_cast, PlaceHolder_CC, t_mma_1)


@T.macro
def GraphNode_0(
    PlaceHolder_A: T.Tensor((64, 64), "float16"),
    PlaceHolder_B: T.Tensor((4096, 64), "float16"),
    PlaceHolder_C: T.Tensor((4096, 64), "float16"),
    t_mma_1: T.Tensor((64, 64), "float32"),
):
    PlaceHolder_AA = T.alloc_shared(
        (
            64,
            64,
        ),
        "float16",
    )
    T.annotate_layout({PlaceHolder_AA: tl.layout.make_swizzled_layout(PlaceHolder_AA)})
    T.copy(PlaceHolder_A, PlaceHolder_AA)
    for i0 in T.Pipelined(64, num_stages=3):
        GraphNode_0_iter(
            PlaceHolder_AA=PlaceHolder_AA,
            PlaceHolder_B=_S(PlaceHolder_B[64 * i0 : 64 * i0 + 64, 0 : 0 + 64], squeeze_dim=[]),
            PlaceHolder_C=_S(PlaceHolder_C[64 * i0 : 64 * i0 + 64, 0 : 0 + 64], squeeze_dim=[]),
            t_mma_1=t_mma_1,
        )


@T.macro
def main_kernel_iter(
    PlaceHolder_A: T.Tensor((64, 64), "float16"),
    PlaceHolder_B: T.Tensor((4096, 64), "float16"),
    PlaceHolder_C: T.Tensor((4096, 64), "float16"),
    OutputNode: T.Tensor((64, 64), "float16"),
):
    t_mma_1 = T.alloc_fragment(
        (
            64,
            64,
        ),
        "float32",
    )
    T.clear(t_mma_1)
    GraphNode_0(PlaceHolder_A, PlaceHolder_B, PlaceHolder_C, t_mma_1)
    t_mma_1_intermediate = T.alloc_shared(
        (
            64,
            64,
        ),
        "float16",
    )
    T.annotate_layout({t_mma_1_intermediate: tl.layout.make_swizzled_layout(t_mma_1_intermediate)})
    T.copy(t_mma_1, t_mma_1_intermediate)
    T.copy(t_mma_1_intermediate, OutputNode)


@T.prim_func
def main_kernel(
    PlaceHolder_A: T.Tensor((32, 4096, 64), "float16"),
    PlaceHolder_B: T.Tensor((32, 4096, 64), "float16"),
    PlaceHolder_C: T.Tensor((32, 4096, 64), "float16"),
    OutputNode: T.Tensor((32, 4096, 64), "float16"),
):
    with T.Kernel(32, 64, threads=(128)) as (i0, i1):
        T.use_swizzle(16)
        main_kernel_iter(
            PlaceHolder_A=_S(
                PlaceHolder_A[1 * i0 : 1 * i0 + 1, 64 * i1 : 64 * i1 + 64, 0 : 0 + 64],
                squeeze_dim=[0],
            ),
            PlaceHolder_B=_S(
                PlaceHolder_B[1 * i0 : 1 * i0 + 1, 0 : 0 + 4096, 0 : 0 + 64], squeeze_dim=[0]
            ),
            PlaceHolder_C=_S(
                PlaceHolder_C[1 * i0 : 1 * i0 + 1, 0 : 0 + 4096, 0 : 0 + 64], squeeze_dim=[0]
            ),
            OutputNode=_S(
                OutputNode[1 * i0 : 1 * i0 + 1, 64 * i1 : 64 * i1 + 64, 0 : 0 + 64], squeeze_dim=[0]
            ),
        )
