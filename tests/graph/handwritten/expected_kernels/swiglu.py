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
def ukernel_mma_ssr(A, B, C):

    A_ptr = A

    B_ptr = B

    C_ptr = C

    T.gemm(
        A_ptr,
        B_ptr,
        C_ptr,
        transpose_A=False,
        transpose_B=False,
        policy=T.GemmWarpPolicy.Square,
    )


@T.macro
def ukernel_silu_2dims_rr(A, B):
    for i0, i1 in T.Parallel(*B.shape):
        B[i0, i1] = A[i0, i1] * T.sigmoid(A[i0, i1])


@T.macro
def ukernel_mul_2dims_rrr(A, B, C):
    for i0, i1 in T.Parallel(*C.shape):
        C[i0, i1] = A[i0, i1] * B[i0, i1]


@T.macro
def GraphNode_0_iter(
    PlaceHolder_A: T.Tensor((128, 32), "float16"),
    PlaceHolder_B: T.Tensor((32, 128), "float16"),
    PlaceHolder_C: T.Tensor((32, 128), "float16"),
    t_mma_0: T.Tensor((128, 128), "float32"),
    t_mma_1: T.Tensor((128, 128), "float32"),
):
    PlaceHolder_AA = T.alloc_shared(
        (
            128,
            32,
        ),
        "float16",
    )
    T.annotate_layout({PlaceHolder_AA: tl.layout.make_swizzled_layout(PlaceHolder_AA)})
    T.copy(PlaceHolder_A, PlaceHolder_AA)
    PlaceHolder_CC = T.alloc_shared(
        (
            32,
            128,
        ),
        "float16",
    )
    T.annotate_layout({PlaceHolder_CC: tl.layout.make_swizzled_layout(PlaceHolder_CC)})
    T.copy(PlaceHolder_C, PlaceHolder_CC)
    ukernel_mma_ssr(PlaceHolder_AA, PlaceHolder_CC, t_mma_0)
    PlaceHolder_BB = T.alloc_shared(
        (
            32,
            128,
        ),
        "float16",
    )
    T.annotate_layout({PlaceHolder_BB: tl.layout.make_swizzled_layout(PlaceHolder_BB)})
    T.copy(PlaceHolder_B, PlaceHolder_BB)
    ukernel_mma_ssr(PlaceHolder_AA, PlaceHolder_BB, t_mma_1)


@T.macro
def GraphNode_0(
    PlaceHolder_A: T.Tensor((128, 4096), "float16"),
    PlaceHolder_B: T.Tensor((4096, 128), "float16"),
    PlaceHolder_C: T.Tensor((4096, 128), "float16"),
    t_mma_0: T.Tensor((128, 128), "float32"),
    t_mma_1: T.Tensor((128, 128), "float32"),
):
    for i0 in T.Pipelined(128, num_stages=3):
        GraphNode_0_iter(
            PlaceHolder_A=_S(PlaceHolder_A[0 : 0 + 128, 32 * i0 : 32 * i0 + 32], squeeze_dim=[]),
            PlaceHolder_B=_S(PlaceHolder_B[32 * i0 : 32 * i0 + 32, 0 : 0 + 128], squeeze_dim=[]),
            PlaceHolder_C=_S(PlaceHolder_C[32 * i0 : 32 * i0 + 32, 0 : 0 + 128], squeeze_dim=[]),
            t_mma_0=t_mma_0,
            t_mma_1=t_mma_1,
        )


@T.macro
def main_kernel_iter(
    PlaceHolder_A: T.Tensor((128, 4096), "float16"),
    PlaceHolder_B: T.Tensor((4096, 128), "float16"),
    PlaceHolder_C: T.Tensor((4096, 128), "float16"),
    OutputNode: T.Tensor((128, 128), "float16"),
):
    t_mma_0 = T.alloc_fragment(
        (
            128,
            128,
        ),
        "float32",
    )
    T.clear(t_mma_0)
    t_mma_1 = T.alloc_fragment(
        (
            128,
            128,
        ),
        "float32",
    )
    T.clear(t_mma_1)
    GraphNode_0(PlaceHolder_A, PlaceHolder_B, PlaceHolder_C, t_mma_0, t_mma_1)
    t_silu_0 = T.alloc_fragment(
        (
            128,
            128,
        ),
        "float32",
    )
    ukernel_silu_2dims_rr(t_mma_0, t_silu_0)
    t_mul_0 = T.alloc_fragment(
        (
            128,
            128,
        ),
        "float32",
    )
    ukernel_mul_2dims_rrr(t_silu_0, t_mma_1, t_mul_0)
    t_mul_0_intermediate = T.alloc_shared(
        (
            128,
            128,
        ),
        "float16",
    )
    T.annotate_layout({t_mul_0_intermediate: tl.layout.make_swizzled_layout(t_mul_0_intermediate)})
    T.copy(t_mul_0, t_mul_0_intermediate)
    T.copy(t_mul_0_intermediate, OutputNode)


@T.prim_func
def main_kernel(
    PlaceHolder_A: T.Tensor((256, 4096), "float16"),
    PlaceHolder_B: T.Tensor((4096, 6144), "float16"),
    PlaceHolder_C: T.Tensor((4096, 6144), "float16"),
    OutputNode: T.Tensor((256, 6144), "float16"),
):
    with T.Kernel(2, 48, threads=(128)) as (i0, i1):
        T.use_swizzle(16)
        main_kernel_iter(
            PlaceHolder_A=_S(
                PlaceHolder_A[128 * i0 : 128 * i0 + 128, 0 : 0 + 4096], squeeze_dim=[]
            ),
            PlaceHolder_B=_S(
                PlaceHolder_B[0 : 0 + 4096, 128 * i1 : 128 * i1 + 128], squeeze_dim=[]
            ),
            PlaceHolder_C=_S(
                PlaceHolder_C[0 : 0 + 4096, 128 * i1 : 128 * i1 + 128], squeeze_dim=[]
            ),
            OutputNode=_S(
                OutputNode[128 * i0 : 128 * i0 + 128, 128 * i1 : 128 * i1 + 128], squeeze_dim=[]
            ),
        )
