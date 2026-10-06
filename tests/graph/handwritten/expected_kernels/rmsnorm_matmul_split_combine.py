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
def ukernel_reduction_axis0_sr(A, B):
    A_ptr = T.alloc_fragment(A.shape, "float16")
    T.copy(A, A_ptr)

    B_ptr = B

    T.reduce_sum(A_ptr, B_ptr, dim=0, clear=False)


@T.macro
def ukernel_div_const_4096_0_1dims_rr(A, C):
    for i0 in T.Parallel(*C.shape):

        C[i0] = A[i0] / 4096.0


@T.macro
def ukernel_sqrt_1dims_rr(A, B):
    for i0 in T.Parallel(*B.shape):
        B[i0] = T.sqrt(A[i0])


@T.macro
def ukernel_div_const_1_0_const_on_left_1dims_rr(A, C):
    for i0 in T.Parallel(*C.shape):

        C[i0] = 1.0 / A[i0]


@T.macro
def ukernel_mul_broadcast_0_2dims_rrr(A, B, C):
    for i0, i1 in T.Parallel(*C.shape):
        C[i0, i1] = A[i0, i1] * B[i0]


@T.macro
def GraphNode_0_iter(
    PlaceHolder_A: T.Tensor((2, 128), "float16"),
    PlaceHolder_B: T.Tensor((2, 128, 128), "float16"),
    t_rowsum_combine_0: T.Tensor((128), "float32"),
    t_mma_combine_0: T.Tensor((128, 128), "float32"),
):
    PlaceHolder_AA = T.alloc_shared(
        (
            2,
            128,
        ),
        "float16",
    )
    T.copy(PlaceHolder_A, PlaceHolder_AA)
    ukernel_reduction_axis0_sr(PlaceHolder_AA, t_rowsum_combine_0)
    PlaceHolder_BB = T.alloc_shared(
        (
            2,
            128,
            128,
        ),
        "float16",
    )
    T.copy(PlaceHolder_B, PlaceHolder_BB)
    ukernel_reduction_axis0_sr(PlaceHolder_BB, t_mma_combine_0)


@T.macro
def GraphNode_0(
    PlaceHolder_A: T.Tensor((4, 128), "float16"),
    PlaceHolder_B: T.Tensor((4, 128, 128), "float16"),
    t_rowsum_combine_0: T.Tensor((128), "float32"),
    t_mma_combine_0: T.Tensor((128, 128), "float32"),
):
    for i0 in T.Pipelined(2, num_stages=2):
        GraphNode_0_iter(
            PlaceHolder_A=_S(PlaceHolder_A[2 * i0 : 2 * i0 + 2, 0 : 0 + 128], squeeze_dim=[]),
            PlaceHolder_B=_S(
                PlaceHolder_B[2 * i0 : 2 * i0 + 2, 0 : 0 + 128, 0 : 0 + 128], squeeze_dim=[]
            ),
            t_rowsum_combine_0=t_rowsum_combine_0,
            t_mma_combine_0=t_mma_combine_0,
        )


@T.macro
def main_kernel_iter(
    PlaceHolder_A: T.Tensor((4, 128), "float16"),
    PlaceHolder_B: T.Tensor((4, 128, 128), "float16"),
    OutputNode: T.Tensor((128, 128), "float16"),
):
    t_rowsum_combine_0 = T.alloc_fragment((128,), "float32")
    T.clear(t_rowsum_combine_0)
    t_mma_combine_0 = T.alloc_fragment(
        (
            128,
            128,
        ),
        "float32",
    )
    T.clear(t_mma_combine_0)
    GraphNode_0(PlaceHolder_A, PlaceHolder_B, t_rowsum_combine_0, t_mma_combine_0)
    t_div_0 = T.alloc_fragment((128,), "float32")
    ukernel_div_const_4096_0_1dims_rr(t_rowsum_combine_0, t_div_0)
    t_sqrt_0 = T.alloc_fragment((128,), "float32")
    ukernel_sqrt_1dims_rr(t_div_0, t_sqrt_0)
    t_div_1 = T.alloc_fragment((128,), "float32")
    ukernel_div_const_1_0_const_on_left_1dims_rr(t_sqrt_0, t_div_1)
    t_mul_0 = T.alloc_fragment(
        (
            128,
            128,
        ),
        "float16",
    )
    ukernel_mul_broadcast_0_2dims_rrr(t_mma_combine_0, t_div_1, t_mul_0)
    T.copy(t_mul_0, OutputNode)


@T.prim_func
def main_kernel(
    PlaceHolder_A: T.Tensor((4, 1024), "float16"),
    PlaceHolder_B: T.Tensor((4, 1024, 6144), "float16"),
    OutputNode: T.Tensor((1024, 6144), "float16"),
):
    with T.Kernel(8, 48, threads=(128)) as (i0, i1):
        T.use_swizzle(16)
        main_kernel_iter(
            PlaceHolder_A=_S(PlaceHolder_A[0 : 0 + 4, 128 * i0 : 128 * i0 + 128], squeeze_dim=[]),
            PlaceHolder_B=_S(
                PlaceHolder_B[0 : 0 + 4, 128 * i0 : 128 * i0 + 128, 128 * i1 : 128 * i1 + 128],
                squeeze_dim=[],
            ),
            OutputNode=_S(
                OutputNode[128 * i0 : 128 * i0 + 128, 128 * i1 : 128 * i1 + 128], squeeze_dim=[]
            ),
        )
