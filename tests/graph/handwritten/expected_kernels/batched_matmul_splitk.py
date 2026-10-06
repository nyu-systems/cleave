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
def split_ukernel_mma_ssr(A, B, C):

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
def split_GraphNode_0_iter(
    PlaceHolder_A: T.Tensor((128, 32), "float16"),
    PlaceHolder_B: T.Tensor((32, 128), "float16"),
    t_mma_0: T.Tensor((128, 128), "float32"),
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
    PlaceHolder_BB = T.alloc_shared(
        (
            32,
            128,
        ),
        "float16",
    )
    T.annotate_layout({PlaceHolder_BB: tl.layout.make_swizzled_layout(PlaceHolder_BB)})
    T.copy(PlaceHolder_B, PlaceHolder_BB)
    split_ukernel_mma_ssr(PlaceHolder_AA, PlaceHolder_BB, t_mma_0)


@T.macro
def split_GraphNode_0(
    PlaceHolder_A: T.Tensor((128, 1024), "float16"),
    PlaceHolder_B: T.Tensor((1024, 128), "float16"),
    t_mma_0: T.Tensor((128, 128), "float32"),
):
    for i0 in T.Pipelined(32, num_stages=3):
        split_GraphNode_0_iter(
            PlaceHolder_A=_S(PlaceHolder_A[0 : 0 + 128, 32 * i0 : 32 * i0 + 32], squeeze_dim=[]),
            PlaceHolder_B=_S(PlaceHolder_B[32 * i0 : 32 * i0 + 32, 0 : 0 + 128], squeeze_dim=[]),
            t_mma_0=t_mma_0,
        )


@T.macro
def split_main_kernel_iter(
    PlaceHolder_A: T.Tensor((128, 1024), "float16"),
    PlaceHolder_B: T.Tensor((1024, 128), "float16"),
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
    split_GraphNode_0(PlaceHolder_A, PlaceHolder_B, t_mma_0)
    t_mma_0_intermediate = T.alloc_shared(
        (
            128,
            128,
        ),
        "float16",
    )
    T.annotate_layout({t_mma_0_intermediate: tl.layout.make_swizzled_layout(t_mma_0_intermediate)})
    T.copy(t_mma_0, t_mma_0_intermediate)
    T.copy(t_mma_0_intermediate, OutputNode)


@T.macro
def combine_ukernel_reduction_axis1_sr(A, B):
    A_ptr = T.alloc_fragment(A.shape, "float16")
    T.copy(A, A_ptr)

    B_ptr = B

    T.reduce_sum(A_ptr, B_ptr, dim=-3, clear=False)


@T.macro
def combine_GraphNode_0_iter(
    PlaceHolder_A: T.Tensor((2, 128, 128), "float16"),
    t_reduction_0: T.Tensor((128, 128), "float32"),
):
    PlaceHolder_AA = T.alloc_shared(
        (
            2,
            128,
            128,
        ),
        "float16",
    )
    T.copy(PlaceHolder_A, PlaceHolder_AA)
    combine_ukernel_reduction_axis1_sr(PlaceHolder_AA, t_reduction_0)


@T.macro
def combine_GraphNode_0(
    PlaceHolder_A: T.Tensor((4, 128, 128), "float16"),
    t_reduction_0: T.Tensor((128, 128), "float32"),
):
    for i0 in T.Pipelined(2, num_stages=2):
        combine_GraphNode_0_iter(
            PlaceHolder_A=_S(
                PlaceHolder_A[2 * i0 : 2 * i0 + 2, 0 : 0 + 128, 0 : 0 + 128], squeeze_dim=[]
            ),
            t_reduction_0=t_reduction_0,
        )


@T.macro
def combine_main_kernel_iter(
    PlaceHolder_A: T.Tensor((4, 128, 128), "float16"),
    OutputNode: T.Tensor((128, 128), "float16"),
):
    t_reduction_0 = T.alloc_fragment(
        (
            128,
            128,
        ),
        "float32",
    )
    T.clear(t_reduction_0)
    combine_GraphNode_0(PlaceHolder_A, t_reduction_0)
    T.copy(t_reduction_0, OutputNode)


@T.prim_func
def main_kernel(
    PlaceHolder_A: T.Tensor((2, 128, 4096), "float16"),
    PlaceHolder_B: T.Tensor((2, 4096, 128), "float16"),
    Partial_A_split: T.Tensor((2, 4, 128, 128), "float16"),
    Partial_A_combine: T.Tensor((2, 4, 128, 128), "float16"),
    OutputNode: T.Tensor((2, 128, 128), "float16"),
):
    with T.Kernel(8, 1, 1, threads=(128)) as (i0, i1, i2):
        T.use_swizzle(16)
        split_main_kernel_iter(
            PlaceHolder_A=_S(
                PlaceHolder_A[
                    1 * (i0 // 4) : 1 * (i0 // 4) + 1,
                    128 * i1 : 128 * i1 + 128,
                    1024 * (i0 % 4) : 1024 * (i0 % 4) + 1024,
                ],
                squeeze_dim=[0],
            ),
            PlaceHolder_B=_S(
                PlaceHolder_B[
                    1 * (i0 // 4) : 1 * (i0 // 4) + 1,
                    1024 * (i0 % 4) : 1024 * (i0 % 4) + 1024,
                    128 * i2 : 128 * i2 + 128,
                ],
                squeeze_dim=[0],
            ),
            OutputNode=_S(
                Partial_A_split[
                    1 * (i0 // 4) : 1 * (i0 // 4) + 1,
                    (i0 % 4) : (i0 % 4) + 1,
                    128 * i1 : 128 * i1 + 128,
                    128 * i2 : 128 * i2 + 128,
                ],
                squeeze_dim=[0, 1],
            ),
        )
    with T.Kernel(2, threads=(128)) as (i0):
        T.use_swizzle(16)
        combine_main_kernel_iter(
            PlaceHolder_A=_S(
                Partial_A_combine[1 * i0 : 1 * i0 + 1, 0 : 0 + 4, 0 : 0 + 128, 0 : 0 + 128],
                squeeze_dim=[0],
            ),
            OutputNode=_S(
                OutputNode[1 * i0 : 1 * i0 + 1, 0 : 0 + 128, 0 : 0 + 128], squeeze_dim=[0]
            ),
        )
