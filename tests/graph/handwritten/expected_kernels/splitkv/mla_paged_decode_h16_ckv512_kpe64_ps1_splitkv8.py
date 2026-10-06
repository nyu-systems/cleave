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
    "tl.disable_safe_memory_legalize": True,
    "tl.enable_fast_math": True,
    "tl.ptxas_register_usage_level": 10,
}
batch = T.symbolic("batch")
num_pages = T.symbolic("num_pages")
len_indptr = T.symbolic("len_indptr")
n_idx = T.symbolic("n_idx")


@T.macro
def split_ukernel_index_gather_cibuf_gs(IDX, base, tile_base, valid_len, DST):
    DST_reg = T.alloc_fragment(DST.shape, "int32")
    for x0 in T.Parallel(*DST.shape):
        DST_reg[x0] = IDX[
            T.min(base + T.max(T.min(tile_base + x0, valid_len - 1), 0), IDX.shape[0] - 1)
        ]
    T.copy(DST_reg, DST)


@T.macro
def split_ukernel_paged_gather_paged_base_cibuf_crbuf_gsr(SRC, IDX, base, DST):
    for g0, g1 in T.Parallel(*DST.shape):
        DST[g0, g1] = SRC[T.min(IDX[T.min(base + g0, IDX.shape[0] - 1)], SRC.shape[0] - 1), g1]


@T.macro
def split_ukernel_mma_ssr_transB_clear(A, B, C):

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
        policy=T.GemmWarpPolicy.FullCol,
    )


@T.macro
def split_ukernel_add_2dims_rrr(A, B, C):
    for i0, i1 in T.Parallel(*C.shape):
        C[i0, i1] = A[i0, i1] + B[i0, i1]


@T.macro
def split_ukernel_maskedexp_scale_rt2_2dims_rr(A, tile_base, valid_len, scale2, B):
    for m0, m1 in T.Parallel(*B.shape):
        B[m0, m1] = T.if_then_else(tile_base + m1 < valid_len, T.exp2(A[m0, m1] * scale2), 0.0)


@T.macro
def split_ukernel_accumulate_2dims_rr(A, B):
    for i0, i1 in T.Parallel(*B.shape):
        B[i0, i1] += A[i0, i1]


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
        policy=T.GemmWarpPolicy.FullCol,
    )


@T.macro
def split_ukernel_row_gather_valid16_r3_gs(SRC, row0, head0, DST):
    DST_reg = T.alloc_fragment(DST.shape, DST.dtype)
    for r0, r1 in T.Parallel(*DST.shape):
        DST_reg[r0, r1] = T.if_then_else(
            r0 < 16,
            SRC[row0, head0 + T.min(r0, 16 - 1), r1],
            T.cast(0, DST.dtype),
        )
    T.copy(DST_reg, DST)


@T.macro
def split_ukernel_reduction_axis1_rr_clear(A, B):

    A_ptr = A

    B_ptr = B

    T.reduce_sum(A_ptr, B_ptr, dim=-1, clear=True)


@T.macro
def split_ukernel_copy_1dims_rg(A, B):
    for i0 in T.Parallel(*B.shape):
        B[i0] = T.cast(A[i0], B.dtype)


@T.macro
def split_ukernel_copy_2dims_rg(A, B):
    for i0, i1 in T.Parallel(*B.shape):
        B[i0, i1] = T.cast(A[i0, i1], B.dtype)


@T.macro
def split_GraphNode_0_iter(
    PlaceHolder_QNN: T.Tensor((16, 512), "bfloat16"),
    PlaceHolder_CKVV: T.Tensor((num_pages, 512), "bfloat16"),
    PlaceHolder_KVII: T.Tensor((n_idx), "int32"),
    PlaceHolder_KPEE: T.Tensor((num_pages, 64), "bfloat16"),
    PlaceHolder_QPP: T.Tensor((16, 64), "bfloat16"),
    tile_base: T.int32,
    valid_len: T.int32,
    scale2: T.float32,
    kv_base: T.int32,
    win_base: T.int32,
    win_off: T.int32,
    win_off_next: T.int32,
    split_base: T.int32,
    split_end: T.int32,
    t_paged_gather_ckv_pf: T.Tensor((64, 512), "bfloat16"),
    t_paged_gather_kpe_pf: T.Tensor((64, 64), "bfloat16"),
    t_idx_window: T.Tensor((512), "int32"),
    t_psum_0: T.Tensor((16, 64), "float32"),
    t_mma_1: T.Tensor((16, 512), "float32"),
):
    if (win_off == 0) | (tile_base == split_base):
        split_ukernel_index_gather_cibuf_gs(
            PlaceHolder_KVII, kv_base, win_base, valid_len, t_idx_window
        )
        split_ukernel_paged_gather_paged_base_cibuf_crbuf_gsr(
            PlaceHolder_CKVV, t_idx_window, win_off, t_paged_gather_ckv_pf
        )
        split_ukernel_paged_gather_paged_base_cibuf_crbuf_gsr(
            PlaceHolder_KPEE, t_idx_window, win_off, t_paged_gather_kpe_pf
        )
    t_paged_gather_ckv = T.alloc_shared(
        (
            64,
            512,
        ),
        "bfloat16",
    )
    T.annotate_layout({t_paged_gather_ckv: tl.layout.make_swizzled_layout(t_paged_gather_ckv)})
    T.copy(t_paged_gather_ckv_pf, t_paged_gather_ckv)
    t_paged_gather_kpe = T.alloc_shared(
        (
            64,
            64,
        ),
        "bfloat16",
    )
    T.annotate_layout({t_paged_gather_kpe: tl.layout.make_swizzled_layout(t_paged_gather_kpe)})
    T.copy(t_paged_gather_kpe_pf, t_paged_gather_kpe)
    if (win_off + 64 < 512) & (tile_base + 64 < split_end):
        split_ukernel_paged_gather_paged_base_cibuf_crbuf_gsr(
            PlaceHolder_CKVV, t_idx_window, win_off_next, t_paged_gather_ckv_pf
        )
        split_ukernel_paged_gather_paged_base_cibuf_crbuf_gsr(
            PlaceHolder_KPEE, t_idx_window, win_off_next, t_paged_gather_kpe_pf
        )
    t_mma_0 = T.alloc_fragment(
        (
            16,
            64,
        ),
        "float32",
    )
    split_ukernel_mma_ssr_transB_clear(PlaceHolder_QNN, t_paged_gather_ckv, t_mma_0)
    t_mma_pe = T.alloc_fragment(
        (
            16,
            64,
        ),
        "float32",
    )
    split_ukernel_mma_ssr_transB_clear(PlaceHolder_QPP, t_paged_gather_kpe, t_mma_pe)
    t_add_0 = T.alloc_fragment(
        (
            16,
            64,
        ),
        "float32",
    )
    split_ukernel_add_2dims_rrr(t_mma_0, t_mma_pe, t_add_0)
    t_exp_0 = T.alloc_fragment(
        (
            16,
            64,
        ),
        "float32",
    )
    split_ukernel_maskedexp_scale_rt2_2dims_rr(t_add_0, tile_base, valid_len, scale2, t_exp_0)
    split_ukernel_accumulate_2dims_rr(t_exp_0, t_psum_0)
    t_exp_0_cast = T.alloc_shared(
        (
            16,
            64,
        ),
        "bfloat16",
    )
    T.annotate_layout({t_exp_0_cast: tl.layout.make_swizzled_layout(t_exp_0_cast)})
    T.copy(t_exp_0, t_exp_0_cast)
    split_ukernel_mma_ssr(t_exp_0_cast, t_paged_gather_ckv, t_mma_1)


@T.macro
def split_GraphNode_0(
    PlaceHolder_QN: T.Tensor((batch, 16, 512), "bfloat16"),
    PlaceHolder_CKV: T.Tensor((num_pages, 1, 512), "bfloat16"),
    PlaceHolder_KVI: T.Tensor((n_idx), "int32"),
    PlaceHolder_KPE: T.Tensor((num_pages, 1, 64), "bfloat16"),
    PlaceHolder_QP: T.Tensor((batch, 16, 64), "bfloat16"),
    q_row0: T.int32,
    q_head0: T.int32,
    kv_base: T.int32,
    valid_len: T.int32,
    scale2: T.float32,
    tile0: T.int32,
    my_trip: T.int32,
    split_end: T.int32,
    t_psum_0: T.Tensor((16, 64), "float32"),
    t_mma_1: T.Tensor((16, 512), "float32"),
):
    PlaceHolder_QNN = T.alloc_shared(
        (
            16,
            512,
        ),
        "bfloat16",
    )
    T.annotate_layout({PlaceHolder_QNN: tl.layout.make_swizzled_layout(PlaceHolder_QNN)})
    split_ukernel_row_gather_valid16_r3_gs(PlaceHolder_QN, q_row0, q_head0, PlaceHolder_QNN)
    PlaceHolder_CKVV = PlaceHolder_CKV
    PlaceHolder_KVII = PlaceHolder_KVI
    PlaceHolder_KPEE = PlaceHolder_KPE
    PlaceHolder_QPP = T.alloc_shared(
        (
            16,
            64,
        ),
        "bfloat16",
    )
    T.annotate_layout({PlaceHolder_QPP: tl.layout.make_swizzled_layout(PlaceHolder_QPP)})
    split_ukernel_row_gather_valid16_r3_gs(PlaceHolder_QP, q_row0, q_head0, PlaceHolder_QPP)
    t_idx_window = T.alloc_shared((512,), "int32")
    t_paged_gather_ckv_pf = T.alloc_fragment(
        (
            64,
            512,
        ),
        "bfloat16",
    )
    t_paged_gather_kpe_pf = T.alloc_fragment(
        (
            64,
            64,
        ),
        "bfloat16",
    )
    for i0 in T.serial(my_trip):
        split_GraphNode_0_iter(
            PlaceHolder_QNN=PlaceHolder_QNN,
            PlaceHolder_CKVV=PlaceHolder_CKVV,
            PlaceHolder_KVII=PlaceHolder_KVII,
            PlaceHolder_KPEE=PlaceHolder_KPEE,
            PlaceHolder_QPP=PlaceHolder_QPP,
            tile_base=64 * (tile0 + i0),
            valid_len=valid_len,
            scale2=scale2,
            kv_base=kv_base,
            win_base=(tile0 + i0) // 8 * 512,
            win_off=(tile0 + i0) % 8 * 64,
            win_off_next=(tile0 + (i0 + 1)) % 8 * 64,
            split_base=64 * tile0,
            split_end=split_end,
            t_idx_window=t_idx_window,
            t_paged_gather_ckv_pf=t_paged_gather_ckv_pf,
            t_paged_gather_kpe_pf=t_paged_gather_kpe_pf,
            t_psum_0=t_psum_0,
            t_mma_1=t_mma_1,
        )


@T.macro
def split_main_kernel_iter(
    PlaceHolder_QN: T.Tensor((batch, 16, 512), "bfloat16"),
    PlaceHolder_QP: T.Tensor((batch, 16, 64), "bfloat16"),
    PlaceHolder_CKV: T.Tensor((num_pages, 512), "bfloat16"),
    PlaceHolder_KPE: T.Tensor((num_pages, 64), "bfloat16"),
    PlaceHolder_KVI: T.Tensor((n_idx), "int32"),
    OutputNode_0: T.Tensor((16), "float32"),
    OutputNode_1: T.Tensor((16, 512), "float32"),
    q_row0: T.int32,
    q_head0: T.int32,
    kv_base: T.int32,
    valid_len: T.int32,
    scale2: T.float32,
    tile0: T.int32,
    my_trip: T.int32,
    split_end: T.int32,
):
    t_psum_0 = T.alloc_fragment(
        (
            16,
            64,
        ),
        "float32",
    )
    T.clear(t_psum_0)
    t_mma_1 = T.alloc_fragment(
        (
            16,
            512,
        ),
        "float32",
    )
    T.clear(t_mma_1)
    split_GraphNode_0(
        PlaceHolder_QN,
        PlaceHolder_CKV,
        PlaceHolder_KVI,
        PlaceHolder_KPE,
        PlaceHolder_QP,
        q_row0,
        q_head0,
        kv_base,
        valid_len,
        scale2,
        tile0,
        my_trip,
        split_end,
        t_psum_0,
        t_mma_1,
    )
    t_rowsum_0 = T.alloc_fragment((16,), "float32")
    split_ukernel_reduction_axis1_rr_clear(t_psum_0, t_rowsum_0)
    split_ukernel_copy_1dims_rg(t_rowsum_0, OutputNode_0)
    split_ukernel_copy_2dims_rg(t_mma_1, OutputNode_1)


@T.macro
def combine_ukernel_accumulate_1dims_sr(A, B):
    for i0 in T.Parallel(*B.shape):
        B[i0] += A[i0]


@T.macro
def combine_ukernel_accumulate_2dims_sr(A, B):
    for i0, i1 in T.Parallel(*B.shape):
        B[i0, i1] += A[i0, i1]


@T.macro
def combine_ukernel_div_broadcast_0_dfloor_1eminus_30_2dims_rrg(A, B, C):
    for i0, i1 in T.Parallel(*C.shape):
        C[i0, i1] = A[i0, i1] / T.max(B[i0], 1e-30)


@T.macro
def combine_ukernel_log2_1dims_rg(A, B):
    for i0 in T.Parallel(*B.shape):
        B[i0] = T.log2(A[i0])


@T.macro
def combine_GraphNode_0_iter(
    PlaceHolder_ROWSUM: T.Tensor((16), "float32"),
    PlaceHolder_VALUE: T.Tensor((16, 128), "float32"),
    t_rowsum_combine_0: T.Tensor((16), "float32"),
    t_value_combine_0: T.Tensor((16, 128), "float32"),
):
    PlaceHolder_ROWSUMM = T.alloc_shared((16,), "float32")
    T.copy(PlaceHolder_ROWSUM, PlaceHolder_ROWSUMM)
    combine_ukernel_accumulate_1dims_sr(PlaceHolder_ROWSUMM, t_rowsum_combine_0)
    PlaceHolder_VALUEE = T.alloc_shared(
        (
            16,
            128,
        ),
        "float32",
    )
    T.copy(PlaceHolder_VALUE, PlaceHolder_VALUEE)
    combine_ukernel_accumulate_2dims_sr(PlaceHolder_VALUEE, t_value_combine_0)


@T.macro
def combine_GraphNode_0(
    PlaceHolder_ROWSUM: T.Tensor((8, 16), "float32"),
    PlaceHolder_VALUE: T.Tensor((8, 16, 128), "float32"),
    t_rowsum_combine_0: T.Tensor((16), "float32"),
    t_value_combine_0: T.Tensor((16, 128), "float32"),
):
    for i0 in T.serial(8):
        combine_GraphNode_0_iter(
            PlaceHolder_ROWSUM=_S(
                PlaceHolder_ROWSUM[1 * i0 : 1 * i0 + 1, 0 : 0 + 16], squeeze_dim=[0]
            ),
            PlaceHolder_VALUE=_S(
                PlaceHolder_VALUE[1 * i0 : 1 * i0 + 1, 0 : 0 + 16, 0 : 0 + 128], squeeze_dim=[0]
            ),
            t_rowsum_combine_0=t_rowsum_combine_0,
            t_value_combine_0=t_value_combine_0,
        )


@T.macro
def combine_main_kernel_iter(
    PlaceHolder_ROWSUM: T.Tensor((8, 16), "float32"),
    PlaceHolder_VALUE: T.Tensor((8, 16, 128), "float32"),
    OutputNode_0: T.Tensor((16, 128), "bfloat16"),
    OutputNode_1: T.Tensor((16), "float32"),
):
    t_rowsum_combine_0 = T.alloc_fragment((16,), "float32")
    T.clear(t_rowsum_combine_0)
    t_value_combine_0 = T.alloc_fragment(
        (
            16,
            128,
        ),
        "float32",
    )
    T.clear(t_value_combine_0)
    combine_GraphNode_0(
        PlaceHolder_ROWSUM, PlaceHolder_VALUE, t_rowsum_combine_0, t_value_combine_0
    )
    combine_ukernel_div_broadcast_0_dfloor_1eminus_30_2dims_rrg(
        t_value_combine_0, t_rowsum_combine_0, OutputNode_0
    )
    combine_ukernel_log2_1dims_rg(t_rowsum_combine_0, OutputNode_1)


@T.prim_func
def main_kernel(
    PlaceHolder_QN: T.Tensor((batch, 16, 512), "bfloat16"),
    PlaceHolder_QP: T.Tensor((batch, 16, 64), "bfloat16"),
    PlaceHolder_CKV: T.Tensor((num_pages, 1, 512), "bfloat16"),
    PlaceHolder_KPE: T.Tensor((num_pages, 1, 64), "bfloat16"),
    PlaceHolder_KV_INDPTR: T.Tensor((len_indptr), "int32"),
    PlaceHolder_KVI: T.Tensor((n_idx), "int32"),
    sm_scale: T.float32,
    Partial_ROWSUM_split: T.Tensor((8, batch, 16), "float32"),
    Partial_ROWSUM_combine: T.Tensor((8, batch, 16), "float32"),
    Partial_VALUE_split: T.Tensor((8, batch, 16, 512), "float32"),
    Partial_VALUE_combine: T.Tensor((8, batch, 16, 512), "float32"),
    OutputNode_0: T.Tensor((batch, 16, 512), "bfloat16"),
    OutputNode_1: T.Tensor((batch, 16), "float32"),
):
    with T.Kernel(batch, 8, threads=(256)) as (i0, i1):
        T.use_swizzle(16)
        split_main_kernel_iter(
            PlaceHolder_QN=_S(
                PlaceHolder_QN[0 : 0 + batch, 0 : 0 + 16, 0 : 0 + 512], squeeze_dim=[]
            ),
            PlaceHolder_QP=_S(
                PlaceHolder_QP[0 : 0 + batch, 0 : 0 + 16, 0 : 0 + 64], squeeze_dim=[]
            ),
            PlaceHolder_CKV=_S(
                PlaceHolder_CKV[0 : 0 + num_pages, 0 : 0 + 1, 0 : 0 + 512], squeeze_dim=[1]
            ),
            PlaceHolder_KPE=_S(
                PlaceHolder_KPE[0 : 0 + num_pages, 0 : 0 + 1, 0 : 0 + 64], squeeze_dim=[1]
            ),
            PlaceHolder_KVI=_S(PlaceHolder_KVI[0 : 0 + n_idx], squeeze_dim=[]),
            OutputNode_0=_S(
                Partial_ROWSUM_split[i1 : i1 + 1, i0 : i0 + 1, 16 * 0 : 16 * 0 + 16],
                squeeze_dim=[0, 1],
            ),
            OutputNode_1=_S(
                Partial_VALUE_split[i1 : i1 + 1, i0 : i0 + 1, 16 * 0 : 16 * 0 + 16, 0 : 0 + 512],
                squeeze_dim=[0, 1],
            ),
            q_row0=i0,
            q_head0=0,
            kv_base=PlaceHolder_KV_INDPTR[i0],
            valid_len=PlaceHolder_KV_INDPTR[i0 + 1] - PlaceHolder_KV_INDPTR[i0],
            scale2=sm_scale * 1.4426950408889634,
            tile0=i1
            * T.ceildiv(
                T.ceildiv(PlaceHolder_KV_INDPTR[i0 + 1] - PlaceHolder_KV_INDPTR[i0], 64), 8
            ),
            my_trip=T.max(
                T.min(
                    T.ceildiv(
                        T.ceildiv(PlaceHolder_KV_INDPTR[i0 + 1] - PlaceHolder_KV_INDPTR[i0], 64), 8
                    ),
                    T.ceildiv(PlaceHolder_KV_INDPTR[i0 + 1] - PlaceHolder_KV_INDPTR[i0], 64)
                    - i1
                    * T.ceildiv(
                        T.ceildiv(PlaceHolder_KV_INDPTR[i0 + 1] - PlaceHolder_KV_INDPTR[i0], 64), 8
                    ),
                ),
                0,
            ),
            split_end=T.min(
                PlaceHolder_KV_INDPTR[i0 + 1] - PlaceHolder_KV_INDPTR[i0],
                64
                * (
                    i1
                    * T.ceildiv(
                        T.ceildiv(PlaceHolder_KV_INDPTR[i0 + 1] - PlaceHolder_KV_INDPTR[i0], 64), 8
                    )
                    + T.ceildiv(
                        T.ceildiv(PlaceHolder_KV_INDPTR[i0 + 1] - PlaceHolder_KV_INDPTR[i0], 64), 8
                    )
                ),
            ),
        )
    with T.Kernel(batch, 4, threads=(128)) as (i0, i1):
        T.use_swizzle(16)
        combine_main_kernel_iter(
            PlaceHolder_ROWSUM=_S(
                Partial_ROWSUM_combine[0 : 0 + 8, i0 : i0 + 1, 16 * 0 : 16 * 0 + 16],
                squeeze_dim=[1],
            ),
            PlaceHolder_VALUE=_S(
                Partial_VALUE_combine[
                    0 : 0 + 8, i0 : i0 + 1, 16 * 0 : 16 * 0 + 16, 128 * i1 : 128 * i1 + 128
                ],
                squeeze_dim=[1],
            ),
            OutputNode_0=_S(
                OutputNode_0[i0 : i0 + 1, 16 * 0 : 16 * 0 + 16, 128 * i1 : 128 * i1 + 128],
                squeeze_dim=[0],
            ),
            OutputNode_1=_S(OutputNode_1[i0 : i0 + 1, 16 * 0 : 16 * 0 + 16], squeeze_dim=[0]),
        )
