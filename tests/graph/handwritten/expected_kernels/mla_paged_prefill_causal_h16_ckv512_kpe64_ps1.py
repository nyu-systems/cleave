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
    "tl.disable_tma_lower": True,
    "tl.disable_warp_specialized": True,
    "tl.enable_fast_math": True,
    "tl.ptxas_register_usage_level": 10,
}
total_q = T.symbolic("total_q")
num_pages = T.symbolic("num_pages")
len_indptr = T.symbolic("len_indptr")
n_idx = T.symbolic("n_idx")


@T.macro
def ukernel_index_gather_cibuf_gs(IDX, base, tile_base, valid_len, DST):
    DST_reg = T.alloc_fragment(DST.shape, "int32")
    for x0 in T.Parallel(*DST.shape):
        DST_reg[x0] = IDX[
            T.min(base + T.max(T.min(tile_base + x0, valid_len - 1), 0), IDX.shape[0] - 1)
        ]
    T.copy(DST_reg, DST)


@T.macro
def ukernel_paged_gather_paged_base_cibuf_crbuf_gss(SRC, IDX, base, DST):
    DST_reg = T.alloc_fragment(DST.shape, DST.dtype)
    for g0, g1 in T.Parallel(*DST.shape):
        DST_reg[g0, g1] = SRC[T.min(IDX[T.min(base + g0, IDX.shape[0] - 1)], SRC.shape[0] - 1), g1]
    T.copy(DST_reg, DST)


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
        policy=T.GemmWarpPolicy.FullCol,
    )


@T.macro
def ukernel_add_2dims_rrr(A, B, C):
    for i0, i1 in T.Parallel(*C.shape):
        C[i0, i1] = A[i0, i1] + B[i0, i1]


@T.macro
def ukernel_causal_maskedexp_scale_rt2_mgrp16_2dims_rr(A, tile_base, valid_len, diag, scale2, B):
    for m0, m1 in T.Parallel(*B.shape):
        B[m0, m1] = T.if_then_else(
            (tile_base + m1 < valid_len) & (tile_base + m1 <= diag + m0 // 16),
            T.exp2(A[m0, m1] * scale2),
            0.0,
        )


@T.macro
def ukernel_accumulate_2dims_rr(A, B):
    for i0, i1 in T.Parallel(*B.shape):
        B[i0, i1] += A[i0, i1]


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
        policy=T.GemmWarpPolicy.FullCol,
    )


@T.macro
def ukernel_reduction_axis1_rr_clear(A, B):

    A_ptr = A

    B_ptr = B

    T.reduce_sum(A_ptr, B_ptr, dim=-1, clear=True)


@T.macro
def ukernel_masked_div_broadcast_0_dfloor_1eminus_30_mgrp16_dst2_2dims_rrg(A, B, n_rows, C):
    for i0, i1 in T.Parallel(*A.shape):
        if i0 // 16 < n_rows:
            C[i0 % 16, i1] = A[i0, i1] / T.max(B[i0], 1e-30)


@T.macro
def ukernel_masked_log2_mgrp16_dst2_1dims_rg(A, n_rows, B):
    for i0 in T.Parallel(*A.shape):
        if i0 // 16 < n_rows:
            B[i0 % 16] = T.log2(A[i0])


@T.macro
def ukernel_indptr_scan_tile1_gl(IDX, slot, n_entries, DST):
    _cum = T.alloc_local((1,), "int32")
    DST[0] = -1
    DST[1] = 0
    _cum[0] = 0
    for s0 in T.serial(n_entries - 1):
        _n = T.ceildiv(IDX[s0 + 1] - IDX[s0], 1)
        _hit = T.if_then_else((_cum[0] <= slot) & (slot < _cum[0] + _n), 1, 0)
        DST[0] += _hit * (s0 + 1)
        DST[1] += _hit * (slot - _cum[0])
        _cum[0] += _n


@T.macro
def GraphNode_0_iter(
    PlaceHolder_Q_NOPEE: T.Tensor((16, 512), "bfloat16"),
    PlaceHolder_CKV_CACHEE: T.Tensor((num_pages, 512), "bfloat16"),
    PlaceHolder_KV_INDICESS: T.Tensor((n_idx), "int32"),
    PlaceHolder_KPE_CACHEE: T.Tensor((num_pages, 64), "bfloat16"),
    PlaceHolder_Q_PEE: T.Tensor((16, 64), "bfloat16"),
    tile_base: T.int32,
    valid_len: T.int32,
    diag: T.int32,
    scale2: T.float32,
    kv_base: T.int32,
    win_base: T.int32,
    win_off: T.int32,
    t_idx_window: T.Tensor((256), "int32"),
    t_psum_0: T.Tensor((16, 64), "float32"),
    t_mma_1: T.Tensor((16, 512), "float32"),
):
    if win_off == 0:
        ukernel_index_gather_cibuf_gs(
            PlaceHolder_KV_INDICESS, kv_base, win_base, valid_len, t_idx_window
        )
    t_paged_gather_ckv = T.alloc_shared(
        (
            64,
            512,
        ),
        "bfloat16",
    )
    T.annotate_layout({t_paged_gather_ckv: tl.layout.make_swizzled_layout(t_paged_gather_ckv)})
    ukernel_paged_gather_paged_base_cibuf_crbuf_gss(
        PlaceHolder_CKV_CACHEE, t_idx_window, win_off, t_paged_gather_ckv
    )
    t_mma_0 = T.alloc_fragment(
        (
            16,
            64,
        ),
        "float32",
    )
    ukernel_mma_ssr_transB_clear(PlaceHolder_Q_NOPEE, t_paged_gather_ckv, t_mma_0)
    t_paged_gather_kpe = T.alloc_shared(
        (
            64,
            64,
        ),
        "bfloat16",
    )
    T.annotate_layout({t_paged_gather_kpe: tl.layout.make_swizzled_layout(t_paged_gather_kpe)})
    ukernel_paged_gather_paged_base_cibuf_crbuf_gss(
        PlaceHolder_KPE_CACHEE, t_idx_window, win_off, t_paged_gather_kpe
    )
    t_mma_pe = T.alloc_fragment(
        (
            16,
            64,
        ),
        "float32",
    )
    ukernel_mma_ssr_transB_clear(PlaceHolder_Q_PEE, t_paged_gather_kpe, t_mma_pe)
    t_add_0 = T.alloc_fragment(
        (
            16,
            64,
        ),
        "float32",
    )
    ukernel_add_2dims_rrr(t_mma_0, t_mma_pe, t_add_0)
    t_exp_0 = T.alloc_fragment(
        (
            16,
            64,
        ),
        "float32",
    )
    ukernel_causal_maskedexp_scale_rt2_mgrp16_2dims_rr(
        t_add_0, tile_base, valid_len, diag, scale2, t_exp_0
    )
    ukernel_accumulate_2dims_rr(t_exp_0, t_psum_0)
    t_exp_0_cast = T.alloc_shared(
        (
            16,
            64,
        ),
        "bfloat16",
    )
    T.annotate_layout({t_exp_0_cast: tl.layout.make_swizzled_layout(t_exp_0_cast)})
    T.copy(t_exp_0, t_exp_0_cast)
    ukernel_mma_ssr(t_exp_0_cast, t_paged_gather_ckv, t_mma_1)


@T.macro
def GraphNode_0(
    PlaceHolder_Q_NOPE: T.Tensor((16, 512), "bfloat16"),
    PlaceHolder_CKV_CACHE: T.Tensor((num_pages, 1, 512), "bfloat16"),
    PlaceHolder_KV_INDICES: T.Tensor((n_idx), "int32"),
    PlaceHolder_KPE_CACHE: T.Tensor((num_pages, 1, 64), "bfloat16"),
    PlaceHolder_Q_PE: T.Tensor((16, 64), "bfloat16"),
    kv_base: T.int32,
    valid_len: T.int32,
    diag: T.int32,
    scale2: T.float32,
    t_psum_0: T.Tensor((16, 64), "float32"),
    t_mma_1: T.Tensor((16, 512), "float32"),
):
    PlaceHolder_Q_NOPEE = T.alloc_shared(
        (
            16,
            512,
        ),
        "bfloat16",
    )
    T.annotate_layout({PlaceHolder_Q_NOPEE: tl.layout.make_swizzled_layout(PlaceHolder_Q_NOPEE)})
    T.copy(PlaceHolder_Q_NOPE, PlaceHolder_Q_NOPEE)
    PlaceHolder_CKV_CACHEE = PlaceHolder_CKV_CACHE
    PlaceHolder_KV_INDICESS = PlaceHolder_KV_INDICES
    PlaceHolder_KPE_CACHEE = PlaceHolder_KPE_CACHE
    PlaceHolder_Q_PEE = T.alloc_shared(
        (
            16,
            64,
        ),
        "bfloat16",
    )
    T.annotate_layout({PlaceHolder_Q_PEE: tl.layout.make_swizzled_layout(PlaceHolder_Q_PEE)})
    T.copy(PlaceHolder_Q_PE, PlaceHolder_Q_PEE)
    t_idx_window = T.alloc_shared((256,), "int32")
    for i0 in T.serial(T.ceildiv(valid_len, 64)):
        GraphNode_0_iter(
            PlaceHolder_Q_NOPEE=PlaceHolder_Q_NOPEE,
            PlaceHolder_CKV_CACHEE=PlaceHolder_CKV_CACHEE,
            PlaceHolder_KV_INDICESS=PlaceHolder_KV_INDICESS,
            PlaceHolder_KPE_CACHEE=PlaceHolder_KPE_CACHEE,
            PlaceHolder_Q_PEE=PlaceHolder_Q_PEE,
            tile_base=64 * i0,
            valid_len=valid_len,
            diag=diag,
            scale2=scale2,
            kv_base=kv_base,
            win_base=i0 // 4 * 256,
            win_off=i0 % 4 * 64,
            t_idx_window=t_idx_window,
            t_psum_0=t_psum_0,
            t_mma_1=t_mma_1,
        )


@T.macro
def main_kernel_iter(
    PlaceHolder_Q_NOPE: T.Tensor((16, 512), "bfloat16"),
    PlaceHolder_Q_PE: T.Tensor((16, 64), "bfloat16"),
    PlaceHolder_CKV_CACHE: T.Tensor((num_pages, 512), "bfloat16"),
    PlaceHolder_KPE_CACHE: T.Tensor((num_pages, 64), "bfloat16"),
    PlaceHolder_KV_INDICES: T.Tensor((n_idx), "int32"),
    OutputNode_0: T.Tensor((16, 512), "bfloat16"),
    OutputNode_1: T.Tensor((16), "float32"),
    n_rows: T.int32,
    kv_base: T.int32,
    valid_len: T.int32,
    diag: T.int32,
    scale2: T.float32,
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
    GraphNode_0(
        PlaceHolder_Q_NOPE,
        PlaceHolder_CKV_CACHE,
        PlaceHolder_KV_INDICES,
        PlaceHolder_KPE_CACHE,
        PlaceHolder_Q_PE,
        kv_base,
        valid_len,
        diag,
        scale2,
        t_psum_0,
        t_mma_1,
    )
    t_rowsum_0 = T.alloc_fragment((16,), "float32")
    ukernel_reduction_axis1_rr_clear(t_psum_0, t_rowsum_0)
    ukernel_masked_div_broadcast_0_dfloor_1eminus_30_mgrp16_dst2_2dims_rrg(
        t_mma_1, t_rowsum_0, n_rows, OutputNode_0
    )
    ukernel_masked_log2_mgrp16_dst2_1dims_rg(t_rowsum_0, n_rows, OutputNode_1)


@T.prim_func
def main_kernel(
    PlaceHolder_Q_NOPE: T.Tensor((total_q, 16, 512), "bfloat16"),
    PlaceHolder_Q_PE: T.Tensor((total_q, 16, 64), "bfloat16"),
    PlaceHolder_CKV_CACHE: T.Tensor((num_pages, 1, 512), "bfloat16"),
    PlaceHolder_KPE_CACHE: T.Tensor((num_pages, 1, 64), "bfloat16"),
    PlaceHolder_QO_INDPTR: T.Tensor((len_indptr), "int32"),
    PlaceHolder_KV_INDPTR: T.Tensor((len_indptr), "int32"),
    PlaceHolder_KV_INDICES: T.Tensor((n_idx), "int32"),
    sm_scale: T.float32,
    OutputNode_0: T.Tensor((total_q, 16, 512), "bfloat16"),
    OutputNode_1: T.Tensor((total_q, 16), "float32"),
):
    with T.Kernel(T.ceildiv(total_q, 1) + len_indptr - 2, threads=(128)) as (i0):
        _wl = T.alloc_local((2,), "int32")
        ukernel_indptr_scan_tile1_gl(PlaceHolder_QO_INDPTR, i0, len_indptr, _wl)
        if _wl[0] >= 0:
            main_kernel_iter(
                PlaceHolder_Q_NOPE=PlaceHolder_Q_NOPE[
                    PlaceHolder_QO_INDPTR[_wl[0]] + (_wl[1] * 1), 0 : 0 + 16, 0 : 0 + 512
                ],
                PlaceHolder_Q_PE=PlaceHolder_Q_PE[
                    PlaceHolder_QO_INDPTR[_wl[0]] + (_wl[1] * 1), 0 : 0 + 16, 0 : 0 + 64
                ],
                PlaceHolder_CKV_CACHE=_S(
                    PlaceHolder_CKV_CACHE[0 : 0 + num_pages, 0 : 0 + 1, 0 : 0 + 512],
                    squeeze_dim=[1],
                ),
                PlaceHolder_KPE_CACHE=_S(
                    PlaceHolder_KPE_CACHE[0 : 0 + num_pages, 0 : 0 + 1, 0 : 0 + 64], squeeze_dim=[1]
                ),
                PlaceHolder_KV_INDICES=_S(PlaceHolder_KV_INDICES[0 : 0 + n_idx], squeeze_dim=[]),
                OutputNode_0=_S(
                    OutputNode_0[
                        PlaceHolder_QO_INDPTR[_wl[0]]
                        + (_wl[1] * 1) : PlaceHolder_QO_INDPTR[_wl[0]]
                        + (_wl[1] * 1)
                        + 1,
                        0 : 0 + 16,
                        0 : 0 + 512,
                    ],
                    squeeze_dim=[0],
                ),
                OutputNode_1=_S(
                    OutputNode_1[
                        PlaceHolder_QO_INDPTR[_wl[0]]
                        + (_wl[1] * 1) : PlaceHolder_QO_INDPTR[_wl[0]]
                        + (_wl[1] * 1)
                        + 1,
                        0 : 0 + 16,
                    ],
                    squeeze_dim=[0],
                ),
                n_rows=T.min(
                    1,
                    (PlaceHolder_QO_INDPTR[_wl[0] + 1] - PlaceHolder_QO_INDPTR[_wl[0]])
                    - (_wl[1] * 1),
                ),
                kv_base=PlaceHolder_KV_INDPTR[_wl[0]],
                valid_len=T.max(
                    T.min(
                        (PlaceHolder_KV_INDPTR[_wl[0] + 1] - PlaceHolder_KV_INDPTR[_wl[0]]),
                        (
                            (PlaceHolder_KV_INDPTR[_wl[0] + 1] - PlaceHolder_KV_INDPTR[_wl[0]])
                            - (PlaceHolder_QO_INDPTR[_wl[0] + 1] - PlaceHolder_QO_INDPTR[_wl[0]])
                            + (_wl[1] * 1)
                        )
                        + T.min(
                            1,
                            (PlaceHolder_QO_INDPTR[_wl[0] + 1] - PlaceHolder_QO_INDPTR[_wl[0]])
                            - (_wl[1] * 1),
                        ),
                    ),
                    0,
                ),
                diag=(
                    (PlaceHolder_KV_INDPTR[_wl[0] + 1] - PlaceHolder_KV_INDPTR[_wl[0]])
                    - (PlaceHolder_QO_INDPTR[_wl[0] + 1] - PlaceHolder_QO_INDPTR[_wl[0]])
                    + (_wl[1] * 1)
                ),
                scale2=sm_scale * 1.4426950408889634,
            )
