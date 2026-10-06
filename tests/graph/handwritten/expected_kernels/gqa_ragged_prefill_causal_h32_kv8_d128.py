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
total_q = T.symbolic("total_q")
total_kv = T.symbolic("total_kv")
len_indptr = T.symbolic("len_indptr")


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
def ukernel_causal_maskedexp_scale_rt2_2dims_rr(A, tile_base, valid_len, diag, scale2, B):
    for m0, m1 in T.Parallel(*B.shape):
        B[m0, m1] = T.if_then_else(
            (tile_base + m1 < valid_len) & (tile_base + m1 <= diag + m0),
            T.exp2(A[m0, m1] * scale2),
            0.0,
        )


@T.macro
def ukernel_reduction_axis1_rr(A, B):

    A_ptr = A

    B_ptr = B

    T.reduce_sum(A_ptr, B_ptr, dim=-1, clear=False)


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
        policy=T.GemmWarpPolicy.FullCol,
    )


@T.macro
def ukernel_masked_div_broadcast_0_dfloor_1eminus_30_2dims_rrg(A, B, n_rows, C):
    for i0, i1 in T.Parallel(*C.shape):
        if i0 < n_rows:
            C[i0, i1] = A[i0, i1] / T.max(B[i0], 1e-30)


@T.macro
def ukernel_masked_log2_1dims_rg(A, n_rows, B):
    for i0 in T.Parallel(*B.shape):
        if i0 < n_rows:
            B[i0] = T.log2(A[i0])


@T.macro
def ukernel_indptr_scan_tile64_gl(IDX, slot, n_entries, DST):
    _cum = T.alloc_local((1,), "int32")
    DST[0] = -1
    DST[1] = 0
    _cum[0] = 0
    for s0 in T.serial(n_entries - 1):
        _n = T.ceildiv(IDX[s0 + 1] - IDX[s0], 64)
        _hit = T.if_then_else((_cum[0] <= slot) & (slot < _cum[0] + _n), 1, 0)
        DST[0] += _hit * (s0 + 1)
        DST[1] += _hit * (slot - _cum[0])
        _cum[0] += _n


@T.macro
def GraphNode_0_iter(
    PlaceHolder_QQ: T.Tensor((64, 128), "bfloat16"),
    PlaceHolder_K: T.Tensor((64, 128), "bfloat16"),
    PlaceHolder_V: T.Tensor((64, 128), "bfloat16"),
    tile_base: T.int32,
    valid_len: T.int32,
    diag: T.int32,
    scale2: T.float32,
    kv_row0: T.int32,
    kv_head: T.int32,
    t_rowsum_0: T.Tensor((64), "float32"),
    t_mma_1: T.Tensor((64, 128), "float32"),
):
    PlaceHolder_KK = T.alloc_shared(
        (
            64,
            128,
        ),
        "bfloat16",
    )
    T.annotate_layout({PlaceHolder_KK: tl.layout.make_swizzled_layout(PlaceHolder_KK)})
    T.copy(PlaceHolder_K, PlaceHolder_KK)
    t_mma_0 = T.alloc_fragment(
        (
            64,
            64,
        ),
        "float32",
    )
    ukernel_mma_ssr_transB_clear(PlaceHolder_QQ, PlaceHolder_KK, t_mma_0)
    t_exp_0 = T.alloc_fragment(
        (
            64,
            64,
        ),
        "float32",
    )
    ukernel_causal_maskedexp_scale_rt2_2dims_rr(
        t_mma_0, tile_base, valid_len, diag, scale2, t_exp_0
    )
    ukernel_reduction_axis1_rr(t_exp_0, t_rowsum_0)
    PlaceHolder_VV = T.alloc_shared(
        (
            64,
            128,
        ),
        "bfloat16",
    )
    T.annotate_layout({PlaceHolder_VV: tl.layout.make_swizzled_layout(PlaceHolder_VV)})
    T.copy(PlaceHolder_V, PlaceHolder_VV)
    t_exp_0_cast = T.alloc_fragment(
        (
            64,
            64,
        ),
        "bfloat16",
    )
    T.copy(t_exp_0, t_exp_0_cast)
    ukernel_mma_rsr(t_exp_0_cast, PlaceHolder_VV, t_mma_1)


@T.macro
def GraphNode_0(
    PlaceHolder_Q: T.Tensor((64, 128), "bfloat16"),
    PlaceHolder_K: T.Tensor((total_kv, 8, 128), "bfloat16"),
    PlaceHolder_V: T.Tensor((total_kv, 8, 128), "bfloat16"),
    q_row0: T.int32,
    q_tile_base: T.int32,
    q_len: T.int32,
    q_head: T.int32,
    kv_row0: T.int32,
    valid_len: T.int32,
    diag: T.int32,
    kv_head: T.int32,
    scale2: T.float32,
    t_rowsum_0: T.Tensor((64), "float32"),
    t_mma_1: T.Tensor((64, 128), "float32"),
):
    PlaceHolder_QQ = T.alloc_shared(
        (
            64,
            128,
        ),
        "bfloat16",
    )
    T.annotate_layout({PlaceHolder_QQ: tl.layout.make_swizzled_layout(PlaceHolder_QQ)})
    T.copy(PlaceHolder_Q, PlaceHolder_QQ)
    for i0 in T.Pipelined(T.ceildiv(valid_len, 64), num_stages=2):
        GraphNode_0_iter(
            PlaceHolder_QQ=PlaceHolder_QQ,
            PlaceHolder_K=PlaceHolder_K[
                kv_row0 + 64 * i0 : kv_row0 + 64 * i0 + 64, kv_head, 0 : 0 + 128
            ],
            PlaceHolder_V=PlaceHolder_V[
                kv_row0 + 64 * i0 : kv_row0 + 64 * i0 + 64, kv_head, 0 : 0 + 128
            ],
            tile_base=64 * i0,
            valid_len=valid_len,
            diag=diag,
            scale2=scale2,
            kv_row0=kv_row0,
            kv_head=kv_head,
            t_rowsum_0=t_rowsum_0,
            t_mma_1=t_mma_1,
        )


@T.macro
def main_kernel_iter(
    PlaceHolder_Q: T.Tensor((64, 128), "bfloat16"),
    PlaceHolder_K: T.Tensor((total_kv, 8, 128), "bfloat16"),
    PlaceHolder_V: T.Tensor((total_kv, 8, 128), "bfloat16"),
    OutputNode_0: T.Tensor((64, 128), "bfloat16"),
    OutputNode_1: T.Tensor((64), "float32"),
    n_rows: T.int32,
    q_row0: T.int32,
    q_tile_base: T.int32,
    q_len: T.int32,
    q_head: T.int32,
    kv_row0: T.int32,
    valid_len: T.int32,
    diag: T.int32,
    kv_head: T.int32,
    scale2: T.float32,
):
    t_rowsum_0 = T.alloc_fragment((64,), "float32")
    T.clear(t_rowsum_0)
    t_mma_1 = T.alloc_fragment(
        (
            64,
            128,
        ),
        "float32",
    )
    T.clear(t_mma_1)
    GraphNode_0(
        PlaceHolder_Q,
        PlaceHolder_K,
        PlaceHolder_V,
        q_row0,
        q_tile_base,
        q_len,
        q_head,
        kv_row0,
        valid_len,
        diag,
        kv_head,
        scale2,
        t_rowsum_0,
        t_mma_1,
    )
    ukernel_masked_div_broadcast_0_dfloor_1eminus_30_2dims_rrg(
        t_mma_1, t_rowsum_0, n_rows, OutputNode_0
    )
    ukernel_masked_log2_1dims_rg(t_rowsum_0, n_rows, OutputNode_1)


@T.prim_func
def main_kernel(
    PlaceHolder_Q: T.Tensor((total_q, 32, 128), "bfloat16"),
    PlaceHolder_K: T.Tensor((total_kv, 8, 128), "bfloat16"),
    PlaceHolder_V: T.Tensor((total_kv, 8, 128), "bfloat16"),
    PlaceHolder_QO_INDPTR: T.Tensor((len_indptr), "int32"),
    PlaceHolder_KV_INDPTR: T.Tensor((len_indptr), "int32"),
    sm_scale: T.float32,
    OutputNode_0: T.Tensor((total_q, 32, 128), "bfloat16"),
    OutputNode_1: T.Tensor((total_q, 32), "float32"),
):
    with T.Kernel(T.ceildiv(total_q, 64) + len_indptr - 2, 32, threads=(128)) as (i0, i1):
        T.use_swizzle(16)
        _wl = T.alloc_local((2,), "int32")
        ukernel_indptr_scan_tile64_gl(PlaceHolder_QO_INDPTR, i0, len_indptr, _wl)
        if _wl[0] >= 0:
            main_kernel_iter(
                PlaceHolder_Q=PlaceHolder_Q[
                    PlaceHolder_QO_INDPTR[_wl[0]]
                    + (_wl[1] * 64) : PlaceHolder_QO_INDPTR[_wl[0]]
                    + (_wl[1] * 64)
                    + 64,
                    i1,
                    0 : 0 + 128,
                ],
                PlaceHolder_K=_S(
                    PlaceHolder_K[0 : 0 + total_kv, 0 : 0 + 8, 0 : 0 + 128], squeeze_dim=[]
                ),
                PlaceHolder_V=_S(
                    PlaceHolder_V[0 : 0 + total_kv, 0 : 0 + 8, 0 : 0 + 128], squeeze_dim=[]
                ),
                OutputNode_0=_S(
                    OutputNode_0[
                        PlaceHolder_QO_INDPTR[_wl[0]]
                        + (_wl[1] * 64) : PlaceHolder_QO_INDPTR[_wl[0]]
                        + (_wl[1] * 64)
                        + 64,
                        i1 : i1 + 1,
                        0 : 0 + 128,
                    ],
                    squeeze_dim=[1],
                ),
                OutputNode_1=_S(
                    OutputNode_1[
                        PlaceHolder_QO_INDPTR[_wl[0]]
                        + (_wl[1] * 64) : PlaceHolder_QO_INDPTR[_wl[0]]
                        + (_wl[1] * 64)
                        + 64,
                        i1 : i1 + 1,
                    ],
                    squeeze_dim=[1],
                ),
                n_rows=T.min(
                    64,
                    (PlaceHolder_QO_INDPTR[_wl[0] + 1] - PlaceHolder_QO_INDPTR[_wl[0]])
                    - (_wl[1] * 64),
                ),
                q_row0=PlaceHolder_QO_INDPTR[_wl[0]],
                q_tile_base=(_wl[1] * 64),
                q_len=(PlaceHolder_QO_INDPTR[_wl[0] + 1] - PlaceHolder_QO_INDPTR[_wl[0]]),
                q_head=i1,
                kv_row0=PlaceHolder_KV_INDPTR[_wl[0]],
                valid_len=T.max(
                    T.min(
                        (PlaceHolder_KV_INDPTR[_wl[0] + 1] - PlaceHolder_KV_INDPTR[_wl[0]]),
                        (
                            (PlaceHolder_KV_INDPTR[_wl[0] + 1] - PlaceHolder_KV_INDPTR[_wl[0]])
                            - (PlaceHolder_QO_INDPTR[_wl[0] + 1] - PlaceHolder_QO_INDPTR[_wl[0]])
                            + (_wl[1] * 64)
                        )
                        + T.min(
                            64,
                            (PlaceHolder_QO_INDPTR[_wl[0] + 1] - PlaceHolder_QO_INDPTR[_wl[0]])
                            - (_wl[1] * 64),
                        ),
                    ),
                    0,
                ),
                diag=(
                    (PlaceHolder_KV_INDPTR[_wl[0] + 1] - PlaceHolder_KV_INDPTR[_wl[0]])
                    - (PlaceHolder_QO_INDPTR[_wl[0] + 1] - PlaceHolder_QO_INDPTR[_wl[0]])
                    + (_wl[1] * 64)
                ),
                kv_head=i1 // 4,
                scale2=sm_scale * 1.4426950408889634,
            )
