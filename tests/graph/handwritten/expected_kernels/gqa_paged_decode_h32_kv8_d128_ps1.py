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
def ukernel_paged_gather_paged_base_head_cibuf_crbuf_ggr(SRC, IDX, base, head, DST):
    for g0, g1 in T.Parallel(*DST.shape):
        DST[g0, g1] = SRC[
            T.min(IDX[T.min(base + g0, IDX.shape[0] - 1)], SRC.shape[0] - 1), head, g1
        ]


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
def ukernel_maskedexp_scale_rt2_2dims_rr(A, tile_base, valid_len, scale2, B):
    for m0, m1 in T.Parallel(*B.shape):
        B[m0, m1] = T.if_then_else(tile_base + m1 < valid_len, T.exp2(A[m0, m1] * scale2), 0.0)


@T.macro
def ukernel_reduction_axis1_rr(A, B):

    A_ptr = A

    B_ptr = B

    T.reduce_sum(A_ptr, B_ptr, dim=-1, clear=False)


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
def ukernel_row_gather_valid4_r3_gs(SRC, row0, head0, DST):
    DST_reg = T.alloc_fragment(DST.shape, DST.dtype)
    for r0, r1 in T.Parallel(*DST.shape):
        DST_reg[r0, r1] = T.if_then_else(
            r0 < 4,
            SRC[row0, head0 + T.min(r0, 4 - 1), r1],
            T.cast(0, DST.dtype),
        )
    T.copy(DST_reg, DST)


@T.macro
def ukernel_div_broadcast_0_dfloor_1eminus_30_2dims_rrg(A, B, C):
    for i0, i1 in T.Parallel(*C.shape):
        C[i0, i1] = A[i0, i1] / T.max(B[i0], 1e-30)


@T.macro
def ukernel_log2_1dims_rg(A, B):
    for i0 in T.Parallel(*B.shape):
        B[i0] = T.log2(A[i0])


@T.macro
def GraphNode_0_iter(
    PlaceHolder_QQ: T.Tensor((16, 128), "bfloat16"),
    PlaceHolder_K_CACHEE: T.Tensor((num_pages, 8, 128), "bfloat16"),
    PlaceHolder_KV_INDICESS: T.Tensor((n_idx), "int32"),
    PlaceHolder_V_CACHEE: T.Tensor((num_pages, 8, 128), "bfloat16"),
    tile_base: T.int32,
    valid_len: T.int32,
    scale2: T.float32,
    kv_pos0_next: T.int32,
    head: T.int32,
    t_paged_gather_k_pf: T.Tensor((64, 128), "bfloat16"),
    t_paged_gather_v_pf: T.Tensor((64, 128), "bfloat16"),
    t_rowsum_0: T.Tensor((16), "float32"),
    t_mma_1: T.Tensor((16, 128), "float32"),
):
    t_paged_gather_k = T.alloc_shared(
        (
            64,
            128,
        ),
        "bfloat16",
    )
    T.annotate_layout({t_paged_gather_k: tl.layout.make_swizzled_layout(t_paged_gather_k)})
    T.copy(t_paged_gather_k_pf, t_paged_gather_k)
    t_paged_gather_v = T.alloc_shared(
        (
            64,
            128,
        ),
        "bfloat16",
    )
    T.annotate_layout({t_paged_gather_v: tl.layout.make_swizzled_layout(t_paged_gather_v)})
    T.copy(t_paged_gather_v_pf, t_paged_gather_v)
    if tile_base + 64 < valid_len:
        ukernel_paged_gather_paged_base_head_cibuf_crbuf_ggr(
            PlaceHolder_K_CACHEE, PlaceHolder_KV_INDICESS, kv_pos0_next, head, t_paged_gather_k_pf
        )
        ukernel_paged_gather_paged_base_head_cibuf_crbuf_ggr(
            PlaceHolder_V_CACHEE, PlaceHolder_KV_INDICESS, kv_pos0_next, head, t_paged_gather_v_pf
        )
    t_mma_0 = T.alloc_fragment(
        (
            16,
            64,
        ),
        "float32",
    )
    ukernel_mma_ssr_transB_clear(PlaceHolder_QQ, t_paged_gather_k, t_mma_0)
    t_exp_0 = T.alloc_fragment(
        (
            16,
            64,
        ),
        "float32",
    )
    ukernel_maskedexp_scale_rt2_2dims_rr(t_mma_0, tile_base, valid_len, scale2, t_exp_0)
    ukernel_reduction_axis1_rr(t_exp_0, t_rowsum_0)
    t_exp_0_cast = T.alloc_shared(
        (
            16,
            64,
        ),
        "bfloat16",
    )
    T.annotate_layout({t_exp_0_cast: tl.layout.make_swizzled_layout(t_exp_0_cast)})
    T.copy(t_exp_0, t_exp_0_cast)
    ukernel_mma_ssr(t_exp_0_cast, t_paged_gather_v, t_mma_1)


@T.macro
def GraphNode_0(
    PlaceHolder_Q: T.Tensor((batch, 32, 128), "bfloat16"),
    PlaceHolder_K_CACHE: T.Tensor((num_pages, 1, 8, 128), "bfloat16"),
    PlaceHolder_KV_INDICES: T.Tensor((n_idx), "int32"),
    PlaceHolder_V_CACHE: T.Tensor((num_pages, 1, 8, 128), "bfloat16"),
    q_row0: T.int32,
    q_head0: T.int32,
    kv_base: T.int32,
    valid_len: T.int32,
    head: T.int32,
    scale2: T.float32,
    t_rowsum_0: T.Tensor((16), "float32"),
    t_mma_1: T.Tensor((16, 128), "float32"),
):
    PlaceHolder_QQ = T.alloc_shared(
        (
            16,
            128,
        ),
        "bfloat16",
    )
    T.annotate_layout({PlaceHolder_QQ: tl.layout.make_swizzled_layout(PlaceHolder_QQ)})
    ukernel_row_gather_valid4_r3_gs(PlaceHolder_Q, q_row0, q_head0, PlaceHolder_QQ)
    PlaceHolder_K_CACHEE = PlaceHolder_K_CACHE
    PlaceHolder_KV_INDICESS = PlaceHolder_KV_INDICES
    PlaceHolder_V_CACHEE = PlaceHolder_V_CACHE
    t_paged_gather_k_pf = T.alloc_fragment(
        (
            64,
            128,
        ),
        "bfloat16",
    )
    ukernel_paged_gather_paged_base_head_cibuf_crbuf_ggr(
        PlaceHolder_K_CACHEE, PlaceHolder_KV_INDICESS, kv_base + 64 * 0, head, t_paged_gather_k_pf
    )
    t_paged_gather_v_pf = T.alloc_fragment(
        (
            64,
            128,
        ),
        "bfloat16",
    )
    ukernel_paged_gather_paged_base_head_cibuf_crbuf_ggr(
        PlaceHolder_V_CACHEE, PlaceHolder_KV_INDICESS, kv_base + 64 * 0, head, t_paged_gather_v_pf
    )
    for i0 in T.serial(T.ceildiv(valid_len, 64)):
        GraphNode_0_iter(
            PlaceHolder_QQ=PlaceHolder_QQ,
            PlaceHolder_K_CACHEE=PlaceHolder_K_CACHEE,
            PlaceHolder_KV_INDICESS=PlaceHolder_KV_INDICESS,
            PlaceHolder_V_CACHEE=PlaceHolder_V_CACHEE,
            tile_base=64 * i0,
            valid_len=valid_len,
            scale2=scale2,
            kv_pos0_next=kv_base + 64 * (i0 + 1),
            head=head,
            t_paged_gather_k_pf=t_paged_gather_k_pf,
            t_paged_gather_v_pf=t_paged_gather_v_pf,
            t_rowsum_0=t_rowsum_0,
            t_mma_1=t_mma_1,
        )


@T.macro
def main_kernel_iter(
    PlaceHolder_Q: T.Tensor((batch, 32, 128), "bfloat16"),
    PlaceHolder_K_CACHE: T.Tensor((num_pages, 8, 128), "bfloat16"),
    PlaceHolder_V_CACHE: T.Tensor((num_pages, 8, 128), "bfloat16"),
    PlaceHolder_KV_INDICES: T.Tensor((n_idx), "int32"),
    OutputNode_0: T.Tensor((4, 128), "bfloat16"),
    OutputNode_1: T.Tensor((4), "float32"),
    q_row0: T.int32,
    q_head0: T.int32,
    kv_base: T.int32,
    valid_len: T.int32,
    head: T.int32,
    scale2: T.float32,
):
    t_rowsum_0 = T.alloc_fragment((16,), "float32")
    T.clear(t_rowsum_0)
    t_mma_1 = T.alloc_fragment(
        (
            16,
            128,
        ),
        "float32",
    )
    T.clear(t_mma_1)
    GraphNode_0(
        PlaceHolder_Q,
        PlaceHolder_K_CACHE,
        PlaceHolder_KV_INDICES,
        PlaceHolder_V_CACHE,
        q_row0,
        q_head0,
        kv_base,
        valid_len,
        head,
        scale2,
        t_rowsum_0,
        t_mma_1,
    )
    ukernel_div_broadcast_0_dfloor_1eminus_30_2dims_rrg(t_mma_1, t_rowsum_0, OutputNode_0)
    ukernel_log2_1dims_rg(t_rowsum_0, OutputNode_1)


@T.prim_func
def main_kernel(
    PlaceHolder_Q: T.Tensor((batch, 32, 128), "bfloat16"),
    PlaceHolder_K_CACHE: T.Tensor((num_pages, 1, 8, 128), "bfloat16"),
    PlaceHolder_V_CACHE: T.Tensor((num_pages, 1, 8, 128), "bfloat16"),
    PlaceHolder_KV_INDPTR: T.Tensor((len_indptr), "int32"),
    PlaceHolder_KV_INDICES: T.Tensor((n_idx), "int32"),
    sm_scale: T.float32,
    OutputNode_0: T.Tensor((batch, 32, 128), "bfloat16"),
    OutputNode_1: T.Tensor((batch, 32), "float32"),
):
    with T.Kernel(batch, 8, threads=(128)) as (i0, i1):
        main_kernel_iter(
            PlaceHolder_Q=_S(PlaceHolder_Q[0 : 0 + batch, 0 : 0 + 32, 0 : 0 + 128], squeeze_dim=[]),
            PlaceHolder_K_CACHE=_S(
                PlaceHolder_K_CACHE[0 : 0 + num_pages, 0 : 0 + 1, 0 : 0 + 8, 0 : 0 + 128],
                squeeze_dim=[1],
            ),
            PlaceHolder_V_CACHE=_S(
                PlaceHolder_V_CACHE[0 : 0 + num_pages, 0 : 0 + 1, 0 : 0 + 8, 0 : 0 + 128],
                squeeze_dim=[1],
            ),
            PlaceHolder_KV_INDICES=_S(PlaceHolder_KV_INDICES[0 : 0 + n_idx], squeeze_dim=[]),
            OutputNode_0=_S(
                OutputNode_0[i0 : i0 + 1, 4 * i1 : 4 * i1 + 4, 0 : 0 + 128], squeeze_dim=[0]
            ),
            OutputNode_1=_S(OutputNode_1[i0 : i0 + 1, 4 * i1 : 4 * i1 + 4], squeeze_dim=[0]),
            q_row0=i0,
            q_head0=i1 * 4,
            kv_base=PlaceHolder_KV_INDPTR[i0],
            valid_len=PlaceHolder_KV_INDPTR[i0 + 1] - PlaceHolder_KV_INDPTR[i0],
            head=i1,
            scale2=sm_scale * 1.4426950408889634,
        )
