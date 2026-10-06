import tilelang.language as T

from tests._graphs import matmul_matmul_bhsd_graph
from cleave.util import compile_and_run_prim_func

DTYPE = "float16"


def matmul_matmul_bhsd(
    batch, heads, seq_len, dim, block_M=64, block_N=64, num_stages=3, threads=128
):

    dtype = "float16"
    accum_dtype = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((batch * heads, seq_len, dim), dtype),
        B: T.Tensor((batch * heads, seq_len, dim), dtype),
        C: T.Tensor((batch * heads, seq_len, dim), dtype),
        Output: T.Tensor((batch * heads, seq_len, dim), dtype),
    ):
        with T.Kernel(batch * heads, T.ceildiv(seq_len, block_M), threads=threads) as (
            bi,
            bj,
        ):
            A_shared = T.alloc_shared([block_M, dim], dtype)
            B_shared = T.alloc_shared([block_N, dim], dtype)
            C_shared = T.alloc_shared([block_N, dim], dtype)

            t_mma_0 = T.alloc_fragment([block_M, block_N], accum_dtype)
            t_mma_0_cast = T.alloc_fragment([block_M, block_N], dtype)
            t_mma_1 = T.alloc_fragment([block_M, dim], accum_dtype)

            T.copy(A[bi, bj * block_M : (bj + 1) * block_M, :], A_shared)

            T.clear(t_mma_1)

            loop_range = T.ceildiv(seq_len, block_N)

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(B[bi, k * block_N : (k + 1) * block_N, :], B_shared)
                T.copy(C[bi, k * block_N : (k + 1) * block_N, :], C_shared)

                T.clear(t_mma_0)
                T.gemm(
                    A_shared,
                    B_shared,
                    t_mma_0,
                    transpose_A=False,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                T.copy(t_mma_0, t_mma_0_cast)

                T.gemm(
                    t_mma_0_cast,
                    C_shared,
                    t_mma_1,
                    transpose_A=False,
                    transpose_B=False,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            T.copy(t_mma_1, Output[bi, bj * block_M : (bj + 1) * block_M, :])

    return main


def test_matmul_matmul_bhsd():
    dims = (1, 32, 4096, 4096, 64)
    batch, heads, seq_len, _, dim = dims
    func = matmul_matmul_bhsd(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        dim=dim,
        block_M=64,
        block_N=64,
        num_stages=3,
        threads=128,
    )
    compile_and_run_prim_func(
        func,
        graph=matmul_matmul_bhsd_graph(batch, heads, seq_len, seq_len, dim),
        dtype=DTYPE,
    )
