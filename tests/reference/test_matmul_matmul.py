import tilelang.language as T

from tests._graphs import matmul_matmul_graph
from cleave.util import compile_and_run_prim_func

DTYPE = "float16"


def matmul_matmul(M, N, K, block_M=64, block_N=64, num_stages=3, threads=128):

    dtype = "float16"
    accum_dtype = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((N, K), dtype),
        Output: T.Tensor((M, K), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bi:
            A_shared = T.alloc_shared([block_M, K], dtype)
            B_shared = T.alloc_shared([K, block_N], dtype)
            C_shared = T.alloc_shared([block_N, K], dtype)

            t_mma_0 = T.alloc_fragment([block_M, block_N], accum_dtype)
            t_mma_0_cast = T.alloc_fragment([block_M, block_N], dtype)
            t_mma_1 = T.alloc_fragment([block_M, K], accum_dtype)

            T.copy(A[bi * block_M : (bi + 1) * block_M, :], A_shared)

            T.clear(t_mma_1)

            loop_range = T.ceildiv(N, block_N)

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(B[:, k * block_N : (k + 1) * block_N], B_shared)
                T.copy(C[k * block_N : (k + 1) * block_N, :], C_shared)

                T.clear(t_mma_0)
                T.gemm(
                    A_shared,
                    B_shared,
                    t_mma_0,
                    transpose_A=False,
                    transpose_B=False,
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

            T.copy(t_mma_1, Output[bi * block_M : (bi + 1) * block_M, :])

    return main


def test_matmul_matmul_small_k():
    M, N, K = 4096, 4096, 32
    func = matmul_matmul(M=M, N=N, K=K, block_M=64, block_N=64, num_stages=3, threads=128)
    compile_and_run_prim_func(
        func, graph=matmul_matmul_graph(M, N, K), dtype=DTYPE
    )


def test_matmul_matmul_small_n():
    M, N, K = 1024, 64, 64
    func = matmul_matmul(M=M, N=N, K=K, block_M=64, block_N=64, num_stages=3, threads=128)
    compile_and_run_prim_func(
        func, graph=matmul_matmul_graph(M, N, K), dtype=DTYPE
    )
