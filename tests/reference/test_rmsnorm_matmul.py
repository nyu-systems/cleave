import tilelang.language as T
import tilelang

from tests._graphs import rmsnorm_matmul_graph
from cleave.util import compile_and_run_prim_func

DTYPE = "float16"


def rmsnorm_matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_stages=5,
    dtype="float16",
    accum_dtype="float32",
    threads=128,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            A_local = T.alloc_fragment((block_M, block_K), dtype)
            A_local_squared = T.alloc_fragment((block_M, block_K), "float32")
            A_squared_sum = T.alloc_fragment((block_M,), "float32")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(A_squared_sum)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.copy(A_shared, A_local)
                for i, j in T.Parallel(block_M, block_K):
                    A_local_squared[i, j] = A_local[i, j] * A_local[i, j]
                T.reduce_sum(A_local_squared, A_squared_sum, dim=1, clear=False)
                T.gemm(
                    A_local,
                    B_shared,
                    C_local,
                    transpose_A=False,
                    transpose_B=False,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for i in T.Parallel(block_M):
                A_squared_sum[i] = T.rsqrt(A_squared_sum[i] / K + 1e-12)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] *= A_squared_sum[i]
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def test_rmsnorm_matmul():
    dims = (1024, 6144, 4096)
    M, N, K = dims
    func = rmsnorm_matmul(
        M=M,
        N=N,
        K=K,
        block_M=128,
        block_N=128,
        block_K=32,
        num_stages=3,
        threads=128,
    )
    compile_and_run_prim_func(
        func, graph=rmsnorm_matmul_graph(M, N, K), dtype=DTYPE
    )
