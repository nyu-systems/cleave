import tilelang.language as T

from tests._graphs import swiglu_graph
from cleave.util import compile_and_run_prim_func

DTYPE = "float16"


def swiglu(M, N, K, block_M=128, block_N=128, block_K=32, num_stages=3, threads=128):
    dtype = "float16"
    accum_dtype = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((K, N), dtype),
        Output: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads
        ) as (i0, i1):
            t_mma_0 = T.alloc_fragment((block_M, block_N), accum_dtype)
            t_mma_1 = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(t_mma_0)
            T.clear(t_mma_1)

            for i2 in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_shared = T.alloc_shared((block_K, block_N), dtype)
                A_local = T.alloc_fragment((block_M, block_K), dtype)
                T.copy(
                    A[
                        i0 * block_M : (i0 + 1) * block_M,
                        i2 * block_K : (i2 + 1) * block_K,
                    ],
                    A_shared,
                )
                T.copy(
                    B[
                        i2 * block_K : (i2 + 1) * block_K,
                        i1 * block_N : (i1 + 1) * block_N,
                    ],
                    B_shared,
                )
                T.copy(
                    C[
                        i2 * block_K : (i2 + 1) * block_K,
                        i1 * block_N : (i1 + 1) * block_N,
                    ],
                    C_shared,
                )
                T.copy(A_shared, A_local)
                T.gemm(
                    A_local,
                    C_shared,
                    t_mma_0,
                    transpose_A=False,
                    transpose_B=False,
                    policy=T.GemmWarpPolicy.Square,
                )
                T.gemm(
                    A_local,
                    B_shared,
                    t_mma_1,
                    transpose_A=False,
                    transpose_B=False,
                    policy=T.GemmWarpPolicy.Square,
                )
            for i, j in T.Parallel(block_M, block_N):
                t_mma_0[i, j] = t_mma_0[i, j] * T.sigmoid(t_mma_0[i, j]) * t_mma_1[i, j]

            T.copy(
                t_mma_0,
                Output[
                    i0 * block_M : (i0 + 1) * block_M, i1 * block_N : (i1 + 1) * block_N
                ],
            )

    return main


def test_swiglu():
    dims = (256, 6144, 4096)
    M, N, K = dims
    func = swiglu(
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
        func, graph=swiglu_graph(M, N, K), dtype=DTYPE
    )
