import tilelang.language as T

from tests._graphs import matmul_rowsum_graph
from cleave.util import compile_and_run_prim_func

DTYPE = "float16"


def matmul_rowsum(
    M, N, K, block_M=128, block_N=128, block_K=32, num_stages=3, threads=128
):
    dtype = "float16"
    accum_dtype = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        Output: T.Tensor((M,), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as i0:
            t_rowsum_0 = T.alloc_fragment((block_M,), accum_dtype)
            T.clear(t_rowsum_0)

            for i1 in T.Pipelined(T.ceildiv(N, block_N), num_stages=num_stages):
                t_mma_0 = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(t_mma_0)

                for i2 in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                    A_shared = T.alloc_shared((block_M, block_K), dtype)
                    B_shared = T.alloc_shared((block_K, block_N), dtype)
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
                    T.gemm(
                        A_shared,
                        B_shared,
                        t_mma_0,
                        transpose_A=False,
                        transpose_B=False,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                T.reduce_sum(t_mma_0, t_rowsum_0, dim=1, clear=False)

            T.copy(t_rowsum_0, Output[i0 * block_M : (i0 + 1) * block_M])

    return main


def _run_matmul_rowsum(dims):
    M, N, K = dims
    func = matmul_rowsum(
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
        func, graph=matmul_rowsum_graph(M, N, K), dtype=DTYPE
    )


def test_matmul_rowsum():
    _run_matmul_rowsum((1024, 1024, 1024))


def test_matmul_rowsum_small_k():
    _run_matmul_rowsum((4096, 4096, 32))


def test_matmul_rowsum_small_n():
    _run_matmul_rowsum((1024, 128, 128))
