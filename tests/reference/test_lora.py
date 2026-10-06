import tilelang.language as T

from tests._graphs import lora_graph
from cleave.util import compile_and_run_prim_func

DTYPE = "float16"


def lora(M, N, K, R, block_M=16, block_N=64, block_K=64, num_stages=2, threads=32):
    # threads MUST be 32 (single warp): the (block_M, R) = (16, 16) t_D -> t_D_cast
    # fp16 cast fails TVM layout inference at >= 64 threads. The other tile knobs
    # (block_M/block_N/block_K/num_stages) are free.
    # NOTE: this hand-written fused kernel is the correctness reference for the
    # LoRA graph; demo/lora.py instead drives the superoptimizer search path.
    assert R % 16 == 0, "LoRA rank R must be a multiple of 16 (WGMMA layout constraint)"
    dtype = "float16"
    accum_dtype = "float32"

    @T.prim_func
    def main(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((K, N), dtype),
        A: T.Tensor((K, R), dtype),
        B: T.Tensor((R, N), dtype),
        Output: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads
        ) as (i0, i1):
            t_C = T.alloc_fragment((block_M, block_N), accum_dtype)  # base  X @ W
            t_D = T.alloc_fragment((block_M, R), accum_dtype)  # down  X @ A
            t_D_cast = T.alloc_fragment((block_M, R), dtype)  # fp32 -> fp16
            t_E = T.alloc_fragment((block_M, block_N), accum_dtype)  # up    D @ B
            B_shared = T.alloc_shared((R, block_N), dtype)
            T.clear(t_C)
            T.clear(t_D)

            for i2 in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                X_shared = T.alloc_shared((block_M, block_K), dtype)
                W_shared = T.alloc_shared((block_K, block_N), dtype)
                A_shared = T.alloc_shared((block_K, R), dtype)
                X_local = T.alloc_fragment((block_M, block_K), dtype)
                T.copy(
                    X[
                        i0 * block_M : (i0 + 1) * block_M,
                        i2 * block_K : (i2 + 1) * block_K,
                    ],
                    X_shared,
                )
                T.copy(
                    W[
                        i2 * block_K : (i2 + 1) * block_K,
                        i1 * block_N : (i1 + 1) * block_N,
                    ],
                    W_shared,
                )
                T.copy(A[i2 * block_K : (i2 + 1) * block_K, :], A_shared)
                T.copy(X_shared, X_local)
                T.gemm(
                    X_local,
                    W_shared,
                    t_C,
                    transpose_A=False,
                    transpose_B=False,
                    policy=T.GemmWarpPolicy.Square,
                )
                T.gemm(
                    X_local,
                    A_shared,
                    t_D,
                    transpose_A=False,
                    transpose_B=False,
                    policy=T.GemmWarpPolicy.Square,
                )

            T.copy(B[:, i1 * block_N : (i1 + 1) * block_N], B_shared)
            T.copy(t_D, t_D_cast)
            T.clear(t_E)
            T.gemm(
                t_D_cast,
                B_shared,
                t_E,
                transpose_A=False,
                transpose_B=False,
                policy=T.GemmWarpPolicy.Square,
            )

            for i, j in T.Parallel(block_M, block_N):
                t_C[i, j] = t_C[i, j] + t_E[i, j]

            T.copy(
                t_C,
                Output[
                    i0 * block_M : (i0 + 1) * block_M, i1 * block_N : (i1 + 1) * block_N
                ],
            )

    return main


def test_lora():
    M, N, K, R = 16, 4096, 4096, 16
    func = lora(
        M=M,
        N=N,
        K=K,
        R=R,
        block_M=16,
        block_N=64,
        block_K=64,
        num_stages=2,
        threads=32,
    )
    compile_and_run_prim_func(func, graph=lora_graph(M, N, K, R), dtype=DTYPE)
