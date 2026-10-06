import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial

from tests._graphs import mha_bhsd_graph
from cleave.util import compile_and_run_prim_func

DTYPE = "float16"


def mha_bhsd(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=128,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch * heads, seq_len, dim]
    dtype = "float16"
    accum_dtype = "float32"

    @T.macro
    def MMA0(
        K: T.Tensor(shape, dtype),
        Q_shared: T.SharedBuffer([block_M, dim], dtype),
        K_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        k: T.int32,
        bx: T.int32,
        by: T.int32,
    ):
        T.copy(K[by, k * block_N : (k + 1) * block_N, :], K_shared)
        if is_causal:
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(
                    bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)
                )
        else:
            T.clear(acc_s)
        T.gemm(
            Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
        )

    @T.macro
    def MMA1(
        V: T.Tensor(shape, dtype),
        V_shared: T.SharedBuffer([block_M, dim], dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        k: T.int32,
        by: T.int32,
    ):
        T.copy(V[by, k * block_N : (k + 1) * block_N, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def Softmax(
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        scores_max: T.FragmentBuffer([block_M], accum_dtype),
        scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        scores_sum: T.FragmentBuffer([block_M], accum_dtype),
        logsum: T.FragmentBuffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        # T.fill(scores_max, -T.infinity(accum_dtype)) # This is a BUG from TileLang, otherwise `logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]` can revert max
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        # To do causal softmax, we need to set the scores_max to 0 if it is -inf
        # This process is called Check_inf in FlashAttention3 code, and it only need to be done
        # in the first ceil_div(kBlockM, kBlockN) steps.
        # for i in T.Parallel(block_M):
        #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        for i, j in T.Parallel(block_M, block_N):
            # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            # max * log_2(e)) This allows the compiler to use the ffma
            # instruction instead of fadd and fmul separately.
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def Rescale(
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads * batch, threads=threads) as (
            bx,
            by,
        ):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N)
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                MMA0(K, Q_shared, K_shared, acc_s, k, bx, by)
                Softmax(
                    acc_s,
                    acc_s_cast,
                    scores_max,
                    scores_max_prev,
                    scores_scale,
                    scores_sum,
                    logsum,
                )
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, by)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[by, bx * block_M : (bx + 1) * block_M, :])

    return main


def test_mha_bhsd():
    dims = (1, 32, 4096, 4096, 64)
    batch, heads, seq_len, _, dim = dims
    func = mha_bhsd(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        dim=dim,
        is_causal=False,
        block_M=64,
        block_N=64,
        num_stages=3,
        threads=128,
    )
    compile_and_run_prim_func(
        func,
        graph=mha_bhsd_graph(batch, heads, seq_len, seq_len, dim),
        dtype=DTYPE,
    )
