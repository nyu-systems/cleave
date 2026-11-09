"""Y = RMSNorm(X) @ W (M=4096, N=14336, K=4096, fp16).
``--preset split-k``: M=128, N=6144, split K by 4,8,16,32.

    python demo/rmsnorm_matmul.py
    python demo/rmsnorm_matmul.py --preset split-k --topk 100
"""
import cleave
import cleave.lang as cl


@cleave.jit
def rmsnorm_matmul(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    return cl.rms_norm(x, dim=-1) @ w


if __name__ == "__main__":
    cleave.examples.main(
        rmsnorm_matmul,
        shapes=dict(M=4096, N=14336, K=4096),
        dtype=cl.f16,
        presets={"split-k": dict(M=128, N=6144, split=cl.Split("K", (4, 8, 16, 32)))},
    )
