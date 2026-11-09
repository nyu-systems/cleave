"""C = X @ W (M=128, N=128, K=4096, fp16).

    python demo/matmul.py                       # solo kernels
    python demo/matmul.py --split K:4,8,16      # + split-K partial/combine plans
    python demo/matmul.py --preset fast -v      # smaller tiling pool, per-stage progress
"""
import cleave
import cleave.lang as cl


@cleave.jit
def matmul(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    return x @ w


if __name__ == "__main__":
    cleave.examples.main(matmul, shapes=dict(M=128, N=128, K=4096), dtype=cl.f16)
