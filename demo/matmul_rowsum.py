"""Y = rowsum(X @ W) (M=4096, N=4096, K=64, fp16)."""
import cleave
import cleave.lang as cl


@cleave.jit
def matmul_rowsum(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    return (x @ w).sum(-1, keepdim=True)


if __name__ == "__main__":
    cleave.examples.main(matmul_rowsum, shapes=dict(M=4096, N=4096, K=64), dtype=cl.f16)
