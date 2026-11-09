"""Y = SiLU(X @ W0) * (X @ W1) (M=4096, N=14336, K=4096, fp16)."""
import cleave
import cleave.lang as cl


@cleave.jit
def swiglu(x: cl.Tensor["M", "K"], w0: cl.Tensor["K", "N"], w1: cl.Tensor["K", "N"]):
    gate, up = x @ w0, x @ w1
    return cl.silu(gate) * up


if __name__ == "__main__":
    cleave.examples.main(swiglu, shapes=dict(M=4096, N=14336, K=4096), dtype=cl.f16)
