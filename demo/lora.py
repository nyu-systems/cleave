"""LoRA: Y = X @ W + (X @ A) @ B (M=16, K=N=4096, rank R=16, fp16; Mirage's demo_lora dims).

Written with the imperative ``cleave.Program`` builder instead of ``@cleave.jit``; both give
the same Program. Factory keywords become flags (``--M 32 --R 64``).
"""
import cleave
import cleave.lang as cl


def lora(M=16, N=4096, K=4096, R=16) -> cleave.Program:
    p = cleave.Program("lora", dtype=cl.f16)
    M, K, N, R = p.dims(M=M, K=K, N=N, R=R)
    x, w = p.input("x", (M, K)), p.input("w", (K, N))
    a, b = p.input("a", (K, R)), p.input("b", (R, N))
    p.output(x @ w + (x @ a) @ b, name="y")
    return p


if __name__ == "__main__":
    cleave.examples.main(lora)
