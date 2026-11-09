"""O = (Q @ K^T) @ V, BHSD (B*H=32, S_Q=S_KV=4096, D=64, fp16).
``--preset split``: S_Q=64, split S_KV by 4,8,16,32, max_ops=8 (``--split none --max-ops 5``
for its unsplit mode).
"""
import cleave
import cleave.lang as cl


@cleave.jit
def matmul_matmul(q: cl.Tensor["BH", "SQ", "D"], k: cl.Tensor["BH", "SKV", "D"], v: cl.Tensor["BH", "SKV", "D"]):
    return (q @ k.mT) @ v


if __name__ == "__main__":
    cleave.examples.main(
        matmul_matmul,
        shapes=dict(BH=32, SQ=4096, SKV=4096, D=64),
        dtype=cl.f16,
        presets={"split": dict(SQ=64, split=cl.Split("SKV", (4, 8, 16, 32)), max_ops=8)},
    )
