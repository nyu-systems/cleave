"""``import cleave.lang as cl``: dims, dtypes, annotations and ops of the Cleave frontend."""

from .frontend.composites import (  # noqa: F401
    exp, matmul, max, mean, rms_norm, rsqrt, silu, sqrt, sum,
)
from .frontend.paged import PageTable, ScheduleHints, paged_gather, softmax  # noqa: F401
from .frontend.tensor import (  # noqa: F401
    DType, Dim, Scalar, Split, Tensor, bf16, dims, f16, f32, i32,
)

__all__ = [
    "DType", "Dim", "PageTable", "Scalar", "ScheduleHints", "Split", "Tensor", "Tolerance", "bf16",
    "dims", "exp", "f16", "f32", "i32", "matmul", "max", "mean", "paged_gather", "rms_norm", "rsqrt",
    "silu", "softmax", "sqrt", "sum", "workloads",
]


def __getattr__(name):  # heavy (tilelang) imports only on use
    if name == "Tolerance":
        from .runtime.validate import Tolerance

        return Tolerance
    if name == "workloads":
        from .frontend import paged

        return paged.workloads
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
