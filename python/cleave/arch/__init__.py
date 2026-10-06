from .arch_base import Arch
from .cuda import CUDA
from typing import Union
from tvm.target import Target


def get_arch(target: Union[str, Target] = "cuda") -> Arch:
    if isinstance(target, str):
        target = Target(target)

    if target.kind.name == "cuda":
        return CUDA(target)
    else:
        raise ValueError(f"Unsupported target: {target.kind.name}")


from .cuda import (
    is_cuda_arch,
    is_volta_arch,
    is_ampere_arch,
    is_ada_arch,
    is_hopper_arch,
    is_tensorcore_supported_precision,
    has_mma_support,
)
