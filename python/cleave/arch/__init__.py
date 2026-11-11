import tilelang  # noqa: F401 - puts its bundled tvm on sys.path

from .arch_base import Arch
from .cuda import CUDA
from typing import Union
from tvm.target import Target


def get_arch(target: Union[str, Target] = "cuda") -> Arch:
    if target == "cuda":
        import os

        from ..core.target import no_cuda_device

        if no_cuda_device():  # describe one (see cleave.core.target)
            arch = os.environ.get("CLEAVE_TARGET_ARCH", "sm_90a").rstrip("a")
            target = {"kind": "cuda", "arch": arch, "max_num_threads": 1024, "thread_warp_size": 32}
    if isinstance(target, (str, dict)):
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
