import tvm
from tvm.target import Target
from .arch_base import Arch
from typing import List, Union
from .driver import cuda_driver


def _target_attr(target: Target, name: str):
    """Read a Target property that may sit on the object or inside ``Target.attrs``.

    Several of these moved from attributes into ``attrs`` upstream, so try both rather
    than pinning to whichever spelling a given tvm happens to use.
    """
    value = getattr(target, name, None)
    if value is None:
        value = target.attrs.get(name, None)
    return value


def check_sm_version(arch: str) -> int:
    sm_version = arch.replace("sm_", "")
    return int(sm_version) if sm_version.isdigit() else -1


def is_cuda_arch(arch: Arch) -> bool:
    return isinstance(arch, CUDA)


def is_volta_arch(arch: Arch) -> bool:
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version >= 70)
    conditions.append(arch.sm_version < 80)
    return all(conditions)


def is_ampere_arch(arch: Arch) -> bool:
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version >= 80 and arch.sm_version < 89)
    return all(conditions)


def is_ada_arch(arch: Arch) -> bool:
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version == 89)
    return all(conditions)


def is_hopper_arch(arch: Arch) -> bool:
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version == 90)
    return all(conditions)


def has_mma_support(arch: Arch) -> bool:
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version >= 80)
    return all(conditions)


volta_tensorcore_supported = [
    ("float16", "float32"),
    ("float16", "float16"),
]
ampere_tensorcore_supported = [
    ("bfloat16", "float32"),
    ("float16", "float32"),
    ("float16", "float16"),
    ("int8", "int32"),
    ("int4", "int32"),
    ("int2", "int32"),
    ("int1", "int32"),
]
ada_tensorcore_supported = [
    ("bfloat16", "float32"),
    ("float16", "float32"),
    ("float16", "float16"),
    ("int8", "int32"),
    ("e5m2_float8", "float32"),
    ("e4m3_float8", "float32"),
]
hopper_tensorcore_supported = ada_tensorcore_supported


# TODO: we should consider the dtype of the input a and b
# instead of assuming both a and b share the same dtype.
# As the tensorcore may supports e4m3_float8 * e5m2_float8
def is_tensorcore_supported_precision(
    in_dtype: str, accum_dtype: str, arch: Arch
) -> bool:

    if is_volta_arch(arch):
        return (in_dtype, accum_dtype) in volta_tensorcore_supported
    elif is_ampere_arch(arch):
        return (in_dtype, accum_dtype) in ampere_tensorcore_supported
    elif is_ada_arch(arch):
        return (in_dtype, accum_dtype) in ada_tensorcore_supported
    elif is_hopper_arch(arch):
        return (in_dtype, accum_dtype) in hopper_tensorcore_supported
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


class TensorInstruction(object):

    def __init__(
        self,
        name: str,
        shape: List[int],
    ):
        self.name: str = name
        # only hold the shape of M and N
        self.shape: List[int] = shape


class CUDA(Arch):

    def __init__(self, target: Union[Target, str]):
        if isinstance(target, str):
            target = tvm.target.Target(target)
        self.target = target
        self.sm_version = check_sm_version(_target_attr(self.target, "arch"))
        device = tvm.runtime.cuda(0)
        if not device.exist:
            raise RuntimeError("Cannot find cuda device 0.")
        self.name = cuda_driver.get_device_name()
        self.device: tvm.runtime.Device = device
        self.platform: str = "CUDA"
        # TODO (david): maybe static shared memory, can be improved in future
        # cuda driver API to get dynamic shared memory size?
        # self.smem_cap = cuda_driver.get_shared_memory_per_block()
        self.smem_cap: int = 164 * 2**10  # 164KB for A100
        self.compute_max_core = device.multi_processor_count
        self.warp_size = device.warp_size
        self.compute_capability = device.compute_version.replace(".", "")
        self.reg_cap: int = 65536
        self.max_smem_usage: int = 2 * self.smem_cap
        self.sm_partition: int = 4
        # Falls back to the device, not to `cudaDeviceProp`: a bare "cuda" target carries
        # only arch/max_num_threads/thread_warp_size, and this module's hand-written
        # cudaDeviceProp layout no longer matches the installed CUDA's struct, so reading
        # `l2CacheSize` out of it returns garbage (measured: 1 byte on an H100).
        l2 = _target_attr(target, "l2_cache_size_bytes")
        if l2 is None:
            l2 = getattr(device, "l2_cache_size_bytes", None)
        self.l2_cache_size_bytes: int = int(l2) if l2 else 0
        # the number of transaction size in bytes
        self.transaction_size: List[int] = [32, 128]  # in bytes
        # bandwidth in MB/s, will be used for recommend basic tile size
        # TODO: find some way to get the real bandwidth
        # However, the ratio of bandwidth between different devices can
        # be similar. The bandwidth can work for another devices as well.
        self.bandwidth: List[int] = [750, 12080]
        # get the available tensor instructions during runtime to avoid
        # the dependency of the tensor intrinsics registration
        self.available_tensor_instructions: List[TensorInstruction] = None

    def get_avaliable_tensorintrin_shapes(self):
        self.available_tensor_instructions = (
            TensorInstruction("mma", [16, 16]),
            TensorInstruction("wmma", [16, 16]),
        )
        return [t.shape for t in self.available_tensor_instructions]

    def __repr__(self):
        return f"CUDA({self.target})"
