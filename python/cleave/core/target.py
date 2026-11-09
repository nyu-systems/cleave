from __future__ import annotations

import functools
import os


class Target:
    """A compilation target whose device is only probed on first use.

    With no visible GPU, ``CLEAVE_TARGET_ARCH`` (default ``sm_90a``) describes the device so
    graph building, tiling search and codegen still run (they need limits, not a device).
    """

    def __init__(self, kind: str = "cuda"):
        self.kind = kind

    @functools.cached_property
    def tvm_target(self):
        from tilelang.carver.arch import get_arch

        try:
            return get_arch(self.kind).target
        except RuntimeError:
            if self.kind != "cuda" or not no_cuda_device():
                raise
            target = _offline_target(os.environ.get("CLEAVE_TARGET_ARCH", "sm_90a"))
            _allow_offline_carver(target)
            return target

    @functools.cached_property
    def arch(self):
        from ..arch import get_arch

        return get_arch(self.kind)

    def __repr__(self):
        return f"Target({self.kind!r})"


def no_cuda_device() -> bool:
    """True only when no CUDA device exists; a present device that fails to probe raises."""
    import glob

    import tvm

    if tvm.runtime.cuda(0).exist:
        return False
    if os.environ.get("CUDA_VISIBLE_DEVICES", "x").strip() in ("", "-1"):
        return True
    if not glob.glob("/dev/nvidia[0-9]*"):
        return True
    raise RuntimeError(
        "cuda device 0 failed to probe although NVIDIA devices are present; refusing the "
        "offline target (set CUDA_VISIBLE_DEVICES='' to run without a GPU)"
    )


def _offline_target(arch: str):
    import tvm

    return tvm.target.Target(
        {"kind": "cuda", "arch": arch, "max_num_threads": 1024, "thread_warp_size": 32}
    )


def _allow_offline_carver(target):
    """carver's matmul analysis probes the device only for its sm version; give it one."""
    import tilelang.carver.matmul_analysis as ma
    from tilelang.carver.arch.cuda import CUDA, check_sm_version

    if getattr(ma.get_arch, "_cleave_offline", False):
        return
    probe = ma.get_arch

    def get_arch(t="cuda"):
        try:
            return probe(t)
        except RuntimeError:
            if not no_cuda_device():
                raise
            arch = CUDA.__new__(CUDA)
            arch.target, arch.platform = target, "CUDA"
            arch.sm_version = check_sm_version(target.attrs["arch"])
            return arch

    get_arch._cleave_offline = True
    ma.get_arch = get_arch


@functools.lru_cache(maxsize=None)
def _target(kind: str) -> Target:
    return Target(kind)


def default_target(kind: str = "cuda") -> Target:
    """The process-wide target of ``kind`` (one instance per kind)."""
    return _target(kind)


def current_target(kind: str = "cuda") -> Target:
    """The active ``Session``'s target when it has one of this kind, else the default."""
    from .session import active_session

    session = active_session()
    target = None if session is None else session.explicit_target
    return target if target is not None and target.kind == kind else default_target(kind)
