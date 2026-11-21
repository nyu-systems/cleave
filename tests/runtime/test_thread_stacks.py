"""Whoever holds the GPU lock leaves its CUDA context with the thread stacks it came with.

A kernel whose threads keep large local arrays needs more stack than a context has, 1 KiB
per thread. The driver then allocates the larger stack for every thread the device can
run and keeps it when the kernel is done: 4.3 GiB of device memory for the 15 KiB frame of
a row-sum quotient on 4096 columns, held by the process until it exits. Every worker of
``run_candidates`` that had launched such a kernel went on holding it, idle, and the
workers after it failed with ``CUDA_ERROR_OUT_OF_MEMORY`` at launch or
``torch.OutOfMemoryError`` while loading their inputs: 7 of the 28 kernels of
``test_end2end[rowsum_div]``, each of them valid, a different 4 to 7 in every run.

``_gpu_exec_guard`` puts the context's stack limit back before it gives up the lock, which
frees the stacks at once, so at most one process on the device holds any.

(The quotient's own kernels keep no such frame any more: its divisor is staged,
``tests/graph/apply_tiling/test_broadcast_over_input.py``. The guard is for whatever
kernel comes with one next.)
"""

import ctypes

import pytest
import torch

from cleave.runtime import gpu


class _FakeDriver:
    """``libcuda`` as far as the guard uses it: one context with a stack limit."""

    def __init__(self, limit=1024, status=0):
        self.limit, self.status, self.calls = limit, status, []

    def cuCtxGetLimit(self, value, which):
        assert which == 0  # CU_LIMIT_STACK_SIZE
        if self.status == 0:
            value._obj.value = self.limit
        return self.status

    def cuCtxSetLimit(self, which, value):
        assert which == 0 and isinstance(value, ctypes.c_size_t)
        self.calls.append(("set", value.value))
        self.limit = value.value
        return 0

    def cuCtxSynchronize(self):
        self.calls.append(("synchronize",))
        return 0


@pytest.fixture
def driver(monkeypatch, tmp_path):
    fake = _FakeDriver()
    monkeypatch.setattr(gpu, "_DRIVER", fake)
    monkeypatch.setattr(gpu, "_CUDA_CONTEXT_READY", True)
    monkeypatch.setattr(gpu, "_ensure_cuda_context", lambda: None)
    monkeypatch.setattr(gpu, "_PARALLEL_ENV_CONFIGURED", True)
    monkeypatch.setattr(gpu, "_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    return fake


def test_the_device_is_first_touched_under_its_lock(driver, monkeypatch):
    """A worker makes its CUDA context (a fill is launched) once it holds the device lock,
    not while it waits for it: another worker of its run is validating or timing a kernel
    then, and a kernel that runs while another process launches on the device can go on
    running for good (the candidate "that never returns": see ``_gpu_exec_guard``)."""
    import os

    held_by = []
    monkeypatch.setattr(gpu, "_ensure_cuda_context",
                        lambda: held_by.append(gpu.lock_state(gpu._GPU_LOCK_PATH)[0]))
    with gpu._gpu_exec_guard():
        assert held_by == [os.getpid()]
    assert gpu.lock_state(gpu._GPU_LOCK_PATH)[0] is None  # ... and it is given up after

    # a context that cannot be made (an exception past the helper) still gives the lock up
    def broken():
        raise RuntimeError("no device")

    monkeypatch.setattr(gpu, "_ensure_cuda_context", broken)
    with pytest.raises(RuntimeError, match="no device"):
        with gpu._gpu_exec_guard():
            raise AssertionError("not reached")
    assert gpu.lock_state(gpu._GPU_LOCK_PATH)[0] is None


def test_a_worker_probes_its_context_under_the_device_lock(driver, monkeypatch):
    """After a candidate that failed, a worker checks that its CUDA context still works: a
    synchronize and a fill that is read back. That is device work too."""
    import os
    import sys
    import types

    from cleave.runtime import workers

    seen = []

    def held():
        seen.append(gpu.lock_state(gpu._GPU_LOCK_PATH)[0] == os.getpid())

    class _Value:
        def item(self):
            held()
            return 0.0

    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_initialized=lambda: True, synchronize=held),
        zeros=lambda *a, **kw: _Value())
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert workers._cuda_context_healthy() is True
    assert seen == [True, True]
    assert gpu.lock_state(gpu._GPU_LOCK_PATH)[0] is None

    # a context that a fault broke: the probe raises, the worker retires, the lock is free
    def faulted():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    fake.cuda.synchronize = faulted
    assert workers._cuda_context_healthy() is False
    assert gpu.lock_state(gpu._GPU_LOCK_PATH)[0] is None
    # ... and a process that never touched the device has nothing to probe, or to wait for
    fake.cuda.is_initialized = lambda: False
    assert workers._cuda_context_healthy() is True


def test_the_guard_puts_back_a_stack_limit_that_a_launch_raised(driver):
    with gpu._gpu_exec_guard():
        driver.limit = 15552  # what launching the row-sum quotient does to the context
    # ... after the kernel is done, and before the lock is given up
    assert driver.limit == 1024 and driver.calls == [("synchronize",), ("set", 1024)]


def test_the_guard_leaves_a_context_alone_whose_stack_limit_did_not_move(driver):
    with gpu._gpu_exec_guard():
        pass
    assert driver.limit == 1024 and driver.calls == []


def test_the_limit_put_back_is_the_one_the_holder_came_with(driver):
    driver.limit = 4096  # set by whoever owns the process: not ours to lower
    with gpu._gpu_exec_guard():
        driver.limit = 15552
    assert driver.limit == 4096 and driver.calls[-1] == ("set", 4096)


def test_a_holder_that_fails_gives_back_its_stacks_and_the_lock(driver):
    with pytest.raises(RuntimeError, match="kernel fault"):
        with gpu._gpu_exec_guard():
            driver.limit = 15552
            raise RuntimeError("kernel fault")
    assert driver.limit == 1024
    with gpu._gpu_exec_guard():  # the lock is free again
        pass


def test_a_context_that_cannot_be_asked_is_left_alone(driver):
    """A context a kernel fault has torn down answers with an error, at any point."""
    driver.status = 201  # CUDA_ERROR_INVALID_CONTEXT
    with gpu._gpu_exec_guard():
        driver.limit = 15552
    assert driver.calls == []
    driver.status = 0
    with gpu._gpu_exec_guard():
        driver.status = 201  # torn down by the launch
    assert driver.calls == []


def test_a_process_without_a_context_does_not_load_the_driver(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "_DRIVER", None)
    monkeypatch.setattr(gpu, "_CUDA_CONTEXT_READY", False)
    monkeypatch.setattr(gpu, "_ensure_cuda_context", lambda: None)
    monkeypatch.setattr(gpu, "_PARALLEL_ENV_CONFIGURED", True)
    monkeypatch.setattr(gpu, "_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    with gpu._gpu_exec_guard():
        assert gpu._stack_limit() is None
    assert gpu._DRIVER is None


def test_gpu_a_kernel_with_large_thread_frames_keeps_no_device_memory():
    """The row-sum quotient on 1024 columns as it was built before its divisor was staged
    (``tests/graph/apply_tiling/test_broadcast_over_input.py``): each thread keeps a
    1024-element row, a frame of 3264 bytes. Launching it used to leave the process with
    0.56 GiB more device memory than it came with, for good."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from cleave.codegen import TilelangCodeGen
    from cleave.runtime.compile import _compile, _load_main_kernel_from_code
    from cleave.runtime.reference import graph_out_indices, make_inputs, order_inputs_for_kernel
    from cleave.runtime.reference import placeholder_order
    from cleave.runtime.validate import launch
    from tests.graph.apply_tiling import test_broadcast_over_input as quotient

    def in_use():
        free, total = torch.cuda.mem_get_info()
        return total - free

    cand = quotient._candidate([2, 1024], 64, dims=(64, 1024))
    tiled = quotient._annotated(cand)
    ((div, _, _),) = quotient._broadcasts(tiled)
    div.resolve_operand_layout = None  # the quotient takes the layout of the sum again
    code = TilelangCodeGen().gen(tiled)
    kernel = _compile(_load_main_kernel_from_code(code), graph_out_indices(cand.flat_graph))
    graph = cand.flat_graph
    by_name = {ph.name: t for ph, t in zip(placeholder_order(graph), make_inputs(graph, "float16"))}
    inputs = order_inputs_for_kernel(by_name, kernel)
    with gpu._gpu_exec_guard():
        limit, before = gpu._stack_limit(), in_use()
        out = launch(kernel, inputs)
        raised, inside = gpu._stack_limit(), in_use()
        assert all(torch.isfinite(o.float()).all()
                   for o in (out if isinstance(out, (list, tuple)) else [out]))
        del out
    assert limit is not None and raised > limit  # the launch did take larger stacks ...
    assert inside - before > 64 * 2**20  # ... on every thread of the device
    # ... and they are back once the lock is: the limit, and the memory with it
    assert gpu._stack_limit() == limit
    assert in_use() - before < (inside - before) // 4
