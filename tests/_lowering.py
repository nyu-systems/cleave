"""Lower a generated kernel to CUDA source without a GPU.

``tilelang.lower`` for an offline ``sm_90a`` target runs the whole TileLang pipeline --
layout inference, pipelining, tensor-core lowering -- on the CPU, the way the JIT does it
(same pass configs, taken from the generated module). ``device_compile=True`` also runs the
device compile (nvcc), which needs the CUDA toolchain but still no device.
"""

from __future__ import annotations


def lower_to_cuda(code: str, *, device_compile: bool = False, arch: str = "sm_90a") -> str:
    import tilelang
    import tvm
    from tvm.target import Target

    from cleave.runtime.compile import _load_main_kernel_from_code

    main_kernel = _load_main_kernel_from_code(code)
    target = Target({"kind": "cuda", "arch": arch})
    configs = dict(main_kernel._cleave_pass_configs)
    with tvm.transform.PassContext(opt_level=3, config=configs), target:
        artifact = tilelang.lower(main_kernel, target=target, enable_device_compile=device_compile)
    return artifact.kernel_source


def run_lowered(ir_graph, program, inputs) -> list:
    """Torch reference of a lowered flat graph on a Program's inputs (program order).

    The inputs reach the graph the way a compiled kernel gets them: bound to placeholders
    by ``pipeline.plans.bind_inputs``, then fed in the graph's argument order.
    """
    from cleave.pipeline.plans import bind_inputs
    from cleave.runtime.reference import execute, placeholder_order

    names = bind_inputs(ir_graph, program)
    assert names is not None, "placeholders do not bind to the program inputs"
    by_name = dict(zip(names, inputs))
    return execute(ir_graph, [by_name[n.name] for n in placeholder_order(ir_graph)])
