from __future__ import annotations

import difflib
from pathlib import Path

import tilelang as tl

from cleave.codegen import TilelangCodeGen
from cleave.util import (
    DEFAULT_TOLERANCE,
    Tolerance,
    compile_and_run,
    compile_and_run_prim_func,
)
from cleave.util.env import _gpu_exec_guard
from cleave.util.graph_executor import build_inputs_and_reference, torch_dtype
from cleave.util.runtime import (
    _compile,
    _load_main_kernel_from_code,
    _validate_kernel_outputs,
)

from tests.graph.handwritten.handwritten_schedules import ensure_main_graph_threads

# Scratch parameter-name suffixes emitted by the fused split-k codegen: the split
# kernel writes ``<base>_split`` and the combine kernel reads ``<base>_combine``,
# but both name the SAME physical partial buffer (see codegen ``_get_partial_name``).
_SCRATCH_SUFFIXES = ("_split", "_combine")

EXPECTED_KERNELS_DIR = Path(__file__).resolve().parent / "expected_kernels"

# The tolerance the paged/ragged gather kernels are held to. They reproduce the
# SOL-ExecBench references far more tightly than the legacy loose bound (measured off-tree:
# output cosine >= 0.99999, lse error <= 1e-5), so the 0.5 bound the older kernels
# grandfather in would not be meaningful for them. ``equal_nan`` stays off: a causal row
# with an empty KV window is defined to be 0 in the output and -inf in the lse, not NaN,
# so a NaN anywhere is a real defect and must fail (SOL-ExecBench's own sanity check
# rejects NaN unconditionally).
GATHER_TOLERANCE = Tolerance(rtol=0.01, atol=0.01, equal_nan=False)


def _expected_kernel_path(kernel_name: str) -> Path:
    return EXPECTED_KERNELS_DIR / f"{kernel_name}.py"


def assert_expected_tilelang_kernel(kernel_name: str, generated_code: str) -> None:
    expected_path = _expected_kernel_path(kernel_name)
    expected_code = expected_path.read_text(encoding="utf-8")
    if generated_code == expected_code:
        return

    diff = "".join(
        difflib.unified_diff(
            expected_code.splitlines(keepends=True),
            generated_code.splitlines(keepends=True),
            fromfile=str(expected_path),
            tofile=f"<generated:{kernel_name}>",
        )
    )
    raise AssertionError(
        f"Generated Tilelang kernel does not match {expected_path}.\n{diff}"
    )


def _kernel_input_buffers(kernel):
    """The compiled kernel's input buffers, in parameter order.

    Mirrors :func:`cleave.util.graph_executor.order_inputs_for_kernel`'s param walk,
    but returns the buffers (not just names) so callers can size scratch tensors
    from the kernel's own declared shapes/dtypes rather than hand-coded dims.
    """
    adapter = kernel.adapter
    prim_func = adapter.prim_func
    buffer_map = prim_func.buffer_map
    result_idx = set(adapter.result_idx)
    return [
        buffer_map[param]
        for param_idx, param in enumerate(prim_func.params)
        if param_idx not in result_idx and param in buffer_map
    ]


def _run_with_scratch(
    code_str, *, source_graph, dtype, tolerance: Tolerance = DEFAULT_TOLERANCE
):
    """Validate a fused split-k kernel against a plain ``source_graph`` reference.

    The fused split+combine kernel takes A, B *plus* two internal partial buffers
    (``Partial_A_split`` / ``Partial_A_combine``) as parameters, so its argument
    surface is not a single cleave graph. But its numerics ARE just the source
    graph's (e.g. ``A @ B``): the partials are scratch the kernel fills itself.

    So we keep ``source_graph`` as the sole ground truth -- it supplies the real
    inputs (by placeholder name) and the reference -- and allocate every remaining
    kernel parameter as uninitialized scratch, sized from the kernel's own buffer
    map. Parameters sharing a base name (``<base>_split`` / ``<base>_combine``)
    alias ONE physical buffer, matching how the split kernel's writes are read back
    by the combine kernel.
    """
    import torch

    main_kernel = _load_main_kernel_from_code(code_str)
    # Through cleave's own compile helper, so the pass configs the generated module
    # declares are honoured here as they are everywhere else.
    kernel = _compile(main_kernel, [-1])

    with _gpu_exec_guard():
        inputs_by_name, reference, _ = build_inputs_and_reference(source_graph, dtype)

        scratch_by_base: dict[str, "torch.Tensor"] = {}
        ordered_inputs = []
        for buffer in _kernel_input_buffers(kernel):
            name = buffer.name
            if name in inputs_by_name:
                ordered_inputs.append(inputs_by_name[name])
                continue
            # A parameter the source graph does not supply is kernel scratch. Alias
            # the ``_split``/``_combine`` pair for one base name to a single buffer.
            base = name
            for suffix in _SCRATCH_SUFFIXES:
                if name.endswith(suffix):
                    base = name[: -len(suffix)]
                    break
            if base not in scratch_by_base:
                shape = tuple(int(s) for s in buffer.shape)
                scratch_by_base[base] = torch.empty(
                    shape, device="cuda", dtype=torch_dtype(str(buffer.dtype))
                )
            ordered_inputs.append(scratch_by_base[base])

        _validate_kernel_outputs(kernel, ordered_inputs, reference, tolerance=tolerance)


def run_handwritten_test(
    graph,
    *,
    source_graph=None,
    scratch_source_graph=None,
    dtype: str,
    kernel_name: str,
    reference_fn=None,
    make_inputs_fn=None,
    workloads_fn=None,
    out_idx=None,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
) -> None:
    """Assert the scheduled ``graph`` codegens to the expected kernel and runs.

    ``graph`` is the (post-tiling) scheduled graph exercised by the codegen
    equality check. The numeric reference is normally derived by walking an
    equivalent pre-tiling cleave ``source_graph`` with the graph executor.

    A fused kernel whose parameter list carries internal scratch buffers on top of
    the source graph's inputs (e.g. the split-k partial buffers) passes that plain
    ground-truth graph as ``scratch_source_graph``: its inputs/reference drive the
    check and the extra kernel parameters are allocated as scratch (see
    :func:`_run_with_scratch`).

    As a generic escape hatch, a caller may instead supply explicit
    ``make_inputs_fn`` / ``reference_fn`` closures plus ``out_idx`` (mirroring
    ``cleave.util.compile_and_run_prim_func``). No in-tree test needs this today --
    every layout, including interior-head ``bshd`` attention (via ``PermuteView``),
    is expressible as a ``source_graph`` -- but the path is kept for one-off kernels
    whose reference has no cleave-graph equivalent.

    A kernel with symbolic extents serves every shape of its operator from one compile, so
    it is checked against a sequence of workloads rather than a single one: ``workloads_fn``
    yields ``(inputs, reference)`` pairs and ``out_idx`` names the outputs.
    """
    ensure_main_graph_threads(graph)
    code_str = TilelangCodeGen().gen(graph)
    print("Generated Code:")
    print(code_str)

    assert_expected_tilelang_kernel(kernel_name, code_str)

    if scratch_source_graph is not None:
        _run_with_scratch(
            code_str,
            source_graph=scratch_source_graph,
            dtype=dtype,
            tolerance=tolerance,
        )
        return

    if reference_fn is not None or workloads_fn is not None:
        # Generated kernels are code strings, so reuse compile_and_run_prim_func's
        # closure path by first materializing the ``main_kernel`` prim func.
        main_kernel = _load_main_kernel_from_code(code_str)
        compile_and_run_prim_func(
            main_kernel,
            dtype=dtype,
            reference_fn=reference_fn,
            make_inputs_fn=make_inputs_fn,
            workloads_fn=workloads_fn,
            out_idx=out_idx,
            tolerance=tolerance,
        )
        return

    compile_and_run(code_str, graph=source_graph, dtype=dtype, tolerance=tolerance)
