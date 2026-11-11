"""The handwritten-schedule cases: one row per expected kernel.

``test_handwritten.py`` codegens and runs each case; ``test_expected_codegen.py`` only
codegens it. ``CASES`` maps the kernel name to its :class:`Case`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Callable

from cleave.dynamic import GATHER_TOLERANCE
from cleave.runtime.validate import DEFAULT_TOLERANCE, Tolerance

from tests import _graphs as g
from cleave.dynamic.workloads import cases


def _schedule(name):
    return importlib.import_module(f"tests.graph.handwritten.handwritten_schedules.{name}")


@dataclass(frozen=True)
class Case:
    """``parts`` are ``(schedule, dims)`` pairs; more than one is a fused split-k kernel.

    ``source(*dims)`` builds the reference graph. A fused case checks against
    ``source(*source_dims)`` with scratch partials; a paged case (``out_idx`` set) runs
    the captured workloads for the operator instead.
    """

    name: str
    parts: tuple
    source: Callable | None = None
    source_dims: tuple | None = None
    out_idx: list | None = None
    workload_kw: dict = field(default_factory=dict)
    dtype: str = "float16"
    tolerance: Tolerance = DEFAULT_TOLERANCE

    def graph(self):
        graphs = [_schedule(s).get_handwritten_scheduled_graph(dims=d) for s, d in self.parts]
        return graphs[0] if len(graphs) == 1 else graphs

    def check_kwargs(self):
        kw = dict(dtype=self.dtype, kernel_name=self.name, tolerance=self.tolerance)
        dims = self.parts[0][1]
        if self.out_idx is not None:
            kw.update(
                out_idx=self.out_idx,
                workloads_fn=lambda: cases(self.name, dims, **self.workload_kw),
            )
        elif len(self.parts) > 1:
            kw.update(scratch_source_graph=self.source(*self.source_dims))
        else:
            kw.update(source_graph=self.source(*dims))
        return kw


def _static(name, dims, source):
    return Case(name, ((name, dims),), source=source)


def _fused(name, partial, combine, partial_dims, source):
    *lead, split_k, m, n, k_partial = partial_dims
    return Case(
        name,
        ((partial, partial_dims), (combine, (*lead, split_k, m, n))),
        source=source,
        source_dims=(*lead, m, n, split_k * k_partial),
    )


def _paged(name, dims, out_idx, **workload_kw):
    if name.startswith("mla"):
        workload_kw["scale"] = _schedule(name).SM_SCALE
    return Case(
        name, ((name, dims),), out_idx=out_idx, workload_kw=workload_kw,
        dtype="bfloat16", tolerance=GATHER_TOLERANCE,
    )


_ALL = [
    _fused("batched_matmul_splitk", "batched_matmul_splitk_partial",
           "batched_matmul_splitk_combine", (2, 4, 128, 128, 1024), g.batched_matmul_graph),
    _static("batched_matmul_splitk_combine", (2, 4, 128, 128),
            g.batched_matmul_splitk_combine_graph),
    _static("batched_matmul_splitk_partial", (2, 4, 128, 128, 1024),
            g.batched_matmul_splitk_partial_graph),
    _static("gqa_split_partial", (1, 32, 8, 4, 4096, 1024, 64), g.gqa_split_partial_graph),
    _static("matmul", (128, 128, 4096), g.matmul_graph),
    _static("matmul_matmul_bhsd", (1, 32, 4096, 4096, 64), g.matmul_matmul_bhsd_graph),
    _static("matmul_matmul_bshd", (1, 4096, 4096, 32, 64), g.matmul_matmul_bshd_graph),
    _static("matmul_matmul_small_k", (4096, 4096, 32), g.matmul_matmul_graph),
    _static("matmul_matmul_small_n", (1024, 64, 64), g.matmul_matmul_graph),
    _static("matmul_rowsum", (1024, 1024, 1024), g.matmul_rowsum_graph),
    _static("matmul_rowsum_small_k", (4096, 4096, 32), g.matmul_rowsum_graph),
    _static("matmul_rowsum_small_n", (1024, 128, 128), g.matmul_rowsum_graph),
    _fused("matmul_splitk", "matmul_splitk_partial", "matmul_splitk_combine",
           (4, 128, 128, 1024), g.matmul_graph),
    _static("matmul_splitk_combine", (4, 128, 128), g.matmul_splitk_combine_graph),
    _static("matmul_splitk_partial", (4, 128, 128, 1024), g.matmul_splitk_partial_graph),
    _static("mha_bhsd", (1, 32, 4096, 4096, 64), g.mha_bhsd_graph),
    _static("mha_bshd", (1, 4096, 4096, 32, 64), g.mha_bshd_graph),
    _static("mha_split_combine", (32, 4, 4096, 64), g.mha_split_combine_graph),
    _static("mha_split_partial", (1, 32, 4, 4096, 1024, 64), g.mha_split_partial_graph),
    _static("mqa_split_partial", (1, 32, 1, 4, 4096, 1024, 64), g.gqa_split_partial_graph),
    _static("rmsnorm_matmul", (1024, 6144, 4096), g.rmsnorm_matmul_graph),
    _static("rmsnorm_matmul_split_combine", (4, 1024, 6144, 4096),
            g.rmsnorm_matmul_split_combine_graph),
    _static("rmsnorm_matmul_split_partial", (4, 1024, 6144, 1024),
            g.rmsnorm_matmul_split_partial_graph),
    _static("swiglu", (256, 6144, 4096), g.swiglu_graph),
    # dims: num_qo_heads, num_kv_heads, head_dim[, page_size] / heads, ckv, kpe for MLA
    _paged("gqa_paged_decode_h32_kv4_d128_ps1", (32, 4, 128), [6, 7]),
    _paged("gqa_paged_decode_h32_kv8_d128_ps1", (32, 8, 128), [6, 7]),
    _paged("gqa_paged_decode_h32_kv8_d128_ps64", (32, 8, 128, 64), [7, 8], page_size=64),
    _paged("gqa_paged_prefill_causal_h32_kv4_d128_ps1", (32, 4, 128), [7, 8]),
    _paged("gqa_paged_prefill_causal_h32_kv8_d128_ps1", (32, 8, 128), [7, 8]),
    _paged("gqa_ragged_prefill_causal_h32_kv4_d128", (32, 4, 128), [6, 7]),
    _paged("gqa_ragged_prefill_causal_h32_kv8_d128", (32, 8, 128), [6, 7]),
    _paged("mla_paged_decode_h16_ckv512_kpe64_ps1", (16, 512, 64), [7, 8]),
    _paged("mla_paged_prefill_causal_h16_ckv512_kpe64_ps1", (16, 512, 64), [8, 9]),
]
CASES = {case.name: case for case in _ALL}
