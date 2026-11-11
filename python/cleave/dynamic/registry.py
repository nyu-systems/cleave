"""The shipped dynamic (paged / ragged / MLA) attention operators.

Each entry pins the spec that names the operator's dynamic inputs and the dense tiling its
tuned kernel was measured at; ``tests/graph/apply_paging/test_expected_kernels.py`` holds
``kernel_source()`` to the checked-in expected kernel byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable

from .paging import (
    MLA_PAGED_PREFILL_CAUSAL,
    PAGED_DECODE_SLAB,
    PAGED_PREFILL_CAUSAL,
    RAGGED_PREFILL_CAUSAL,
    MLASpec,
    PagingSpec,
)
from ..runtime.validate import Tolerance
from . import workloads as _wl

# The paged/ragged gather kernels reproduce the SOL-ExecBench references far more tightly
# than the default loose bound (output cosine >= 0.99999, lse error <= 1e-5). ``equal_nan``
# stays off: an empty causal row is defined as 0 / -inf, so a NaN is a real defect.
GATHER_TOLERANCE = Tolerance(rtol=0.01, atol=0.01, equal_nan=False)

# The expected kernels are a test contract and stay in the source tree; only a checkout has them.
EXPECTED_KERNELS_DIR = (
    Path(__file__).resolve().parents[3] / "tests" / "graph" / "handwritten" / "expected_kernels"
)


@dataclass(frozen=True)
class DynamicOperator:
    name: str
    spec: PagingSpec | MLASpec
    tuned_tiling: dict
    # What the name fixes: (qo_heads, kv_heads, head_dim[, page_size]) or (qo_heads, ckv, kpe).
    dims: tuple
    # Captured workload axes, smallest first.
    workloads: tuple
    # axes -> (inputs, expected outputs) for one workload.
    reference: Callable = field(repr=False)
    tolerance: Tolerance = GATHER_TOLERANCE
    page_size: int = 1
    scale: float | None = None

    @property
    def is_mla(self) -> bool:
        return isinstance(self.spec, MLASpec)

    def cases(self):
        """(inputs, expected outputs) for every captured workload, built one at a time."""
        return (self.reference(axes) for axes in self.workloads)

    def keyed_workloads(self, *, keep: int = 0):
        return keyed_workloads(self.name, keep=keep)

    def kernel_source(self) -> str:
        """The TileLang the pipeline emits for this operator at its tuned tiling."""
        from .dense import mla_paged_kernel_source, paged_kernel_source

        build = mla_paged_kernel_source if self.is_mla else paged_kernel_source
        return build(self.spec, **self.tuned_tiling)

    @property
    def expected_kernel_path(self) -> Path:
        return EXPECTED_KERNELS_DIR / f"{self.name}.py"


def _operator(name, spec, tuned_tiling, dims, page_size=1, scale=None) -> DynamicOperator:
    return DynamicOperator(
        name=name,
        spec=spec,
        tuned_tiling=dict(tuned_tiling),
        dims=dims,
        workloads=_wl.WORKLOADS[name],
        reference=partial(_wl.case, name, dims=dims, page_size=page_size, scale=scale),
        page_size=page_size,
        scale=scale,
    )


_GQA = dict(query_rows=16, kv_tile=64, head_dim=128)
_GQA_PREFILL = dict(query_rows=64, kv_tile=16, head_dim=128)

OPERATORS: dict[str, DynamicOperator] = {
    op.name: op
    for op in (
        _operator(
            "gqa_paged_decode_h32_kv8_d128_ps1",
            PagingSpec(num_qo_heads=32, num_kv_heads=8),
            _GQA,
            (32, 8, 128),
        ),
        _operator(
            "gqa_paged_decode_h32_kv4_d128_ps1",
            # kv4 puts the kv head on the leading grid axis; kv8 measured better the other way.
            PagingSpec(num_qo_heads=32, num_kv_heads=4, grid_order=("head", "batch")),
            _GQA,
            (32, 4, 128),
        ),
        _operator(
            "gqa_paged_decode_h32_kv8_d128_ps64",
            # A 64-token page holds a whole KV tile, so the tile is a contiguous slab copied
            # as a BufferRegion (which TMA can lower). Bounds predication stays ON: it is
            # what predicates those slab copies.
            PagingSpec(
                num_qo_heads=32,
                num_kv_heads=8,
                kind=PAGED_DECODE_SLAB,
                page_size=64,
                grid_order=("head", "batch"),
                pass_configs={},
            ),
            _GQA,
            (32, 8, 128, 64),
            page_size=64,
        ),
        _operator(
            "gqa_paged_prefill_causal_h32_kv8_d128_ps1",
            PagingSpec(num_qo_heads=32, num_kv_heads=8, kind=PAGED_PREFILL_CAUSAL, causal=True),
            # A prefill block's M axis fuses (token, head slot); the narrow KV tile is
            # measured -- the captured requests' causal windows are a handful of tokens.
            _GQA_PREFILL,
            (32, 8, 128),
        ),
        _operator(
            "gqa_paged_prefill_causal_h32_kv4_d128_ps1",
            PagingSpec(num_qo_heads=32, num_kv_heads=4, kind=PAGED_PREFILL_CAUSAL, causal=True),
            _GQA_PREFILL,
            (32, 4, 128),
        ),
        _operator(
            "gqa_ragged_prefill_causal_h32_kv8_d128",
            # No page table: the keys are packed, so the KV tile is an affine slice. That
            # lets the loop pipeline, and the L2 swizzle pays because neighbouring blocks
            # share keys.
            PagingSpec(
                num_qo_heads=32,
                num_kv_heads=8,
                kind=RAGGED_PREFILL_CAUSAL,
                causal=True,
                rasterization=None,
                cache_inputs=("K", "V"),
            ),
            dict(query_rows=64, kv_tile=64, head_dim=128),
            (32, 8, 128),
        ),
        _operator(
            "gqa_ragged_prefill_causal_h32_kv4_d128",
            PagingSpec(
                num_qo_heads=32,
                num_kv_heads=4,
                kind=RAGGED_PREFILL_CAUSAL,
                causal=True,
                rasterization=None,
                cache_inputs=("K", "V"),
            ),
            dict(query_rows=64, kv_tile=64, head_dim=128),
            (32, 4, 128),
        ),
        _operator(
            "mla_paged_decode_h16_ckv512_kpe64_ps1",
            # 256 threads and the look-ahead read are one decision: the BN x 512 compressed
            # tile spills as a fragment at 128 threads.
            MLASpec(num_qo_heads=16),
            dict(query_rows=16, kv_tile=64, n_threads=256, ckv=512, kpe=64),
            (16, 512, 64),
            scale=_wl.MLA_SM_SCALE,
        ),
        _operator(
            "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1",
            # One 16-row query block is too little work for 8 warps, so neither the wider
            # block nor the prefetch.
            MLASpec(
                num_qo_heads=16,
                kind=MLA_PAGED_PREFILL_CAUSAL,
                causal=True,
                index_window_tiles=4,
                prefetch=False,
                rasterization=1,
                query_inputs=("Q_NOPE", "Q_PE"),
                cache_inputs=("CKV_CACHE", "KPE_CACHE"),
                index_input="KV_INDICES",
            ),
            dict(query_rows=16, kv_tile=64, n_threads=128, ckv=512, kpe=64),
            (16, 512, 64),
            scale=_wl.MLA_SM_SCALE,
        ),
    )
}


def get(name: str) -> DynamicOperator:
    try:
        return OPERATORS[name]
    except KeyError:
        raise KeyError(f"unknown dynamic operator {name!r}; known: {sorted(OPERATORS)}") from None


def operator_kernel_source(operator: str) -> str:
    return OPERATORS[operator].kernel_source()


def workloads(operator):
    """(inputs, expected outputs) for every captured workload of ``operator``."""
    return OPERATORS[operator].cases()


def keyed_workloads(operator, *, keep: int = 0):
    """A zero-arg factory yielding ``(workload_key, inputs, reference)``.

    ``keep=0`` is every captured workload; ``keep>0`` a stratified sample of that many (a
    prefix would sample only the degenerate shapes, since ``WORKLOADS`` is smallest-first).
    """
    from ..runtime.autotune import sampled, workload_key

    op = OPERATORS[operator]
    # Every case seeds its own generator, so skipping one does not perturb the rest.
    wanted = [op.workloads[i] for i in sorted(set(sampled(range(len(op.workloads)), keep)))]

    def factory():
        for axes in wanted:
            yield (workload_key(axes), *op.reference(axes))

    return factory
