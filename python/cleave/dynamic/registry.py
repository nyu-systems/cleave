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


# The absolute part of every lse bound. It is not measured (no captured row has a zero scale)
# and it is not an operator's slack: it is what an lse of exactly 0 can be off by, 4 fp32
# ulps of a sum of 1 in log2 units (4 * 2**-23 / ln 2 = 6.9e-7), rounded up to a 1-2-5 step.
LSE_ATOL = 1e-6


def _tolerance(out_rtol, lse_rtol):
    """``(out, lse)`` bounds of one operator, each set from its measured error.

    Both are relative to the row. For the bf16 output the row is the head_dim vector of one
    query row and head, and there is no absolute part: an empty causal row is exactly 0.
    For the fp32 lse the row is the heads of one query row. Its error grows with its
    magnitude (measured per operator over |lse| of 2 .. 25 on the captured workloads and up
    to 3000 on the adversarial ones: the largest absolute error rises with the row max, from
    2e-6 to 4e-3, the largest relative one stays level, between 2e-7 and 5e-6), so an
    absolute bound sized on one logit scale is wrong on every other: at |lse| = 450 (logits
    x100) one fp32 ulp is already 3e-5. ``LSE_ATOL`` only keeps a row whose lse is 0 from
    demanding a bit-exact answer; an empty row's -inf is matched exactly either way.

    ``lse_rtol`` is the next 1-2-5 step at or above 4x the largest row-relative error the
    operator's valid kernels measured over three seeds on its normal inputs: its captured
    workloads and the hand-made requests of its tests that leave the inputs as generated,
    single kernels and split-KV pairs alike. Two kinds of comparison are counted and are no
    input to it.

    One is an lse output whose every entry is within ``LSE_ATOL``. It is inside the bound
    whatever the rtol, and its row-relative error is the floor's share of a row near 0, not
    an error that grows with the row.

    The other is a comparison on adversarial inputs: the operator's adversarial workloads
    and the hand-made requests that transform the inputs (logits x100 through the queries
    or the scale, a ramp or one dominant key along the keys, a mirrored scale). The softmax
    of every layer is exact, so every kernel is validated on them too, at this same bound:
    it is declared from the normal inputs and never loosened for an adversarial case, and a
    kernel that needs more than it there fails. What they measure, next to the normal
    inputs: 4.1e-7 .. 6.9e-7 against 2.3e-7 .. 5.7e-7 for the GQA layers, 1.1e-6 against
    1.1e-6 for MLA decode and 5.0e-6 against 1.5e-6 for MLA prefill, each inside its bound
    by a factor of 2 to 7.

    The MLA prefill figure is not all the kernel's. The reference is float32 as well, and
    where an adversarial workload makes the queries and a key row constant -- every product
    of a 576-term MLA score of one sign -- it is the reference that is off: against the
    value those inputs allow to be formed with two roundings, the kernel is within 1.1e-7
    and the reference 3.2e-6 away.

    ``out`` follows the same rule at 2x: bf16 has 8 significand bits, so one ulp is 7.8e-3
    of the row max, every kernel measures 1 ulp (7.8e-3), and 4x would be the 5e-2 ceiling,
    looser than the elementwise 0.01 + 0.01|ref| these operators were held to before for any
    row whose max is above 0.25. 2e-2 is 2.56 ulps.
    (``python -m tests._measure_tolerances`` re-measures; ``tests/measured_errors.json``
    is the measurement these values are pinned to.)
    """
    return (Tolerance(rtol=out_rtol), Tolerance(rtol=lse_rtol, atol=LSE_ATOL))


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
    # The declared (out, lse) validation bounds; every operator states its own.
    tolerance: tuple
    # (distribution, axes) adversarial softmax workloads. The softmax of every layer is
    # exact (a running max), so it holds at any logit scale, and every validation of the
    # operator -- single kernel or split-KV pair -- runs these after the captured ones.
    # They are never timed.
    adversarial: tuple = ()
    page_size: int = 1
    scale: float | None = None

    @property
    def is_mla(self) -> bool:
        return isinstance(self.spec, MLASpec)

    def cases(self):
        """(inputs, expected outputs) for every captured workload, built one at a time."""
        return (self.reference(axes) for axes in self.workloads)

    def keyed_workloads(self, *, keep: int = 0, adversarial: bool = False):
        return keyed_workloads(self.name, keep=keep, adversarial=adversarial)

    def kernel_source(self) -> str:
        """The TileLang the pipeline emits for this operator at its tuned tiling."""
        from .dense import mla_paged_kernel_source, paged_kernel_source

        build = mla_paged_kernel_source if self.is_mla else paged_kernel_source
        return build(self.spec, **self.tuned_tiling)

    @property
    def expected_kernel_path(self) -> Path:
        return EXPECTED_KERNELS_DIR / f"{self.name}.py"


def _operator(name, spec, tuned_tiling, dims, tolerance, page_size=1, scale=None) -> DynamicOperator:
    return DynamicOperator(
        name=name,
        spec=spec,
        tuned_tiling=dict(tuned_tiling),
        dims=dims,
        workloads=_wl.WORKLOADS[name],
        reference=partial(_wl.case, name, dims=dims, page_size=page_size, scale=scale),
        tolerance=tolerance,
        adversarial=_wl.ADVERSARIAL_WORKLOADS[name],
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
            _tolerance(out_rtol=2e-2, lse_rtol=1e-6),
        ),
        _operator(
            "gqa_paged_decode_h32_kv4_d128_ps1",
            # kv4 puts the kv head on the leading grid axis; kv8 measured better the other way.
            PagingSpec(num_qo_heads=32, num_kv_heads=4, grid_order=("head", "batch")),
            _GQA,
            (32, 4, 128),
            _tolerance(out_rtol=2e-2, lse_rtol=1e-6),
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
            _tolerance(out_rtol=2e-2, lse_rtol=2e-6),
            page_size=64,
        ),
        _operator(
            "gqa_paged_prefill_causal_h32_kv8_d128_ps1",
            PagingSpec(num_qo_heads=32, num_kv_heads=8, kind=PAGED_PREFILL_CAUSAL, causal=True),
            # A prefill block's M axis fuses (token, head slot); the narrow KV tile is
            # measured -- the captured requests' causal windows are a handful of tokens.
            _GQA_PREFILL,
            (32, 8, 128),
            _tolerance(out_rtol=2e-2, lse_rtol=2e-6),
        ),
        _operator(
            "gqa_paged_prefill_causal_h32_kv4_d128_ps1",
            PagingSpec(num_qo_heads=32, num_kv_heads=4, kind=PAGED_PREFILL_CAUSAL, causal=True),
            _GQA_PREFILL,
            (32, 4, 128),
            _tolerance(out_rtol=2e-2, lse_rtol=2e-6),
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
            _tolerance(out_rtol=2e-2, lse_rtol=5e-6),
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
            _tolerance(out_rtol=2e-2, lse_rtol=2e-6),
        ),
        _operator(
            "mla_paged_decode_h16_ckv512_kpe64_ps1",
            # 256 threads and the look-ahead read are one decision: the BN x 512 compressed
            # tile spills as a fragment at 128 threads.
            MLASpec(num_qo_heads=16),
            dict(query_rows=16, kv_tile=64, n_threads=256, ckv=512, kpe=64),
            (16, 512, 64),
            _tolerance(out_rtol=2e-2, lse_rtol=5e-6),
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
            _tolerance(out_rtol=2e-2, lse_rtol=1e-5),
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


def keyed_workloads(operator, *, keep: int = 0, adversarial: bool = False):
    """A zero-arg factory yielding ``(workload_key, inputs, reference)``.

    ``keep=0`` is every captured workload; ``keep>0`` a stratified sample of that many (a
    prefix would sample only the degenerate shapes, since ``WORKLOADS`` is smallest-first).
    ``adversarial`` appends the operator's adversarial softmax workloads
    (``workloads.ADVERSARIAL_WORKLOADS``), keyed ``<workload>@<distribution>``.
    """
    from ..runtime.autotune import sampled, workload_key

    op = OPERATORS[operator]
    # Every case seeds its own generator, so skipping one does not perturb the rest.
    wanted = [op.workloads[i] for i in sorted(set(sampled(range(len(op.workloads)), keep)))]

    def factory():
        for axes in wanted:
            yield (workload_key(axes), *op.reference(axes))
        if adversarial:
            for distribution, axes in op.adversarial:
                yield (f"{workload_key(axes)}@{distribution}",
                       *op.reference(axes, distribution=distribution))

    return factory
