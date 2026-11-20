"""The validation tolerance each dense test operator declares. There is no default bound.

One entry per operator, keyed by the graph builder the fixture validates against
(``tests/_graphs.py``, ``cleave.dynamic.dense``) or by the name of a traced test program.
The paged operators declare theirs in ``cleave.dynamic.registry``; the demos in ``demo/``.

Every value is measured, not chosen (``python -m tests._measure_tolerances`` re-measures and
prints the table): the next 1-2-5 step at or above 4x the largest row-normalized error of the
operator's valid candidates over three seeds, and never below one fp16 ulp of the row max
(1e-3), which two correct results may differ by. Errors are bimodal -- valid kernels within a
few ulp (<= 1.6e-3, 3.9e-3 for the quotient ``rowsum_div``), wrong ones above 1e-1 -- so a
candidate that needs more than its operator's value is wrong, not unlucky. Never raise a
value to make a kernel pass.

The measurement itself is checked in as ``tests/measured_errors.json`` and
``tests/test_tolerance_declarations.py`` holds every entry here to what the rule gives for
it, so a value cannot be edited by hand in either direction: re-measure and ``--pin``.
"""

from __future__ import annotations

from cleave.runtime.validate import Tolerance

CEILING = 5e-2  # no operator may declare a looser rtol than this
# ... and an fp32 output (the paged lse) none looser than this. Valid kernels measure
# 2e-7 .. 1.6e-6 of the row max; a kernel that loses one key in 8192 is 1.4e-5 off, which
# this ceiling keeps out of the "clearly valid" range (a quarter of it) when re-measuring.
FP32_CEILING = 2e-5

TOLERANCES = {
    # ---- tests/_graphs.py
    "matmul_graph": Tolerance(rtol=5e-3),
    "batched_matmul_graph": Tolerance(rtol=5e-3),
    "matmul_splitk_partial_graph": Tolerance(rtol=5e-3),
    "matmul_splitk_combine_graph": Tolerance(rtol=1e-3),
    "batched_matmul_splitk_partial_graph": Tolerance(rtol=5e-3),
    "batched_matmul_splitk_combine_graph": Tolerance(rtol=1e-3),
    "matmul_matmul_graph": Tolerance(rtol=5e-3),
    "matmul_matmul_bhsd_graph": Tolerance(rtol=5e-3),
    "matmul_matmul_bshd_graph": Tolerance(rtol=5e-3),
    "matmul_rowsum_graph": Tolerance(rtol=5e-3),
    "rowsum_graph": Tolerance(rtol=2e-3),
    # Measured once, by running its deselected end2end case by hand (22 valid candidates).
    "rowsum_div_graph": Tolerance(rtol=2e-2),
    "rmsnorm_matmul_graph": Tolerance(rtol=5e-3),
    "rmsnorm_matmul_split_partial_graph": Tolerance(rtol=5e-3),
    # No kernel of this operator validates today (its tests are deselected: TileLang fails
    # before one exists), so there is nothing to measure: the one-ulp floor until one does.
    "rmsnorm_matmul_split_combine_graph": Tolerance(rtol=1e-3),
    "swiglu_graph": Tolerance(rtol=5e-3),
    "lora_graph": Tolerance(rtol=5e-3),
    "mha_bhsd_graph": Tolerance(rtol=5e-3),
    "mha_bshd_graph": Tolerance(rtol=1e-2),
    "gqa_bhsd_graph": Tolerance(rtol=5e-3),
    "mqa_bhsd_graph": Tolerance(rtol=5e-3),
    "qknorm_graph": Tolerance(rtol=1e-2),
    "mla_prefill_bhsd_graph": Tolerance(rtol=5e-3),
    "mla_decode_bhsd_graph": Tolerance(rtol=5e-3),
    "gqa_split_partial_graph": Tolerance(rtol=5e-3),
    "mha_split_partial_graph": Tolerance(rtol=5e-3),
    "mha_split_combine_graph": Tolerance(rtol=5e-3),
    # (matmul_softmax_graph has no entry: its test is ignored because apply_tiling cannot
    # schedule it, so no kernel exists to measure. Measure it when that changes.)
    # ---- cleave.dynamic.dense
    "flash_attention_sd_graph": Tolerance(rtol=5e-3),
    # ---- traced test programs (tests/frontend/_programs.py, test_exact_softmax.py)
    "matmul": Tolerance(rtol=5e-3),
    "softmax": Tolerance(rtol=1e-3),
    "softmax_mid": Tolerance(rtol=2e-3),  # over the middle axis of a batched input
    "exact_attention": Tolerance(rtol=5e-3),
    # the fixed-shift attention, split: partials, combines and the chained Kernel
    "attention": Tolerance(rtol=5e-3),
}


def tolerance_for(operator) -> Tolerance:
    """The declared tolerance of a graph builder (or its name). Undeclared is an error."""
    name = operator if isinstance(operator, str) else operator.__name__
    try:
        return TOLERANCES[name]
    except KeyError:
        raise KeyError(
            f"no tolerance is declared for {name!r}; measure it and add it to "
            "tests/_tolerances.py"
        ) from None
