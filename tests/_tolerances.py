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
value to make a kernel pass. (The outputs are float16 but for one operator, ``matmul_tf32``;
a float32 output has its own ulp and its own ceiling.)

The measurement itself is checked in as ``tests/measured_errors.json`` and
``tests/test_tolerance_declarations.py`` holds every entry here to what the rule gives for
it, so a value cannot be edited by hand in either direction: re-measure and ``--pin``.

An operator whose kernels return an output with a boundary role declares one bound per
role instead of one ``Tolerance``: the split partial of an exact softmax returns its float32
shift next to its float16 sums, and the rule gives each role the bound of its own outputs.
"""

from __future__ import annotations

from cleave.runtime.validate import Tolerance

CEILING = 5e-2  # no operator may declare a looser rtol than this
# ... and an fp32 output (the paged lse, the TF32 matmul) none looser than this. Valid
# kernels measure 2e-7 .. 1.6e-6 of the row max; a kernel that loses one key in 8192 is
# 1.4e-5 off, which this ceiling keeps out of the "clearly valid" range (a quarter of it)
# when re-measuring.
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
    "rowsum_div_graph": Tolerance(rtol=2e-2),
    "rmsnorm_matmul_graph": Tolerance(rtol=5e-3),
    "rmsnorm_matmul_split_partial_graph": Tolerance(rtol=5e-3),
    "rmsnorm_matmul_split_combine_graph": Tolerance(rtol=5e-3),
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
    # the attention with its softmax written out by hand on a constant shift (a plain exp, no
    # request), split: partials, combines and the chained Kernel
    "attention": Tolerance(rtol=5e-3),
    # the exact attention, split: partials, combines and the chained Kernel. By boundary
    # role: a partial returns its shift (the max of its split, float32) next to its sums.
    "exact_split_attention": {"": Tolerance(rtol=1e-2), "shift": Tolerance(rtol=5e-6)},
    # the exact softmax of an input, split over the softmax axis: the partial reads its
    # logits from storage, so its shift is the max of stored values
    "exact_split_rows": {"": Tolerance(rtol=5e-3), "shift": Tolerance(rtol=2e-7)},
    # ---- traced programs of tests/graph/apply_tiling
    # two reductions that share nothing, every tiling whose two steps agree
    # (test_unrelated_reductions.py): two outputs, and the two products added
    "two_matmuls": Tolerance(rtol=5e-3),
    "sum_of_matmuls": Tolerance(rtol=5e-3),
    # a returned product that another op reads, every tiling that is built
    # (test_returned_intermediates.py): next to a second matmul, and next to its double
    "product_and_chain": Tolerance(rtol=5e-3),
    "product_and_double": Tolerance(rtol=5e-3),
    # rows divided by the root of their looped sum of squares, every tiling that is built
    # (test_broadcast_over_input.py)
    "l2_normalize": Tolerance(rtol=5e-3),
    # ---- the one float32 operator (tests/reference/test_matmul_tf32.py)
    # A float32 matmul on operands TF32 holds exactly, against their float32 product. On
    # ordinary float32 operands the same kernels are 1e-3 off, the TF32 instruction's
    # truncation, which no float32 bound admits: that case has no entry and never will.
    "matmul_tf32": Tolerance(rtol=5e-6),
}


def tolerance_for(operator):
    """The declared tolerance of a graph builder (or its name): a ``Tolerance``, or one per
    boundary role. Undeclared is an error."""
    name = operator if isinstance(operator, str) else operator.__name__
    try:
        return TOLERANCES[name]
    except KeyError:
        raise KeyError(
            f"no tolerance is declared for {name!r}; measure it and add it to "
            "tests/_tolerances.py"
        ) from None
