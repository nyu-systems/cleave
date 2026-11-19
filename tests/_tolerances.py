"""The validation tolerance each dense test operator declares. There is no default bound.

One entry per operator, keyed by the graph builder the fixture validates against
(``tests/_graphs.py``, ``cleave.dynamic.dense``) or by the name of a traced test program.
The paged operators declare theirs in ``cleave.dynamic.registry``; the demos in ``demo/``.

The values below are a provisional, uniform 1e-2 (row-relative): each is replaced by the
operator's measured error in the commit that follows. Valid kernels sit near 1e-3, wrong ones
above 1e-1, so a candidate that needs more than this is wrong, not unlucky. Never raise a
value to make a kernel pass.
"""

from __future__ import annotations

from cleave.runtime.validate import Tolerance

CEILING = 5e-2  # no operator may declare a looser rtol than this

TOLERANCES = {
    # ---- tests/_graphs.py
    "matmul_graph": Tolerance(rtol=1e-2),
    "batched_matmul_graph": Tolerance(rtol=1e-2),
    "matmul_splitk_partial_graph": Tolerance(rtol=1e-2),
    "matmul_splitk_combine_graph": Tolerance(rtol=1e-2),
    "batched_matmul_splitk_partial_graph": Tolerance(rtol=1e-2),
    "batched_matmul_splitk_combine_graph": Tolerance(rtol=1e-2),
    "matmul_matmul_graph": Tolerance(rtol=1e-2),
    "matmul_matmul_bhsd_graph": Tolerance(rtol=1e-2),
    "matmul_matmul_bshd_graph": Tolerance(rtol=1e-2),
    "matmul_rowsum_graph": Tolerance(rtol=1e-2),
    "rowsum_graph": Tolerance(rtol=1e-2),
    "rmsnorm_matmul_graph": Tolerance(rtol=1e-2),
    "rmsnorm_matmul_split_partial_graph": Tolerance(rtol=1e-2),
    "rmsnorm_matmul_split_combine_graph": Tolerance(rtol=1e-2),
    "swiglu_graph": Tolerance(rtol=1e-2),
    "lora_graph": Tolerance(rtol=1e-2),
    "mha_bhsd_graph": Tolerance(rtol=1e-2),
    "mha_bshd_graph": Tolerance(rtol=1e-2),
    "gqa_bhsd_graph": Tolerance(rtol=1e-2),
    "mqa_bhsd_graph": Tolerance(rtol=1e-2),
    "qknorm_graph": Tolerance(rtol=1e-2),
    "mla_prefill_bhsd_graph": Tolerance(rtol=1e-2),
    "mla_decode_bhsd_graph": Tolerance(rtol=1e-2),
    "gqa_split_partial_graph": Tolerance(rtol=1e-2),
    "mha_split_partial_graph": Tolerance(rtol=1e-2),
    "mha_split_combine_graph": Tolerance(rtol=1e-2),
    # ---- cleave.dynamic.dense
    "flash_attention_sd_graph": Tolerance(rtol=1e-2),
    # ---- traced test programs (tests/frontend/_programs.py, test_exact_softmax.py)
    "matmul": Tolerance(rtol=1e-2),
    "softmax": Tolerance(rtol=1e-2),
    "exact_attention": Tolerance(rtol=1e-2),
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
