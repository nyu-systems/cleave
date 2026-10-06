"""The paging pass reproduces every tuned kernel, byte for byte.

This is the load-bearing test of the whole dynamic-attention pipeline. Each of the nine
operators has a hand-tuned schedule whose emitted TileLang is checked in under
``expected_kernels/``, and whose measured standing against production attention kernels was
established from that text. If the pass -- fed a searched dense body and the operator's
spec -- emits the same text, then the searched pipeline inherits those measurements exactly,
with no benchmark to re-run and no tolerance to argue about.

It also pins the specs themselves. The registry in ``_test_utils.OPERATOR_CASES`` is what
the demos under ``demo/dynamic/`` use, so a wrong input name or a wrong tuning flag fails
here rather than silently shipping a kernel that is merely correct.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tilelang")

from tests.graph.apply_paging._test_utils import OPERATOR_CASES, operator_kernel_source
from tests.graph.handwritten import _paged_workloads as paged_workloads
from tests.graph.handwritten._test_utils import (
    EXPECTED_KERNELS_DIR,
    assert_expected_tilelang_kernel,
)

# The families this pass is responsible for. A golden whose name starts with one of these
# is a dynamic-access operator and must have a spec in the registry.
PAGED_FAMILIES = ("gqa_paged_", "gqa_ragged_", "mla_paged_")


@pytest.mark.parametrize("operator", sorted(OPERATOR_CASES))
def test_reproduces_the_expected_kernel(operator):
    assert_expected_tilelang_kernel(operator, operator_kernel_source(operator))


def test_every_shipped_operator_is_covered():
    """Both registries must cover every dynamic-access kernel that has a golden.

    Keyed off the golden files themselves rather than off another hand-kept list, so no two
    sides can drift together: adding a paged operator to ``expected_kernels/`` without
    teaching the pass to build it, or without giving it dims and workloads, fails here.

    ``glob`` and not ``rglob``, and kept that way deliberately: the benchmark's wrapper
    modules used to live in a subdirectory here with stems that match these prefixes, and
    although they now live under ``bench/assets/``, a recursive walk would pick up any future
    subdirectory the same way. The prefix filter is a convention, not a guarantee -- a paged
    operator named outside these three families would slip past, so extend the tuple when a
    family is added.
    """
    goldens = {
        path.stem
        for path in EXPECTED_KERNELS_DIR.glob("*.py")
        if path.stem.startswith(PAGED_FAMILIES)
    }
    assert goldens, f"found no paged goldens in {EXPECTED_KERNELS_DIR}"
    assert set(OPERATOR_CASES) == goldens, (
        "the paging registry and the checked-in goldens have diverged: "
        f"only in registry {sorted(set(OPERATOR_CASES) - goldens)}, "
        f"only in goldens {sorted(goldens - set(OPERATOR_CASES))}"
    )
    # The third list: dims, page size and workloads. A golden with a spec but no entry here
    # builds and reproduces its kernel, then dies with a KeyError the moment anything tries
    # to run it.
    assert set(paged_workloads.OPERATORS) == goldens, (
        "the workload registry and the checked-in goldens have diverged: "
        f"only in workloads {sorted(set(paged_workloads.OPERATORS) - goldens)}, "
        f"only in goldens {sorted(goldens - set(paged_workloads.OPERATORS))}"
    )
