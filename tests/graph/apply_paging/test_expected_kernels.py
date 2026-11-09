"""The paging pass reproduces every tuned kernel byte for byte, at the registry's specs
(``cleave.dynamic.OPERATORS``), so the searched pipeline inherits their measurements.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tilelang")

from tests.graph.apply_paging._test_utils import OPERATOR_CASES
from cleave.dynamic.registry import operator_kernel_source
from cleave.dynamic import OPERATORS
from tests.graph.handwritten._test_utils import (
    EXPECTED_KERNELS_DIR,
    assert_expected_tilelang_kernel,
)

# The families this pass is responsible for. An expected kernel whose name starts with one of these
# is a dynamic-access operator and must have a spec in the registry.
PAGED_FAMILIES = ("gqa_paged_", "gqa_ragged_", "mla_paged_")


@pytest.mark.parametrize("operator", sorted(OPERATOR_CASES))
def test_reproduces_the_expected_kernel(operator):
    assert_expected_tilelang_kernel(operator, operator_kernel_source(operator))


def test_every_shipped_operator_is_covered():
    """Every paged expected kernel in ``expected_kernels/`` must have a registry entry.

    ``glob``, not ``rglob``, so subdirectories are not picked up; extend the prefix tuple
    when a family is added.
    """
    expected = {
        path.stem
        for path in EXPECTED_KERNELS_DIR.glob("*.py")
        if path.stem.startswith(PAGED_FAMILIES)
    }
    assert expected, f"found no paged expected kernels in {EXPECTED_KERNELS_DIR}"
    assert set(OPERATOR_CASES) == expected, (
        "the paging registry and the checked-in expected kernels have diverged: "
        f"only in registry {sorted(set(OPERATOR_CASES) - expected)}, "
        f"only in expected kernels {sorted(expected - set(OPERATOR_CASES))}"
    )
    # The third list: dims, page size and workloads. An expected kernel with a spec but no entry here
    # builds and reproduces its kernel, then dies with a KeyError the moment anything tries
    # to run it.
    assert set(OPERATORS) == expected, (
        "the workload registry and the checked-in expected kernels have diverged: "
        f"only in workloads {sorted(set(OPERATORS) - expected)}, "
        f"only in expected kernels {sorted(expected - set(OPERATORS))}"
    )
