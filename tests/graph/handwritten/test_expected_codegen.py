"""Codegen-only expected-kernel check: each handwritten case's graph vs its expected kernel."""

import pytest

pytest.importorskip("tilelang")

from tests.graph.handwritten._cases import CASES
from tests.graph.handwritten._test_utils import (
    assert_expected_tilelang_kernel,
    gen_handwritten_code,
    update_expected_tilelang_kernel,
)


@pytest.mark.parametrize("name", sorted(CASES))
def test_expected_codegen(name, request):
    code = gen_handwritten_code(CASES[name].graph())
    if request.config.getoption("--update-expected"):
        update_expected_tilelang_kernel(name, code)
    else:
        assert_expected_tilelang_kernel(name, code)
