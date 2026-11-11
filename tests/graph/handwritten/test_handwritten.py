import pytest

from tests.graph.handwritten._cases import CASES
from tests.graph.handwritten._test_utils import run_handwritten_test


@pytest.mark.parametrize("name", sorted(CASES))
def test_handwritten(name):
    case = CASES[name]
    run_handwritten_test(case.graph(), **case.check_kwargs())
