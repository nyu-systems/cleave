"""Shared helpers for the apply_paging tests.

Each test drives the real pipeline as far as codegen -- dense source graph, pinned
tiling, ``apply_tiling``, ``apply_paging`` -- and then asserts something about the
result. The tiling is pinned rather than searched because these tests are about the
access layer; which tilings the policy *ranks* first is tested under tests/schedule/.
"""

from cleave.dynamic.registry import OPERATORS

# name -> (spec, tuned dense tiling), the shape the tests index.
OPERATOR_CASES: dict = {name: (op.spec, op.tuned_tiling) for name, op in OPERATORS.items()}
