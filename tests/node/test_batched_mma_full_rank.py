import pytest

from tilelang.carver.roller import Edge  # noqa: F401 (ensures tilelang/tvm path setup)
from cleave.arch import get_arch
from cleave.graph import OutputNode
from cleave.ops import Matmul
from cleave.policy.default import TCPolicy


def _build_batched_mma_with_broadcast_batch() -> Matmul:
    a, b, c, d, e, f = 1, 32, 1, 4096, 4096, 64
    return Matmul(
        (a * b, d, f, e),
        input_batch_dims_A=(a * b,),
        input_batch_dims_B=(a * c,),
        dtype="float16",
        out_dtype="float32",
        name="matmul_localized",
        warp_policy="rows",
    )


def _build_policy_and_batched_mma() -> tuple[TCPolicy, Matmul]:
    node = _build_batched_mma_with_broadcast_batch()
    node.set_tag("tensorcore_config", True)
    output_node = OutputNode(node, dtype="float16")
    return TCPolicy([output_node], get_arch("cuda")), node


@pytest.mark.parametrize(
    "output_tile, rstep, expected_placeholder_tile",
    [
        ([1, 64, 32], None, [1, 4096, 32]),
        ([1, 64, 64], None, [1, 4096, 64]),
        ([1, 64, 32], 64, [1, 64, 32]),
        ([1, 64, 64], 64, [1, 64, 64]),
    ],
)
def test_batched_mma_full_rank_input_propagation(
    output_tile, rstep, expected_placeholder_tile
):
    policy, node = _build_policy_and_batched_mma()
    rstep_map = (
        {axis.var.name: rstep for axis in node.raxis} if rstep is not None else None
    )
    propagated_inputs = policy._propagate_inputs_full_rank(
        node, output_tile, rstep=rstep_map
    )
    propagated_input = propagated_inputs[1]

    assert len(propagated_input) == 3
    assert propagated_input == expected_placeholder_tile
