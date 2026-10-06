import pytest

pytest.importorskip("tilelang")
pytest.importorskip("tvm")

from cleave.ops import Div


@pytest.mark.parametrize(
    ("dims", "expected_space_dim"),
    [
        ((1, 128, 128), [1, 128, 128]),
        ((2, 128, 128), [2, 128, 128]),
    ],
    ids=["b1_singleton_batch", "b2_batch"],
)
def test_div_get_space_dim_preserves_batch_dim(dims, expected_space_dim):
    div = Div(
        dims,
        broadcast=0,
        dtype="float32",
        out_dtype="float32",
        name=f"div_{dims[0]}",
    )

    assert div.get_space_dim() == expected_space_dim
