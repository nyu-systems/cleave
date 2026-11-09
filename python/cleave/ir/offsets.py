"""A tile's start index per axis as a function of the loop indices (``Iterator`` offsets)."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union


@dataclass(frozen=True)
class OffsetTerm:
    """One axis of a tile offset: ``[scale * ](base[ // divisor])``.

    ``base`` is an int (index of the offset's argument) or a fixed expression string.
    """

    base: Union[int, str]
    scale: Optional[int] = None
    divisor: int = 1

    def render(self, args) -> str:
        expr = f"{args[self.base]}" if isinstance(self.base, int) else self.base
        if self.divisor > 1:
            expr = f"({expr} // {self.divisor})"
        return expr if self.scale is None else f"{self.scale} * {expr}"


@dataclass(frozen=True)
class TileOffset:
    """A tile's start index per axis as a function of the loop indices; ``None`` is 0."""

    terms: Tuple[Optional[OffsetTerm], ...]
    num_args: int

    def __call__(self, *args) -> Tuple[str, ...]:
        if len(args) != self.num_args:
            raise TypeError(
                f"TileOffset takes {self.num_args} indices, got {len(args)}"
            )
        return tuple("0" if t is None else t.render(args) for t in self.terms)

    @property
    def __code__(self):  # arity, as a lambda's ``__code__.co_argcount`` reported it
        return SimpleNamespace(co_argcount=self.num_args)


def get_offset_lambda(
    tile_shape: List[int],
    arg_indices: List[Optional[int]],
    num_args: int,
    iter_repeat_factors: Optional[List[int]] = None,
) -> TileOffset:
    assert len(arg_indices) == len(
        tile_shape
    ), "arg_indices must have the same length as tile_shape"
    assert num_args >= 1, "num_args must be positive"
    iter_repeat_factors = iter_repeat_factors or [1] * len(tile_shape)
    assert len(iter_repeat_factors) == len(
        tile_shape
    ), "iter_repeat_factors must have the same length as tile_shape"

    terms = []
    for t, arg_idx, factor in zip(tile_shape, arg_indices, iter_repeat_factors):
        if arg_idx is None:
            terms.append(None)
            continue
        assert (
            0 <= arg_idx < num_args
        ), f"Invalid lambda arg index {arg_idx} for lambda arity {num_args}"
        assert factor >= 1, "iter_repeat_factors must be >= 1."
        terms.append(OffsetTerm(arg_idx, scale=t, divisor=factor))
    return TileOffset(tuple(terms), num_args)
