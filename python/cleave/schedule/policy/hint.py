from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from tilelang.carver.roller.hint import Stride  # noqa: F401  (re-exported)


@dataclass(eq=False, repr=False)
class TileDict:
    output_tile: Any
    rstep_map: Dict
    flat_graph: Any
    # schedule config (keyed by Node)
    tile_map: Dict = field(default_factory=dict)
    itile_map: Dict = field(default_factory=dict)
    cached_tensors_map: Dict = field(default_factory=dict)
    output_strides_map: Dict = field(default_factory=dict)
    tensor_strides_map: Dict = field(default_factory=dict)
    n_threads: Optional[int] = None  # set by the policy per candidate

    # analysis
    traffic: int = -1
    smem_cost: int = -1
    block_per_SM: int = -1
    num_wave: int = -1
    grid_size: int = -1
    valid: bool = True

    def get_tile(self, func) -> list[int]:
        return self.tile_map[func]

    def get_itile(self, func) -> list[int]:
        return self.itile_map[func]

    def get_rstep(self, node) -> dict[str, int]:
        return self.rstep_map[node]

    def __hash__(self) -> int:
        return hash(tuple(self.output_tile))
