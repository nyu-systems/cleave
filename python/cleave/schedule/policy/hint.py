from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from tilelang.carver.roller.hint import Stride  # noqa: F401  (re-exported)

# accumulation contexts of an output (``TileDict.accumulation``, ``Graph.accumulation``)
INTRA, INTER = "intra", "inter"


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
    # Accumulation context of each output (keyed by the node that feeds an OutputNode),
    # set only for a graph whose outputs reduce over orthogonal axes: "intra" outputs
    # define the launch tile and accumulate over the CTA's loop; an "inter" output's tile
    # follows the loop variable and is summed across CTAs. Empty: every output is intra.
    accumulation: Dict = field(default_factory=dict)

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
