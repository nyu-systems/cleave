import numpy as np


class Stride:
    """
    Manages stride information for a given axis of a tensor.
    """

    def __init__(self, stride: int = 1, ax: int = -1) -> None:
        # which axis to put stride on
        self._ax: int = int(ax)
        # the stride size of the axis
        self._stride: int = int(stride)

    @property
    def ax(self) -> int:
        return self._ax

    @property
    def stride(self) -> int:
        return self._stride

    def compute_elements_from_shape(self, shape):
        original_shape = np.prod(shape)
        if not self.is_valid():
            strided_elem = original_shape
        else:
            assert self.ax < len(shape)
            strided_elem = np.prod(shape[0 : self.ax + 1]) * self.stride
            assert strided_elem >= original_shape
        return int(strided_elem)

    def is_valid(self):
        return self.ax >= 0

    def __repr__(self):
        return f"<Stride, {self._ax}, {self._stride}>"


class TileDict:
    def __init__(self, output_tile, rstep_map, flat_graph) -> None:
        self.output_tile = output_tile
        self.rstep_map = rstep_map
        # schedule config
        self.tile_map = {}
        self.itile_map = {}
        self.cached_tensors_map = {}
        self.output_strides_map = {}
        self.tensor_strides_map = {}

        # analysis
        self.traffic = -1
        self.smem_cost = -1
        self.block_per_SM = -1
        self.num_wave = -1
        self.grid_size = -1
        self.valid = True
        self.flat_graph = flat_graph

    def get_tile(self, func) -> list[int]:
        return self.tile_map[func]

    def get_itile(self, func) -> list[int]:
        return self.itile_map[func]

    def get_rstep(self, node) -> dict[str, int]:
        return self.rstep_map[node]

    def __hash__(self) -> int:
        return hash(tuple(self.output_tile))
