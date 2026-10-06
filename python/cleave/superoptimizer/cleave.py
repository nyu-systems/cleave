from . import core
from .core import DTensor


class TileGraph:
    def __init__(self, graph):
        self.cygraph = graph

    def create_tiled_input(
        self, dims: tuple, can_split=None, dtype: core.dtype = core.float16
    ) -> DTensor:
        if isinstance(can_split, core.dtype):
            dtype = can_split
            can_split = None
        if can_split is None:
            return self.cygraph.create_tiled_input(dims, dtype=dtype)
        return self.cygraph.create_tiled_input(dims, can_split, dtype)

    def create_tiled_output(self, tensor: DTensor):
        return self.cygraph.create_tiled_output(tensor)

    def matmul(
        self, A: DTensor, B: DTensor, trans_A: bool = False, trans_B: bool = False
    ) -> DTensor:
        return self.cygraph.matmul(A, B, trans_A, trans_B)

    def add(self, A, B) -> DTensor:
        return self.cygraph.add(A, B)

    def mul(self, A, B) -> DTensor:
        return self.cygraph.mul(A, B)

    def div(self, A, B) -> DTensor:
        return self.cygraph.div(A, B)

    def exp(self, A: DTensor) -> DTensor:
        return self.cygraph.exp(A)

    def sqrt(self, A: DTensor) -> DTensor:
        return self.cygraph.sqrt(A)

    def silu(self, A: DTensor) -> DTensor:
        return self.cygraph.silu(A)

    def reduction(self, A: DTensor, dim: int) -> DTensor:
        return self.cygraph.reduction(A, dim)

    def setup_search_config(self):
        return self.cygraph.setup_search_config()

    def search_tile_graph(
        self,
        checkpoint_file_name: str = "checkpoint.json",
        enable_concat_matmul_transformation: bool = False,
        show_config: bool = False,
        config=None,
    ):
        return self.cygraph.search_tile_graph(
            checkpoint_file_name,
            enable_concat_matmul_transformation,
            show_config,
            config,
        )


def create_symbolic_tile_graph(concrete_graph: TileGraph):
    symbolic_graph, symbol_map = core.create_symbolic_tile_graph(
        concrete_graph.cygraph
    )
    return TileGraph(symbolic_graph), symbol_map
