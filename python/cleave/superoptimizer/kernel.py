from typing import *

from .core import *
from .visualizer import *
from .utils import *

class Handle:
    def __init__(self, handles=[], remain_op=None) -> None:
        self.handles = handles
        self.remain_op = remain_op

    def wait(self):
        for handle in self.handles:
            handle.wait()
        if self.remain_op:
            self.remain_op()


class KNGraph:
    def __init__(self, graph):
        self.cygraph = graph

        self._is_compiled = False
        self.run = None
        self._valid_cuda_kernels = False
        self._cached_results = None
        self.visualizer = None

        self.backend = "cuda"

    def new_input(self, dims: tuple, dtype: dtype = float16) -> DTensor:
        return self.cygraph.new_input(dims, dtype)

    def mark_output(self, A: DTensor):
        return self.cygraph.mark_output(A)

    def matmul(
        self, A: DTensor, B: DTensor, trans_A: bool = False, trans_B: bool = False
    ) -> DTensor:
        return self.cygraph.matmul(A, B, trans_A, trans_B)

    def reduction(self, A: DTensor, dim: int):
        return self.cygraph.reduction(A, dim)

    def exp(self, A: DTensor):
        return self.cygraph.exp(A)

    def silu(self, A: DTensor):
        return self.cygraph.silu(A)

    def gelu(self, A: DTensor):
        return self.cygraph.gelu(A)

    def relu(self, A: DTensor):
        return self.cygraph.relu(A)
    
    def clamp(self, A: DTensor, min_val: float, max_val: float):
        return self.cygraph.clamp(A, min_val, max_val)

    def add(self, A: DTensor, B: DTensor):
        return self.cygraph.add(A, B)

    def mul(self, A: DTensor, B: DTensor):
        return self.cygraph.mul(A, B)

    def div(self, A: DTensor, B: DTensor):
        return self.cygraph.div(A, B)

    def rms_norm(self, A: DTensor, normalized_shape: tuple):
        return self.cygraph.rms_norm(A, normalized_shape)

    def valid_kernels(self):
        assert self._is_compiled, "Should check kernel validness after compilation"
        return self._valid_cuda_kernels

    def __call__(self, **kwargs):
        raise NotImplementedError("__call__ is not implemented yet")

    def search(
        self,
        previous_checkpoint: str = None,
        config: str = None,
    ):
        cygraphs = search(
            self.cygraph,
            previous_checkpoint=previous_checkpoint,
            default_config=config,
        )
        return [KNGraph(g) for g in cygraphs]

    def visualize(self, file_name):
        operators = self.cygraph.get_graph_structure()
        self.visualizer = visualizer(file_name)
        self.visualizer.draw_graphs(operators)
