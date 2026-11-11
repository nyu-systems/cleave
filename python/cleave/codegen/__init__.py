"""TileLang source generation for a tiled (hierarchical) ``Graph``."""

from typing import Sequence

from ..ir.graph import Graph
from .buffer import BASE_INDENT, TAB, CodeBuffer
from .format import IMPORT_CODE, UTIL_CODE, format_source
from .kernel import KernelEmitter
from .macros import MacroEmitter
from .operands import OperandResolver
from .split import SplitEmitter
from .validate import GraphValidator

__all__ = [
    "TilelangCodeGen",
    "CodeBuffer",
    "TAB",
    "BASE_INDENT",
    "IMPORT_CODE",
    "UTIL_CODE",
]


class TilelangCodeGen(KernelEmitter, SplitEmitter, OperandResolver, MacroEmitter, GraphValidator):
    def __init__(self, apply_rasterization: int = 16, debug: bool = False):
        self.apply_rasterization = apply_rasterization
        self.debug = debug
        self.all_names = set()
        self._reset()

    def _reset(self):
        self.graph_code = ""
        self.symbol_code = ""
        self.pass_code = ""
        self._reset_macros()

    def gen(self, graph: Graph | Sequence[Graph]):
        self._reset()
        graphs = list(graph) if isinstance(graph, (list, tuple)) else [graph]
        for g in graphs:
            self._validate_graph(g, is_main=True)
        self._declare_dyn_dims(graph)
        self._declare_pass_configs(graph)
        if len(graphs) > 1:
            self._gen_split_kernel(graphs)
        else:
            self.gen_graph(graphs[0], name="main_kernel", is_main=True)
        return self.format_code()

    def emit(self, code: str):
        self.graph_code += code

    def format_code(self):
        return format_source(
            self.debug,
            self.pass_code,
            self.symbol_code,
            self.prim_code,
            self.graph_code,
        )
