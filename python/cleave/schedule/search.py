from typing import Callable, List, Optional

from ..core.target import current_target
from ..ir.graph import Graph, OutputNode
from ..ir.ops import Matmul
from .policy.default import DefaultPolicy, TCPolicy
from .policy.hint import TileDict


def search_tiling(
    main_graph: Graph,
    topk: int,
    force_fusion: bool = False,
    accept: Optional[Callable[[TileDict], bool]] = None,
) -> List[Graph]:
    """The policy's top ``topk`` tilings; ``accept`` drops candidates before truncation."""
    arch = current_target().arch
    is_tensorcore = False
    output_nodes = []
    for node in main_graph.nodes:
        if isinstance(node, Matmul):
            is_tensorcore = True
        elif isinstance(node, OutputNode):
            output_nodes.append(node)
    policy = (
        TCPolicy(output_nodes, arch, force_fusion=force_fusion, flat_graph=main_graph)
        if is_tensorcore
        else DefaultPolicy(
            output_nodes, arch, force_fusion=force_fusion, flat_graph=main_graph
        )
    )
    candidates = policy.emit_config(topk, accept=accept)
    return candidates
