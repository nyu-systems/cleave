"""Cleave package."""

from typing import List

from ._cache_env import configure_cache_env as _configure_cache_env

_configure_cache_env()

__all__: List[str] = [
    "TilelangCodeGen",
    "Graph",
    "GraphNode",
    "OutputNode",
    "PlaceHolderNode",
    "passes",
    "superoptimizer",
    "search_tiling",
    "apply_tiling",
]


def __getattr__(name: str):
    if name == "TilelangCodeGen":
        from .codegen import TilelangCodeGen

        return TilelangCodeGen
    if name in {"Graph", "GraphNode", "OutputNode", "PlaceHolderNode"}:
        from .graph import Graph, GraphNode, OutputNode, PlaceHolderNode

        return {
            "Graph": Graph,
            "GraphNode": GraphNode,
            "OutputNode": OutputNode,
            "PlaceHolderNode": PlaceHolderNode,
        }[name]
    if name in {"search_tiling", "apply_tiling"}:
        from .schedule import apply_tiling, search_tiling

        return {"search_tiling": search_tiling, "apply_tiling": apply_tiling}[name]
    if name == "passes":
        from importlib import import_module

        return import_module(".passes", __name__)
    if name == "superoptimizer":
        from importlib import import_module

        return import_module(".superoptimizer", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
