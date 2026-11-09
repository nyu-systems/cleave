"""Cleave package."""

from importlib import import_module

_LAZY = {
    "TilelangCodeGen": ".codegen",
    "Graph": ".ir.graph",
    "GraphNode": ".ir.graph",
    "OutputNode": ".ir.graph",
    "PlaceHolderNode": ".ir.graph",
    "search_tiling": ".schedule",
    "apply_tiling": ".schedule",
    # frontend
    "Program": ".frontend",
    "trace": ".frontend",
    "jit": ".frontend",
    "JitFunction": ".frontend",
    "InputSpec": ".frontend",
    "OutputSpec": ".frontend",
    # pipeline
    "compile": ".pipeline",
    "superoptimize": ".pipeline",
    "build": ".pipeline",
    "evaluate": ".pipeline",
    "load": ".pipeline",
    "list_presets": ".pipeline",
    "SuperoptConfig": ".pipeline",
    "ScheduleConfig": ".pipeline",
    "EvalConfig": ".pipeline",
    "GraphSet": ".pipeline",
    "SearchedGraph": ".pipeline",
    "Plan": ".pipeline",
    "PlanSet": ".pipeline",
    "BuildSet": ".pipeline",
    "Report": ".pipeline",
    "Kernel": ".pipeline",
    "CleaveError": ".core.errors",
    "Session": ".core.session",
}
_SUBMODULES = {
    "core", "ir", "superoptimizer", "lower", "schedule", "codegen", "runtime", "dynamic", "frontend",
    "pipeline", "lang", "examples",
}

__all__ = sorted([*_LAZY, *_SUBMODULES])


def __getattr__(name: str):
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    if name in _LAZY:
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
