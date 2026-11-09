"""Tiling search (``search_tiling``) and application (``apply_tiling``).

The module is also callable: ``cleave.schedule(graphs, ...)`` runs the pipeline's schedule stage.
"""

import sys
import types

from .search import search_tiling
from .tiling import apply_tiling, assign_tensor_map

__all__ = ["apply_tiling", "assign_tensor_map", "search_tiling"]


class _ScheduleModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        from ..pipeline import schedule

        return schedule(*args, **kwargs)


sys.modules[__name__].__class__ = _ScheduleModule
