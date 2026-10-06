from pathlib import Path
import ctypes

def _preload_in_tree_z3():
    try:
        import z3
    except ImportError:
        return
    z3_build = Path(z3.__file__).resolve().parent / "lib"
    for libname in ("libz3.so.4.15", "libz3.so"):
        lib_path = z3_build / libname
        if lib_path.exists():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            return

try:
    from .core import *
except ImportError as exc:
    if "libz3" not in str(exc):
        raise
    _preload_in_tree_z3()
    from .core import *

from .kernel import *
from .cleave import *

class InputNotFoundError(Exception):
    """Raised when cannot find input tensors """
    pass

def new_kernel_graph():
    kgraph = core.CyKNGraph()
    return KNGraph(kgraph)

def new_tile_graph(split_enabled=False, symbolic=False):
    if isinstance(split_enabled, bool):
        normalized_split_enabled = split_enabled
    else:
        try:
            normalized_split_enabled = any(split_enabled)
        except TypeError:
            normalized_split_enabled = bool(split_enabled)
    tgraph = core.CyTileGraph(normalized_split_enabled, symbolic)
    return TileGraph(tgraph)

# Current Version
__version__ = "0.2.2"
