import textwrap

import tvm
from tilelang.carver.roller.node import Edge

from ..graph import PlaceHolderNode


class MacroOp:
    """An op whose TileLang macro is written out directly, not derived from a prim_func.

    ``codegen.emit_op_node`` needs exactly two things from a node -- a macro name and a
    macro body -- so an op that has a hand-written body does not need to be a
    ``ComputeOpNode`` and pay for ``PrimFuncNode.__init__``, which would wire up
    ``PlaceHolderNode``s the op has no use for. Subclasses are either outside the
    dataflow entirely (a prologue scan) or an alternative producer of a tile that is
    already in it (a constant fill).
    """

    # Least destination size the body indexes. Asserted by the codegen, since a
    # destination smaller than this would be written past without complaint.
    min_size = 1

    def _format_code(self, code: str) -> str:
        return textwrap.dedent(code) + "\n"

    def get_macro_name(self, **kwargs) -> str:
        raise NotImplementedError

    def gen_macro(self, **kwargs) -> str:
        raise NotImplementedError


def const_token(value) -> str:
    """A macro-name fragment for a numeric literal, safe inside a Python identifier."""
    return str(value).replace(".", "_").replace("-", "minus_").replace("+", "plus_")


def mgroup_row(idx: str, group: int) -> str:
    """The token index inside a fused ``M = tokens * group`` row axis."""
    return idx if group == 1 else f"{idx} // {group}"


def mgroup_slot(idx: str, group: int) -> str:
    """The group slot inside a fused ``M = tokens * group`` row axis."""
    return "0" if group == 1 else f"{idx} % {group}"


def _is_repeat_broadcast_shape(
    src_shape: tuple[int, ...], dst_shape: tuple[int, ...]
) -> bool:
    if len(src_shape) != len(dst_shape):
        return False
    changed = False
    for src_dim, dst_dim in zip(src_shape, dst_shape):
        try:
            src_dim_i = int(src_dim)
            dst_dim_i = int(dst_dim)
        except (TypeError, ValueError):
            return False
        if src_dim_i == dst_dim_i:
            continue
        if src_dim_i <= 0 or dst_dim_i % src_dim_i != 0:
            return False
        changed = True
    return changed


def _is_expandable_broadcast_shape(
    src_shape: tuple[int, ...], dst_shape: tuple[int, ...]
) -> bool:
    if len(src_shape) >= len(dst_shape):
        return False

    src_dims = tuple(int(v) for v in src_shape)
    dst_dims = tuple(int(v) for v in dst_shape)

    def _search(src_idx: int, dst_idx: int, changed: bool) -> bool:
        if src_idx == len(src_dims):
            return changed and all(dim == 1 for dim in dst_dims[dst_idx:])
        if dst_idx == len(dst_dims):
            return False

        src_dim = src_dims[src_idx]
        dst_dim = dst_dims[dst_idx]
        if src_dim > 0 and dst_dim % src_dim == 0:
            if _search(src_idx + 1, dst_idx + 1, changed or src_dim != dst_dim):
                return True

        if dst_dim == 1 and _search(src_idx, dst_idx + 1, True):
            return True

        return False

    return _search(0, 0, False)


def wire_input(src_node, dst_node, dst_id: int, src_id: int = 0) -> None:
    if isinstance(src_node, PlaceHolderNode):
        if hasattr(dst_node, "input_is_batch_dim") and getattr(
            dst_node, "input_is_batch_dim"
        ):
            src_node.is_batch_dim = dst_node.input_is_batch_dim[dst_id]
        elif hasattr(dst_node, "is_batch_dim"):
            src_node.is_batch_dim = dst_node.is_batch_dim
        if hasattr(dst_node, "input_buffers") and dst_id < len(dst_node.input_buffers):
            buffer = dst_node.input_buffers[dst_id]
            has_shape = (
                src_id < len(src_node._shapes) and src_node._shapes[src_id] is not None
            )
            preserve_src_shape = False
            if has_shape:
                src_shape = tuple(src_node.get_shape(src_id))
                dst_shape = tuple(buffer.shape)
                preserve_src_shape = _is_repeat_broadcast_shape(
                    src_shape, dst_shape
                ) or _is_expandable_broadcast_shape(src_shape, dst_shape)
            if not preserve_src_shape:
                src_node.set_shape(buffer.shape, src_id, overwrite=True)
            if src_id >= len(src_node._dtypes) or src_node._dtypes[src_id] is None:
                src_node.set_dtype(tvm.DataType(buffer.dtype), src_id)

    old_edge = dst_node.inputs[dst_id]
    if old_edge in old_edge.src_node._out_edges:
        old_edge.src_node._out_edges.remove(old_edge)
    edge = Edge(src_node, dst_node, src_id, dst_id)
    src_node._out_edges.append(edge)
    dst_node.set_inputs(dst_id, edge)
