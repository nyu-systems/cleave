"""Searched structures -> Cleave graphs: ``[graph]`` solo, or a split structure cut into
``[partial, combine]`` (see :mod:`cleave.lower.split`)."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .convert import to_ir
from .split import (
    _CUT_CACHE_SIZE,
    Rejected,
    _augment_split_symbol_mapping,
    _contains_split_ops,
    _infer_symbolic_split_factor_from_graphs,
    _reduction_axis_from_op,
    _split_axis_from_op,
    _tensor_dims,
    cut_split_graph,
)


def _normalize_split_graph_layouts(graph_bundle):
    if len(graph_bundle) != 2:
        return graph_bundle

    def _normalize_split_tensor_shape(tensor: Dict) -> None:
        dims = _tensor_dims(tensor)
        if len(dims) <= 3:
            return
        position = len(dims) - 3
        normalized_dims = dims[1:]
        normalized_dims.insert(position, dims[0])
        dim_capacity = max(len(tensor.get("dim", [])), len(normalized_dims))
        tensor["dim"] = list(normalized_dims) + [0] * (dim_capacity - len(normalized_dims))
        tensor["num_dims"] = len(normalized_dims)

    normalized_bundle = deepcopy(graph_bundle)
    tensor_is_split: Dict[int, bool] = {}
    for op in normalized_bundle[0]:
        op_type = op.get("op_type", "")
        input_tensors = op.get("input_tensors", [])
        output_tensors = op.get("output_tensors", [])

        if op_type == "kn_input_op":
            for tensor in output_tensors:
                if tensor.get("guid") is not None:
                    tensor_is_split[tensor["guid"]] = False
            continue

        input_split = False
        for tensor in input_tensors:
            guid = tensor.get("guid")
            is_split = guid is not None and tensor_is_split.get(guid, False)
            input_split = input_split or is_split
            if is_split:
                _normalize_split_tensor_shape(tensor)

        if _split_axis_from_op(op_type) is not None:
            for tensor in output_tensors:
                _normalize_split_tensor_shape(tensor)
                if tensor.get("guid") is not None:
                    tensor_is_split[tensor["guid"]] = True
            continue

        output_split = input_split and _reduction_axis_from_op(op_type) != 0
        if output_split:
            for tensor in output_tensors:
                _normalize_split_tensor_shape(tensor)
        for tensor in output_tensors:
            if tensor.get("guid") is not None:
                tensor_is_split[tensor["guid"]] = output_split

    return normalized_bundle


def _align_split_boundary_inputs(graph_bundle):
    if len(graph_bundle) != 2:
        return graph_bundle

    symbolic_first_graph = to_ir(graph_bundle[0])
    boundary_shapes = [
        tuple(node.get_shape()) for node in symbolic_first_graph.nodes if node.is_output()
    ]
    combine_graph = deepcopy(graph_bundle[1])
    input_ops = [op for op in combine_graph if op.get("op_type") == "kn_input_op"]
    assert len(boundary_shapes) == len(input_ops), (
        "Split boundary mismatch: found %d symbolic outputs but %d combine inputs"
        % (len(boundary_shapes), len(input_ops))
    )

    split_axis_by_guid: Dict[int, int] = {}
    for op, shape in zip(input_ops, boundary_shapes):
        output_tensors = op.get("output_tensors", [])
        assert len(output_tensors) == 1, (
            "Expected one output tensor for combine input op, got %d" % len(output_tensors)
        )
        tensor = output_tensors[0]
        raw_shape = _tensor_dims(tensor)
        dim_capacity = max(len(tensor.get("dim", [])), len(shape))
        tensor["dim"] = list(shape) + [0] * (dim_capacity - len(shape))
        tensor["num_dims"] = len(shape)
        guid = tensor.get("guid")
        if guid is None:
            continue
        for split_pos in range(len(raw_shape)):
            candidate_shape = list(raw_shape[1:])
            candidate_shape.insert(split_pos, raw_shape[0])
            if tuple(candidate_shape) == tuple(shape) or (
                len(candidate_shape) == len(shape) + 1
                and candidate_shape[-1] == 1
                and tuple(candidate_shape[:-1]) == tuple(shape)
            ):
                split_axis_by_guid[guid] = split_pos
                break

    for op in combine_graph:
        input_split_axis = next(
            (split_axis_by_guid[t["guid"]] for t in op.get("input_tensors", [])
             if t.get("guid") in split_axis_by_guid),
            None,
        )
        if input_split_axis is None:
            continue

        reduction_axis = _reduction_axis_from_op(op.get("op_type", ""))
        output_split_axis = input_split_axis
        if reduction_axis is not None:
            if reduction_axis == 0 and input_split_axis != 0:
                op["op_type"] = f"kn_reduce_sum_{input_split_axis}_op"
                reduction_axis = input_split_axis
            if reduction_axis == input_split_axis:
                output_split_axis = None
            elif reduction_axis < input_split_axis:
                output_split_axis = input_split_axis - 1

        for tensor in op.get("output_tensors", []):
            guid = tensor.get("guid")
            if guid is None:
                continue
            if output_split_axis is None:
                split_axis_by_guid.pop(guid, None)
            else:
                split_axis_by_guid[guid] = output_split_axis

    return [graph_bundle[0], combine_graph]


def _freeze_structure(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_structure(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_structure(item) for item in value)
    return value


@dataclass
class _Lowerable:
    """One searched structure ready to convert: ``[graph]`` or ``[partial, combine]``."""

    raw: List[List[Dict]]
    converted: List[List[Dict]]  # split layouts normalized, combine inputs aligned
    symbolic_split_factor: Optional[int]  # as reported by the cut, else inferred


_PREPARED: "OrderedDict[Tuple, _Lowerable]" = OrderedDict()


def _prepare(raw: List[List[Dict]], symbolic_split_factor: Optional[int], key) -> _Lowerable:
    hit = _PREPARED.get(key)
    if hit is not None:
        _PREPARED.move_to_end(key)
        return hit
    raw = deepcopy(raw)  # cached: must not alias the caller's structures
    converted = _align_split_boundary_inputs(_normalize_split_graph_layouts(raw))
    inferred = symbolic_split_factor
    if inferred is None:
        inferred = _infer_symbolic_split_factor_from_graphs(raw)
    prepared = _Lowerable(raw, converted, inferred)
    _PREPARED[key] = prepared
    if len(_PREPARED) > _CUT_CACHE_SIZE:
        _PREPARED.popitem(last=False)
    return prepared


def lowerable(graph_structures: list) -> Tuple[List[_Lowerable], List[Tuple[int, Rejected]]]:
    """Deduplicated, conversion-ready structures plus ``(index, Rejected)`` per refused one."""
    out, rejected, seen = [], [], set()
    for i, gs in enumerate(graph_structures):
        if _contains_split_ops(gs):
            result = cut_split_graph(gs)
            if isinstance(result, Rejected):
                rejected.append((i, result))
                continue
            bundle, symbolic = result
        else:
            bundle, symbolic = [gs], None
        key = (_freeze_structure(bundle), symbolic)
        if key in seen:
            continue
        seen.add(key)
        out.append(_prepare(bundle, symbolic, key))
    return out, rejected


def to_bundles(
    graph_structures: list,
    symbol_map: dict = None,
    split_factors: list = None,
    *,
    dtype: str = "float16",
    accum_dtype: str = "float32",
    out_dtype: str = "float16",
    expand_broadcast_inputs: bool = False,
):
    """Cleave graphs for each searched structure: ``[graph]`` or ``[partial, combine]``.

    ``dtype``/``accum_dtype``/``out_dtype`` are handed to ``to_ir``.

    ``expand_broadcast_inputs`` widens a split partial's broadcast input (rank-2 Q against
    rank-3 split K) to its operand buffer's rank, so ``apply_tiling`` doesn't fail on the
    misaligned axis. Opt-in: it changes the dense kernel signature (Q becomes ``(1, M, D)``).
    """
    structures, rejected = lowerable(graph_structures)
    for i, r in rejected:
        print(f"Invalid split graph (no. {i}): {r.reason}, skipping")

    cleave_graphs = []
    for s in structures:
        factors = [None]
        if s.symbolic_split_factor is not None and split_factors:
            factors = [int(v) for v in split_factors]
        for split_factor in factors:
            local_symbol_map = _augment_split_symbol_mapping(
                s.raw,
                None if symbol_map is None else dict(symbol_map),
                symbolic_split_factor=s.symbolic_split_factor,
                split_factor=split_factor,
            )
            # Split-combine graphs keep the explicit sqrt + reciprocal-div chain so their
            # Cleave form matches the reference combine graph.
            fuse = [True] if len(s.converted) == 1 else [True, False]
            bundle = [
                to_ir(gs, symbol_map=local_symbol_map, fuse_rsqrt=f, dtype=dtype,
                      accum_dtype=accum_dtype, out_dtype=out_dtype)
                for gs, f in zip(s.converted, fuse)
            ]
            if expand_broadcast_inputs and len(bundle) == 2:
                _expand_broadcast_inputs(bundle[0])
            cleave_graphs.append(bundle)
    return cleave_graphs


def _expand_broadcast_inputs(graph) -> None:
    """Give an input placeholder the rank of the operand buffer it broadcasts into.

    Only a placeholder whose batch flags outnumber its dims is touched -- that is the
    inconsistent state ``wire_input`` leaves when it preserves a lower-rank input feeding
    a batched operand -- and it takes exactly the consuming operand's buffer shape.
    """
    from ..ir.graph import PlaceHolderNode

    for node in graph.nodes:
        if not isinstance(node, PlaceHolderNode) or not node.outputs:
            continue
        flags = getattr(node, "is_batch_dim", None)
        shape = tuple(node.get_shape(0))
        if not isinstance(flags, (list, tuple)) or len(flags) <= len(shape):
            continue
        edge = node.outputs[0]
        buffers = getattr(edge.dst_node, "input_buffers", None)
        if buffers is None or edge.dst_id >= len(buffers):
            continue
        target = tuple(int(x) for x in buffers[edge.dst_id].shape)
        if len(target) != len(flags):
            continue
        node.set_shape(target, overwrite=True)
