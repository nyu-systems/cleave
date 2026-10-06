from copy import deepcopy
from dataclasses import dataclass
from itertools import zip_longest
from typing import Dict, List, Optional, Tuple

from .superoptimizer import graph_structure_to_cleave


_UNARY_PULLBACK_OPS = {
    "kn_exp_op",
    "kn_silu_op",
    "kn_sqrt_op",
}

_ELEMENTWISE_OPS = {
    "kn_add_op",
    "kn_div_op",
    "kn_mul_op",
    "kn_sub_op",
}


def _split_axis_from_op(op_type: str) -> Optional[int]:
    if op_type.startswith("kn_split_") and op_type.endswith("_op"):
        return int(op_type.split("_")[2])
    return None


def _reduction_axis_from_op(op_type: str) -> Optional[int]:
    if op_type.startswith("kn_reduction_") and op_type.endswith("_op"):
        return int(op_type.split("_")[2])
    return None


def _tensor_dims(tensor: Dict) -> List[int]:
    dims = [int(v) for v in tensor.get("dim", [])]
    num_dims = tensor.get("num_dims")
    if isinstance(num_dims, int) and num_dims > 0:
        dims = dims[:num_dims]
    return dims


def _raw_tensor_dims(tensor: Dict) -> List[int]:
    return [int(v) for v in tensor.get("dim", [])]


def _resolve_dim(dim: int, symbol_map: Optional[Dict[int, int]] = None) -> int:
    if symbol_map and dim in symbol_map:
        return symbol_map[dim]
    return dim


def _contains_split_ops(graph_structure) -> bool:
    if not isinstance(graph_structure, list):
        return False
    return any(
        _split_axis_from_op(op.get("op_type", "")) is not None for op in graph_structure
    )


def _infer_split_factor(op: Dict) -> Optional[int]:
    output_tensors = op.get("output_tensors", [])
    if len(output_tensors) != 1:
        return None
    output_dims = _tensor_dims(output_tensors[0])
    if not output_dims:
        return None
    factor = int(output_dims[0])
    if factor <= 1:
        return None
    return factor


def _infer_symbolic_split_factor_from_graphs(graph_structures) -> Optional[int]:
    split_factors = set()

    for graph_structure in graph_structures:
        if not isinstance(graph_structure, list):
            continue

        for op in graph_structure:
            if _split_axis_from_op(op.get("op_type", "")) is None:
                continue

            split_factor = _infer_split_factor(op)
            if split_factor is None:
                continue
            split_factors.add(split_factor)

    if not split_factors:
        return None
    if len(split_factors) != 1:
        raise ValueError(
            "Inconsistent symbolic split factors found across split graphs: "
            f"{sorted(split_factors)}"
        )
    return next(iter(split_factors))


def _is_scalar_like_unary(op: Dict) -> bool:
    op_type = op.get("op_type", "")
    input_tensors = op.get("input_tensors", [])
    if op_type in _UNARY_PULLBACK_OPS:
        return len(input_tensors) == 1
    if op_type == "kn_add_op":
        return len(input_tensors) == 1
    if op_type == "kn_div_op":
        return len(input_tensors) == 1 and op.get("scalar") is not None
    return False


def _looks_like_classic_split_k(graph_structure, split_factor: int) -> bool:
    if not isinstance(split_factor, int) or split_factor <= 1:
        return False

    non_trivial_ops = [
        op
        for op in graph_structure
        if op.get("op_type") not in {"kn_input_op", "kn_output_op"}
        and _split_axis_from_op(op.get("op_type", "")) is None
    ]
    if len(non_trivial_ops) != 2:
        return False
    if sum(op.get("op_type") == "kn_matmul_op" for op in non_trivial_ops) != 1:
        return False
    if (
        sum(
            _reduction_axis_from_op(op.get("op_type", "")) == 0
            for op in non_trivial_ops
        )
        != 1
    ):
        return False

    producers = {}
    for op in graph_structure:
        for tensor in op.get("output_tensors", []):
            guid = tensor.get("guid")
            if guid is not None:
                producers[guid] = op

    for reduction_op in graph_structure:
        reduction_axis = _reduction_axis_from_op(reduction_op.get("op_type", ""))
        if reduction_axis is None:
            continue

        input_tensors = reduction_op.get("input_tensors", [])
        if len(input_tensors) != 1:
            continue

        reduction_input_dims = _tensor_dims(input_tensors[0])
        if len(reduction_input_dims) < 3:
            continue
        if reduction_axis != 0 or reduction_input_dims[0] != split_factor:
            continue

        matmul_op = producers.get(input_tensors[0].get("guid"))
        if matmul_op is None or matmul_op.get("op_type") != "kn_matmul_op":
            continue

        matmul_inputs = matmul_op.get("input_tensors", [])
        if len(matmul_inputs) != 2:
            continue

        all_inputs_are_split = True
        for matmul_input in matmul_inputs:
            split_op = producers.get(matmul_input.get("guid"))
            split_axis = (
                None
                if split_op is None
                else _split_axis_from_op(split_op.get("op_type", ""))
            )
            if split_axis is None:
                all_inputs_are_split = False
                break

            split_input_tensors = split_op.get("input_tensors", [])
            split_output_tensors = split_op.get("output_tensors", [])
            if len(split_input_tensors) != 1 or len(split_output_tensors) != 1:
                all_inputs_are_split = False
                break

            split_input_dims = _tensor_dims(split_input_tensors[0])
            split_output_dims = _tensor_dims(split_output_tensors[0])
            if not (0 <= split_axis < len(split_input_dims)):
                all_inputs_are_split = False
                break
            if split_input_dims[split_axis] % split_factor != 0:
                all_inputs_are_split = False
                break

            expected_output_dims = [split_factor]
            expected_output_dims.extend(
                (
                    split_input_dims[i] // split_factor
                    if i == split_axis
                    else split_input_dims[i]
                )
                for i in range(len(split_input_dims))
            )
            if expected_output_dims != split_output_dims:
                all_inputs_are_split = False
                break

        if all_inputs_are_split:
            return True

    return False


def _augment_split_symbol_mapping(
    graph_structures,
    symbol_map: Optional[Dict[int, int]],
    symbolic_split_factor: Optional[int],
    split_factor: Optional[int],
) -> Optional[Dict[int, int]]:
    if symbol_map is None:
        return symbol_map

    if symbolic_split_factor is None:
        symbolic_split_factor = _infer_symbolic_split_factor_from_graphs(
            graph_structures
        )
    if symbolic_split_factor is None:
        return symbol_map

    if split_factor is None:
        split_factor = int(symbol_map.get(symbolic_split_factor, symbolic_split_factor))

    mapped_split_factor = symbol_map.get(symbolic_split_factor)
    if mapped_split_factor is not None and mapped_split_factor != split_factor:
        raise ValueError(
            "Conflicting split-factor mapping for symbolic dimension "
            f"{symbolic_split_factor}: symbol_map has {mapped_split_factor}, "
            f"requested split_factor is {split_factor}"
        )
    symbol_map[symbolic_split_factor] = split_factor

    for graph_structure in graph_structures:
        for op in graph_structure:
            split_axis = _split_axis_from_op(op.get("op_type", ""))
            if split_axis is None:
                continue

            input_tensors = op.get("input_tensors", [])
            output_tensors = op.get("output_tensors", [])
            if len(input_tensors) != 1 or len(output_tensors) != 1:
                continue

            raw_input_dims = _tensor_dims(input_tensors[0])
            raw_output_dims = _tensor_dims(output_tensors[0])
            if len(raw_output_dims) != len(raw_input_dims) + 1:
                continue

            split_position = 0
            if not (0 <= split_axis < len(raw_input_dims)):
                continue
            if split_position < 0 or split_position >= len(raw_output_dims):
                continue

            resolved_input_dims = [_resolve_dim(v, symbol_map) for v in raw_input_dims]
            if resolved_input_dims[split_axis] % split_factor != 0:
                raise ValueError(
                    f"Split factor {split_factor} does not divide resolved input extent "
                    f"{resolved_input_dims[split_axis]} for split axis {split_axis}"
                )

            expected_inner_extent = resolved_input_dims[split_axis] // split_factor
            raw_inner_symbol = raw_output_dims[split_axis + 1]
            resolved_inner_extent = _resolve_dim(raw_inner_symbol, symbol_map)
            if resolved_inner_extent == expected_inner_extent:
                continue

            if raw_inner_symbol in symbol_map:
                raise ValueError(
                    "Conflicting split-derived mapping for symbolic dimension "
                    f"{raw_inner_symbol}: symbol_map has {symbol_map[raw_inner_symbol]}, "
                    f"expected {expected_inner_extent} for split_factor {split_factor}"
                )

            symbol_map[raw_inner_symbol] = expected_inner_extent

    return symbol_map


@dataclass
class _TensorState:
    guid: int
    shape: Tuple[int, ...]
    tokens: Tuple[Tuple[str, int, int], ...]
    template: Dict
    can_split_flags: Tuple[bool, ...]
    can_reduce_flags: Tuple[bool, ...]
    is_split: bool


@dataclass
class _CanonicalOp:
    op_type: str
    input_tensors: List[_TensorState]
    output_tensors: List[_TensorState]
    template_op: Optional[Dict]


def split_graph(graph_structure):
    def _renumber_graph(ops):
        guid_map = {}
        next_guid = 10000001
        renumbered_ops = []

        def _renumber_tensor(tensor):
            nonlocal next_guid

            new_tensor = deepcopy(tensor)
            guid = new_tensor.get("guid")
            if guid is not None:
                if guid not in guid_map:
                    guid_map[guid] = next_guid
                    next_guid += 1
                new_tensor["guid"] = guid_map[guid]
            return new_tensor

        for op in ops:
            new_op = deepcopy(op)
            new_op["input_tensors"] = [
                _renumber_tensor(tensor) for tensor in op.get("input_tensors", [])
            ]
            new_op["output_tensors"] = [
                _renumber_tensor(tensor) for tensor in op.get("output_tensors", [])
            ]
            renumbered_ops.append(new_op)

        return renumbered_ops

    if not _contains_split_ops(graph_structure):
        return None

    producers: Dict[int, int] = {}
    consumers: Dict[int, List[Tuple[int, int]]] = {}
    for op_idx, op in enumerate(graph_structure):
        for tensor in op.get("output_tensors", []):
            guid = tensor.get("guid")
            if guid is not None:
                producers[guid] = op_idx
        for input_idx, tensor in enumerate(op.get("input_tensors", [])):
            guid = tensor.get("guid")
            if guid is not None:
                consumers.setdefault(guid, []).append((op_idx, input_idx))

    split_factor = None
    split_output_to_source_guid: Dict[int, int] = {}
    seed_axes: Dict[Tuple[int, int], Dict] = {}
    split_axis_by_tensor_guid: Dict[int, int] = {}

    def _map_matmul_output_axis_to_input(
        op: Dict, axis: int
    ) -> Optional[Tuple[int, int]]:
        input_tensors = op.get("input_tensors", [])
        output_tensors = op.get("output_tensors", [])
        if len(input_tensors) != 2 or len(output_tensors) != 1:
            return None
        output_rank = len(_tensor_dims(output_tensors[0]))
        if axis < 0 or axis >= output_rank:
            return None
        if axis < output_rank - 2:
            return None

        transpose_a = bool(op.get("trans_A", False))
        transpose_b = bool(op.get("trans_B", False))
        if axis == output_rank - 2:
            a_rank = len(_tensor_dims(input_tensors[0]))
            mapped_axis = a_rank - 1 if transpose_a else a_rank - 2
            return input_tensors[0]["guid"], mapped_axis
        if axis == output_rank - 1:
            b_rank = len(_tensor_dims(input_tensors[1]))
            mapped_axis = b_rank - 2 if transpose_b else b_rank - 1
            return input_tensors[1]["guid"], mapped_axis
        return None

    for op in graph_structure:
        split_axis = _split_axis_from_op(op.get("op_type", ""))
        if split_axis is None:
            continue

        input_tensors = op.get("input_tensors", [])
        output_tensors = op.get("output_tensors", [])
        if len(input_tensors) != 1 or len(output_tensors) != 1:
            return None

        factor = _infer_split_factor(op)
        if factor is None:
            return None
        if split_factor is None:
            split_factor = factor
        elif split_factor != factor:
            return None

        split_output_to_source_guid[output_tensors[0]["guid"]] = input_tensors[0][
            "guid"
        ]

        current_guid = input_tensors[0].get("guid")
        current_axis = split_axis
        while True:
            existing_axis = split_axis_by_tensor_guid.get(current_guid)
            if existing_axis is not None and existing_axis != current_axis:
                return None
            split_axis_by_tensor_guid[current_guid] = current_axis

            producer_idx = producers.get(current_guid)
            if producer_idx is None:
                return None
            producer_op = graph_structure[producer_idx]
            producer_type = producer_op.get("op_type", "")

            if producer_type == "kn_input_op":
                seed_key = (current_guid, current_axis)
                seed_axes.setdefault(
                    seed_key,
                    {
                        "source_tensor": producer_op.get("output_tensors", [])[0],
                        "axis": current_axis,
                    },
                )
                break

            if len(consumers.get(current_guid, [])) != 1:
                return None

            if _is_scalar_like_unary(producer_op):
                producer_inputs = producer_op.get("input_tensors", [])
                if len(producer_inputs) != 1:
                    return None
                current_guid = producer_inputs[0]["guid"]
                continue

            if producer_type == "kn_matmul_op":
                mapped = _map_matmul_output_axis_to_input(producer_op, current_axis)
                if mapped is None:
                    return None
                current_guid, current_axis = mapped
                continue

            return None

    if split_factor is None or not seed_axes:
        return None

    seed_order = sorted(
        seed_axes, key=lambda item: (producers.get(item[0], -1), item[1])
    )

    next_tensor_guid = 1

    def _new_tensor_state(
        shape: Tuple[int, ...],
        tokens: Tuple[Tuple[str, int, int], ...],
        template: Dict,
        can_split_flags: Optional[Tuple[bool, ...]] = None,
        can_reduce_flags: Optional[Tuple[bool, ...]] = None,
        is_split: bool = False,
    ) -> _TensorState:
        nonlocal next_tensor_guid
        state = _TensorState(
            guid=10_000_000 + next_tensor_guid,
            shape=tuple(int(v) for v in shape),
            tokens=tuple(tokens),
            template=deepcopy(template),
            can_split_flags=tuple(can_split_flags or (False,) * len(shape)),
            can_reduce_flags=tuple(can_reduce_flags or (False,) * len(shape)),
            is_split=is_split,
        )
        next_tensor_guid += 1
        return state

    def _input_tokens(tensor: Dict) -> Tuple[Tuple[str, int, int], ...]:
        guid = tensor.get("guid")
        return tuple(("orig", guid, axis) for axis in range(len(_tensor_dims(tensor))))

    def _template_reduce_flags(
        tensor: Dict, rank: Optional[int] = None
    ) -> Tuple[bool, ...]:
        shape_rank = len(_tensor_dims(tensor)) if rank is None else rank
        return tuple(bool(v) for v in tensor.get("can_reduce", [])[:shape_rank])

    def _split_state(src_state: _TensorState, axis: int) -> _TensorState:
        if axis < 0 or axis >= len(src_state.shape):
            raise ValueError(f"Invalid split axis {axis} for shape {src_state.shape}")
        if src_state.shape[axis] % split_factor != 0:
            raise ValueError(
                f"Split factor {split_factor} does not divide shape {src_state.shape} on axis {axis}"
            )

        new_shape = [split_factor]
        new_shape.extend(
            src_state.shape[i] // split_factor if i == axis else src_state.shape[i]
            for i in range(len(src_state.shape))
        )
        split_token = ("split", src_state.template.get("guid", src_state.guid), axis)
        new_tokens = (split_token,) + src_state.tokens
        return _new_tensor_state(
            tuple(new_shape),
            new_tokens,
            src_state.template,
            can_split_flags=tuple(False for _ in new_shape),
            can_reduce_flags=(True,) + src_state.can_reduce_flags,
            is_split=True,
        )

    def _split_transformed_shape(shape: Tuple[int, ...], axis: int) -> Tuple[int, ...]:
        if axis < 0 or axis >= len(shape):
            raise ValueError(f"Invalid split axis {axis} for shape {shape}")
        if shape[axis] % split_factor != 0:
            raise ValueError(
                f"Split factor {split_factor} does not divide shape {shape} on axis {axis}"
            )
        return tuple(
            [split_factor]
            + [
                shape[i] // split_factor if i == axis else shape[i]
                for i in range(len(shape))
            ]
        )

    def _build_output_state(
        output_tensor: Dict,
        *,
        split_axis: Optional[int] = None,
        split_token: Optional[Tuple[str, int, int]] = None,
        already_split: bool = False,
    ) -> _TensorState:
        base_shape = tuple(_tensor_dims(output_tensor))
        base_flags = _template_reduce_flags(output_tensor, len(base_shape))
        output_guid = output_tensor.get("guid", -1)

        if split_axis is not None:
            out_shape = _split_transformed_shape(base_shape, split_axis)
            can_reduce_flags = (True,) + base_flags
            first_token = split_token or ("split", output_guid, split_axis)
            out_tokens = (first_token,) + tuple(
                ("op", output_guid, axis) for axis in range(len(base_shape))
            )
            return _new_tensor_state(
                out_shape,
                out_tokens,
                output_tensor,
                can_split_flags=tuple(False for _ in out_shape),
                can_reduce_flags=can_reduce_flags,
                is_split=True,
            )

        if already_split:
            out_shape = base_shape
            first_token = split_token or ("split", output_guid, 0)
            out_tokens = (
                (first_token,)
                + tuple(("op", output_guid, axis) for axis in range(1, len(base_shape)))
                if base_shape
                else ()
            )
            return _new_tensor_state(
                out_shape,
                out_tokens,
                output_tensor,
                can_split_flags=tuple(False for _ in out_shape),
                can_reduce_flags=base_flags,
                is_split=True,
            )

        return _new_tensor_state(
            base_shape,
            tuple(("op", output_guid, axis) for axis in range(len(base_shape))),
            output_tensor,
            can_split_flags=tuple(False for _ in base_shape),
            can_reduce_flags=base_flags,
            is_split=False,
        )

    split_input_ops: List[_CanonicalOp] = []
    split_seed_ops: List[_CanonicalOp] = []
    split_graph_ops: List[_CanonicalOp] = []
    combine_input_ops: List[_CanonicalOp] = []
    combine_graph_ops: List[_CanonicalOp] = []
    base_tensors: Dict[int, _TensorState] = {}
    split_tensors: Dict[int, _TensorState] = {}
    combine_tensors: Dict[int, _TensorState] = {}
    boundary_inputs: Dict[int, _TensorState] = {}
    combine_started = False

    def _get_boundary_input(split_state: _TensorState) -> _TensorState:
        existing = boundary_inputs.get(split_state.guid)
        if existing is not None:
            return existing

        split_graph_ops.append(
            _CanonicalOp(
                op_type="kn_output_op",
                input_tensors=[split_state],
                output_tensors=[],
                template_op=None,
            )
        )
        boundary_state = _new_tensor_state(
            split_state.shape,
            split_state.tokens,
            split_state.template,
            can_split_flags=tuple(False for _ in split_state.shape),
            can_reduce_flags=split_state.can_reduce_flags,
            is_split=True,
        )
        combine_input_ops.append(
            _CanonicalOp(
                op_type="kn_input_op",
                input_tensors=[],
                output_tensors=[boundary_state],
                template_op=None,
            )
        )
        boundary_inputs[split_state.guid] = boundary_state
        return boundary_state

    seed_axis_by_guid: Dict[int, int] = {}
    for seed_guid, seed_axis in seed_order:
        if seed_guid in seed_axis_by_guid and seed_axis_by_guid[seed_guid] != seed_axis:
            return None
        seed_axis_by_guid[seed_guid] = seed_axis

    for op in graph_structure:
        if op.get("op_type") != "kn_input_op":
            continue
        output_tensors = op.get("output_tensors", [])
        if len(output_tensors) != 1:
            return None

        output_tensor = output_tensors[0]
        shape = tuple(_tensor_dims(output_tensor))
        base_state = _new_tensor_state(
            shape,
            _input_tokens(output_tensor),
            output_tensor,
            can_split_flags=tuple(
                bool(v) for v in output_tensor.get("can_split", [])[: len(shape)]
            ),
            can_reduce_flags=_template_reduce_flags(output_tensor, len(shape)),
            is_split=False,
        )
        tensor_guid = output_tensor.get("guid")
        base_tensors[tensor_guid] = base_state
        split_input_ops.append(
            _CanonicalOp(
                op_type="kn_input_op",
                input_tensors=[],
                output_tensors=[base_state],
                template_op=op,
            )
        )

        if tensor_guid in seed_axis_by_guid:
            split_state = _split_state(base_state, seed_axis_by_guid[tensor_guid])
            split_tensors[tensor_guid] = split_state
            split_seed_ops.append(
                _CanonicalOp(
                    op_type=f"kn_split_{seed_axis_by_guid[tensor_guid]}_op",
                    input_tensors=[base_state],
                    output_tensors=[split_state],
                    template_op=None,
                )
            )

    if not split_tensors:
        return None

    def _resolve_input_state(input_tensor: Dict) -> Tuple[Optional[_TensorState], str]:
        guid = input_tensor.get("guid")
        if guid in combine_tensors:
            return combine_tensors[guid], "combine"
        if guid in split_output_to_source_guid:
            source_guid = split_output_to_source_guid[guid]
            return split_tensors.get(source_guid), "split"
        if guid in split_tensors:
            return split_tensors[guid], "split"
        if guid in base_tensors:
            return base_tensors[guid], "base"
        return None, "invalid"

    def _emit_op(
        op_type: str,
        input_states: List[_TensorState],
        output_states: List[_TensorState],
        template_op: Optional[Dict],
        graph_id: int,
    ) -> None:
        op_state = _CanonicalOp(
            op_type=op_type,
            input_tensors=input_states,
            output_tensors=output_states,
            template_op=template_op,
        )
        if graph_id == 0:
            split_graph_ops.append(op_state)
        else:
            combine_graph_ops.append(op_state)

    for op in graph_structure:
        op_type = op.get("op_type", "")
        if op_type in {"kn_input_op", "kn_output_op"}:
            continue
        if _split_axis_from_op(op_type) is not None:
            continue

        resolved_inputs: List[Optional[_TensorState]] = []
        input_kinds: List[str] = []
        for tensor in op.get("input_tensors", []):
            state, kind = _resolve_input_state(tensor)
            resolved_inputs.append(state)
            input_kinds.append(kind)

        has_combine_input = any(kind == "combine" for kind in input_kinds)
        has_split_input = any(kind == "split" for kind in input_kinds)
        if not has_combine_input and not has_split_input:
            continue
        if any(state is None for state in resolved_inputs):
            return None
        if has_combine_input and has_split_input:
            return None

        graph_id = 1 if has_combine_input else 0
        if graph_id == 1 and any(kind == "base" for kind in input_kinds):
            return None

        cast_inputs = [state for state in resolved_inputs if state is not None]
        reduction_axis = _reduction_axis_from_op(op_type)
        if combine_started and graph_id == 0 and reduction_axis != 0:
            return None
        split_inputs = [state for state in cast_inputs if state.is_split]
        split_token = split_inputs[0].tokens[0] if split_inputs else None
        output_tensor = op.get("output_tensors", [{}])[0]
        output_guid = output_tensor.get("guid")
        logical_split_axis = split_axis_by_tensor_guid.get(output_guid)
        already_split_output = (
            graph_id == 0 and bool(split_inputs) and logical_split_axis is None
        )

        if op_type == "kn_matmul_op":
            if len(cast_inputs) != 2:
                return None
            output_state = _build_output_state(
                output_tensor,
                split_axis=logical_split_axis,
                split_token=split_token,
                already_split=already_split_output,
            )
            if graph_id == 0:
                if not output_state.is_split:
                    return None
                split_tensors[output_guid] = output_state
            else:
                combine_tensors[output_guid] = output_state
            _emit_op(op_type, cast_inputs, [output_state], op, graph_id)
            continue

        if reduction_axis is not None:
            if len(cast_inputs) != 1:
                return None
            input_state = cast_inputs[0]
            if graph_id == 0 and reduction_axis == 0:
                combine_started = True
                boundary_input = _get_boundary_input(input_state)
                output_state = _build_output_state(output_tensor)
                combine_tensors[output_guid] = output_state
                _emit_op(op_type, [boundary_input], [output_state], op, 1)
                continue

            output_state = _build_output_state(
                output_tensor,
                split_axis=logical_split_axis,
                split_token=split_token,
                already_split=already_split_output,
            )
            if graph_id == 0:
                if not output_state.is_split:
                    return None
                split_tensors[output_guid] = output_state
            else:
                combine_tensors[output_guid] = output_state
            _emit_op(op_type, [input_state], [output_state], op, graph_id)
            continue

        if op_type in _ELEMENTWISE_OPS or _is_scalar_like_unary(op):
            output_state = _build_output_state(
                output_tensor,
                split_axis=logical_split_axis,
                split_token=split_token,
                already_split=already_split_output,
            )
            if graph_id == 0:
                if not output_state.is_split:
                    return None
                split_tensors[output_guid] = output_state
            else:
                combine_tensors[output_guid] = output_state
            _emit_op(op_type, cast_inputs, [output_state], op, graph_id)
            continue

        return None

    if not combine_input_ops or not combine_graph_ops:
        return None

    for op in graph_structure:
        if op.get("op_type") != "kn_output_op":
            continue
        input_tensors = op.get("input_tensors", [])
        if len(input_tensors) != 1:
            return None
        output_state, kind = _resolve_input_state(input_tensors[0])
        if output_state is None or kind != "combine":
            return None
        combine_graph_ops.append(
            _CanonicalOp(
                op_type="kn_output_op",
                input_tensors=[output_state],
                output_tensors=[],
                template_op=op,
            )
        )

    if not any(op.op_type == "kn_output_op" for op in combine_graph_ops):
        return None

    def _serialize_tensor(tensor_state: _TensorState) -> Dict:
        template = tensor_state.template or {}
        serialized = deepcopy(template)
        raw_dims = _raw_tensor_dims(template)
        dim_capacity = max(len(raw_dims), len(tensor_state.shape))
        serialized["dim"] = list(tensor_state.shape) + [0] * (
            dim_capacity - len(tensor_state.shape)
        )
        serialized["guid"] = tensor_state.guid
        serialized["num_dims"] = len(tensor_state.shape)

        if "can_reduce" in template:
            flags_len = max(len(template.get("can_reduce", [])), dim_capacity)
            can_reduce = [False] * flags_len
            for axis, flag in enumerate(tensor_state.can_reduce_flags):
                if axis >= flags_len:
                    break
                can_reduce[axis] = bool(flag)
            serialized["can_reduce"] = can_reduce

        if "can_split" in template:
            flags_len = max(len(template.get("can_split", [])), dim_capacity)
            can_split = [False] * flags_len
            for axis, flag in enumerate(tensor_state.can_split_flags):
                if axis >= flags_len:
                    break
                can_split[axis] = bool(flag)
            serialized["can_split"] = can_split

        return serialized

    def _serialize_op(op: _CanonicalOp) -> Dict:
        if op.template_op is None:
            return {
                "op_type": op.op_type,
                "input_tensors": [
                    _serialize_tensor(tensor) for tensor in op.input_tensors
                ],
                "output_tensors": [
                    _serialize_tensor(tensor) for tensor in op.output_tensors
                ],
            }

        serialized = deepcopy(op.template_op)
        serialized["op_type"] = op.op_type
        serialized["input_tensors"] = [
            _serialize_tensor(tensor) for tensor in op.input_tensors
        ]
        serialized["output_tensors"] = [
            _serialize_tensor(tensor) for tensor in op.output_tensors
        ]
        return serialized

    first_graph = [
        _serialize_op(op) for op in split_input_ops + split_seed_ops + split_graph_ops
    ]
    second_graph = [_serialize_op(op) for op in combine_input_ops + combine_graph_ops]

    symbolic_split_factor = None
    if any(
        len(_tensor_dims(op.get("output_tensors", [{}])[0])) >= 4
        for op in graph_structure
        if _split_axis_from_op(op.get("op_type", "")) is not None
    ) or _looks_like_classic_split_k(graph_structure, split_factor):
        symbolic_split_factor = split_factor

    return [
        _renumber_graph(first_graph),
        _renumber_graph(second_graph),
    ], symbolic_split_factor


def is_split_graph(graph_structure):
    if not isinstance(graph_structure, list) or not graph_structure:
        return False
    if any(isinstance(op, list) for op in graph_structure):
        return False
    if not _contains_split_ops(graph_structure):
        return False
    return split_graph(graph_structure) is not None


def process_graphs(
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

    ``dtype``/``accum_dtype``/``out_dtype`` are handed to ``graph_structure_to_cleave`` so a
    caller working in another precision does not have to rebuild the conversion.

    ``expand_broadcast_inputs`` widens a split partial's input placeholders to the rank of
    the operand buffer they feed when the two disagree. The integration keeps an input at
    its declared rank when it broadcasts into a batched matmul (a rank-2 Q against a
    rank-3 split K), but hands it the matmul's rank-3 batch flags; the tiling policy then
    tiles the rank-3 flags and maps them onto the rank-2 shape, and ``apply_tiling`` fails
    on the misaligned axis ("Expected exactly one launch-axis mapping for iter dim 0").
    With the flag on, the input takes the buffer's shape -- a leading unit extent -- and
    the same body tiles. Opt-in because it changes the emitted kernel's signature for the
    dense pipeline (Q becomes ``(1, M, D)``); the paged access layer, which rebuilds its
    inputs from the operator's spec, is indifferent.
    """
    def _normalize_split_graph_layouts(graph_bundle):
        if len(graph_bundle) != 2:
            return graph_bundle

        def _normalize_split_tensor_shape(tensor: Dict) -> None:
            dims = _tensor_dims(tensor)
            if len(dims) <= 3:
                return

            position = len(dims) - 3
            if position <= 0:
                return

            normalized_dims = dims[1:]
            normalized_dims.insert(position, dims[0])
            dim_capacity = max(len(tensor.get("dim", [])), len(normalized_dims))
            tensor["dim"] = list(normalized_dims) + [0] * (
                dim_capacity - len(normalized_dims)
            )
            tensor["num_dims"] = len(normalized_dims)

        normalized_bundle = deepcopy(graph_bundle)
        for graph_idx, graph in enumerate(normalized_bundle):
            if graph_idx != 0:
                continue

            tensor_is_split: Dict[int, bool] = {}

            for op in graph:
                op_type = op.get("op_type", "")
                input_tensors = op.get("input_tensors", [])
                output_tensors = op.get("output_tensors", [])

                if op_type == "kn_input_op":
                    for tensor in output_tensors:
                        guid = tensor.get("guid")
                        if guid is not None:
                            tensor_is_split[guid] = False
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
                        guid = tensor.get("guid")
                        if guid is not None:
                            tensor_is_split[guid] = True
                    continue

                reduction_axis = _reduction_axis_from_op(op_type)
                output_split = input_split
                if input_split and reduction_axis == 0:
                    output_split = False

                if output_split:
                    for tensor in output_tensors:
                        _normalize_split_tensor_shape(tensor)

                for tensor in output_tensors:
                    guid = tensor.get("guid")
                    if guid is not None:
                        tensor_is_split[guid] = output_split

        return normalized_bundle

    def _align_split_boundary_inputs(graph_bundle):
        if len(graph_bundle) != 2:
            return graph_bundle

        symbolic_first_graph = graph_structure_to_cleave(graph_bundle[0])
        boundary_shapes = [
            tuple(node.get_shape())
            for node in symbolic_first_graph.nodes
            if node.is_output()
        ]
        combine_graph = deepcopy(graph_bundle[1])
        input_ops = [op for op in combine_graph if op.get("op_type") == "kn_input_op"]
        assert len(boundary_shapes) == len(input_ops), (
            "Split boundary mismatch: found %d symbolic outputs but %d combine inputs"
            % (len(boundary_shapes), len(input_ops))
        )

        for op, shape in zip(input_ops, boundary_shapes):
            output_tensors = op.get("output_tensors", [])
            assert (
                len(output_tensors) == 1
            ), "Expected one output tensor for combine input op, got %d" % len(
                output_tensors
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
                if tuple(candidate_shape) == tuple(shape):
                    tensor["normalized_split_axis"] = split_pos
                    break
                if (
                    len(candidate_shape) == len(shape) + 1
                    and candidate_shape[-1] == 1
                    and tuple(candidate_shape[:-1]) == tuple(shape)
                ):
                    tensor["normalized_split_axis"] = split_pos
                    break

        split_axis_by_guid: Dict[int, int] = {}
        for op in input_ops:
            output_tensors = op.get("output_tensors", [])
            if len(output_tensors) != 1:
                continue
            tensor = output_tensors[0]
            guid = tensor.get("guid")
            split_axis = tensor.pop("normalized_split_axis", None)
            if guid is not None and split_axis is not None:
                split_axis_by_guid[guid] = split_axis

        for op in combine_graph:
            op_type = op.get("op_type", "")
            input_tensors = op.get("input_tensors", [])
            output_tensors = op.get("output_tensors", [])

            input_split_axis = None
            for tensor in input_tensors:
                guid = tensor.get("guid")
                if guid is None or guid not in split_axis_by_guid:
                    continue
                input_split_axis = split_axis_by_guid[guid]
                break

            if input_split_axis is None:
                continue

            reduction_axis = _reduction_axis_from_op(op_type)
            output_split_axis = input_split_axis
            if reduction_axis is not None:
                if reduction_axis == 0 and input_split_axis != 0:
                    op["op_type"] = f"kn_reduction_{input_split_axis}_op"
                    reduction_axis = input_split_axis
                if reduction_axis == input_split_axis:
                    output_split_axis = None
                elif reduction_axis < input_split_axis:
                    output_split_axis = input_split_axis - 1

            for tensor in output_tensors:
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
            return tuple(
                sorted((key, _freeze_structure(item)) for key, item in value.items())
            )
        if isinstance(value, list):
            return tuple(_freeze_structure(item) for item in value)
        return value

    graphs = []
    seen_graphs = set()
    for i, gs in enumerate(graph_structures):
        if _contains_split_ops(gs):
            split_result = split_graph(gs)
            if split_result is None:
                print(f"Invalid split graph (no. {i}), skipping")
                continue

            split_graphs, symbolic_split_factor = split_result
            if len(split_graphs) != 2:
                print(f"Invalid split graph (no. {i}), skipping")
                continue

            key = (
                _freeze_structure(split_graphs),
                symbolic_split_factor,
            )
            if key not in seen_graphs:
                seen_graphs.add(key)
                graphs.append((split_graphs, symbolic_split_factor))
        else:
            candidate = [gs]
            key = (_freeze_structure(candidate), None)
            if key not in seen_graphs:
                seen_graphs.add(key)
                graphs.append((candidate, None))

    cleave_graphs = []
    for graph, symbolic_split_factor in graphs:
        raw_graph_bundle = graph
        graph_for_conversion = _normalize_split_graph_layouts(raw_graph_bundle)
        graph_for_conversion = _align_split_boundary_inputs(graph_for_conversion)
        inferred_symbolic_split_factor = symbolic_split_factor
        if inferred_symbolic_split_factor is None:
            inferred_symbolic_split_factor = _infer_symbolic_split_factor_from_graphs(
                raw_graph_bundle
            )

        candidate_split_factors = [None]
        if inferred_symbolic_split_factor is not None and split_factors:
            candidate_split_factors = [int(v) for v in split_factors]

        for split_factor_value in candidate_split_factors:
            local_symbol_mapping = (
                None if symbol_map is None else dict(symbol_map)
            )
            local_symbol_mapping = _augment_split_symbol_mapping(
                raw_graph_bundle,
                local_symbol_mapping,
                symbolic_split_factor=inferred_symbolic_split_factor,
                split_factor=split_factor_value,
            )

            fuse_rsqrt_flags = [True] * len(graph_for_conversion)
            if len(graph_for_conversion) == 2:
                # Preserve the explicit sqrt + reciprocal-div chain in split-combine
                # graphs so their Cleave form matches the reference combine graph.
                fuse_rsqrt_flags[1] = False

            bundle = [
                graph_structure_to_cleave(
                    gs,
                    symbol_map=local_symbol_mapping,
                    fuse_rsqrt=fuse_rsqrt,
                    dtype=dtype,
                    accum_dtype=accum_dtype,
                    out_dtype=out_dtype,
                )
                for gs, fuse_rsqrt in zip(graph_for_conversion, fuse_rsqrt_flags)
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
    from ..graph import PlaceHolderNode

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
