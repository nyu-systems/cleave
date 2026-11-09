"""Cut a searched split structure into ``[partial, combine]`` (``cut_split_graph``, memoized;
a refusal is a ``Rejected(reason)``)."""

import json
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


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
        return int(op_type.split("_")[-2])
    return None


def _reduction_axis_from_op(op_type: str) -> Optional[int]:
    if op_type.startswith("kn_reduce_sum_") and op_type.endswith("_op"):
        return int(op_type.split("_")[-2])
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

            if not (0 <= split_axis < len(raw_input_dims)):
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


@dataclass(frozen=True)
class Rejected:
    """Why a structure with split ops admits no partial/combine cut."""

    reason: str


SplitResult = Tuple[List[List[Dict]], Optional[int]]  # ([partial, combine], symbolic factor)


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
        new_op["input_tensors"] = [_renumber_tensor(t) for t in op.get("input_tensors", [])]
        new_op["output_tensors"] = [_renumber_tensor(t) for t in op.get("output_tensors", [])]
        renumbered_ops.append(new_op)
    return renumbered_ops


def _map_matmul_output_axis_to_input(op: Dict, axis: int) -> Optional[Tuple[int, int]]:
    input_tensors = op.get("input_tensors", [])
    output_tensors = op.get("output_tensors", [])
    if len(input_tensors) != 2 or len(output_tensors) != 1:
        return None
    output_rank = len(_tensor_dims(output_tensors[0]))
    if axis < 0 or axis >= output_rank or axis < output_rank - 2:
        return None
    if axis == output_rank - 2:
        a_rank = len(_tensor_dims(input_tensors[0]))
        return input_tensors[0]["guid"], a_rank - 1 if op.get("trans_A", False) else a_rank - 2
    b_rank = len(_tensor_dims(input_tensors[1]))
    return input_tensors[1]["guid"], b_rank - 2 if op.get("trans_B", False) else b_rank - 1


def _template_reduce_flags(tensor: Dict, rank: Optional[int] = None) -> Tuple[bool, ...]:
    shape_rank = len(_tensor_dims(tensor)) if rank is None else rank
    return tuple(bool(v) for v in tensor.get("can_reduce", [])[:shape_rank])


def _serialize_tensor(tensor_state: _TensorState) -> Dict:
    template = tensor_state.template or {}
    serialized = deepcopy(template)
    raw_dims = _raw_tensor_dims(template)
    dim_capacity = max(len(raw_dims), len(tensor_state.shape))
    serialized["dim"] = list(tensor_state.shape) + [0] * (dim_capacity - len(tensor_state.shape))
    serialized["guid"] = tensor_state.guid
    serialized["num_dims"] = len(tensor_state.shape)
    for key, flags in (("can_reduce", tensor_state.can_reduce_flags),
                       ("can_split", tensor_state.can_split_flags)):
        if key in template:
            flags_len = max(len(template.get(key, [])), dim_capacity)
            out = [False] * flags_len
            for axis, flag in enumerate(flags[:flags_len]):
                out[axis] = bool(flag)
            serialized[key] = out
    return serialized


def _serialize_op(op: _CanonicalOp) -> Dict:
    serialized = {} if op.template_op is None else deepcopy(op.template_op)
    serialized["op_type"] = op.op_type
    serialized["input_tensors"] = [_serialize_tensor(t) for t in op.input_tensors]
    serialized["output_tensors"] = [_serialize_tensor(t) for t in op.output_tensors]
    return serialized


class _Cut:
    """Cuts a structure with split ops into a partial graph and a combine graph.

    Phases: ``trace_seeds`` (walk each split back to the graph input it splits),
    ``emit_inputs``, ``emit_body`` (route each op to partial or combine), ``emit_outputs``,
    then ``result``. A phase returns ``Rejected`` to stop.
    """

    def __init__(self, graph_structure: List[Dict]):
        self.gs = graph_structure
        self.producers: Dict[int, int] = {}
        self.consumers: Dict[int, List[Tuple[int, int]]] = {}
        for op_idx, op in enumerate(graph_structure):
            for tensor in op.get("output_tensors", []):
                if tensor.get("guid") is not None:
                    self.producers[tensor["guid"]] = op_idx
            for input_idx, tensor in enumerate(op.get("input_tensors", [])):
                if tensor.get("guid") is not None:
                    self.consumers.setdefault(tensor["guid"], []).append((op_idx, input_idx))
        self.split_factor: Optional[int] = None
        self.split_source: Dict[int, int] = {}  # split op output guid -> its input guid
        self.seed_axes: Dict[Tuple[int, int], int] = {}  # (graph input guid, axis)
        self.axis_by_guid: Dict[int, int] = {}  # logical split axis of each traced tensor
        self.next_tensor_guid = 1
        self.split_input_ops: List[_CanonicalOp] = []
        self.split_seed_ops: List[_CanonicalOp] = []
        self.split_graph_ops: List[_CanonicalOp] = []
        self.combine_input_ops: List[_CanonicalOp] = []
        self.combine_graph_ops: List[_CanonicalOp] = []
        self.base_tensors: Dict[int, _TensorState] = {}
        self.split_tensors: Dict[int, _TensorState] = {}
        self.combine_tensors: Dict[int, _TensorState] = {}
        self.boundary_inputs: Dict[int, _TensorState] = {}
        self.combine_started = False

    def run(self) -> Union[SplitResult, Rejected]:
        for phase in (self.trace_seeds, self.emit_inputs, self.emit_body, self.emit_outputs):
            rejected = phase()
            if rejected is not None:
                return rejected
        return self.result()

    # ---------------------------------------------------------------- trace
    def trace_seeds(self) -> Optional[Rejected]:
        for op in self.gs:
            op_type = op.get("op_type", "")
            split_axis = _split_axis_from_op(op_type)
            if split_axis is None:
                continue
            ins, outs = op.get("input_tensors", []), op.get("output_tensors", [])
            if len(ins) != 1 or len(outs) != 1:
                return Rejected(f"{op_type} needs one input and one output")
            factor = _infer_split_factor(op)
            if factor is None:
                return Rejected(f"{op_type} has no split factor > 1")
            if self.split_factor is None:
                self.split_factor = factor
            elif self.split_factor != factor:
                return Rejected(f"mixed split factors {self.split_factor} and {factor}")
            self.split_source[outs[0]["guid"]] = ins[0]["guid"]
            rejected = self._trace(ins[0].get("guid"), split_axis)
            if rejected is not None:
                return rejected
        if self.split_factor is None or not self.seed_axes:
            return Rejected("no split reaches a graph input")
        seed_order = sorted(self.seed_axes, key=lambda k: (self.producers.get(k[0], -1), k[1]))
        self.seed_axis_by_guid: Dict[int, int] = {}
        for guid, axis in seed_order:
            if self.seed_axis_by_guid.get(guid, axis) != axis:
                return Rejected(f"input {guid} split on axes {self.seed_axis_by_guid[guid]} and {axis}")
            self.seed_axis_by_guid[guid] = axis
        return None

    def _trace(self, guid, axis) -> Optional[Rejected]:
        """Follow a split back through single-consumer unary ops and matmuls to an input."""
        while True:
            existing = self.axis_by_guid.get(guid)
            if existing is not None and existing != axis:
                return Rejected(f"tensor {guid} split on axes {existing} and {axis}")
            self.axis_by_guid[guid] = axis
            producer_idx = self.producers.get(guid)
            if producer_idx is None:
                return Rejected(f"tensor {guid} has no producer")
            producer = self.gs[producer_idx]
            producer_type = producer.get("op_type", "")
            if producer_type == "kn_input_op":
                self.seed_axes.setdefault((guid, axis), axis)
                return None
            n_consumers = len(self.consumers.get(guid, []))
            if n_consumers != 1:
                return Rejected(f"split source {guid} ({producer_type}) has {n_consumers} consumers")
            if _is_scalar_like_unary(producer):
                producer_inputs = producer.get("input_tensors", [])
                if len(producer_inputs) != 1:
                    return Rejected(f"{producer_type} on the split path needs one input")
                guid = producer_inputs[0]["guid"]
                continue
            if producer_type == "kn_matmul_op":
                mapped = _map_matmul_output_axis_to_input(producer, axis)
                if mapped is None:
                    return Rejected(f"split axis {axis} does not map through a matmul input")
                guid, axis = mapped
                continue
            return Rejected(f"split cannot pass back through {producer_type}")

    # ---------------------------------------------------------------- tensor states
    def _new_state(self, shape, tokens, template, can_split_flags=None, can_reduce_flags=None,
                   is_split=False) -> _TensorState:
        state = _TensorState(
            guid=10_000_000 + self.next_tensor_guid,
            shape=tuple(int(v) for v in shape),
            tokens=tuple(tokens),
            template=deepcopy(template),
            can_split_flags=tuple(can_split_flags or (False,) * len(shape)),
            can_reduce_flags=tuple(can_reduce_flags or (False,) * len(shape)),
            is_split=is_split,
        )
        self.next_tensor_guid += 1
        return state

    def _split_shape(self, shape: Tuple[int, ...], axis: int) -> Tuple[int, ...]:
        f = self.split_factor
        if axis < 0 or axis >= len(shape):
            raise ValueError(f"Invalid split axis {axis} for shape {shape}")
        if shape[axis] % f != 0:
            raise ValueError(f"Split factor {f} does not divide shape {shape} on axis {axis}")
        return tuple([f] + [shape[i] // f if i == axis else shape[i] for i in range(len(shape))])

    def _split_state(self, src: _TensorState, axis: int) -> _TensorState:
        new_shape = self._split_shape(src.shape, axis)
        split_token = ("split", src.template.get("guid", src.guid), axis)
        return self._new_state(new_shape, (split_token,) + src.tokens, src.template,
                               can_split_flags=tuple(False for _ in new_shape),
                               can_reduce_flags=(True,) + src.can_reduce_flags, is_split=True)

    def _output_state(self, output_tensor: Dict, *, split_axis=None, split_token=None,
                      already_split=False) -> _TensorState:
        base_shape = tuple(_tensor_dims(output_tensor))
        base_flags = _template_reduce_flags(output_tensor, len(base_shape))
        guid = output_tensor.get("guid", -1)
        if split_axis is not None:
            out_shape = self._split_shape(base_shape, split_axis)
            first = split_token or ("split", guid, split_axis)
            tokens = (first,) + tuple(("op", guid, a) for a in range(len(base_shape)))
            return self._new_state(out_shape, tokens, output_tensor,
                                   can_split_flags=tuple(False for _ in out_shape),
                                   can_reduce_flags=(True,) + base_flags, is_split=True)
        if already_split:
            first = split_token or ("split", guid, 0)
            tokens = ((first,) + tuple(("op", guid, a) for a in range(1, len(base_shape)))
                      if base_shape else ())
            return self._new_state(base_shape, tokens, output_tensor,
                                   can_split_flags=tuple(False for _ in base_shape),
                                   can_reduce_flags=base_flags, is_split=True)
        return self._new_state(base_shape, tuple(("op", guid, a) for a in range(len(base_shape))),
                               output_tensor, can_split_flags=tuple(False for _ in base_shape),
                               can_reduce_flags=base_flags, is_split=False)

    def _boundary_input(self, split_state: _TensorState) -> _TensorState:
        """Partial output -> combine input pair for a tensor crossing the cut (made once)."""
        existing = self.boundary_inputs.get(split_state.guid)
        if existing is not None:
            return existing
        self.split_graph_ops.append(_CanonicalOp("kn_output_op", [split_state], [], None))
        boundary = self._new_state(split_state.shape, split_state.tokens, split_state.template,
                                   can_split_flags=tuple(False for _ in split_state.shape),
                                   can_reduce_flags=split_state.can_reduce_flags, is_split=True)
        self.combine_input_ops.append(_CanonicalOp("kn_input_op", [], [boundary], None))
        self.boundary_inputs[split_state.guid] = boundary
        return boundary

    def _resolve_input(self, tensor: Dict) -> Tuple[Optional[_TensorState], str]:
        guid = tensor.get("guid")
        if guid in self.combine_tensors:
            return self.combine_tensors[guid], "combine"
        if guid in self.split_source:
            return self.split_tensors.get(self.split_source[guid]), "split"
        if guid in self.split_tensors:
            return self.split_tensors[guid], "split"
        if guid in self.base_tensors:
            return self.base_tensors[guid], "base"
        return None, "invalid"

    # ---------------------------------------------------------------- emit
    def emit_inputs(self) -> Optional[Rejected]:
        for op in self.gs:
            if op.get("op_type") != "kn_input_op":
                continue
            outs = op.get("output_tensors", [])
            if len(outs) != 1:
                return Rejected("kn_input_op needs one output")
            tensor = outs[0]
            shape = tuple(_tensor_dims(tensor))
            base = self._new_state(
                shape, tuple(("orig", tensor.get("guid"), a) for a in range(len(shape))), tensor,
                can_split_flags=tuple(bool(v) for v in tensor.get("can_split", [])[: len(shape)]),
                can_reduce_flags=_template_reduce_flags(tensor, len(shape)), is_split=False)
            guid = tensor.get("guid")
            self.base_tensors[guid] = base
            self.split_input_ops.append(_CanonicalOp("kn_input_op", [], [base], op))
            if guid in self.seed_axis_by_guid:
                axis = self.seed_axis_by_guid[guid]
                split = self._split_state(base, axis)
                self.split_tensors[guid] = split
                self.split_seed_ops.append(_CanonicalOp(f"kn_split_{axis}_op", [base], [split], None))
        if not self.split_tensors:
            return Rejected("no graph input is split")
        return None

    def emit_body(self) -> Optional[Rejected]:
        for op in self.gs:
            op_type = op.get("op_type", "")
            if op_type in ("kn_input_op", "kn_output_op") or _split_axis_from_op(op_type) is not None:
                continue
            resolved, kinds = [], []
            for tensor in op.get("input_tensors", []):
                state, kind = self._resolve_input(tensor)
                resolved.append(state)
                kinds.append(kind)
            has_combine, has_split = "combine" in kinds, "split" in kinds
            if not has_combine and not has_split:
                continue
            if any(state is None for state in resolved):
                return Rejected(f"{op_type} reads a tensor with no state on either side of the cut")
            if has_combine and has_split:
                return Rejected(f"{op_type} mixes partial and combine inputs")
            graph_id = 1 if has_combine else 0
            if graph_id == 1 and "base" in kinds:
                return Rejected(f"combine-side {op_type} reads an unsplit graph input")
            reduction_axis = _reduction_axis_from_op(op_type)
            if self.combine_started and graph_id == 0 and reduction_axis != 0:
                return Rejected(f"partial-side {op_type} after the combine started")
            rejected = self._emit_body_op(op, op_type, resolved, graph_id, reduction_axis)
            if rejected is not None:
                return rejected
        if not self.combine_input_ops or not self.combine_graph_ops:
            return Rejected("no reduction over the split axis (empty combine graph)")
        return None

    def _emit_body_op(self, op, op_type, inputs, graph_id, reduction_axis) -> Optional[Rejected]:
        split_inputs = [s for s in inputs if s.is_split]
        split_token = split_inputs[0].tokens[0] if split_inputs else None
        output_tensor = op.get("output_tensors", [{}])[0]
        output_guid = output_tensor.get("guid")
        logical_axis = self.axis_by_guid.get(output_guid)
        already_split = graph_id == 0 and bool(split_inputs) and logical_axis is None

        if op_type.startswith("kn_reduce_max_"):
            # partial maxima would need a max-combine; the cut only builds sum-combines
            return Rejected(f"{op_type} is not supported in a split graph (no max-combine)")
        if op_type == "kn_matmul_op":
            if len(inputs) != 2:
                return Rejected("kn_matmul_op needs two inputs")
        elif reduction_axis is not None:
            if len(inputs) != 1:
                return Rejected(f"{op_type} needs one input")
            if graph_id == 0 and reduction_axis == 0:  # sums the splits: starts the combine
                self.combine_started = True
                boundary = self._boundary_input(inputs[0])
                out = self._output_state(output_tensor)
                self.combine_tensors[output_guid] = out
                self.combine_graph_ops.append(_CanonicalOp(op_type, [boundary], [out], op))
                return None
        elif not (op_type in _ELEMENTWISE_OPS or _is_scalar_like_unary(op)):
            return Rejected(f"{op_type} is not supported in a split graph")

        out = self._output_state(output_tensor, split_axis=logical_axis, split_token=split_token,
                                 already_split=already_split)
        if graph_id == 0:
            if not out.is_split:
                return Rejected(f"partial-side {op_type} output is not split")
            self.split_tensors[output_guid] = out
            self.split_graph_ops.append(_CanonicalOp(op_type, list(inputs), [out], op))
        else:
            self.combine_tensors[output_guid] = out
            self.combine_graph_ops.append(_CanonicalOp(op_type, list(inputs), [out], op))
        return None

    def emit_outputs(self) -> Optional[Rejected]:
        for op in self.gs:
            if op.get("op_type") != "kn_output_op":
                continue
            ins = op.get("input_tensors", [])
            if len(ins) != 1:
                return Rejected("kn_output_op needs one input")
            state, kind = self._resolve_input(ins[0])
            if state is None or kind != "combine":
                return Rejected(f"graph output {ins[0].get('guid')} is not computed by the combine")
            self.combine_graph_ops.append(_CanonicalOp("kn_output_op", [state], [], op))
        if not any(op.op_type == "kn_output_op" for op in self.combine_graph_ops):
            return Rejected("combine graph has no output")
        return None

    def result(self) -> SplitResult:
        first = [_serialize_op(op)
                 for op in self.split_input_ops + self.split_seed_ops + self.split_graph_ops]
        second = [_serialize_op(op) for op in self.combine_input_ops + self.combine_graph_ops]
        symbolic = None
        if any(
            len(_tensor_dims(op.get("output_tensors", [{}])[0])) >= 4
            for op in self.gs
            if _split_axis_from_op(op.get("op_type", "")) is not None
        ) or _looks_like_classic_split_k(self.gs, self.split_factor):
            symbolic = self.split_factor
        return [_renumber_graph(first), _renumber_graph(second)], symbolic


_CUT_CACHE: "OrderedDict[str, Union[SplitResult, Rejected]]" = OrderedDict()
_CUT_CACHE_SIZE = 1024


def _structure_key(graph_structure) -> str:
    return json.dumps(graph_structure, sort_keys=True, default=repr)


def cut_split_graph(graph_structure) -> Union[SplitResult, Rejected]:
    """``([partial, combine], symbolic_split_factor)`` or ``Rejected(reason)``.

    Memoized on the structure's content; treat the result as read-only.
    """
    if not _contains_split_ops(graph_structure):
        return Rejected("no split ops")
    key = _structure_key(graph_structure)
    hit = _CUT_CACHE.get(key)
    if hit is not None:
        _CUT_CACHE.move_to_end(key)
        return hit
    result = _Cut(graph_structure).run()
    _CUT_CACHE[key] = result
    if len(_CUT_CACHE) > _CUT_CACHE_SIZE:
        _CUT_CACHE.popitem(last=False)
    return result


def split_graph(graph_structure) -> Optional[SplitResult]:
    result = cut_split_graph(graph_structure)
    return None if isinstance(result, Rejected) else deepcopy(result)


def is_split_graph(graph_structure):
    if not isinstance(graph_structure, list) or not graph_structure:
        return False
    if any(isinstance(op, list) for op in graph_structure):
        return False
    return not isinstance(cut_split_graph(graph_structure), Rejected)
