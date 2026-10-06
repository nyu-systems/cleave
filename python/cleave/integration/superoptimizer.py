from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import tilelang as tl
import tvm
from tilelang.carver.roller import Edge

from ..graph import Graph, OutputNode, PlaceHolderNode
from ..ops import (
    Matmul,
    Add,
    Div,
    Mul,
    Sub,
    Exp,
    Reduction,
    Silu,
    Sqrt,
    RSqrt,
    SplitView,
)
from ..ops.base import wire_input
from ..util.graph_utils import find_topo_sort, gather_all_nodes


_SUPPORTED_OPS = {
    "kn_input_op",
    "kn_output_op",
    "kn_matmul_op",
    "kn_split_0_op",
    "kn_split_1_op",
    "kn_split_2_op",
    "kn_split_3_op",
    "kn_split_4_op",
    "kn_add_op",
    "kn_exp_op",
    "kn_silu_op",
    "kn_sqrt_op",
    "kn_reduction_0_op",
    "kn_reduction_1_op",
    "kn_reduction_2_op",
    "kn_reduction_3_op",
    "kn_reduction_4_op",
    "kn_div_op",
    "kn_mul_op",
    "kn_sub_op",
}


def _dtype_from_tensor(tensor_desc: Optional[Dict], default: str) -> str:
    if not tensor_desc:
        return default
    dtype_name = tensor_desc.get("dtype")
    if dtype_name is None:
        return default
    return str(dtype_name)


def _shape_from_tensor(
    tensor_desc: Dict,
    symbol_map: Optional[Dict[int, int]] = None,
) -> Tuple[int, ...]:
    dims_list = [int(v) for v in tensor_desc.get("dim", [])]
    num_dims = tensor_desc.get("num_dims")
    if isinstance(num_dims, int) and num_dims > 0:
        dims_list = dims_list[:num_dims]
    dims = tuple(dims_list)
    if not symbol_map:
        return dims
    mapped_dims = []
    for v in dims:
        assert v in symbol_map, (
            f"Dimension {v} not found in symbol_map while mapping shape {dims}. "
            f"Available symbol_map keys: {sorted(symbol_map.keys())}"
        )
        mapped_dims.append(symbol_map[v])
    return tuple(mapped_dims)


def _infer_broadcast_index(dims: Tuple[int, ...], other_shape: Tuple[int, ...]) -> int:
    if tuple(dims) == tuple(other_shape):
        return -1
    if len(dims) < 2:
        return -1
    batch_dims = dims[:-2]
    m_dim, n_dim = dims[-2], dims[-1]
    if tuple(other_shape) == tuple(batch_dims + (m_dim,)):
        return 0
    if tuple(other_shape) == tuple(batch_dims + (n_dim,)):
        return 1
    return -1


def _infer_matmul_config(
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
    transpose_a: bool,
    transpose_b: bool,
) -> Tuple[Tuple[int, ...], bool, bool, Tuple[int, ...], Tuple[int, ...]]:
    def _normalize_input_batch_dims(
        input_batch_dims: Tuple[int, ...],
        output_batch_dims: Tuple[int, ...],
        input_name: str,
    ) -> Tuple[int, ...]:
        if len(input_batch_dims) > len(output_batch_dims):
            raise ValueError(
                f"Invalid matmul input shapes: input {input_name} has batch rank "
                f"{len(input_batch_dims)} > output batch rank {len(output_batch_dims)}"
            )
        if not output_batch_dims:
            return ()

        n_input = len(input_batch_dims)
        n_output = len(output_batch_dims)
        if n_input == 0:
            return (1,) * n_output

        compatible_candidates = []

        def _build_candidate(
            selected_positions: Tuple[int, ...],
        ) -> Optional[Tuple[int, ...]]:
            candidate = [1] * n_output
            for input_idx, out_idx in enumerate(selected_positions):
                candidate[out_idx] = int(input_batch_dims[input_idx])

            for out_dim, in_dim in zip(output_batch_dims, candidate):
                out_dim_i = int(out_dim)
                in_dim_i = int(in_dim)
                if in_dim_i <= 0 or out_dim_i % in_dim_i != 0:
                    return None
            return tuple(candidate)

        def _search(
            next_out_idx: int,
            next_input_idx: int,
            selected_positions: Tuple[int, ...],
        ) -> None:
            remaining_inputs = n_input - next_input_idx
            remaining_outputs = n_output - next_out_idx
            if remaining_outputs < remaining_inputs:
                return
            if next_input_idx == n_input:
                candidate = _build_candidate(selected_positions)
                if candidate is None:
                    return
                exact_matches = sum(
                    int(output_batch_dims[out_idx]) == int(input_batch_dims[input_idx])
                    for input_idx, out_idx in enumerate(selected_positions)
                )
                movement_cost = sum(
                    abs(out_idx - input_idx)
                    for input_idx, out_idx in enumerate(selected_positions)
                )
                compatible_candidates.append(
                    (
                        exact_matches,
                        -movement_cost,
                        selected_positions,
                        candidate,
                    )
                )
                return

            max_out_idx = n_output - remaining_inputs
            for out_idx in range(next_out_idx, max_out_idx + 1):
                _search(
                    out_idx + 1,
                    next_input_idx + 1,
                    selected_positions + (out_idx,),
                )

        _search(0, 0, ())
        if compatible_candidates:
            compatible_candidates.sort(reverse=True)
            return compatible_candidates[0][3]

        if len(input_batch_dims) < len(output_batch_dims):
            padded_dims = (1,) * (
                len(output_batch_dims) - len(input_batch_dims)
            ) + tuple(int(v) for v in input_batch_dims)
        else:
            padded_dims = tuple(int(v) for v in input_batch_dims)

        for out_dim, in_dim in zip(output_batch_dims, padded_dims):
            out_dim_i = int(out_dim)
            in_dim_i = int(in_dim)
            if in_dim_i <= 0 or out_dim_i % in_dim_i != 0:
                raise ValueError(
                    f"Invalid matmul input shapes: input {input_name} batch dim "
                    f"{in_dim_i} is incompatible with output batch dim {out_dim_i}"
                )
        return padded_dims

    # Support matvec-style broadcast: (K,) x (K, N) -> (N,)
    if len(a_shape) == 1 and len(b_shape) >= 2:
        batch_dims = b_shape[:-2]
        k_dim = b_shape[-1] if transpose_b else b_shape[-2]
        n_dim = b_shape[-2] if transpose_b else b_shape[-1]
        if a_shape[-1] != k_dim:
            raise ValueError(f"Invalid matmul input shapes: {a_shape}, {b_shape}")
        m_dim = 1
        dims = (*batch_dims, m_dim, n_dim, k_dim)
        return dims, transpose_a, transpose_b, batch_dims, batch_dims
    if len(a_shape) >= 2 and len(b_shape) == 1:
        batch_dims = a_shape[:-2]
        m_dim = a_shape[-1] if transpose_a else a_shape[-2]
        k_dim = a_shape[-2] if transpose_a else a_shape[-1]
        if b_shape[-1] != k_dim:
            raise ValueError(f"Invalid matmul input shapes: {a_shape}, {b_shape}")
        n_dim = 1
        dims = (*batch_dims, m_dim, n_dim, k_dim)
        return dims, transpose_a, transpose_b, batch_dims, batch_dims
    if len(a_shape) < 2 or len(b_shape) < 2:
        raise ValueError(f"Invalid matmul input shapes: {a_shape}, {b_shape}")

    k_dim = a_shape[-2] if transpose_a else a_shape[-1]
    b_k_dim = b_shape[-1] if transpose_b else b_shape[-2]
    if k_dim != b_k_dim:
        raise ValueError(f"Invalid matmul input shapes: {a_shape}, {b_shape}")

    if len(out_shape) >= 2:
        batch_dims = out_shape[:-2]
        m_dim, n_dim = out_shape[-2], out_shape[-1]
    else:
        batch_dims = a_shape[:-2]
        m_dim = a_shape[-1] if transpose_a else a_shape[-2]
        n_dim = b_shape[-2] if transpose_b else b_shape[-1]
    a_batch_dims = _normalize_input_batch_dims(a_shape[:-2], batch_dims, "A")
    b_batch_dims = _normalize_input_batch_dims(b_shape[:-2], batch_dims, "B")
    dims = (*batch_dims, m_dim, n_dim, k_dim)
    return dims, transpose_a, transpose_b, a_batch_dims, b_batch_dims


def graph_structure_to_cleave(
    graph_structure: List[Dict],
    dtype: str = "float16",
    accum_dtype: str = "float32",
    out_dtype: str = "float16",
    symbol_map: Optional[Dict[int, int]] = None,
    fuse_rsqrt: bool = True,
) -> Graph:
    tensor_producer: Dict[int, Tuple[object, int]] = {}
    tensor_shape: Dict[int, Tuple[int, ...]] = {}
    tensor_dtype: Dict[int, str] = {}
    tensor_consumers: Dict[int, List[str]] = {}

    output_nodes: List[OutputNode] = []

    if symbol_map is not None:
        symbol_map[1] = 1  # ensure the constant dim '1' is always mapped to 1

    for op in graph_structure:
        op_type = op.get("op_type", "")
        for tensor in op.get("input_tensors", []):
            guid = tensor.get("guid")
            if guid is None:
                continue
            if tensor.get("dtype") is not None:
                tensor_dtype[guid] = _dtype_from_tensor(
                    tensor, tensor_dtype.get(guid, dtype)
                )
            tensor_consumers.setdefault(guid, []).append(op_type)
        for tensor in op.get("output_tensors", []):
            guid = tensor.get("guid")
            if guid is None:
                continue
            if tensor.get("dtype") is not None:
                tensor_dtype[guid] = _dtype_from_tensor(
                    tensor, tensor_dtype.get(guid, dtype)
                )

    def _elementwise_out_dtype(output_guid: Optional[int]) -> str:
        if output_guid is None:
            return accum_dtype
        metadata_dtype = tensor_dtype.get(output_guid)
        if metadata_dtype is not None:
            return metadata_dtype
        consumers = tensor_consumers.get(output_guid, [])
        if consumers and all(op_type == "kn_output_op" for op_type in consumers):
            return out_dtype
        return accum_dtype

    def _producer_out_dtype(guid: Optional[int]) -> str:
        if guid is None:
            return dtype

        src_node, src_id = tensor_producer.get(guid, (None, 0))
        if src_node is None:
            return tensor_dtype.get(guid, dtype)

        producer_out_dtype = getattr(src_node, "out_dtype", None)
        if producer_out_dtype is not None:
            return str(producer_out_dtype)

        producer_dtype = getattr(src_node, "dtype", None)
        if producer_dtype is not None:
            return str(producer_dtype)

        dtypes = getattr(src_node, "_dtypes", None)
        if dtypes and src_id < len(dtypes) and dtypes[src_id] is not None:
            return str(dtypes[src_id])

        return dtype

    def _common_input_dtype(*guids: Optional[int]) -> str:
        input_dtypes = [_producer_out_dtype(guid) for guid in guids if guid is not None]
        if not input_dtypes:
            return dtype
        if len(set(input_dtypes)) == 1:
            return input_dtypes[0]
        return accum_dtype

    def _elevated_n_batch_dims_from_node(src_node, src_shape, out_dims):
        # An elementwise op inherits the same number of grid-elided (batch)
        # leading dims as its producing context. A Reduction (or an already
        # elevated rmsnorm-chain op) marks every dim except the innermost as
        # batch, so its rank-2 (batch, seq) output tiles to a 1-D (block_S,)
        # buffer. Downstream same-rank elementwise ops must match by setting
        # n_batch_dims = rank-1; the default dims[:-2] gives 0 and emits a 2-D
        # loop that cannot unpack the 1-D tile. Returns None (keep default)
        # unless the producer marks all-but-last as batch.
        if src_node is None or src_shape is None:
            return None
        src_nbd = getattr(src_node, "n_batch_dims", None)
        if src_nbd is None:
            return None
        src_rank = len(src_shape)
        if src_rank >= 2 and src_nbd == src_rank - 1 and len(out_dims) == src_rank:
            return len(out_dims) - 1
        return None

    def _elevated_n_batch_dims(input_guid, out_dims):
        if input_guid is None:
            return None
        src_node, _ = tensor_producer.get(input_guid, (None, None))
        src_shape = tensor_shape.get(input_guid)
        return _elevated_n_batch_dims_from_node(src_node, src_shape, out_dims)

    for op_idx, op in enumerate(graph_structure):
        op_type = op.get("op_type", "")
        if op_type not in _SUPPORTED_OPS:
            raise ValueError(f"Unsupported op_type '{op_type}' at index {op_idx}")

        input_tensors = op.get("input_tensors", [])
        output_tensors = op.get("output_tensors", [])

        if op_type == "kn_input_op":
            if not output_tensors:
                continue
            tensor = output_tensors[0]
            guid = tensor["guid"]
            shape = _shape_from_tensor(tensor, symbol_map)
            input_dtype = _dtype_from_tensor(tensor, dtype)
            node = PlaceHolderNode(name=f"input_{guid}")
            node.set_shape(shape)
            node.set_dtype(tvm.DataType(input_dtype))
            tensor_producer[guid] = (node, 0)
            tensor_shape[guid] = shape
            tensor_dtype[guid] = input_dtype
            continue

        if op_type == "kn_output_op":
            if not input_tensors:
                continue
            input_guid = input_tensors[0]["guid"]
            if input_guid not in tensor_producer:
                raise ValueError(f"Output tensor guid {input_guid} has no producer.")
            src_node, src_id = tensor_producer[input_guid]
            output_nodes.append(
                OutputNode(src_node, dtype=tensor_dtype.get(input_guid, out_dtype))
            )
            continue

        # regular ops
        skip_default_wiring = False
        if op_type.startswith("kn_split_") and op_type.endswith("_op"):
            if len(input_tensors) != 1 or len(output_tensors) != 1:
                raise ValueError(f"{op_type} expects one input and one output tensor.")
            axis = int(op_type.split("_")[2])
            guid = input_tensors[0]["guid"]
            src_shape = tensor_shape.get(
                guid, _shape_from_tensor(input_tensors[0], symbol_map)
            )
            out_shape = _shape_from_tensor(output_tensors[0], symbol_map)
            if len(out_shape) != len(src_shape) + 1:
                raise ValueError(
                    f"Invalid split output rank: src {src_shape}, out {out_shape}, axis={axis}"
                )
            split_pos = len(src_shape) + 1 - 3
            factor = None
            supported_layouts = []
            for position in (0, split_pos):
                candidate_factor = int(out_shape[position])
                if candidate_factor <= 0 or src_shape[axis] % candidate_factor != 0:
                    continue

                candidate_shape = list(src_shape)
                candidate_shape[axis] = src_shape[axis] // candidate_factor
                candidate_shape.insert(position, candidate_factor)
                supported_layouts.append(tuple(candidate_shape))
                if tuple(candidate_shape) == tuple(out_shape):
                    factor = candidate_factor
                    break

            if factor is None:
                raise ValueError(
                    "Unsupported split layout: expected one of the supported raw split layouts; "
                    f"src={src_shape}, axis={axis}, out={out_shape}, "
                    f"supported={supported_layouts}"
                )
            src_node = tensor_producer.get(guid, (None, None))[0]
            in_dtype = _producer_out_dtype(guid)
            node = SplitView(
                src_shape,
                axis=axis,
                factor=factor,
                name=f"split_{axis}_{op_idx}",
                dtype=in_dtype,
                split_dim_is_batch=True,
            )
        elif op_type == "kn_matmul_op":
            a_guid = input_tensors[0]["guid"]
            b_guid = input_tensors[1]["guid"]
            a_shape = tensor_shape.get(
                a_guid, _shape_from_tensor(input_tensors[0], symbol_map)
            )
            b_shape = tensor_shape.get(
                b_guid, _shape_from_tensor(input_tensors[1], symbol_map)
            )
            out_shape = _shape_from_tensor(output_tensors[0], symbol_map)
            transpose_a = bool(op.get("trans_A", False))
            transpose_b = bool(op.get("trans_B", False))
            (
                dims,
                transpose_a,
                transpose_b,
                a_batch_dims,
                b_batch_dims,
            ) = _infer_matmul_config(
                a_shape, b_shape, out_shape, transpose_a, transpose_b
            )
            node = Matmul(
                dims,
                input_batch_dims_A=a_batch_dims,
                input_batch_dims_B=b_batch_dims,
                name=f"mma_{op_idx}",
                transpose_A=transpose_a,
                transpose_B=transpose_b,
                warp_policy="rows",
                dtype=dtype,
                out_dtype=accum_dtype,
            )
        elif op_type == "kn_exp_op":
            guid = input_tensors[0]["guid"]
            shape = tensor_shape.get(
                guid, _shape_from_tensor(input_tensors[0], symbol_map)
            )
            output_guid = output_tensors[0]["guid"] if output_tensors else None
            node = Exp(
                shape,
                name=f"exp_{op_idx}",
                dtype=_producer_out_dtype(guid),
                out_dtype=_elementwise_out_dtype(output_guid),
                n_batch_dims=_elevated_n_batch_dims(guid, shape),
            )
        elif op_type == "kn_silu_op":
            guid = input_tensors[0]["guid"]
            shape = tensor_shape.get(
                guid, _shape_from_tensor(input_tensors[0], symbol_map)
            )
            output_guid = output_tensors[0]["guid"] if output_tensors else None
            node = Silu(
                shape,
                name=f"silu_{op_idx}",
                dtype=_producer_out_dtype(guid),
                out_dtype=_elementwise_out_dtype(output_guid),
                n_batch_dims=_elevated_n_batch_dims(guid, shape),
            )
        elif op_type == "kn_sqrt_op":
            guid = input_tensors[0]["guid"]
            shape = tensor_shape.get(
                guid, _shape_from_tensor(input_tensors[0], symbol_map)
            )
            output_guid = output_tensors[0]["guid"] if output_tensors else None
            node = Sqrt(
                shape,
                name=f"sqrt_{op_idx}",
                dtype=_producer_out_dtype(guid),
                out_dtype=_elementwise_out_dtype(output_guid),
                n_batch_dims=_elevated_n_batch_dims(guid, shape),
            )
        elif op_type in (
            "kn_reduction_0_op",
            "kn_reduction_1_op",
            "kn_reduction_2_op",
            "kn_reduction_3_op",
            "kn_reduction_4_op",
        ):
            guid = input_tensors[0]["guid"]
            shape = tensor_shape.get(
                guid, _shape_from_tensor(input_tensors[0], symbol_map)
            )
            axis = int(op_type.split("_")[2])
            input_dtype = _producer_out_dtype(guid)
            reduction_axis = -1 if axis == len(shape) - 1 else axis
            node = Reduction(
                shape,
                axis=reduction_axis,
                name=f"reduction_{axis}_{op_idx}",
                dtype=input_dtype,
                out_dtype=accum_dtype,
            )
        elif op_type == "kn_div_op":
            scalar = op.get("scalar")
            scalar_on_left = bool(op.get("scalar_on_left", False))
            output_guid = output_tensors[0]["guid"] if output_tensors else None
            op_out_dtype = _elementwise_out_dtype(output_guid)
            if scalar is not None:
                if not input_tensors:
                    raise ValueError("kn_div_op with scalar requires one input tensor.")
                tensor_guid = input_tensors[0]["guid"]
                if fuse_rsqrt and scalar_on_left and float(scalar) == 1.0:
                    src_node, src_id = tensor_producer.get(tensor_guid, (None, None))
                    if isinstance(src_node, Sqrt) and src_node.inputs:
                        sqrt_in_edge = src_node.inputs[0]
                        # detach the unused sqrt node to avoid dangling consumers.
                        if sqrt_in_edge in sqrt_in_edge.src_node._out_edges:
                            sqrt_in_edge.src_node._out_edges.remove(sqrt_in_edge)
                        if sqrt_in_edge in src_node._in_edges:
                            src_node._in_edges.remove(sqrt_in_edge)
                        in_shape = tuple(src_node.input_buffers[0].shape)
                        rsqrt_nbd = _elevated_n_batch_dims_from_node(
                            sqrt_in_edge.src_node, in_shape, in_shape
                        )
                        node = RSqrt(
                            in_shape,
                            name=f"rsqrt_{op_idx}",
                            dtype=accum_dtype,
                            out_dtype=op_out_dtype,
                            n_batch_dims=rsqrt_nbd,
                        )
                        wire_input(
                            sqrt_in_edge.src_node,
                            node,
                            0,
                            src_id=sqrt_in_edge.src_id,
                        )
                        skip_default_wiring = True
                    else:
                        lhs_shape = tensor_shape.get(
                            tensor_guid, _shape_from_tensor(input_tensors[0], symbol_map)
                        )
                        node = Div(
                            lhs_shape,
                            const=float(scalar),
                            const_on_left=scalar_on_left,
                            name=f"div_{op_idx}",
                            dtype=accum_dtype,
                            out_dtype=op_out_dtype,
                            n_batch_dims=_elevated_n_batch_dims(tensor_guid, lhs_shape),
                        )
                else:
                    lhs_shape = tensor_shape.get(
                        tensor_guid, _shape_from_tensor(input_tensors[0], symbol_map)
                    )
                    node = Div(
                        lhs_shape,
                        const=float(scalar),
                        const_on_left=scalar_on_left,
                        name=f"div_{op_idx}",
                        dtype=_producer_out_dtype(tensor_guid),
                        out_dtype=op_out_dtype,
                        n_batch_dims=_elevated_n_batch_dims(tensor_guid, lhs_shape),
                    )
            else:
                lhs_guid = input_tensors[0]["guid"]
                rhs_guid = input_tensors[1]["guid"]
                lhs_shape = tensor_shape.get(
                    lhs_guid, _shape_from_tensor(input_tensors[0], symbol_map)
                )
                rhs_shape = tensor_shape.get(
                    rhs_guid, _shape_from_tensor(input_tensors[1], symbol_map)
                )
                broadcast = _infer_broadcast_index(lhs_shape, rhs_shape)
                op_out_dtype = _common_input_dtype(lhs_guid, rhs_guid)
                node = Div(
                    lhs_shape,
                    broadcast=broadcast,
                    name=f"div_{op_idx}",
                    dtype=_common_input_dtype(lhs_guid, rhs_guid),
                    out_dtype=op_out_dtype,
                    n_batch_dims=_elevated_n_batch_dims(lhs_guid, lhs_shape),
                )
        elif op_type == "kn_mul_op":
            scalar = op.get("scalar")
            output_guid = output_tensors[0]["guid"] if output_tensors else None
            op_out_dtype = _elementwise_out_dtype(output_guid)
            if scalar is not None:
                if not input_tensors:
                    raise ValueError("kn_mul_op with scalar requires one input tensor.")
                lhs_guid = input_tensors[0]["guid"]
                lhs_shape = tensor_shape.get(
                    lhs_guid, _shape_from_tensor(input_tensors[0], symbol_map)
                )
                node = Mul(
                    lhs_shape,
                    const=float(scalar),
                    name=f"mul_{op_idx}",
                    dtype=_producer_out_dtype(lhs_guid),
                    out_dtype=op_out_dtype,
                    n_batch_dims=_elevated_n_batch_dims(lhs_guid, lhs_shape),
                )
            else:
                if len(input_tensors) < 2:
                    raise ValueError("kn_mul_op requires two input tensors.")
                lhs_guid = input_tensors[0]["guid"]
                rhs_guid = input_tensors[1]["guid"]
                lhs_node = tensor_producer.get(lhs_guid, (None, None))[0]
                rhs_node = tensor_producer.get(rhs_guid, (None, None))[0]
                # Ensure deterministic ordering for swiglu-like patterns: prefer silu as lhs
                if isinstance(rhs_node, Silu) and not isinstance(lhs_node, Silu):
                    lhs_guid, rhs_guid = rhs_guid, lhs_guid
                    input_tensors = [input_tensors[1], input_tensors[0]]
                lhs_shape = tensor_shape.get(
                    lhs_guid, _shape_from_tensor(input_tensors[0], symbol_map)
                )
                rhs_shape = tensor_shape.get(
                    rhs_guid, _shape_from_tensor(input_tensors[1], symbol_map)
                )
                broadcast = _infer_broadcast_index(lhs_shape, rhs_shape)
                node = Mul(
                    lhs_shape,
                    broadcast=broadcast,
                    name=f"mul_{op_idx}",
                    dtype=_common_input_dtype(lhs_guid, rhs_guid),
                    out_dtype=op_out_dtype,
                    n_batch_dims=_elevated_n_batch_dims(lhs_guid, lhs_shape),
                )
        elif op_type == "kn_add_op":
            scalar = op.get("scalar")
            output_guid = output_tensors[0]["guid"] if output_tensors else None
            op_out_dtype = _elementwise_out_dtype(output_guid)
            if scalar is not None:
                if not input_tensors:
                    raise ValueError("kn_add_op with scalar requires one input tensor.")
                lhs_guid = input_tensors[0]["guid"]
                lhs_shape = tensor_shape.get(
                    lhs_guid, _shape_from_tensor(input_tensors[0], symbol_map)
                )
                node = Add(
                    lhs_shape,
                    const=float(scalar),
                    name=f"add_{op_idx}",
                    dtype=_producer_out_dtype(lhs_guid),
                    out_dtype=op_out_dtype,
                    n_batch_dims=_elevated_n_batch_dims(lhs_guid, lhs_shape),
                )
            else:
                lhs_guid = input_tensors[0]["guid"]
                rhs_guid = input_tensors[1]["guid"]
                lhs_shape = tensor_shape.get(
                    lhs_guid, _shape_from_tensor(input_tensors[0], symbol_map)
                )
                rhs_shape = tensor_shape.get(
                    rhs_guid, _shape_from_tensor(input_tensors[1], symbol_map)
                )
                broadcast = _infer_broadcast_index(lhs_shape, rhs_shape)
                node = Add(
                    lhs_shape,
                    broadcast=broadcast,
                    name=f"add_{op_idx}",
                    dtype=_common_input_dtype(lhs_guid, rhs_guid),
                    out_dtype=op_out_dtype,
                    n_batch_dims=_elevated_n_batch_dims(lhs_guid, lhs_shape),
                )
        elif op_type == "kn_sub_op":
            lhs_guid = input_tensors[0]["guid"]
            rhs_guid = input_tensors[1]["guid"]
            output_guid = output_tensors[0]["guid"] if output_tensors else None
            op_out_dtype = _elementwise_out_dtype(output_guid)
            lhs_shape = tensor_shape.get(
                lhs_guid, _shape_from_tensor(input_tensors[0], symbol_map)
            )
            rhs_shape = tensor_shape.get(
                rhs_guid, _shape_from_tensor(input_tensors[1], symbol_map)
            )
            broadcast = _infer_broadcast_index(lhs_shape, rhs_shape)
            node = Sub(
                lhs_shape,
                broadcast=broadcast,
                name=f"sub_{op_idx}",
                dtype=_common_input_dtype(lhs_guid, rhs_guid),
                out_dtype=op_out_dtype,
                n_batch_dims=_elevated_n_batch_dims(lhs_guid, lhs_shape),
            )
        else:
            raise ValueError(f"Unsupported op_type '{op_type}' at index {op_idx}")

        # connect inputs
        if not skip_default_wiring:
            for dst_id, tensor in enumerate(input_tensors):
                guid = tensor["guid"]
                if guid not in tensor_producer:
                    placeholder = PlaceHolderNode(name=f"input_{guid}")
                    placeholder.set_shape(_shape_from_tensor(tensor, symbol_map))
                    placeholder_dtype = _dtype_from_tensor(tensor, dtype)
                    placeholder.set_dtype(tvm.DataType(placeholder_dtype))
                    tensor_producer[guid] = (placeholder, 0)
                    tensor_shape[guid] = _shape_from_tensor(tensor, symbol_map)
                    tensor_dtype[guid] = placeholder_dtype
                src_node, src_id = tensor_producer[guid]
                wire_input(src_node, node, dst_id, src_id=src_id)
                if isinstance(src_node, PlaceHolderNode) and dst_id < len(
                    node.input_buffers
                ):
                    tensor_shape[guid] = tuple(src_node.get_shape(src_id))

        # register output tensors
        for output_id, tensor in enumerate(output_tensors):
            guid = tensor["guid"]
            tensor_producer[guid] = (node, output_id)
            tensor_shape[guid] = tuple(node.get_shape(output_id))

    if not output_nodes:
        raise ValueError("No output nodes found in graph structure.")

    nodes = gather_all_nodes(*output_nodes)
    return Graph(nodes=find_topo_sort([n for n in nodes if n.is_output()]))
