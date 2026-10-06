from ..util import find_topo_sort
from .common import (
    coalesced_factor,
    get_all_round_factors,
    coalesced_tensor_shape,
    factorize,
)
from .bestfit import BestFit
from .hint import TileDict, Stride, Hint
from ..graph import ComputeOpNode, ViewOpNode
from ..util.analysis import (
    build_output_axis_provenance,
    build_reduction_axis_provenance,
)

import copy
import math
import functools
import numpy as np
from queue import Queue, PriorityQueue
from typing import List


DEFAULT_N_THREADS = 128


class DefaultPolicy:
    def __init__(self, output_nodes, arch, force_fusion=False, flat_graph=None) -> None:
        self.arch = arch
        self.ordered_nodes = list(
            filter(
                lambda n: not n.is_placeholder() and not n.is_output(),
                find_topo_sort(output_nodes),
            )
        )

        self.output_nodes = []
        for node in self.ordered_nodes:
            is_topo_output = True
            for edge in node.outputs:
                if not edge.dst_node.is_output():
                    is_topo_output = False
            if is_topo_output:
                self.output_nodes.append(node)

        # pre-compute reduction-axis provenance once and reuse during tile evaluation.
        self.output_axis_provenance = build_output_axis_provenance(self.ordered_nodes)
        self.reduction_axis_provenance = build_reduction_axis_provenance(
            self.ordered_nodes, self.output_axis_provenance
        )
        (
            self.reduction_axis_overlap_pairs,
            self.reduction_axis_overlap_descendants,
        ) = self._build_reduction_axis_overlap_pairs()
        # TODO(david): this might change on a two pass approach
        # (check this commit when implementing two pass itgraph):
        # A descendant reduction axis that overlaps an ancestor reduction (they sum
        # over shared source tokens) must span its FULL extent: e.g. QKNorm's QK^T
        # matmul contracts over HD while an ancestor RMSNorm reduces sum-of-squares
        # over the same HD. Tiling such a contraction axis below full extent makes
        # the fused RMSNorm see only a partial-HD slice per step -> numerically
        # wrong. Pin these axes to full extent so only correct schedules generate.
        self._full_extent_reduction_axis_ids = {}
        for (_anc, desc), axis_pairs in self.reduction_axis_overlap_pairs.items():
            self._full_extent_reduction_axis_ids.setdefault(desc, set()).update(
                desc_ax for (_anc_ax, desc_ax) in axis_pairs
            )
        self.compatible_reduction_groups = self._get_compatible_reduction_groups()
        self.force_fusion = force_fusion
        self.flat_graph = flat_graph
        self.canonical_output_node = self._select_canonical_output_node()

    def _select_canonical_output_node(self):
        if not self.output_nodes:
            return None

        canonical = self.output_nodes[0]
        canonical_rank = len(canonical.get_space_dim())
        for node in self.output_nodes[1:]:
            rank = len(node.get_space_dim())
            if rank > canonical_rank:
                canonical = node
                canonical_rank = rank
        return canonical

    def _get_visible_output_shape(self, node):
        shape = list(node.get_space_dim())
        batch_flags = self._get_visible_batch_dim_flags(node)
        if len(batch_flags) != len(shape):
            return shape

        batch_extents = [
            int(extent) for extent, is_batch_dim in zip(shape, batch_flags) if is_batch_dim
        ]
        non_batch_extents = [
            int(extent) for extent, is_batch_dim in zip(shape, batch_flags) if not is_batch_dim
        ]
        if not batch_extents:
            return shape
        return [int(np.prod(batch_extents))] + non_batch_extents

    def _get_visible_output_batch_dim_flags(self, node):
        visible_shape = self._get_visible_output_shape(node)
        full_shape = list(node.get_space_dim())
        if len(visible_shape) == len(full_shape):
            return self._get_visible_batch_dim_flags(node)
        return [True] + [False] * (len(visible_shape) - 1)

    def _expand_output_tile_to_node_rank(self, node, tile):
        shape = list(node.get_space_dim())
        tile = list(tile)
        if len(tile) == len(shape):
            return tile

        batch_flags = self._get_visible_batch_dim_flags(node)
        if len(batch_flags) != len(shape):
            return tile

        batch_axes = [i for i, is_batch_dim in enumerate(batch_flags) if is_batch_dim]
        non_batch_axes = [i for i, is_batch_dim in enumerate(batch_flags) if not is_batch_dim]
        if not batch_axes:
            return tile
        if len(tile) != 1 + len(non_batch_axes):
            return tile
        if int(tile[0]) != 1:
            raise NotImplementedError(
                "Collapsed batch tiles greater than 1 are not supported."
            )

        expanded_tile = []
        non_batch_idx = 1
        for axis in range(len(shape)):
            if axis in batch_axes:
                expanded_tile.append(1)
            else:
                expanded_tile.append(int(tile[non_batch_idx]))
                non_batch_idx += 1
        return expanded_tile

    def _project_canonical_tile_to_output(self, canonical_tile, node):
        canonical_node = self.canonical_output_node
        if canonical_node is None or node == canonical_node:
            return list(canonical_tile)

        canonical_shape = list(canonical_node.get_space_dim())
        node_shape = list(node.get_space_dim())
        canonical_batch_flags = self._get_visible_batch_dim_flags(canonical_node)
        node_batch_flags = self._get_visible_batch_dim_flags(node)
        if len(canonical_batch_flags) != len(canonical_shape) or len(node_batch_flags) != len(
            node_shape
        ):
            raise NotImplementedError("Output batch flags do not match node rank.")

        canonical_batch_axes = [
            i for i, is_batch_dim in enumerate(canonical_batch_flags) if is_batch_dim
        ]
        node_batch_axes = [
            i for i, is_batch_dim in enumerate(node_batch_flags) if is_batch_dim
        ]
        if len(node_batch_axes) != len(canonical_batch_axes):
            raise NotImplementedError(
                "Same-launch multi-output currently requires matching batch-axis rank."
            )

        canonical_non_batch_axes = [
            i for i, is_batch_dim in enumerate(canonical_batch_flags) if not is_batch_dim
        ]
        node_non_batch_axes = [
            i for i, is_batch_dim in enumerate(node_batch_flags) if not is_batch_dim
        ]
        if len(node_non_batch_axes) > len(canonical_non_batch_axes):
            raise NotImplementedError(
                "Sibling outputs cannot have more non-batch axes than the canonical output."
            )

        projected_tile = [int(canonical_tile[axis]) for axis in canonical_batch_axes]
        for idx, axis in enumerate(node_non_batch_axes):
            canonical_axis = canonical_non_batch_axes[idx]
            if int(node_shape[axis]) != int(canonical_shape[canonical_axis]):
                raise NotImplementedError(
                    "Same-launch multi-output currently requires sibling outputs to drop only trailing non-batch axes."
                )
            projected_tile.append(int(canonical_tile[canonical_axis]))
        return projected_tile

    def _normalize_shape_to_source_rank(
        self, node, shape, src_shape, formal_shape=None
    ):
        shape = list(shape)
        src_shape = list(src_shape)
        if formal_shape is not None:
            formal_shape = [node.extent_wrapper(dim) for dim in formal_shape]

        if len(shape) > len(src_shape):
            if formal_shape is None or len(formal_shape) != len(shape):
                return None

            projected_shape = []
            src_idx = 0
            for dim_idx, (extent, formal_extent) in enumerate(zip(shape, formal_shape)):
                remaining_shape_dims = len(shape) - dim_idx
                remaining_src_dims = len(src_shape) - src_idx
                if remaining_shape_dims > remaining_src_dims and int(formal_extent) == 1:
                    continue
                if src_idx >= len(src_shape):
                    return None
                projected_shape.append(extent)
                src_idx += 1

            if len(projected_shape) != len(src_shape):
                return None
            shape = projected_shape
        elif len(shape) < len(src_shape):
            shape = [1] * (len(src_shape) - len(shape)) + shape

        return [node.extent_wrapper(j) for j in list(map(min, zip(shape, src_shape)))]

    def _normalize_input_shapes_to_source_rank(self, node, input_shapes):
        normalized_shapes = list(input_shapes)
        for i, edge in enumerate(node.inputs):
            if i >= len(normalized_shapes):
                break
            src_shape = edge.src_node.get_shape(edge.src_id)
            formal_shape = (
                node.input_buffers[i].shape if i < len(node.input_buffers) else src_shape
            )
            normalized_shape = self._normalize_shape_to_source_rank(
                node,
                normalized_shapes[i],
                src_shape,
                formal_shape=formal_shape,
            )
            if normalized_shape is not None:
                normalized_shapes[i] = normalized_shape
        return normalized_shapes

    def _get_output_tile_map(self, tile):
        tile_map = {}
        canonical_tile = self._expand_output_tile_to_node_rank(
            self.canonical_output_node, tile
        )
        for node in self.output_nodes:
            tile_map[node] = self._project_canonical_tile_to_output(
                canonical_tile, node
            )
        return tile_map

    def _get_visible_batch_dim_flags(self, node):
        shape = list(node.get_space_dim())
        batch_flags = list(getattr(node, "is_batch_dim", []))
        if not batch_flags:
            return [False for _ in shape]
        if len(batch_flags) == len(shape):
            return batch_flags

        n_non_batch_dims = sum(1 for is_batch_dim in batch_flags if not is_batch_dim)
        visible_batch_dims = max(0, min(len(shape), len(shape) - n_non_batch_dims))
        return [i < visible_batch_dims for i in range(len(shape))]

    def _propagate_inputs_full_rank(self, node, tile, rstep=None):
        if rstep is None:
            rstep = {}
        compensated_shapes = self._propagate_inputs_from_output_rank_collapse(
            node, tile, rstep
        )
        if compensated_shapes is not None:
            return compensated_shapes

        raw_base_shapes = list(node.propagate_inputs(tile, rstep=rstep))
        base_shapes = self._normalize_input_shapes_to_source_rank(node, raw_base_shapes)
        if not self._has_input_rank_drop(node, raw_base_shapes) and not self._has_input_rank_drop(
            node, base_shapes
        ):
            return base_shapes

        recovery_candidates = self._get_input_shape_recovery_candidates(
            node, tile, rstep
        )
        return self._merge_input_shape_recovery(
            node, raw_base_shapes, base_shapes, recovery_candidates
        )

    def _has_input_rank_drop(self, node, input_shapes):
        for i, edge in enumerate(node.inputs):
            if i >= len(input_shapes):
                return True
            src_shape = edge.src_node.get_shape(edge.src_id)
            if len(input_shapes[i]) != len(src_shape):
                return True
        return False

    def _get_input_shape_recovery_candidates(self, node, tile, rstep):
        candidates = []
        compensated_shapes = self._propagate_inputs_from_output_rank_collapse(
            node, tile, rstep
        )
        if compensated_shapes is not None:
            candidates.append(compensated_shapes)
        if node.reduction_block is not None:
            try:
                candidates.append(node.propagate_inputs_on_reduction(tile, rstep=rstep))
            except Exception:
                pass
        return candidates

    def _merge_input_shape_recovery(
        self, node, raw_base_shapes, base_shapes, recovery_candidates
    ):
        normalized_shapes = list(base_shapes)

        for i, edge in enumerate(node.inputs):
            src_shape = edge.src_node.get_shape(edge.src_id)
            raw_shape = raw_base_shapes[i] if i < len(raw_base_shapes) else None
            needs_recovery = raw_shape is None or len(raw_shape) != len(src_shape)
            if not needs_recovery and len(normalized_shapes[i]) == len(src_shape):
                continue

            for candidate_shapes in recovery_candidates:
                if candidate_shapes is None or i >= len(candidate_shapes):
                    continue
                candidate = self._normalize_shape_to_source_rank(
                    node,
                    candidate_shapes[i],
                    src_shape,
                    formal_shape=(
                        node.input_buffers[i].shape
                        if i < len(node.input_buffers)
                        else src_shape
                    ),
                )
                if candidate is None:
                    continue
                normalized_shapes[i] = candidate
                break

        return normalized_shapes

    def _propagate_inputs_from_output_rank_collapse(self, node, tile, rstep):
        if len(node.output_buffers) == 0:
            return None

        try:
            target_block = (
                node.reduction_block
                if node.reduction_block is not None
                else node.schedule_stages[0]
            )
            block_info = node.block_analyzer.get_block_info(target_block)
            spatial_extents = [
                int(node.extent_wrapper(it.dom.extent))
                for it in block_info.iters
                if it.kind == "S"
            ]
        except Exception:
            return None

        output_shape = list(node.get_shape(0))
        if len(tile) != len(output_shape) or len(spatial_extents) >= len(output_shape):
            return None

        compressed_tile = []
        logical_idx = 0
        for spatial_extent in spatial_extents:
            if logical_idx >= len(output_shape):
                return None

            grouped_extent = 1
            grouped_tile = 1
            while logical_idx < len(output_shape) and grouped_extent < spatial_extent:
                grouped_extent *= int(output_shape[logical_idx])
                grouped_tile *= int(tile[logical_idx])
                logical_idx += 1

            if grouped_extent != spatial_extent:
                return None
            compressed_tile.append(grouped_tile)

        if logical_idx != len(output_shape):
            return None

        compensated = []
        if node.reduction_block is not None:
            try:
                compressed_shapes = node.propagate_inputs_on_reduction(
                    compressed_tile, rstep=rstep
                )
            except Exception:
                return None
            if len(compressed_shapes) != len(node.input_buffers):
                return None
            for i, shape in enumerate(compressed_shapes):
                src_shape = node.inputs[i].src_node.get_shape(node.inputs[i].src_id)
                trimmed = self._normalize_shape_to_source_rank(
                    node,
                    shape,
                    src_shape,
                    formal_shape=node.input_buffers[i].shape,
                )
                if trimmed is None:
                    return None
                compensated.append(trimmed)
            return compensated

        targets = [t.name for t in node.args[: len(node.input_buffers)]]
        try:
            raw_shapes, _ = node.propagate(compressed_tile, rstep, targets)
        except Exception:
            return None

        for i, arg in enumerate(node.args[: len(node.input_buffers)]):
            src_shape = node.inputs[i].src_node.get_shape(node.inputs[i].src_id)
            trimmed = self._normalize_shape_to_source_rank(
                node,
                raw_shapes[arg.name],
                src_shape,
                formal_shape=node.input_buffers[i].shape,
            )
            if trimmed is None:
                return None
            compensated.append(trimmed)

        return compensated if len(compensated) == len(node.input_buffers) else None

    def compute_workload_per_item(self, output_tile) -> float:
        op_tile_map = self._get_output_tile_map(output_tile)
        compute = 0
        num_item = int(np.prod(output_tile))
        for node in reversed(self.ordered_nodes):
            tile = op_tile_map[node]
            dep = self._propagate_inputs_full_rank(node, tile)
            compute += int(np.prod(tile))
            for i, edge in enumerate(node.inputs):
                op_tile_map[edge.src_node] = dep[i]
        return float(compute / num_item)

    # get the minimum tile that could satisfy no redundancy computation
    def get_base_tile(self):
        out_node = self.canonical_output_node
        shape = self._get_visible_output_shape(out_node)
        visible_batch_flags = self._get_visible_output_batch_dim_flags(out_node)
        base_tile = [1 for _ in shape]
        wpi = self.compute_workload_per_item(base_tile)
        for dim, n in enumerate(shape):
            if visible_batch_flags[dim]:
                continue
            factors = [n]
            for factor in factors:
                if factor == base_tile[dim]:
                    continue
                tile = base_tile.copy()
                tile[dim] = factor
                new_wpi = self.compute_workload_per_item(tile)
                if new_wpi < wpi:
                    wpi, base_tile = new_wpi, tile
                else:
                    break

        return base_tile

    def get_node_reduce_step_candidates(self, node):
        # general idea : use factor first, since it does not require extra boundary check
        #                for large prime number, which is rare case, use power of 2.
        full_extent_axes = self._full_extent_reduction_axis_ids.get(node, set())
        results = {}
        for ax_id, k_iter in enumerate(node.raxis):
            if ax_id in full_extent_axes:
                # Overlaps an ancestor reduction (see __init__): must stay full so
                # the fused reduction is computed over the whole axis.
                results[k_iter.var.name] = [int(k_iter.dom.extent)]
                continue
            all_factors = get_all_round_factors(int(k_iter.dom.extent))
            if len(all_factors) == 2 and int(k_iter.dom.extent) > 64:
                all_factors = [1]
                while all_factors[-1] * 2 < int(k_iter.dom.extent):
                    all_factors.append(all_factors[-1] * 2)
            results[k_iter.var.name] = all_factors
        return results

    def score_block_size(self, n):
        num_wrap = (n + self.arch.warp_size - 1) // self.arch.warp_size
        r1 = max(num_wrap / self.arch.sm_partition, self.arch.sm_partition / num_wrap)
        r2 = (num_wrap * self.arch.warp_size - n) / n
        return (r1, r2)

    def _assign_reduce_step(self, node):
        if node.reduction_block is None:
            return {}

        raxis = node.raxis
        tile = [1] * len(node.get_space_dim())
        all_steps = self.get_node_reduce_step_candidates(node)

        def sim(a: int, b: int):
            return (2 * a * b) / (a * a + b * b)

        def _score(rstep_id):
            rstep = {k: all_steps[k][rstep_id[k]] for k in rstep_id}
            score = 0
            shape = self._propagate_inputs_full_rank(node, tile, rstep=rstep)
            for i, input_buffer in enumerate(node.input_buffers):
                read_transaction_elements = self.arch.transaction_size[1] // (
                    (node.get_buffer_dtype(input_buffer).bits + 7) // 8
                )
                score += sim(
                    int(coalesced_factor(shape[i], input_buffer.shape)),
                    read_transaction_elements,
                )
            return score

        def _enlarge(rstep_id):
            candidates = []
            candidates.append((rstep_id, _score(rstep_id)))
            for ax in rstep_id:
                if rstep_id[ax] + 1 == len(all_steps[ax]):
                    continue
                r = rstep_id.copy()
                r[ax] += 1
                candidates.append((r, _score(r)))
            best = max(candidates, key=lambda x: x[1])
            return best

        # enlarge rstep to ensure read is coaleased
        cur_rstep_id = {ax.var.name: 0 for ax in raxis}
        cur_score = _score(cur_rstep_id)
        while True:
            if cur_score == 0:
                break
            new_rstep, new_score = _enlarge(cur_rstep_id)
            if new_score <= cur_score:
                break
            else:
                cur_rstep_id, cur_score = new_rstep, new_score
        rstep = {k: all_steps[k][cur_rstep_id[k]] for k in cur_rstep_id}
        return rstep

    def _build_reduction_axis_overlap_pairs(self):
        from ..util import get_op_ancestors

        reduction_nodes = [
            n for n in self.ordered_nodes if n in self.reduction_axis_provenance
        ]
        overlap_pairs = {}
        overlap_descendants = {}
        memo_ancestors = {
            n: [
                a
                for a in get_op_ancestors(n, self.ordered_nodes)
                if a in reduction_nodes
            ]
            for n in reduction_nodes
        }

        for desc_node in reduction_nodes:
            desc_axis_tokens = self.reduction_axis_provenance[desc_node]
            for anc_node in memo_ancestors[desc_node]:
                anc_axis_tokens = self.reduction_axis_provenance[anc_node]
                shared_axis_pairs = set()
                for anc_ax, anc_tokens in enumerate(anc_axis_tokens):
                    if not anc_tokens:
                        continue
                    for desc_ax, desc_tokens in enumerate(desc_axis_tokens):
                        if not desc_tokens:
                            continue
                        if anc_tokens.intersection(desc_tokens):
                            shared_axis_pairs.add((anc_ax, desc_ax))

                if shared_axis_pairs:
                    overlap_pairs[(anc_node, desc_node)] = shared_axis_pairs
                    overlap_descendants.setdefault(anc_node, set()).add(desc_node)

        return overlap_pairs, overlap_descendants

    def _get_active_reduction_axes(self, node, rstep):
        active_axes = set()
        if not node.raxis:
            return active_axes
        for ax_id, raxis in enumerate(node.raxis):
            step = int(rstep.get(raxis.var.name, int(raxis.dom.extent)))
            if step < int(raxis.dom.extent):
                active_axes.add(ax_id)
        return active_axes

    def _compute_memory_traffic(self, output_tile, rstep_map):
        op_tile_map = self._get_output_tile_map(output_tile)
        op_itile_map = op_tile_map.copy()
        traffic = 0
        irregular_fusion = False
        active_desc_reduction_axes = {}
        for node in reversed(self.ordered_nodes):
            # if we are not forcing fusion check whether we have
            # reduction axis that overlaps with consumers.
            # this is basically a way to prune dependent nodes
            # that iterate over the same axis (e.g., vanilla attention).
            if node.raxis and not self.force_fusion:
                node_active_axes = self._get_active_reduction_axes(
                    node, rstep_map.get(node, {})
                )
                if node_active_axes:
                    for desc_node in self.reduction_axis_overlap_descendants.get(
                        node, set()
                    ):
                        desc_active_axes = active_desc_reduction_axes.get(desc_node)
                        if not desc_active_axes:
                            continue
                        shared_axis_pairs = self.reduction_axis_overlap_pairs.get(
                            (node, desc_node), set()
                        )
                        for anc_ax, desc_ax in shared_axis_pairs:
                            if (
                                anc_ax in node_active_axes
                                and desc_ax in desc_active_axes
                            ):
                                irregular_fusion = True
                                break
                        if irregular_fusion:
                            break

                    active_desc_reduction_axes[node] = node_active_axes

            tile, itile = op_tile_map[node], op_itile_map[node]
            input_shapes = self._propagate_inputs_full_rank(node, tile)
            input_iter_shapes = self._propagate_inputs_full_rank(
                node, itile, rstep=rstep_map[node]
            )
            output_shapes = node.propagate_outputs(tile)
            for i, edge in enumerate(node.inputs):
                op_tile_map[edge.src_node], op_itile_map[edge.src_node] = (
                    input_shapes[i],
                    input_iter_shapes[i],
                )
                if edge.src_node.is_placeholder():
                    nbytes = (edge.src_node.get_dtype().bits + 7) // 8
                    read_transaction_elements = self.arch.transaction_size[1] // nbytes
                    traffic += (
                        coalesced_tensor_shape(
                            input_shapes[i],
                            edge.src_node.get_shape(),
                            read_transaction_elements,
                        )
                        * nbytes
                    )
                # TODO (david): add extra traffic whenever we need to load from higher-level memory again?
                # e.g., nodes out of reduction block? this may happen when force_fusion is set to True.
            for edge in node.outputs:
                if edge.dst_node.is_output():
                    nbytes = (edge.src_node.get_dtype().bits + 7) // 8
                    write_transaction_elements = self.arch.transaction_size[0] // nbytes
                    traffic += (
                        coalesced_tensor_shape(
                            output_shapes[edge.src_id],
                            node.get_shape(edge.src_id),
                            write_transaction_elements,
                        )
                        * nbytes
                    )

        return traffic, op_tile_map, op_itile_map, irregular_fusion

    def compute_node_stride_map(self, node, td: TileDict):
        output_strides = {
            int(i + len(node.input_buffers)): Stride()
            for i, _ in enumerate(node.output_buffers)
        }
        tensor_strides = {}
        return output_strides, tensor_strides

    def _compute_stride_map(self, td: TileDict):
        output_strides_map = {}
        tensor_strides_map = {}
        for node in self.ordered_nodes:
            output_strides_map[node], tensor_strides_map[node] = (
                self.compute_node_stride_map(node, td)
            )
        td.output_strides_map, td.tensor_strides_map = (
            output_strides_map,
            tensor_strides_map,
        )

    def infer_node_smem_usage(self, td, node):
        # view ops are alias-only and should not contribute internal shared-memory footprint.
        if isinstance(node, ViewOpNode):
            return 0, []
        return node.footprint(
            td.get_itile(node), td.get_rstep(node), td.tensor_strides_map[node]
        )

    def _compute_shared_memory_usage(self, td):
        self._compute_stride_map(td)
        allocator = BestFit()
        block_map = {}
        processed = set()
        cached_tensors_map = {}

        def resolve_materialized_source(node, out_id):
            visited = set()
            src_node, src_id = node, out_id
            while isinstance(src_node, ViewOpNode):
                key = (src_node, src_id)
                if key in visited:
                    break
                visited.add(key)
                if not src_node.inputs:
                    break
                in_edge = src_node.inputs[0]
                src_node, src_id = in_edge.src_node, in_edge.src_id
            return src_node, src_id

        def has_unprocessed_consumer(node, out_id):
            for edge in node.outputs:
                if edge.src_id != out_id:
                    continue
                dst = edge.dst_node
                if isinstance(dst, ViewOpNode):
                    if len(dst.outputs) == 0:
                        if dst not in processed:
                            return True
                        continue
                    view_out_ids = {out_edge.src_id for out_edge in dst.outputs}
                    for view_out_id in view_out_ids:
                        if has_unprocessed_consumer(dst, view_out_id):
                            return True
                    continue
                if dst not in processed:
                    return True
            return False

        def can_free(node, out_id):
            return not has_unprocessed_consumer(node, out_id)

        for node in self.ordered_nodes:
            node_internal_bytes, cached_tensors_map[node] = self.infer_node_smem_usage(
                td, node
            )
            block = allocator.malloc(node_internal_bytes)
            allocator.free(block)
            # free inputs
            processed.add(node)
            for edge in node.inputs:
                src_node, src_id = resolve_materialized_source(
                    edge.src_node, edge.src_id
                )
                if src_node.is_placeholder():
                    continue
                if can_free(src_node, src_id):
                    block = block_map.pop((src_node, src_id), None)
                    if block is not None:
                        allocator.free(block)
            # alloc outputs
            for edge in node.outputs:
                if isinstance(node, ViewOpNode):
                    continue
                if (
                    not edge.dst_node.is_output()
                    and (node, edge.src_id) not in block_map
                ):
                    dtype_bytes = (node.get_dtype(edge.src_id).bits + 7) // 8
                    stride = td.output_strides_map[node][len(node.inputs) + edge.src_id]
                    output_elem = stride.compute_elements_from_shape(td.get_itile(node))
                    block_map[(node, edge.src_id)] = allocator.malloc(
                        output_elem * dtype_bytes
                    )

        assert len(block_map) == 0
        return allocator.limit, cached_tensors_map

    def compute_tile_dict(self, output_tile: list[int], rstep_map):
        td = TileDict(output_tile, rstep_map, flat_graph=self.flat_graph)
        td.n_threads = DEFAULT_N_THREADS
        td.traffic, td.tile_map, td.itile_map, irregular_fusion = (
            self._compute_memory_traffic(output_tile, rstep_map)
        )
        td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)
        if td.smem_cost > self.arch.smem_cap or (
            irregular_fusion and not self.force_fusion
        ):
            td.valid = False
            return td
        output_shape = self._get_visible_output_shape(self.canonical_output_node)
        td.grid_size = int(
            np.prod([(y + x - 1) // x for x, y in zip(output_tile, output_shape)])
        )
        # estimated reg usage
        reg_usage = int(
            2
            * max(
                [
                    np.prod(td.get_itile(node)) * node.get_dtype().bits / 32
                    for node in self.ordered_nodes
                ]
            )
        )
        if reg_usage > self.arch.reg_cap:
            td.valid = False
            return td
        td.block_per_SM = min(
            self.arch.max_smem_usage // max(td.smem_cost, 1),
            self.arch.reg_cap // max(reg_usage, 1),
            self.arch.sm_partition,
        )
        td.num_wave = int(
            np.ceil(td.grid_size / int(td.block_per_SM * self.arch.compute_max_core))
        )
        return td

    def check_tile_shape_isvalid(self, td: TileDict) -> bool:
        for node in self.ordered_nodes:
            if np.prod(td.get_tile(node)) == 0:
                return False
            node_grid_size = np.prod(
                [
                    (y + x - 1) // x
                    for x, y in zip(td.get_tile(node), node.get_space_dim())
                ]
            )
            if node_grid_size != td.grid_size:
                return False
            if (
                hasattr(node, "reduce_op")
                and node.reduce_op is not None
                and len(node.reduce_op.axis) == len(td.output_tile)
            ):
                for i, tile_extent in enumerate(td.output_tile):
                    if node.reduce_op.axis[i].dom.extent % tile_extent:
                        return False

        return True

    def recommend_block_size(self, td: TileDict) -> List[int]:
        node_space_sizes = [
            int(np.prod(td.get_tile(node))) for node in self.ordered_nodes
        ]
        max_block_size = functools.reduce(math.gcd, node_space_sizes)

        if (
            max_block_size < self.arch.warp_size * self.arch.sm_partition
            and max_block_size == min(node_space_sizes)
        ):
            node_reduce_sizes = [
                int(np.prod(list(td.get_rstep(node).values())))
                for node in self.ordered_nodes
            ]
            total_sizes = [x * y for x, y in zip(node_space_sizes, node_reduce_sizes)]
            max_possible_size = functools.reduce(math.gcd, total_sizes)
            possible_block_sizes = list(
                filter(
                    lambda x: x % max_block_size == 0 and x <= 1024,
                    get_all_round_factors(max_possible_size),
                )
            )
            possible_block_sizes = list(
                filter(  # either be a factor of space or cover fully cover the space
                    lambda x: all([x % s == 0 or s % x == 0 for s in node_space_sizes]),
                    possible_block_sizes,
                )
            )
            factor_ordered = sorted(possible_block_sizes, key=self.score_block_size)
            return factor_ordered
        else:
            possible_block_sizes = get_all_round_factors(max_block_size)
            possible_block_sizes = list(
                filter(lambda x: x <= 1024, possible_block_sizes)
            )
        factor_ordered = sorted(possible_block_sizes, key=self.score_block_size)
        return factor_ordered

    def assign_block_size(self, td: TileDict, topk=1):
        block_size_ordered = self.recommend_block_size(td)
        for block_size in block_size_ordered:
            result = {}
            failed = False
            for node in self.ordered_nodes:
                result[node] = self._assign_block_size(node, td, block_size)
                if result[node] is None:
                    failed = True
                    break
            if failed:
                continue
            else:
                yield result
                topk -= 1
                if topk == 0:
                    break

    def _assign_block_size(self, node: ComputeOpNode, td: TileDict, block_size: int):
        tile, rsteps = td.get_tile(node), td.get_rstep(node)
        factors = factorize(block_size)
        cur_threads = [1 for _ in tile]
        reduce_thread = {k: 1 for k in rsteps}
        ndim = len(tile)

        def _score(node, thread):  # small is better
            score = 0
            block_tile = [int(np.ceil(tile[i] / thread[i])) for i in range(ndim)]
            shape = self._propagate_inputs_full_rank(node, block_tile)
            for i, _ in enumerate(node.input_buffers):
                score += np.prod(shape[i]) / self.arch.bandwidth[1]
            for buffer in node.output_buffers:
                if node.is_output():  # TODO: check if this is correct
                    score += (
                        coalesced_tensor_shape(thread, buffer.shape, 8)
                        / self.arch.bandwidth[0]
                    )
            return score

        for factor in reversed(factors):
            score_map = {}
            for i in range(ndim):
                if cur_threads[i] >= tile[i]:
                    continue
                if (tile[i] % (cur_threads[i] * factor)) != 0:
                    continue
                cur_threads[i] *= factor
                score_map[i] = (_score(node, cur_threads), i)
                cur_threads[i] //= factor
            if len(score_map) > 0:
                # assign to space axis
                dim_order = sorted(score_map.keys(), key=lambda x: score_map[x])
                cur_threads[dim_order[0]] *= factor
            else:
                # assign to reduce axis
                target_ax = None
                for ax, ax_len in reversed(list(rsteps.items())):
                    if ax_len % (reduce_thread[ax] * factor) == 0:
                        target_ax = ax
                        break
                assert target_ax
                reduce_thread[target_ax] *= factor

        codegen_dict = Hint()
        codegen_dict.block = tile
        codegen_dict.thread = cur_threads
        codegen_dict.rstep = [rsteps[ax.var.name] for ax in node.raxis]
        codegen_dict.reduce_thread = [reduce_thread[ax.var.name] for ax in node.raxis]
        codegen_dict.cached_tensors = td.cached_tensors_map[node]

        if (
            node.get_dtype().bits == 16
        ):  # set step=2 for 16bit case to ensure coalesced access
            codegen_dict._step = [1 for _ in range(ndim)]
            for i in reversed(range(ndim)):
                if codegen_dict.block[i] // codegen_dict.thread[i] % 2 == 0:
                    codegen_dict._step[i] = 2
                    break
        elif (
            node.get_dtype().bits == 8
        ):  # set step=4 for 8bit case to ensure coalesced access
            codegen_dict._step = [1 for _ in range(ndim)]
            for i in reversed(range(ndim)):
                if codegen_dict.block[i] // codegen_dict.thread[i] % 4 == 0:
                    codegen_dict._step[i] = 4
                    break
        # Plan vectorize
        codegen_dict.vectorize = self._plan_vectorize(node, td, block_size)
        codegen_dict.arch = self.arch
        codegen_dict.opt_shapes = node.get_tag("opt_shapes")
        return codegen_dict

    def _plan_vectorize(self, node: ComputeOpNode, td: TileDict, block_size: int):
        def is_cont(shape, vec):
            if len(shape) == 0:
                return vec == 1
            last = shape[-1]
            if last == 1:
                return is_cont(shape[0:-1], vec // last)
            else:
                return last % vec == 0

        def is_shape_aligned(shape, factor):
            return int(np.prod(shape)) % factor == 0

        def is_type_allowed(dtype, vec):
            return dtype.bits * vec <= 128

        vectorize_sizes = [16, 8, 4, 2]
        dtypes = node.get_reduce_inputs_dtype()
        shapes = node.propagate_reduction_inputs(td.get_tile(node), td.get_rstep(node))
        vectorize_result = {}
        for tensor, shape in shapes.items():
            for v in vectorize_sizes:
                if (
                    is_shape_aligned(shape, block_size * v)
                    and is_cont(shape, v)
                    and is_type_allowed(dtypes[tensor], v)
                ):
                    vectorize_result[tensor] = v
                    break
        return vectorize_result

    def dfs_smem_tile(self, init_tile, rstep_map):
        visible_shape = self._get_visible_output_shape(self.canonical_output_node)
        _steps = [get_all_round_factors(n) for n in visible_shape]
        steps = [step[step.index(t) :] for step, t in zip(_steps, init_tile)]
        visible_batch_flags = self._get_visible_output_batch_dim_flags(
            self.canonical_output_node
        )
        steps = [
            (
                [
                    1,
                ]
                if visible_batch_flags[i]
                else step
            )
            for i, step in enumerate(steps)
        ]
        for i in range(len(steps)):
            added = list(
                filter(
                    lambda s: s < steps[i][-1]
                    and s > steps[i][0]
                    and s not in steps[i],
                    [2, 4, 8, 16, 32],
                )
            )
            steps[i].extend(added)
            steps[i] = sorted(steps[i])
        visited_tiles = {}
        queue = Queue()

        def add_to_queue(tile):
            if tuple(tile) in visited_tiles:
                return
            td = self.compute_tile_dict(tile, rstep_map)
            visited_tiles[tuple(tile)] = td
            if td.valid:
                queue.put(tile)

        add_to_queue(init_tile)
        while not queue.empty():
            tile = queue.get()
            dim_ids = [step.index(t) for step, t in zip(steps, tile)]
            for i in reversed(range(len(dim_ids))):
                if dim_ids[i] + 1 < len(steps[i]):
                    new_tile = tile.copy()
                    new_tile[i] = steps[i][dim_ids[i] + 1]
                    add_to_queue(new_tile)

        visited_tiles = filter(lambda td: td.valid, visited_tiles.values())
        return list(visited_tiles)

    def expand_iterative_tiles(self, base_tiles, base_rstep_map):
        def prio(td):
            return (td.traffic + 1) * td.num_wave

        tile_counter = 0

        def add_to_queue(tile, rstep_map):
            nonlocal tile_counter
            if (
                tuple(tile),
                tuple((k, tuple(v.items())) for k, v in rstep_map.items()),
            ) in visited_tiles:
                return
            td = self.compute_tile_dict(tile, rstep_map)
            visited_tiles[
                (
                    tuple(tile),
                    tuple((k, tuple(v.items())) for k, v in rstep_map.items()),
                )
            ] = td
            if td.valid:
                queue.put((prio(td), tile_counter, (tile, rstep_map)))
                tile_counter += 1

        visited_tiles = {}
        queue = PriorityQueue()
        for base_td in base_tiles:
            _steps = {
                n: list(self.get_node_reduce_step_candidates(n).values())[0]
                for n in base_rstep_map
                if n.raxis
            }
            steps = {
                n: step[step.index(base_rstep_map[n][n.raxis[0].var.name]) :]
                for n, step in _steps.items()
            }

            add_to_queue(base_td.output_tile, base_rstep_map)
            while not queue.empty():
                _, _, (tile, rstep_map) = queue.get()
                rdim_ids = {
                    n: [step.index(rstep_map[n][ax.var.name]) for ax in n.raxis]
                    for n, step in steps.items()
                    if n in rstep_map and n.raxis
                }

                for group in self.compatible_reduction_groups:
                    node = group[0]
                    assert (
                        node in rdim_ids
                    ), "Node with no reduction axis in compatible reduction group"
                    dim_id = rdim_ids[node]

                    for i in reversed(range(len(dim_id))):
                        if dim_id[i] + 1 < len(steps[node]):
                            new_rstep_map = rstep_map.copy()

                            for member in group:
                                if member not in rdim_ids:
                                    continue
                                member_dim_id = rdim_ids[member]
                                if (
                                    i < len(member_dim_id)
                                    and member_dim_id[i] + 1 < len(steps[member])
                                ):
                                    new_rstep = new_rstep_map[member].copy()
                                    new_rstep[member.raxis[i].var.name] = steps[member][
                                        member_dim_id[i] + 1
                                    ]
                                    new_rstep_map[member] = new_rstep

                            add_to_queue(tile, new_rstep_map)

        visited_tiles = filter(lambda td: td.valid, visited_tiles.values())
        sorted_tiles = sorted(visited_tiles, key=lambda td: prio(td))
        return sorted_tiles

    def propose_iterative_tiles(self, base_tile):
        # 1. find minimum reduction step for each reduction node
        base_iter_map = {
            node: self._assign_reduce_step(node) for node in self.ordered_nodes
        }

        # 2. find all possible output tiles that work with the minumum reduction step
        base_iter_tiles = self.dfs_smem_tile(base_tile, base_iter_map)

        # 3. for each output tile, try to expand the reduction steps
        smem_tile_candidates = self.expand_iterative_tiles(
            base_iter_tiles, base_iter_map
        )

        return smem_tile_candidates

    def emit_config(self, topk):
        base_tile = self.get_base_tile()
        if base_tile is None:
            return []
        smem_tile_candidates = self.propose_iterative_tiles(base_tile)
        smem_tile_candidates = self.get_num_threads(smem_tile_candidates)

        return smem_tile_candidates[:topk]

    def get_num_threads(self, candidates):
        for candidate in candidates:
            candidate.n_threads = DEFAULT_N_THREADS
        return candidates

    def _get_compatible_reduction_groups(self):
        from ..util import get_op_ancestors

        reduction_nodes = [n for n in self.ordered_nodes if n.raxis]

        memo_ancestors = {n: set(get_op_ancestors(n)) for n in reduction_nodes}

        def get_descendant_reduction_nodes(node):
            from ..util import gather_all_nodes

            descendants = gather_all_nodes(
                node, ascending=False, descending=True, change_name=False
            )
            return {d for d in descendants if d in reduction_nodes and d != node}

        memo_descendant_reductions = {
            n: get_descendant_reduction_nodes(n) for n in reduction_nodes
        }

        def are_compatible(node1, node2):
            # 1. intersecting ancestor path
            if not memo_ancestors[node1].intersection(memo_ancestors[node2]):
                return False

            # 2. descendant reduction nodes must match
            if memo_descendant_reductions[node1] != memo_descendant_reductions[node2]:
                return False

            return True

        adj = {n: [] for n in reduction_nodes}
        for i in range(len(reduction_nodes)):
            for j in range(i + 1, len(reduction_nodes)):
                node1, node2 = reduction_nodes[i], reduction_nodes[j]
                if are_compatible(node1, node2):
                    adj[node1].append(node2)
                    adj[node2].append(node1)

        groups = []
        visited = set()
        for node in reduction_nodes:
            if node not in visited:
                group = []
                q = [node]
                visited.add(node)
                head = 0
                while head < len(q):
                    curr = q[head]
                    head += 1
                    group.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
                groups.append(group)

        return groups


class TCPolicy(DefaultPolicy):
    def __init__(self, output_nodes, arch, force_fusion=False, flat_graph=None):
        super().__init__(
            output_nodes, arch, force_fusion=force_fusion, flat_graph=flat_graph
        )
        self.wmma = [64, 32, 32]  # M, N, K

    def _assign_reduce_step(self, node):
        if not node.get_tag("tensorcore_config"):
            return super()._assign_reduce_step(node)
        # get reduce input size
        target_transaction = self.arch.transaction_size[0] * 2
        # 512 bytes // type bits
        reduce_input_dtype = node.get_buffer_dtype(
            node.block_analyzer.get_input_buffers(node.reduction_block)[0]
        )
        basic = (target_transaction * 8) // reduce_input_dtype.bits

        full_extent_axes = self._full_extent_reduction_axis_ids.get(node, set())
        result = {}
        for ax_id, iter_info in enumerate(node.raxis):
            iter_name = iter_info.var.name
            iter_dom = iter_info.dom.extent
            if ax_id in full_extent_axes:
                # Overlaps an ancestor reduction (see __init__): seed at full so
                # the fused reduction is correct (never a partial-axis slice).
                result[iter_name] = int(iter_dom)
            elif iter_dom % 16 > 0:
                result[iter_name] = (
                    16 if iter_dom < basic else basic
                )  # for the case of padding
            elif iter_dom % basic == 0:
                result[iter_name] = basic
            else:
                return super()._assign_reduce_step(node)
        return result

    def get_base_tile(self):
        out_node = self.canonical_output_node
        shape = self._get_visible_output_shape(out_node)
        visible_batch_flags = self._get_visible_output_batch_dim_flags(out_node)
        visible_batch_dims = sum(1 for is_batch_dim in visible_batch_flags if is_batch_dim)
        base_tile = [
            (
                1
                if i < visible_batch_dims
                else min(self.wmma[i - visible_batch_dims], shape[i])
            )
            for i in range(len(shape))
        ]
        return base_tile

    # Thread counts (i.e. warp counts, warp_size=32) tried per tile candidate.
    # 128/256 (4/8 warps) are the primary configs for every tensorcore graph;
    # 64/32 (2/1 warps) are added ONLY for candidates that contain a matmul whose
    # output tile is too small to place 4 warps (see below), so ordinary
    # operators pay no extra compile time.
    PRIMARY_N_THREADS = (DEFAULT_N_THREADS, 256)
    LOW_N_THREADS = (64, 32)

    def _requires_low_thread_counts(self, candidate):
        """Does this candidate have a matmul tile that 128 threads cannot tile?

        A fused kernel uses a single thread count for all its matmuls, and
        tilelang's ``makeGemmFragmentC`` requires ``warp_m % 16 == 0`` and
        ``warp_n % 8 == 0`` (``warp_m = block_m / m_warp``, ``warp_n = block_n /
        n_warp``). A matmul output tile ``(block_m, block_n)`` therefore admits at
        most ``(block_m // 16) * (block_n // 8)`` warps. When that ceiling is below
        4, 128 threads (4 warps) and 256 (8 warps) both fail to compile and only
        1-2 warp (32/64-thread) configs can schedule -- the case hit by LoRA's
        ``(16, 16)`` down-projection ``X@A`` tile (1 * 2 = 2 warps). Ordinary
        matmul tiles (e.g. ``(64, 32)`` -> 16 warps) never trip this, so they keep
        just the 128/256 configs.
        """
        for node in self.ordered_nodes:
            if not node.get_tag("tensorcore_config"):
                continue
            tile = candidate.get_tile(node)
            if len(tile) < 2:
                continue
            block_m, block_n = int(tile[-2]), int(tile[-1])
            if (block_m // 16) * (block_n // 8) < 4:
                return True
        return False

    def get_num_threads(self, candidates):
        expanded_candidates = []
        for candidate in candidates:
            n_threads_options = list(self.PRIMARY_N_THREADS)
            if self._requires_low_thread_counts(candidate):
                n_threads_options.extend(self.LOW_N_THREADS)
            for n_threads in n_threads_options:
                expanded = copy.copy(candidate)
                expanded.n_threads = n_threads
                expanded_candidates.append(expanded)
        return expanded_candidates
