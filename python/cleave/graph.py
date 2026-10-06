from dataclasses import dataclass, field
from typing import Callable, List, Union, Dict, Tuple, Optional

from tilelang.carver.roller.node import (
    Edge,
    Node,
    get_analyzer_by_tir,
    BlockAnalyzer,
    analysis,
    get_reduction_blocks,
    pre_order_traverse,
)
from tilelang.carver.utils import get_tensorized_func_and_tags
from tilelang.carver.arch import get_arch

import tvm
from tvm import tirx
from tvm import DataType
from tvm.s_tir import Schedule
from tvm.s_tir.schedule import SBlockRV
from tvm.tirx import PrimFunc

import functools
import textwrap
import numpy as np

ARCH = get_arch("cuda")


class Iterator:
    def __init__(
        self,
        offset: Callable[[str], tuple[str]],
        shape: tuple[int],
        tile_size: tuple[int],
    ):
        self.offset = offset  # offset('i') -> index expression tuple
        self.shape = shape
        self.tile_size = tile_size

    def __repr__(self):
        return f"Iterator(offset={self.offset}, shape={self.shape}, tile_size={self.tile_size})"


@dataclass
class ScalarScan:
    """One prologue statement: a small buffer plus a device-side scan that fills it.

    Emitted inside ``with T.Kernel(...)`` before the ``*_iter`` call, so both ``smap``
    expressions and ``block_guard`` may read ``buffer``. ``inputs`` are graph-level
    tensor nodes, resolved to this kernel's own parameter names at emission;
    ``scalar_args`` is a callable over the grid index names returning expression
    strings, exactly like an ``smap`` entry.

    This is how a work list that used to be precomputed on the host becomes a device
    scan over the operators' own CSR ``indptr`` inputs.
    """

    op: "PrologueOp"  # noqa: F821 - cleave.ops.prologue.PrologueOp, imported lazily
    buffer: str
    size: int
    dtype: str = "int32"
    scope: str = "l"
    inputs: Tuple[Node, ...] = ()
    scalar_args: Optional[Callable[..., Tuple[str, ...]]] = None


@dataclass
class Graph:
    nodes: List[Node]

    # tensor mapping for input and output nodes, used for mapping both
    # threadBlock index and iteration number to tensor regions.
    tmap: Union[List[dict[Node, Iterator]], dict[Node, Iterator]] = field(
        default_factory=dict
    )
    extent: Union[int, Tuple[int, ...], None] = (
        None  # number of iterations, e.g., number of threadBlocks, for loop extent
    )
    tinfo_map: dict[Node, tuple[str, str]] = field(
        default_factory=dict
    )  # (tensor name, 'g'/'s')
    tlayout_map: dict[Node, str] = field(
        default_factory=dict
    )
    args: List[Node] = field(default_factory=list)  # input and output nodes
    accum_nodes: List[Node] = field(
        default_factory=list
    )  # nodes that require accumulation
    binding: dict[str, Node] = field(
        default_factory=dict
    )  # map PlaceHolder/Output nodes in the supergraph to nodes in the subgraph
    locked: bool = False  # whether the graph is locked for further modification
    sargs: List[Tuple[str, str]] = field(
        default_factory=list
    )  # scalar (non-tensor) parameters as (name, dtype)
    smap: dict[str, Callable[..., str]] = field(
        default_factory=dict
    )  # scalar name -> expression in terms of this graph's iteration vars, and of any
    # buffer a `prologue` scan filled, since those are emitted first
    iter_sargs: Optional[List[Tuple[str, str]]] = (
        None  # subset of sargs threaded into the *_iter macro; None -> all of sargs
    )
    loop_kind: str = "pipelined"  # 'pipelined' | 'serial' for the iterative loop
    loop_trip: Optional[str] = None  # runtime trip-count expression; defaults to extent
    loop_stages: Optional[int] = None  # pipeline depth; defaults to min(3, extent).
    # Only worth setting when the loop's inputs are AFFINE slices: tilelang double-buffers
    # those, and a gathered (non-affine) source gets reordered without being buffered,
    # which is a correctness bug rather than a missed optimisation.
    prefetch_nodes: List[Node] = field(
        default_factory=list
    )  # nodes of an iterative graph whose global read is issued one iteration ahead.
    # tilelang's own pipeliner cannot help a gather: it double-buffers bulk copies, and a
    # gather is an indexed compute loop, so `loop_stages` leaves the load latency exposed
    # in front of the tile's first consumer. A prefetched node instead keeps a fragment
    # alive across iterations: the tile the previous iteration fetched is copied into the
    # node's own (shared) buffer at the top of the body, then the NEXT tile's global read
    # is issued before any of this tile's math. One buffer, not two, because the shared
    # tile is the second version; and no barrier of its own, because the fragment is
    # thread-local. Every scalar the node reads is re-evaluated at `i + 1` through `smap`,
    # so a node is prefetchable exactly when its address depends on nothing but the loop
    # index and the graph's scalars.
    prefetch_guard: Optional[str] = (
        None  # condition, in terms of `iter_sargs`, under which the look-ahead read runs.
        # Without one the last iteration reads one tile past the end -- in bounds whenever
        # the node clamps to its buffer, but pointless traffic. Iterative graphs only.
    )
    window_nodes: List[Node] = field(
        default_factory=list
    )  # nodes of an iterative graph whose buffer is hoisted out of the loop and refilled
    # only on the iterations where `window_guard` holds, the rest reading what an earlier
    # iteration left there. For a tile whose contents cover several iterations' worth of
    # work -- a page-table window, say -- this turns a per-iteration staging cost into a
    # per-window one, without the second loop tier that expressing the window as its own
    # graph would need. The node's consumers index into it with a scalar giving the offset
    # of their own iteration inside the window.
    window_guard: Optional[str] = (
        None  # condition, in terms of `iter_sargs`, marking the iterations that refill the
        # `window_nodes`. Required whenever there are any: without it every iteration would
        # refill, which is what hoisting is meant to avoid.
    )
    dyn_dims: List[str] = field(
        default_factory=list
    )  # symbolic extent names; each must appear bare in some tensor shape so the
    # runtime can infer it, after which it may be used in extent/grid expressions
    pass_configs: Dict[str, bool] = field(
        default_factory=dict
    )
    region_args: List[Node] = field(
        default_factory=list
    )  # args whose tile is handed down as a rank-reduced BufferRegion (a scalar index for
    # each unit extent) instead of an `_S()` view. Only legal when no tier re-indexes the
    # argument -- it may be forwarded and copied, nothing else. Worth it because tilelang
    # can bounds-predicate and bulk-copy a real BufferRegion, but sees an `_S()` view as a
    # buffer of exactly the slice's extent: no predication (so an overrunning slice is an
    # illegal access rather than a masked read) and no bulk copy.
    scalar_params: List[Tuple[str, str]] = field(
        default_factory=list
    )  # scalar (non-tensor) *kernel* inputs as (name, dtype), e.g. sm_scale. Distinct
    # from `sargs`, which are per-block values this graph derives and threads into its
    # own `*_iter` macro: these are declared inputs of the operator, emitted in the main
    # prim_func signature between the tensor inputs and the outputs, and referenced from
    # `smap` expressions by name. Main graph only.
    prologue: List[ScalarScan] = field(
        default_factory=list
    )  # statements emitted inside `with T.Kernel(...)` before the *_iter call. Main
    # graph only, so that `smap` and `block_guard` can read the scanned buffers.
    rasterization: Optional[int] = (
        None  # per-kernel L2 swizzle width; None defers to the codegen's default. Set to 1
        # to emit no `T.use_swizzle` at all. Whether it pays is an operator-by-operator
        # question and MEASURED both ways: swizzling costs a per-CTA index remap and buys
        # L2 locality between neighbouring blocks, so it wins where blocks share KV pages
        # and loses where the grid is small or the blocks are already spread over the cache.
        # Main graph only -- an inner tier has no grid to rasterise.
    )
    work_guard: Optional[Callable[..., str]] = (
        None  # inner predicate selecting the working body over `else_nodes`. Emitted
        # nested inside `block_guard`, so a slot that is not part of the grid at all is
        # still skipped entirely rather than filled.
    )
    else_nodes: List[Node] = field(
        default_factory=list
    )  # ops emitted in `work_guard`'s else branch, each a single macro call writing its own
    # tile of an already-declared output. These are NOT in `nodes`: they are an alternative
    # way to produce the same answer, not part of the dataflow. Used for the rows an
    # attention kernel has no work for, where running the full epilogue to arrive at
    # 0 / -inf costs several times what storing it directly does.
    block_guard: Optional[Callable[..., str]] = (
        None  # whole-body activation predicate for one grid slot, called with the grid
        # index names. Main graph only. CONTRACT: (a) the predicate must be CTA-uniform,
        # because the guarded region contains T.copy/T.gemm/T.reduce_* and therefore
        # implicit __syncthreads(); (b) the union of active slots must cover every
        # declared output element -- nothing else checks that.
    )


class PlaceHolderNode(Node):
    def __init__(self, name=""):
        super().__init__(name="PlaceHolder_" + name)

    def is_placeholder(self):
        return True

    def get_ir(self) -> str:
        return "placeholder"

    def __hash__(self):
        return hash((self.name, tuple(e.dst_node for e in self.outputs)))

    def __eq__(self, other):
        if not isinstance(other, PlaceHolderNode):
            return False
        if self.name != other.name:
            return False
        if len(self.outputs) != len(other.outputs):
            return False
        return all(
            e1.dst_node == e2.dst_node for e1, e2 in zip(self.outputs, other.outputs)
        )


class PrimFuncNode(Node):
    def __init__(
        self,
        prim_func: PrimFunc,
        tags: Optional[Dict] = None,
        name: str = "PrimFuncNode",
    ) -> None:
        super().__init__(tags, name=name)
        self.prim_func = self._specialize_func(prim_func)
        self.sch: Schedule = Schedule(self.prim_func)
        self.block_analyzer: BlockAnalyzer = BlockAnalyzer(self.sch)
        self.schedule_stages: List[SBlockRV] = []
        self.blocks: List[SBlockRV] = []
        self.output_blocks: List[SBlockRV] = None
        self.reduction_block: SBlockRV = None
        self.raxis = []
        self.input_buffers = []
        self.output_buffers = []
        self.buffers = []
        self.args = []
        self._analysis_funcinfo()
        self._assign_placeholder_node()
        self.ana = get_analyzer_by_tir(self.block_analyzer, self.blocks)

        # set input shapes and dtypes
        for edge, buffer in zip(self.inputs, self.input_buffers):
            edge.src_node.set_shape(buffer.shape, edge.src_id)
            edge.src_node.set_dtype(tvm.DataType(buffer.dtype), edge.src_id)
        for output_id, buffer in enumerate(self.output_buffers):
            self.set_shape(buffer.shape, output_id)
            self.set_dtype(tvm.DataType(buffer.dtype), output_id)

    def _assign_placeholder_node(self):
        inputs: List[Node] = []
        for buffer in self.input_buffers:
            inputs.append(PlaceHolderNode(buffer.name))

        for dst_id, n in enumerate(inputs):
            if isinstance(n, Node):
                n = (n, 0)
            assert len(n) == 2
            src_node, src_id = n[0], n[1]
            edge = Edge(src_node, self, src_id, dst_id)
            self._in_edges.append(edge)
            src_node._out_edges.append(edge)

    def _specialize_func(self, func: PrimFunc):
        # Specialize the function to make it more friendly for analysis.
        # set attrs
        for k, v in func.attrs.items():
            self.set_tag(k, v)
        if self.get_tag("is_speclized"):
            return func
        opt_shapes = self.get_tag("opt_shapes")
        if opt_shapes:
            for name, shape in opt_shapes.items():
                var = analysis.find_var_from_func(func, name)
                if var is not None:
                    func = func.specialize({var: shape.astype(var.dtype)})
        return func

    def _analysis_funcinfo(self):
        root_block = analysis.get_root_block(self.sch)
        blocks = self.sch.get_child_blocks(root_block)
        self.blocks = blocks

        self.output_blocks = self.sch.get_output_blocks(root_block)
        reduction_blocks = get_reduction_blocks(self.sch, blocks)
        if reduction_blocks is None:
            self.reduction_block = None
            self.schedule_stages.append(*self.output_blocks)
        else:
            # analysis on the last reduction block
            self.reduction_block = reduction_blocks[-1]
            # set raxis
            reduce_block_info = self.block_analyzer.get_block_info(self.reduction_block)
            for iter in reduce_block_info.iters:
                if iter.kind == "R":
                    self.raxis.append(iter)
            self.schedule_stages.append(self.reduction_block)

        # collect output buffers
        for output_block in self.output_blocks:
            for write in self.sch.get(output_block).writes:
                if write not in self.output_buffers:
                    self.output_buffers.append(write.buffer)

        for param in self.prim_func.params:
            if param not in self.prim_func.buffer_map:
                # in case of dynamic symbolic may in params
                continue
            buffer = self.prim_func.buffer_map[param]
            if buffer not in self.output_buffers:
                self.input_buffers.append(buffer)

        self.args = self.input_buffers + self.output_buffers
        self.buffers = [buffer for buffer in self.prim_func.buffer_map.values()]

        # set dtype
        self.set_dtype(tvm.DataType(self.output_buffers[0].dtype))

    def get_opt_shape(self, name) -> int:
        opt_shapes = self.get_tag("opt_shapes")
        if opt_shapes is None:
            return None
        return opt_shapes[name]

    def extent_wrapper(self, value) -> int:
        if isinstance(value, tvm.tirx.Var):
            return self.get_opt_shape(value.name)
        elif isinstance(value, tvm.tirx.IntImm):
            return int(value)
        else:
            return value

    @functools.lru_cache()
    def get_space_dim(self) -> List[int]:
        dim_size = []
        if self.reduction_block:
            block_info = self.block_analyzer.get_block_info(self.reduction_block)
            for iter in block_info.iters:
                if iter.kind == "S":
                    if isinstance(iter.dom.extent, tvm.tirx.IntImm):
                        dim_size.append(int(iter.dom.extent))
                    else:
                        assert isinstance(iter.dom.extent, tvm.tirx.Var)
                        dim_size.append(self.get_opt_shape(iter.dom.extent.name))
        else:
            # assume outer stage has the same shape
            loops = self.sch.get_loops(self.schedule_stages[0])
            for loop in loops:
                dim_size.append(int(self.sch.get(loop).extent))
        dim_size = [int(x) for x in dim_size]

        # add 1-batch dim back if needed
        is_batch_dim = getattr(self, "is_batch_dim", None)
        if not is_batch_dim:
            return dim_size
        if not self.output_buffers:
            return dim_size

        output_shape = [
            int(self.extent_wrapper(ext)) for ext in self.output_buffers[0].shape
        ]
        if len(output_shape) != len(is_batch_dim) or len(dim_size) >= len(output_shape):
            return dim_size

        recovered_dim_size = []
        dim_idx = 0
        out_idx = 0
        while out_idx < len(output_shape):
            if dim_idx >= len(dim_size):
                return dim_size

            out_extent = output_shape[out_idx]
            if dim_size[dim_idx] == out_extent:
                recovered_dim_size.append(dim_size[dim_idx])
                dim_idx += 1
                out_idx += 1
                continue

            if is_batch_dim[out_idx]:
                batch_product = 1
                batch_extents = []
                cursor = out_idx
                while (
                    cursor < len(output_shape)
                    and is_batch_dim[cursor]
                    and batch_product < dim_size[dim_idx]
                ):
                    batch_product *= output_shape[cursor]
                    batch_extents.append(output_shape[cursor])
                    cursor += 1

                if batch_extents and batch_product == dim_size[dim_idx]:
                    recovered_dim_size.extend(batch_extents)
                    dim_idx += 1
                    out_idx = cursor
                    continue

                if out_extent == 1:
                    recovered_dim_size.append(1)
                    out_idx += 1
                    continue

            return dim_size

        if dim_idx != len(dim_size):
            return dim_size
        return recovered_dim_size

    def set_dtype(self, dtype: tvm.DataType, id=0) -> None:
        assert isinstance(dtype, tvm.DataType), type(dtype)
        if dtype == tvm.DataType("bool"):
            dtype = tvm.DataType("int8")
        if len(self._dtypes) <= id:
            self._dtypes.extend([None for _ in range(id - len(self._dtypes) + 1)])
        elif self._dtypes[id] is not None:
            assert self._dtypes[id] == dtype, (self._dtypes, dtype)
        self._dtypes[id] = dtype

    def get_buffer_dtype(self, buffer: tirx.Buffer) -> tvm.DataType:
        return tvm.DataType(buffer.dtype)

    def propagate(self, tile, rstep: Optional[Dict] = None, targets=None):
        if rstep is None:
            rstep = {}
        shape = {
            self.block_analyzer.get_output_buffers(block)[0].name: [
                tvm.arith.ConstIntBound(0, val - 1) for val in tile
            ]
            for block in self.schedule_stages
        }
        return self.ana.infer(shape, rstep, targets)

    def propagate_inputs(self, tile, rstep: Optional[Dict] = None) -> List[List[int]]:
        if rstep is None:
            rstep = {}
        read_idx_offset = len(self.input_buffers)
        targets = [t.name for t in self.args[:read_idx_offset]]
        shapes, intermediate_bind = self.propagate(tile, rstep, targets)
        results = []
        for i, arg in enumerate(self.args[:read_idx_offset]):
            if arg.name in intermediate_bind:
                results.append(shapes[arg.name])
                continue
            # should not exceed original shape
            trimmed_shape = [
                self.extent_wrapper(j)
                for j in list(
                    map(min, zip(shapes[arg.name], self.input_buffers[i].shape))
                )
            ]
            results.append(trimmed_shape)
        return results

    # Propagate inputs only on reduction block
    def propagate_inputs_on_reduction(
        self, tile, rstep: Optional[Dict] = None
    ) -> List[List[int]]:
        if rstep is None:
            rstep = {}
        reduction_block = self.reduction_block
        args = self.block_analyzer.get_input_buffers(reduction_block)
        targets = [t.name for t in args]
        shapes, intermediate_bind = self.propagate(tile, rstep, targets)
        results = []
        for i, arg in enumerate(args):
            if arg.name in intermediate_bind:
                results.append(shapes[arg.name])
                continue
            # should not exceed original shape
            propagate_shape = shapes[arg.name]
            buffer_shape = args[i].shape
            if len(buffer_shape) > len(propagate_shape):
                buffer_shape = buffer_shape[-len(propagate_shape) :]
            trimmed_shape = [
                self.extent_wrapper(j)
                for j in list(map(min, zip(propagate_shape, buffer_shape)))
            ]
            results.append(trimmed_shape)
        return results

    def propagate_outputs(self, tile, rstep: Optional[Dict] = None) -> List[List[int]]:
        if rstep is None:
            rstep = {}
        read_idx_offset = len(self.input_buffers)
        targets = [t.name for t in self.args[read_idx_offset:]]
        shapes, _ = self.propagate(tile, rstep, targets)
        results = []
        for i, arg in enumerate(self.args[read_idx_offset:]):
            # should not exceed original shape
            trimmed_shape = list(
                map(min, zip(shapes[arg.name], self.input_buffers[i].shape))
            )
            results.append(trimmed_shape)
        return results

    def propagate_reduction_inputs(
        self, shape, rstep: Optional[Dict] = None
    ) -> Dict[str, List[int]]:
        if rstep is None:
            rstep = {}
        if self.reduction_block is None:
            return {}
        targets = [
            b.name for b in self.block_analyzer.get_input_buffers(self.reduction_block)
        ]
        results, _ = self.propagate(shape, rstep, targets)
        return results

    def get_reduce_inputs_dtype(self):
        if self.reduction_block is None:
            return {}
        return {
            b.name: tvm.DataType(b.dtype)
            for b in self.block_analyzer.get_input_buffers(self.reduction_block)
        }

    @functools.lru_cache()
    def infer_tensorcore_axis(self) -> Tuple[int]:
        # axis is fixed for one expression, so only inference and cached
        assert self.get_tag("tensorcore_config")

        C_ax_m, C_ax_n = self.get_tag("tensorcore_config")
        wmma_m, wmma_n, wmma_k = [16, 16, 16]  # just for testing, any number is ok

        output_buffer_shape = (
            self.block_analyzer.sch.get(self.reduction_block).writes[0].buffer.shape
        )
        valid_region = []
        for region in output_buffer_shape:
            if region.value == 1:
                continue
            valid_region.append(region)

        num_nvalid_regions = len(output_buffer_shape) - len(valid_region)
        self.set_tag("num_nvalid_regions", num_nvalid_regions)

        def get_cl_shapes(c_ax_m, c_ax_n, num_nvalid_regions):
            spatial_dim = self.get_space_dim()
            assert len(valid_region) == len(
                spatial_dim
            ), f" {valid_region} mismatch with {spatial_dim}"
            cl_shapes = [1] * len(spatial_dim)
            cl_shapes[c_ax_m - num_nvalid_regions] = wmma_m
            cl_shapes[c_ax_n - num_nvalid_regions] = wmma_n
            return cl_shapes

        CL_shape = get_cl_shapes(C_ax_m, C_ax_n, num_nvalid_regions)
        self.set_tag(
            "tensorcore_config", [s - num_nvalid_regions for s in [C_ax_m, C_ax_n]]
        )
        shapes = self.propagate_reduction_inputs(
            CL_shape, {x.var.name: 1 for x in self.raxis}
        )
        A_deps, B_deps = shapes.values()
        A_ax_m = A_deps.index(wmma_m)
        B_ax_n = B_deps.index(wmma_n)

        CL_shape = [1] * len(self.get_space_dim())
        shapes = self.propagate_reduction_inputs(
            CL_shape, {x.var.name: wmma_k for x in self.raxis}
        )
        A_deps, B_deps = shapes.values()
        A_ax_k = len(A_deps) - 1 - A_deps[::-1].index(wmma_k)
        B_ax_k = len(B_deps) - 1 - B_deps[::-1].index(wmma_k)
        tc_axis = (A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n)
        return tc_axis

    def footprint(self, shape, rstep, stride_map: Optional[Dict] = None) -> int:
        if stride_map is None:
            stride_map = {}
        result = 0
        shapes, _ = self.propagate(shape, rstep)

        def is_broadcast_pattern(buffer, output_buffer):
            return (
                buffer in self.args
                and len(shapes[output_buffer.name]) > len(shapes[buffer.name])
                and np.prod(shapes[output_buffer.name]) > np.prod(shapes[buffer.name])
            )

        def is_after_reduce_stage(block):
            if not self.reduction_block:
                return False
            reduce_dependent_blocks = getattr(self, "reduce_dependent_blocks", None)
            if reduce_dependent_blocks is None:
                reduce_dependent_blocks = set()
                pre_order_traverse(
                    self.block_analyzer,
                    [self.reduction_block],
                    lambda block: reduce_dependent_blocks.add(block),
                )
                self.reduce_dependent_blocks = reduce_dependent_blocks
            return block not in reduce_dependent_blocks

        # compute cached stages
        cached_tensor = []
        for block in self.blocks:
            output_buffer = self.block_analyzer.get_output_buffers(block)[0]
            for buffer in self.block_analyzer.get_input_buffers(block):
                cache = buffer.name not in cached_tensor and (
                    is_broadcast_pattern(buffer, output_buffer)
                    or self.block_analyzer.get_block_info(block).is_reduction()
                )
                if not cache:
                    continue
                cached_tensor.append(buffer.name)
                if is_after_reduce_stage(block):
                    continue  # cache after reduce op can often reuse buffer in reduce stage

                if buffer.name in stride_map:
                    num_elem = stride_map[buffer.name].compute_elements_from_shape(
                        shapes[buffer.name]
                    )
                else:
                    num_elem = np.prod(shapes[buffer.name])
                buffer_len = num_elem * int((tvm.DataType(buffer.dtype).bits + 7) // 8)
                buffer_len = (buffer_len + 31) // 32 * 32
                result += buffer_len
        return result, cached_tensor

    def get_input_buffers(self) -> List[tirx.Buffer]:
        return self.block_analyzer.input_buffers


class ComputeOpNode(PrimFuncNode):
    def __init__(
        self,
        func,
        name,
        is_batch_dim: Optional[List[bool]] = None,
        input_is_batch_dim: Optional[List[List[bool]]] = None,
    ):
        self.is_batch_dim = is_batch_dim
        self.input_is_batch_dim = input_is_batch_dim
        tensorized_func, tags = get_tensorized_func_and_tags(func, ARCH.target)
        super().__init__(tensorized_func, name=name, tags=tags)

    def _assign_placeholder_node(self):
        super()._assign_placeholder_node()
        for input in self._in_edges:
            assert isinstance(
                input.src_node, PlaceHolderNode
            ), f"At this point, input src_node {input.src_node} should always be a PlaceHolderNode."
            if self.input_is_batch_dim is None:
                input.src_node.is_batch_dim = self.is_batch_dim
            else:
                input.src_node.is_batch_dim = self.input_is_batch_dim[input.dst_id]

    def create_prim_func(self):
        raise NotImplementedError("create_prim_func must be implemented by subclasses.")

    def gen_macro(self, **kwargs) -> str:
        raise NotImplementedError("gen_macro must be implemented by subclasses.")

    def _format_code(self, code: str) -> str:
        return textwrap.dedent(code) + "\n"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return type(self) is type(other) and self.name == other.name


class ViewOpNode(PrimFuncNode):
    def __init__(
        self,
        func,
        name,
        is_batch_dim: Optional[List[bool]] = None,
        input_is_batch_dim: Optional[List[List[bool]]] = None,
    ):
        self.is_batch_dim = is_batch_dim
        self.input_is_batch_dim = input_is_batch_dim
        tensorized_func, tags = get_tensorized_func_and_tags(func, ARCH.target)
        super().__init__(tensorized_func, name=name, tags=tags)

    def _assign_placeholder_node(self):
        super()._assign_placeholder_node()
        for input in self._in_edges:
            assert isinstance(
                input.src_node, PlaceHolderNode
            ), f"At this point, input src_node {input.src_node} should always be a PlaceHolderNode."
            if self.input_is_batch_dim is None:
                input.src_node.is_batch_dim = self.is_batch_dim
            else:
                input.src_node.is_batch_dim = self.input_is_batch_dim[input.dst_id]

    def create_prim_func(self):
        raise NotImplementedError("create_prim_func must be implemented by subclasses.")

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return type(self) is type(other) and self.name == other.name


class OutputNode(Node):
    def __init__(self, node, dtype, id=0):
        # connect node and output node
        super().__init__(name="OutputNode")

        assert isinstance(
            node, (ComputeOpNode, ViewOpNode, GraphNode)
        ), f"OutputNode can only connect to ComputeOpNode, ViewOpNode or GraphNode, but got {type(node)}."

        # initialize edge and connect
        src_node, src_id = node, id
        edge = Edge(src_node, self, src_id, 0)
        self._in_edges.append(edge)
        src_node._out_edges.append(edge)
        self.set_dtype(DataType(dtype))

        if isinstance(node, (ComputeOpNode, ViewOpNode)):
            self.set_shape(node.get_shape(id))
            self.is_batch_dim = node.is_batch_dim
        elif isinstance(node, GraphNode):
            node._bind_output_node(self)

    def is_output(self):
        return True

    def get_ir(self) -> str:
        return "output"

    def __hash__(self):
        edge = self.inputs[0]
        return hash((edge.src_node.name, edge.src_id))

    def __eq__(self, other):
        return (
            isinstance(other, OutputNode)
            and self.inputs[0].src_node == other.inputs[0].src_node
            and self.inputs[0].src_id == other.inputs[0].src_id
        )


class GraphNode(Node):
    def __init__(self, name: str, graph: Graph):
        super().__init__(name=name)
        self.graph = graph
        self._assign_placeholder_nodes()
        self._assign_graph_node_info()
        self._assign_accum_nodes()

    def _assign_placeholder_nodes(self, skip=[]):
        new_placeholders = {}

        # collect and bind placeholder nodes
        for outer_node in self.graph.nodes:
            if isinstance(outer_node, PlaceHolderNode) and outer_node not in skip:
                # create a inner placeholder node
                inner_node = PlaceHolderNode()
                inner_node.is_batch_dim = outer_node.is_batch_dim
                inner_node.name, outer_node.name = (
                    outer_node.name + outer_node.name[-1],
                    outer_node.name,
                )
                new_placeholders[outer_node.name] = inner_node

                # connect the outer node to the graph node
                outer_edge = Edge(outer_node, self, 0, 0)
                self._in_edges.append(outer_edge)
                for edge in list(outer_node._out_edges):
                    if edge.dst_node in self.graph.nodes:
                        # transform edge into inner edge and remove from outer node
                        edge.src_node = inner_node
                        inner_node._out_edges.append(edge)
                        outer_node._out_edges.remove(edge)
                outer_node._out_edges.append(outer_edge)

                inner_node._shapes = outer_node._shapes
                inner_node._dtypes = outer_node._dtypes
                self.graph.binding[outer_node.name] = inner_node
                self.graph.binding[inner_node.name] = outer_node

                # get index of the outer node in the graph node inputs
                input_idx = self.graph.nodes.index(outer_node)
                self.graph.nodes[input_idx] = (
                    inner_node  # replace outer node with inner node in graph nodes
                )

        # update inner graph bindings
        self._update_inner_graph_bindings(new_placeholders)

    def _update_inner_graph_bindings(self, new_placeholders):
        new_ph = {}

        for node in self.graph.nodes:
            if isinstance(node, GraphNode):
                for (
                    old_outer_placeholder_name,
                    new_outer_placeholder,
                ) in new_placeholders.items():
                    if old_outer_placeholder_name in node.graph.binding:
                        inner_placeholder = node.graph.binding[
                            old_outer_placeholder_name
                        ]

                        if inner_placeholder.name == new_outer_placeholder.name:
                            inner_placeholder.name = (
                                new_outer_placeholder.name
                                + new_outer_placeholder.name[-1]
                            )
                            new_ph[new_outer_placeholder.name] = inner_placeholder

                        node.graph.binding[new_outer_placeholder.name] = (
                            inner_placeholder
                        )
                        node.graph.binding[inner_placeholder.name] = (
                            new_outer_placeholder
                        )
                        del node.graph.binding[old_outer_placeholder_name]
                node._update_inner_graph_bindings(new_ph)

    def _assign_graph_node_info(self):
        # assign tensor info based on explicit output nodes, if present
        output_idx = 0
        for node in self.graph.nodes:
            if isinstance(node, OutputNode):
                self.set_shape(node.get_shape(0), id=output_idx, overwrite=True)
                self.set_dtype(node.get_dtype(0), id=output_idx)
                output_idx += 1

    def _assign_accum_nodes(self):
        # TODO: support nested graph accum_nodes
        for node in self.graph.nodes:
            if isinstance(node, ComputeOpNode) and all(
                o.dst_node not in self.graph.nodes for o in node.outputs
            ):
                if node in self.graph.accum_nodes:
                    continue

                self.graph.accum_nodes.append(node)
                output_id = len(self.graph.accum_nodes) - 1
                self.set_shape(node.get_shape(0), id=output_id, overwrite=True)
                self.set_dtype(node.get_dtype(0), id=output_id)
                for edge in list(node.outputs):
                    assert (
                        edge.dst_node not in self.graph.nodes
                    ), f"Destination node {edge.dst_node.name} should not be in inner graph nodes."
                    new_edge = AccumEdge(
                        self,
                        edge.dst_node,
                        len(self.outputs),
                        edge.dst_id,
                        accum_node=node,
                    )
                    self.outputs.append(new_edge), node.outputs.remove(edge)
                    edge.dst_node.set_inputs(new_edge.dst_id, new_edge)
                assert (
                    len(node.outputs) == 0
                ), f"All outputs of accum_node {node.name} should be removed after assigning AccumEdge."

    def _resolve_output_binding(self, output_id: int):
        if output_id < len(self.graph.accum_nodes):
            bound_node = self.graph.accum_nodes[output_id]
            shape = bound_node.get_shape(0)
            is_batch_dim = getattr(bound_node, "is_batch_dim", None)
            return bound_node, shape, is_batch_dim

        inner_outputs = [node for node in self.graph.nodes if isinstance(node, OutputNode)]
        if output_id < len(inner_outputs):
            inner_output = inner_outputs[output_id]
            bound_node = inner_output.inputs[0].src_node
            return bound_node, inner_output.get_shape(0), inner_output.is_batch_dim

        raise IndexError(
            f"GraphNode {self.name} does not have output id {output_id}."
        )

    def _bind_output_node(self, output_node):
        output_id = output_node.inputs[0].src_id
        bound_node, shape, is_batch_dim = self._resolve_output_binding(output_id)
        output_node.set_shape(shape, overwrite=True)
        output_node.is_batch_dim = is_batch_dim
        self.graph.binding[bound_node.name] = output_node
        self.graph.binding[output_node.name] = bound_node

    def __hash__(self):
        assert (
            self.graph.locked
        ), "Graph must be locked (in immutable state) before hashing GraphNode."
        return hash((self.name, tuple(self.graph.nodes)))

    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return False
        if self.name != other.name:
            return False
        return len(self.graph.nodes) == len(other.graph.nodes) and all(
            n1 == n2 for n1, n2 in zip(self.graph.nodes, other.graph.nodes)
        )


@dataclass
class AccumEdge(Edge):
    accum_node: "Node"

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        assert isinstance(
            self.accum_node, Node
        ), f"Invalid accum_node type: {type(self.accum_node)}"
        assert isinstance(
            self.src_node, GraphNode
        ), f"Invalid src_node type: {type(self.src_node)}"
        assert (
            self.accum_node in self.src_node.graph.accum_nodes
        ), f"accum_node {self.accum_node.name} not in src_node's graph accum_nodes"
