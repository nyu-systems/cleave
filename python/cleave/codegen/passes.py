from ..ir.graph import (
    AccumEdge,
    Graph,
    PlaceHolderNode,
    OutputNode,
    GraphNode,
    ComputeOpNode,
    ViewOpNode,
)
from ..ir.graph_utils import find_topo_sort
from ..ir.ops import Add, Div, Exp, Matmul, Mul, ReduceSum, Sub
from .operands import kept_extents

# A Hopper warp-group gemm (WGMMA): four warps make a group, and one instruction computes
# a tile of 64 rows, 16 to a warp, in columns that are a multiple of 8 to a warp.
_WARP_THREADS = 32
_GROUP_WARPS = 4
_WARP_ROWS = 16
_WARP_COLS = 8


def wgmma_row_warps(warp_policy: str, m: int, n: int, n_threads: int) -> tuple:
    """How many warps TileLang may lay along the rows of a warp-group gemm.

    The gemm of an ``(m, n)`` tile on ``n_threads`` threads under ``Matmul.warp_policy``;
    empty when TileLang does not issue it as a warp-group gemm at all (fewer than 64 rows,
    or a block that is not whole warp groups). This is ``ComputeWgmmaWarpPartition`` of the
    vendored TileLang (``src/cuda/op/gemm.cc``): ``rows`` takes the most row warps that
    divide the tile, ``cols`` the fewest. ``even`` picks by a ratio it computes in single
    precision, so every partition it could pick is returned.
    """
    num_warps = n_threads // _WARP_THREADS
    if n_threads % _WARP_THREADS or num_warps % _GROUP_WARPS or m < 4 * _WARP_ROWS:
        return ()
    valid = [
        row_warps
        for row_warps in range(_GROUP_WARPS, num_warps + 1, _GROUP_WARPS)
        if num_warps % row_warps == 0
        and m % (row_warps * _WARP_ROWS) == 0
        and n % (num_warps // row_warps * _WARP_COLS) == 0
    ]
    if not valid or warp_policy == "even":
        return tuple(valid)
    return (max(valid) if warp_policy == "rows" else min(valid),)


def register_operand_is_misplaced(warp_policy: str, m: int, n: int, n_threads: int) -> bool:
    """Whether TileLang computes a warp-group gemm of an ``(m, n)`` tile on the wrong rows
    when its first operand is a fragment.

    TileLang 0.1.13 lays the two fragments of such a gemm out differently. With ``G`` warp
    groups along the rows and ``R`` instructions per group, warp ``j`` of group ``g`` holds
    for instruction ``r`` the 16 rows from

    * ``64 * G * r + 64 * g + 16 * j`` of the operand (``make_mma_load_layout``: all the
      warps side by side, then repeated), and
    * ``64 * R * g + 64 * r + 16 * j`` of the result (``make_mma_store_layout``: a group
      owns consecutive rows).

    The instruction multiplies whatever the registers of its group hold, so the result
    rows it writes are the product of other operand rows wherever the two differ: for
    every ``(g, r)`` but the first and the last as soon as ``G >= 2`` and ``R >= 2``. On
    256 threads a 256-row tile comes out with rows 64..127 and 128..191 exchanged, a
    512-row tile with six of its eight 64-row bands misplaced. Nothing fails: the kernel
    lowers, runs, and is wrong. With one group, or one instruction per group, the two
    layouts are the same.
    """
    return any(
        row_warps // _GROUP_WARPS >= 2 and m // (row_warps * _WARP_ROWS) >= 2
        for row_warps in wgmma_row_warps(warp_policy, m, n, n_threads)
    )


# TODO: have all levels of the graph with OutputNode (this should make this function smoother)
def annotate_args(graph: Graph, is_main: bool = False) -> Graph:
    """The kernel's parameter list: its inputs in topological order, then its outputs.

    Outputs go last whatever the topological order says. With several outputs it can put
    one in front of an input -- the row sum of an un-looped split partial is finished
    before the value matmul's ``V`` is reached -- and everything that launches a kernel
    takes the trailing arguments for its outputs (``runtime.reference.graph_out_indices``,
    ``placeholder_order``): such a kernel was handed ``V`` for its row-sum buffer.
    """
    outputs = []
    for node in find_topo_sort(graph.nodes):
        if is_main:
            if isinstance(node, PlaceHolderNode):
                graph.args.append(node)
            elif isinstance(node, OutputNode):
                outputs.append(node)
        if isinstance(node, GraphNode):
            in_args = [i.src_node for i in node.inputs]
            # TODO: output nodes
            node.graph.args.extend(in_args)
            annotate_args(node.graph, is_main=False)
    graph.args.extend(outputs)

    return graph


def annotate_tinfo(graph: Graph, is_main: bool = False) -> Graph:
    all_names = set([node.name for node in graph.nodes])
    assert len(all_names) == len(graph.nodes), "node names should be unique"

    def getname(name: str) -> str:
        name = name.replace(".", "_")
        if name in all_names:
            idx = 1
            while f"{name}_{idx}" in all_names:
                idx += 1
            name = f"{name}_{idx}"
        all_names.add(name)
        return name

    for node in find_topo_sort(graph.nodes):
        if isinstance(node, (PlaceHolderNode, OutputNode)):
            graph.tinfo_map[node] = (
                node.name,
                "g" if is_main else ("r" if len(node.outputs) > 1 else "s"),
            )
        elif isinstance(node, ComputeOpNode):
            graph.tinfo_map[node] = (getname("t_" + node.name), "r")

            # special handling for tilelang matmul that does not support two fragment inputs
            if isinstance(node, Matmul):
                input_tinfos = [
                    (
                        graph.tinfo_map[edge.src_node][1]
                        if not isinstance(edge.src_node, GraphNode)
                        else edge.src_node.graph.tinfo_map[edge.accum_node][1]
                    )
                    for edge in node.inputs
                ]
                assert len(input_tinfos) == 2, "Matmul should have exactly two inputs"
                # put smallest input in shared memory
                if input_tinfos == ["r", "r"]:
                    import numpy as np

                    input_placeholders = [
                        edge.src_node
                        for edge in node.inputs
                        if isinstance(edge.src_node, PlaceHolderNode)
                    ]
                    if len(input_placeholders) == 0:
                        # TODO(david): this is a limitation of tilelang that
                        # forces us to keep at least one operand in smem, might
                        # be solved in a upcoming version.
                        # Both operands are computed fragments (e.g. QKNorm's
                        # QK^T where Q and K are RMSNorm outputs). Hopper WGMMA
                        # requires the B (transpose_B) operand in shared memory
                        # (gemm.cc:436-437). Stage BOTH operands through smem:
                        # operand 1 (B) satisfies WGMMA; operand 0 (A) decouples
                        # the matmul's TC-fragment layout demand from the shared
                        # input fragment (Q), which is also consumed by the
                        # RMSNorm reduction path in a conflicting linear layout.
                        # NOTE(david): annotate_mla_accumulation_layout_conflict resets
                        # stage_fragment_operands_in_shared on every matmul, so if
                        # CompilerPassOptions.mla_decode_pass is ever enabled on a
                        # graph that also has a two-fragment matmul it would clobber
                        # this; today mla_decode_pass defaults False and no graph has
                        # both patterns.
                        node.stage_fragment_operands_in_shared = (0, 1)
                    else:
                        def _tile_volume(p):
                            tile = graph.tmap[p].tile_size
                            if any(not isinstance(t, int) for t in tile):
                                return float("inf")  # whole-buffer operand
                            return int(np.prod(tile))

                        smallest_placeholder = min(input_placeholders, key=_tile_volume)
                        graph.tinfo_map[smallest_placeholder] = (
                            graph.tinfo_map[smallest_placeholder][0],
                            "s",
                        )
        elif isinstance(node, ViewOpNode):
            src_tinfo = graph.tinfo_map[node.inputs[0].src_node]
            graph.tinfo_map[node] = (getname("v_" + node.name), src_tinfo[1])
        elif isinstance(node, GraphNode):
            # ensure that tinfo_map also reflects all args of the subgraph
            node.graph.tinfo_map = {
                k: v for k, v in graph.tinfo_map.items() if k in node.graph.args
            }
            annotate_tinfo(node.graph, is_main=False)

    return graph


def annotate_tc_layout(graph: Graph) -> Graph:
    node_to_graph = {}
    graph_to_owner = {}

    def collect_hierarchy(cur_graph: Graph, owner: GraphNode = None) -> None:
        if owner is not None:
            graph_to_owner[id(cur_graph)] = owner

        for cur_node in cur_graph.nodes:
            node_to_graph[id(cur_node)] = cur_graph
            if isinstance(cur_node, GraphNode):
                collect_hierarchy(cur_node.graph, cur_node)

    def annotate_graph(cur_graph: Graph) -> None:
        init_graph_layout(cur_graph)

        for cur_node in cur_graph.nodes:
            if isinstance(cur_node, GraphNode):
                annotate_graph(cur_node.graph)
            elif isinstance(cur_node, Matmul):
                propagate_tc_layout(cur_node, node_to_graph, graph_to_owner)

    collect_hierarchy(graph)
    annotate_graph(graph)
    return graph


def annotate_fragment_layout_conflicts(graph: Graph) -> Graph:
    """Stage the divisor of a ``sum / sum`` divide through shared memory.

    ``value.sum(split) / rowsum.sum(split)`` -- the split-KV combine -- divides one
    reduced fragment by another that was reduced from a tensor of lower rank. TileLang
    lays each of them out from its own input, and no layout of the two serves the divide
    (``no available layout found``). Read from shared memory, the divisor has no fragment
    layout to agree with. This holds wherever the scheduler put the two sums: both in one
    loop, each in a loop of its own, or either one in the same tier as the divide when
    its step is the whole split axis.

    The same holds for a multiply, and for a second operand that is not the lower-rank
    sum itself but computed from it elementwise: the RMSNorm split-K combine multiplies
    the summed matmul partials by ``1 / sqrt(rowsum.sum(split) / K)``, a chain of three
    ops that all keep the row sum's fragment layout.

    And for a matmul that reads a sum (or such a chain from one) as an operand:
    ``sum(x, 0, keepdim=True) @ w`` with ``x`` of shape ``(B, M, K)``. The sum's fragment
    is laid out from the tensor it reduces, the gemm wants its operand in the tensor-core
    layout, and nothing serves both. It is staged like the operands of a matmul of two
    computed values (``annotate_tinfo``).

    And for the first operand of a matmul whose block is two warp groups tall, wherever
    that operand is a fragment (``stage_misplaced_register_operands``).

    And for a loop's accumulator of lower rank that is broadcast over an operand read
    straight from global memory: ``x / sum(x, -1)`` with the sum looped, where the
    quotient's tile is the whole row and the kernel's input is read where it lies. The
    quotient is then the only other fragment of the op, and TileLang gives it the layout
    of the sum, which is the layout of the slices the sum was reduced from: with slices
    of 512 columns or fewer on 128 threads, one or a few rows per thread, each held whole.
    On 4096 columns every thread computed and kept 4096 to 32768 elements -- up to 32
    times the work, a frame of 8 to 64 KiB, 512-fold unrolled loops that take the device
    compiler two minutes and more, and 4.3 GiB of thread stacks at launch. With the sum
    read from shared memory the quotient is laid out across the threads of the block.
    The same holds one or more elementwise ops away from the sum: ``x / sqrt(sum(x * x))``.
    """

    def reset_graph(cur_graph: Graph) -> None:
        for cur_node in cur_graph.nodes:
            if hasattr(cur_node, "resolve_operand_layout"):
                cur_node.resolve_operand_layout = None
            if isinstance(cur_node, GraphNode):
                reset_graph(cur_node.graph)

    def get_input_scope(cur_graph: Graph, input_edge) -> str | None:
        accum_node = getattr(input_edge, "accum_node", None)
        if accum_node is not None and isinstance(input_edge.src_node, GraphNode):
            return input_edge.src_node.graph.tinfo_map.get(accum_node, (None, None))[1]
        return cur_graph.tinfo_map.get(input_edge.src_node, (None, None))[1]

    def through_elementwise_chain(edge):
        """The edge that feeds the chain of one-operand, shape-keeping ops ``edge`` reads
        the end of (``edge`` itself when it reads no such op): ``1 / sqrt(x / K)`` -> x."""
        while not isinstance(edge, AccumEdge):
            node = edge.src_node
            if (
                not isinstance(node, ComputeOpNode)
                or isinstance(node, ReduceSum)
                or len(node.inputs) != 1
            ):
                break
            below = node.inputs[0]
            source = getattr(below, "accum_node", None) or below.src_node
            if tuple(source.get_shape(0)) != tuple(node.get_shape(0)):
                break
            edge = below
        return edge

    def is_independent_reduction_div(cur_node, cur_graph: Graph) -> bool:
        if cur_node.const is not None or cur_node.broadcast != 0:
            return False
        if len(cur_node.inputs) != 2:
            return False

        lhs_edge, rhs_edge = cur_node.inputs
        # The node that produces each operand: a loop's accumulator, or a reduction of
        # this tier (one whose step is its whole axis is not looped). The second operand
        # may come from its sum through ops that keep the sum's shape, and so its layout.
        rhs_sum_edge = through_elementwise_chain(rhs_edge)
        lhs_accum = getattr(lhs_edge, "accum_node", None) or lhs_edge.src_node
        rhs_accum = getattr(rhs_sum_edge, "accum_node", None) or rhs_sum_edge.src_node

        if not isinstance(lhs_accum, ReduceSum) or not isinstance(rhs_accum, ReduceSum):
            return False
        if get_input_scope(cur_graph, rhs_edge) != "r":
            return False

        lhs_shape = tuple(lhs_accum.get_shape(0))
        rhs_shape = tuple(rhs_accum.get_shape(0))
        if len(lhs_shape) != len(rhs_shape) + 1:
            return False
        if lhs_shape[:-1] != rhs_shape:
            return False

        # Both sums run over the leading axis of the tensor they are stored in (the split
        # axis, behind batch dims the block holds one element of). ``stored_axis``, not
        # ``axis``: for an axis named from the end ``axis`` stays the built one.
        if lhs_accum.stored_axis != 0 or rhs_accum.stored_axis != 0:
            return False
        if len(lhs_accum.inputs) != 1 or len(rhs_accum.inputs) != 1:
            return False

        lhs_src = lhs_accum.inputs[0].src_node
        rhs_src = rhs_accum.inputs[0].src_node
        if lhs_src is rhs_src:
            return False

        lhs_input_shape = tuple(
            lhs_accum.inputs[0].src_node.get_shape(lhs_accum.inputs[0].src_id)
        )
        rhs_input_shape = tuple(
            rhs_accum.inputs[0].src_node.get_shape(rhs_accum.inputs[0].src_id)
        )
        if len(lhs_input_shape) != len(rhs_input_shape) + 1:
            return False

        return True

    def accumulator_broadcast_over_input(cur_node, cur_graph: Graph) -> bool:
        """``x <op> f`` with ``x`` read from global memory and ``f`` a loop's accumulator
        that lacks an axis of it, or computed from one elementwise.

        The accumulator has the layout TileLang gave the slices it was reduced from, and
        a narrow slice puts each row on a few threads only. (A fragment computed in the
        tier of the op from the whole tile -- the max of an in-tile softmax -- is held by
        every thread, and next to a first operand in shared memory, the weight of a paged
        split combine, the result is laid out across the threads as it is.)
        """
        if cur_node.const is not None or cur_node.broadcast == -1:
            return False
        if len(cur_node.inputs) != 2:
            return False
        lhs_edge, rhs_edge = cur_node.inputs
        # The accumulator itself, or what a chain of one-operand, shape-keeping ops made
        # of it: ``x / sqrt(sum(x * x, -1))`` divides by a fragment with the sum's layout.
        accum_edge = through_elementwise_chain(rhs_edge)
        if not isinstance(accum_edge.src_node, GraphNode):
            return False
        if getattr(accum_edge, "accum_node", None) is None:
            return False
        return (
            get_input_scope(cur_graph, lhs_edge) == "g"
            and get_input_scope(cur_graph, rhs_edge) == "r"
        )

    def reduced_operands(cur_node, cur_graph: Graph) -> set:
        """The operands of a matmul that are fragments a reduction filled: the sum itself
        (in this tier, or a loop's accumulator) or what a chain of one-operand,
        shape-keeping ops made of it."""
        found = set()
        for edge in cur_node.inputs:
            if get_input_scope(cur_graph, edge) != "r":
                continue
            sum_edge = through_elementwise_chain(edge)
            producer = getattr(sum_edge, "accum_node", None) or sum_edge.src_node
            if isinstance(producer, ReduceSum):
                found.add(edge.dst_id)
        return found

    def annotate_graph(cur_graph: Graph) -> None:
        for cur_node in cur_graph.nodes:
            if isinstance(cur_node, GraphNode):
                annotate_graph(cur_node.graph)
                continue
            if isinstance(cur_node, (Div, Mul)) and is_independent_reduction_div(
                cur_node, cur_graph
            ):
                cur_node.resolve_operand_layout = "rhs"
            if isinstance(
                cur_node, (Add, Sub, Mul, Div)
            ) and accumulator_broadcast_over_input(cur_node, cur_graph):
                cur_node.resolve_operand_layout = "rhs"
            if isinstance(cur_node, Matmul) and len(cur_node.inputs) == 2:
                staged = set(cur_node.stage_fragment_operands_in_shared)
                reduced = reduced_operands(cur_node, cur_graph)
                if not reduced <= staged:
                    cur_node.stage_fragment_operands_in_shared = tuple(sorted(staged | reduced))

    reset_graph(graph)
    annotate_graph(graph)
    return stage_misplaced_register_operands(graph)


def stage_misplaced_register_operands(graph: Graph) -> Graph:
    """Stage the first operand of a matmul that TileLang would multiply on the wrong rows.

    A matmul reads its first operand from a fragment when that operand is computed
    (``exp(s) @ v``), is an input another node of the loop reads too (``x`` under
    ``x @ w`` next to ``sum(x * x)``, or under two matmuls), or has to be cast. On a block
    two warp groups tall whose tile needs more than one instruction per group, TileLang
    lays that fragment out for other rows than the ones each instruction writes
    (``register_operand_is_misplaced``), and the kernel is silently wrong: six tilings of
    the RMSNorm split-K partial and six of SwiGLU, all 256 threads with row tiles of 256
    or 512, were built, launched and failed validation with half or three quarters of
    their rows exchanged. Read from shared memory the operand has no fragment layout, and
    the gemm addresses it per warp group itself.

    An input of the loop that only matmuls read (``x`` under ``x @ w1`` and ``x @ w2``)
    is a fragment for no reader's sake: it is loaded into shared memory instead, as an
    input with a single reader is, and no fragment is left to copy back. Any other
    operand is staged (``Matmul.stage_fragment_operands_in_shared``).

    ``graph`` is the main tier (it carries ``n_threads``). Run after every pass that
    writes ``stage_fragment_operands_in_shared``.
    """
    n_threads = getattr(graph, "n_threads", None)
    if not isinstance(n_threads, int):
        return graph

    def reads_a_fragment(cur_graph: Graph, cur_node: Matmul) -> bool:
        edge = cur_node.inputs[0]
        src = edge.src_node
        if isinstance(src, GraphNode):
            accum_node = getattr(edge, "accum_node", None)
            if accum_node is None:
                return False
            scope = src.graph.tinfo_map.get(accum_node, (None, None))[1]
            in_dtype = accum_node.out_dtype
        else:
            scope = cur_graph.tinfo_map.get(src, (None, None))[1]
            in_dtype = (
                src.out_dtype if isinstance(src, ComputeOpNode) else str(src.get_dtype(0))
            )
        # A cast is emitted into a fragment unless the operand is staged
        # (``OperandResolver._resolve_operand``).
        return scope == "r" or in_dtype != cur_node.dtype

    def annotate_graph(cur_graph: Graph) -> None:
        for cur_node in cur_graph.nodes:
            if isinstance(cur_node, GraphNode):
                annotate_graph(cur_node.graph)
                continue
            if not isinstance(cur_node, Matmul) or len(cur_node.inputs) != 2:
                continue
            if 0 in cur_node.stage_fragment_operands_in_shared:
                continue
            # Only 16-bit operands go to the warp-group instruction.
            if cur_node.dtype not in ("float16", "bfloat16"):
                continue
            it = cur_graph.tmap.get(cur_node)
            extents = [] if it is None else kept_extents(it.tile_size)
            if len(extents) < 2:
                continue
            try:
                rows, cols = int(extents[-2]), int(extents[-1])
            except (TypeError, ValueError):
                continue  # a symbolic extent: not a tile the static path builds
            if not register_operand_is_misplaced(cur_node.warp_policy, rows, cols, n_threads):
                continue
            if not reads_a_fragment(cur_graph, cur_node):
                continue
            src = cur_node.inputs[0].src_node
            if (
                isinstance(src, PlaceHolderNode)
                and str(src.get_dtype(0)) == cur_node.dtype
                and all(isinstance(edge.dst_node, Matmul) for edge in src.outputs)
            ):
                cur_graph.tinfo_map[src] = (cur_graph.tinfo_map[src][0], "s")
                continue
            cur_node.stage_fragment_operands_in_shared = tuple(
                sorted({0, *cur_node.stage_fragment_operands_in_shared})
            )

    annotate_graph(graph)
    return graph


def annotate_mla_accumulation_layout_conflict(graph: Graph) -> Graph:
    def reset_graph(cur_graph: Graph) -> None:
        for cur_node in cur_graph.nodes:
            if isinstance(cur_node, Matmul):
                cur_node.stage_fragment_operands_in_shared = ()
            if isinstance(cur_node, GraphNode):
                reset_graph(cur_node.graph)

    def has_reduction_consumer(exp_node: Exp) -> bool:
        return any(isinstance(edge.dst_node, ReduceSum) for edge in exp_node.outputs)

    def has_upstream_score_matmul(exp_node: Exp, rhs_node: PlaceHolderNode) -> bool:
        worklist = [edge.src_node for edge in exp_node.inputs]
        visited = set()

        while worklist:
            cur_node = worklist.pop()
            cur_node_id = id(cur_node)
            if cur_node_id in visited:
                continue
            visited.add(cur_node_id)

            if isinstance(cur_node, Matmul) and len(cur_node.inputs) == 2:
                if cur_node.transpose_B and cur_node.inputs[1].src_node is rhs_node:
                    return True

            worklist.extend(edge.src_node for edge in getattr(cur_node, "inputs", []))

        return False

    def annotate_graph(cur_graph: Graph) -> None:
        for cur_node in cur_graph.nodes:
            if isinstance(cur_node, GraphNode):
                annotate_graph(cur_node.graph)
                continue

            if not isinstance(cur_node, Matmul) or len(cur_node.inputs) != 2:
                continue

            lhs_node = cur_node.inputs[0].src_node
            rhs_node = cur_node.inputs[1].src_node
            if not isinstance(lhs_node, Exp):
                continue
            if not isinstance(rhs_node, PlaceHolderNode):
                continue
            if not has_reduction_consumer(lhs_node):
                continue
            if not has_upstream_score_matmul(lhs_node, rhs_node):
                continue

            # MLA decode reuses the KV tensor for both score and value paths.
            # Stage the softmax fragment through swizzled shared memory before
            # the value matmul to avoid TileLang fragment layout conflicts.
            cur_node.stage_fragment_operands_in_shared = (0,)

    reset_graph(graph)
    annotate_graph(graph)
    # The reset above also dropped what this rule staged; it holds whatever the pattern.
    return stage_misplaced_register_operands(graph)


def init_graph_layout(graph: Graph) -> Graph:
    graph.tlayout_map.clear()
    for node in graph.nodes:
        if isinstance(node, (PlaceHolderNode, OutputNode)):
            graph.tlayout_map[node] = "linear"

    return graph


def propagate_tc_layout(node, node_to_graph, graph_to_owner):
    from collections import deque

    def get_graph(cur_node):
        return node_to_graph.get(id(cur_node))

    # Operands the gemm itself reads. A computed one is a swizzle-able shared buffer just
    # like a staged placeholder -- a gathered KV tile arrives this way -- but only right
    # here: further up its producer chain sit things like the 1-D index tile, which no
    # tensor-core instruction touches.
    gemm_operands = {id(edge.src_node) for edge in node.inputs}

    def mark_tc(cur_node):
        cur_graph = get_graph(cur_node)
        if cur_graph is None:
            return
        if not isinstance(cur_node, (PlaceHolderNode, OutputNode)):
            scope = cur_graph.tinfo_map.get(cur_node, (None, None))[-1]
            if not (
                isinstance(cur_node, ComputeOpNode)
                and scope == "s"
                and id(cur_node) in gemm_operands
            ):
                return
        cur_graph.tlayout_map[cur_node] = "tc"

    def is_reduction_node(cur_node):
        return len(getattr(cur_node, "raxis", [])) > 0

    def move_to_upper_graph_producer(cur_node):
        cur_graph = get_graph(cur_node)
        if cur_graph is None:
            return []

        owner = graph_to_owner.get(id(cur_graph))
        if owner is None:
            return []

        outer_node = cur_graph.binding.get(cur_node.name)
        return [outer_node] if outer_node is not None else []

    def move_to_upper_graph_consumers(cur_node):
        cur_graph = get_graph(cur_node)
        if cur_graph is None:
            return []

        owner = graph_to_owner.get(id(cur_graph))
        if owner is None:
            return []

        next_nodes = []
        bound_node = cur_graph.binding.get(cur_node.name)
        if isinstance(bound_node, OutputNode):
            next_nodes.append(bound_node)

        next_nodes.extend(
            edge.dst_node
            for edge in owner.outputs
            if getattr(edge, "accum_node", None) is cur_node
        )
        return next_nodes

    def get_next_nodes(cur_node, direction):
        if direction == "producers":
            if cur_node.inputs:
                return [edge.src_node for edge in cur_node.inputs]
            return move_to_upper_graph_producer(cur_node)

        if cur_node.outputs:
            return [edge.dst_node for edge in cur_node.outputs]
        return move_to_upper_graph_consumers(cur_node)

    frontier = deque([(node, "producers", True), (node, "consumers", True)])
    visited = {(id(node), "producers"), (id(node), "consumers")}

    while frontier:
        cur_node, direction, is_root = frontier.popleft()
        mark_tc(cur_node)

        if not is_root and (
            isinstance(cur_node, GraphNode) or is_reduction_node(cur_node)
        ):
            continue

        for next_node in get_next_nodes(cur_node, direction):
            state = (id(next_node), direction)
            if state in visited:
                continue
            visited.add(state)
            frontier.append((next_node, direction, False))
