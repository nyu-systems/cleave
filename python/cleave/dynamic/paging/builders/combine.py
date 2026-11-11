"""The combine half of a split-KV decode: sum the partials over the splits, normalise."""

from __future__ import annotations

from ....ir.graph import Graph, GraphNode, Iterator
from ....ir.ops import Accumulate, ReduceSum
from ....ir.graph_utils import gather_all_nodes, lock_graph
from ..kinds import PARTIAL_ROWSUM, PARTIAL_VALUE
from ..stages import _normalise, _run_passes
from ..tensor_maps import _head_offsets, _output_maps, _output_tinfo, _whole


def _build_decode_combine(
    spec,
    heads: int,
    group: int,
    head_dim: int,
    out_dtype: str,
    accum_dtype: str,
    comb,
    grid_order,
    batch_dim: str,
    rasterization,
) -> Graph:
    """The combine half of a split-KV decode: sum the partials over the splits, normalise.

    One block owns ``group`` rows (a kv head's query heads; the whole head block for MLA)
    by ``comb.cols`` columns of one request, reads the ``split`` partials of that tile in
    ``comb.split_step``-deep slices through a loop tier, and applies the single-kernel
    layer's own epilogue -- the floored divide and the log2 -- to the sums.
    """
    comb.validate(head_dim)
    split, step, cols = comb.split, comb.split_step, comb.cols
    n_col_blocks = head_dim // cols
    n_row_blocks = heads // group

    if step > 1:
        rowsum_c = ReduceSum(
            (step, group),
            axis=0,
            dtype=accum_dtype,
            out_dtype=accum_dtype,
            name="rowsum_combine_0",
        )
        value_c = ReduceSum(
            (step, group, cols),
            axis=0,
            dtype=accum_dtype,
            out_dtype=accum_dtype,
            name="value_combine_0",
        )
    else:
        # One partial per iteration: the split axis of the slice is a unit extent, which
        # codegen squeezes away, so there is no axis left to reduce -- the sum over the
        # splits is the loop itself, and each iteration adds its slice elementwise.
        rowsum_c = Accumulate(
            (group,), dtype=accum_dtype, out_dtype=accum_dtype, name="rowsum_combine_0"
        )
        value_c = Accumulate(
            (group, cols), dtype=accum_dtype, out_dtype=accum_dtype, name="value_combine_0"
        )
    ph_d = rowsum_c.inputs[0].src_node
    ph_o = value_c.inputs[0].src_node
    ph_d.name = f"PlaceHolder_{PARTIAL_ROWSUM}"
    ph_o.name = f"PlaceHolder_{PARTIAL_VALUE}"

    inner = Graph(nodes=gather_all_nodes(rowsum_c, value_c))
    inner.extent = split // step
    inner.loop_kind = comb.loop_kind
    inner.loop_stages = comb.loop_stages if comb.loop_kind == "pipelined" else None
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    d_id = inner.accum_nodes.index(rowsum_c)
    o_id = inner.accum_nodes.index(value_c)
    inner_d = inner.binding[ph_d.name]
    inner_o = inner.binding[ph_o.name]

    div, log2, outputs = _normalise(
        loop_node, (value_c, o_id), (rowsum_c, d_id), (group, cols),
        accum_dtype, out_dtype, spec.emit_lse,
    )

    main = Graph(nodes=[ph_d, ph_o] + gather_all_nodes(loop_node))
    main.dyn_dims = [batch_dim]
    main.n_threads = comb.n_threads
    main.pass_configs = dict(spec.pass_configs)
    main.rasterization = rasterization
    # The same grid as the partial's, without its split axis, plus a column axis when a
    # block owns less than the head dim. A single-kv-head operator (MLA) has no head axis.
    batch_first = tuple(grid_order) == ("batch", "head")
    if n_row_blocks == 1:
        lead = (f"{batch_dim}",)
    elif batch_first:
        lead = (f"{batch_dim}", f"{n_row_blocks}")
    else:
        lead = (f"{n_row_blocks}", f"{batch_dim}")
    main.extent = lead + ((f"{n_col_blocks}",) if n_col_blocks > 1 else ())

    def grid(*iters):
        """(request, row block, column offset) whatever the launch order."""
        if n_row_blocks == 1:
            b, h, rest = iters[0], "0", iters[1:]
        else:
            b, h = (iters[0], iters[1]) if batch_first else (iters[1], iters[0])
            rest = iters[2:]
        c = f"{cols} * {rest[0]}" if rest else "0"
        return b, h, c

    def _d_offset(*iters):
        b, h, _ = grid(*iters)
        return ("0", f"{b}", f"{group} * {h}")

    def _o_offset(*iters):
        b, h, c = grid(*iters)
        return ("0", f"{b}", f"{group} * {h}", c)

    out_offset, lse_offset = _head_offsets(grid, group, cols=True)
    main.tmap[ph_d] = Iterator(
        _d_offset, shape=(split, batch_dim, heads), tile_size=(split, 1, group)
    )
    main.tmap[ph_o] = Iterator(
        _o_offset,
        shape=(split, batch_dim, heads, head_dim),
        tile_size=(split, 1, group, cols),
    )
    # One entry per accumulator, in the loop tier's own accumulator order.
    main.tmap[loop_node] = [
        _whole((group,)) if node is rowsum_c else _whole((group, cols))
        for node in inner.accum_nodes
    ]
    _output_maps(
        main,
        div,
        log2,
        outputs,
        Iterator(out_offset, shape=(batch_dim, heads, head_dim), tile_size=(1, group, cols)),
        Iterator(lse_offset, shape=(batch_dim, heads), tile_size=(1, group)),
    )

    # A tile spanning every split is the whole slice, which codegen hands down unsliced
    # and therefore expects at a literal zero offset.
    s_off = (lambda i: "0") if step == split else (lambda i: f"{step} * {i}")
    inner.tmap[inner_d] = Iterator(
        lambda i: (s_off(i), "0"), shape=(split, group), tile_size=(step, group)
    )
    inner.tmap[inner_o] = Iterator(
        lambda i: (s_off(i), "0", "0"),
        shape=(split, group, cols),
        tile_size=(step, group, cols),
    )
    inner.tmap[rowsum_c] = _whole((group,))
    inner.tmap[value_c] = _whole((group, cols))

    main.tinfo_map = {
        ph_d: (ph_d.name, "g"),
        ph_o: (ph_o.name, "g"),
        loop_node: ("t_GraphNode_0", "r"),
    }
    _output_tinfo(main, div, log2, outputs)
    inner.tinfo_map = {
        ph_d: (ph_d.name, "g"),
        ph_o: (ph_o.name, "g"),
        inner_d: (inner_d.name, comb.input_scope),
        inner_o: (inner_o.name, comb.input_scope),
        rowsum_c: ("t_rowsum_combine_0", "r"),
        value_c: ("t_value_combine_0", "r"),
    }
    return _run_passes(main, inner)
