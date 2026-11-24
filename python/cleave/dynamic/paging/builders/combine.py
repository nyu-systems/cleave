"""The combine half of a split-KV decode: weigh the partials, sum them over the splits,
normalise."""

from __future__ import annotations

from ....ir.graph import Graph, GraphNode, Iterator
from ....ir.ops import (
    NEG_FLT_MAX,
    Accumulate,
    AccumulateMax,
    Exp2,
    Mul,
    ReduceMax,
    ReduceSum,
    Sub,
)
from ....ir.graph_utils import gather_all_nodes, lock_graph
from ....ir.rolling import rolling_nodes
from ..kinds import PARTIAL_MAX, PARTIAL_ROWSUM, PARTIAL_VALUE
from ..stages import _connect, _normalise, _run_passes
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
    """The combine half of a split-KV decode: weigh the partials, sum them, normalise.

    One block owns ``group`` rows (a kv head's query heads; the whole head block for MLA)
    by ``comb.cols`` columns of one request, reads the ``split`` partials of that tile in
    ``comb.split_step``-deep slices through a loop tier, and applies the single-kernel
    layer's own epilogue -- the floored divide and the lse -- to the sums.

    A split's row sum ``l_i`` and values ``o_i`` are sums of ``exp2(s - m_i)`` for that
    split's own running max ``m_i``, the third partial. The combine is the same online
    softmax one level up, with a split where the single kernel has a KV tile: it carries
    ``M``, the max of the ``m_i`` read so far, and per iteration

        M_prev = M                                    lag node, before M is updated
        M      = max(M, max_i m_i)                    starts at -FLT_MAX
        w_i    = exp2(m_i - M)                        the slice's weights
        alpha  = exp2(M_prev - M)
        L      = L * alpha + sum_i l_i * w_i
        O      = O * alpha + sum_i o_i * w_i

    and after the loop ``out = O / max(L, floor)``, ``lse = log2(L) + M``. A split that
    ran no tile stores ``l = 0``, ``o = 0`` and the finite value its max started from: its
    weight is ``exp2(-FLT_MAX - M) = 0`` next to any split that ran, and ``exp2(0)`` times
    zero when none did, which leaves ``L = 0`` and ``M`` at its start -- the 0 and the -inf
    the operator defines for a request with no keys.
    """
    comb.validate(head_dim)
    split, step, cols = comb.split, comb.split_step, comb.cols
    n_col_blocks = head_dim // cols
    n_row_blocks = heads // group

    f32 = dict(dtype=accum_dtype, out_dtype=accum_dtype)
    if step > 1:
        # The slice keeps its split axis: the max and the two sums fold it.
        rows, tile = (step, group), (step, group, cols)
        max_c = ReduceMax(rows, axis=0, name="combine_max", accum_init=NEG_FLT_MAX, **f32)
        rowsum_c = ReduceSum(rows, axis=0, name="rowsum_combine_0", **f32)
        value_c = ReduceSum(tile, axis=0, name="value_combine_0", **f32)
        # m_i[s, g] - M[g]: the max is one value per row, broadcast over the slice's splits.
        shift = Sub(rows, broadcast=1, name="combine_shift", n_batch_dims=0, **f32)
    else:
        # One partial per iteration: the split axis of the slice is a unit extent, which
        # codegen squeezes away, so there is no axis left to reduce -- the max and the sums
        # over the splits are the loop itself, and each iteration folds its slice in
        # elementwise.
        rows, tile = (group,), (group, cols)
        max_c = AccumulateMax(rows, name="combine_max", accum_init=NEG_FLT_MAX, **f32)
        rowsum_c = Accumulate(rows, name="rowsum_combine_0", **f32)
        value_c = Accumulate(tile, name="value_combine_0", **f32)
        shift = Sub(rows, name="combine_shift", n_batch_dims=0, **f32)
    weight = Exp2(rows, name="combine_weight", n_batch_dims=0, **f32)
    rowsum_w = Mul(rows, name="rowsum_weighted_0", n_batch_dims=0, **f32)
    # The weight lacks only the column axis of the values: broadcast over it.
    value_w = Mul(tile, broadcast=0, name="value_weighted_0", n_batch_dims=0, **f32)
    max_prev, max_step, alpha = rolling_nodes(
        max_c, stem="combine", rows=(group,), n_batch_dims=0, exp_cls=Exp2
    )

    ph_d = rowsum_w.inputs[0].src_node
    ph_o = value_w.inputs[0].src_node
    ph_m = max_c.inputs[0].src_node
    ph_d.name = f"PlaceHolder_{PARTIAL_ROWSUM}"
    ph_o.name = f"PlaceHolder_{PARTIAL_VALUE}"
    ph_m.name = f"PlaceHolder_{PARTIAL_MAX}"
    # One max slice feeds the running max and the weights.
    _connect(ph_m, shift, 0, 0)
    _connect(max_c, shift, 0, 1)
    _connect(shift, weight, 0, 0)
    _connect(weight, rowsum_w, 0, 1)
    _connect(weight, value_w, 0, 1)
    _connect(rowsum_w, rowsum_c, 0, 0)
    _connect(value_w, value_c, 0, 0)

    inner = Graph(nodes=gather_all_nodes(rowsum_c, value_c))
    # Declared before the GraphNode exists, as in the partial's loop: the factor has no
    # dataflow consumer and must not be taken for an accumulator.
    inner.lag_nodes = [max_prev]
    inner.accum_scale = {rowsum_c: alpha, value_c: alpha}
    inner.extent = split // step
    inner.loop_kind = comb.loop_kind
    inner.loop_stages = comb.loop_stages if comb.loop_kind == "pipelined" else None
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    d_id = inner.accum_nodes.index(rowsum_c)
    o_id = inner.accum_nodes.index(value_c)
    # Read inside the loop (by the weights), so carried by declaration; appended last.
    m_id = loop_node.carry(max_c)
    inner_d = inner.binding[ph_d.name]
    inner_o = inner.binding[ph_o.name]
    inner_m = inner.binding[ph_m.name]

    div, lse, outputs = _normalise(
        loop_node, (rowsum_c, d_id), (value_c, o_id), (max_c, m_id), (group, cols),
        accum_dtype, out_dtype, spec.emit_lse,
    )

    main = Graph(nodes=[ph_d, ph_o, ph_m] + gather_all_nodes(loop_node))
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
    for ph in (ph_d, ph_m):  # one value per (split, request, head)
        main.tmap[ph] = Iterator(
            _d_offset, shape=(split, batch_dim, heads), tile_size=(split, 1, group)
        )
    main.tmap[ph_o] = Iterator(
        _o_offset,
        shape=(split, batch_dim, heads, head_dim),
        tile_size=(split, 1, group, cols),
    )
    # One entry per accumulator, in the loop tier's own accumulator order.
    main.tmap[loop_node] = [
        _whole((group, cols)) if node is value_c else _whole((group,))
        for node in inner.accum_nodes
    ]
    _output_maps(
        main,
        div,
        lse,
        outputs,
        Iterator(out_offset, shape=(batch_dim, heads, head_dim), tile_size=(1, group, cols)),
        Iterator(lse_offset, shape=(batch_dim, heads), tile_size=(1, group)),
    )

    # A tile spanning every split is the whole slice, which codegen hands down unsliced
    # and therefore expects at a literal zero offset.
    s_off = (lambda i: "0") if step == split else (lambda i: f"{step} * {i}")
    for inner_ph in (inner_d, inner_m):
        inner.tmap[inner_ph] = Iterator(
            lambda i: (s_off(i), "0"), shape=(split, group), tile_size=(step, group)
        )
    inner.tmap[inner_o] = Iterator(
        lambda i: (s_off(i), "0", "0"),
        shape=(split, group, cols),
        tile_size=(step, group, cols),
    )
    for node in (shift, weight, rowsum_w):
        inner.tmap[node] = _whole(rows)
    inner.tmap[value_w] = _whole(tile)
    for node in (rowsum_c, max_c, max_prev, max_step, alpha):
        inner.tmap[node] = _whole((group,))
    inner.tmap[value_c] = _whole((group, cols))

    main.tinfo_map = {
        ph_d: (ph_d.name, "g"),
        ph_o: (ph_o.name, "g"),
        ph_m: (ph_m.name, "g"),
        loop_node: ("t_GraphNode_0", "r"),
    }
    _output_tinfo(main, div, lse, outputs)
    inner.tinfo_map = {
        ph_d: (ph_d.name, "g"),
        ph_o: (ph_o.name, "g"),
        ph_m: (ph_m.name, "g"),
        inner_d: (inner_d.name, comb.input_scope),
        inner_o: (inner_o.name, comb.input_scope),
        # The max slice has two readers, the running max and the weights: it is loaded
        # once, into a fragment both read, whatever the staging of the other two.
        inner_m: (inner_m.name, "r"),
        max_c: ("t_combine_max", "r"),
        max_prev: ("t_combine_max_prev", "r"),
        max_step: ("t_combine_max_step", "r"),
        alpha: ("t_combine_alpha", "r"),
        shift: ("t_combine_shift", "r"),
        weight: ("t_combine_weight", "r"),
        rowsum_w: ("t_rowsum_weighted_0", "r"),
        value_w: ("t_value_weighted_0", "r"),
        rowsum_c: ("t_rowsum_combine_0", "r"),
        value_c: ("t_value_combine_0", "r"),
    }
    return _run_passes(main, inner)
