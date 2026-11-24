"""Building blocks the access-layer builders share."""

from __future__ import annotations

from types import SimpleNamespace

from tilelang.carver.roller import Edge

from ...codegen import passes
from ...ir.graph import AccumEdge, Graph, GraphNode, OutputNode, PlaceHolderNode, ScalarScan
from ...ir.ops import (
    NEG_FLT_MAX,
    CausalMaskedExp,
    CausalMaskedScore,
    Copy,
    Div,
    Exp2,
    IndexGather,
    IndptrTileScan,
    Lse2,
    MaskedExp,
    MaskedScore,
    Matmul,
    PagedGather,
    RangeGather,
    ReduceMax,
    ReduceSum,
    RowGather,
)
from ...ir.ops.gather import BUFFER_BOUND
from ...ir.rolling import rolling_nodes
from .kinds import REGISTER_P_KINDS

# exp2 is a single instruction where exp is a call, so the scale rides in the log2
# domain: exp(x * s) == exp2(x * s * LOG2E).
LOG2E = 1.4426950408889634
# Floors the softmax denominator. A request whose KV window is empty runs zero KV tiles,
# leaving a zero numerator over a zero denominator; the floor turns that 0/0 into the 0
# the operator defines rather than the NaN a benchmark's sanity check would reject.
DENOM_FLOOR = 1e-30


def _connect(src, dst, src_id, dst_id) -> None:
    edge = Edge(src, dst, src_id, dst_id)
    src._out_edges.append(edge)
    dst.set_inputs(dst_id, edge)


def _accum_edge(loop_node, dst, src_id, dst_id, accum) -> None:
    """Feed ``dst``'s input ``dst_id`` from the loop tier's accumulator ``accum``."""
    edge = AccumEdge(loop_node, dst, src_id, dst_id, accum_node=accum)
    loop_node._out_edges.append(edge)
    dst.set_inputs(dst_id, edge)


def _int32():
    import tvm

    return tvm.DataType("int32")


def _index_input(name: str) -> PlaceHolderNode:
    """An int32 kernel input read for its values, not gathered from, so it stays out of
    the *_iter call."""
    node = PlaceHolderNode(name=name)
    node.set_shape((1,))
    node.set_dtype(_int32())
    node.index_only = True
    node.name = f"PlaceHolder_{name}"
    return node


def _output(node, dtype, i: int) -> OutputNode:
    out = OutputNode(node, dtype=dtype)
    out.name = f"OutputNode_{i}"
    return out


def _passthrough(*names) -> dict:
    """Loop-tier ``smap`` entries that forward a block scalar unchanged."""
    return {name: (lambda v: lambda i: v)(name) for name in names}


def _exp(dense, rows: int, causal: bool = False, **kwargs):
    """The masked exp of the exact softmax: ``exp2(score - max)``, the literal 0 where masked.

    Its operand is the masked score (``_score``), already scaled into the log2 domain, and
    its second input that score's running row max; so it takes the mask scalars alone, the
    scale being the score's.
    """
    cls, mask = (
        (CausalMaskedExp, ("tile_base", "valid_len", "diag"))
        if causal
        else (MaskedExp, ("tile_base", "valid_len"))
    )
    return cls(
        (rows, dense.kv_tile),
        scale=None,
        scalar_args=mask,
        shift=True,
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="exp_0",
        n_batch_dims=0,
        **kwargs,
    )


def _score(dense, rows: int, causal: bool = False, **kwargs):
    """The exp operand of the exact softmax: the score scaled into the log2 domain, masked
    columns at the finite value the running max starts from."""
    cls, args = (
        (CausalMaskedScore, ("tile_base", "valid_len", "diag", "scale2"))
        if causal
        else (MaskedScore, ("tile_base", "valid_len", "scale2"))
    )
    return cls(
        (rows, dense.kv_tile),
        scalar_args=args,
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="score_0",
        n_batch_dims=0,
        **kwargs,
    )


def _online_softmax(dense, rows: int, scores, causal: bool = False, **mask):
    """The exact (online) softmax of ``scores``, a loop tier's tile of raw attention scores.

        score   = _score     <- scores           the exp operand, masked, in the log2 domain
        row_max = ReduceMax  <- score            carried across the loop, from -FLT_MAX
        exp     = _exp       <- (score, row_max)                    P = exp2(score - max)
        alpha   = exp2(max_prev - row_max)       what the loop's accumulators are scaled by

    Every layer builds its softmax here, so the max is always taken after scaling and
    masking, and the mask of the score is the mask of the exp (``mask`` goes to both).
    Returns the nodes (``score``, ``row_max``, ``exp``, ``max_prev``, ``max_step``,
    ``alpha``). The caller feeds ``exp`` to its accumulators, declares the update on its
    loop tier before the ``GraphNode`` exists (``_declare_rolling``), and then carries the
    max (``loop_node.carry(row_max)``).
    """
    score = _score(dense, rows, causal, **mask)
    exp = _exp(dense, rows, causal, **mask)
    # The max is taken over the scaled, masked score and is float32 like it: its start
    # value, the most negative finite float32, is no value of a narrower type.
    row_max = ReduceMax(
        (rows, dense.kv_tile),
        axis=-1,
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="exp_0_max",
        accum_init=NEG_FLT_MAX,
    )
    max_prev, max_step, alpha = rolling_nodes(
        row_max, stem="exp_0", rows=(rows,), n_batch_dims=0, exp_cls=Exp2
    )
    _connect(scores, score, 0, 0)
    _connect(score, row_max, 0, 0)
    _connect(score, exp, 0, 0)
    _connect(row_max, exp, 0, 1)
    return SimpleNamespace(
        score=score, row_max=row_max, exp=exp, max_prev=max_prev, max_step=max_step,
        alpha=alpha,
    )


def _declare_rolling(inner: Graph, rolling, *accumulators) -> None:
    """Declare the rolling update on a loop tier: the snapshot of the max, and the factor
    every one of ``accumulators`` is multiplied by before it takes this tile.

    Before the tier's ``GraphNode`` exists: the factor has no dataflow consumer, and must
    not be taken for an accumulator. Every accumulator named here sums ``exp2(s - m)``
    terms, so every one is rescaled when the max moves.
    """
    inner.lag_nodes = [rolling.max_prev]
    inner.accum_scale = {acc: rolling.alpha for acc in accumulators}


def _accumulators(loop_node, denom, value_mma, rolling) -> tuple:
    """``(accumulator, index in the loop tier)`` for the softmax denominator, the value
    matmul and the carried max, in that order.

    The first two are accumulators by dataflow. The max is read inside the loop, so it is
    carried by declaration; it is appended, which leaves the other two indices as they are.
    """
    accum = loop_node.graph.accum_nodes
    found = ((denom, accum.index(denom)), (value_mma, accum.index(value_mma)))
    return (*found, (rolling.row_max, loop_node.carry(rolling.row_max)))


def _gqa_body(spec, dense, rows: int, causal: bool = False, **mask) -> tuple:
    """Q @ K^T -> exact softmax -> {row sum, @ V}:
    ``(score_mma, exp, rowsum, value_mma, rolling)``, the last being ``_online_softmax``'s
    nodes; ``causal`` and ``mask`` select and parametrise the softmax mask.

    P is staged through shared for the value matmul unless the layer is one of
    ``REGISTER_P_KINDS``, where tilelang feeds the fragment to the second gemm in place
    (wgmma_rs).
    """
    bn, head_dim = dense.kv_tile, dense.head_dim
    score_mma = Matmul(
        (rows, bn, head_dim),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_0",
        transpose_B=True,
        warp_policy=spec.warp_policy,
    )
    rowsum = ReduceSum(
        (rows, bn),
        axis=-1,
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="rowsum_0",
    )
    value_mma = Matmul(
        (rows, head_dim, bn),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_1",
        warp_policy=spec.warp_policy,
    )
    if spec.kind not in REGISTER_P_KINDS:
        value_mma.stage_fragment_operands_in_shared = (0,)
    rolling = _online_softmax(dense, rows, score_mma, causal, **mask)
    _connect(rolling.exp, rowsum, 0, 0)
    _connect(rolling.exp, value_mma, 0, 0)
    return score_mma, rolling.exp, rowsum, value_mma, rolling


def _paged_gathers(spec, dense, head_scalar: str) -> tuple:
    """The K and V row gathers off a token-unit page table, sharing one index input.

    They read the page table in place rather than through a tile resolved into shared: the
    thread that gathers row g0 is not the one that would have resolved entry g0, so a
    resolved tile costs a barrier before it can be read and another before it is reused.
    """
    gather_kwargs = dict(
        source_rows=spec.pages_dim,
        index_extent=spec.index_dim,
        page_size=spec.page_size,
        paged_axes=True,
        source_heads=spec.num_kv_heads,
        head_scalar=True,
        clamp_index=BUFFER_BOUND,
        clamp_row=BUFFER_BOUND,
        scalar_args=("kv_pos0", head_scalar),
        dtype=dense.dtype,
        out_dtype=dense.dtype,
    )
    shape = (dense.kv_tile, dense.head_dim)
    gather_k = PagedGather(shape, name="paged_gather_k", **gather_kwargs)
    gather_v = PagedGather(shape, name="paged_gather_v", **gather_kwargs)
    return gather_k, gather_v


def _wire_gathers(gather_k, gather_v, score_mma, value_mma) -> tuple:
    """Feed the matmuls from the gathers; ``(ph_k, ph_v, ph_idx)``.

    One page table feeds both gathers, so K's index placeholder is rewired to V rather than
    each gather declaring its own copy of the same kernel input.
    """
    _connect(gather_k, score_mma, 0, 1)
    _connect(gather_v, value_mma, 0, 1)
    ph_idx = gather_k.inputs[1].src_node
    _connect(ph_idx, gather_v, 0, 1)
    return gather_k.inputs[0].src_node, gather_v.inputs[0].src_node, ph_idx


def _name_inputs(spec, ph_q, ph_k, ph_v, ph_idx=None) -> None:
    ph_q.name = f"PlaceHolder_{spec.query_input}"
    ph_k.name = f"PlaceHolder_{spec.cache_inputs[0]}"
    ph_v.name = f"PlaceHolder_{spec.cache_inputs[1]}"
    if ph_idx is not None:
        ph_idx.name = f"PlaceHolder_{spec.index_input}"


def _normalise(
    loop_node, rowsum, value, row_max, shape, accum_dtype, out_dtype, emit_lse,
    div_cls=Div, lse_cls=Lse2, **kwargs,
) -> tuple:
    """The floored divide and, with ``emit_lse``, the log-sum-exp ``log2(l) + m``.

    ``rowsum``/``value``/``row_max`` are ``(accumulator, index in the loop tier)``, as
    ``_accumulators`` returns them; ``kwargs`` go to both the divide and the lse (row
    predication). Returns ``(div, lse, outputs)``.

    The row sum is a sum of ``exp2(s - m)``, so the lse adds the carried max back. It reads
    the row sum unfloored -- a row with nothing live has ``l = 0`` and the max still at its
    finite start, which gives the -inf the operator defines -- and the divide keeps its
    floor, which is a guard for exactly that row and nothing else: a row with any live key
    has a row sum of at least 1.
    """
    div = div_cls(
        shape,
        broadcast=0,
        denom_floor=DENOM_FLOOR,
        dtype=accum_dtype,
        out_dtype=out_dtype,
        name="div_0",
        n_batch_dims=0,
        **kwargs,
    )
    _accum_edge(loop_node, div, value[1], 0, value[0])
    _accum_edge(loop_node, div, rowsum[1], 1, rowsum[0])
    outputs = [_output(div, out_dtype, 0)]
    lse = None
    if emit_lse:
        lse = lse_cls(
            shape[:1],
            dtype=accum_dtype,
            out_dtype=accum_dtype,
            name="lse_0",
            n_batch_dims=0,
            **kwargs,
        )
        _accum_edge(loop_node, lse, rowsum[1], 0, rowsum[0])
        _accum_edge(loop_node, lse, row_max[1], 1, row_max[0])
        outputs.append(_output(lse, accum_dtype, 1))
    return div, lse, outputs


def _partial_stores(rows: int, cols: int, accum_dtype: str):
    """The three raw stores of a split-KV partial: ``(store_d, store_o, store_m)``.

    Row sum first, then the value matmul: the order the dense split-K expected kernels use, and the
    order the combine declares its inputs in -- the fused emitter pairs them positionally.
    The max comes last. A split's row sum and values are sums of ``exp2(s - m)`` for the
    split's own running max ``m``, so they mean nothing without it: the combine weighs
    every split by ``exp2(m - M)``, ``M`` being the max over the splits. It leaves the
    block as it is, in float32 -- a split that ran no tile stores the finite value the max
    starts from, next to a zero row sum.
    """
    store_d = Copy(
        (rows,),
        dtype=accum_dtype,
        out_dtype=accum_dtype,
        name="partial_rowsum",
        n_batch_dims=0,
    )
    store_o = Copy(
        (rows, cols),
        dtype=accum_dtype,
        out_dtype=accum_dtype,
        name="partial_value",
        n_batch_dims=0,
    )
    store_m = Copy(
        (rows,),
        dtype=accum_dtype,
        out_dtype=accum_dtype,
        name="partial_max",
        n_batch_dims=0,
    )
    return store_d, store_o, store_m


def _decode_epilogue(spec, dense, loop_node, rowsum, value, row_max, split):
    """The GQA decode layers' epilogue: normalise and emit the lse, or store raw partials.

    ``rowsum``/``value``/``row_max`` are ``(accumulator, index in the loop tier)``. Returns
    ``(div, lse, stores, outputs)``; the unused half is None.
    """
    group, head_dim = spec.group, dense.head_dim
    if split is None:
        div, lse, outputs = _normalise(
            loop_node,
            rowsum,
            value,
            row_max,
            (group, head_dim),
            dense.accum_dtype,
            dense.out_dtype,
            spec.emit_lse,
        )
        return div, lse, None, outputs
    stores = _partial_stores(group, head_dim, dense.accum_dtype)
    outputs = []
    for i, (store, (accum, accum_id)) in enumerate(zip(stores, (rowsum, value, row_max))):
        _accum_edge(loop_node, store, accum_id, 0, accum)
        outputs.append(_output(store, dense.accum_dtype, i))
    return None, None, stores, outputs


def _decode_grid(spec, split):
    """A decode grid over (request, kv head) in ``spec.grid_order``, plus a trailing split
    axis for a split-KV partial: ``(extent, grid, scalar)``.

    ``grid`` names the launch indices ``(request, head, *rest)`` whatever the order;
    ``scalar`` adapts an ``smap`` builder written against those names.
    """
    batch, n_kv = spec.batch_dim, spec.num_kv_heads
    batch_first = tuple(spec.grid_order) == ("batch", "head")
    extent = (f"{batch}", f"{n_kv}") if batch_first else (f"{n_kv}", f"{batch}")
    if split is not None:
        extent += (f"{split}",)

    def grid(*iters):
        lead = iters[:2] if batch_first else tuple(reversed(iters[:2]))
        return (*lead, *iters[2:])

    def scalar(build):
        return lambda *iters: build(*grid(*iters))

    return extent, grid, scalar


def _worklist(ph_qo, ph_kv, rows: int):
    """The per-block expressions of a prefill work list.

    ``_wl[0]`` is the request and ``_wl[1]`` the tile within it, both from the device-side
    scan; ``rows`` is the query tokens one block owns. ``diag`` is the end-aligned causal
    bound for the tile's first row, and ``live`` the KV window causality leaves the whole
    tile (``live <= 0``: every row's window is empty).
    """
    q_beg = f"{ph_qo.name}[_wl[0]]"
    q_end = f"{ph_qo.name}[_wl[0] + 1]"
    kv_beg = f"{ph_kv.name}[_wl[0]]"
    kv_end = f"{ph_kv.name}[_wl[0] + 1]"
    q_len = f"({q_end} - {q_beg})"
    kv_len = f"({kv_end} - {kv_beg})"
    tile_base = f"(_wl[1] * {rows})"
    n_rows = f"T.min({rows}, {q_len} - {tile_base})"
    diag = f"({kv_len} - {q_len} + {tile_base})"
    live = f"T.min({kv_len}, {diag} + {n_rows})"
    return SimpleNamespace(
        q_beg=q_beg,
        kv_beg=kv_beg,
        q_len=q_len,
        tile_base=tile_base,
        n_rows=n_rows,
        diag=diag,
        live=live,
        q_abs=f"{q_beg} + {tile_base}",
    )


def _worklist_scan(ph_qo, rows: int, indptr_dim: str) -> list:
    """The prologue that scans the query indptr into a block's ``_wl`` (request, tile)."""
    return [
        ScalarScan(
            op=IndptrTileScan(rows=rows),
            buffer="_wl",
            size=2,
            inputs=(ph_qo,),
            scalar_args=lambda i, *_: (i, indptr_dim),
        )
    ]


def _split_range_scalars(split: int, bn: int, valid_len, scalar) -> dict:
    """``smap`` entries (``tile0``, ``my_trip``, ``split_end``) giving a split block its KV tiles.

    Tiles are dealt ``ceil(n_tiles / split)`` per split; a trailing split may get none and
    stores zero partials.
    """
    n_tiles = lambda b: f"T.ceildiv({valid_len(b)}, {bn})"  # noqa: E731
    per_split = lambda b: f"T.ceildiv({n_tiles(b)}, {split})"  # noqa: E731
    return {
        "tile0": scalar(lambda b, h, s: f"{s} * {per_split(b)}"),
        "my_trip": scalar(
            lambda b, h, s: f"T.max(T.min({per_split(b)}, {n_tiles(b)} - {s} * {per_split(b)}), 0)"
        ),
        "split_end": scalar(
            lambda b, h, s: f"T.min({valid_len(b)}, {bn} * ({s} * {per_split(b)} + {per_split(b)}))"
        ),
    }


def _run_passes(main: Graph, inner: Graph = None, index_tile=None) -> Graph:
    """Run the annotation passes (not ``annotate_tinfo``: the builders write ``tinfo_map``).

    The index tile is relabelled linear: ``annotate_tc_layout`` reaches the int32 page table
    through the gathers and would mark it tensor-core.
    """
    _assert_loop_kind_matches_its_source(main)
    main = passes.annotate_args(main, is_main=True)
    main = passes.annotate_tc_layout(main)
    main = passes.annotate_fragment_layout_conflicts(main)
    if index_tile is not None:
        inner.tlayout_map[index_tile] = "linear"
    return main


def _assert_loop_kind_matches_its_source(main: Graph) -> None:
    """A nested tier that pipelines a gathered tile is a bug: pipelining reorders an unbuffered read.

    Only nested tiers and their own nodes are checked (a hoisted Q gather is not in the loop).
    """
    gathers = (PagedGather, RowGather, RangeGather, IndexGather)
    tiers = [n.graph for n in main.nodes if isinstance(n, GraphNode)]
    while tiers:
        tier = tiers.pop()
        gathered = [n.name for n in tier.nodes if isinstance(n, gathers)]
        if tier.loop_kind == "pipelined":
            assert not gathered, (
                f"a pipelined loop stages {gathered} with a gather; pipelining reorders "
                "reads the gather has not buffered, so this loop must be serial (a gather "
                "layer overlaps with prefetch_nodes instead)"
            )
        else:
            assert not tier.region_args, (
                f"a {tier.loop_kind!r} loop hands down region_args "
                f"{list(tier.region_args)}; the BufferRegion exists so tilelang can reach "
                "for TMA, whose mbarriers need a pipelined producer/consumer body to "
                "supply arrivals"
            )
        tiers += [n.graph for n in tier.nodes if isinstance(n, GraphNode)]
