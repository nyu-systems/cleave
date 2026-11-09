"""The paged multi-head latent attention layers: decode, and causal prefill."""

from __future__ import annotations

from ....ir.graph import Graph, GraphNode, Iterator
from ....ir.ops import (
    Accumulate,
    Add,
    Div,
    IndexGather,
    Log2,
    MaskedDiv,
    MaskedLog2,
    Matmul,
    PagedGather,
    ReduceSum,
    RowGather,
)
from ....ir.ops.gather import BUFFER_BOUND
from ....ir.graph_utils import gather_all_nodes, lock_graph
from ..stages import (
    DENOM_FLOOR,
    LOG2E,
    _accum_edge,
    _connect,
    _exp,
    _index_input,
    _output,
    _partial_stores,
    _passthrough,
    _run_passes,
    _split_range_scalars,
    _worklist,
    _worklist_scan,
)
from ..tensor_maps import _mla_tensor_maps


def _mla_loop_tier(spec, dense, exp):
    """The nodes inside the KV loop, shared by both MLA layers.

    Only the softmax mask differs between them, so ``exp`` is passed in: the decode layer
    masks the tail of the request's KV window, the prefill layer also masks the diagonal.
    Everything else -- the shared page-table window, the two gathers, the two score
    matmuls and their sum, the (M, N) denominator accumulator, and the value matmul reading
    the *same* gathered ckv tile -- is identical.
    """
    mq, bn = dense.query_rows, dense.kv_tile
    ckv, kpe = dense.head_dim_ckv, dense.head_dim_kpe
    window = bn * spec.index_window_tiles

    idx_window = IndexGather(
        (window,),
        index_extent=spec.index_dim,
        row_stride=1,
        page_size=spec.page_size,
        clamp_index=BUFFER_BOUND,
        scalar_args=("kv_base", "win_base", "valid_len"),
        name="idx_window",
    )
    # MLA has a single kv head, so the cache carries only (page, slot, dim) and the gather
    # picks no head; page_size 1 leaves no slot axis for the macro either.
    gather_kwargs = dict(
        source_rows=spec.pages_dim,
        index_extent=window,
        page_size=spec.page_size,
        paged_axes=True,
        clamp_index=BUFFER_BOUND,
        clamp_row=BUFFER_BOUND,
        scalar_args=("win_off",),
        dtype=dense.dtype,
        out_dtype=dense.dtype,
    )
    gather_ckv = PagedGather((bn, ckv), name="paged_gather_ckv", **gather_kwargs)
    gather_kpe = PagedGather((bn, kpe), name="paged_gather_kpe", **gather_kwargs)
    ckv_mma = Matmul(
        (mq, bn, ckv),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_0",
        transpose_B=True,
        warp_policy=spec.warp_policy,
    )
    pe_mma = Matmul(
        (mq, bn, kpe),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_pe",
        transpose_B=True,
        warp_policy=spec.warp_policy,
    )
    add = Add(
        (mq, bn),
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="add_0",
        n_batch_dims=0,
    )
    # The denominator is accumulated as a whole (M, N) tile and folded to a row vector once,
    # after the loop, rather than folded every iteration. A row fold is cross-lane work and
    # MLA runs the most iterations of any of these operators, so paying it once per block
    # instead of once per tile is worth the registers the tile costs.
    psum = Accumulate(
        (mq, bn), dtype=dense.accum_dtype, out_dtype=dense.accum_dtype, name="psum_0"
    )
    value_mma = Matmul(
        (mq, ckv, bn),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_1",
        warp_policy=spec.warp_policy,
    )
    value_mma.stage_fragment_operands_in_shared = (0,)

    _connect(gather_ckv, ckv_mma, 0, 1)
    _connect(gather_kpe, pe_mma, 0, 1)
    _connect(ckv_mma, add, 0, 0)
    _connect(pe_mma, add, 0, 1)
    _connect(add, exp, 0, 0)
    _connect(exp, psum, 0, 0)
    _connect(exp, value_mma, 0, 0)
    # The compressed cache is also the value tensor: MLA has no separate V, so the value
    # matmul reads the tile the score matmul already gathered rather than a second one.
    _connect(gather_ckv, value_mma, 0, 1)

    ph_qn = ckv_mma.inputs[0].src_node
    ph_ckv = gather_ckv.inputs[0].src_node
    # One window feeds both gathers, so the kpe gather is rewired to it rather than
    # declaring a second copy of the same page table.
    _connect(idx_window, gather_ckv, 0, 1)
    _connect(idx_window, gather_kpe, 0, 1)
    ph_idx = idx_window.inputs[0].src_node
    ph_qp = pe_mma.inputs[0].src_node
    ph_kpe = gather_kpe.inputs[0].src_node

    ph_qn.name = f"PlaceHolder_{spec.query_inputs[0]}"
    ph_ckv.name = f"PlaceHolder_{spec.cache_inputs[0]}"
    ph_idx.name = f"PlaceHolder_{spec.index_input}"
    ph_qp.name = f"PlaceHolder_{spec.query_inputs[1]}"
    ph_kpe.name = f"PlaceHolder_{spec.cache_inputs[1]}"

    return dict(
        idx_window=idx_window,
        gather_ckv=gather_ckv,
        gather_kpe=gather_kpe,
        ckv_mma=ckv_mma,
        pe_mma=pe_mma,
        add=add,
        exp=exp,
        psum=psum,
        value_mma=value_mma,
        ph_qn=ph_qn,
        ph_qp=ph_qp,
        ph_ckv=ph_ckv,
        ph_kpe=ph_kpe,
        ph_idx=ph_idx,
    )


def _mla_window_scalars(spec, dense, split: bool = False) -> dict:
    """The window bookkeeping the loop tier's iteration scalars need.

    With ``split`` the loop index counts the block's own tiles and the request's tile is
    ``tile0 + i``, so every position is taken relative to that.
    """
    bn, iwt = dense.kv_tile, spec.index_window_tiles
    tile = (lambda i: f"(tile0 + {i})") if split else (lambda i: f"{i}")
    return {
        "tile_base": lambda i: f"{bn} * {tile(i)}",
        # The window this tile falls in, and the tile's offset inside it, both in tokens.
        "win_base": lambda i: f"{tile(i)} // {iwt} * {bn * iwt}",
        "win_off": lambda i: f"{tile(i)} % {iwt} * {bn}",
    }


def _fold_denominator(dense, loop_node, psum, psum_id, rows: int):
    """Fold the accumulated (M, N) denominator to a row vector, once, after the loop."""
    rowsum = ReduceSum(
        (rows, dense.kv_tile),
        axis=-1,
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="rowsum_0",
    )
    _accum_edge(loop_node, rowsum, psum_id, 0, psum)
    return rowsum


def _mla_normalise(spec, dense, loop_node, rowsum, value_mma, value_id, rows, **masked):
    """The divide by the folded denominator and its log2; row-predicated with ``masked``."""
    div_cls, log2_cls = (MaskedDiv, MaskedLog2) if masked else (Div, Log2)
    div = div_cls(
        (rows, dense.head_dim_ckv),
        broadcast=0,
        denom_floor=DENOM_FLOOR,
        dtype=dense.accum_dtype,
        out_dtype=dense.out_dtype,
        name="div_0",
        n_batch_dims=0,
        **masked,
    )
    _accum_edge(loop_node, div, value_id, 0, value_mma)
    _connect(rowsum, div, 0, 1)
    outputs = [_output(div, dense.out_dtype, 0)]
    log2 = None
    if spec.emit_lse:
        log2 = log2_cls(
            (rows,),
            dtype=dense.accum_dtype,
            out_dtype=dense.accum_dtype,
            name="log2_0",
            n_batch_dims=0,
            **masked,
        )
        _connect(rowsum, log2, 0, 0)
        outputs.append(_output(log2, dense.accum_dtype, 1))
    return div, log2, outputs


def _build_mla_paged_decode(spec, dense, split: int | None = None) -> Graph:
    """A paged MLA decode: one CTA per request, every query head in the tile.

    With ``split``, the partial half of the split-KV form (see ``_build_paged_decode``):
    the grid is ``(request, split)``, each block walks its own range of the request's KV
    tiles, and the epilogue stores the folded denominator and the value accumulator raw.
    The page-table window is refilled on the block's first tile as well as at every
    window boundary, since a split may start in the middle of a window; codegen re-seeds
    the look-ahead from the fresh window on the same iteration, so nothing else changes.
    """
    mq, bn = dense.query_rows, dense.kv_tile
    ckv, kpe = dense.head_dim_ckv, dense.head_dim_kpe
    batch = spec.batch_dim
    window = bn * spec.index_window_tiles
    if split is not None:
        assert isinstance(split, int) and split > 1, f"split={split!r} is not a split"

    made = _mla_loop_tier(spec, dense, _exp(dense, mq))
    idx_window, gather_ckv, gather_kpe = (
        made["idx_window"],
        made["gather_ckv"],
        made["gather_kpe"],
    )
    psum, value_mma = made["psum"], made["value_mma"]
    ph_qn, ph_qp = made["ph_qn"], made["ph_qp"]
    ph_ckv, ph_kpe, ph_idx = made["ph_ckv"], made["ph_kpe"], made["ph_idx"]

    inner = Graph(nodes=gather_all_nodes(psum, value_mma))
    inner.extent = 1
    inner.loop_kind = "serial"
    inner.window_nodes = [idx_window]
    # A split block's first tile need not sit at a window boundary, so it refills too.
    inner.window_guard = (
        "win_off == 0" if split is None else "(win_off == 0) | (tile_base == split_base)"
    )
    kv_end = "valid_len" if split is None else "split_end"
    if spec.prefetch:
        # Both cache tiles are read one iteration ahead, out of the same shared window. The
        # look-ahead stops at the window's edge, so the first tile of every window is not
        # covered by it; codegen re-seeds the fragments on the iteration that refills the
        # window, which is exactly where that tile's indices become available.
        inner.prefetch_nodes = [gather_ckv, gather_kpe]
        inner.prefetch_guard = (
            f"(win_off + {bn} < {window}) & (tile_base + {bn} < {kv_end})"
        )
    inner.sargs = [
        ("q_row0", "int32"),
        ("q_head0", "int32"),
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("scale2", "float32"),
    ]
    inner.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("scale2", "float32"),
        ("kv_base", "int32"),
        ("win_base", "int32"),
        ("win_off", "int32"),
    ]
    inner.smap = {
        **_passthrough("q_row0", "q_head0", "kv_base", "valid_len", "scale2"),
        **_mla_window_scalars(spec, dense, split=split is not None),
    }
    if split is None:
        inner.loop_trip = f"T.ceildiv(valid_len, {bn})"
    else:
        inner.loop_trip = "my_trip"
        inner.sargs += [("tile0", "int32"), ("my_trip", "int32"), ("split_end", "int32")]
        inner.iter_sargs += [("split_base", "int32"), ("split_end", "int32")]
        inner.smap.update(_passthrough("tile0", "my_trip", "split_end"))
        inner.smap["split_base"] = lambda i: f"{bn} * tile0"
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    psum_id = inner.accum_nodes.index(psum)
    value_id = inner.accum_nodes.index(value_mma)
    inner_qn = inner.binding[ph_qn.name]
    inner_qp = inner.binding[ph_qp.name]
    inner_idx = inner.binding[ph_idx.name]

    # A decode request is one query token, so the tile's rows walk that token's head axis
    # from head 0 and the whole head count is valid.
    for tile_name, source, width, binding in (
        ("qn_tile", ph_qn, ckv, inner_qn),
        ("qp_tile", ph_qp, kpe, inner_qp),
    ):
        tile = RowGather(
            (mq, width),
            source_rows=batch,
            n_valid=mq,
            src_dims=3,
            source_heads=spec.num_qo_heads,
            scalar_args=("q_row0", "q_head0"),
            dtype=dense.dtype,
            out_dtype=dense.dtype,
            name=tile_name,
        )
        tile.source_args = (source.name,)
        binding.producer_op = tile

    # --- the epilogue: fold the accumulated denominator, then normalise -------------
    rowsum = _fold_denominator(dense, loop_node, psum, psum_id, mq)
    store_d = store_o = None
    if split is None:
        div, log2, outputs = _mla_normalise(spec, dense, loop_node, rowsum, value_mma, value_id, mq)
    else:
        # The folded denominator and the raw value accumulator, for the combine to sum.
        div = log2 = None
        store_d, store_o = _partial_stores(mq, ckv, dense.accum_dtype)
        _connect(rowsum, store_d, 0, 0)
        _accum_edge(loop_node, store_o, value_id, 0, value_mma)
        outputs = [_output(store_d, dense.accum_dtype, 0), _output(store_o, dense.accum_dtype, 1)]

    ph_indptr = _index_input(spec.indptr_input)
    main = Graph(
        nodes=[ph_qn, ph_qp, ph_ckv, ph_kpe, ph_indptr, ph_idx]
        + gather_all_nodes(loop_node)
    )
    main.dyn_dims = [batch, spec.pages_dim, spec.indptr_dim, spec.index_dim]
    main.scalar_params = [(spec.scale_input, "float32")]
    main.pass_configs = dict(spec.pass_configs)
    main.n_threads = dense.n_threads
    main.rasterization = spec.rasterization
    # One CTA per request: MLA has a single kv head, so there is no head axis to split the
    # grid on and all query heads are the tile's rows. A split-KV partial adds the split
    # as the trailing grid axis.
    main.extent = (batch,) if split is None else (batch, f"{split}")
    main.sargs = list(inner.sargs)
    indptr = ph_indptr.name
    main.smap = {
        "q_row0": lambda i, *_: f"{i}",
        "q_head0": lambda i, *_: "0",
        "kv_base": lambda i, *_: f"{indptr}[{i}]",
        "valid_len": lambda i, *_: f"{indptr}[{i} + 1] - {indptr}[{i}]",
        "scale2": lambda i, *_: f"{spec.scale_input} * {LOG2E}",
    }
    if split is not None:
        main.smap.update(
            _split_range_scalars(
                split,
                bn,
                lambda b: f"{indptr}[{b} + 1] - {indptr}[{b}]",
                # (request, head block, split) for a grid of (request, split).
                lambda build: (lambda i, s: build(i, "0", s)),
            )
        )
    lock_graph(main)

    _mla_tensor_maps(
        main,
        inner,
        spec,
        dense,
        made=made,
        loop_node=loop_node,
        rowsum=rowsum,
        div=div,
        log2=log2,
        outputs=outputs,
        indptrs=[ph_indptr],
        q_shape_lead=batch,
        q_tiles=None,
        out_rows=1,
        out_offset=lambda i, *_: (f"{i}", "0", "0"),
        lse_offset=lambda i, *_: (f"{i}", "0"),
        split=split,
        store_d=store_d,
        store_o=store_o,
    )
    return _run_passes(main, inner, inner_idx)


def _build_mla_paged_prefill_causal(spec, dense) -> Graph:
    """A causal paged MLA prefill over a ragged query axis.

    MLA has a single kv head, so a CTA owns *all* query heads of the tokens in its tile:
    the M axis fuses (token, head) and one staged KV tile feeds every head. The tuned
    schedule has ``query_rows == num_qo_heads``, i.e. one token per tile, and that is
    required rather than incidental -- the query tiles are affine regions over one token's
    contiguous block of heads, and a tile spanning several tokens is not one region.
    """
    mq, bn = dense.query_rows, dense.kv_tile
    tokens, indptr_dim = spec.tokens_dim, spec.indptr_dim
    # `apply_paging` has already checked that mq == heads, for both MLA layers.
    heads = spec.num_qo_heads
    btok = mq // heads

    made = _mla_loop_tier(spec, dense, _exp(dense, mq, causal=True, row_group=heads))
    idx_window, psum, value_mma = made["idx_window"], made["psum"], made["value_mma"]
    ph_qn, ph_qp = made["ph_qn"], made["ph_qp"]
    ph_ckv, ph_kpe, ph_idx = made["ph_ckv"], made["ph_kpe"], made["ph_idx"]

    inner = Graph(nodes=gather_all_nodes(psum, value_mma))
    inner.extent = 1
    inner.loop_kind = "serial"
    inner.loop_trip = f"T.ceildiv(valid_len, {bn})"
    inner.window_nodes = [idx_window]
    inner.window_guard = "win_off == 0"
    inner.sargs = [
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("scale2", "float32"),
    ]
    inner.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("scale2", "float32"),
        ("kv_base", "int32"),
        ("win_base", "int32"),
        ("win_off", "int32"),
    ]
    inner.smap = {
        **_passthrough("kv_base", "valid_len", "diag", "scale2"),
        **_mla_window_scalars(spec, dense),
    }
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    psum_id = inner.accum_nodes.index(psum)
    value_id = inner.accum_nodes.index(value_mma)
    # The two query tensors are copied in as regions by the outer tier rather than tiled
    # here, so nothing below reads their loop-tier bindings -- but they still have to exist,
    # or the region would have nothing to copy into.
    for name in (ph_qn.name, ph_qp.name):
        assert name in inner.binding, (
            f"{name} did not resolve in the loop tier, so its region copy has no destination"
        )
    inner_idx = inner.binding[ph_idx.name]

    rowsum = _fold_denominator(dense, loop_node, psum, psum_id, mq)
    # The epilogue is row-predicated: the last tile of a request holds fewer than `mq` rows
    # and the rows past it belong to the next request, so they must not be written.
    div, log2, outputs = _mla_normalise(
        spec, dense, loop_node, rowsum, value_mma, value_id, mq,
        scalar_args=("n_rows",), row_group=heads, fuse_dst=False,
    )

    ph_qo = _index_input(spec.query_indptr_input)
    ph_kv = _index_input(spec.indptr_input)
    main = Graph(
        nodes=[ph_qn, ph_qp, ph_ckv, ph_kpe, ph_qo, ph_kv, ph_idx]
        + gather_all_nodes(loop_node)
    )
    main.dyn_dims = [tokens, spec.pages_dim, indptr_dim, spec.index_dim]
    main.scalar_params = [(spec.scale_input, "float32")]
    main.n_threads = dense.n_threads
    main.rasterization = spec.rasterization
    # One CTA per query tile; every head lives in the tile. See the paged prefill layer
    # for why the slack term covers the work list.
    main.extent = (f"T.ceildiv({tokens}, {btok}) + {indptr_dim} - 2",)
    main.prologue = _worklist_scan(ph_qo, btok, indptr_dim)
    main.block_guard = lambda i: "_wl[0] >= 0"
    main.pass_configs = dict(spec.pass_configs)
    # A BufferRegion lets tilelang reach for TMA, whose mbarrier protocol assumes a
    # pipelined producer/consumer body; these nested serial macros provide no arrivals for
    # it and the kernel spins on the barrier. Nothing here needs TMA: the region exists to
    # replace a per-element gather with an affine copy, and this tile can never overrun
    # (its leading extent is btok and the other two are the buffer's own).
    main.pass_configs["tl.disable_tma_lower"] = True
    main.pass_configs["tl.disable_warp_specialized"] = True
    main.sargs = [
        ("n_rows", "int32"),
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("scale2", "float32"),
    ]
    wl = _worklist(ph_qo, ph_kv, btok)
    main.smap = {
        "n_rows": lambda i: wl.n_rows,
        "kv_base": lambda i: wl.kv_beg,
        "valid_len": lambda i: f"T.max({wl.live}, 0)",
        "diag": lambda i: wl.diag,
        "scale2": lambda i: f"{spec.scale_input} * {LOG2E}",
    }
    lock_graph(main)

    q_abs = wl.q_abs
    _mla_tensor_maps(
        main,
        inner,
        spec,
        dense,
        made=made,
        loop_node=loop_node,
        rowsum=rowsum,
        div=div,
        log2=log2,
        outputs=outputs,
        indptrs=[ph_qo, ph_kv],
        q_shape_lead=tokens,
        # One token's whole block of heads, which is contiguous in (total_q, head, dim): the
        # tile the M axis wants is an affine slice, so it goes down as a region instead of a
        # gather. The caches cannot follow -- page_size is 1, so consecutive kv positions
        # are unrelated pages and the read is irreducibly a gather.
        q_tiles=lambda shape: Iterator(
            lambda i: (q_abs, "0", "0"),
            shape=shape,
            tile_size=(btok, heads, shape[-1]),
        ),
        out_rows=btok,
        out_offset=lambda i: (q_abs, "0", "0"),
        lse_offset=lambda i: (q_abs, "0"),
    )
    main.region_args = [ph_qn, ph_qp]
    return _run_passes(main, inner, inner_idx)
