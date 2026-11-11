"""Both tiers' ``tmap``/``tinfo_map`` for the access layers.

The paged tiers are built rather than patched: a dense schedule's maps describe slices of
dense tensors, and every buffer here either changed scope (an index tile becomes a linear
shared buffer), changed shape (a cache is handed down whole and gathered from), or did not
exist (the page table, the lse).
"""

from __future__ import annotations

from ...ir.graph import Iterator


def _whole(shape) -> Iterator:
    """A buffer handed down whole: the tier takes no slice of its own."""
    return Iterator(
        lambda *iters: tuple("0" for _ in shape),
        shape=tuple(shape),
        tile_size=tuple(shape),
    )


def _head_offsets(grid, group: int, cols: bool = False) -> tuple:
    """``(out, lse)`` offsets of a block owning ``group`` heads of one request.

    ``grid`` maps the launch indices to ``(request, head block, ...)``; with ``cols`` its
    third entry is the block's column offset, otherwise the block owns every column.
    """

    def out(*iters):
        b, h, *rest = grid(*iters)
        return (f"{b}", f"{group} * {h}", rest[0] if cols else "0")

    def lse(*iters):
        b, h = grid(*iters)[:2]
        return (f"{b}", f"{group} * {h}")

    return out, lse


def _partial_output_maps(main, outputs, store_d, store_o, split, heads, group, head_dim,
                         batch_dim, grid) -> None:
    """``tmap``/``tinfo_map`` entries for a split-KV partial's two stores.

    The partials are indexed by split first, so one split's contribution to every request
    is contiguous and the combine reads a request's splits at one stride. ``grid`` maps the
    launch indices to ``(request, head block, split)``.
    """

    def _d_offset(*iters):
        b, h, s = grid(*iters)
        return (f"{s}", f"{b}", f"{group} * {h}")

    def _o_offset(*iters):
        b, h, s = grid(*iters)
        return (f"{s}", f"{b}", f"{group} * {h}", "0")

    main.tmap[store_d] = Iterator(
        _d_offset, shape=(split, batch_dim, heads), tile_size=(1, 1, group)
    )
    main.tmap[outputs[0]] = main.tmap[store_d]
    main.tmap[store_o] = Iterator(
        _o_offset, shape=(split, batch_dim, heads, head_dim), tile_size=(1, 1, group, head_dim)
    )
    main.tmap[outputs[1]] = main.tmap[store_o]
    main.tinfo_map[store_d] = ("OutputNode_0", "g")
    main.tinfo_map[outputs[0]] = ("OutputNode_0", "g")
    main.tinfo_map[store_o] = ("OutputNode_1", "g")
    main.tinfo_map[outputs[1]] = ("OutputNode_1", "g")


def _output_maps(main, div, log2, outputs, out_iter, lse_iter, extra=()) -> None:
    """Map the divide, the log2 and their outputs (and any ``extra`` producers of the same
    tiles, in ``(out, lse)`` order) to the two output tensors."""
    main.tmap[div] = out_iter
    main.tmap[outputs[0]] = out_iter
    if extra:
        main.tmap[extra[0]] = out_iter
    if log2 is not None:
        main.tmap[log2] = lse_iter
        main.tmap[outputs[1]] = lse_iter
        if extra:
            main.tmap[extra[1]] = lse_iter


def _output_tinfo(main, div, log2, outputs, extra=()) -> None:
    main.tinfo_map[div] = ("OutputNode_0", "g")
    for node in extra[:1]:
        main.tinfo_map[node] = ("OutputNode_0", "g")
    main.tinfo_map[outputs[0]] = ("OutputNode_0", "g")
    if log2 is not None:
        main.tinfo_map[log2] = ("OutputNode_1", "g")
        for node in extra[1:2]:
            main.tinfo_map[node] = ("OutputNode_1", "g")
        main.tinfo_map[outputs[1]] = ("OutputNode_1", "g")


def _decode_output_maps(main, spec, head_dim, grid, n) -> None:
    """The GQA decode layers' outputs: ``(request, head group)`` tiles, or split partials."""
    group = spec.group
    if n["split"] is None:
        out, lse = _head_offsets(grid, group)
        _output_maps(
            main,
            n["div"],
            n["log2"],
            n["outputs"],
            Iterator(
                out,
                shape=(spec.batch_dim, spec.num_qo_heads, head_dim),
                tile_size=(1, group, head_dim),
            ),
            Iterator(lse, shape=(spec.batch_dim, spec.num_qo_heads), tile_size=(1, group)),
        )
        _output_tinfo(main, n["div"], n["log2"], n["outputs"])
    else:
        _partial_output_maps(
            main, n["outputs"], n["store_d"], n["store_o"], n["split"],
            spec.num_qo_heads, group, head_dim, spec.batch_dim, grid,
        )


def _accum_tiles(rows: int, cols: int) -> list:
    """The loop node's two accumulator tiles: the row sum, then the value matmul."""
    return [_whole((rows,)), _whole((rows, cols))]


def _build_tensor_maps(main, inner, spec, dense, built, grid) -> None:
    """Fill both tiers' maps for the row-gather decode layer."""
    mq, bn, head_dim = dense.query_rows, dense.kv_tile, dense.head_dim
    b = built
    q_shape = (spec.batch_dim, spec.num_qo_heads, head_dim)
    cache_shape = (spec.pages_dim, spec.page_size, spec.num_kv_heads, head_dim)
    inner_k, inner_v = inner.binding[b.ph_k.name], inner.binding[b.ph_v.name]

    main.tmap[b.ph_q] = _whole(q_shape)
    main.tmap[b.ph_k] = _whole(cache_shape)
    main.tmap[b.ph_v] = _whole(cache_shape)
    main.tmap[b.ph_idx] = _whole((spec.index_dim,))
    main.tmap[b.ph_indptr] = _whole((spec.indptr_dim,))
    main.tmap[b.loop_node] = _accum_tiles(mq, head_dim)

    inner.extent = 1
    inner.tmap[b.ph_q] = _whole(q_shape)
    inner.tmap[b.inner_q] = _whole((mq, head_dim))
    inner.tmap[b.inner_idx] = _whole((spec.index_dim,))
    inner.tmap[inner_k] = _whole(cache_shape)
    inner.tmap[inner_v] = _whole(cache_shape)
    inner.tmap[b.gather_k] = _whole((bn, head_dim))
    inner.tmap[b.gather_v] = _whole((bn, head_dim))
    inner.tmap[b.score_mma] = _whole((mq, bn))
    inner.tmap[b.exp] = inner.tmap[b.score_mma]
    inner.tmap[b.rowsum] = _whole((mq,))
    inner.tmap[b.value_mma] = _whole((mq, head_dim))

    main.tinfo_map = {
        b.ph_q: (b.ph_q.name, "g"),
        b.ph_k: (b.ph_k.name, "g"),
        b.ph_v: (b.ph_v.name, "g"),
        b.ph_idx: (b.ph_idx.name, "g"),
        b.ph_indptr: (b.ph_indptr.name, "g"),
        b.loop_node: ("t_GraphNode_0", "r"),
    }
    _decode_output_maps(main, spec, head_dim, grid, vars(b))

    inner.tinfo_map = {
        b.ph_q: (b.ph_q.name, "g"),
        b.ph_k: (b.ph_k.name, "g"),
        b.ph_v: (b.ph_v.name, "g"),
        b.ph_idx: (b.ph_idx.name, "g"),
        b.inner_q: (b.inner_q.name, "s"),
        b.inner_idx: (b.inner_idx.name, "g"),
        inner_k: (inner_k.name, "g"),
        inner_v: (inner_v.name, "g"),
        b.gather_k: ("t_paged_gather_k", "s"),
        b.gather_v: ("t_paged_gather_v", "s"),
        b.score_mma: ("t_mma_0", "r"),
        b.exp: ("t_exp_0", "r"),
        b.rowsum: ("t_rowsum_0", "r"),
        b.value_mma: ("t_mma_1", "r"),
    }


def _slab_tensor_maps(main, inner, spec, dense, *, grid, nodes) -> None:
    """Fill both tiers' maps for the slab layer.

    The one interesting entry is the KV slice: a 4-D Iterator over the cache whose leading
    index is a page-table read and whose tile is ``(1, BN, 1, head_dim)`` -- one page, BN
    consecutive tokens of it, one kv head, the whole head dim.
    """
    mq, bn, head_dim = dense.query_rows, dense.kv_tile, dense.head_dim
    page = spec.page_size
    n = nodes
    inner_idx, inner_k, inner_v = n["inner_idx"], n["inner_k"], n["inner_v"]
    split = n.get("split")

    q_shape = (spec.batch_dim, spec.num_qo_heads, head_dim)
    cache_shape = (spec.pages_dim, page, spec.num_kv_heads, head_dim)

    main.tmap[n["ph_q"]] = _whole(q_shape)
    main.tmap[n["ph_k"]] = _whole(cache_shape)
    main.tmap[n["ph_v"]] = _whole(cache_shape)
    main.tmap[n["ph_idx"]] = _whole((spec.index_dim,))
    main.tmap[n["ph_indptr"]] = _whole((spec.indptr_dim,))
    main.tmap[n["ph_last"]] = _whole((spec.batch_dim,))
    main.tmap[n["loop_node"]] = _accum_tiles(mq, head_dim)

    inner.extent = 1
    inner.tmap[n["ph_q"]] = _whole(q_shape)
    inner.tmap[n["inner_q"]] = _whole((mq, head_dim))
    inner.tmap[inner_idx] = _whole((spec.index_dim,))

    # `kv_page_base` counts pages and `tile_base` counts tokens within the request, so the
    # page holding tile i is `kv_page_base + tile_base / page_size`, and the offset inside
    # it `tile_base % page_size`. Both index reads are clamped, so a corrupt page table
    # reads a wrong page rather than off the end of the pool. A split-KV block's tile i is
    # the request's tile `tile0 + i`.
    def tokens(i):
        return f"({bn} * {i})" if split is None else f"({bn} * (tile0 + {i}))"

    def page_of(i):
        return (
            f"T.min({inner_idx.name}[T.min(kv_page_base + {tokens(i)} // {page}, "
            f"{spec.index_dim} - 1)], {spec.pages_dim} - 1)"
        )

    def slab():
        return Iterator(
            lambda i: (page_of(i), f"{tokens(i)} % {page}", "head", "0"),
            shape=cache_shape,
            tile_size=(1, bn, 1, head_dim),
        )

    inner.tmap[inner_k] = slab()
    inner.tmap[inner_v] = slab()
    inner.tmap[n["score_mma"]] = _whole((mq, bn))
    inner.tmap[n["exp"]] = inner.tmap[n["score_mma"]]
    inner.tmap[n["rowsum"]] = _whole((mq,))
    inner.tmap[n["value_mma"]] = _whole((mq, head_dim))

    main.tinfo_map = {
        n[k]: (n[k].name, "g") for k in ("ph_q", "ph_k", "ph_v", "ph_idx", "ph_indptr", "ph_last")
    }
    main.tinfo_map[n["loop_node"]] = ("t_GraphNode_0", "r")
    _decode_output_maps(main, spec, head_dim, grid, n)

    inner.tinfo_map = {
        **{n[k]: (n[k].name, "g") for k in ("ph_q", "ph_k", "ph_v", "ph_idx")},
        n["inner_q"]: (n["inner_q"].name, "s"),
        inner_idx: (inner_idx.name, "g"),
        # Shared, unlike the ps1 layer's global cache view: the slab is copied into shared
        # memory by the TMA the region unlocks, and it is the matmul operand itself.
        inner_k: (inner_k.name, "s"),
        inner_v: (inner_v.name, "s"),
        n["score_mma"]: ("t_mma_0", "r"),
        n["exp"]: ("t_exp_0", "r"),
        n["rowsum"]: ("t_rowsum_0", "r"),
        n["value_mma"]: ("t_mma_1", "r"),
    }


def _prefill_tensor_maps(main, inner, spec, dense, *, btok, q_abs, nodes) -> None:
    """Both tiers' ``tmap``/``tinfo_map`` for the causal prefill layer."""
    bm, bn, head_dim = dense.query_rows, dense.kv_tile, dense.head_dim
    group = spec.group
    n = nodes
    q_shape = (spec.tokens_dim, spec.num_qo_heads, head_dim)
    cache_shape = (spec.pages_dim, spec.page_size, spec.num_kv_heads, head_dim)
    inner_k, inner_v = inner.binding[n["ph_k"].name], inner.binding[n["ph_v"].name]

    main.tmap[n["ph_q"]] = _whole(q_shape)
    main.tmap[n["ph_k"]] = _whole(cache_shape)
    main.tmap[n["ph_v"]] = _whole(cache_shape)
    main.tmap[n["ph_idx"]] = _whole((spec.index_dim,))
    main.tmap[n["ph_qo"]] = _whole((spec.indptr_dim,))
    main.tmap[n["ph_kv"]] = _whole((spec.indptr_dim,))
    main.tmap[n["loop_node"]] = _accum_tiles(bm, head_dim)
    _output_maps(
        main,
        n["div"],
        n["log2"],
        n["outputs"],
        Iterator(
            lambda i, h: (q_abs, f"{h} * {group}", "0"),
            shape=q_shape,
            tile_size=(btok, group, head_dim),
        ),
        Iterator(
            lambda i, h: (q_abs, f"{h} * {group}"),
            shape=(spec.tokens_dim, spec.num_qo_heads),
            tile_size=(btok, group),
        ),
        extra=n["fills"],
    )

    inner.extent = 1
    inner.tmap[n["ph_q"]] = _whole(q_shape)
    inner.tmap[n["inner_q"]] = _whole((bm, head_dim))
    inner.tmap[n["inner_idx"]] = _whole((spec.index_dim,))
    inner.tmap[inner_k] = _whole(cache_shape)
    inner.tmap[inner_v] = _whole(cache_shape)
    inner.tmap[n["gather_k"]] = _whole((bn, head_dim))
    inner.tmap[n["gather_v"]] = _whole((bn, head_dim))
    inner.tmap[n["score_mma"]] = _whole((bm, bn))
    inner.tmap[n["exp"]] = inner.tmap[n["score_mma"]]
    inner.tmap[n["rowsum"]] = _whole((bm,))
    inner.tmap[n["value_mma"]] = _whole((bm, head_dim))

    main.tinfo_map = {
        n[k]: (n[k].name, "g") for k in ("ph_q", "ph_k", "ph_v", "ph_idx", "ph_qo", "ph_kv")
    }
    main.tinfo_map[n["loop_node"]] = ("t_GraphNode_0", "r")
    _output_tinfo(main, n["div"], n["log2"], n["outputs"], extra=n["fills"])

    inner.tinfo_map = {
        **{n[k]: (n[k].name, "g") for k in ("ph_q", "ph_k", "ph_v", "ph_idx")},
        n["inner_q"]: (n["inner_q"].name, "s"),
        n["inner_idx"]: (n["inner_idx"].name, "g"),
        inner_k: (inner_k.name, "g"),
        inner_v: (inner_v.name, "g"),
        n["gather_k"]: ("t_paged_gather_k", "s"),
        n["gather_v"]: ("t_paged_gather_v", "s"),
        n["score_mma"]: ("t_mma_0", "r"),
        n["exp"]: ("t_exp_0", "r"),
        n["rowsum"]: ("t_rowsum_0", "r"),
        n["value_mma"]: ("t_mma_1", "r"),
    }


def _mla_tensor_maps(
    main,
    inner,
    spec,
    dense,
    *,
    made,
    loop_node,
    rowsum,
    div,
    log2,
    outputs,
    indptrs,
    q_shape_lead,
    q_tiles,
    out_rows,
    out_offset,
    lse_offset,
    split=None,
    store_d=None,
    store_o=None,
) -> None:
    """Fill both tiers' maps for either MLA layer.

    ``q_tiles`` is what separates them: None means the queries are handed down whole and a
    ``RowGather`` stages the tile (decode), and an Iterator factory means the tile is an
    affine region of the ragged query axis (prefill).
    """
    mq, bn = dense.query_rows, dense.kv_tile
    ckv, kpe = dense.head_dim_ckv, dense.head_dim_kpe
    heads, window = spec.num_qo_heads, bn * spec.index_window_tiles
    ph_qn, ph_qp = made["ph_qn"], made["ph_qp"]
    ph_ckv, ph_kpe, ph_idx = made["ph_ckv"], made["ph_kpe"], made["ph_idx"]
    idx_window = made["idx_window"]
    gather_ckv, gather_kpe = made["gather_ckv"], made["gather_kpe"]
    ckv_mma, pe_mma, add = made["ckv_mma"], made["pe_mma"], made["add"]
    exp, psum, value_mma = made["exp"], made["psum"], made["value_mma"]
    inner_qn = inner.binding[ph_qn.name]
    inner_qp = inner.binding[ph_qp.name]
    inner_idx = inner.binding[ph_idx.name]

    qn_shape = (q_shape_lead, heads, ckv)
    qp_shape = (q_shape_lead, heads, kpe)
    ckv_shape = (spec.pages_dim, spec.page_size, ckv)
    kpe_shape = (spec.pages_dim, spec.page_size, kpe)

    if q_tiles is None:
        main.tmap[ph_qn] = _whole(qn_shape)
        main.tmap[ph_qp] = _whole(qp_shape)
    else:
        main.tmap[ph_qn] = q_tiles(qn_shape)
        main.tmap[ph_qp] = q_tiles(qp_shape)
    main.tmap[ph_ckv] = _whole(ckv_shape)
    main.tmap[ph_kpe] = _whole(kpe_shape)
    main.tmap[ph_idx] = _whole((spec.index_dim,))
    for indptr in indptrs:
        main.tmap[indptr] = _whole((spec.indptr_dim,))
    main.tmap[loop_node] = _accum_tiles(mq, ckv)

    main.tmap[rowsum] = _whole((mq,))
    if split is None:
        _output_maps(
            main,
            div,
            log2,
            outputs,
            Iterator(
                out_offset, shape=(q_shape_lead, heads, ckv), tile_size=(out_rows, heads, ckv)
            ),
            Iterator(lse_offset, shape=(q_shape_lead, heads), tile_size=(out_rows, heads)),
        )

    inner.extent = 1
    # A gathered query tile needs the whole tensor in scope inside the tier; a region tile
    # arrives already sliced, so the tier sees only the slice.
    if q_tiles is None:
        inner.tmap[ph_qn] = _whole(qn_shape)
        inner.tmap[ph_qp] = _whole(qp_shape)
    inner.tmap[inner_qn] = _whole((mq, ckv))
    inner.tmap[inner_qp] = _whole((mq, kpe))
    inner.tmap[inner_idx] = _whole((spec.index_dim,))
    inner.tmap[inner.binding[ph_ckv.name]] = _whole(ckv_shape)
    inner.tmap[inner.binding[ph_kpe.name]] = _whole(kpe_shape)
    inner.tmap[idx_window] = _whole((window,))
    inner.tmap[gather_ckv] = _whole((bn, ckv))
    inner.tmap[gather_kpe] = _whole((bn, kpe))
    inner.tmap[ckv_mma] = _whole((mq, bn))
    inner.tmap[pe_mma] = inner.tmap[ckv_mma]
    inner.tmap[add] = inner.tmap[ckv_mma]
    inner.tmap[exp] = inner.tmap[ckv_mma]
    inner.tmap[psum] = inner.tmap[exp]
    inner.tmap[value_mma] = _whole((mq, ckv))

    main.tinfo_map = {
        ph_qn: (ph_qn.name, "g"),
        ph_qp: (ph_qp.name, "g"),
        ph_ckv: (ph_ckv.name, "g"),
        ph_kpe: (ph_kpe.name, "g"),
        ph_idx: (ph_idx.name, "g"),
        **{indptr: (indptr.name, "g") for indptr in indptrs},
        loop_node: ("t_GraphNode_0", "r"),
        rowsum: ("t_rowsum_0", "r"),
    }
    if split is None:
        _output_tinfo(main, div, log2, outputs)
    else:
        # A decode grid of (request, split): the whole head block is the row tile.
        _partial_output_maps(
            main, outputs, store_d, store_o, split, heads, heads, ckv, q_shape_lead,
            lambda i, s: (i, "0", s),
        )

    inner.tinfo_map = {
        ph_qn: (ph_qn.name, "g"),
        ph_qp: (ph_qp.name, "g"),
        ph_ckv: (ph_ckv.name, "g"),
        ph_kpe: (ph_kpe.name, "g"),
        ph_idx: (ph_idx.name, "g"),
        inner_qn: (inner_qn.name, "s"),
        inner_qp: (inner_qp.name, "s"),
        inner_idx: (inner_idx.name, "g"),
        inner.binding[ph_ckv.name]: (inner.binding[ph_ckv.name].name, "g"),
        inner.binding[ph_kpe.name]: (inner.binding[ph_kpe.name].name, "g"),
        idx_window: ("t_idx_window", "s"),
        gather_ckv: ("t_paged_gather_ckv", "s"),
        gather_kpe: ("t_paged_gather_kpe", "s"),
        ckv_mma: ("t_mma_0", "r"),
        pe_mma: ("t_mma_pe", "r"),
        add: ("t_add_0", "r"),
        exp: ("t_exp_0", "r"),
        psum: ("t_psum_0", "r"),
        value_mma: ("t_mma_1", "r"),
    }
