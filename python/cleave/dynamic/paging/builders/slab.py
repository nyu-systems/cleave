"""The paged decode layer at page_size > 1: each KV tile is a contiguous slab of one page."""

from __future__ import annotations

from ....ir.graph import Graph, GraphNode, PlaceHolderNode
from ....ir.ops import RowGather
from ....ir.graph_utils import gather_all_nodes, lock_graph
from ..stages import (
    LOG2E,
    _decode_epilogue,
    _decode_grid,
    _exp,
    _gqa_body,
    _index_input,
    _int32,
    _name_inputs,
    _passthrough,
    _run_passes,
    _split_range_scalars,
)
from ..tensor_maps import _slab_tensor_maps


def _build_paged_decode_slab(spec, dense, split: int | None = None) -> Graph:
    """Paged decode whose KV tile is a contiguous slab inside one page (``page_size % kv_tile == 0``).

    No gather: the cache is sliced by a page-table-indexed ``tmap`` and copied as a BufferRegion
    in a 2-stage pipelined loop (TMA). ``KV_INDPTR`` counts pages, so the token length comes
    from ``KV_LAST_PAGE_LEN``. With ``split``, the partial half of the split-KV form.
    """
    mq, bn, head_dim = dense.query_rows, dense.kv_tile, dense.head_dim
    group, page = spec.group, spec.page_size
    if split is not None:
        assert isinstance(split, int) and split > 1, f"split={split!r} is not a split"

    # --- the loop tier: score one slab, accumulate ---------------------------------
    score_mma, exp, rowsum, value_mma = _gqa_body(spec, dense, mq, lambda: _exp(dense, mq))
    # The caches are the matmuls' own operands: there is no gather between them and the
    # tile, only the slab slice that `_slab_tensor_maps` writes.
    ph_q = score_mma.inputs[0].src_node
    ph_k = score_mma.inputs[1].src_node
    ph_v = value_mma.inputs[1].src_node
    # The page table feeds no node -- it is read for the addresses in those slab slices --
    # but it is a declared kernel input and has to be in scope inside the tier that reads
    # it, so it is listed among both tiers' nodes.
    ph_idx = PlaceHolderNode(name=spec.index_input)
    ph_idx.set_shape((1,))
    ph_idx.set_dtype(_int32())
    ph_idx.is_batch_dim = False
    _name_inputs(spec, ph_q, ph_k, ph_v, ph_idx)

    inner = Graph(nodes=[ph_idx] + gather_all_nodes(rowsum, value_mma))
    inner.extent = 1
    # Pipelined for TMA, not for latency: a copy is only lowered to a bulk transfer when
    # it sits inside a pipelined producer/consumer loop.
    inner.loop_kind = "pipelined"
    inner.loop_stages = 2
    inner.sargs = [
        ("q_row0", "int32"),
        ("q_head0", "int32"),
        ("kv_page_base", "int32"),
        ("valid_len", "int32"),
        ("head", "int32"),
        ("scale2", "float32"),
    ]
    inner.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("scale2", "float32"),
    ]
    inner.smap = _passthrough(
        "q_row0", "q_head0", "kv_page_base", "head", "valid_len", "scale2"
    )
    if split is None:
        inner.loop_trip = f"T.ceildiv(valid_len, {bn})"
        inner.smap["tile_base"] = lambda i: f"{bn} * {i}"
    else:
        inner.loop_trip = "my_trip"
        inner.sargs += [("tile0", "int32"), ("my_trip", "int32")]
        inner.smap.update(_passthrough("tile0", "my_trip"))
        inner.smap["tile_base"] = lambda i: f"{bn} * (tile0 + {i})"
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    rowsum_id = inner.accum_nodes.index(rowsum)
    value_id = inner.accum_nodes.index(value_mma)
    inner_q = inner.binding[ph_q.name]
    inner_idx = inner.binding[ph_idx.name]
    inner_k = inner.binding[ph_k.name]
    inner_v = inner.binding[ph_v.name]
    # A slab lies inside one page addressed by a scalar page id, which tilelang can turn
    # into a TMA descriptor. This is the only reason the tiles are regions.
    inner.region_args = [inner_k, inner_v]

    q_tile = RowGather(
        (mq, head_dim),
        source_rows=spec.batch_dim,
        n_valid=group,
        src_dims=3,
        source_heads=spec.num_qo_heads,
        scalar_args=("q_row0", "q_head0"),
        dtype=dense.dtype,
        out_dtype=dense.dtype,
        name="q_tile",
    )
    q_tile.source_args = (ph_q.name,)
    inner_q.producer_op = q_tile

    div, log2, store_d, store_o, outputs = _decode_epilogue(
        spec, dense, loop_node, rowsum, rowsum_id, value_mma, value_id, split
    )
    ph_indptr = _index_input(spec.indptr_input)
    ph_last = _index_input(spec.last_page_len_input)

    main = Graph(
        nodes=[ph_q, ph_k, ph_v, ph_indptr, ph_idx, ph_last]
        + gather_all_nodes(loop_node)
    )
    main.dyn_dims = [spec.batch_dim, spec.pages_dim, spec.indptr_dim, spec.index_dim]
    main.scalar_params = [(spec.scale_input, "float32")]
    main.n_threads = dense.n_threads
    main.pass_configs = dict(spec.pass_configs)
    main.rasterization = spec.rasterization
    main.extent, grid, scalar = _decode_grid(spec, split)
    main.sargs = list(inner.sargs)

    indptr, last = ph_indptr.name, ph_last.name

    def n_pages(b):
        return f"({indptr}[{b} + 1] - {indptr}[{b}])"

    # Every page but the last is full; the last holds kv_last_page_len tokens. The whole
    # sum is clamped to the pages the request owns, which is what keeps a corrupt or
    # differently-conventioned entry from attending to the *next* request's pages --
    # clamping only the last term to `n_pages * page_size` would leave it up to
    # `(n_pages - 1) * page_size` tokens of room to do so. The clamp doubles as the
    # empty-request case: at zero pages it is min(anything, 0) = 0.
    def valid_len(b):
        return (
            f"T.min(T.max({n_pages(b)} - 1, 0) * {page}"
            f" + T.max({last}[{b}], 0), {n_pages(b)} * {page})"
        )

    main.smap = {
        "q_row0": scalar(lambda b, h, *_: f"{b}"),
        "q_head0": scalar(lambda b, h, *_: f"{h} * {group}"),
        # In pages: the slab slice multiplies back up to tokens itself.
        "kv_page_base": scalar(lambda b, h, *_: f"{indptr}[{b}]"),
        "valid_len": scalar(lambda b, h, *_: valid_len(b)),
        "head": scalar(lambda b, h, *_: f"{h}"),
        "scale2": scalar(lambda b, h, *_: f"{spec.scale_input} * {LOG2E}"),
    }
    if split is not None:
        main.smap.update(_split_range_scalars(split, bn, valid_len, scalar))
    lock_graph(main)

    _slab_tensor_maps(
        main,
        inner,
        spec,
        dense,
        grid=grid,
        nodes=dict(
            ph_q=ph_q,
            ph_k=ph_k,
            ph_v=ph_v,
            ph_idx=ph_idx,
            ph_indptr=ph_indptr,
            ph_last=ph_last,
            loop_node=loop_node,
            inner_q=inner_q,
            inner_idx=inner_idx,
            inner_k=inner_k,
            inner_v=inner_v,
            score_mma=score_mma,
            exp=exp,
            rowsum=rowsum,
            value_mma=value_mma,
            div=div,
            log2=log2,
            outputs=outputs,
            split=split,
            store_d=store_d,
            store_o=store_o,
        ),
    )
    return _run_passes(main, inner, inner_idx)
