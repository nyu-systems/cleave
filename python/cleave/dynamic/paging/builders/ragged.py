"""The causal prefill layer over ragged (already-contiguous) keys."""

from __future__ import annotations

from ....ir.graph import Graph, GraphNode, Iterator
from ....ir.ops import MaskedDiv, MaskedLog2
from ....ir.graph_utils import gather_all_nodes, lock_graph
from ..stages import (
    LOG2E,
    _exp,
    _gqa_body,
    _index_input,
    _name_inputs,
    _normalise,
    _passthrough,
    _run_passes,
    _worklist,
    _worklist_scan,
)
from ..tensor_maps import _accum_tiles, _output_maps, _output_tinfo, _whole


def _build_ragged_prefill_causal(spec, dense) -> Graph:
    """Causal prefill over ragged (contiguous) keys: the KV tile is an affine BufferRegion.

    Hence a pipelined loop (TMA), and one CTA per query head with the kv head derived from it.
    """
    bm, bn, head_dim = dense.query_rows, dense.kv_tile, dense.head_dim
    group = spec.group
    tokens, kv_tokens = spec.tokens_dim, spec.kv_tokens_dim
    indptr_dim = spec.indptr_dim

    # No `row_group`: a row of this tile is a query token, not a (token, head slot) pair.
    # P stays in its fragment rather than being staged through shared: tilelang then feeds
    # it to the second matmul in place (wgmma_rs) instead of a stmatrix round trip.
    score_mma, exp, rowsum, value_mma = _gqa_body(
        spec, dense, bm, lambda: _exp(dense, bm, causal=True), stage_p=False
    )
    ph_q = score_mma.inputs[0].src_node
    ph_k = score_mma.inputs[1].src_node
    ph_v = value_mma.inputs[1].src_node
    _name_inputs(spec, ph_q, ph_k, ph_v)

    inner = Graph(nodes=gather_all_nodes(rowsum, value_mma))
    inner.extent = 1
    inner.loop_kind = "pipelined"
    inner.loop_stages = 2
    inner.loop_trip = f"T.ceildiv(valid_len, {bn})"
    body_sargs = [
        ("q_row0", "int32"),
        ("q_tile_base", "int32"),
        ("q_len", "int32"),
        ("q_head", "int32"),
        ("kv_row0", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("kv_head", "int32"),
        ("scale2", "float32"),
    ]
    inner.sargs = list(body_sargs)
    inner.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("scale2", "float32"),
        ("kv_row0", "int32"),
        ("kv_head", "int32"),
    ]
    inner.smap = {
        **_passthrough(*(name for name, _ in body_sargs)),
        "tile_base": lambda i: f"{bn} * {i}",
    }
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    rowsum_id = inner.accum_nodes.index(rowsum)
    value_id = inner.accum_nodes.index(value_mma)
    inner_q = inner.binding[ph_q.name]
    inner_k = inner.binding[ph_k.name]
    inner_v = inner.binding[ph_v.name]

    div, log2, outputs = _normalise(
        loop_node,
        (value_mma, value_id),
        (rowsum, rowsum_id),
        (bm, head_dim),
        dense.accum_dtype,
        dense.out_dtype,
        spec.emit_lse,
        div_cls=MaskedDiv,
        log2_cls=MaskedLog2,
        scalar_args=("n_rows",),
    )
    ph_qo = _index_input(spec.query_indptr_input)
    ph_kv = _index_input(spec.indptr_input)

    main = Graph(nodes=[ph_q, ph_k, ph_v, ph_qo, ph_kv] + gather_all_nodes(loop_node))
    main.dyn_dims = [tokens, kv_tokens, indptr_dim]
    main.scalar_params = [(spec.scale_input, "float32")]
    main.pass_configs = dict(spec.pass_configs)
    main.n_threads = dense.n_threads
    main.rasterization = spec.rasterization
    # One CTA per (query tile, query head); see the paged prefill layer for why the slack
    # term covers the work list.
    main.extent = (
        f"T.ceildiv({tokens}, {bm}) + {indptr_dim} - 2",
        f"{spec.num_qo_heads}",
    )
    main.sargs = [("n_rows", "int32")] + body_sargs

    wl = _worklist(ph_qo, ph_kv, bm)
    main.smap = {
        "n_rows": lambda i, h: wl.n_rows,
        "q_row0": lambda i, h: wl.q_beg,
        "q_tile_base": lambda i, h: wl.tile_base,
        "q_len": lambda i, h: wl.q_len,
        "q_head": lambda i, h: f"{h}",
        "kv_row0": lambda i, h: wl.kv_beg,
        "valid_len": lambda i, h: f"T.max({wl.live}, 0)",
        "diag": lambda i, h: wl.diag,
        # The grid is over QUERY heads, so the kv head a block reads follows from it.
        "kv_head": lambda i, h: f"{h} // {group}",
        "scale2": lambda i, h: f"{spec.scale_input} * {LOG2E}",
    }
    main.prologue = _worklist_scan(ph_qo, bm, indptr_dim)
    main.block_guard = lambda i, h: "_wl[0] >= 0"
    # A BufferRegion is what lets tilelang bounds-predicate and bulk-copy the slice; an
    # `_S()` view is just a buffer of the slice's extent.
    main.region_args = [ph_q]
    lock_graph(main)

    q_abs = wl.q_abs
    q_shape = (tokens, spec.num_qo_heads, head_dim)
    kv_shape = (kv_tokens, spec.num_kv_heads, head_dim)

    main.tmap[ph_q] = Iterator(
        lambda i, h: (q_abs, f"{h}", "0"), shape=q_shape, tile_size=(bm, 1, head_dim)
    )
    main.tmap[ph_k] = _whole(kv_shape)
    main.tmap[ph_v] = _whole(kv_shape)
    main.tmap[ph_qo] = _whole((indptr_dim,))
    main.tmap[ph_kv] = _whole((indptr_dim,))
    main.tmap[loop_node] = _accum_tiles(bm, head_dim)
    _output_maps(
        main,
        div,
        log2,
        outputs,
        Iterator(
            lambda i, h: (q_abs, f"{h}", "0"), shape=q_shape, tile_size=(bm, 1, head_dim)
        ),
        Iterator(
            lambda i, h: (q_abs, f"{h}"),
            shape=(tokens, spec.num_qo_heads),
            tile_size=(bm, 1),
        ),
    )

    def kv_tile():
        """One request's KV window, `bn` tokens at a time -- an affine slice, so TMA can
        take it and the loop can be pipelined."""
        return Iterator(
            lambda i: (f"kv_row0 + {bn} * {i}", "kv_head", "0"),
            shape=kv_shape,
            tile_size=(bn, 1, head_dim),
        )

    inner.extent = 1
    inner.tmap[inner_q] = _whole((bm, head_dim))
    inner.tmap[inner_k] = kv_tile()
    inner.tmap[inner_v] = kv_tile()
    inner.region_args = [inner_k, inner_v]
    inner.tmap[score_mma] = _whole((bm, bn))
    inner.tmap[exp] = inner.tmap[score_mma]
    inner.tmap[rowsum] = _whole((bm,))
    inner.tmap[value_mma] = _whole((bm, head_dim))

    main.tinfo_map = {node: (node.name, "g") for node in (ph_q, ph_k, ph_v, ph_qo, ph_kv)}
    main.tinfo_map[loop_node] = ("t_GraphNode_0", "r")
    _output_tinfo(main, div, log2, outputs)
    inner.tinfo_map = {
        **{node: (node.name, "g") for node in (ph_q, ph_k, ph_v)},
        **{node: (node.name, "s") for node in (inner_q, inner_k, inner_v)},
        score_mma: ("t_mma_0", "r"),
        exp: ("t_exp_0", "r"),
        rowsum: ("t_rowsum_0", "r"),
        value_mma: ("t_mma_1", "r"),
    }
    return _run_passes(main)
