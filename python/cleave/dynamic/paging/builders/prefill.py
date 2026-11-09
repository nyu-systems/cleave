"""The causal prefill layer over a paged cache, keyed on a device-scanned work list."""

from __future__ import annotations

from ....ir.graph import Graph, GraphNode
from ....ir.ops import ConstFill, MaskedDiv, MaskedLog2, RangeGather
from ....ir.ops.gather import BUFFER_BOUND
from ....ir.graph_utils import gather_all_nodes, lock_graph
from ..stages import (
    LOG2E,
    _exp,
    _gqa_body,
    _index_input,
    _name_inputs,
    _normalise,
    _paged_gathers,
    _passthrough,
    _run_passes,
    _wire_gathers,
    _worklist,
    _worklist_scan,
)
from ..tensor_maps import _prefill_tensor_maps


def _build_paged_prefill_causal(spec, dense) -> Graph:
    """Causal prefill over a paged cache, gridded over query tiles found by a device-side scan.

    M fuses (token, head slot); tiles with an empty causal window take the fill-only branch.
    """
    bm, bn, head_dim = dense.query_rows, dense.kv_tile, dense.head_dim
    group, n_kv = spec.group, spec.num_kv_heads
    # Query tokens per block: the M axis is (token, head slot), so a block owns this many
    # tokens of a single kv head's group.
    btok = bm // group
    tokens, pages = spec.tokens_dim, spec.pages_dim
    indptr_dim, index_dim = spec.indptr_dim, spec.index_dim

    # --- the loop tier ---------------------------------------------------------------
    gather_k, gather_v = _paged_gathers(spec, dense, "kv_head")
    # `diag` is the tile's own end-aligned causal bound; `row_group` is what makes a row
    # index mean a token rather than a head slot.
    score_mma, exp, rowsum, value_mma = _gqa_body(
        spec, dense, bm, lambda: _exp(dense, bm, causal=True, row_group=group)
    )
    ph_k, ph_v, ph_idx = _wire_gathers(gather_k, gather_v, score_mma, value_mma)
    ph_q = score_mma.inputs[0].src_node
    _name_inputs(spec, ph_q, ph_k, ph_v, ph_idx)

    inner = Graph(nodes=gather_all_nodes(rowsum, value_mma))
    inner.extent = 1
    inner.loop_kind = "serial"
    inner.loop_trip = f"T.ceildiv(valid_len, {bn})"
    inner.prefetch_nodes = [gather_k, gather_v]
    inner.prefetch_guard = f"tile_base + {bn} < valid_len"
    body_sargs = [
        ("q_row0", "int32"),
        ("q_tile_base", "int32"),
        ("q_len", "int32"),
        ("q_head0", "int32"),
        ("kv_base", "int32"),
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
        ("kv_pos0", "int32"),
        ("kv_head", "int32"),
    ]
    inner.smap = {
        **_passthrough(*(name for name, _ in body_sargs)),
        "tile_base": lambda i: f"{bn} * {i}",
        # The tile's first kv position, in tokens: where the gathers read the page table.
        "kv_pos0": lambda i: f"kv_base + {bn} * {i}",
    }
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    rowsum_id = inner.accum_nodes.index(rowsum)
    value_id = inner.accum_nodes.index(value_mma)
    inner_q = inner.binding[ph_q.name]
    inner_idx = inner.binding[ph_idx.name]

    # Q comes from a ragged token axis, so the tile is a token RANGE rather than a row and
    # a head; `row_group` splits the fused M axis back into (token, head slot).
    q_tile = RangeGather(
        (bm, head_dim),
        source_rows=tokens,
        src_dims=3,
        source_heads=spec.num_qo_heads,
        row_group=group,
        clamp_row=BUFFER_BOUND,
        scalar_args=("q_row0", "q_tile_base", "q_len", "q_head0"),
        dtype=dense.dtype,
        out_dtype=dense.dtype,
        name="q_tile",
    )
    q_tile.source_args = (ph_q.name,)
    inner_q.producer_op = q_tile

    # --- the epilogue ----------------------------------------------------------------
    # `n_rows` is how many of the tile's tokens the request actually has; the last tile of
    # a request is partial far more often than not, so both stores are row-predicated.
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
        row_group=group,
    )

    # The alternative producers for a tile with no causal window. They write the same
    # tiles as the epilogue, at the same per-kv-head shape -- one CTA per kv head, so the
    # store keeps the parallelism the attention path has.
    fills = [
        ConstFill(
            (bm, head_dim),
            value=0.0,
            out_dtype=dense.out_dtype,
            row_group=group,
            scalar_args=("n_rows",),
            name="fill_o",
        )
    ]
    if spec.emit_lse:
        fills.append(
            ConstFill(
                (bm,),
                value=float("-inf"),
                out_dtype=dense.accum_dtype,
                row_group=group,
                scalar_args=("n_rows",),
                name="fill_lse",
            )
        )

    ph_qo = _index_input(spec.query_indptr_input)
    ph_kv = _index_input(spec.indptr_input)

    main = Graph(
        nodes=[ph_q, ph_k, ph_v, ph_qo, ph_kv, ph_idx] + gather_all_nodes(loop_node)
    )
    main.dyn_dims = [tokens, pages, indptr_dim, index_dim]
    main.scalar_params = [(spec.scale_input, "float32")]
    main.pass_configs = dict(spec.pass_configs)
    main.n_threads = dense.n_threads
    main.rasterization = spec.rasterization
    # One CTA per (query tile, kv head). sum(ceil(q_len/btok)) is at most
    # ceil(total_q/btok) + n_requests - 1 (ceil is subadditive up to one per term), so this
    # grid always covers the work list; the slack slots scan to request -1 and the body
    # guard skips them.
    main.extent = (
        f"T.ceildiv({tokens}, {btok}) + {indptr_dim} - 2",
        f"{n_kv}",
    )
    main.sargs = [("n_rows", "int32")] + body_sargs

    wl = _worklist(ph_qo, ph_kv, btok)
    main.smap = {
        "n_rows": lambda i, h: wl.n_rows,
        "q_row0": lambda i, h: wl.q_beg,
        "q_tile_base": lambda i, h: wl.tile_base,
        "q_len": lambda i, h: wl.q_len,
        "q_head0": lambda i, h: f"{h} * {group}",
        "kv_base": lambda i, h: wl.kv_beg,
        "valid_len": lambda i, h: f"T.max({wl.live}, 0)",
        "diag": lambda i, h: wl.diag,
        "kv_head": lambda i, h: f"{h}",
        "scale2": lambda i, h: f"{spec.scale_input} * {LOG2E}",
    }
    main.prologue = _worklist_scan(ph_qo, btok, indptr_dim)
    main.block_guard = lambda i, h: "_wl[0] >= 0"
    # With an empty causal window the KV loop runs zero trips, so the accumulators stay
    # zero and the epilogue's floored divide yields exactly the 0 and -inf the fills
    # write -- the two branches agree by construction. Taking the fill instead skips
    # clearing the accumulator, gathering Q, and the divide.
    main.work_guard = lambda i, h: f"{wl.live} > 0"
    main.else_nodes = list(fills)
    lock_graph(main)

    _prefill_tensor_maps(
        main,
        inner,
        spec,
        dense,
        btok=btok,
        q_abs=wl.q_abs,
        nodes=dict(
            ph_q=ph_q,
            ph_k=ph_k,
            ph_v=ph_v,
            ph_idx=ph_idx,
            ph_qo=ph_qo,
            ph_kv=ph_kv,
            loop_node=loop_node,
            inner_q=inner_q,
            inner_idx=inner_idx,
            gather_k=gather_k,
            gather_v=gather_v,
            score_mma=score_mma,
            exp=exp,
            rowsum=rowsum,
            value_mma=value_mma,
            div=div,
            log2=log2,
            outputs=outputs,
            fills=fills,
        ),
    )
    return _run_passes(main, inner, inner_idx)
