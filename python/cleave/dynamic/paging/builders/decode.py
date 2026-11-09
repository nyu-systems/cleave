"""The paged decode layer at one token per page: KV rows gathered off a page table."""

from __future__ import annotations

from dataclasses import dataclass

from ....ir.graph import Graph, GraphNode
from ....ir.ops import RowGather
from ....ir.graph_utils import gather_all_nodes, lock_graph
from ..stages import (
    LOG2E,
    _decode_epilogue,
    _decode_grid,
    _exp,
    _gqa_body,
    _index_input,
    _name_inputs,
    _paged_gathers,
    _passthrough,
    _run_passes,
    _split_range_scalars,
    _wire_gathers,
)
from ..tensor_maps import _build_tensor_maps


@dataclass
class _Built:
    """The nodes ``_build_paged_decode`` created, for the tensor-map step to key on."""

    ph_q: object
    ph_k: object
    ph_v: object
    ph_idx: object
    ph_indptr: object
    loop_node: object
    inner_q: object
    inner_idx: object
    gather_k: object
    gather_v: object
    score_mma: object
    exp: object
    rowsum: object
    value_mma: object
    div: object
    log2: object
    outputs: list
    # Split-KV partial only: the number of splits and the two stores that replace the
    # divide and the log-sum-exp. None/absent for the single-kernel layer.
    split: int = None
    store_d: object = None
    store_o: object = None


def _build_paged_decode(spec, dense, split: int | None = None) -> Graph:
    """The paged decode layer, or with ``split`` the partial half of its split-KV form.

    A split block runs ``my_trip`` tiles from ``tile0``, prefetches no further than
    ``split_end``, and stores raw fp32 accumulators for ``_build_decode_combine`` to sum.
    """
    mq, bn, head_dim = dense.query_rows, dense.kv_tile, dense.head_dim
    group = spec.group
    if split is not None:
        assert isinstance(split, int) and split > 1, f"split={split!r} is not a split"

    # --- the loop tier: gather a KV tile, score it, accumulate ----------------------
    gather_k, gather_v = _paged_gathers(spec, dense, "head")
    # One MaskedExp replaces the dense Mul(1/sqrt(D)) + Sub(max estimate) + Exp chain: none
    # of the tuned kernels subtract a max.
    score_mma, exp, rowsum, value_mma = _gqa_body(spec, dense, mq, lambda: _exp(dense, mq))
    ph_k, ph_v, ph_idx = _wire_gathers(gather_k, gather_v, score_mma, value_mma)
    ph_q = score_mma.inputs[0].src_node
    _name_inputs(spec, ph_q, ph_k, ph_v, ph_idx)

    inner = Graph(nodes=gather_all_nodes(rowsum, value_mma))
    # A serial loop with a runtime trip count never reads the static extent; the real
    # bound is loop_trip, which is per request.
    inner.extent = 1
    inner.loop_kind = "serial"
    # Both KV tiles are read one iteration ahead, so the gathers' global latency is
    # covered by the tile's own matmuls and softmax instead of standing in front of them.
    # This is the loop's only source of load/compute overlap: the reads are gathers,
    # which tilelang's pipeliner cannot double-buffer.
    inner.prefetch_nodes = [gather_k, gather_v]
    inner.sargs = [
        ("q_row0", "int32"),
        ("q_head0", "int32"),
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("head", "int32"),
        ("scale2", "float32"),
    ]
    inner.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("scale2", "float32"),
        ("kv_pos0", "int32"),
        ("head", "int32"),
    ]
    inner.smap = _passthrough("q_row0", "q_head0", "kv_base", "head", "valid_len", "scale2")
    if split is None:
        inner.loop_trip = f"T.ceildiv(valid_len, {bn})"
        inner.prefetch_guard = f"tile_base + {bn} < valid_len"
        inner.smap["tile_base"] = lambda i: f"{bn} * {i}"
        # The tile's first kv position, in tokens: where the gathers read the page table.
        inner.smap["kv_pos0"] = lambda i: f"kv_base + {bn} * {i}"
    else:
        # This block's tiles are `tile0 .. tile0 + my_trip` of the request's, and the
        # look-ahead must not run past the split's own end.
        inner.loop_trip = "my_trip"
        inner.prefetch_guard = f"tile_base + {bn} < split_end"
        inner.sargs += [("tile0", "int32"), ("my_trip", "int32"), ("split_end", "int32")]
        inner.iter_sargs += [("split_end", "int32")]
        inner.smap.update(_passthrough("tile0", "my_trip", "split_end"))
        inner.smap["tile_base"] = lambda i: f"{bn} * (tile0 + {i})"
        inner.smap["kv_pos0"] = lambda i: f"kv_base + {bn} * (tile0 + {i})"
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    rowsum_id = inner.accum_nodes.index(rowsum)
    value_id = inner.accum_nodes.index(value_mma)
    inner_q = inner.binding[ph_q.name]
    inner_idx = inner.binding[ph_idx.name]

    # Q is staged once per block by a gather that zero-fills the rows past the head
    # group, so a padded query tile costs no correctness.
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

    # Inputs lead the node list in the operator's declared order, both so the emitted
    # signature matches it and because a symbolic output extent is resolved at call time
    # from an already-bound input.
    main = Graph(
        nodes=[ph_q, ph_k, ph_v, ph_indptr, ph_idx] + gather_all_nodes(loop_node)
    )
    main.dyn_dims = [spec.batch_dim, spec.pages_dim, spec.indptr_dim, spec.index_dim]
    main.scalar_params = [(spec.scale_input, "float32")]
    main.n_threads = dense.n_threads
    main.pass_configs = dict(spec.pass_configs)
    # Unconditional in every builder: None means "defer to the codegen default".
    main.rasterization = spec.rasterization
    # One CTA per (request, kv head): a decode request is a single query token, so its
    # whole head group fits one tile. A split-KV partial adds the split as the trailing
    # (slowest) grid axis, so the blocks sharing a request's page table launch together.
    main.extent, grid, scalar = _decode_grid(spec, split)
    main.sargs = list(inner.sargs)

    indptr = ph_indptr.name
    main.smap = {
        "q_row0": scalar(lambda b, h, *_: f"{b}"),
        "q_head0": scalar(lambda b, h, *_: f"{h} * {group}"),
        "kv_base": scalar(lambda b, h, *_: f"{indptr}[{b}]"),
        "valid_len": scalar(lambda b, h, *_: f"{indptr}[{b} + 1] - {indptr}[{b}]"),
        "head": scalar(lambda b, h, *_: f"{h}"),
        "scale2": scalar(lambda b, h, *_: f"{spec.scale_input} * {LOG2E}"),
    }
    if split is not None:
        main.smap.update(
            _split_range_scalars(
                split, bn, lambda b: f"{indptr}[{b} + 1] - {indptr}[{b}]", scalar
            )
        )
    lock_graph(main)

    built = _Built(
        ph_q=ph_q,
        ph_k=ph_k,
        ph_v=ph_v,
        ph_idx=ph_idx,
        ph_indptr=ph_indptr,
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
        split=split,
        store_d=store_d,
        store_o=store_o,
    )
    _build_tensor_maps(main, inner, spec, dense, built, grid)
    return _run_passes(main, inner, inner_idx)
