"""Recognise a scheduled dense attention graph and read off the tiling it was given.

The access layer rebuilds the body from the spec and keeps only the tiling, so these are
real recognizers rather than type censuses: anything accepted and not checked is something
the emitted kernel silently disagrees with. A dense graph that computes a biased,
differently scaled or differently wired attention must be rejected here, because nothing
downstream will notice.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...ir.graph import Graph, GraphNode, OutputNode, PlaceHolderNode
from ...ir.ops import Add, Div, Exp, Matmul, Mul, ReduceSum, Sub
from .errors import NotDenseAttention
from .specs import CombineTiling


@dataclass(frozen=True)
class DenseAttention:
    """The tiling the dense stages chose, read off a scheduled attention graph."""

    query_rows: int  # M tile: query rows per block
    kv_tile: int  # N tile of the score matmul: KV rows per iteration
    head_dim: int
    n_threads: int
    dtype: str
    accum_dtype: str
    out_dtype: str


@dataclass(frozen=True)
class DenseMLA:
    """The tiling the dense stages chose, read off a scheduled dense MLA graph."""

    query_rows: int
    kv_tile: int
    head_dim_ckv: int
    head_dim_kpe: int
    n_threads: int
    dtype: str
    accum_dtype: str
    out_dtype: str


def _sole(nodes, what):
    nodes = list(nodes)
    if len(nodes) != 1:
        raise NotDenseAttention(f"expected exactly one {what}, found {[n.name for n in nodes]}")
    return nodes[0]


def _is_constant_scaling(node) -> bool:
    """Whether ``node`` scales or shifts its input by a compile-time constant.

    A constant is foldable into the runtime softmax scale; a second *tensor* operand is a
    bias or a mask, which the access layer would drop from the kernel's signature
    entirely, so those must not be mistaken for scaling.
    """
    return (
        isinstance(node, (Mul, Div, Add, Sub))
        and len(node.inputs) == 1
        and getattr(node, "const", None) is not None
    )


def _chain_to(node, target):
    """The constant-scaling nodes between ``target`` and ``node``, walking input 0.

    None when the path is anything else -- an extra tensor operand, another matmul, a
    second nonlinearity.
    """
    chain, current = [], node
    for _ in range(8):
        source = current.inputs[0].src_node
        if source is target:
            return chain
        if not _is_constant_scaling(source):
            return None
        chain.append(source)
        current = source
    return None


def _folded_scale(chain) -> float:
    """The multiplier ``chain`` applies to the scores; shifts do not affect it.

    A shift is dropped by the access layer, which is sound: softmax is shift-invariant,
    and the paged operator's log-sum-exp is defined without one (none of the shipped
    kernels subtract a max).
    """
    scale = 1.0
    for node in chain:
        if isinstance(node, Mul):
            scale *= float(node.const)
        elif isinstance(node, Div):
            if getattr(node, "const_on_left", False):
                raise NotDenseAttention(
                    "the scores are divided INTO a constant, which is not a scaling"
                )
            scale /= float(node.const)
    return scale


def _assert_output_is_the_divide(graph: Graph, div, out) -> None:
    """The dense body must *end* at the softmax divide.

    The ``accums`` check constrains what feeds the divide; this constrains what consumes
    it. Both access layers store the divide's tile straight to the output, so an op the
    dense body applies after the divide -- a scaling, an activation, anything -- is not
    slower in the paged kernel, it is absent from it. That is a wrong kernel produced from
    a graph the recognizer accepted, which is the one failure mode this whole module is
    written to avoid.
    """
    if out.inputs[0].src_node is not div:
        raise NotDenseAttention(
            f"the output is produced by {out.inputs[0].src_node.name}, not the softmax divide; "
            "the access layer stores the divide's tile, so whatever the dense body computes "
            "after it would be dropped from the paged kernel"
        )
    # Closing the class rather than the one instance: the outer tier is rebuilt as exactly
    # the reduction loop plus the divide, so any other op there is discarded too, even one
    # that does not sit between the divide and the output.
    extra = [
        n
        for n in graph.nodes
        if n is not div
        and not isinstance(n, (GraphNode, PlaceHolderNode, OutputNode))
    ]
    if extra:
        raise NotDenseAttention(
            f"the outer tier also holds {[n.name for n in extra]}; the access layer rebuilds "
            "only the reduction loop and the softmax divide, and would drop these"
        )


def _split_of(graph: Graph) -> int:
    """The split factor of a scheduled split-K partial: its grid is the split axis alone.

    The dense stages tile a partial body so that one block owns one split's whole
    (query rows, head dim) tile -- anything else would be a block the paging pass discards
    -- so the only grid axis left is the split.
    """
    extent = graph.extent if isinstance(graph.extent, tuple) else (graph.extent,)
    if not (len(extent) == 1 and int(extent[0]) > 1):
        raise NotDenseAttention(
            f"a split-K partial's grid must be the split axis alone, got {graph.extent}; "
            "re-tile with the whole query tile and head dim in one block"
        )
    return int(extent[0])


def _assert_split_slices(graph: Graph, placeholders, split: int, what: str) -> None:
    """Each KV operand's tile must be exactly a 1/split slice of its dense extent."""
    for ph in placeholders:
        tmap = graph.tmap[ph]
        whole, tile = int(tmap.shape[0]), int(tmap.tile_size[0])
        if not (getattr(ph, "_split_axis", None) == 0 and whole == split * tile):
            raise NotDenseAttention(
                f"{what} {ph.name} is tiled {tile} of {whole} rows on split axis "
                f"{getattr(ph, '_split_axis', None)}; a split-K partial slices its KV axis "
                f"into {split} equal contiguous ranges, one per block"
            )


def _partial_outputs(graph: Graph, rowsum, value_mma) -> str:
    """Check the partial stores its two accumulators raw, and return their output dtype.

    Both must be stored unchanged: a partial that normalises by its own denominator, or
    scales anything, cannot be summed by the combine. The access layer stores the
    accumulators in their accumulation precision regardless of the dense output dtype,
    so what this returns is only the operator's *final* output dtype.
    """
    outs = [n for n in graph.nodes if isinstance(n, OutputNode)]
    if len(outs) != 2:
        raise NotDenseAttention(
            f"a split-K partial stores exactly its two accumulators, found {len(outs)} outputs"
        )
    stored = []
    for out in outs:
        edge = out.inputs[0]
        accum = getattr(edge, "accum_node", None)
        if not (accum is not None and edge.src_node.__class__.__name__ == "GraphNode"):
            raise NotDenseAttention(
                f"output {out.name} is produced by {edge.src_node.name}, not stored straight "
                "from the reduction loop; a partial that post-processes its accumulators "
                "cannot be summed by the combine"
            )
        stored.append(accum)
    if {id(n) for n in stored} != {id(rowsum), id(value_mma)}:
        raise NotDenseAttention(
            "the two outputs must be the row sum and the value matmul; got "
            f"{[n.name for n in stored]}"
        )
    extra = [
        n for n in graph.nodes if not isinstance(n, (GraphNode, PlaceHolderNode, OutputNode))
    ]
    if extra:
        raise NotDenseAttention(
            f"the outer tier also holds {[n.name for n in extra]}; a split-K partial's outer "
            "tier is the reduction loop and its two stores, nothing else"
        )
    dtypes = {str(o.get_dtype(0)) for o in outs}
    if len(dtypes) != 1:
        raise NotDenseAttention(f"the two partial outputs disagree on dtype: {dtypes}")
    return dtypes.pop()


def graph_tile(graph: Graph, node) -> tuple:
    iterator = graph.tmap[node]
    return tuple(iterator.tile_size)


class _Normalised:
    """The single-kernel epilogue: the body ends at the softmax divide, on one block."""

    # Index of the reduction extent in the query operand's tile.
    reduce_axis = 1

    def __init__(self, graph: Graph):
        self.div = _sole([n for n in graph.nodes if isinstance(n, Div)], "softmax divide")
        # The value dtype is the OutputNode's, not the divide's: the divide accumulates in
        # fp32 and the store is what casts.
        self.out = _sole([n for n in graph.nodes if isinstance(n, OutputNode)], "output")

    def outputs(self, graph: Graph, rowsum, value_mma) -> str:
        div = self.div
        if getattr(div, "broadcast", None) != 0:
            raise NotDenseAttention(
                f"the divide must broadcast the denominator over rows, got {div.broadcast}"
            )
        # Ordered, not a set: a body that divides the row sum BY the value matmul carries
        # the same two accumulators, and the layer -- which always builds the numerator
        # from the value matmul -- would silently compute something else.
        accums = [getattr(edge, "accum_node", None) for edge in div.inputs]
        if accums[:2] != [value_mma, rowsum]:
            raise NotDenseAttention(
                "the divide must take the value matmul over the row sum, in that order; got "
                f"{[None if n is None else n.name for n in accums]}"
            )
        _assert_output_is_the_divide(graph, div, self.out)
        return str(self.out.get_dtype(0))

    def grid(self, graph: Graph, inner: Graph, kv, what: str):
        # The layer writes its own grid; a dense schedule with blocks of its own loses them.
        if graph.extent not in (1, (1,)):
            raise NotDenseAttention(
                f"the dense schedule has a grid of {graph.extent}, which paging would discard; "
                "it must tile to a single block"
            )
        return None


class _Partial:
    """The split-K partial epilogue: no divide, both accumulators stored raw, KV sliced."""

    reduce_axis = -1

    def __init__(self, graph: Graph):
        if [n for n in graph.nodes if isinstance(n, Div)]:
            raise NotDenseAttention(
                "a split-K partial must not divide: the softmax denominator is only known once "
                "every split has contributed, so the divide belongs to the combine"
            )

    def outputs(self, graph: Graph, rowsum, value_mma) -> str:
        return _partial_outputs(graph, rowsum, value_mma)

    def grid(self, graph: Graph, inner: Graph, kv, what: str) -> int:
        split = _split_of(graph)
        _assert_split_slices(graph, [inner.binding[ph.name] for ph in kv], split, what)
        return split


def _loop_body(graph: Graph):
    """The reduction tier and its two accumulators: ``(inner, rowsum, value_mma)``."""
    inner = _sole(
        [n for n in graph.nodes if isinstance(n, GraphNode)], "reduction subgraph"
    ).graph
    if len(inner.accum_nodes) != 2:
        raise NotDenseAttention(
            "expected two accumulators (the row sum and the value matmul), got "
            f"{[n.name for n in inner.accum_nodes]}"
        )
    rowsum = _sole(
        [n for n in inner.accum_nodes if type(n) is ReduceSum], "row-sum accumulator"
    )
    value_mma = _sole(
        [n for n in inner.accum_nodes if isinstance(n, Matmul)], "value matmul"
    )
    return inner, rowsum, value_mma


def _score_mmas(inner: Graph, value_mma) -> list:
    return [
        n
        for n in inner.nodes
        if isinstance(n, Matmul) and n is not value_mma and n.transpose_B
    ]


def _check_softmax_consumers(rowsum, value_mma, exp, value_name: str) -> None:
    if rowsum.inputs[0].src_node is not exp:
        raise NotDenseAttention(
            f"the row sum reduces {rowsum.inputs[0].src_node.name}, not the softmax exp"
        )
    if value_mma.inputs[0].src_node is not exp:
        raise NotDenseAttention(
            f"the value matmul consumes {value_mma.inputs[0].src_node.name}, not the exp"
        )
    if value_mma.transpose_B:
        raise NotDenseAttention(f"the value matmul must not transpose {value_name}")


def _check_scale(scale_chain, head_dim: int) -> None:
    """The folded constant must be 1/sqrt(D), or the runtime scale means something else."""
    folded = _folded_scale(scale_chain)
    expected = head_dim**-0.5
    if not abs(folded - expected) <= 1e-6 * max(1.0, abs(expected)):
        raise NotDenseAttention(
            f"the dense body scales the scores by {folded}, not 1/sqrt({head_dim}) = "
            f"{expected}; the runtime scale this layer substitutes would not match"
        )


def _match_attention(graph: Graph, epilogue) -> tuple:
    """Q @ K^T -> constant scaling -> exp -> {row sum, @ V} -> ``epilogue``."""
    inner, rowsum, value_mma = _loop_body(graph)
    score_mma = _sole(_score_mmas(inner, value_mma), "score matmul (transpose_B)")
    exp = _sole([n for n in inner.nodes if isinstance(n, Exp)], "softmax exp")
    tail = epilogue(graph)
    scale_chain = _chain_to(exp, score_mma)
    if scale_chain is None:
        raise NotDenseAttention(
            "the softmax exp does not read the score matmul through constant scaling alone; "
            "this body is not the attention the access layer would rebuild"
        )
    _check_softmax_consumers(rowsum, value_mma, exp, "V")
    # The tiling carries one dtype pair, read off the score matmul; a mixed-precision body
    # would be silently rebuilt in the score matmul's precision throughout.
    if not ((str(value_mma.dtype), str(value_mma.out_dtype)) == (
        str(score_mma.dtype),
        str(score_mma.out_dtype),
    )):
        raise NotDenseAttention(
            f"the value matmul runs {value_mma.dtype}/{value_mma.out_dtype} but the score "
            f"matmul runs {score_mma.dtype}/{score_mma.out_dtype}; the access layer builds one "
            "precision for both and cannot express the difference"
        )
    out_dtype = tail.outputs(graph, rowsum, value_mma)
    placeholders = [n for n in inner.nodes if isinstance(n, PlaceHolderNode)]
    if len(placeholders) != 3:
        raise NotDenseAttention(
            f"expected exactly Q, K and V in the loop tier, got {[n.name for n in placeholders]}"
        )
    for operand, name in ((score_mma.inputs[0], "Q"), (score_mma.inputs[1], "K")):
        if not isinstance(operand.src_node, PlaceHolderNode):
            raise NotDenseAttention(
                f"the score matmul's {name} operand is {operand.src_node.name}, not an input"
            )
    if not isinstance(value_mma.inputs[1].src_node, PlaceHolderNode):
        raise NotDenseAttention(
            "the value matmul's V operand is not an input"
        )

    score_tile = graph_tile(inner, score_mma)
    value_tile = graph_tile(inner, value_mma)
    if not (len(score_tile) == 2 and len(value_tile) == 2):
        raise NotDenseAttention(
            f"expected 2-D score/value tiles, got {score_tile}/{value_tile}"
        )
    if score_tile[0] != value_tile[0]:
        raise NotDenseAttention(
            f"score and value matmuls disagree on the query tile: {score_tile} vs {value_tile}"
        )
    # The head dim comes off the node, not the tile: they differ exactly when the policy
    # split the head dim, and a kernel built from the fraction is a different operator.
    head_dim = int(value_mma.dims[-2])
    if int(value_tile[1]) != head_dim:
        raise NotDenseAttention(
            f"the head dim is split into {value_tile[1]}-wide blocks (of {head_dim}); the "
            "access layer replaces the grid, so those blocks would be lost. Re-tile with the "
            "whole head dim in one block."
        )
    if int(graph_tile(inner, score_mma.inputs[0].src_node)[tail.reduce_axis]) != head_dim:
        raise NotDenseAttention(
            "the score matmul reduces over part of the head dim; it must reduce over all of it"
        )
    kv = (score_mma.inputs[1].src_node, value_mma.inputs[1].src_node)
    split = tail.grid(graph, inner, kv, "KV operand")
    _check_scale(scale_chain, head_dim)
    dense = DenseAttention(
        query_rows=int(score_tile[0]),
        kv_tile=int(score_tile[1]),
        head_dim=head_dim,
        n_threads=int(getattr(graph, "n_threads", 128)),
        dtype=str(score_mma.dtype),
        accum_dtype=str(score_mma.out_dtype),
        out_dtype=out_dtype,
    )
    return dense, split


def _match_mla(graph: Graph, epilogue) -> tuple:
    """(Q @ CKV^T) + (Q_pe @ KPE^T) -> constant scaling -> exp -> {row sum, @ CKV} -> ``epilogue``.

    The value matmul's operand is the very placeholder the wider score matmul reduced
    against: MLA aliases the compressed cache as V.
    """
    inner, rowsum, value_mma = _loop_body(graph)
    score_mmas = _score_mmas(inner, value_mma)
    if len(score_mmas) != 2:
        raise NotDenseAttention(
            "expected two score matmuls (compressed kv and positional), got "
            f"{[n.name for n in score_mmas]}"
        )
    exp = _sole([n for n in inner.nodes if isinstance(n, Exp)], "softmax exp")
    tail = epilogue(graph)
    add = _sole(
        [n for n in inner.nodes if isinstance(n, Add) and len(n.inputs) == 2],
        "score sum (the two matmuls added)",
    )
    added = {id(edge.src_node) for edge in add.inputs}
    if added != {id(score_mmas[0]), id(score_mmas[1])}:
        raise NotDenseAttention(
            "the score sum does not add the two score matmuls; this body is not the MLA "
            "attention the access layer would rebuild"
        )
    scale_chain = _chain_to(exp, add)
    if scale_chain is None:
        raise NotDenseAttention(
            "the softmax exp does not read the summed scores through constant scaling alone"
        )
    _check_softmax_consumers(rowsum, value_mma, exp, "the cache")
    for other in (score_mmas[0], score_mmas[1], value_mma):
        if not ((str(other.dtype), str(other.out_dtype)) == (
            str(score_mmas[0].dtype),
            str(score_mmas[0].out_dtype),
        )):
            raise NotDenseAttention(
                f"{other.name} runs {other.dtype}/{other.out_dtype} but {score_mmas[0].name} "
                f"runs {score_mmas[0].dtype}/{score_mmas[0].out_dtype}; the access layer builds "
                "one precision for all three matmuls"
            )
    out_dtype = tail.outputs(graph, rowsum, value_mma)
    placeholders = [n for n in inner.nodes if isinstance(n, PlaceHolderNode)]
    if len(placeholders) != 4:
        raise NotDenseAttention(
            "expected exactly the two query tensors and the two caches in the loop tier, got "
            f"{[n.name for n in placeholders]}"
        )
    # Both operands of each score matmul, but only the cache operand of the value matmul.
    operands = [(mma, slot) for mma in score_mmas for slot in (0, 1)]
    operands.append((value_mma, 1))
    for mma, slot in operands:
        if not isinstance(mma.inputs[slot].src_node, PlaceHolderNode):
            raise NotDenseAttention(
                f"{mma.name}'s operand {slot} is "
                f"{mma.inputs[slot].src_node.name}, not an input"
            )
    # The compressed cache is the value tensor; that decides which score matmul is ckv's.
    # A shared cache or query needs no check: `_sole` and the placeholder census refuse both.
    cache = value_mma.inputs[1].src_node
    ckv_mma = _sole(
        [n for n in score_mmas if n.inputs[1].src_node is cache],
        "score matmul reading the value cache (MLA aliases ckv as V)",
    )
    pe_mma = score_mmas[0] if score_mmas[1] is ckv_mma else score_mmas[1]

    score_tile = graph_tile(inner, ckv_mma)
    pe_tile = graph_tile(inner, pe_mma)
    value_tile = graph_tile(inner, value_mma)
    if not (len(score_tile) == 2 and len(value_tile) == 2):
        raise NotDenseAttention(
            f"expected 2-D score/value tiles, got {score_tile}/{value_tile}"
        )
    if score_tile != pe_tile:
        raise NotDenseAttention(
            f"the two score matmuls disagree on the tile: {score_tile} vs {pe_tile}"
        )
    if score_tile[0] != value_tile[0]:
        raise NotDenseAttention(
            f"score and value matmuls disagree on the query tile: {score_tile} vs {value_tile}"
        )
    head_dim_ckv = int(value_mma.dims[-2])
    head_dim_kpe = int(pe_mma.dims[-1])
    if int(value_tile[1]) != head_dim_ckv:
        raise NotDenseAttention(
            f"the compressed head dim is split into {value_tile[1]}-wide blocks (of "
            f"{head_dim_ckv}); the access layer replaces the grid, so those blocks would be "
            "lost. Re-tile with the whole head dim in one block."
        )
    axis = tail.reduce_axis
    if int(graph_tile(inner, ckv_mma.inputs[0].src_node)[axis]) != head_dim_ckv:
        raise NotDenseAttention(
            "the ckv score matmul reduces over part of the compressed head dim; it must "
            "reduce over all of it"
        )
    if int(graph_tile(inner, pe_mma.inputs[0].src_node)[axis]) != head_dim_kpe:
        raise NotDenseAttention(
            "the positional score matmul reduces over part of the RoPE head dim; it must "
            "reduce over all of it"
        )
    split = tail.grid(graph, inner, (cache, pe_mma.inputs[1].src_node), "cache operand")
    # MLA's true scale arrives at runtime (it comes from the pre-absorption qk width and a
    # YaRN correction), but a body scaled by anything but 1/sqrt(ckv) is not attention.
    _check_scale(scale_chain, head_dim_ckv)
    dense = DenseMLA(
        query_rows=int(score_tile[0]),
        kv_tile=int(score_tile[1]),
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        n_threads=int(getattr(graph, "n_threads", 128)),
        dtype=str(ckv_mma.dtype),
        accum_dtype=str(ckv_mma.out_dtype),
        out_dtype=out_dtype,
    )
    return dense, split


def read_dense_attention(graph: Graph) -> DenseAttention:
    """Recognise a scheduled dense attention graph and extract its tiling."""
    return _match_attention(graph, _Normalised)[0]


def read_dense_partial(graph: Graph) -> tuple:
    """Recognise a scheduled split-K *partial* of dense attention; ``(tiling, split)``.

    ``read_dense_attention``'s body minus the divide, both accumulators stored raw and the
    KV operands each tiled to a contiguous 1/split slice, one split per block.
    """
    return _match_attention(graph, _Partial)


def read_dense_mla(graph: Graph) -> DenseMLA:
    """Recognise a scheduled dense MLA graph and extract its tiling."""
    return _match_mla(graph, _Normalised)[0]


def read_dense_mla_partial(graph: Graph) -> tuple:
    """Recognise a scheduled split-K partial of dense MLA attention; ``(tiling, split)``."""
    return _match_mla(graph, _Partial)


def read_dense_combine(graph: Graph) -> CombineTiling:
    """Recognise a scheduled split-K *combine* and extract its tiling.

    The body: two axis-0 reductions over ``(split, rows)`` and ``(split, rows, cols)``
    partials, then a divide broadcasting the first over the rows of the second, stored
    straight to the output. The tiling read off it is how many partials one loop
    iteration folds (the value sum's split step), how many output columns one block owns,
    and the block size -- the three things the dynamic combine keeps. The rows a block
    owns are not read: in the dynamic kernel they are the head group, fixed by the operator.

    The scheduler places each reduction independently: inside its own loop tier stepping
    through the splits, or -- when its step is the whole split -- in the outer tier with no
    loop. Any mix is legal; the value sum's placement is the one that matters for cost, so
    it is the one whose step and staging are kept.
    """
    div = _sole([n for n in graph.nodes if isinstance(n, Div)], "softmax divide")
    out = _sole([n for n in graph.nodes if isinstance(n, OutputNode)], "output")
    if getattr(div, "broadcast", None) != 0:
        raise NotDenseAttention(
            f"the divide must broadcast the denominator over rows, got {div.broadcast}"
        )
    if out.inputs[0].src_node is not div:
        raise NotDenseAttention(
            f"the output is produced by {out.inputs[0].src_node.name}, not the softmax divide; "
            "the access layer stores the divide's tile, so whatever the dense body computes "
            "after it would be dropped from the paged kernel"
        )
    # Where each reduction lives: (node, tier graph, or None for the outer tier).
    placed = []
    for node in graph.nodes:
        if type(node) is ReduceSum:  # a ReduceMax is never the row sum
            placed.append((node, None))
        elif isinstance(node, GraphNode):
            for inner_node in node.graph.nodes:
                if type(inner_node) is ReduceSum:
                    if inner_node not in node.graph.accum_nodes:
                        raise NotDenseAttention(
                            f"{inner_node.name} sits in a loop tier without accumulating "
                            "across it"
                        )
                    placed.append((inner_node, node.graph))
        elif not isinstance(node, (PlaceHolderNode, OutputNode, Div)):
            raise NotDenseAttention(
                f"the combine also holds {node.name}; the access layer rebuilds only the "
                "two split sums and the divide, and would drop it"
            )
    if not (len(placed) == 2 and all(r.axis == 0 for r, _ in placed)):
        raise NotDenseAttention(
            "expected two reductions over the leading (split) axis, got "
            f"{[(n.name, getattr(n, 'axis', None)) for n, _ in placed]}"
        )
    by_rank = {len(r.dims): (r, tier) for r, tier in placed}
    if set(by_rank) != {2, 3}:
        raise NotDenseAttention(
            "expected one (split, rows) and one (split, rows, cols) reduction, got ranks "
            f"{sorted(len(r.dims) for r, _ in placed)}"
        )
    value_sum, value_tier = by_rank[3]
    rowsum_sum, _ = by_rank[2]
    summed = [getattr(e, "accum_node", None) or e.src_node for e in div.inputs]
    if summed[:2] != [value_sum, rowsum_sum]:
        raise NotDenseAttention(
            "the divide must take the summed value partials over the summed row sums, in that "
            f"order; got {[n.name for n in summed]}"
        )
    ph_value = value_sum.inputs[0].src_node
    ph_rowsum = rowsum_sum.inputs[0].src_node
    if not (isinstance(ph_value, PlaceHolderNode) and isinstance(ph_rowsum, PlaceHolderNode)):
        raise NotDenseAttention(
            "both reductions must read a partial tensor directly"
        )

    def outer(ph, tier):
        return ph if tier is None else tier.binding[ph.name]

    # The split count is the partials' leading extent; the step is the value sum's tile.
    split = int(graph.tmap[outer(ph_value, value_tier)].shape[0])
    rowsum_tier = by_rank[2][1]
    if int(graph.tmap[outer(ph_rowsum, rowsum_tier)].shape[0]) != split:
        raise NotDenseAttention(
            "the two partials disagree on the split count"
        )
    if value_tier is None:
        step, scope = split, "g"
    else:
        # The policy may hand the split axis a tile wider than its extent; that is the
        # whole extent in one step, not a step the loop could take.
        step = min(int(graph_tile(value_tier, ph_value)[0]), split)
        scope = value_tier.tinfo_map.get(ph_value, (None, "s"))[-1]
    div_tile = graph_tile(graph, div)
    if int(div_tile[-1]) != int(graph_tile(graph, out)[-1]):
        raise NotDenseAttention(
            "the divide and the output disagree on the column tile"
        )
    return CombineTiling(
        split=split,
        split_step=step,
        cols=int(div_tile[-1]),
        n_threads=int(getattr(graph, "n_threads", 128)),
        input_scope=scope if scope in ("s", "g") else "s",
    )
