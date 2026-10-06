"""The recognizers must reject a body the access layer would misrepresent.

The layer rebuilds the attention body from the spec and keeps only the tiling, so any
dense graph it accepts without checking is one the emitted kernel silently disagrees
with. Each case here is a dense graph that used to be accepted and produce the shipped
kernel regardless -- a biased score, an unscaled score, a row sum over the wrong tensor.

Both recognizers are covered. ``read_dense_attention`` takes the Q@K^T/softmax/P@V body
the four GQA layers are built from; ``read_dense_mla`` takes the latent one, whose two
score matmuls are *summed* and whose value matmul reads the very cache the wider score
matmul reduced against. They share the checks on the softmax epilogue and duplicate the
rest, so a body that is rejected by one says nothing about the other.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tilelang")

from tests._graphs import (
    _connect,
    _finalize,
    flash_attention_sd_graph,
    mla_attention_sd_graph,
)
from tests.graph.apply_paging._test_utils import (
    ACCUM_DTYPE,
    DTYPE,
    KV_PROXY,
    build_paged_graph,
    tiled_dense_attention,
)
from cleave.graph import OutputNode
from cleave.ops import Add, Div, Exp, Matmul, Mul, Reduction, Sub
from cleave.paging import (
    DenseAttention,
    DenseMLA,
    MLASpec,
    PagingSpec,
    apply_paging,
    read_dense_attention,
    read_dense_mla,
)

MQ, D, BN = 16, 128, 64
SPEC = PagingSpec(num_qo_heads=32, num_kv_heads=8)


def _tail_op(shape, name):
    """An op the access layer has no place for, to hang off the end of a body.

    A scaling stands in for every one of them: an activation, a second divide, anything the
    dense body applies after the softmax divide fires the same assert in the same place, so
    one per recognizer is enough. What differs between the two recognizers is the call site.
    """
    return Mul(shape, const=2.0, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name=name)


def _reroute(op, dst, slot):
    """Insert ``op`` in front of ``dst``'s ``slot``, dropping the input it displaces."""
    stale = dst.inputs[slot].src_node
    _connect(stale, op, 0, 0)
    _connect(op, dst, 0, slot)
    stale._out_edges = [e for e in stale._out_edges if e.dst_node is not dst]


def _dense(**overrides):
    """The dense attention body, with one edge or constant deliberately wrong."""
    scale = overrides.get("scale", D**-0.5)
    mma_0 = Matmul(
        (MQ, KV_PROXY, D),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    mul_0 = Mul((MQ, KV_PROXY), const=scale, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="mul_0")
    sub_0 = Sub(
        (MQ, KV_PROXY),
        const=(50.0 * scale),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="sub_0",
    )
    exp_0 = Exp((MQ, KV_PROXY), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="exp_0")
    mma_1 = Matmul(
        (MQ, D, KV_PROXY),
        dtype=overrides.get("value_dtype", DTYPE),
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
        transpose_B=overrides.get("value_transpose", False),
    )
    rowsum_0 = Reduction(
        (MQ, KV_PROXY), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    div_0 = Div(
        (MQ, D),
        broadcast=overrides.get("broadcast", 0),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )
    nodes = [mma_0, mul_0, sub_0, exp_0, mma_1, rowsum_0, div_0]

    _connect(mma_0, mul_0, 0, 0)
    _connect(mul_0, sub_0, 0, 0)
    tail = sub_0
    if overrides.get("bias"):
        # An additive score bias: a fourth input the layer would drop from the signature.
        add_0 = Add(
            (MQ, KV_PROXY), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="add_0"
        )
        _connect(tail, add_0, 0, 0)
        nodes.append(add_0)
        tail = add_0
    _connect(tail, exp_0, 0, 0)
    # The row sum and the value matmul normally both read the exp.
    _connect(exp_0 if not overrides.get("rowsum_reads_scores") else tail, rowsum_0, 0, 0)
    _connect(exp_0 if not overrides.get("value_reads_scores") else tail, mma_1, 0, 0)
    _connect(mma_1, div_0, 0, 0)
    _connect(rowsum_0, div_0, 0, 1)
    if overrides.get("shared_cache"):
        # One tensor serving as both K and V. The layer declares a (K, V) pair and stages
        # each with its own gather, so it would rebuild a body reading two caches.
        stale = mma_1.inputs[1].src_node
        _connect(mma_0.inputs[1].src_node, mma_1, 0, 1)
        stale._out_edges = [e for e in stale._out_edges if e.dst_node is not mma_1]
    # An operand the layer would have to read from a tensor it does not declare.
    if overrides.get("q_pre"):
        _reroute(Mul((MQ, D), const=2.0, dtype=DTYPE, out_dtype=DTYPE, name="mul_q"), mma_0, 0)
    if overrides.get("v_pre"):
        _reroute(Mul((KV_PROXY, D), const=2.0, dtype=DTYPE, out_dtype=DTYPE, name="mul_v"), mma_1, 1)
    out_of = div_0
    if overrides.get("post"):
        out_of = _tail_op((MQ, D), "mul_post")
        _connect(div_0, out_of, 0, 0)
        nodes.append(out_of)
    OutputNode(out_of, dtype=DTYPE)
    return _finalize(*nodes)


def _paged_from(dense):
    return apply_paging(tiled_dense_attention(dense, MQ, D, BN), SPEC)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        # A bias tensor the emitted kernel would not read at all.
        (dict(bias=True), "not the attention the access layer would rebuild"),
        # A scale that is not the operator's, so the runtime sm_scale means something else.
        (dict(scale=2.0 * D**-0.5), "not 1/sqrt"),
        (dict(scale=1.0), "not 1/sqrt"),
        # Wiring that computes a different function.
        (dict(rowsum_reads_scores=True), "not the softmax exp"),
        (dict(value_reads_scores=True), "not the exp"),
        (dict(value_transpose=True), "must not transpose V"),
        (dict(broadcast=1), "must broadcast the denominator"),
        # The tiling this returns carries ONE dtype pair, read off the score matmul, so a
        # mixed-precision body would be rebuilt entirely in the score matmul's precision.
        (dict(value_dtype="float16"), "one precision for both"),
        # The layer stores the divide's tile straight to the output, so an op the dense body
        # applies after the divide is not slower in the paged kernel, it is absent from it.
        # Six such bodies used to be accepted and emit the shipped kernel byte for byte.
        # One tail op is enough: every one of them fires the same assert in the same place.
        (dict(post=True), "not the softmax divide"),
        # A matmul operand that is a computed tile rather than a declared input: the layer
        # rebuilds the matmuls against the spec's tensors, so the computation would vanish.
        (dict(q_pre=True), "Q operand is mul_q, not an input"),
        (dict(v_pre=True), "V operand is not an input"),
        # One tensor as both K and V. The layer gathers a K tile and a V tile from two
        # declared caches, so it would rebuild a body that reads the shared one twice.
        (dict(shared_cache=True), "exactly Q, K and V"),
    ],
)
def test_rejects_a_body_it_would_misrepresent(overrides, expected):
    with pytest.raises(AssertionError, match=expected):
        _paged_from(_dense(**overrides))


def _shipped_dense():
    return flash_attention_sd_graph(MQ, KV_PROXY, D, dtype=DTYPE, out_dtype=DTYPE)


def test_accepts_the_dense_attention_body():
    """The tiling it returns is the whole of what the access layer keeps.

    Every field here is what a builder reads in place of the dense graph, so a recognizer
    that handed back the shipped operator's numbers for any body it accepted would be the
    same fault as a builder hardcoding them -- and a call that drops the return value, as
    this one used to, cannot tell the difference.
    """
    assert read_dense_attention(
        tiled_dense_attention(_shipped_dense(), MQ, D, BN)
    ) == DenseAttention(
        query_rows=MQ,
        kv_tile=BN,
        head_dim=D,
        n_threads=128,
        dtype=DTYPE,
        accum_dtype=ACCUM_DTYPE,
        out_dtype=DTYPE,
    )


def test_rejects_a_split_head_dim():
    """A head dim the policy split is a fraction of the operator's, not the whole of it."""
    with pytest.raises(AssertionError, match="head dim is split"):
        apply_paging(tiled_dense_attention(_shipped_dense(), MQ, D // 2, BN), SPEC)


def test_rejects_a_partial_reduction_over_the_head_dim():
    """A score matmul that reduces over half the head dim computes half a score.

    The twin of the split-head-dim case above and of the MLA cases below: there the *output*
    tile is a fraction of D, here the output tile is whole and the reduction is split. The
    layer rebuilds the matmul against the operator's full width either way, so the split
    would be dropped and the kernel would attend on the wrong reduction.
    """
    steps = {"mma_0": {"v2": D // 2}, "mma_1": {"v2": BN}, "rowsum_0": {"v1": BN}}
    with pytest.raises(AssertionError, match="part of the head dim"):
        apply_paging(tiled_dense_attention(_shipped_dense(), MQ, D, BN, steps=steps), SPEC)


def test_rejects_a_dense_schedule_that_still_has_a_grid():
    """The paged grid is written from scratch, so a dense schedule's own blocks are lost.

    The twin of ``test_rejects_a_latent_schedule_that_still_has_a_grid``: 32 query rows
    tiled 16 at a time is a grid of 2, and the pass would emit its (batch, kv head) grid
    over it and compute half the query rows.
    """
    dense = flash_attention_sd_graph(2 * MQ, KV_PROXY, D, dtype=DTYPE, out_dtype=DTYPE)
    with pytest.raises(AssertionError, match="has a grid of"):
        apply_paging(tiled_dense_attention(dense, MQ, D, BN), SPEC)


def test_rejects_a_query_tile_smaller_than_the_head_group():
    """kv4 needs 8 query rows; a 4-row tile would read past the accumulator."""
    with pytest.raises(AssertionError, match="head group needs"):
        build_paged_graph(
            PagingSpec(num_qo_heads=32, num_kv_heads=4), query_rows=4, kv_tile=BN
        )


# --- the latent body ---------------------------------------------------------------
#
# The MLA recognizer duplicates every check above against a different body, so none of the
# cases above says anything about it: the dense MLA graph has two score matmuls whose
# results are summed, and its value matmul reads the compressed cache the first of them
# reduced against rather than a V of its own.

CKV, KPE = 512, 64
MLA_SPEC = MLASpec(num_qo_heads=MQ)


def _mla_steps(ckv, kpe, kv_tile=BN):
    """Reduction steps for a body with two score matmuls: each reduces its whole width."""
    return {
        "mma_0": {"v2": ckv},
        "mma_pe": {"v2": kpe},
        "mma_1": {"v2": kv_tile},
        "rowsum_0": {"v1": kv_tile},
    }


def _dense_mla(**overrides):
    """The dense latent-attention body, with one edge, dtype or constant deliberately wrong.

    ``kpe`` is an override because the two cases that make the score matmuls share an
    operand need the shared tensor to have one width.
    """
    ckv, kpe = overrides.get("ckv", CKV), overrides.get("kpe", KPE)
    s_q = overrides.get("query_tokens", MQ)
    scale = overrides.get("scale", ckv**-0.5)
    mma_0 = Matmul(
        (s_q, KV_PROXY, ckv),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    q, kv = mma_0.inputs[0].src_node, mma_0.inputs[1].src_node
    q.name, kv.name = "PlaceHolder_A", "PlaceHolder_B"
    mma_pe = Matmul(
        (s_q, KV_PROXY, kpe),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_pe",
        transpose_B=True,
        warp_policy="rows",
    )
    # MLA's nope and pe queries are separate inputs of different widths, and its two caches
    # likewise; either sharing is a body that computes something else.
    for slot, shared, name in ((0, overrides.get("shared_query"), "PlaceHolder_C"),
                               (1, overrides.get("shared_cache"), "PlaceHolder_D")):
        stale = mma_pe.inputs[slot].src_node
        if shared:
            _connect(q if slot == 0 else kv, mma_pe, 0, slot)
            stale._out_edges = [e for e in stale._out_edges if e.dst_node is not mma_pe]
        else:
            stale.name = name
    add_0 = Add((s_q, KV_PROXY), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="add_0")
    mul_0 = Mul(
        (s_q, KV_PROXY), const=scale, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="mul_0"
    )
    sub_0 = Sub(
        (s_q, KV_PROXY),
        const=(50.0 * scale),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="sub_0",
    )
    exp_0 = Exp((s_q, KV_PROXY), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="exp_0")
    mma_1 = Matmul(
        (s_q, ckv, KV_PROXY),
        dtype=overrides.get("value_dtype", DTYPE),
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
        transpose_B=overrides.get("value_transpose", False),
    )
    rowsum_0 = Reduction(
        (s_q, KV_PROXY), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    div_0 = Div(
        (s_q, ckv),
        broadcast=overrides.get("broadcast", 0),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )
    nodes = [mma_0, mma_pe, add_0, mul_0, sub_0, exp_0, mma_1, rowsum_0, div_0]

    _connect(mma_0, add_0, 0, 0)
    if overrides.get("scaled_pe"):
        # The positional scores scaled on their own: the sum no longer adds the two score
        # matmuls, and the layer's single runtime scale cannot express a per-term one.
        mul_pe = Mul(
            (s_q, KV_PROXY), const=2.0, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="mul_pe"
        )
        _connect(mma_pe, mul_pe, 0, 0)
        _connect(mul_pe, add_0, 0, 1)
        nodes.append(mul_pe)
    else:
        _connect(mma_pe, add_0, 0, 1)
    _connect(add_0, mul_0, 0, 0)
    _connect(mul_0, sub_0, 0, 0)
    if overrides.get("tensor_scale"):
        # A second *tensor* operand in the scaling chain -- a mask or a bias, not a constant
        # the runtime scale can absorb -- so the chain the recognizer walks is not scaling
        # alone, and the tensor is an input the layer would drop from the signature.
        mul_mask = Mul(
            (s_q, KV_PROXY), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="mul_mask"
        )
        _connect(sub_0, mul_mask, 0, 0)
        _connect(mul_mask, exp_0, 0, 0)
        nodes.append(mul_mask)
    else:
        _connect(sub_0, exp_0, 0, 0)
    _connect(exp_0 if not overrides.get("value_reads_scores") else sub_0, mma_1, 0, 0)
    # The compressed cache IS the value tensor; a value matmul with its own cache is a fifth
    # input the layer does not declare.
    if not overrides.get("value_own_cache"):
        _connect(kv, mma_1, 0, 1)
    _connect(exp_0 if not overrides.get("rowsum_reads_scores") else sub_0, rowsum_0, 0, 0)
    _connect(mma_1, div_0, 0, 0)
    _connect(rowsum_0, div_0, 0, 1)
    if overrides.get("q_pre"):
        _reroute(Mul((s_q, ckv), const=2.0, dtype=DTYPE, out_dtype=DTYPE, name="mul_q"), mma_0, 0)
    out_of = div_0
    if overrides.get("post"):
        out_of = _tail_op((s_q, ckv), "mul_post")
        _connect(div_0, out_of, 0, 0)
        nodes.append(out_of)
    OutputNode(out_of, dtype=DTYPE)
    return _finalize(*nodes)


def _tiled_mla(dense, *, ckv=CKV, kpe=KPE, out_ckv=None, steps=None, query_rows=MQ):
    """``dense`` at the tuned MLA tiling: one 16-row query tile, 256 threads."""
    return tiled_dense_attention(
        dense,
        query_rows,
        out_ckv or ckv,
        BN,
        n_threads=256,
        steps=steps or _mla_steps(ckv, kpe),
    )


def _paged_mla_from(overrides):
    ckv, kpe = overrides.get("ckv", CKV), overrides.get("kpe", KPE)
    return apply_paging(_tiled_mla(_dense_mla(**overrides), ckv=ckv, kpe=kpe), MLA_SPEC)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (dict(value_transpose=True), "must not transpose the cache"),
        # MLA's true scale comes from the pre-absorption qk width and arrives at runtime, but
        # a body scaled by anything other than the conventional constant for its own
        # reduction width is computing something other than attention.
        (dict(scale=2.0 * CKV**-0.5), "not 1/sqrt"),
        (dict(scale=1.0), "not 1/sqrt"),
        (dict(rowsum_reads_scores=True), "not the softmax exp"),
        (dict(value_reads_scores=True), "not the exp"),
        (dict(scaled_pe=True), "does not add the two score matmuls"),
        # The loop tier takes exactly four inputs. Sharing an operand between the two score
        # matmuls, or giving the value matmul a cache of its own, changes that count -- which
        # is the check that fires, ahead of the two that name the sharing directly.
        (dict(shared_cache=True, kpe=CKV), "the two query tensors and the two caches"),
        (dict(shared_query=True, kpe=CKV), "the two query tensors and the two caches"),
        (dict(value_own_cache=True), "the two query tensors and the two caches"),
        # The returned tiling carries one dtype pair, taken from the ckv score matmul, so all
        # three matmuls have to agree or the layer rebuilds a mixed-precision body in one
        # precision.
        (dict(value_dtype="float16"), "one precision for all three matmuls"),
        # Both layers store the divide's tile straight to the output. One tail op is enough:
        # every one of them fires the same assert in the same place.
        (dict(post=True), "not the softmax divide"),
        (dict(q_pre=True), "operand 0 is mul_q, not an input"),
        # The exp must read the summed scores through constant scaling alone. A tensor in
        # that chain is a mask or a bias -- a fifth input the emitted kernel does not take.
        (dict(tensor_scale=True), "constant scaling"),
        (dict(broadcast=1), "must broadcast the denominator"),
    ],
)
def test_rejects_a_latent_body_it_would_misrepresent(overrides, expected):
    with pytest.raises(AssertionError, match=expected):
        _paged_mla_from(overrides)


def test_accepts_the_dense_mla_body():
    """As for the dense attention body: the returned tiling is all the layer keeps.

    The two widths are separate fields because the latent body has two of them, and the
    builders read both -- the compressed one becomes the value matmul's N and the RoPE one
    the positional matmul's K -- so a recognizer that returned either of the shipped
    operator's numbers regardless of the graph would go unnoticed.
    """
    assert read_dense_mla(_tiled_mla(_shipped_mla())) == DenseMLA(
        query_rows=MQ,
        kv_tile=BN,
        head_dim_ckv=CKV,
        head_dim_kpe=KPE,
        n_threads=256,
        dtype=DTYPE,
        accum_dtype=ACCUM_DTYPE,
        out_dtype=DTYPE,
    )


def _shipped_mla():
    return mla_attention_sd_graph(MQ, KV_PROXY, CKV, KPE, dtype=DTYPE, out_dtype=DTYPE)


def test_rejects_a_split_compressed_head_dim():
    """The access layer replaces the grid, so head-dim blocks would be lost with it."""
    with pytest.raises(AssertionError, match="compressed head dim is split"):
        apply_paging(_tiled_mla(_shipped_mla(), out_ckv=CKV // 2), MLA_SPEC)


@pytest.mark.parametrize(
    "matmul, width, expected",
    [
        ("mma_0", CKV, "part of the compressed head dim"),
        ("mma_pe", KPE, "part of the RoPE head dim"),
    ],
)
def test_rejects_a_partial_reduction_over_a_head_dim(matmul, width, expected):
    """A score matmul that reduces over half its width computes half a score.

    The layer rebuilds both matmuls against the operator's full widths, so the split would
    be dropped and the kernel would silently attend on the wrong reduction.
    """
    steps = _mla_steps(CKV, KPE)
    steps[matmul] = {"v2": width // 2}
    with pytest.raises(AssertionError, match=expected):
        apply_paging(_tiled_mla(_shipped_mla(), steps=steps), MLA_SPEC)


def test_rejects_a_latent_schedule_that_still_has_a_grid():
    """The paged grid is written from scratch, so a dense schedule's own blocks are lost."""
    dense = mla_attention_sd_graph(2 * MQ, KV_PROXY, CKV, KPE, dtype=DTYPE, out_dtype=DTYPE)
    with pytest.raises(AssertionError, match="has a grid of"):
        apply_paging(_tiled_mla(dense), MLASpec(num_qo_heads=MQ))
