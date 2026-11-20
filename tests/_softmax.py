"""Shift-free softmax programs and their ``Stabilize`` records, for the rewrite tests (CPU).

The stabilization rewrite (``cleave.schedule.stabilize``) consumes a record the scale
analysis writes on each softmax ``Exp`` at lowering time. The programs here are written out
(the requested ``exp``, ``sum``, ``div``) and lowered as identity graphs; :func:`softmax_ir`
then CHECKS the record lowering wrote on every softmax Exp against the one the test expects:
same softmax axis, same scaled reductions (read off the graph: the first reductions /
contractions reached from the Exp), an obstacle to the rolling form exactly where one is
expected, same exported max.

There is no other mode: an Exp that lowering left without a record fails the test that
lowered it, and so does a program that cannot make the request. The rewrite tests therefore
always run on the analysis's own records.
"""

from __future__ import annotations

import cleave.lang as cl
from cleave.frontend import trace
from cleave.ir.graph import OutputNode
from cleave.ir.ops import Div, Exp, Matmul, Mul, ReduceMax, ReduceSum, Stabilize
from cleave.pipeline.plans import lower
from cleave.pipeline.superopt import GraphSet

REDUCERS = (ReduceSum, ReduceMax, Matmul)


def stable_exp(x, dim: int = -1):
    """The ``exp`` of a softmax over ``dim``, as the frontend requests it (what
    ``cl.softmax`` records in front of its sum and its division)."""
    return x.program._stable_exp(x, dim)


def exact_softmax(x, dim: int = -1):
    """``softmax(x, dim)`` as the frontend records it: the requested exp, its sum, the
    division. No max and no shift."""
    return cl.softmax(x, dim)


# ------------------------------------------------------------------------------- programs


def flash_attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                    v: cl.Tensor["H", "SKV", "D"]):
    """Flash form: the row sum and the value matmul are siblings, the division comes last."""
    e = stable_exp((q @ k.mT) / q.shape[-1] ** 0.5)
    l = cl.sum(e, -1, keepdim=True)
    return (e @ v) / l


def traced_attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                     v: cl.Tensor["H", "SKV", "D"]):
    """Traced order: normalize, then the value matmul. The exp is read next to its sum."""
    e = stable_exp((q @ k.mT) / q.shape[-1] ** 0.5)
    return (e / cl.sum(e, -1, keepdim=True)) @ v


def unscaled_attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                       v: cl.Tensor["H", "SKV", "D"]):
    """Traced order without the 1/sqrt(d): the exp reads the QK^T matmul itself."""
    e = stable_exp(q @ k.mT)
    return (e / cl.sum(e, -1, keepdim=True)) @ v


def softmax_rows(x: cl.Tensor["R", "C"]):
    e = stable_exp(x)
    return e / cl.sum(e, -1, keepdim=True)


def flash_rows(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """Flash form whose scores are an input: the exp reads storage, not a float32 tile."""
    e = stable_exp(x)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def column_average(x: cl.Tensor["H", "K", "N"], y: cl.Tensor["H", "K", "N"]):
    """``sum_k(softmax_k(x) * y)``: the softmax axis is the FIRST non-batch axis of the exp;
    both sums reduce it, so their accumulators lead with the remaining axis ``N``."""
    e = stable_exp(x, 1)
    return cl.sum(e * y, 1, keepdim=True) / cl.sum(e, 1, keepdim=True)


def column_average_2d(x: cl.Tensor["K", "N"], y: cl.Tensor["K", "N"]):
    """:func:`column_average` without the batch dim: no unit dim sits in front of the
    softmax axis, so a tile that holds all of ``K`` can take the max in place."""
    e = stable_exp(x, 0)
    return cl.sum(e * y, 0, keepdim=True) / cl.sum(e, 0, keepdim=True)


def two_softmaxes(a: cl.Tensor["R", "C"], b: cl.Tensor["R", "C"]):
    """``sum(pa * pb) / (sum(pa) * sum(pb))``: one sum is scaled by both softmaxes."""
    pa, pb = stable_exp(a), stable_exp(b)
    s = cl.sum(pa * pb, -1, keepdim=True)
    return s / (cl.sum(pa, -1, keepdim=True) * cl.sum(pb, -1, keepdim=True))


def max_normalized(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """``(e @ v) / max(e)``: a max of the scaled value; it cannot be rescaled from zero."""
    e = stable_exp(x)
    return (e @ v) / cl.max(e, -1, keepdim=True)


def column_weighted(v: cl.Tensor["M", "K"], x: cl.Tensor["K", "N"]):
    """``(v @ e) / sum(e, 0)``: the factor lives on the accumulator's TRAILING axis."""
    e = stable_exp(x, 0)
    return (v @ e) / cl.sum(e, 0, keepdim=True)


def sum_next_to_exp(x: cl.Tensor["R", "C"]):
    """``sum(e + l) / l``: the exp is added to its own finished row sum."""
    e = stable_exp(x)
    l = cl.sum(e, -1, keepdim=True)
    return cl.sum(e + l, -1, keepdim=True) / l


# -------------------------------------------------------------------------------- records


def lowered(fn, shapes, dtype=cl.f16):
    """The identity (un-searched) graph of ``fn``, lowered to IR."""
    (graph,) = list(GraphSet.identity(trace(fn, shapes, dtype)))
    (ir,) = lower(graph)
    return ir


def exps(ir) -> list:
    return [n for n in ir.nodes if type(n) is Exp]


def _computed_from(node, source) -> bool:
    stack, seen = [node], set()
    while stack:
        n = stack.pop()
        if n is source:
            return True
        if id(n) not in seen:
            seen.add(id(n))
            stack.extend(e.src_node for e in n.inputs)
    return False


def scaled_reductions(exp) -> tuple:
    """``((name, 1), ...)``: the reductions / contractions that sum a value still carrying
    the scale of ``exp``: the first ones reached from it. The walk stops where the value is
    divided by one of the exp's own reductions: the quotient no longer carries the scale,
    so a matmul behind ``exp / sum(exp)`` is not one of them."""
    found, stack, seen = [], [exp], set()
    while stack:
        node = stack.pop(0)
        for edge in node.outputs:
            dst = edge.dst_node
            if id(dst) in seen or isinstance(dst, OutputNode):
                continue
            seen.add(id(dst))
            if isinstance(dst, REDUCERS):
                found.append(dst.name)
            elif (type(dst) is Div and dst.const is None and edge.dst_id == 0
                  and _computed_from(dst.inputs[1].src_node, exp)):
                continue
            else:
                stack.append(dst)
    return tuple((name, 1) for name in found)


def hand_record(exp):
    """The record of a softmax Exp read off the graph by hand, or ``None`` where this rule
    cannot tell.

    A second derivation of what the scale analysis writes, independent of it: the tests
    hold the two against each other (``tests/schedule/search_tiling/
    test_stabilize_census.py``). Nothing sets a record from it. The rule covers a softmax
    over the LAST axis of the Exp:

    * ``scaled``: :func:`scaled_reductions`; each of them has to consume the Exp's last
      axis (a ``ReduceSum`` / ``ReduceMax`` over the last axis, or a ``Matmul`` whose left
      operand it is);
    * ``rolling``: the one obstacle searched softmax graphs have, the Exp read next to one
      of its own finished reductions (traced order: ``exp / sum(exp)`` before the value
      matmul).

    """
    scaled = scaled_reductions(exp)
    if not scaled:
        return None
    names = {name for name, _ in scaled}
    last = len(exp.get_shape()) - 1
    reducers, stack, seen = [], [exp], {id(exp)}
    while stack:
        node = stack.pop()
        for edge in node.outputs:
            dst = edge.dst_node
            if isinstance(dst, OutputNode):
                continue
            if dst.name in names:
                if isinstance(dst, Matmul):
                    over_last = edge.dst_id == 0 and not getattr(dst, "transpose_A", False)
                else:
                    over_last = dst.axis_arg in (-1, last)
                over_last = over_last and tuple(edge.src_node.get_shape()) == tuple(exp.get_shape())
                if not over_last:
                    return None
                if not any(dst is r for r in reducers):
                    reducers.append(dst)
            elif id(dst) not in seen:
                seen.add(id(dst))
                stack.append(dst)
    rolling = ""
    stack, seen = [exp], {id(exp)}
    while stack and not rolling:
        node = stack.pop()
        for edge in node.outputs:
            dst = edge.dst_node
            if isinstance(dst, OutputNode) or dst.name in names or id(dst) in seen:
                continue
            seen.add(id(dst))
            finished = next((r for r in reducers if _computed_from(dst, r)), None)
            if finished is not None:
                rolling = (f"{dst.name} reads the exp next to the finished reduction "
                           f"{finished.name}")
                break
            stack.append(dst)
    return Stabilize(axes=(-1,), scaled=scaled, rolling=rolling)


def softmax_ir(fn, shapes, *softmax, dtype=cl.f16):
    """Lower ``fn`` and check that each Exp carries the record ``softmax`` describes.

    One dict per Exp of the lowered graph, in graph order: ``axis`` (from the end, default
    -1), ``rolling`` (the obstacle text, default none), ``export`` (default none). No dict
    at all means one softmax over the last axis that has a rolling form.
    """
    ir = lowered(fn, shapes, dtype)
    nodes = exps(ir)
    softmax = softmax or ({},) * len(nodes)
    assert len(softmax) == len(nodes), f"{len(nodes)} exps, {len(softmax)} expectations"
    for exp, want in zip(nodes, softmax):
        record = Stabilize(
            axes=(want.get("axis", -1),),
            scaled=scaled_reductions(exp),
            rolling=want.get("rolling", ""),
            export=want.get("export"),
        )
        got = exp.stabilize  # written by the scale analysis: it has to be there, and agree
        assert got is not None, f"{exp.name}: lowering wrote no record (expected {record})"
        assert got.axes == record.axes, (exp.name, got, record)
        assert dict(got.scaled) == dict(record.scaled), (exp.name, got, record)
        assert bool(got.rolling) == bool(record.rolling), (exp.name, got, record)
        assert got.export == record.export, (exp.name, got, record)
    return ir


def raw_statistics_graph(BH, S_Q, S_KV, D):
    """The shape of a split partial, without a split: the kernel's outputs are the
    unnormalized statistics of one softmax, and the shift they are relative to.

        scores = Q K^T / sqrt(D)
        exp    = Exp(scores)              marked; its shift is the exported max below
        l      = ReduceSum(exp)       ->  OutputNode_0  (BH, S_Q)      float16
        o      = exp @ V              ->  OutputNode_1  (BH, S_Q, D)   float16
        m      = ReduceMax(scores)    ->  OutputNode_2  (BH, S_Q)      float32

    Every output has degree 1 in the softmax's scale, so nothing cancels a wrong shift:
    ``l`` and ``o`` are only right relative to exactly the ``m`` the kernel returns.
    """
    from tests._graphs import ACCUM_DTYPE, DTYPE, OUT_DTYPE, _connect, _finalize

    f32 = dict(dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE)
    scores = (BH, S_Q, S_KV)
    mma_0 = Matmul((BH, S_Q, S_KV, D), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_0",
                   transpose_B=True, warp_policy="rows")
    mul_0 = Mul(scores, const=1.0 / (D**0.5), name="mul_0", **f32)
    exp_0 = Exp(scores, name="exp_0", **f32)
    rowsum_0 = ReduceSum(scores, name="rowsum_0", **f32)
    mma_1 = Matmul((BH, S_Q, D, S_KV), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_1",
                   warp_policy="rows")
    row_max = ReduceMax(scores, axis=-1, name="reduce_max_2_exp_0", **f32)
    _connect(mma_0, mul_0, 0, 0)
    _connect(mul_0, exp_0, 0, 0)
    _connect(exp_0, rowsum_0, 0, 0)
    _connect(exp_0, mma_1, 0, 0)
    _connect(mul_0, row_max, 0, 0)
    for i, (node, dtype) in enumerate(
        ((rowsum_0, OUT_DTYPE), (mma_1, OUT_DTYPE), (row_max, "float32"))
    ):
        OutputNode(node, dtype=dtype).name = f"OutputNode_{i}"
    exp_0.stabilize = Stabilize(
        axes=(-1,), scaled=(("rowsum_0", 1), ("mma_1", 1)), export=row_max.name
    )
    return _finalize(mma_0, mul_0, exp_0, rowsum_0, mma_1, row_max)
