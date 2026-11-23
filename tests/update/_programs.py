"""Programs with the max written out, and how the engine tests reach their obstacles.

Every program here records its own ``exp(x - max(x))`` (``cl.softmax`` records the same);
the engine is called directly (``engine.run``) or through ``pipeline.plans.lower``.
"""

from __future__ import annotations

import functools

import cleave.lang as cl
from cleave import pipeline as P
from cleave.frontend import trace
from cleave.ir.update_rule import loop_key
from cleave.pipeline.superopt import GraphSet
from cleave.pipeline.update import engine
from cleave.pipeline.update.instantiate import Recipe
from cleave.schedule.streaming import obstacles
from cleave.schedule.tiling import reduction_ancestors, scheduled_reductions

EPS = 1e-6


def _max(x, dim=-1):
    return cl.max(x, dim, keepdim=True)


# ------------------------------------------------------------------------------- programs


def flash_attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                    v: cl.Tensor["H", "SKV", "D"]):
    """Flash form: the row sum and the value matmul are siblings, the division comes last."""
    s = (q @ k.mT) / q.shape[-1] ** 0.5
    e = cl.exp(s - _max(s))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def traced_attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
                     v: cl.Tensor["H", "SKV", "D"]):
    """Traced order: normalize, then the value matmul (two passes over the row)."""
    s = (q @ k.mT) / q.shape[-1] ** 0.5
    e = cl.exp(s - _max(s))
    return (e / cl.sum(e, -1, keepdim=True)) @ v


def flash_rows(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """Flash form whose logits are an input."""
    e = cl.exp(x - _max(x))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def softmax_rows(x: cl.Tensor["R", "C"]):
    """A returned softmax: the exp is divided by its own finished sum (two passes)."""
    e = cl.exp(x - _max(x))
    return e / cl.sum(e, -1, keepdim=True)


def column_average(x: cl.Tensor["H", "K", "N"], y: cl.Tensor["H", "K", "N"]):
    """``sum_k(softmax_k(x) * y)``: the softmax axis is the FIRST non-batch axis."""
    e = cl.exp(x - _max(x, 1))
    return cl.sum(e * y, 1, keepdim=True) / cl.sum(e, 1, keepdim=True)


def column_average_2d(x: cl.Tensor["K", "N"], y: cl.Tensor["K", "N"]):
    e = cl.exp(x - _max(x, 0))
    return cl.sum(e * y, 0, keepdim=True) / cl.sum(e, 0, keepdim=True)


def two_softmaxes(a: cl.Tensor["R", "C"], b: cl.Tensor["R", "C"]):
    """``sum(pa * pb) / (sum(pa) * sum(pb))``: one sum is scaled by both softmaxes."""
    pa, pb = cl.exp(a - _max(a)), cl.exp(b - _max(b))
    s = cl.sum(pa * pb, -1, keepdim=True)
    return s / (cl.sum(pa, -1, keepdim=True) * cl.sum(pb, -1, keepdim=True))


def scaled_rows(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """``exp((x - max(x)) / 8)``: a constant between the max and the sums."""
    e = cl.exp((x - _max(x)) / 8.0)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def tenth_rows(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """``exp((x - max(x)) * 0.1)``: a constant that is no decimal in binary."""
    e = cl.exp((x - _max(x)) * 0.1)
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def left_constant(x: cl.Tensor["R", "C"], v: cl.Tensor["C", "D"]):
    """``1 / exp(max(x) - x)``: a constant on the LEFT of a division between the max and
    the sums (the same softmax weights as ``flash_rows``)."""
    e = 1.0 / cl.exp(x * -1.0 + _max(x))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def gqa(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"],
        v: cl.Tensor["HKV", "SKV", "D"]):
    s = (q @ k.mT) / q.shape[-1] ** 0.5
    e = cl.exp(s - _max(s))
    return (e @ v) / cl.sum(e, -1, keepdim=True)


def mla_decode(q: cl.Tensor["B", "H", "D"], kv: cl.Tensor["B", "SKV", "D"],
               q_pe: cl.Tensor["B", "H", "PE"], k_pe: cl.Tensor["B", "SKV", "PE"]):
    """demo/mla_decode.py: the latent kv is both K and V."""
    return cl.softmax((q @ kv.mT + q_pe @ k_pe.mT) / 128 ** 0.5, dim=-1) @ kv


def qknorm(q: cl.Tensor["HQ", "SQ", "D"], k: cl.Tensor["HKV", "SKV", "D"],
           v: cl.Tensor["HKV", "SKV", "D"]):
    """demo/qknorm.py: RMS-normalized q and k, then attention."""
    q_sq, k_sq = q * q, k * k
    q_ms, k_ms = q_sq.mean(-1, keepdim=True), k_sq.mean(-1, keepdim=True)
    qn = q * cl.rsqrt(q_ms + EPS)
    kn = k * cl.rsqrt(k_ms + EPS)
    return cl.softmax((qn @ kn.mT) / qn.shape[-1] ** 0.5, dim=-1) @ v


def attention(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"],
              v: cl.Tensor["H", "SKV", "D"]):
    return cl.softmax((q @ k.mT) / q.shape[-1] ** 0.5, dim=-1) @ v


# --------------------------------------------------------------------------------- access


@functools.lru_cache(maxsize=None)
def searched(fn, shapes: tuple, max_ops=None, dtype=cl.f16):
    """``(program, graphs)`` of ``fn`` (``shapes`` as sorted items)."""
    program = trace(fn, dict(shapes), dtype)
    config = P.SuperoptConfig(max_ops=max_ops) if max_ops else P.SuperoptConfig()
    return program, P.superoptimize(program, config)


def identity(fn, shapes: dict, dtype=cl.f16):
    """The un-searched graph of ``fn`` (``SearchedGraph`` with no symbol map)."""
    (graph,) = list(GraphSet.identity(trace(fn, shapes, dtype)))
    return graph


def lowered(graph):
    """The flat graph of ``graph`` as ``plans.lower`` lowers it, without any derivation."""
    from cleave.pipeline.plans import _lower_structure

    (ir,) = _lower_structure(graph, graph.structure)
    return ir


def the_obstacle(ir):
    """The one obstacle of ``ir`` that passes the structural rules."""
    nodes, red = scheduled_reductions(ir)
    found = [ob for ob in obstacles(nodes, red, reduction_ancestors(nodes)) if not ob.refused]
    (ob,) = found
    return ob


def derivation(graph, ir=None, **kw):
    """``engine.run`` for the one obstacle of ``graph``'s flat graph (no memo, no store)."""
    ir = lowered(graph) if ir is None else ir
    ob = the_obstacle(ir)
    return engine.run(ir, ob, Recipe.of(graph), key=loop_key(ir, ob.full, ob.loops), **kw)


MHA = tuple(dict(H=32, SQ=4096, SKV=4096, D=64).items())  # demo/mha_prefill.py


def flash(graphs):
    """The graph whose row sum and value matmul are siblings (the divide comes last)."""
    (graph,) = [g for g in graphs if g.summary().endswith("sum>matmul>div")]
    return graph


@functools.lru_cache(maxsize=None)
def mha_derivation(budgets=(4,)):
    """The derivation of MHA prefill's flash form (``max_ops=11``) at ``budgets``."""
    _, graphs = searched(attention, MHA, 11)
    return derivation(flash(graphs), budgets=budgets)
