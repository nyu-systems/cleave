"""Search-derived update rules for streaming obstacles.

``derive`` is called by ``pipeline.plans.lower`` for every streaming obstacle the scheduler's
own criterion finds on a freshly lowered flat graph (``schedule.streaming.obstacles``). It

  1. re-instantiates the flat graph at small shapes and evaluates it on prefixes of the
     looped dim (``reference``): the specification ``U(F(H), T) = F(H ++ T)``;
  2. enumerates update expressions per accumulator hole (``search``);
  3. proves each survivor over the reals with z3 (``proof``);
  4. ranks the proven ones (``rank``) and
  5. writes the chosen rule on the full-extent node as a ``Streaming`` record
     (``cleave.ir.update_rule``). The policy un-pins a loop only where it finds that record,
     and ``apply_tiling`` assembles the loop from it.

Nothing in here names softmax: the op set, the contribution of one element and the
references are all read off the flat graph.
"""

from __future__ import annotations

import time

from ...ir.update_rule import loop_key


def rebuilder(searched):
    """``rebuild(symbol_map, dtype)`` for a searched graph: the same structure lowered with
    other extents."""

    def rebuild(symbol_map, dtype):
        from ...lower.process import to_bundles

        bundles = to_bundles([searched.structure], symbol_map=dict(symbol_map),
                             split_factors=None, dtype=dtype, out_dtype=dtype)
        (bundle,) = bundles
        (graph,) = bundle
        return graph

    return rebuild


_DERIVED: dict = {}  # loop_key -> Derivation (one per loop structure per process)


def derive(flat_graph, obstacle, rebuild, symbols, *, budget: int = 4, looped=None):
    """The ``Derivation`` for one obstacle of ``flat_graph`` (cached per loop structure).
    ``looped``: the dim (a key of ``symbols``) the full-extent node reduces, when the
    caller knows it."""
    from .engine import run

    key = loop_key(flat_graph, obstacle.full, tuple(obstacle.loops))
    if key not in _DERIVED:
        t0 = time.perf_counter()
        _DERIVED[key] = run(flat_graph, obstacle, rebuild, symbols, budget=budget, key=key,
                            looped=looped)
        _DERIVED[key].seconds = time.perf_counter() - t0
    return _DERIVED[key]
