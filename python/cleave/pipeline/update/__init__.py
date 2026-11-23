"""Search-derived update rules for streaming obstacles.

``derive`` is called by ``pipeline.plans.lower`` for every streaming obstacle the scheduler's
own criterion finds on a freshly lowered flat graph (``schedule.streaming.obstacles``). It

  1. builds the flat graph again at small shapes (``instantiate.Recipe``) and evaluates it
     on prefixes of the looped dim (``reference``): the specification
     ``U(F(H), T) = F(H ++ T)``;
  2. enumerates update expressions per accumulator hole (``search``);
  3. proves each survivor over the reals with z3 (``proof``);
  4. ranks the proven ones the assembly can emit (``rank``) and
  5. returns a ``Derivation`` whose ``record()`` is the ``Streaming`` record
     (``cleave.ir.update_rule``) ``plans.lower`` writes on the full-extent node. The policy
     un-pins a loop only where it finds that record, and ``apply_tiling`` assembles the
     loop from it.

Nothing in here names softmax: the op set, the contribution of one element and the
references are all read off the flat graph. Every reason an obstacle keeps its pin is a
typed refusal (``texts``); ``derive`` never raises.

A derivation is made once per loop structure (``update_rule.loop_key``): in a process
(``_DERIVED``) and on disk (``store``, every stored rule proved again on load).
"""

from __future__ import annotations

import time

from ...ir.update_rule import loop_key

_DERIVED: dict = {}  # loop_key -> Derivation (one per loop structure per process)


def derive(flat_graph, obstacle, recipe):
    """The ``Derivation`` for one obstacle of ``flat_graph``, made at most once per loop
    structure; ``recipe`` builds the graph again at other extents."""
    from . import store
    from .engine import run
    from .texts import bug

    t0 = time.perf_counter()
    try:
        key = loop_key(flat_graph, obstacle.full, tuple(obstacle.loops))
        if key in _DERIVED:
            return _DERIVED[key]
        found = store.load(key, flat_graph, obstacle)
        if found is None:
            found = run(flat_graph, obstacle, recipe, key=key)
            store.save(found)
        else:
            found.seconds = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - never out of plans.lower: the pin stays
        from .engine import Derivation

        return Derivation("", obstacle.full, tuple(obstacle.loops), refused=bug(e), bug=True,
                          seconds=time.perf_counter() - t0)
    if not found.bug:
        _DERIVED[key] = found
    return found
