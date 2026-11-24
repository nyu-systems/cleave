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
(``_DERIVED``) and on disk (``store``, every stored rule checked and proved again on load).
The loop key has no extents, so two programs can share it; a rule holds for both (it is
proved on the flat graph, which the key pins), and so does a refusal read off the flat graph
alone.
A refusal that depends on the recipe (E1-E6, E10-E12, E15: the symbols, the extents, the
references) is kept under the loop key AND the recipe's key: another program of the same
loop structure derives its own. A rule of the loop structure is looked up first.
"""

from __future__ import annotations

import time

from ...ir.update_rule import loop_key

# loop_key -> Derivation (a rule, or a refusal of the flat graph alone), and
# (loop_key, Recipe.key) -> Derivation (a refusal of that recipe)
_DERIVED: dict = {}


def derive(flat_graph, obstacle, recipe):
    """The ``Derivation`` for one obstacle of ``flat_graph``, made at most once per loop
    structure (a refusal that depends on the recipe: once per loop structure and recipe);
    ``recipe`` builds the graph again at other extents."""
    from . import store
    from .engine import run
    from .texts import bug

    t0 = time.perf_counter()
    try:
        key = loop_key(flat_graph, obstacle.full, tuple(obstacle.loops))
        if key in _DERIVED:
            return _DERIVED[key]
        mine = recipe.key
        if (key, mine) in _DERIVED:
            return _DERIVED[(key, mine)]
        found = store.load(key, flat_graph, obstacle)
        if found is None:
            found = store.load(key, flat_graph, obstacle, recipe_key=mine)
        if found is None:
            found = run(flat_graph, obstacle, recipe, key=key)
            store.save(found)
            found = found.light()
        else:
            found.seconds = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - never out of plans.lower: the pin stays
        from .engine import Derivation

        return Derivation("", obstacle.full, tuple(obstacle.loops), refused=bug(e), bug=True,
                          seconds=time.perf_counter() - t0)
    if not found.bug:
        _DERIVED[(key, found.recipe_key) if found.recipe_key else key] = found
    return found
