"""Search-derived update rules for reduction conflicts, tested in the first pass's field.

``derive`` is called by ``pipeline.plans.lower`` for every reduction conflict of a freshly
lowered flat graph (``schedule.streaming.obstacles``: a reduction that a later reduction over
the same dim depends on, whatever the ops). It

  1. builds the flat graph again at the first pass's test sizes with the looped dim at a few
     elements (``instantiate.Recipe``) and evaluates it on prefixes of that dim in the field
     of the first pass's fingerprint (``reference``, ``field``): the program's own states,
     ``F(H)`` and ``F(H ++ T)``;
  2. enumerates update expressions per accumulator and keeps those that reproduce
     ``F(H ++ T)`` exactly on every element, residue by residue, where neither the test data
     nor the update divides by zero in the field (``search``);
  3. replays every combination of the ones the assembly can emit in floating point and
     ranks them (``rank``), and
  4. returns a ``Derivation`` whose ``record()`` is the ``Streaming`` record
     (``cleave.ir.update_rule``) ``plans.lower`` writes on the full-extent node. The policy
     un-pins a loop only where it finds that record, and ``apply_tiling`` assembles the
     loop from it.

Nothing in here names an op of a program: the ops a candidate may use and the test data
are read off the flat graph. A conflict without a rule keeps its pin, with one of the
reasons of ``texts`` (no test, no rule, no rule survives the replay); ``derive`` never
raises.

A derivation is made once per loop structure (``update_rule.loop_key``): in a process
(``_DERIVED``) and on disk (``store``, every stored rule tested again on load). The loop key
has no extents, so two programs can share it; a rule holds for both. A refusal is kept under
the loop key AND the recipe's key: another program of the same loop structure derives its
own. A rule of the loop structure is looked up first.
"""

from __future__ import annotations

import time

from ...ir.update_rule import loop_key

# loop_key -> Derivation (a rule), and (loop_key, Recipe.key) -> Derivation (a refusal)
_DERIVED: dict = {}


def derive(flat_graph, obstacle, recipe):
    """The ``Derivation`` for one conflict of ``flat_graph``, made at most once per loop
    structure (a refusal: once per loop structure and recipe); ``recipe`` builds the graph
    again at other extents."""
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
        found = store.load(key, flat_graph, obstacle, recipe)
        if found is None:
            found = store.load(key, flat_graph, obstacle, recipe, recipe_key=mine)
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
