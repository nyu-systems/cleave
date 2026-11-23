"""The on-disk store of derivations (DESIGN 2.7, place 3).

One JSON file per loop structure, under ``cache.root() / "update"``, at the digest of

    "update", ENGINE_DIGEST, SETTINGS, the record key (``update_rule.loop_key``)[, recipe key]

(``cache.digest`` mixes in ``SCHEMA`` and ``build_id()``). ``ENGINE_DIGEST`` is a digest of
the engine's own sources (``ir/update_rule.py`` and every module of ``pipeline/update``);
``SETTINGS`` the budgets, the loose filter, the reference-set constants and the z3 version.
Content: ``{"key", "recipe", "full", "loops", "rules", "refused", "report"}``; written
atomically.

The loop structure's entry (``recipe`` "") holds a rule, or a refusal read off the flat
graph alone (T5, E7, E8, E9, E14). A refusal that depends on the recipe (the extents, the
symbols, the data of the references: E1-E6, E10-E12, E15) is stored for that recipe only,
under the record key and ``Recipe.key``: another program of the same loop structure
derives its own.

A stored rule is never trusted: on load every rule is proved again on the graph at hand
(``proof.prove(update_of(F), ...)``). A rule that is not proved, a key or recipe that
differs from the stored one, or an unreadable file is a miss: the caller derives and
overwrites. A stored refusal is returned as it is (it only keeps a pin). ``E13`` is never
stored.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from ...ir.update_rule import update_of

_HERE = Path(__file__).resolve().parent


def engine_sources() -> list:
    """The files the engine's digest is taken over, in a fixed order."""
    return [_HERE.parent.parent / "ir" / "update_rule.py"] + sorted(_HERE.glob("*.py"))


def engine_digest(paths=None) -> str:
    """sha256 over the name and bytes of every engine source."""
    h = hashlib.sha256()
    for p in paths if paths is not None else engine_sources():
        p = Path(p)
        h.update(p.name.encode() + b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


ENGINE_DIGEST = engine_digest()


def settings() -> list:
    """Everything outside the sources that a derivation depends on."""
    import z3

    from . import engine, reference, search

    return [list(engine.BUDGETS), search.LOOSE_RTOL, reference.TILE, reference.N,
            list(reference.PREFIXES), list(reference.SEEDS_NORMAL), list(reference.TRY_SEEDS),
            [list(s) for s in reference.SHAPINGS], [list(c) for c in engine.RANK_CASES],
            engine.RANK_N, z3.get_version_string()]


def entry_key(record_key: str, recipe_key: str = "") -> str:
    from .. import cache

    parts = [record_key] + ([recipe_key] if recipe_key else [])
    return cache.digest("update", ENGINE_DIGEST, settings(), *parts)


def path_of(record_key: str, recipe_key: str = "") -> Path:
    """The entry of a loop structure (``recipe_key`` ""), or of one recipe's refusal."""
    from .. import cache

    k = entry_key(record_key, recipe_key)
    return cache.root() / "update" / k[:2] / f"{k}.json"


def _tuples(x):
    """An expression read back from JSON: lists become tuples."""
    return tuple(_tuples(y) for y in x) if isinstance(x, list) else x


def save(d) -> Path | None:
    """Write the derivation ``d`` (not an ``E13``). Returns the path, or None when the
    store cannot be written (the derivation is still served)."""
    from .engine import report

    if d.bug:
        return None
    entry = {"key": d.key, "recipe": d.recipe_key, "full": d.full, "loops": list(d.loops),
             "rules": [[acc, factor] for acc, factor in d.rules], "refused": d.refused,
             "report": report(d)}
    try:
        p = path_of(d.key, d.recipe_key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.stem, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(entry, f, sort_keys=True)
        os.replace(tmp, p)
        return p
    except OSError:
        return None


def load(record_key: str, flat_graph, obstacle, recipe_key: str = ""):
    """The stored derivation of ``record_key`` (with ``recipe_key``: that recipe's stored
    refusal) with every rule proved again on ``flat_graph``; None on a miss."""
    from .engine import Derivation
    from .proof import prove

    try:
        entry = json.loads(path_of(record_key, recipe_key).read_text())
        if (entry["key"] != record_key or entry["recipe"] != recipe_key
                or entry["full"] != obstacle.full
                or tuple(entry["loops"]) != tuple(obstacle.loops)):
            return None
        rules = tuple((acc, _tuples(factor)) for acc, factor in entry["rules"])
        refused, text = str(entry["refused"]), str(entry["report"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    d = Derivation(record_key, obstacle.full, tuple(obstacle.loops), refused=refused,
                   from_store=True, report_text=text, recipe_key=recipe_key)
    if refused:
        return d
    if recipe_key:  # only refusals are stored for one recipe
        return None
    if tuple(acc for acc, _ in rules) != tuple(obstacle.loops):
        return None
    for acc, factor in rules:
        if not prove(update_of(factor), flat_graph, obstacle.full, acc).proved:
            return None
    d.rules = rules
    d.chosen = {acc: update_of(factor) for acc, factor in rules}
    return d
