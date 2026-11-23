"""Searched update rules: the plain data the scheduler, the engine and the assembly share.

A reduction that wants to loop over a dim can sit under a value that needs the whole dim --
the row max of ``exp(x - max(x))`` -- which is itself a reduction over that dim and feeds
back into the loop. Such a pair is an :class:`Obstacle`. Where an update rule was found and
proven for every reduction that loops under the full-extent value
(``cleave.pipeline.update``), the full-extent node of the flat graph carries a
:class:`Streaming` record, and the loop is built with the value carried across its
iterations (``cleave.schedule.streaming``).

Everything here is data, hashing and expression helpers; nothing above ``ir`` is imported.

An *expression* is a nested tuple: a leaf ``"a"`` (the accumulator before the iteration),
``"m"`` / ``"m'"`` (the full-extent value before / after it), ``"r"`` (what the tile
computation adds, computed with ``m'``); ``(unary, e)``; or ``(op, e1, e2)`` with ``op`` in
``add sub mul div``. Constants never occur.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass

# ------------------------------------------------------------------------------ the switch
#
# The searched path is dormant until the switch-over: ``cl.softmax`` records the request of
# the hand-derived pass unless this is on. Read at four places only (``frontend.composites.
# softmax``, ``pipeline.plans.lower``, ``runtime.inputs.softmax_axis`` and, through that,
# ``pipeline.build._reference_fn``) and, so that a kernel saved under one setting is never
# returned under the other, in the kernel-cache key (``pipeline.compile._kernel_key``).

SWITCH_ENV = "CLEAVE_SEARCHED_UPDATES"
_SEARCHED = os.environ.get(SWITCH_ENV, "") not in ("", "0")


def searched_updates() -> bool:
    """Whether ``cl.softmax`` records the explicit max and update rules are searched."""
    return _SEARCHED


@contextmanager
def use_searched_updates(on: bool = True):
    """The switch set to ``on`` inside the block (also in the environment, so a process
    started inside it starts with the same setting), restored on exit."""
    global _SEARCHED
    before, env_before = _SEARCHED, os.environ.get(SWITCH_ENV)
    _SEARCHED = bool(on)
    os.environ[SWITCH_ENV] = "1" if on else "0"
    try:
        yield
    finally:
        _SEARCHED = before
        if env_before is None:
            os.environ.pop(SWITCH_ENV, None)
        else:
            os.environ[SWITCH_ENV] = env_before


# ----------------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Obstacle:
    """One full-extent value and the reductions that want to loop under it.

    It is also the carried-state proposal: the state of the loop is ``loops`` (the
    accumulators) plus ``full``. Nothing else is state.
    """

    full: str  # name of the full-extent reduction (the row max)
    loops: tuple  # names of the reductions that want to loop under it, in graph order
    refused: str = ""  # why the overlap is NOT a streaming obstacle ("" = it is one)


@dataclass(frozen=True)
class Streaming:
    """Plain-data record on the full-extent node of a flat graph (``node.streaming``).

    ``rules``: ``((accumulator name, factor expression), ...)`` in ``loops`` order. A factor
    is an expression over ``"m"`` and ``"m'"`` only, ``("exp", ("sub", "m", "m'"))``: the
    accumulator is multiplied by it, in place, right before its own macro accumulates the
    tile (the update is ``update_of(factor)``). ``key`` is ``loop_key`` of the graph the
    rules were derived for; a record is honoured only on a graph with that key.
    """

    loops: tuple
    rules: tuple
    key: str


# ---------------------------------------------------------------------------- expressions

LEAVES = ("a", "m", "m'", "r")
BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def show(expr) -> str:
    """``expr`` as text, fully parenthesized: ``(r + (a * exp((m - m'))))``."""
    if isinstance(expr, str):
        return expr
    op, *args = expr
    if op in BINARY:
        return f"({show(args[0])} {BINARY[op]} {show(args[1])})"
    return f"{op}({show(args[0])})"


def size(expr) -> int:
    """Op nodes of the tree (a shared subterm counts every time it is written)."""
    return 0 if isinstance(expr, str) else 1 + sum(size(a) for a in expr[1:])


def subterms(*exprs) -> list:
    """The distinct op subterms of ``exprs`` taken together, operands before the op that
    reads them, in the order they are first needed: the nodes ``ir.rolling.build_expression``
    makes for them, and what they cost a loop per iteration."""
    found = []

    def walk(e):
        if isinstance(e, str) or e in found:
            return
        for x in e[1:]:
            walk(x)
        if e not in found:
            found.append(e)

    for e in exprs:
        walk(e)
    return found


def uses(expr, leaf: str) -> bool:
    """Whether ``leaf`` occurs in ``expr``."""
    return expr == leaf if isinstance(expr, str) else any(uses(x, leaf) for x in expr[1:])


def repair_of(update):
    """The factor ``F`` of an update of the form "repair, then accumulate" that the
    assembly can emit, or None.

    ``update`` is ``("add", "r", ("mul", "a", F))`` up to the order of the operands of
    ``add`` and ``mul``, and ``F`` uses neither ``"a"`` nor ``"r"``. Anything else has no
    factor: the loop-carried mechanism multiplies an accumulator in place and then lets its
    own macro add the tile, nothing more.
    """
    if isinstance(update, str) or update[0] != "add":
        return None
    x, y = update[1], update[2]
    g = x if y == "r" else y if x == "r" else None
    if g is None or isinstance(g, str) or g[0] != "mul":
        return None
    factor = g[2] if g[1] == "a" else g[1] if g[2] == "a" else None
    if factor is None or uses(factor, "a") or uses(factor, "r"):
        return None
    return factor


def update_of(factor):
    """The canonical update of a factor: ``a * factor + r``."""
    return ("add", "r", ("mul", "a", factor))


# ------------------------------------------------------------------------------- identity
#
# What makes two flat graphs the same loop structure. A rule holds for every shape and
# every tile of a loop structure, so the key has no extents and no tiling in it.
#
# Every attribute of a class lowering can emit (``lower.convert._RULES``) that says what a
# node computes or how it is laid out is in exactly one of the two lists
# (``tests/update/test_update_rule.py`` checks every ``_macro_attrs`` entry of those classes):
#
# SEMANTIC: changes what the node computes. EXTENTS: made of extents (``factor`` is the
# number of splits of a ``SplitView``: the partials of every split factor share one key), or
# a storage / layout choice that does not change the value. (Not listed: a node's name and
# wiring, which the signature holds itself; ``out_dtype``, which it holds with ``dtypes``;
# and bookkeeping that does not reach the value: ``op_index``, ``input_index``, ``boundary``.)
SEMANTIC = (
    "n_batch_dims", "is_batch_dim", "input_is_batch_dim",
    "const", "const_on_left", "broadcast", "denom_floor",
    "axis", "flat_axis", "axis_arg", "flat_axis_arg",
    "transpose_A", "transpose_B", "perm", "split_dim_is_batch", "split_pos", "inner_pos",
    "accum_init",
)
EXTENTS = (
    # extents
    "dims", "output_dims", "batch_dims", "raxis", "M", "N", "K",
    "src_shape", "input_shape", "output_shape", "inner_extent", "factor",
    "input_batch_dims_A", "input_batch_dims_B",
    # storage / layout
    "dtype", "resolve_operand_layout", "warp_policy", "stored_axis",
)
KEY_VERSION = 1


def _dtype_of(node) -> str:
    try:
        return str(node.get_dtype(0))
    except IndexError:  # a placeholder whose dtype slot was never filled
        return ""


def signature(graph, dtypes: bool) -> list:
    """``graph`` as plain data, without extents: per node, in order, its class, name,
    wiring (by operand), semantic attributes and -- with ``dtypes`` -- the dtype of its
    value and the dtype it computes in."""
    rows = []
    for n in graph.nodes:
        edges = sorted(getattr(n, "inputs", ()), key=lambda e: e.dst_id)
        row = [
            type(n).__name__,
            n.name,
            [[e.src_node.name, int(e.src_id), int(e.dst_id)] for e in edges],
            [[k, repr(getattr(n, k))] for k in SEMANTIC if hasattr(n, k)],
        ]
        if dtypes:
            out_dtype = getattr(n, "out_dtype", None)
            row.append([_dtype_of(n)] + ([str(out_dtype)] if out_dtype is not None else []))
        else:
            row.append([])
        rows.append(row)
    return rows


def loop_key(flat_graph, full: str, loops: tuple) -> str:
    """The identity of one loop structure: the flat graph (``signature`` with dtypes), the
    full-extent node and the reductions that loop under it. A ``Streaming`` record is
    derived for, cached by and honoured on exactly this."""
    payload = [signature(flat_graph, dtypes=True), full, list(loops), KEY_VERSION]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
