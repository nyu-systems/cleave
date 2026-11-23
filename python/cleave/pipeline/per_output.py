"""Per-output search and merge (``SuperoptConfig(per_output=...)``).

A joint search over several outputs grows with the product of their search spaces; one output at
a time it stays small. Each output gets its own program (``Program.for_output``: that output, the
inputs it reads, the ``equals=`` inputs whose sources are all kept) and goes through the normal
``superoptimize`` (its own cache entry and auto ``max_ops``, no split). One searched graph per
output is then merged into one multi-output structure over ALL the parent's inputs, in order (an
input no merged graph reads stays an input: ``bind_inputs`` binds it to None), with common
sub-expressions eliminated: an op of the same kind, with the same attributes, over the same
operands (commutative ``add``/``mul`` operands compared as a set) and with the same result
shape is computed once. Each part is a verified graph of its output, and merging only shares
values that are equal by construction, so every merged graph computes every output.

Combinations are enumerated in order of increasing index sum (the parts' first graphs first;
lexicographic within one sum), deduplicated after CSE, and capped at ``MAX_MERGED`` distinct
merged graphs: the cartesian product of a few graphs per output grows fast and the scheduler
tiles every graph it is given.
"""

from __future__ import annotations

import json
from typing import Iterator, Sequence

from ..core.errors import SuperoptError

MAX_MERGED = 64  # distinct merged graphs kept per program
_COMMUTATIVE = ("kn_add_op", "kn_mul_op")
_TENSORS = ("input_tensors", "output_tensors")


def combinations(counts: Sequence[int]) -> Iterator[tuple[int, ...]]:
    """Every index tuple (``0 <= t[i] < counts[i]``), by increasing sum, then lexicographic."""
    if not counts or min(counts) < 1:
        return
    for total in range(sum(c - 1 for c in counts) + 1):
        yield from _with_sum(tuple(counts), total)


def _with_sum(counts, total):
    if len(counts) == 1:
        if total < counts[0]:
            yield (total,)
        return
    rest = sum(c - 1 for c in counts[1:])
    for first in range(max(0, total - rest), min(counts[0] - 1, total) + 1):
        for tail in _with_sum(counts[1:], total - first):
            yield (first,) + tail


def symbol_translation(part_inputs: list[dict], parent_inputs: list[dict], part_symbols: dict,
                       kept: Sequence[int]) -> dict:
    """Map a part's dim symbols to the parent's, through the inputs they share.

    ``part_inputs[j]`` is parent input ``kept[j]``; both are ``kn_input_op`` output tensors.
    """
    table = {}
    for j, k in enumerate(kept):
        mine, theirs = part_inputs[j]["dim"], parent_inputs[k]["dim"]
        if len(mine) != len(theirs):
            raise SuperoptError(f"per-output merge: input {k} has rank {len(mine)} in a part, "
                                f"{len(theirs)} in the program")
        for a, b in zip(mine, theirs):
            if a in part_symbols and table.setdefault(a, b) != b:
                raise SuperoptError(f"per-output merge: symbol {a} maps to {table[a]} and {b}")
    return table


def merge(parts: Sequence[list[dict]], kepts: Sequence[Sequence[int]], tables: Sequence[dict],
          parent_inputs: list[dict]) -> list[dict]:
    """One structure computing every part's output(s), in part order, over ``parent_inputs``.

    ``parts[i]``: a one-output ``get_graph_structure()`` list whose ``kn_input_op`` j is parent
    input ``kepts[i][j]``; ``tables[i]``: its symbol translation. Guids are fresh but NOT
    renumbered (the caller renumbers, as for any searched graph).
    """
    ops = [{"op_type": "kn_input_op", "input_tensors": [], "output_tensors": [dict(t)]}
           for t in parent_inputs]
    tensors = {("input", k): dict(t) for k, t in enumerate(parent_inputs)}
    seen: dict = {}  # CSE key -> the merged op's output keys
    next_guid = 1 + max((t["guid"] for t in parent_inputs), default=0)
    outputs = []
    for part, kept, table in zip(parts, kepts, tables):
        local = {}  # the part's guid -> merged tensor key
        n_inputs = 0
        for op in part:
            kind = op["op_type"]
            if kind == "kn_input_op":
                local[op["output_tensors"][0]["guid"]] = ("input", kept[n_inputs])
                n_inputs += 1
                continue
            operands = [local[t["guid"]] for t in op["input_tensors"]]
            if kind == "kn_output_op":
                outputs.extend(operands)
                continue
            attrs = {k: v for k, v in op.items() if k not in _TENSORS and k != "op_type"}
            shapes = [[table.get(d, d) for d in t["dim"]] for t in op["output_tensors"]]
            key_operands = (sorted(operands) if kind in _COMMUTATIVE and len(operands) == 2
                            and "scalar" not in attrs else operands)
            key = json.dumps([kind, attrs, key_operands, shapes], sort_keys=True)
            if key not in seen:
                outs = []
                for t, dims in zip(op["output_tensors"], shapes):
                    t = {**t, "dim": dims, "guid": next_guid}
                    next_guid += 1
                    outs.append(("op", t["guid"]))
                    tensors[outs[-1]] = t
                ops.append({**op, "input_tensors": [dict(tensors[o]) for o in operands],
                            "output_tensors": [dict(tensors[o]) for o in outs]})
                seen[key] = outs
            for t, o in zip(op["output_tensors"], seen[key]):
                local[t["guid"]] = o
    for o in outputs:
        ops.append({"op_type": "kn_output_op", "input_tensors": [dict(tensors[o])],
                    "output_tensors": []})
    known = {d for t in parent_inputs for d in t["dim"]} | {1}
    unknown = sorted({d for op in ops for t in op["output_tensors"] for d in t["dim"]} - known)
    if unknown:  # a dim no input has: the translation through the inputs cannot place it
        raise SuperoptError(f"per-output merge: dims {unknown} are not dims of any input")
    return ops
