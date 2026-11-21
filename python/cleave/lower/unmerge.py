"""Give the exps of a searched structure back to the readers the program wrote them for.

The search works on values: two ``exp`` ops of one tensor are one value to it, and the
graphs it returns compute that value once. ``cl.softmax(x)`` records a plain ``exp(x)``
plus a request, so next to another ``exp`` of the same tensor -- ``cl.softmax(x, -1) +
cl.exp(x)``, or a second softmax of ``x`` over another axis -- the searched graph has one
exp op where the program wrote two, read by the softmax and by everything else. Its
factor does not cancel (the scale analysis sees ``c * exp(x)`` reach an output with degree
1, or a factor that would have to be constant along two axes), so the graph as searched
cannot be stabilized, although the program can.

``unmerged`` enumerates the structures that compute the merged exp once more for a part
of its readers. They are the same function of the inputs, op for op; the scale analysis
then decides, as for any graph, which exp is a softmax exp, and the lowering takes the
first structure it accepts (``cleave.pipeline.plans``).
"""

from __future__ import annotations

import copy
import itertools

EXP = "kn_exp_op"
LIMIT = 256  # structures tried per searched graph


def exp_ops(structure) -> int:
    return sum(op["op_type"] == EXP for op in structure)


def _guids(structure):
    return [t["guid"] for op in structure for t in op["input_tensors"] + op["output_tensors"]]


def _twins(structure):
    """Every structure that computes one exp of ``structure`` a second time, for a proper
    non-empty part of its readers.

    A reader is an (op, input slot) pair: ``e * e`` reads its exp twice, and each slot can
    go to either copy. The first reader stays with the first copy, so each way of dividing
    the readers in two is seen once. The second copy stands right behind the first.
    """
    fresh = max(_guids(structure)) + 1
    for k, op in enumerate(structure):
        if op["op_type"] != EXP:
            continue
        out = op["output_tensors"][0]["guid"]
        readers = [(i, j) for i, reader in enumerate(structure)
                   for j, t in enumerate(reader["input_tensors"]) if t["guid"] == out]
        for size in range(1, len(readers)):
            for group in itertools.combinations(readers[1:], size):
                twin = copy.deepcopy(structure)
                second = copy.deepcopy(twin[k])
                second["output_tensors"][0]["guid"] = fresh
                for i, j in group:
                    twin[i]["input_tensors"][j]["guid"] = fresh
                twin.insert(k + 1, second)
                yield twin


def unmerged(structure, extra: int):
    """The structures with ``extra`` more exps than ``structure``, each computed for a part
    of the readers of an exp it has; at most ``LIMIT`` of them, in a fixed order."""
    found = [structure]
    for _ in range(extra):
        found = list(itertools.islice(
            (twin for s in found for twin in _twins(s)), LIMIT))
    return found if extra > 0 else []
