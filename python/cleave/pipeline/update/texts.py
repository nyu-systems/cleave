"""The engine's typed refusals: why an obstacle keeps its pin.

A refusal is an exception whose text is one of these, filled in. ``Derivation.refused``
holds it, ``pipeline.plans`` puts it on the graph as ``"{full}: {text}"``
(``ir.streaming_refusals``), and the no-tiling report quotes it. Tests pin the texts.
"""

from __future__ import annotations


class Refused(Exception):
    """No update rule is derived for this obstacle; ``str(e)`` says why."""


E1 = ("the looped dim of {full} has no symbol: the graph cannot be built again on a prefix "
      "of it")
E2 = "the flat graph does not lower at small shapes: {error}"
E3 = "the flat graph is not the same graph at small shapes: {difference}"
E4 = ("{input} carries the looped dim on {k} axes: a prefix of the loop is not a prefix of "
      "it")
E5 = ("the value of {full} over a prefix and over the next tile do not combine to its value "
      "over both ({label})")
E6 = ("{acc} of one row changes with {full} of another row: the rows of {full} do not lead "
      "its accumulators")
E7 = ("{node} ({cls}) lies between {full} and {acc}: only elementwise add, sub, mul, div and "
      "exp are read there")
E8 = "{node} has {attr}={value}, which the proof does not model"
E9 = "{node} combines two values computed from {full} that are not indexed alike"
E10 = ("the references do not show the move of {full} and the old value of {acc} in the same "
       "rows ({k} rows, {n} needed)")
E11 = ("no proven update of {acc} of the form 'accumulator * factor + tile' within {budget} "
       "ops ({matched} matched the references, {proved} proved)")
E12 = "every proven update of {loops} is non-finite on an input case ({case})"
E13 = "{type}: {message} (a bug in the update search)"
# Beyond E1-E13: what the proof takes as one free real per element must be the
# same on every prefix of the loop.
E14 = ("{node}, which {acc} reads, is computed from {other}, another reduction over the dim "
       "{full} reduces: it changes from one prefix of the loop to the next")
# Beyond E1-E13: the looped dim's symbol must size nothing else of the state (a
# self-attention whose queries and keys are both written ``S``: a prefix of the keys would
# cut the queries too).
E15 = ("{node} is {a} at {n} elements of the looped dim of {full} and {b} at {m}: its symbol "
       "{symbol} also sizes a dim the loop does not step")
# The structural rule S5 of ``schedule.streaming.obstacles`` (its text T5), checked again
# here: the contribution of one element is only read off a plain sum or matmul.
T5 = ("{d} is a {cls}: the reductions that loop under {full} must be sums or matmuls that "
      "start at zero")


def bug(error: BaseException) -> str:
    """E13 for an unexpected exception: it keeps the pin and names itself."""
    message = str(error).splitlines()[0] if str(error) else ""
    return E13.format(type=type(error).__name__, message=message)
