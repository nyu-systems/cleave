"""Why a reduction conflict gets no update rule: the three outcomes of the search, and a bug.

A refusal is an exception whose text is one of these, filled in. ``Derivation.refused``
holds it, ``pipeline.plans`` puts it on the graph as ``"{full}: {text}"``
(``ir.streaming_refusals``), and the no-tiling report quotes it. Each one only keeps the
conflict's pin: the dim is held whole, as it is for any conflict without a rule.
"""

from __future__ import annotations


class Refused(Exception):
    """No update rule is derived for this conflict; ``str(e)`` says why."""


# The flat graph could not be built again on prefixes of the looped dim, or evaluated
# there: without the program's own states there is nothing to test a candidate against.
NO_TEST = "{full} has no test on prefixes of the dim it reduces: {why}"
# No enumerated update reproduced the program's states.
NO_RULE = ("no update of {acc} of the form 'accumulator * factor + tile' within {budget} ops "
           "reproduces the flat graph on its prefixes")
# Every update that reproduced them fails the float replay of the loop.
NO_REPLAY = "every update of {loops} that reproduces the flat graph fails the float replay ({case})"
BUG = "{type}: {message} (a bug in the update search)"


def bug(error: BaseException) -> str:
    """The refusal of an unexpected exception: it keeps the pin and names itself."""
    message = str(error).splitlines()[0] if str(error) else ""
    return BUG.format(type=type(error).__name__, message=message)
