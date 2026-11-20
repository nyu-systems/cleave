"""The exact-softmax request for tests, until ``cl.softmax`` records it itself.

``cl.softmax(x, dim)`` still traces ``exp(x - max(x)) / sum``; the stabilize request (a
plain ``exp`` record tagged with the softmax axis) is reached through ``Program._stable_exp``.
The switch replaces the body of :func:`exact_softmax` with ``cl.softmax(x, dim)``.
"""

import cleave.lang as cl


def stable_exp(x, dim: int = -1):
    """The ``exp`` of a softmax over ``dim``: a plain exp record carrying the request."""
    return x.program._stable_exp(x, dim)


def exact_softmax(x, dim: int = -1):
    e = stable_exp(x, dim)
    return e / cl.sum(e, dim, keepdim=True)
