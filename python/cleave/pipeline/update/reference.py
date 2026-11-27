"""Test data: the flat graph's own loop states on prefixes of the looped dim.

A candidate update is tested against the program itself: the flat graph built again (by its
recipe, ``instantiate.Recipe``) at the first pass's own test sizes -- the extents the
superoptimizer fingerprints at -- with the looped dim at a few elements, evaluated in the
field of that fingerprint (``field``) on random inputs, and its state read off its nodes by
name.

    old state   F(H)        graph at |H|      on the inputs cut to H
    new state   F(H ++ T)   graph at |H|+|T|  on the inputs cut to H ++ T
    tile        the graph at |T| on the inputs cut to T, with the full-extent value given
                as the NEW one (the tile computation re-wired to read m')
    first iter  old state = the full-extent value's start value and zeros

for every draw of ``DRAWS`` and every prefix of ``STEPS``: every row of every case is a test.
Where the graph cannot be built or evaluated so, there is no test (``NO_TEST``) and no rule.

The same instances also run in floating point, in the program's dtype, for the ranking's
replay of the loop (``rank``): ``sample``, ``state``, ``start``, ``magnitude``.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dfield

import numpy as np

from ...ir.update_rule import signature
from ...runtime.reference import values
from . import field
from .texts import NO_TEST, Refused

TILE = 4  # elements of the looped dim per step of the test
STEPS = (0, 4, 8, 12)  # the prefixes; the longest instance has N elements
N = STEPS[-1] + TILE
DRAWS = (1, 2)  # random field inputs: each draw tests every row at every step
# Distinct extents of the one lowering that finds the looped dim when the provenance does
# not name it.
_PROBE = (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79)


def no_test(full: str, why: str) -> Refused:
    return Refused(NO_TEST.format(full=full, why=why[:300]))


def symbol_order(s):
    """Sort key of a symbol: structure proxies (ints) by value, dim names by name."""
    return (type(s).__name__, s)


def combine_of(node):
    """The reducer of a full-extent node as a torch function (``type(node).combine``)."""
    import torch

    return {"max": torch.maximum, "add": torch.add}[type(node).combine]


def folds_to(combined, whole, magnitude, count: int) -> bool:
    """Whether ``combined`` (the fold of a full-extent value over a prefix and over the next
    tile, as the kernel carries it) is ``whole`` (its value over both, ``count`` elements of
    the looped dim), in floating point: within what float32 arithmetic on the same elements
    in another order or another tiling can differ by, ``(count + 1) * 2**-23 * magnitude()``
    per element, ``magnitude()`` the sum of the absolute values of those elements. A value
    that is not a running reduction of the looped dim misses by far more."""
    bound = (count + 1) * 2.0 ** -23 * magnitude()
    return bool(((combined.float() - whole.float()).abs() <= bound).all())


def carry_init(flat_graph_node, full: str):
    """Where the carried value of ``full`` starts (``carry_init`` of its class; None: zero),
    or no test where its class declares no combine: such a value is not carried."""
    cls = type(flat_graph_node)
    if getattr(cls, "combine", None) is None:
        raise no_test(full, f"{full} is a {cls.__name__}, which declares no combine of its "
                      "value over a prefix and over the next tile: the loop cannot carry it")
    return cls.carry_init


def _node(graph, name):
    for n in graph.nodes:
        if n.name == name:
            return n
    raise KeyError(f"no node {name!r} in the re-instantiated flat graph")


def first_difference(a: list, b: list) -> str:
    """The first row in which two ``update_rule.signature`` lists differ, as text ("" when
    they are equal)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"node {i}: {x} instead of {y}"
    if len(a) != len(b):
        return f"{len(a)} nodes instead of {len(b)}"
    return ""


def probe_looped(recipe, full: str):
    """The symbol ``full`` reduces, found by lowering with a distinct extent per symbol
    (unit symbols stay 1): None when no single symbol has the extent its axis has."""
    free = [s for s in sorted(recipe.symbols, key=symbol_order) if recipe.sizes.get(s) != 1]
    if not free or len(free) >= len(_PROBE):
        return None
    probe = {s: _PROBE[i] for i, s in enumerate(free)}
    sizes = {**{s: 1 for s in recipe.symbols if recipe.sizes.get(s) == 1}, **probe}

    def extent(sizes):
        try:
            graph = recipe.at(sizes, "float32")
        except Exception:  # noqa: BLE001 - lowering refuses these extents
            return None
        return int(_node(graph, full).raxis[0].dom.extent)

    hits = [s for s in free if probe[s] == extent(sizes)]
    if len(hits) != 1:
        return None
    other = _PROBE[len(free)]  # an extent no symbol had
    return hits[0] if extent({**sizes, hits[0]: other}) == other else None


@dataclass(eq=False)
class Flat:
    """One flat graph and the loop a rule is derived for, at the first pass's test sizes.

    ``recipe`` builds the graph again; ``program_graph`` is the program's own (what every
    instance must be, ``signature``); ``full`` the full-extent node and ``accs`` the
    accumulators that loop under it; ``looped`` the symbol (or frozenset of symbols) of the
    looped dim. Every other symbol is at ``recipe.sizes``; ``dtype`` is the dtype the
    instances are lowered in (the program's own for the ranking's float replay)."""

    recipe: object
    program_graph: object
    full: str
    accs: tuple
    looped: object
    dtype: str = "float32"
    tile: int = TILE
    _graphs: dict = dfield(default_factory=dict, repr=False)
    _layout: list = dfield(default=None, repr=False)

    def __post_init__(self):
        self._looped = (set(self.looped) if isinstance(self.looped, frozenset)
                        else {self.looped})
        self._want = signature(self.program_graph, dtypes=self.dtype != "float32")
        self.sizes = {s: self.recipe.sizes[s] for s in self.recipe.symbols
                      if s not in self._looped}

    def graph(self, n: int):
        """The flat graph with the looped dim at ``n``: the program's graph, or no test."""
        n = int(n)
        if n not in self._graphs:
            sizes = {**self.sizes, **{s: n for s in self._looped}}
            try:
                graph = self.recipe.at(sizes, self.dtype)
            except Exception as e:  # noqa: BLE001
                raise no_test(self.full, f"it does not lower with {n} elements of that dim "
                              f"({type(e).__name__}: {e})") from e
            difference = first_difference(signature(graph, dtypes=self.dtype != "float32"),
                                          self._want)
            if difference:
                raise no_test(self.full, f"it is another graph with {n} elements of that "
                              f"dim ({difference})")
            self._graphs[n] = graph
        return self._graphs[n]

    def node(self, n: int, name: str):
        return _node(self.graph(n), name)

    def looped_layout(self) -> list:
        """Per placeholder (``placeholder_order``), ``(axis, groups)``: the axis that carries
        the looped dim (or None) and how many runs of it lie along that axis (an input read
        through a split view holds one run per split). Read off two instances that differ
        in nothing else; an input that grows along two axes has no prefix: no test."""
        if self._layout is None:
            from ...runtime.reference import placeholder_order

            a = placeholder_order(self.graph(self.tile))
            b = placeholder_order(self.graph(2 * self.tile))
            layout = []
            for pa, pb in zip(a, b):
                diff = [(i, int(y) - int(x)) for i, (x, y) in
                        enumerate(zip(pa.get_shape(0), pb.get_shape(0))) if int(x) != int(y)]
                if len(diff) > 1 or (diff and diff[0][1] % self.tile):
                    raise no_test(self.full, f"{pa.name} grows along {len(diff)} axes with "
                                  "that dim: a prefix of the loop is not a prefix of it")
                layout.append((None, 1) if not diff else (diff[0][0], diff[0][1] // self.tile))
            self._layout = layout
        return self._layout

    def cut(self, inputs, start: int, length: int) -> list:
        """``inputs`` restricted to ``[start, start + length)`` of the looped dim."""
        out = []
        for x, (ax, groups) in zip(inputs, self.looped_layout()):
            if ax is None:
                out.append(x)
            elif groups == 1:
                out.append(x.narrow(ax, start, length))
            else:  # ``groups`` runs of the looped dim along this axis: cut each of them
                shape = list(x.shape)
                run = shape[ax] // groups
                y = x.reshape(shape[:ax] + [groups, run] + shape[ax + 1:])
                y = y.narrow(ax + 1, start, length)
                out.append(y.reshape(shape[:ax] + [groups * length] + shape[ax + 1:]))
        return out

    def _shapes(self, n: int) -> list:
        from ...runtime.reference import placeholder_order

        return [tuple(int(d) for d in p.get_shape(0)) for p in placeholder_order(self.graph(n))]

    # ---------------------------------------------------------------- in the field

    def field_sample(self, n: int, seed: int) -> list:
        """Random field inputs of the graph at looped extent ``n``."""
        return field.random_inputs(self._shapes(n), seed)

    def field_state(self, inputs, start: int, length: int, given=None, given_zeros=None):
        """``(values, zero-divisor bits)``: the full-extent value and the accumulators on
        ``[start, start + length)``, in the field (``field.values``)."""
        try:
            env, _, zeros = field.values(self.graph(length), self.cut(inputs, start, length),
                                         given, given_zeros, with_zeros=True)
        except field.NoField as e:
            raise no_test(self.full, str(e)) from e
        names = (self.full, *self.accs)
        return {k: env[k] for k in names}, {k: zeros[k] for k in names}

    def field_start(self, like: dict) -> dict:
        """The state before the first iteration: the full-extent value's start value (its
        ``carry_init``, zero where it has none) and zeros."""
        import torch

        init = carry_init(self.node(self.tile, self.full), self.full)
        start = 0 if init is None else field.scalar(init)
        out = {self.full: torch.full_like(like[self.full], start)}
        for acc in self.accs:
            out[acc] = torch.zeros_like(like[acc])
        return out

    # ---------------------------------------------------------- in floating point

    def sample(self, n: int, seed: int, distribution: str = "normal") -> list:
        """Float inputs of the graph at looped extent ``n``, in the instance's dtype."""
        from ...runtime.reference import make_inputs

        inputs = make_inputs(self.graph(n), self.dtype, seed=seed, device="cpu",
                             distribution=distribution)
        if self.dtype == "float32":
            inputs = [x.float() if x.is_floating_point() else x for x in inputs]
        return list(inputs)

    def state(self, inputs, start: int, length: int, given=None) -> dict:
        """The full-extent value and the accumulators on ``[start, start + length)``
        (float32)."""
        env, _ = values(self.graph(length), self.cut(inputs, start, length), given)
        return {k: env[k].float() for k in (self.full, *self.accs)}

    def magnitude(self, inputs, start: int, length: int):
        """Per element of the full-extent value: the sum of the absolute values of what it
        reduces over ``[start, start + length)`` (float32; ``folds_to``)."""
        env, _ = values(self.graph(length), self.cut(inputs, start, length))
        node = self.node(length, self.full)
        operand = env[node.inputs[0].src_node.name].float().abs()
        return operand.sum(dim=node.axis).reshape(env[self.full].shape)

    def start(self, like: dict) -> dict:
        """The float state before the first iteration: the start value and zeros."""
        import torch

        init = carry_init(self.node(self.tile, self.full), self.full)
        out = {self.full: torch.full_like(like[self.full], 0.0 if init is None else init)}
        for acc in self.accs:
            out[acc] = torch.zeros_like(like[acc])
        return out


@dataclass(eq=False)
class Case:
    """One step ``F(H) -> F(H ++ T)``: everything a candidate update may read, and what it
    must give (field values), and where each of them is computed from a division by zero
    (``zeros``: per part, ``field.zero_bits``)."""

    label: str
    prefix: int
    old: dict  # F(H): full-extent value and accumulators (the start state for H empty)
    new: dict  # F(H ++ T)
    tile: dict  # the tile's results, computed with the NEW full-extent value
    full: str = ""
    zeros: dict = None  # part -> {name: bits}

    @property
    def rows(self) -> int:
        return int(self.old[self.full].numel())

    def view(self, part: str, name: str) -> np.ndarray:
        """``part`` (old / new / tile) of ``name`` as ``(rows, width)`` int64."""
        return getattr(self, part)[name].reshape(self.rows, -1).numpy()

    def zeros_view(self, part: str, name: str) -> np.ndarray:
        """The zero-divisor bits of :meth:`view`."""
        return self.zeros[part][name].reshape(self.rows, -1).numpy()


def extension(flat: Flat, inputs, prefix: int, label: str) -> Case:
    """The case ``F(H) -> F(H ++ T)`` for ``|H| = prefix`` and the next tile ``T``."""
    t = flat.tile
    new, new_zeros = flat.field_state(inputs, 0, prefix + t)
    own, _ = flat.field_state(inputs, prefix, t)  # the tile alone: its own shapes
    for name in (flat.full, *flat.accs):
        shapes = {tuple(new[name].shape), tuple(own[name].shape)}
        if len(shapes) != 1:
            raise no_test(flat.full, f"{name} changes shape with the looped dim "
                          f"({sorted(shapes)}): the dim sizes more than the loop")
    if prefix:
        old, old_zeros = flat.field_state(inputs, 0, prefix)
    else:
        old = flat.field_start(new)
        old_zeros = {k: v.new_zeros(v.shape) for k, v in old.items()}
    tile, tile_zeros = flat.field_state(inputs, prefix, t, given={flat.full: new[flat.full]},
                                        given_zeros={flat.full: new_zeros[flat.full]})
    zeros = {"old": old_zeros, "new": new_zeros, "tile": tile_zeros}
    return Case(label, prefix, old, new, tile, flat.full, zeros)


def reference_set(flat: Flat) -> list:
    """Every case: per draw of ``DRAWS``, every prefix of ``STEPS``."""
    cases = []
    for seed in DRAWS:
        inputs = flat.field_sample(N, seed)
        cases += [extension(flat, inputs, p, f"draw {seed} prefix {p}") for p in STEPS]
    return cases


def block(cases, part: str, name: str) -> np.ndarray:
    """One int64 block ``(rows of every case, width)``: ``part`` of ``name`` in every case."""
    return np.concatenate([c.view(part, name) for c in cases], axis=0)


def zeros_block(cases, part: str, name: str) -> np.ndarray:
    """The zero-divisor bits of :func:`block`."""
    return np.concatenate([c.zeros_view(part, name) for c in cases], axis=0)
