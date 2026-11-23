"""Prefix references: the flat graph itself, re-instantiated at small shapes.

``F(prefix)`` is the state the flat graph computes when it is evaluated on a prefix of the
looped dim. Nothing here knows what an online softmax is: the flat graph is lowered again
with a small ``symbol_map`` (``rebuild``), evaluated node by node with the validation
reference (``cleave.runtime.reference.values``), and the state is read off its nodes by
name. Float32 throughout (or the program's own dtype, for ranking); no FP64.

    old state   F(H)        graph at |H|      on the inputs sliced to H
    new state   F(H ++ T)   graph at |H|+|T|  on the inputs sliced to H ++ T
    tile        the graph at |T| on the inputs sliced to T, with the full-extent value
                given as the NEW one (the tile computation re-wired to read m')
    first iter  old state = the start values (sentinel, zeros); new state F(T)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...ir.ops import NEG_FLT_MAX
from ...runtime.reference import values


class NotStreamable(Exception):
    """The references say this loop has no update rule to search for; the text is the
    reason the obstacle keeps its pin."""


# Extents for the dims that are not looped (none is 1: unit dims lower differently), and the
# distinct ones of the single lowering that finds the looped dim.
_SMALL = (2, 3)
_PROBE = (5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


@dataclass
class Flat:
    """One flat graph and the loop a rule is derived for, at small shapes.

    ``rebuild(symbol_map, dtype)`` lowers the same searched structure with other extents.
    ``symbols`` is its own ``symbol_map``; ``full`` the full-extent node, ``accs`` the
    accumulators that loop under it.
    """

    rebuild: object
    symbols: dict
    full: str
    accs: tuple
    dtype: str = "float32"
    tile: int = 4
    _graphs: dict = field(default_factory=dict)
    looped: object = None  # the symbol of the looped dim (found, not given)
    small: dict = None

    def __post_init__(self):
        syms = sorted(self.symbols)
        if self.looped is None:
            # Which dim does the full-extent node reduce? Lower once with a distinct extent
            # per dim and read the extent of its reduction axis. (A caller that knows the
            # dim -- from the reduction-axis provenance of the flat graph -- passes it.)
            assert len(syms) <= len(_PROBE), "more dims than probe extents"
            probe = {s: _PROBE[i] for i, s in enumerate(syms)}
            graph = self.rebuild(probe, self.dtype)
            extent = int(self.node(graph, self.full).raxis[0].dom.extent)
            hits = [s for s in syms if probe[s] == extent]
            assert len(hits) == 1, f"cannot tell which dim {self.full} reduces: {hits}"
            (self.looped,) = hits
        # Every other dim: 2 or 3 elements (alternating), so a state tensor has a handful
        # of rows and columns and a search level holds a few floats per candidate. Dims
        # that must divide one another (grouped batch dims: 32 query heads over 8 kv heads)
        # do not lower that way; then every dim gets the same extent.
        rest = [s for s in syms if s != self.looped]
        errors = []
        for sizing in ({s: _SMALL[i % len(_SMALL)] for i, s in enumerate(rest)},
                       {s: 2 for s in rest}, {s: 3 for s in rest}):
            try:
                self.rebuild({**sizing, self.looped: self.tile}, self.dtype)
            except Exception as e:  # noqa: BLE001 - lowering refuses these extents
                errors.append(f"{type(e).__name__}: {e}")
                continue
            self.small = sizing
            return
        raise NotStreamable("the flat graph does not lower at small shapes: " + errors[-1][:200])

    @staticmethod
    def node(graph, name):
        return next(n for n in graph.nodes if n.name == name)

    def graph(self, n: int):
        """The flat graph with the looped dim at ``n`` and every other dim small."""
        if n not in self._graphs:
            g = self.rebuild({**self.small, self.looped: int(n)}, self.dtype)
            self._graphs[n] = g
        return self._graphs[n]

    def structure(self, graph) -> tuple:
        return tuple((type(n).__name__, n.name) for n in graph.nodes)

    def looped_axes(self) -> list:
        """Per placeholder (in ``placeholder_order``), the axis that is the looped dim, or
        None: read off two instantiations that differ in nothing else."""
        from ...runtime.reference import placeholder_order

        return [ax for ax, _ in self.looped_layout()]

    def looped_layout(self) -> list:
        """Per placeholder, ``(axis, groups)``: the axis that carries the looped dim (or
        None) and how many times: an input that the graph reads through a split view
        holds ``groups`` consecutive runs of the looped dim along that axis (one per
        split), so its extent grows by ``groups`` for every element of the looped dim."""
        from ...runtime.reference import placeholder_order

        a = placeholder_order(self.graph(self.tile))
        b = placeholder_order(self.graph(2 * self.tile))
        assert [p.name for p in a] == [p.name for p in b]
        layout = []
        for pa, pb in zip(a, b):
            diff = [(i, int(y) - int(x)) for i, (x, y) in
                    enumerate(zip(pa.get_shape(0), pb.get_shape(0))) if int(x) != int(y)]
            assert len(diff) <= 1, f"{pa.name}: the looped dim is on axes {diff}"
            if not diff:
                layout.append((None, 1))
                continue
            axis, grown = diff[0]
            assert grown % self.tile == 0
            layout.append((axis, grown // self.tile))
        return layout

    def sample(self, n: int, seed: int, distribution: str = "normal") -> list:
        """Inputs of the graph at looped extent ``n`` (float32 or the program dtype)."""
        import torch

        from ...runtime.reference import make_inputs

        inputs = make_inputs(self.graph(n), self.dtype, seed=seed, device="cpu",
                             distribution=distribution)
        if self.dtype == "float32":  # the structure may declare its inputs narrower
            inputs = [x.float() if x.is_floating_point() else x for x in inputs]
        return list(inputs)

    def cut(self, inputs, start: int, length: int) -> list:
        """``inputs`` restricted to ``[start, start + length)`` of the looped dim. Whether
        the restriction is the right one is not assumed: ``extension`` checks that the
        full-extent value of prefix and tile combine to that of their concatenation."""
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

    def state(self, inputs, start: int, length: int, given=None):
        """The flat graph on the slice ``[start, start + length)`` of the looped dim:
        ``({name: value} for the full-extent node and the accumulators, outputs)``."""
        env, outs = values(self.graph(length), self.cut(inputs, start, length), given)
        return {k: env[k].float() for k in (self.full, *self.accs)}, outs

    def start(self, like: dict) -> dict:
        """The state before the first iteration: the sentinel and zeros."""
        import torch

        out = {self.full: torch.full_like(like[self.full], NEG_FLT_MAX)}
        for acc in self.accs:
            out[acc] = torch.zeros_like(like[acc])
        return out


@dataclass
class Case:
    """One prefix extension: everything a candidate update may read, and what it must give."""

    label: str
    old: dict  # F(H): full-extent value and accumulators
    new: dict  # F(H ++ T)
    tile: dict  # the tile's finished results, computed with the NEW full-extent value
    moved: float  # share of rows whose full-extent value moved


def extension(flat: Flat, inputs, prefix: int, label: str) -> Case:
    import torch

    t = flat.tile
    new, _ = flat.state(inputs, 0, prefix + t)
    own, _ = flat.state(inputs, prefix, t)  # the tile on its own: only its max is read
    if prefix:
        old, _ = flat.state(inputs, 0, prefix)
    else:
        old = flat.start(new)
    # The hole of the full-extent value is the reducer's own associativity, and it is exact.
    # Where it does not hold the value is not a running reduction of the looped dim at all
    # (its operand depends on the prefix: a max of ``exp(x - max(x))``), or the inputs were
    # not cut along the looped dim: no rule is derived.
    if not torch.equal(torch.maximum(old[flat.full], own[flat.full]), new[flat.full]):
        raise NotStreamable(
            f"the value of {flat.full} over a prefix and over the next tile do not combine "
            f"to its value over both ({label})")
    tile, _ = flat.state(inputs, prefix, t, given={flat.full: new[flat.full]})
    moved = float((new[flat.full] > old[flat.full]).float().mean())
    return Case(label, old, new, tile, moved)


def moving_inputs(flat: Flat, n: int, tries: int = 400):
    """Benign inputs on which the full-extent value moves in EVERY row at EVERY prefix
    extension, by a moderate amount (so that neither the old sum nor the tile's is
    negligible next to the other): the inputs that carry the looped dim are ramped along
    it, and the first seed with the property is taken."""
    import torch

    layout = flat.looped_layout()
    for seed in range(tries):
        for gain in (1.0, 2.0, 4.0, 8.0, 16.0):
            inputs = flat.sample(n, seed)
            ramped = []
            for x, (ax, groups) in zip(inputs, layout):
                level = 1.0 + 0.25 * torch.tanh(x.float())  # positive, within 25% of 1
                if ax is None:
                    ramped.append((gain * level).to(x.dtype))
                    continue
                shape = [1] * x.dim()
                shape[ax] = groups * n
                ramp = (1.0 + 1.5 * torch.arange(n, dtype=torch.float32) / n).repeat(groups)
                ramped.append((level * ramp.reshape(shape)).to(x.dtype))
            cases = [extension(flat, ramped, p, f"moving seed={seed} gain={gain:g} prefix={p}")
                     for p in range(flat.tile, n, flat.tile)]
            moves = torch.cat([(c.new[flat.full] - c.old[flat.full]).flatten() for c in cases])
            # every row moves, by an amount that leaves both terms of an update visible
            if float(moves.min()) > 0.2 and float(moves.max()) < 3.0:
                return seed, ramped, cases
    raise NotStreamable(
        f"no benign inputs found on which {flat.full} moves in every row at every prefix "
        f"extension by a moderate amount ({tries} seeds)")
