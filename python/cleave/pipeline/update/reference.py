"""Prefix references: the flat graph itself, re-instantiated at small shapes (DESIGN 5).

``F(prefix)`` is the state the flat graph computes when it is evaluated on a prefix of the
looped dim. Nothing here knows what an online softmax is: the flat graph is built again by
its recipe (``instantiate.Recipe``) with the looped dim at a few elements and every other
dim small, evaluated node by node with the validation reference
(``cleave.runtime.reference.values``), and the state is read off its nodes by name.
Float32 throughout (or the program's own dtype, for ranking); no FP64.

    old state   F(H)        graph at |H|      on the inputs cut to H
    new state   F(H ++ T)   graph at |H|+|T|  on the inputs cut to H ++ T
    tile        the graph at |T| on the inputs cut to T, with the full-extent value
                given as the NEW one (the tile computation re-wired to read m')
    first iter  old state = the start values (``carry_init``, zeros); new state F(T)

Every guard is a typed refusal (``texts``): the graph must lower at small shapes (E2) and be
the same graph there (E3), the looped symbol must size the looped dim (E1) and nothing else
of the state (E15), the inputs must carry the looped dim on one axis (E4), the cut must be
right (E5: the full-extent value is its reducer's own fold, bit for bit), the rows of the
full-extent value must lead its accumulators (E6), and the references must show what an
update has to repair (E10).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...ir.update_rule import signature
from ...runtime.reference import values
from .search import LOOSE_RTOL
from .texts import E1, E2, E3, E4, E5, E6, E10, E15, Refused

TILE = 4  # elements of the looped dim per iteration of the references
N = 16  # looped extent of a reference sample: tile 4, prefixes 4 / 8 / 12
PREFIXES = (4, 8, 12)
SEEDS_NORMAL = (1, 2)
TRY_SEEDS = tuple(range(16))
SHAPINGS = (("none", 1.0), ("ramp", 1.0), ("ramp", 4.0), ("ramp", 16.0), ("tail", 2.0),
            ("tail", 4.0))
# Distinct extents of the one lowering that finds the looped dim when the provenance does
# not name it.
_PROBE = (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79)


def symbol_order(s):
    """Sort key of a symbol: structure proxies (ints) by value, dim names by name."""
    return (type(s).__name__, s)


def combine_of(node):
    """The reducer of a full-extent node as a torch function (``type(node).combine``)."""
    import torch

    return {"max": torch.maximum}[type(node).combine]


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
    (unit symbols stay 1): None when no symbol has the extent its axis has. A hit is
    confirmed by one more lowering with another extent for that symbol: a literal dim whose
    extent happens to be the probe's is no symbol of it."""
    free = [s for s in sorted(recipe.symbols, key=symbol_order) if recipe.symbols[s] != 1]
    if not free or len(free) >= len(_PROBE):
        return None
    probe = {s: _PROBE[i] for i, s in enumerate(free)}
    sizes = {**{s: 1 for s in recipe.symbols if recipe.symbols[s] == 1}, **probe}

    def extent(sizes):
        try:
            graph = recipe.at(sizes, "float32")
        except Exception as e:  # noqa: BLE001 - lowering refuses these extents
            raise Refused(E2.format(error=f"{type(e).__name__}: {e}"[:200])) from e
        return int(_node(graph, full).raxis[0].dom.extent)

    hits = [s for s in free if probe[s] == extent(sizes)]
    if len(hits) != 1:
        return None
    other = _PROBE[len(free)]  # an extent no symbol had
    return hits[0] if extent({**sizes, hits[0]: other}) == other else None


@dataclass(eq=False)
class Flat:
    """One flat graph and the loop a rule is derived for, at small shapes.

    ``recipe`` builds the graph again (``instantiate.Recipe``); ``program_graph`` is the
    program's own (what every instance must equal, ``signature``); ``full`` the full-extent
    node and ``accs`` the accumulators that loop under it; ``looped`` the symbol (or
    frozenset of symbols) of the looped dim. ``small`` is the extent of every other symbol;
    without it, the first sizing that lowers and passes the guard is chosen.
    """

    recipe: object
    program_graph: object
    full: str
    accs: tuple
    looped: object
    dtype: str = "float32"
    tile: int = TILE
    small: dict = None
    _graphs: dict = field(default_factory=dict, repr=False)
    _layout: list = field(default=None, repr=False)

    def __post_init__(self):
        self._looped = (set(self.looped) if isinstance(self.looped, frozenset)
                        else {self.looped})
        self._want = signature(self.program_graph, dtypes=self.strict)
        if self.small is None:
            self.small = self._choose_sizes()

    @property
    def strict(self) -> bool:
        """The guard compares dtypes too for an instance in the program's own dtype."""
        return self.dtype != "float32"

    def _sizings(self):
        symbols = self.recipe.symbols
        rest = [s for s in sorted(symbols, key=symbol_order) if s not in self._looped]
        units = {s: 1 for s in rest if symbols[s] == 1}
        free = [s for s in rest if symbols[s] != 1]
        real = sorted({int(symbols[s]) for s in free if symbols[s] is not None})
        power = {e: min(e, 2 ** (i + 1)) for i, e in enumerate(real)}
        top = 2 ** (len(real) + 1)
        ladder = [
            {s: (2, 3)[i % 2] for i, s in enumerate(free)},  # alternating 2 / 3
            {s: 2 for s in free},
            {s: 3 for s in free},  # grouped batch dims: equal extents divide one another
            {s: power[int(symbols[s])] if symbols[s] is not None else top for s in free},
        ]
        seen = []
        for sizing in ladder:
            sizing = {**units, **sizing}
            if sizing not in seen:
                seen.append(sizing)
                yield sizing

    def _choose_sizes(self) -> dict:
        errors, differences = [], []
        for sizing in self._sizings():
            sizes = {**sizing, **{s: self.tile for s in self._looped}}
            try:
                graph = self.recipe.at(sizes, self.dtype)
            except Exception as e:  # noqa: BLE001 - lowering refuses these extents
                errors.append(f"{type(e).__name__}: {e}")
                continue
            difference = first_difference(signature(graph, dtypes=self.strict), self._want)
            if difference:
                differences.append(difference)
                continue
            self._graphs[self.tile] = graph
            return sizing
        if differences:
            raise Refused(E3.format(difference=differences[0][:300]))
        raise Refused(E2.format(error=(errors[-1] if errors else "no sizing")[:200]))

    def graph(self, n: int):
        """The flat graph with the looped dim at ``n`` and every other dim small; the same
        graph as the program's, or ``E3``."""
        n = int(n)
        if n not in self._graphs:
            sizes = {**self.small, **{s: n for s in self._looped}}
            try:
                graph = self.recipe.at(sizes, self.dtype)
            except Exception as e:  # noqa: BLE001
                raise Refused(E2.format(error=f"{type(e).__name__}: {e}"[:200])) from e
            difference = first_difference(signature(graph, dtypes=self.strict), self._want)
            if difference:
                raise Refused(E3.format(difference=difference[:300]))
            self._graphs[n] = graph
        return self._graphs[n]

    def node(self, n: int, name: str):
        return _node(self.graph(n), name)

    def check_loop(self) -> None:
        """The looped symbol sizes the looped dim and nothing else of the state: at ``n``
        elements ``full`` reduces ``n`` of them (else E1: the symbol is not its dim's), every
        input carries it on one axis at most (E4, ``looped_layout``), and ``full`` and every
        accumulator have the same shape at ``tile`` and ``2 * tile`` elements (else E15:
        the symbol also names a dim the loop does not step, such as the queries' axis of a
        self-attention written ``q, k: [.., S, ..]``)."""
        t = self.tile
        for n in (t, 2 * t):
            if int(self.node(n, self.full).raxis[0].dom.extent) != n:
                raise Refused(E1.format(full=self.full))
        self.looped_layout()
        symbol = ", ".join(sorted(map(str, self._looped)))
        for name in (self.full, *self.accs):
            a = [int(x) for x in self.node(t, name).get_shape(0)]
            b = [int(x) for x in self.node(2 * t, name).get_shape(0)]
            if a != b:
                raise Refused(E15.format(node=name, a=a, n=t, full=self.full, b=b, m=2 * t,
                                         symbol=symbol))

    def looped_layout(self) -> list:
        """Per placeholder (``placeholder_order``), ``(axis, groups)``: the axis that carries
        the looped dim (or None) and how many runs of it lie along that axis (an input read
        through a split view holds one run per split: it grows by ``groups`` per element).
        Read off two instances that differ in nothing else; more than one axis is E4."""
        if self._layout is None:
            from ...runtime.reference import placeholder_order

            a = placeholder_order(self.graph(self.tile))
            b = placeholder_order(self.graph(2 * self.tile))
            layout = []
            for pa, pb in zip(a, b):
                diff = [(i, int(y) - int(x)) for i, (x, y) in
                        enumerate(zip(pa.get_shape(0), pb.get_shape(0))) if int(x) != int(y)]
                if len(diff) > 1 or (diff and diff[0][1] % self.tile):
                    raise Refused(E4.format(input=pa.name, k=len(diff)))
                layout.append((None, 1) if not diff else (diff[0][0], diff[0][1] // self.tile))
            self._layout = layout
        return self._layout

    def sample(self, n: int, seed: int, distribution: str = "normal") -> list:
        """Inputs of the graph at looped extent ``n``, in the instance's dtype (the float32
        instance reads float32 values of what the program declares)."""
        from ...runtime.reference import make_inputs

        inputs = make_inputs(self.graph(n), self.dtype, seed=seed, device="cpu",
                             distribution=distribution)
        if self.dtype == "float32":
            inputs = [x.float() if x.is_floating_point() else x for x in inputs]
        return list(inputs)

    def stored(self, inputs) -> list:
        """``inputs`` as the program's inputs hold them: rounded to each placeholder's
        dtype (then float32 again for the float32 instance). A shaped sample is computed in
        float32; an input the program declares float16 never holds a value that float16
        cannot, or the graph's own nodes would read it at two precisions."""
        from ...runtime.reference import _placeholder_torch_dtype, placeholder_order, torch_dtype

        out = []
        for x, p in zip(inputs, placeholder_order(self.graph(self.tile))):
            if x.is_floating_point():
                x = x.to(_placeholder_torch_dtype(p, torch_dtype(self.dtype)))
                x = x.float() if self.dtype == "float32" else x
            out.append(x)
        return out

    def cut(self, inputs, start: int, length: int) -> list:
        """``inputs`` restricted to ``[start, start + length)`` of the looped dim. Whether
        the restriction is the right one is not assumed: ``extension`` checks that the
        full-extent values of prefix and tile fold to that of their concatenation."""
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

    def state(self, inputs, start: int, length: int, given=None) -> dict:
        """The full-extent value and the accumulators of the flat graph on the slice
        ``[start, start + length)`` of the looped dim (float32)."""
        env, _ = values(self.graph(length), self.cut(inputs, start, length), given)
        return {k: env[k].float() for k in (self.full, *self.accs)}

    def start(self, like: dict) -> dict:
        """The state before the first iteration: the reducer's start value and zeros."""
        import torch

        init = type(self.node(self.tile, self.full)).carry_init
        out = {self.full: torch.full_like(like[self.full], init)}
        for acc in self.accs:
            out[acc] = torch.zeros_like(like[acc])
        return out


@dataclass(eq=False)
class Case:
    """One prefix extension: everything a candidate update may read, and what it must give."""

    label: str
    prefix: int
    old: dict  # F(H): full-extent value and accumulators (the start state for H empty)
    new: dict  # F(H ++ T)
    tile: dict  # the tile's finished results, computed with the NEW full-extent value
    moved: float  # share of rows whose full-extent value moved
    full: str = ""

    @property
    def rows(self) -> int:
        return int(self.old[self.full].numel())

    def view(self, part: str, name: str):
        """``part`` (old / new / tile) of ``name`` as ``(rows, width)``, float32."""
        return getattr(self, part)[name].float().reshape(self.rows, -1)

    def only(self, keep) -> "Case":
        """The case restricted to the rows ``keep`` (a boolean mask or indices)."""
        import torch

        keep = torch.as_tensor(keep)

        def cut(d):
            return {k: v.float().reshape(self.rows, -1)[keep] for k, v in d.items()}

        old, new = cut(self.old), cut(self.new)
        moved = float((new[self.full] > old[self.full]).float().mean()) if keep.numel() else 0.0
        return Case(f"{self.label} (rows {keep.tolist()})", self.prefix, old, new,
                    cut(self.tile), moved, self.full)


def extension(flat: Flat, inputs, prefix: int, label: str) -> Case:
    """The case ``F(H) -> F(H ++ T)`` for ``|H| = prefix`` and the next tile ``T``."""
    import torch

    t = flat.tile
    new = flat.state(inputs, 0, prefix + t)
    own = flat.state(inputs, prefix, t)  # the tile on its own: only its full-extent value
    old = flat.state(inputs, 0, prefix) if prefix else flat.start(new)
    # The full-extent value's own hole is its reducer's associativity, exact. Where it does
    # not hold, the value is not a running reduction of the looped dim (its operand depends
    # on the prefix, a max of ``exp(x - max(x))``) or the cut is wrong: no rule.
    fold = combine_of(flat.node(t, flat.full))
    if not torch.equal(fold(old[flat.full], own[flat.full]), new[flat.full]):
        raise Refused(E5.format(full=flat.full, label=label))
    tile = flat.state(inputs, prefix, t, given={flat.full: new[flat.full]})
    moved = float((new[flat.full] > old[flat.full]).float().mean())
    return Case(label, prefix, old, new, tile, moved, flat.full)


# --------------------------------------------------------------------------- reference set

MARGIN = 4 * LOOSE_RTOL  # a row discriminates where both errors exceed the search's filter 4x


def discriminating(case: Case, acc: str):
    """Per row of ``case``: does it tell the right update of ``acc`` from "no repair"
    (``a + r``: the move of the full-extent value is visible) AND from "tile only" (``r``:
    the old value is visible)? Relative to the largest ``|target|`` of the row."""
    a, r, t = case.view("old", acc), case.view("tile", acc), case.view("new", acc)
    scale = t.abs().amax(dim=1, keepdim=True) + 1e-30
    move = ((a + r - t).abs() / scale).amax(dim=1)
    old = ((r - t).abs() / scale).amax(dim=1)
    return (move > MARGIN) & (old > MARGIN)


def adequacy(cases, accs) -> dict:
    """``{acc: discriminating rows}`` over the cases with a non-empty prefix."""
    later = [c for c in cases if c.prefix]
    return {acc: sum(int(discriminating(c, acc).sum()) for c in later) for acc in accs}


def check_adequacy(cases, full: str, accs, needed: int) -> dict:
    """``adequacy``, or E10 for the first accumulator with fewer than ``needed`` rows."""
    counts = adequacy(cases, accs)
    for acc in accs:
        if counts[acc] < needed:
            raise Refused(E10.format(full=full, acc=acc, k=counts[acc], n=needed))
    return counts


def shaped(flat: Flat, seed: int, kind: str, gain: float) -> list:
    """A sample at ``N`` shaped so that the full-extent value moves.

    ``ramp xg``: every input becomes ``1 + 0.25 tanh(x)``; the inputs with the looped dim
    are multiplied by a ramp from 1 to 2.5 along it, the others by ``g``.
    ``tail xg``: in the inputs that carry the looped dim (one run), the elements from the
    first tile on are multiplied by ``g``.
    """
    import torch

    inputs = flat.sample(N, seed)
    if kind == "none":
        return inputs  # as sampled: already what the inputs hold
    out = []
    for x, (ax, groups) in zip(inputs, flat.looped_layout()):
        if kind == "ramp":
            level = 1.0 + 0.25 * torch.tanh(x.float())
            if ax is None:
                out.append((gain * level).to(x.dtype))
                continue
            shape = [1] * x.dim()
            shape[ax] = groups * N
            ramp = (1.0 + 1.5 * torch.arange(N, dtype=torch.float32) / N).repeat(groups)
            out.append((level * ramp.reshape(shape)).to(x.dtype))
        else:  # tail
            if ax is None or groups != 1:
                out.append(x)
                continue
            y = x.clone()
            part = y.narrow(ax, flat.tile, N - flat.tile)
            part.copy_((part.float() * gain).to(y.dtype))
            out.append(y)
    return flat.stored(out)


@dataclass
class References:
    cases: list  # Case, in order: per normal seed the first iteration and the prefixes; then
    #              the shaped sample's prefixes
    rows: int  # rows of the full-extent value (one case's worth)
    adequacy: dict  # acc -> discriminating rows over the cases with a non-empty prefix
    shaped: str  # label of the shaped sample that was chosen
    tried: int  # shaped samples tried


def reference_set(flat: Flat, *, check: bool = True) -> References:
    """The reference set of DESIGN 5.4, with its adequacy checked (E10; ``check=False``
    leaves the check to the caller, which then has the cases for its report)."""
    cases = []
    for seed in SEEDS_NORMAL:
        inputs = flat.sample(N, seed)
        cases.append(extension(flat, inputs, 0, f"first seed={seed}"))
        cases += [extension(flat, inputs, p, f"normal seed={seed} prefix={p}") for p in PREFIXES]
    rows = cases[0].rows
    best, best_score, best_label, tried = None, -1, "", 0
    goal = len(PREFIXES) * rows
    for seed in TRY_SEEDS:
        for kind, gain in SHAPINGS:
            tried += 1
            tag = f"shaped seed={seed} {kind} x{gain:g}"
            inputs = shaped(flat, seed, kind, gain)
            sample = [extension(flat, inputs, p, f"{tag} prefix={p}") for p in PREFIXES]
            score = min(sum(int(discriminating(c, acc).sum()) for c in sample)
                        for acc in flat.accs)
            if score > best_score:
                best, best_score, best_label = sample, score, tag
            if best_score >= goal:
                break
        if best_score >= goal:
            break
    cases += best
    counts = (check_adequacy(cases, flat.full, flat.accs, rows) if check
              else adequacy(cases, flat.accs))
    return References(cases, rows, counts, best_label, tried)


def row_dependency(flat: Flat, seed: int = SEEDS_NORMAL[0]) -> None:
    """E6 unless the full-extent value enters each accumulator row by row: raising it in
    one row of the tile instance changes accumulator elements of that row only (the
    accumulator viewed as ``rows + (-1,)``). This is what the search's typing, the kernel's
    row-wise repair and the proof's single symbol per row assume."""
    import torch

    t = flat.tile
    inputs = flat.sample(t, seed)
    env, _ = values(flat.graph(t), inputs)
    m = env[flat.full].float()
    rows = int(m.numel())

    def accs(value):
        out, _ = values(flat.graph(t), inputs, {flat.full: value})
        return {a: out[a].float() for a in flat.accs}

    base = accs(m)
    for acc, value in base.items():
        if value.numel() % rows:
            raise Refused(E6.format(acc=acc, full=flat.full))
    for rho in range(rows):
        bumped = m.clone().reshape(-1)
        bumped[rho] += 1.0
        moved = accs(bumped.reshape(m.shape))
        for acc, value in moved.items():
            changed = (value != base[acc]).reshape(rows, -1)
            others = torch.ones(rows, dtype=torch.bool)
            others[rho] = False
            if bool(changed[others].any()):
                raise Refused(E6.format(acc=acc, full=flat.full))


def block(cases, part: str, name: str) -> np.ndarray:
    """One float32 block ``(rows of every case, width)``: ``part`` (old / new / tile) of
    ``name`` in every case, stacked."""
    out = [c.view(part, name).numpy() for c in cases]
    for x in out:
        assert x.dtype == np.float32, x.dtype
    return np.concatenate(out, axis=0)
