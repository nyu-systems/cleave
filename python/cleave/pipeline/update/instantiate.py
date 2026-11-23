"""Recipes: the program's own flat graph, built again at other extents (DESIGN 5.2).

The references of an update rule are the flat graph evaluated on prefixes of the looped
dim, so the graph has to be built again with the looped dim at a handful of elements and
every other dim small. A :class:`Recipe` knows how one flat graph was made and makes it
again: it is re-instantiated, never re-implemented.

    searched(g)          a searched graph: its structure lowered with another symbol map
    identity(g)          an un-searched graph: the program's records at other dim extents
    partial(g, ...)      the partial of a split built from the loop state of ``g``
    combine(g, ...)      the combine of that split

``symbols`` maps each symbol (a structure proxy, or a program dim name) to the program's
extent (None where the program has none: the split factor and the extent of one split).
``looped`` names the symbol(s) the full-extent value reduces; several symbols tied by a
contraction are resized together (a ``frozenset``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable



@dataclass(eq=False)
class Recipe:
    kind: str
    symbols: dict  # symbol -> the program's extent (None: not a dim of the program)
    dtype: str  # the dtype the program's own graph is lowered in
    build: Callable = field(repr=False)  # (sizes, dtype) -> Graph
    find_looped: Callable = field(repr=False)  # (flat_graph, full) -> symbol | frozenset | None
    origin: Any = field(default=None, repr=False)

    def at(self, sizes: dict, dtype: str):
        """The flat graph with ``sizes`` (symbol -> extent) in ``dtype``."""
        return self.build(dict(sizes), dtype)

    def looped(self, flat_graph, full: str):
        """The symbol(s) of the dim ``full`` reduces, read off the provenance the scheduler
        pins by; None where the provenance does not name them (the caller then probes)."""
        return self.find_looped(flat_graph, full)

    # ------------------------------------------------------------------ constructors

    @classmethod
    def of(cls, g) -> "Recipe":
        """The recipe of a solo graph as ``pipeline.plans.lower`` lowers it."""
        return cls.identity(g) if g.symbol_map is None else cls.searched(g)

    @classmethod
    def searched(cls, g) -> "Recipe":
        def build(sizes, dtype):
            return _one_graph(g.structure, sizes, dtype)

        def find(flat_graph, full):
            inputs = [op for op in g.structure if op["op_type"] == "kn_input_op"]

            def symbol(index, axis):
                if index is None or index >= len(inputs):
                    return None
                dims = _tensor_dims(inputs[index]["output_tensors"][0])
                return dims[axis] if axis < len(dims) else None

            return _looped_from_tokens(flat_graph, full, symbol, set(g.symbol_map),
                                       ranks=[len(_tensor_dims(op["output_tensors"][0]))
                                              for op in inputs])

        return cls("searched", dict(g.symbol_map), g.dtype, build, find, g)

    @classmethod
    def identity(cls, g) -> "Recipe":
        from ...frontend.tensor import Dim

        program = g.program
        symbols = {name: int(extent) for name, extent in program.sizes.items()}

        def build(sizes, dtype):
            return _one_graph(program.to_graph_structure(sizes), None, dtype)

        def find(flat_graph, full):
            def symbol(index, axis):
                if index is None or index >= len(program.inputs):
                    return None
                dims = program.inputs[index].dims
                d = dims[axis] if axis < len(dims) else None
                return d.name if isinstance(d, Dim) else None

            return _looped_from_tokens(flat_graph, full, symbol, set(symbols),
                                       ranks=[len(s.dims) for s in program.inputs])

        return cls("identity", symbols, g.dtype, build, find, g)

    @classmethod
    def partial(cls, g, full: str, accs: tuple, *, solo=None, split_factor=None) -> "Recipe":
        """The partial of a split built from the loop state ``(full, accs)`` of the searched
        graph ``g`` (``lower.state_split``): the solo dims without the looped one, plus the
        split factor and the extent of one split, which is the dim the partial loops."""
        from ...lower.state_split import lower_partial, op_indices, partial_structure

        solo = solo if solo is not None else _one_graph(g.structure, g.symbol_map, g.dtype)
        full_op, acc_ops = op_indices(solo, full, tuple(accs))
        _, factor, looped, inner = partial_structure(g.structure, full_op, acc_ops)
        extent = g.symbol_map.get(looped)
        symbols = {k: v for k, v in g.symbol_map.items() if k != looped}
        symbols[factor] = None if split_factor is None else int(split_factor)
        symbols[inner] = (None if split_factor is None or extent is None
                          else int(extent) // int(split_factor))

        def build(sizes, dtype):
            f = int(sizes.pop(factor))
            sizes[looped] = f * int(sizes.pop(inner))
            return lower_partial(g, full_op, acc_ops, f, symbol_map=sizes, dtype=dtype)

        return cls("partial", symbols, g.dtype, build, lambda flat_graph, name: inner, g)

    @classmethod
    def combine(cls, g, full: str, rules: tuple, split_pos: int, out_dtype: str, *, solo=None,
                split_factor=None) -> "Recipe":
        """The combine of that split (``lower.state_split.combine_graph``): the solo dims
        without the looped one, plus the number of splits, which is the dim it loops."""
        from ...lower.state_split import combine_graph, op_indices, partial_structure

        solo = solo if solo is not None else _one_graph(g.structure, g.symbol_map, g.dtype)
        accs = tuple(a for a, _ in rules)
        full_op, acc_ops = op_indices(solo, full, accs)
        _, factor, looped, _ = partial_structure(g.structure, full_op, acc_ops)
        symbols = {k: v for k, v in g.symbol_map.items() if k != looped}
        symbols[factor] = None if split_factor is None else int(split_factor)

        def build(sizes, dtype):
            f = int(sizes.pop(factor))
            # what the combine keeps of the solo graph does not depend on the looped extent
            base = _one_graph(g.structure, {**sizes, looped: 4}, dtype)
            return combine_graph(base, full, rules, f, split_pos, dtype)

        return cls("combine", symbols, out_dtype, build, lambda flat_graph, name: factor, g)


# ---------------------------------------------------------------------------------- helpers


def _tensor_dims(t) -> list:
    return list(t["dim"][: int(t.get("num_dims", len(t["dim"])))])


def _one_graph(structure, symbol_map, dtype):
    from ...lower.process import to_bundles

    bundles = to_bundles([structure], symbol_map=None if symbol_map is None else dict(symbol_map),
                         split_factors=None, dtype=dtype, out_dtype=dtype)
    if len(bundles) != 1 or len(bundles[0]) != 1:
        raise ValueError(f"the structure lowers to {[len(b) for b in bundles]} graphs, not one")
    return bundles[0][0]


def _looped_from_tokens(flat_graph, full: str, symbol, known: set, ranks: list):
    """The symbol(s) behind the reduction axis of ``full`` and of the reductions that loop
    under it: each provenance token names ``(placeholder, dim)``; the placeholder's
    ``input_index`` says which input it is and ``symbol(index, dim)`` names the symbol of
    that axis. None when a token does not name a known symbol (a placeholder declared in
    another rank than its input, a literal dim): the caller probes instead."""
    from ...schedule.streaming import obstacles
    from ...schedule.tiling import reduction_ancestors, scheduled_reductions

    nodes, red = scheduled_reductions(flat_graph)
    by_name = {n.name: n for n in nodes}
    if full not in by_name or not red.get(by_name[full]):
        return None
    tokens = set(red[by_name[full]][0])
    for ob in obstacles(nodes, red, reduction_ancestors(nodes)):
        if ob.full == full:
            for name in ob.loops:
                if red.get(by_name[name]):
                    tokens |= set(red[by_name[name]][0])
    placeholders = {id(e.src_node): e.src_node for n in nodes for e in n.inputs
                    if e.src_node.is_placeholder()}
    found = set()
    for node_id, _, axis in tokens:
        p = placeholders.get(node_id)
        index = getattr(p, "input_index", None)
        if p is None or index is None or index >= len(ranks) \
                or len(p.get_shape(0)) != ranks[index]:
            return None
        s = symbol(index, axis)
        if s is None or s not in known:
            return None
        found.add(s)
    if not found:
        return None
    return next(iter(found)) if len(found) == 1 else frozenset(found)
