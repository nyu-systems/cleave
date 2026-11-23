"""A split pair BUILT from the loop state of a streaming obstacle.

With the max explicit in the flat graph the superoptimizer returns no split graph (a max
has no sum-combine). The pair is constructed instead, from the solo graph and the loop
state the scheduler proposes for it (``cleave.ir.update_rule.Obstacle``):

  partial   the solo structure up to the loop state, replayed on a split of the looped dim.
            Every split computes ``F(split_i)``: the full-extent value ``m_i`` of its own
            slice and the accumulators relative to it. Its outputs are the accumulators,
            then ``m_i`` (role ``"shift"``, float32). Built at the structure level, in the
            format ``lower.split.cut_split_graph`` emits, and lowered by the same ``to_ir``.

  combine   the solo graph with its loop state substituted: the full-extent value becomes
            ``M = max_i m_i`` over the split axis, each accumulator becomes
            ``sum_i repair(acc_i, m_i, M)`` with the PROVEN repair (the factor of the
            ``Streaming`` record with ``m := m_i``, ``m' := M``), and everything behind the
            accumulators is the solo graph's own (the final divide). Built at the IR level.

Both are ordinary flat graphs with an explicit max in front of their sums: where a tiling
loops the looped dim (the partial's keys, the combine's split axis) the scheduler finds
the same streaming obstacle and serves it with a rule derived for that graph.

Reused from the split plumbing of searched split graphs: ``Boundary`` records and roles,
the stored-layout record, the three-output partial, float32 at the cut, per-role
tolerances, the hand-off.

Nothing calls this module yet: the pipeline reaches it once split programs with a max are
routed here (``pipeline.plans.lower``).
"""

from __future__ import annotations

from copy import deepcopy

from .split import _renumber_graph, _tensor_dims

SHIFT = "shift"
STAT_DTYPE = "float32"


class SplitRefused(Exception):
    """No pair is built from this loop state; the text is the reason."""


def _axis(op_type: str):
    """``(kind, axis)`` of ``kn_reduce_sum_2_op`` / ``kn_reduce_max_2_op``, else None."""
    for kind in ("reduce_sum", "reduce_max"):
        head = f"kn_{kind}_"
        if op_type.startswith(head) and op_type.endswith("_op"):
            return kind, int(op_type[len(head):-3])
    return None


def op_indices(solo, full: str, accs: tuple):
    """``(full_op, acc_ops)``: the indices, in the structure ``solo`` was lowered from, of
    the ops behind the full-extent node and the accumulators (``node.op_index``, recorded
    by ``lower.convert.to_ir``)."""
    by_name = {n.name: n for n in solo.nodes}
    return by_name[full].op_index, tuple(by_name[a].op_index for a in accs)


def partial_structure(structure, full_op: int, acc_ops: tuple):
    """``(partial ops, symbolic split factor, looped proxy, inner proxy)``.

    ``full_op`` / ``acc_ops`` are indices into ``structure`` (:func:`op_indices`): the
    state tensors are the outputs of those ops.
    """
    ops = list(structure)
    full = ops[full_op]
    kind_axis = _axis(full["op_type"])
    if kind_axis is None or kind_axis[0] != "reduce_max":
        raise SplitRefused(f"op {full_op} ({full['op_type']}) is not a max")
    looped = _tensor_dims(full["input_tensors"][0])[kind_axis[1]]
    proxies = {d for op in ops for t in op["input_tensors"] + op["output_tensors"]
               for d in _tensor_dims(t)}
    factor = next((f for f in (2, 3, 5, 7) if looped % f == 0 and f not in proxies
                   and looped // f not in proxies), None)
    if factor is None:
        raise SplitRefused("no symbolic split factor for the looped dim")
    inner = looped // factor

    state = [full["output_tensors"][0]["guid"]] + [
        ops[a]["output_tensors"][0]["guid"] for a in acc_ops]
    produced = {op["output_tensors"][0]["guid"]: op for op in ops if op["output_tensors"]}
    needed, stack = set(), list(state)
    while stack:
        g = stack.pop()
        if g in needed:
            continue
        needed.add(g)
        stack.extend(t["guid"] for t in produced[g]["input_tensors"])

    fresh = iter(range(20000001, 30000000))
    here = {}  # guid of the solo structure -> the tensor that stands for it in the partial
    is_split = set()

    def split_desc(t):
        dims = [factor] + [inner if d == looped else d for d in _tensor_dims(t)]
        out = {"num_dims": len(dims), "dim": dims, "guid": next(fresh)}
        if "dtype" in t:
            out["dtype"] = t["dtype"]
        is_split.add(out["guid"])
        return out

    partial = []
    for op in ops:  # inputs first, each split along the looped dim where it has it
        if op["op_type"] != "kn_input_op":
            continue
        t = op["output_tensors"][0]
        partial.append(deepcopy(op))
        here[t["guid"]] = deepcopy(t)
    for op in ops:
        if op["op_type"] != "kn_input_op":
            continue
        t = op["output_tensors"][0]
        dims = _tensor_dims(t)
        if looped in dims and t["guid"] in needed:
            out = split_desc(t)
            partial.append({"op_type": f"kn_split_{dims.index(looped)}_op",
                            "input_tensors": [deepcopy(t)], "output_tensors": [out]})
            here[t["guid"]] = out
    for op in ops:
        if op["op_type"] in ("kn_input_op", "kn_output_op"):
            continue
        out_t = op["output_tensors"][0]
        if out_t["guid"] not in needed:
            continue
        ins = [deepcopy(here[t["guid"]]) for t in op["input_tensors"]]
        if not any(t["guid"] in is_split for t in ins):
            raise SplitRefused(f"{op['op_type']} on the way to the loop state reads no "
                               "split input")
        new = deepcopy(op)
        new["input_tensors"] = ins
        new["output_tensors"] = [split_desc(out_t)]
        kind_axis = _axis(op["op_type"])
        if kind_axis is not None:  # one more leading dim
            new["op_type"] = f"kn_{kind_axis[0]}_{kind_axis[1] + 1}_op"
        partial.append(new)
        here[out_t["guid"]] = new["output_tensors"][0]
    for g in state[1:] + state[:1]:  # the accumulators, then the full-extent value
        partial.append({"op_type": "kn_output_op", "input_tensors": [deepcopy(here[g])],
                        "output_tensors": []})
    return _renumber_graph(partial), factor, looped, inner


def lower_partial(searched, full_op: int, acc_ops: tuple, split_factor: int, *,
                  symbol_map=None, dtype=None):
    """The lowered partial of ``searched`` (a solo graph) for one split factor: outputs in
    boundary order (the accumulators, then the full-extent value with role ``"shift"``),
    all float32 at the cut.

    ``symbol_map`` / ``dtype`` default to the searched graph's own; given, the same partial
    is built at other extents or in another dtype.
    """
    from tvm import DataType

    from ..ir.graph import Boundary, Graph
    from ..ir.graph_utils import find_topo_sort
    from .convert import to_ir
    from .process import _normalize_split_graph_layouts
    from .split import _augment_split_symbol_mapping

    dtype = dtype or searched.dtype
    partial, factor, looped, inner = partial_structure(searched.structure, full_op, acc_ops)
    local = _augment_split_symbol_mapping(
        [partial], dict(symbol_map or searched.symbol_map), symbolic_split_factor=factor,
        split_factor=int(split_factor))
    normalized = _normalize_split_graph_layouts([partial, []])[0]
    graph = to_ir(normalized, symbol_map=local, fuse_rsqrt=True, dtype=dtype,
                  accum_dtype=STAT_DTYPE, out_dtype=dtype)
    outs = sorted((n for n in graph.nodes if n.is_output()), key=lambda n: n.boundary.index)
    shift = outs[-1]
    shift.boundary = Boundary(shift.boundary.index, SHIFT)
    for out in outs:  # the sums and the max they are relative to cross the cut in float32
        out._dtypes[0] = DataType(STAT_DTYPE)
    # The kernel returns the max first: the policy walks a graph from its outputs in order,
    # and the max lets the scores go once the exp has read them.
    graph = Graph(nodes=find_topo_sort([shift] + outs[:-1]))
    graph.split_symbols = dict(factor=factor, looped=looped, inner=inner)
    return graph


def combine_graph(solo, full: str, rules: tuple, split_factor: int, split_pos: int,
                  out_dtype: str):
    """The combine of a split pair, from the solo flat graph ``solo`` and the proven repair.

    ``rules``: ``((accumulator, factor expression), ...)`` of the ``Streaming`` record, in
    boundary order. ``split_pos``: where the split axis sits in the state tensors (after
    the batch dims, as the partial's normalised layout puts it).
    """
    from ..ir.graph import Boundary, Graph, OutputNode
    from ..ir.graph_utils import find_topo_sort
    from ..ir.ops import Mul, ReduceSum, wire_input
    from ..ir.rolling import build_expression

    solo = solo.clone()
    by_name = {n.name: n for n in solo.nodes}
    full_node = by_name[full]

    def with_split(shape):
        shape = tuple(int(s) for s in shape)
        return shape[:split_pos] + (int(split_factor),) + shape[split_pos:]

    rows = with_split(full_node.get_shape())
    nb = split_pos  # the batch dims in front of the split axis
    # M = the full-extent value over all splits: the reducer of the solo graph, over the
    # split axis
    top = type(full_node)(rows, axis=split_pos, name="state_max", dtype=STAT_DTYPE,
                          out_dtype=STAT_DTYPE)
    m_in = top.inputs[0].src_node
    m_in.name = "PlaceHolder_state_max"
    m_in.boundary = Boundary(len(rules), SHIFT)
    m_in._dtypes[0] = _dtype(STAT_DTYPE)
    made = {}
    outputs_of = {}
    for index, (acc, factor) in enumerate(rules):
        old = by_name[acc]
        dims = with_split(old.get_shape())
        if dims == rows:
            broadcast = -1
        elif dims[:-1] == rows:
            broadcast = 0  # one weight per row of the tile
        else:
            raise SplitRefused(f"{acc} {dims} is not the full-extent value {rows} with at "
                               "most one trailing axis")
        # repair(acc_i, m_i, M) = acc_i * factor(m := m_i, m' := M); M lacks the split axis
        try:
            weight = build_expression(factor, {"m": m_in, "m'": top}, made, stem="state",
                                      rows=rows, n_batch_dims=nb,
                                      exp_cls=_exp_class(solo, full), narrow=("m'",))
        except ValueError as e:
            raise SplitRefused(str(e)) from e
        repaired = Mul(dims, broadcast=broadcast, name=f"{acc}_repaired", dtype=STAT_DTYPE,
                       out_dtype=STAT_DTYPE, n_batch_dims=nb)
        part = repaired.inputs[0].src_node
        part.name = f"PlaceHolder_state_{acc}"
        part.boundary = Boundary(index, "")
        part._dtypes[0] = _dtype(STAT_DTYPE)
        wire_input(weight, repaired, 1)
        summed = ReduceSum(dims, axis=split_pos, name=f"{acc}_combined", dtype=STAT_DTYPE,
                           out_dtype=old.out_dtype)
        wire_input(repaired, summed, 0)
        outputs_of[acc] = summed
    # everything behind the loop state is the solo graph's own
    for acc, summed in outputs_of.items():
        for edge in list(by_name[acc].outputs):
            wire_input(summed, edge.dst_node, edge.dst_id)
    for edge in list(full_node.outputs):  # a reader of the finished max behind the loop
        if not _feeds(edge.dst_node, [by_name[a] for a, _ in rules]):
            wire_input(top, edge.dst_node, edge.dst_id)
    outs = [n for n in solo.nodes if isinstance(n, OutputNode)]
    return Graph(nodes=find_topo_sort(outs))


def _feeds(node, targets) -> bool:
    stack, seen = [node], set()
    while stack:
        n = stack.pop()
        if any(n is t for t in targets):
            return True
        if id(n) in seen:
            continue
        seen.add(id(n))
        stack.extend(e.dst_node for e in n.outputs)
    return False


def _dtype(name):
    from tvm import DataType

    return DataType(name)


def _exp_class(graph, full):
    """The class of the exp between the full-extent value and what it feeds."""
    from ..ir.ops import Exp

    node = next(n for n in graph.nodes if n.name == full)
    stack, seen = [node], set()
    while stack:
        n = stack.pop()
        for e in n.outputs:
            d = e.dst_node
            if id(d) in seen:
                continue
            seen.add(id(d))
            if isinstance(d, Exp):
                return type(d)
            stack.append(d)
    return Exp
