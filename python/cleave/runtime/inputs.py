"""Input distributions a kernel can be validated on: ``normal`` and the adversarial softmax cases.

``normal`` is what :func:`cleave.runtime.reference.make_inputs` samples. The adversarial ones
rewrite that sample afterwards (so the normal stream never shifts), and only what the logits
of a softmax are made of: the operands of the *score* matmuls (the matmuls an ``Exp`` is
reached from without crossing another matmul), and an input that is the logits itself (a
stabilized ``Exp`` that reads an input: ``cl.softmax(x)``; through the view of a split as
well, so the partial of a program split over its softmax axis is held to the same cases).

The cases are laid along the softmax axis of the scores. The operand of the score matmul
whose rows are that axis is called the key operand here and the other the query operand,
after attention, where the softmax runs over the keys (the second operand's rows). For a
softmax over the queries (``cl.softmax(q @ k.mT, dim=-2)``) the first operand's rows are
the softmax axis, and the two operands change places.

``large_logits``  the query operand times 100, so every logit is 100 times larger;
``increasing``    logits that grow along the key axis for every query row, by more than
                  fp32 ``exp`` can hold, so a running max changes at every tile;
``decreasing``    the same ramp the other way round: the row max is the first key, so a
                  running max is final after the first tile and never moves again;
``last_spike``    normal logits, except that the last key scores at least 64 above every
                  other: the max jumps in the last tile, by more than everything
                  accumulated before it survives;
``constant``      every key is the same key: all logits of a row are equal, the softmax is
                  uniform and a running max never moves;
``offset``        normal-sized logits (on a 1/8 grid, see ``OFFSET``) that all sit about
                  1e4 below zero: ``exp`` of any of them is 0, only the differences count.

Logits that are an input get the same rows written into them directly, along the softmax
axis of the ``Exp`` that reads them; their offset is ``LOGIT_OFFSET`` (what fp16 holds next
to a normal-sized term).

The first three are shared with the dynamic operators (``SHARED_DISTRIBUTIONS``); the others
exist for the dense programs only.

Each is requested explicitly (``EvalConfig.distributions`` for a program,
``make_inputs(..., distribution=...)`` for a graph); a program whose softmax is exact is
validated on all of them (``ADVERSARIAL``, see ``cleave.pipeline.build``). A graph with no
score matmul and no logit input has no logits to shape, and asking for an adversarial
distribution on it is an error.

In a graph whose softmax is exact (an ``Exp`` carries a ``Stabilize`` record) only the
logits of the stabilized ``Exp`` nodes are shaped: a plain ``exp`` written next to a softmax
makes no claim about large arguments, and its own matmul is left alone. Where a shaped input
also feeds such a plain ``exp``, the graph has no adversarial case at all
(``why_not_adversarial`` says so): its reference would overflow, by the program's own
definition.
"""

from __future__ import annotations

from ..core.errors import SpecError
from ..ir.graph import PlaceHolderNode
from ..ir.ops import Div, Exp, Matmul, Mul, PermuteView, ReduceMax, SplitView, Sub
from .reference import placeholder_order

# What a dynamic operator can be validated on as well (``cleave.dynamic.reference`` adds two
# of its own), and what only a dense program can.
SHARED_DISTRIBUTIONS = ("normal", "large_logits", "increasing")
DISTRIBUTIONS = SHARED_DISTRIBUTIONS + ("decreasing", "last_spike", "constant", "offset")
# Everything but ``normal``: what a program whose softmax is exact (``cl.softmax``) is held to.
ADVERSARIAL = DISTRIBUTIONS[1:]

LARGE_LOGIT_FACTOR = 100.0
# ``increasing``: q.k ramps from 0 to ``a * INCREASING_SPAN * sqrt(K)`` along the keys (``K`` the
# contracted width, ``a`` in [0.5, ...) per query row), i.e. over ``a * 128`` after the usual
# 1/sqrt(K) scaling -- past ln(fp32 max) = 88.7.
INCREASING_SPAN = 128.0
# ``offset``: every logit is ``-OFFSET`` plus a normal-sized term, after the usual 1/sqrt(K)
# scaling. The offset is the product of one coordinate of the contraction, ``OFFSET_QUERY``
# on the query side; the other coordinates are rounded to multiples of ``OFFSET_GRID``. Their
# products are then multiples of 1/64 and every partial sum of the contraction is exact in
# fp32 (below 2^18 for K up to 512), in whatever order it is accumulated: the kernel and the
# reference see the same logits bit for bit, so the case measures the softmax and not the
# rounding of a sum at magnitude 1e5, where one fp32 ulp is 1e-3 of a logit.
OFFSET = 1.0e4
OFFSET_QUERY = 256.0
OFFSET_GRID = 0.125
# ... and for logits that are an input: fp16 has no room for 1e4 next to a normal-sized term
# (one ulp there is 8). At 512 it is 0.5, and ``exp(-512)`` is 0 in fp32 all the same.
LOGIT_OFFSET = 512.0


def check_distributions(distributions) -> tuple:
    """``distributions`` as a tuple of known names (at least one)."""
    names = (distributions,) if isinstance(distributions, str) else tuple(distributions)
    unknown = [n for n in names if n not in DISTRIBUTIONS]
    if not names or unknown:
        raise SpecError(
            f"input distributions must be drawn from {DISTRIBUTIONS}, got {distributions!r}"
        )
    return names


def adversarial_cases(graph) -> tuple:
    """The adversarial cases ``graph``'s inputs can be shaped into, in ``ADVERSARIAL`` order.

    All of them when every score matmul reads a query input and a separate key input, and
    when the logits are an input themselves; only ``large_logits`` when one input is both
    query and key; none when the graph has neither a score matmul nor a logit input, when an
    operand of a score matmul is computed (nothing here can shape it through what computes
    it), or when a shaped input also feeds a plain ``exp``. :func:`why_not_adversarial`
    gives the reason for every case that is missing.

    The cases shape the logits that can be shaped. A graph with a second exact softmax
    whose logits are neither a score matmul nor an input (``cl.softmax(x + y)``) keeps the
    cases of the first, and :func:`why_not_adversarial` names the softmax they do not reach.
    """
    return _cases(graph)[0]


def why_not_adversarial(graph) -> str:
    """Why ``graph`` is not held to every adversarial case, or why the cases it is held to
    do not reach every exact softmax of it; ``""`` when they are all there and reach all."""
    return _cases(graph)[1]


def _cases(graph) -> tuple:
    scores = score_matmuls(graph)
    try:
        operands = [(_operand(m, 1 - key)[0], _operand(m, key)[0])  # (query, key) inputs
                    for m, key in ((m, key_operand(graph, m)) for m in scores)]
    except SpecError as e:
        return (), str(e)
    logits = [node for node, _, _ in logit_inputs(graph)]
    if not operands and not logits:
        return (), "its logits are neither the result of a score matmul nor an input"
    shaped = [p for pair in operands for p in pair] + logits
    # Only where the softmax is exact: a graph without a record makes no claim at all, and
    # the cases are simply what its inputs can be shaped into.
    exact = any(_stabilized(n) for n in graph.nodes)
    plain = _plain_exp_fed_by(graph, shaped) if exact else None
    if plain is not None:
        exp, placeholder = plain
        return (), (
            f"input {placeholder.name!r} also feeds {exp.name!r}, an exp that is not "
            "stabilized: on shaped logits its reference overflows"
        )
    # A softmax the cases cannot reach is validated on its normal logits whatever the other
    # softmaxes of the graph are shaped into: the cases stand, and the note names it.
    unshaped = _unshaped(graph, logits) if exact else []
    rest = ""
    if unshaped:
        rest = ("the logits of " + ", ".join(repr(n.name) for n in unshaped)
                + " are neither the result of a score matmul nor an input: no case shapes them")
    both = {id(q) for q, _ in operands} & {id(k) for _, k in operands}
    if both:
        name = next(q.name for q, _ in operands if id(q) in both)
        return ("large_logits",), (
            f"input {name!r} is both a query and a key of the scores: it can be scaled, "
            "not ramped" + ("; " + rest if rest else "")
        )
    return ADVERSARIAL, rest


def softmax_axis(node):
    """The axis, counted from the end, that ``node`` is the exp of an exact softmax over,
    or None.

    Either the record of the hand-derived pass (``Exp.stabilize``, with one axis), or a
    structural fact: the exp reads ``x - max(x, axis)``, a ``Sub`` whose second operand is a
    ``ReduceMax`` of its first. Its argument is then never positive, whatever the logits
    are, which is the claim the adversarial cases test.
    """
    record = getattr(node, "stabilize", None)
    if record is not None:
        return record.axes[0] if len(record.axes) == 1 else None
    if type(node) is not Exp:
        return None
    shift = _explicit_max(node.inputs[0].src_node)
    return None if shift is None else shift.flat_axis - len(node.get_shape(0))


def _explicit_max(node):
    """The ``ReduceMax`` of ``node`` = ``x - max(x)``: a ``Sub`` without a constant whose
    second operand is a ``ReduceMax`` of its first (same node, same output); else None."""
    if type(node) is not Sub or node.const is not None:
        return None
    x, m = sorted(node.inputs, key=lambda e: e.dst_id)
    if type(m.src_node) is not ReduceMax:
        return None
    (reduced,) = m.src_node.inputs
    if reduced.src_node is not x.src_node or reduced.src_id != x.src_id:
        return None
    return m.src_node


def _logits_of(exp):
    """The node an exact-softmax exp takes its logits from: what it reads, or the first
    operand of the explicit ``x - max(x)``."""
    src = exp.inputs[0].src_node
    if getattr(exp, "stabilize", None) is None and softmax_axis(exp) is not None:
        return next(e for e in src.inputs if e.dst_id == 0).src_node
    return src


def _stabilized(node) -> bool:
    return getattr(node, "stabilize", None) is not None or softmax_axis(node) is not None


def _unshaped(graph, logits) -> list:
    """The stabilized ``Exp`` nodes no case shapes: no score matmul behind them, and what
    they read is not one of the ``logits`` inputs."""
    wanted = {id(p) for p in logits}
    return [n for n in graph.nodes if isinstance(n, Exp) and _stabilized(n)
            and not _score_matmuls_of([n]) and id(_read(n)[0]) not in wanted]


def _plain_exp_fed_by(graph, placeholders):
    """``(exp, placeholder)`` for an ``Exp`` without a record that is computed from one of
    ``placeholders``, else None. A stabilized ``Exp`` on the way is not crossed: what it
    returns is at most 1, whatever its logits are."""
    wanted = {id(p) for p in placeholders}
    for exp in graph.nodes:
        if not isinstance(exp, Exp) or _stabilized(exp):
            continue
        stack, seen = [exp], set()
        while stack:
            for edge in stack.pop().inputs:
                src = edge.src_node
                if id(src) in wanted:
                    return exp, src
                if id(src) in seen or isinstance(src, PlaceHolderNode) or _stabilized(src):
                    continue
                seen.add(id(src))
                stack.append(src)
    return None


def score_matmuls(graph) -> list:
    """The matmuls whose result reaches an ``Exp`` without crossing another matmul.

    In a graph whose softmax is exact (some ``Exp`` carries a ``Stabilize`` record) only the
    stabilized ones count: the logits of a plain ``exp`` are not softmax logits.
    """
    stack = [n for n in graph.nodes if isinstance(n, Exp)]
    if any(_stabilized(n) for n in stack):
        stack = [n for n in stack if _stabilized(n)]
    return _score_matmuls_of(stack)


def key_operand(graph, matmul) -> int:
    """Which operand of a score matmul has the softmax axis for its rows: 1, the second
    (attention over the keys), unless the stabilized ``Exp`` the scores feed says the
    softmax runs over the first operand's rows.

    The scores are ``(batch.., M, N)`` with ``M`` the rows of the first operand and ``N``
    those of the second. The softmax axis is read off the ``Exp``'s record where the
    ``Exp`` reads the scores through constant scalings only, which keep the axes in
    place. Anywhere else -- no record, another op on the way, another shape -- it is
    taken to be the last axis, as for a graph that makes no claim.
    """
    def by_axis(exp, recorded: bool):
        """The operand the softmax axis of ``exp`` names; ``recorded``: whether the axis is
        to come from the exp's record (it reads the scores) or from its structure."""
        if recorded != (getattr(exp, "stabilize", None) is not None):
            return 1
        axis, rank = softmax_axis(exp), len(exp.get_shape(0))
        if axis is None or tuple(exp.get_shape(0)) != tuple(matmul.get_shape(0)):
            return 1
        return 0 if axis % rank == rank - 2 else 1

    node, seen = matmul, set()
    while id(node) not in seen:
        seen.add(id(node))
        readers = [edge.dst_node for edge in node.outputs]
        # The explicit form, ``exp(s - max(s))``: the max reads the scores next to the
        # subtraction it feeds, which is one path, and the exp that reads the subtraction
        # is a softmax by structure (``softmax_axis``), not by a record. Whoever else reads
        # the max (a split partial returns it) does not move the axis it was taken over.
        shifted = [r for r in readers if _explicit_max(r) is not None]
        if len(shifted) == 1 and all(
            r is shifted[0] or r is _explicit_max(shifted[0]) for r in readers
        ):
            exps = [edge.dst_node for edge in shifted[0].outputs]
            if len(exps) != 1 or type(exps[0]) is not Exp:
                return 1
            return by_axis(exps[0], recorded=False)
        if len(readers) != 1:
            return 1
        (node,) = readers
        if type(node) is Exp:
            return by_axis(node, recorded=True)
        if not (type(node) in (Mul, Div) and node.const is not None
                and not getattr(node, "const_on_left", False)):
            return 1
    return 1


def _score_matmuls_of(exps) -> list:
    found, seen, stack = [], set(), list(exps)
    while stack:
        for edge in stack.pop().inputs:
            src = edge.src_node
            if id(src) in seen or isinstance(src, PlaceHolderNode):
                continue
            seen.add(id(src))
            if isinstance(src, Matmul):
                found.append(src)
            else:
                stack.append(src)
    return found


def logit_inputs(graph) -> list:
    """``(placeholder, softmax axis, sign)`` for every stabilized ``Exp`` whose logits are an
    input: it reads a placeholder, directly or through constant scalings (``sign`` is
    theirs) and views that only reindex it -- the partial of a program split over its
    softmax axis reads its logits through the split. The softmax axis is the record's,
    which only a stabilized ``Exp`` has (a plain ``exp(x)`` does not say what it is summed
    over), given as the axis of the placeholder one step along it moves along.
    """
    found = []
    for node in graph.nodes:
        softmax = softmax_axis(node) if type(node) is Exp else None
        if softmax is None:
            continue
        src, sign, views = _read(node)
        if not isinstance(src, PlaceHolderNode) or any(src is p for p, _, _ in found):
            continue
        axis = _moved(_viewed(src, views[::-1]), softmax, src)
        if axis is not None:
            found.append((src, axis, sign))
    return found


def _read(exp):
    """``(node, sign, views)``: what ``exp`` reads behind the views that only reindex it and
    the constant scalings (``sign`` is theirs); ``views`` from the ``Exp`` inwards."""
    src, sign, views = _logits_of(exp), 1.0, []
    while True:
        if isinstance(src, (SplitView, PermuteView)):
            views.append(src)
        elif type(src) in (Mul, Div) and src.const is not None \
                and not getattr(src, "const_on_left", False):
            sign = -sign if float(src.const) < 0 else sign
        else:
            return src, sign, views
        src = src.inputs[0].src_node


def _viewed(placeholder, views):
    """The flat index of every element of ``placeholder`` where the reindexing ``views``
    (from the placeholder outwards) put it."""
    import torch

    shape = tuple(int(s) for s in placeholder.get_shape(0))
    flat = torch.arange(int(torch.tensor(shape).prod())).reshape(shape)
    for view in views:
        if isinstance(view, PermuteView):
            flat = flat.permute(*view.perm)
        else:  # SplitView: the split axis is created before ``axis`` and moved to split_pos
            axis = view.axis
            dims = list(flat.shape)
            flat = flat.reshape(dims[:axis] + [view.factor, view.inner_extent] + dims[axis + 1:])
            flat = flat.movedim(axis, view.split_pos)
    return flat


def _moved(flat, axis, placeholder):
    """The axis of ``placeholder`` that one step along ``axis`` of ``flat`` (its elements
    as some value arranges them, ``_viewed``) moves along; ``None`` where that axis holds
    one element, ``SpecError`` if the step is not along one axis of the input."""
    import torch

    if flat.shape[axis] < 2:
        return None
    shape = tuple(int(s) for s in placeholder.get_shape(0))
    origin = [0] * flat.dim()
    step = list(origin)
    step[axis] = 1
    a = torch.unravel_index(flat[tuple(origin)], shape)
    b = torch.unravel_index(flat[tuple(step)], shape)
    axes = [i for i, (x, y) in enumerate(zip(a, b)) if int(x) != int(y)]
    if len(axes) != 1:
        raise SpecError(f"cannot map axis {axis} of a value back to its input "
                        f"{placeholder.name!r}")
    return axes[0]


def _operand(matmul, dst_id):
    """``(placeholder, views, sign)`` behind one operand of ``matmul``.

    ``views`` are the reindexing nodes from the placeholder to the operand, in that order;
    ``sign`` is that of the constant scalings in between.
    """
    edge = next(e for e in matmul.inputs if e.dst_id == dst_id)
    node, views, sign = edge.src_node, [], 1.0
    while not isinstance(node, PlaceHolderNode):
        if isinstance(node, (SplitView, PermuteView)):
            views.append(node)
        elif type(node) in (Mul, Div) and node.const is not None \
                and not getattr(node, "const_on_left", False):
            sign = -sign if float(node.const) < 0 else sign
        else:
            raise SpecError(
                f"{matmul.name}: operand {dst_id} of this score matmul is computed by "
                f"{type(node).__name__} {node.name!r}, not read from an input; the adversarial "
                "input distributions only shape plain inputs"
            )
        node = node.inputs[0].src_node
    return node, views[::-1], sign


def _operand_axes(matmul, dst_id, placeholder, views):
    """``(row axis | None, contraction axis | None)`` of ``placeholder``: the axes one step
    along the operand's row (M for A, N for B) and contracted (K) axes moves along."""
    flat = _viewed(placeholder, views)
    if dst_id == 0:
        batch, transposed, rows = matmul.input_batch_dims_A, matmul.transpose_A, matmul.M
    else:  # B is (K, N) unless transposed
        batch, transposed, rows = matmul.input_batch_dims_B, not matmul.transpose_B, matmul.N
    tail = (matmul.K, rows) if transposed else (rows, matmul.K)
    flat = flat.reshape(*batch, *tail)
    row_axis, contracted_axis = (-1, -2) if transposed else (-2, -1)

    def moved(operand_axis):
        try:
            return _moved(flat, operand_axis, placeholder)
        except SpecError:
            raise SpecError(
                f"{matmul.name}: cannot map operand {dst_id} back to its input") from None

    return moved(row_axis), moved(contracted_axis)


def apply_distribution(graph, inputs, distribution):
    """``inputs`` (in :func:`placeholder_order`) rewritten to an adversarial distribution."""
    import torch

    (distribution,) = check_distributions(distribution)
    if distribution == "normal":
        return tuple(inputs)
    scores, logits = score_matmuls(graph), logit_inputs(graph)
    if not scores and not logits:
        raise SpecError(
            f"the {distribution!r} inputs shape softmax logits, but this graph has no score "
            "matmul (a Matmul feeding an Exp) and no stabilized Exp that reads an input"
        )
    placeholders = placeholder_order(graph)
    index = {id(p): i for i, p in enumerate(placeholders)}
    out = list(inputs)
    for node, axis, sign in logits:
        out[index[id(node)]] = _logits(inputs[index[id(node)]], axis, sign, distribution)
    if not scores:
        return tuple(out)
    # per input: the matmul, the placeholder, its views, [the sign,] which operand it is
    queries, keys = {}, {}
    for matmul in scores:
        k_id = key_operand(graph, matmul)
        q_node, q_views, q_sign = _operand(matmul, 1 - k_id)
        k_node, k_views, k_sign = _operand(matmul, k_id)
        queries[index[id(q_node)]] = (matmul, q_node, q_views, 1 - k_id)
        keys[index[id(k_node)]] = (matmul, k_node, k_views, q_sign * k_sign, k_id)

    if distribution == "large_logits":
        for i in queries:
            out[i] = (inputs[i].float() * LARGE_LOGIT_FACTOR).to(inputs[i].dtype)
        return tuple(out)

    both = sorted(set(queries) & set(keys))
    if both:
        raise SpecError(
            f"input {placeholders[both[0]].name!r} is both a query and a key of the scores; "
            f"the {distribution!r} inputs need them to be separate inputs"
        )
    if distribution == "constant":
        for i, (matmul, node, views, _, k_id) in keys.items():
            key_axis, _ = _operand_axes(matmul, k_id, node, views)
            if key_axis is not None:  # the first key, for every key
                out[i] = inputs[i].narrow(key_axis, 0, 1).expand_as(inputs[i]).contiguous()
        return tuple(out)
    if distribution in ("last_spike", "offset"):
        shape = _spike if distribution == "last_spike" else _offset
        for i, (matmul, node, views, q_id) in queries.items():
            _, contracted = _operand_axes(matmul, q_id, node, views)
            out[i] = shape(inputs[i], contracted, matmul.K)
        for i, (matmul, node, views, sign, k_id) in keys.items():
            key_axis, contracted = _operand_axes(matmul, k_id, node, views)
            out[i] = shape(inputs[i], contracted, matmul.K, key=(key_axis, sign))
        return tuple(out)

    for i, (matmul, node, views, q_id) in queries.items():
        _, contracted = _operand_axes(matmul, q_id, node, views)
        x = inputs[i].float()
        # One positive magnitude per query row, constant along the contracted axis.
        magnitude = x.abs() + 0.5
        if contracted is not None:
            magnitude = magnitude.narrow(contracted, 0, 1)
        out[i] = (magnitude / matmul.K ** 0.5).expand_as(x).to(inputs[i].dtype).contiguous()
    for i, (matmul, node, views, sign, k_id) in keys.items():
        key_axis, _ = _operand_axes(matmul, k_id, node, views)
        x = inputs[i]
        shape = [1] * x.dim()
        ramp = torch.zeros(1, device=x.device)
        if key_axis is not None:
            n = x.shape[key_axis]
            shape[key_axis] = n
            ramp = torch.arange(n, device=x.device, dtype=torch.float32) / (n - 1)
            if distribution == "decreasing":
                ramp = ramp.flip(0)
        out[i] = (sign * INCREASING_SPAN * ramp).reshape(shape).expand_as(x).to(x.dtype).contiguous()
    return tuple(out)


def _logits(x, axis, sign, distribution):
    """A logit input in one of the adversarial cases: the rows the score operands are shaped
    to produce, written along ``axis`` directly. ``sign`` is that of the constant scalings
    between the input and the ``Exp``."""
    import torch

    xf, n = x.float(), x.shape[axis]
    first = xf.narrow(axis, 0, 1)
    top = sign * (first.abs() + 0.5) * INCREASING_SPAN  # a * 128, a >= 0.5, per row
    if distribution == "large_logits":
        out = xf * LARGE_LOGIT_FACTOR
    elif distribution in ("increasing", "decreasing"):
        shape = [1] * x.dim()
        shape[axis] = n
        ramp = torch.arange(n, device=x.device, dtype=torch.float32) / max(n - 1, 1)
        if distribution == "decreasing":
            ramp = ramp.flip(0)
        out = top * ramp.reshape(shape)
    elif distribution == "last_spike":
        out = xf.clone()
        out.narrow(axis, n - 1, 1).copy_(top)
    elif distribution == "constant":
        out = first.expand_as(xf)
    else:  # offset: the input's own rounding puts the normal-sized term on its grid
        out = xf - sign * LOGIT_OFFSET
    return out.to(x.dtype).contiguous()


def _first(x, contracted):
    """The first coordinate of ``x`` along the contraction (a view)."""
    return x if contracted is None else x.narrow(contracted, 0, 1)


def _spike(x, contracted, width, key=None):
    """``last_spike``: a query's first coordinate along the contraction becomes a positive
    magnitude ``|q_0| + 0.5``; the last key becomes ``INCREASING_SPAN * sqrt(K)`` there and
    zero elsewhere. It then scores ``magnitude * 128`` (at least 64) after the 1/sqrt(K)
    scaling, while every other key scores as a normal key does."""
    out = x.float().clone()
    if key is None:
        first = _first(out, contracted)
        first.copy_(first.abs() + 0.5)
        return out.to(x.dtype)
    key_axis, sign = key
    last = out if key_axis is None else out.narrow(key_axis, out.shape[key_axis] - 1, 1)
    last.zero_()
    _first(last, contracted).fill_(sign * INCREASING_SPAN * width ** 0.5)
    return out.to(x.dtype)


def _offset(x, contracted, width, key=None):
    """``offset``: every coordinate on the ``OFFSET_GRID``; the first one along the
    contraction is ``OFFSET_QUERY`` for a query and, for every key, the grid value whose
    product with it is ``-OFFSET * sqrt(K)``."""
    import torch

    out = torch.round(x.float() / OFFSET_GRID) * OFFSET_GRID
    value = OFFSET_QUERY
    if key is not None:
        value = -key[1] * round(OFFSET * width ** 0.5 / OFFSET_QUERY / OFFSET_GRID) * OFFSET_GRID
    _first(out, contracted).fill_(value)
    return out.to(x.dtype)
