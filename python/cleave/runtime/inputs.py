"""Input distributions a kernel can be validated on: ``normal`` and the adversarial softmax cases.

``normal`` is what :func:`cleave.runtime.reference.make_inputs` samples. The adversarial ones
rewrite that sample afterwards (so the normal stream never shifts), and only the operands of
the *score* matmuls: the matmuls an ``Exp`` is reached from without crossing another matmul.

``large_logits``  the query operand times 100, so every logit is 100 times larger;
``increasing``    logits that grow along the key axis for every query row, by more than
                  fp32 ``exp`` can hold, so a running max changes at every tile.

Each is requested explicitly (``EvalConfig.distributions`` for a program,
``make_inputs(..., distribution=...)`` for a graph). A graph with no score matmul has no
logits to shape, and asking for an adversarial distribution on it is an error.
"""

from __future__ import annotations

from ..core.errors import SpecError
from ..ir.graph import PlaceHolderNode
from ..ir.ops import Div, Exp, Matmul, Mul, PermuteView, SplitView
from .reference import placeholder_order

DISTRIBUTIONS = ("normal", "large_logits", "increasing")

LARGE_LOGIT_FACTOR = 100.0
# ``increasing``: q.k ramps from 0 to ``a * INCREASING_SPAN * sqrt(K)`` along the keys (``K`` the
# contracted width, ``a`` in [0.5, ...) per query row), i.e. over ``a * 128`` after the usual
# 1/sqrt(K) scaling -- past ln(fp32 max) = 88.7.
INCREASING_SPAN = 128.0


def check_distributions(distributions) -> tuple:
    """``distributions`` as a tuple of known names (at least one)."""
    names = (distributions,) if isinstance(distributions, str) else tuple(distributions)
    unknown = [n for n in names if n not in DISTRIBUTIONS]
    if not names or unknown:
        raise SpecError(
            f"input distributions must be drawn from {DISTRIBUTIONS}, got {distributions!r}"
        )
    return names


def score_matmuls(graph) -> list:
    """The matmuls whose result reaches an ``Exp`` without crossing another matmul."""
    found, seen = [], set()
    stack = [n for n in graph.nodes if isinstance(n, Exp)]
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
    if dst_id == 0:
        batch, transposed, rows = matmul.input_batch_dims_A, matmul.transpose_A, matmul.M
    else:  # B is (K, N) unless transposed
        batch, transposed, rows = matmul.input_batch_dims_B, not matmul.transpose_B, matmul.N
    tail = (matmul.K, rows) if transposed else (rows, matmul.K)
    flat = flat.reshape(*batch, *tail)
    row_axis, contracted_axis = (-1, -2) if transposed else (-2, -1)

    def moved(operand_axis):
        if flat.shape[operand_axis] < 2:
            return None
        origin = [0] * flat.dim()
        step = list(origin)
        step[operand_axis] = 1
        a = torch.unravel_index(flat[tuple(origin)], shape)
        b = torch.unravel_index(flat[tuple(step)], shape)
        axes = [i for i, (x, y) in enumerate(zip(a, b)) if int(x) != int(y)]
        if len(axes) != 1:
            raise SpecError(f"{matmul.name}: cannot map operand {dst_id} back to its input")
        return axes[0]

    return moved(row_axis), moved(contracted_axis)


def apply_distribution(graph, inputs, distribution):
    """``inputs`` (in :func:`placeholder_order`) rewritten to an adversarial distribution."""
    import torch

    (distribution,) = check_distributions(distribution)
    if distribution == "normal":
        return tuple(inputs)
    scores = score_matmuls(graph)
    if not scores:
        raise SpecError(
            f"the {distribution!r} inputs shape softmax logits, but this graph has no score "
            "matmul (a Matmul feeding an Exp)"
        )
    placeholders = placeholder_order(graph)
    index = {id(p): i for i, p in enumerate(placeholders)}
    out = list(inputs)
    queries, keys = {}, {}
    for matmul in scores:
        q_node, q_views, q_sign = _operand(matmul, 0)
        k_node, k_views, k_sign = _operand(matmul, 1)
        queries[index[id(q_node)]] = (matmul, q_node, q_views)
        keys[index[id(k_node)]] = (matmul, k_node, k_views, q_sign * k_sign)

    if distribution == "large_logits":
        for i in queries:
            out[i] = (inputs[i].float() * LARGE_LOGIT_FACTOR).to(inputs[i].dtype)
        return tuple(out)

    both = sorted(set(queries) & set(keys))
    if both:
        raise SpecError(
            f"input {placeholders[both[0]].name!r} is both a query and a key of the scores; "
            "the 'increasing' inputs need them to be separate inputs"
        )
    for i, (matmul, node, views) in queries.items():
        _, contracted = _operand_axes(matmul, 0, node, views)
        x = inputs[i].float()
        # One positive magnitude per query row, constant along the contracted axis.
        magnitude = x.abs() + 0.5
        if contracted is not None:
            magnitude = magnitude.narrow(contracted, 0, 1)
        out[i] = (magnitude / matmul.K ** 0.5).expand_as(x).to(inputs[i].dtype).contiguous()
    for i, (matmul, node, views, sign) in keys.items():
        key_axis, _ = _operand_axes(matmul, 1, node, views)
        x = inputs[i]
        shape = [1] * x.dim()
        ramp = torch.zeros(1, device=x.device)
        if key_axis is not None:
            n = x.shape[key_axis]
            shape[key_axis] = n
            ramp = torch.arange(n, device=x.device, dtype=torch.float32) / (n - 1)
        out[i] = (sign * INCREASING_SPAN * ramp).reshape(shape).expand_as(x).to(x.dtype).contiguous()
    return tuple(out)
