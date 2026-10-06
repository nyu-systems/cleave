"""Torch reference execution for cleave graphs.

The correctness of a compiled candidate is checked by comparing its output against
a reference computed directly from the *input* cleave :class:`~cleave.graph.Graph`
that the superoptimizer and scheduler operate on. Walking that graph node by node
in torch removes the need for a per-op ``op_name`` dispatch table: every operator
in the fixed cleave vocabulary knows its own shape/dtype semantics, so the reference
is whatever the graph says it is -- byte-for-byte the same computation the generated
kernel implements.

``torch`` is imported lazily inside the functions so importing this module (and
therefore ``cleave.util``) never pulls torch into TVM-only codegen paths.
"""

from __future__ import annotations

import hashlib
import operator
import os

from ..graph import OutputNode, PlaceHolderNode
from ..ops import (
    Add,
    CausalMaskedExp,
    Div,
    Exp,
    IndexGather,
    Log2,
    MaskedExp,
    Matmul,
    Mul,
    PagedGather,
    PermuteView,
    RangeGather,
    Reduction,
    RowGather,
    RSqrt,
    Silu,
    SplitView,
    Sqrt,
    Sub,
)
from .graph_utils import find_topo_sort


# Cleave dtype strings -> torch dtype attribute names.
_TORCH_DTYPE_NAME = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float": "float32",
    "float32": "float32",
    "float64": "float64",
    "int8": "int8",
    "int32": "int32",
    "int64": "int64",
    "bool": "bool",
}

_BINARY_OP = {Add: operator.add, Sub: operator.sub, Mul: operator.mul, Div: operator.truediv}


def torch_dtype(dtype):
    import torch

    return getattr(torch, _TORCH_DTYPE_NAME[str(dtype)])


def placeholder_order(graph):
    """Placeholder nodes in the order the generated kernel receives them.

    Mirrors :func:`cleave.passes.annotate_args`, which appends placeholders in
    ``find_topo_sort`` order followed by outputs, so ``kernel(*inputs)`` consumes
    inputs in exactly this order.
    """
    return [n for n in find_topo_sort(graph.nodes) if isinstance(n, PlaceHolderNode)]


def output_order(graph):
    """Output nodes in the order the kernel returns them (matches ``out_idx``)."""
    return [n for n in find_topo_sort(graph.nodes) if isinstance(n, OutputNode)]


def graph_out_indices(graph):
    """``out_idx`` for ``tl.compile``: the trailing ``n_out`` kernel arguments."""
    n_out = len(output_order(graph))
    if n_out == 0:
        raise ValueError("Graph has no output nodes; cannot derive out_idx.")
    return list(range(-n_out, 0))


def fingerprint_seed(graph):
    """Deterministic RNG seed derived from the graph's structure.

    An explicit ``CLEAVE_TEST_SEED`` env var overrides the derived value. The
    fingerprint keeps inputs reproducible per problem and distinct across problems.
    """
    override = os.getenv("CLEAVE_TEST_SEED")
    if override is not None:
        return int(override) & 0xFFFFFFFF

    parts = []
    for node in find_topo_sort(graph.nodes):
        parts.append(f"{type(node).__name__}:{node.name}")
        for output_id in range(max(1, node.num_outputs())):
            try:
                shape = tuple(int(s) for s in node.get_shape(output_id))
                parts.append(f"{shape}:{node.get_dtype(output_id)}")
            except Exception:
                pass
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _positive_placeholders(graph):
    """Placeholders that must be sampled strictly positive to avoid nan/inf.

    A value must be positive if it feeds ``Sqrt``/``RSqrt`` or a tensor ``Div``
    denominator. The requirement propagates backward only through ops whose output
    positivity *requires* positive inputs -- sum-``Reduction``, the bijective views
    (``SplitView``/``PermuteView``), elementwise ``Add``/``Mul`` of tensors, and a
    ``Div`` by a positive constant (``x / c`` with ``c > 0``), which preserves sign.

    It must NOT propagate through sign-breaking or unconditionally-positive ops
    (``Matmul``, ``Exp``, ``Silu``, ``Sub``, ``Sqrt``/``RSqrt``, and the
    sign-breaking ``Div`` forms ``c / x`` and tensor ``x / y``): e.g. ``exp`` is
    always positive regardless of its input, so marking its ancestors positive would
    wrongly force a whole attention score path positive and blow up ``exp``.
    Propagation therefore stops at those boundaries.
    """
    # Nodes whose positive output requires their tensor input(s) to be positive.
    def _propagates(node):
        if isinstance(node, (Reduction, SplitView, PermuteView)):
            return True  # sum of positives / bijective view preserves positivity
        if isinstance(node, Add):
            # tensor+tensor: both must be positive; +positive const: input positive.
            if node.const is None:
                return True
            return float(node.const) >= 0
        if isinstance(node, Mul):
            if node.const is not None:
                return float(node.const) > 0  # scale by positive const
            # A self-square Mul(A, A) is non-negative for any sign of A, so it does
            # NOT require A positive; a genuine tensor*tensor of positives does.
            in_edges = node.inputs
            if (
                len(in_edges) == 2
                and in_edges[0].src_node is in_edges[1].src_node
                and in_edges[0].src_id == in_edges[1].src_id
            ):
                return False
            return True
        if isinstance(node, Div):
            # ``x / c`` with a positive constant on the right preserves the sign of
            # ``x``, so a positive output requires a positive input. The other Div
            # forms are sign-breaking and stop propagation: ``c / x`` (const on the
            # left) and tensor ``x / y`` (there the denominator, not the numerator,
            # is constrained -- handled by the seeding loop below, not here).
            return (
                node.const is not None
                and not node.const_on_left
                and float(node.const) > 0
            )
        return False

    positive = set()
    stack = []

    def _require_positive(src_node):
        if isinstance(src_node, PlaceHolderNode):
            positive.add(src_node)
        else:
            stack.append(src_node)

    for node in graph.nodes:
        if isinstance(node, (Sqrt, RSqrt)):
            _require_positive(node.inputs[0].src_node)
        elif isinstance(node, Div) and node.const is None:
            _require_positive(node.inputs[1].src_node)
        elif isinstance(node, Div) and node.const_on_left:
            _require_positive(node.inputs[0].src_node)

    visited = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if not _propagates(node):
            continue
        for edge in node.inputs:
            _require_positive(edge.src_node)

    return positive


def _eval_matmul(node, args, out_dtype):
    a, b = args[0].to(out_dtype), args[1].to(out_dtype)
    # A physical operand may drop size-1 batch dims (e.g. a (B*H, S, D) input feeds
    # a matmul declaring input_batch_dims=(B*H, 1)). Reshape to the declared operand
    # shape so batch broadcasting lines up, exactly as the kernel's buffer view does.
    m, n, k = node.M, node.N, node.K
    a_shape = (
        (*node.input_batch_dims_A, k, m)
        if node.transpose_A
        else (*node.input_batch_dims_A, m, k)
    )
    b_shape = (
        (*node.input_batch_dims_B, n, k)
        if node.transpose_B
        else (*node.input_batch_dims_B, k, n)
    )
    a = a.reshape(a_shape)
    b = b.reshape(b_shape)
    if node.transpose_A:
        a = a.transpose(-2, -1)
    if node.transpose_B:
        b = b.transpose(-2, -1)
    a = _broadcast_batch(a, node.batch_dims, node.input_batch_dims_A)
    b = _broadcast_batch(b, node.batch_dims, node.input_batch_dims_B)
    return a @ b


def _broadcast_batch(tensor, out_batch, in_batch):
    # `_map_batch_indices` maps out batch index -> floordiv(idx, out//in), i.e.
    # grouped repetition (repeat_interleave), NOT tiled broadcast (expand).
    for dim, (out_extent, in_extent) in enumerate(zip(out_batch, in_batch)):
        if out_extent != in_extent:
            tensor = tensor.repeat_interleave(out_extent // in_extent, dim=dim)
    return tensor


def _eval_binary(node, args, out_dtype):
    a = args[0].to(out_dtype)
    if node.const is not None:
        const = float(node.const)
        if isinstance(node, Div) and node.const_on_left:
            return const / a
        return _BINARY_OP[type(node)](a, const)

    b = args[1].to(out_dtype)
    if node.broadcast == 0:
        b = b.unsqueeze(-1)  # B: (*batch, M) broadcast across the last axis N
    elif node.broadcast == 1:
        b = b.unsqueeze(-2)  # B: (*batch, N) broadcast across M
    return _BINARY_OP[type(node)](a, b)


def _eval_splitview(node, tensor):
    tensor = tensor.to(torch_dtype(node.get_dtype(0)))
    axis, factor, inner = node.axis, node.factor, node.inner_extent
    shape = list(tensor.shape)
    tensor = tensor.reshape(
        tuple(shape[:axis]) + (factor, inner) + tuple(shape[axis + 1 :])
    )
    # The split (factor) axis is created just before `axis` and moved to split_pos.
    return tensor.movedim(axis, node.split_pos)


def _eval_node(node, args):
    import torch

    # SplitView is a pure view (no compute dtype); it handles its own dtype.
    if isinstance(node, SplitView):
        return _eval_splitview(node, args[0])
    if isinstance(node, PermuteView):
        return (
            args[0]
            .to(torch_dtype(node.get_dtype(0)))
            .permute(*node.perm)
            .contiguous()
        )

    # Each op stores its inputs in a buffer of node.dtype (the prim func declares
    # placeholders at in_dtype, then upcasts to out_dtype inside the compute). Model
    # that storage precision so e.g. an fp16 matmul operand fed by an fp32 producer
    # is rounded exactly as the kernel rounds it.
    in_dtype = torch_dtype(node.dtype)
    args = [a.to(in_dtype) for a in args]

    out_dtype = torch_dtype(node.out_dtype)

    if isinstance(node, Matmul):
        return _eval_matmul(node, args, out_dtype)
    if isinstance(node, Reduction):
        return args[0].to(out_dtype).sum(dim=node.axis)
    if type(node) in _BINARY_OP:
        return _eval_binary(node, args, out_dtype)
    if isinstance(node, Exp):
        return torch.exp(args[0].to(out_dtype))
    if isinstance(node, Silu):
        x = args[0].to(out_dtype)
        return x * torch.sigmoid(x)
    if isinstance(node, Sqrt):
        return torch.sqrt(args[0].to(out_dtype))
    if isinstance(node, RSqrt):
        return torch.rsqrt(args[0].to(out_dtype))
    if isinstance(node, Log2):
        return torch.log2(args[0].to(out_dtype))
    if isinstance(
        node,
        (PagedGather, IndexGather, RangeGather, RowGather, MaskedExp, CausalMaskedExp),
    ):
        raise NotImplementedError(
            f"graph_executor has no torch reference for {type(node).__name__} "
            f"(node {node.name!r}). These ops carry their semantics in gen_macro, "
            "parameterized by schedule scalars (tile_base/valid_len/base/head/row0) "
            "that a pre-tiling source graph does not have, so paged/ragged attention "
            "is not yet expressible as a source graph."
        )
    raise TypeError(
        f"graph_executor has no torch reference for node type {type(node).__name__}"
    )


def make_inputs(graph, dtype, *, seed=None, device="cuda"):
    """Sample the graph's placeholder inputs (in kernel argument order)."""
    import torch

    if seed is None:
        seed = fingerprint_seed(graph)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base_dtype = torch_dtype(dtype)
    positive = _positive_placeholders(graph)
    inputs = []
    for placeholder in placeholder_order(graph):
        shape = tuple(int(s) for s in placeholder.get_shape(0))
        dt = _placeholder_torch_dtype(placeholder, base_dtype)
        if placeholder in positive:
            tensor = torch.rand(shape, device=device) + 0.5
        else:
            tensor = torch.randn(shape, device=device)
        inputs.append(tensor.to(dt))
    return tuple(inputs)


def _placeholder_torch_dtype(placeholder, base_dtype):
    """A placeholder's torch dtype, falling back to ``base_dtype`` if unset.

    ``Node.get_dtype`` raises ``IndexError`` (not ``None``) when the dtype slot was
    never populated, so guard for that too.
    """
    try:
        ph_dtype = placeholder.get_dtype(0)
    except IndexError:
        ph_dtype = None
    return torch_dtype(ph_dtype) if ph_dtype is not None else base_dtype


def execute(graph, inputs):
    """Compute the graph's reference outputs from concrete input tensors.

    ``inputs`` are matched positionally to :func:`placeholder_order`. Returns a list
    with one tensor per output node, in :func:`output_order` (== kernel ``out_idx``).
    """
    order = find_topo_sort(graph.nodes)
    placeholders = [n for n in order if isinstance(n, PlaceHolderNode)]
    if len(inputs) != len(placeholders):
        raise ValueError(
            f"Expected {len(placeholders)} inputs, got {len(inputs)} for graph execution."
        )

    env = {}
    for placeholder, tensor in zip(placeholders, inputs):
        env[(placeholder, 0)] = tensor

    outputs = []
    for node in order:
        if isinstance(node, PlaceHolderNode):
            continue
        if isinstance(node, OutputNode):
            edge = node.inputs[0]
            value = env[(edge.src_node, edge.src_id)]
            outputs.append(value.to(torch_dtype(node.get_dtype(0))))
            continue
        args = [
            env[(edge.src_node, edge.src_id)]
            for edge in sorted(node.inputs, key=lambda e: e.dst_id)
        ]
        env[(node, 0)] = _eval_node(node, args)
    return outputs


def reconcile_output_layout(reference, actual_shape):
    """Return ``reference`` reshaped/permuted to the kernel's physical output shape.

    The executor emits an output in its *logical* axis order, but a scheduled kernel
    may store it differently: it can keep or drop size-1 axes (e.g. a split-partial
    rowsum stored ``(split_k, M, 1)``) and, for split-partial outputs, move the split
    axis to the front (``(split_k, B*H, S_Q, ...)`` vs logical ``(B*H, split_k, ...)``).
    Both are pure reindexings recovered by matching non-unit axis extents.

    The extent-matching is only well defined when the non-unit extents are distinct.
    If two non-unit axes share an extent AND a non-identity permutation is required
    to line up with ``actual_shape`` (e.g. a split-partial output where ``split_k ==
    B*H``), the layout is ambiguous and we raise rather than silently risk a wrong
    permutation masking a real mismatch.
    """
    import torch

    ref_shape = tuple(int(s) for s in reference.shape)
    actual_shape = tuple(int(s) for s in actual_shape)
    if ref_shape == actual_shape:
        return reference
    if reference.numel() != int(torch.tensor(actual_shape).prod().item()):
        return reference

    # Drop the size-1 axes each side inserts independently, then line up the
    # remaining (non-unit) axes.
    ref_non_unit = [d for d in ref_shape if d != 1]
    actual_non_unit = [d for d in actual_shape if d != 1]
    if sorted(ref_non_unit) != sorted(actual_non_unit):
        # Different non-unit multisets -> not a reindexing; a bare reshape is the
        # only interpretation (and assert_close will catch a genuine mismatch).
        return reference.reshape(actual_shape)

    squeezed = reference.reshape(ref_non_unit) if ref_non_unit else reference.reshape(())
    if ref_non_unit != actual_non_unit:
        if len(set(ref_non_unit)) != len(ref_non_unit):
            raise ValueError(
                "Ambiguous output layout reconciliation: non-unit extents "
                f"{ref_non_unit} (reference) -> {actual_non_unit} (kernel) require a "
                "permutation but contain duplicate extents, so the axis mapping is "
                "undetermined. Emit the reference in the kernel's physical layout."
            )
        perm = [ref_non_unit.index(d) for d in actual_non_unit]
        squeezed = squeezed.permute(*perm)
    return squeezed.reshape(actual_shape)


def build_inputs_and_reference(graph, dtype, *, seed=None, device="cuda"):
    """Return ``(inputs_by_name, reference, out_idx)`` for a cleave graph.

    ``inputs_by_name`` maps each placeholder name to its tensor. Scheduling can
    reorder the kernel's parameter list relative to the flat graph's topo order, so
    inputs are keyed by name and ordered against the compiled kernel's actual
    parameter names at call time (see :func:`order_inputs_for_kernel`). ``reference``
    is a list with one tensor per output node and ``out_idx`` is the matching
    ``tl.compile`` index.
    """
    placeholders = placeholder_order(graph)
    tensors = make_inputs(graph, dtype, seed=seed, device=device)
    inputs_by_name = {ph.name: t for ph, t in zip(placeholders, tensors)}
    reference = execute(graph, tensors)
    return inputs_by_name, reference, graph_out_indices(graph)


def order_inputs_for_kernel(inputs_by_name, kernel):
    """Order the name-keyed inputs to match a compiled kernel's parameter list.

    ``inputs_by_name`` is an ordered mapping (flat placeholder order) of placeholder
    name -> tensor. Cleave-generated kernels name each input buffer after its
    placeholder, so scheduling reorders are resolved by matching those names. Kernels
    whose input buffers are NOT named after the graph's placeholders (e.g. the
    hand-written reference/handwritten prim funcs, which use ``A``/``B``/``Q``/...)
    fall back to the graph's placeholder order positionally -- the source graphs are
    built to declare placeholders in the kernel's parameter order.
    """
    adapter = kernel.adapter
    prim_func = adapter.prim_func
    buffer_map = prim_func.buffer_map
    result_idx = set(adapter.result_idx)

    kernel_input_names = [
        buffer_map[param].name
        for param_idx, param in enumerate(prim_func.params)
        if param_idx not in result_idx and param in buffer_map
    ]
    if len(kernel_input_names) != len(inputs_by_name):
        raise ValueError(
            f"Kernel consumes {len(kernel_input_names)} inputs but graph produced "
            f"{len(inputs_by_name)} placeholders."
        )

    if all(name in inputs_by_name for name in kernel_input_names):
        return [inputs_by_name[name] for name in kernel_input_names]

    # Hand-written kernel with non-placeholder buffer names: use positional order.
    return list(inputs_by_name.values())
