"""Torch reference execution for cleave graphs.

The correctness of a compiled candidate is checked by comparing its output against
a reference computed directly from the *input* cleave :class:`~cleave.ir.graph.Graph`
that the superoptimizer and scheduler operate on. Walking that graph node by node
in torch removes the need for a per-op ``op_name`` dispatch table: every operator
in the fixed cleave vocabulary knows its own shape/dtype semantics, so the reference
is whatever the graph says it is -- byte-for-byte the same computation the generated
kernel implements.

``torch`` is imported lazily inside the functions so importing this module (and
therefore ``cleave.runtime``) never pulls torch into TVM-only codegen paths.
"""

from __future__ import annotations

import hashlib

from ..core.settings import Settings
from ..ir.graph import OutputNode, PlaceHolderNode
from ..ir.ops import (
    Accumulate,
    AccumulateMax,
    Add,
    Copy,
    GridFloor,
    Div,
    Exp,
    Exp2,
    Log2,
    Lse2,
    Matmul,
    ReduceMax,
    Mul,
    PermuteView,
    ReduceSum,
    RSqrt,
    Silu,
    SplitView,
    Sqrt,
    Sub,
)
from ..ir.ops.registry import NO_REFERENCE
from ..ir.graph_utils import find_topo_sort


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


def torch_dtype(dtype):
    import torch

    return getattr(torch, _TORCH_DTYPE_NAME[str(dtype)])


def placeholder_order(graph):
    """Placeholder nodes in the order the generated kernel receives them.

    Mirrors :func:`cleave.codegen.passes.annotate_args`, which appends placeholders in
    ``find_topo_sort`` order followed by outputs, so ``kernel(*inputs)`` consumes
    inputs in exactly this order.
    """
    return [n for n in find_topo_sort(graph.nodes) if isinstance(n, PlaceHolderNode)]


def output_order(graph):
    """Output nodes in the order the kernel returns them (matches ``out_idx``)."""
    return [n for n in find_topo_sort(graph.nodes) if isinstance(n, OutputNode)]


def output_roles(graph):
    """The boundary role of each output, in :func:`output_order`: ``""`` for an output the
    program or its cut defines (and for one without a record), else what a rewrite named
    the tensor it added to the cut (``"shift"``)."""
    return tuple(getattr(getattr(node, "boundary", None), "role", "") or ""
                 for node in output_order(graph))


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
    override = Settings.from_env().test_seed
    if override is not None:
        return override

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
    positivity *requires* positive inputs -- ``ReduceSum``, the bijective views
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
        if isinstance(node, (ReduceSum, SplitView, PermuteView)):
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
    floor = node.denom_floor
    if node.const is not None:
        const = float(node.const)
        if node.const_on_left:
            return const / (a if floor is None else a.clamp_min(floor))
        return type(node).py_op(a, const)

    b = args[1].to(out_dtype)
    if floor is not None:
        b = b.clamp_min(floor)
    if node.broadcast == 0:
        b = b.unsqueeze(-1)  # B: (*batch, M) broadcast across the last axis N
    elif node.broadcast == 1:
        b = b.unsqueeze(-2)  # B: (*batch, N) broadcast across M
    return type(node).py_op(a, b)


def _eval_splitview(node, tensor):
    tensor = tensor.to(torch_dtype(node.get_dtype(0)))
    axis, factor, inner = node.axis, node.factor, node.inner_extent
    shape = list(tensor.shape)
    tensor = tensor.reshape(
        tuple(shape[:axis]) + (factor, inner) + tuple(shape[axis + 1 :])
    )
    # The split (factor) axis is created just before `axis` and moved to split_pos.
    return tensor.movedim(axis, node.split_pos)


def _compute(fn):
    """Model the op's storage precision: inputs live in node.dtype buffers, math in out_dtype.

    E.g. an fp16 matmul operand fed by an fp32 producer is rounded as the kernel rounds it.
    """

    def run(node, args):
        in_dtype = torch_dtype(node.dtype)
        return fn(node, [a.to(in_dtype) for a in args], torch_dtype(node.out_dtype))

    return run


def _unary(fn):
    return _compute(lambda node, args, out_dtype: fn(args[0].to(out_dtype)))


def _eval_exp(node, x):
    """``exp(x)``, or for an annotated softmax exp ``exp(x - max over its softmax axis)``.

    The annotation says that a positive factor constant along that axis cancels in every
    output of a solo graph, so this is the same function with a representable
    intermediate. Per node: an Exp without the record is the plain ``exp``.
    """
    stabilize = getattr(node, "stabilize", None)
    if stabilize is not None:
        x = x - x.amax(dim=tuple(stabilize.axes), keepdim=True)
    return x.exp()


def _eval_permute(node, args):
    return args[0].to(torch_dtype(node.get_dtype(0))).permute(*node.perm).contiguous()


# Torch reference per op class, matched on the exact type: a subclass (e.g. MaskedDiv)
# never silently inherits its base's semantics. Views handle their own dtype.
REFERENCES = {
    SplitView: lambda node, args: _eval_splitview(node, args[0]),
    PermuteView: _eval_permute,
    Matmul: _compute(_eval_matmul),
    ReduceSum: _compute(lambda node, args, out_dtype: args[0].to(out_dtype).sum(dim=node.axis)),
    ReduceMax: _compute(lambda node, args, out_dtype: args[0].to(out_dtype).amax(dim=node.axis)),
    Add: _compute(_eval_binary),
    Sub: _compute(_eval_binary),
    Mul: _compute(_eval_binary),
    Div: _compute(_eval_binary),
    Exp: _compute(lambda node, args, out_dtype: _eval_exp(node, args[0].to(out_dtype))),
    Silu: _unary(lambda x: x * x.sigmoid()),
    Sqrt: _unary(lambda x: x.sqrt()),
    RSqrt: _unary(lambda x: x.rsqrt()),
    Log2: _unary(lambda x: x.log2()),
    Accumulate: _unary(lambda x: x),
    # One iteration of it, like Accumulate: the max over a single slice is the slice.
    AccumulateMax: _unary(lambda x: x),
    Copy: _unary(lambda x: x),
    GridFloor: _unary(lambda x: (x / GridFloor.GRID).floor() * GridFloor.GRID),
    Exp2: _unary(lambda x: x.exp2()),
    Lse2: _compute(lambda node, args, out_dtype: args[0].to(out_dtype).log2() + args[1].to(out_dtype)),
}


def _eval_node(node, args):
    ref = REFERENCES.get(type(node))
    if ref is not None:
        return ref(node, args)
    name = type(node).__name__
    if name in NO_REFERENCE:
        raise NotImplementedError(
            f"graph_executor has no torch reference for {name} (node {node.name!r}): "
            f"{NO_REFERENCE[name]}."
        )
    raise TypeError(f"graph_executor has no torch reference for node type {name}")


def make_inputs(graph, dtype, *, seed=None, device="cuda", distribution="normal"):
    """Sample the graph's placeholder inputs (in kernel argument order).

    ``distribution`` is ``"normal"`` (iid N(0,1), or U(0.5,1.5) where positivity is needed)
    or one of the adversarial softmax cases of :mod:`cleave.runtime.inputs`, which rewrite
    the normal sample afterwards: the normal stream is the same whatever is requested.
    """
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
    if distribution != "normal":
        from .inputs import apply_distribution

        return apply_distribution(graph, tuple(inputs), distribution)
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


def _check_layout(layout, rank):
    """``(axes, pad)`` of a stored layout (``Graph.stored_layouts``) for a rank-``rank`` output."""
    axes, pad = layout
    axes, pad = tuple(int(a) for a in axes), int(pad)
    if sorted(axes) != list(range(rank)) or pad < 0:
        raise ValueError(
            f"stored layout (axes={axes}, pad={pad}) is not a permutation of the {rank} "
            "axes of the output it describes plus a number of unit dims"
        )
    return axes, pad


def stored_shape(logical_shape, layout):
    """The shape a kernel stores an output of ``logical_shape`` in under ``layout``."""
    logical_shape = tuple(int(s) for s in logical_shape)
    axes, pad = _check_layout(layout, len(logical_shape))
    return tuple(logical_shape[a] for a in axes) + (1,) * pad


def to_stored(logical, layout):
    """``logical`` (an output as the graph defines it) in the layout a kernel stores it in.

    ``layout`` is the ``(axes, pad)`` ``apply_tiling`` recorded for that output:
    ``axes[p]`` is the logical axis at stored position ``p``, followed by ``pad`` unit dims.
    """
    axes, _ = _check_layout(layout, logical.dim())
    return logical.permute(*axes).reshape(stored_shape(logical.shape, layout))


def to_logical(stored, layout, logical_shape):
    """The inverse of :func:`to_stored`: a kernel's stored output as a contiguous tensor of
    ``logical_shape``. Raises if ``stored`` is not that output in that layout."""
    logical_shape = tuple(int(s) for s in logical_shape)
    axes, pad = _check_layout(layout, len(logical_shape))
    expected = stored_shape(logical_shape, layout)
    if tuple(stored.shape) != expected:
        raise ValueError(
            f"a tensor of shape {tuple(stored.shape)} is not an output of logical shape "
            f"{logical_shape} stored as (axes={axes}, pad={pad}), which is {expected}"
        )
    inverse = sorted(range(len(axes)), key=axes.__getitem__)
    return stored.reshape(expected[: len(axes)]).permute(*inverse).contiguous()


def is_transposed(layout) -> bool:
    """Whether ``layout`` permutes the axes of the output it describes. Such an output has
    to be copied to reach the logical layout (:func:`to_logical`); any other layout only
    appends unit dims, which is a view."""
    axes = tuple(int(a) for a in layout[0])
    return axes != tuple(range(len(axes)))


def with_hand_off(kernel, stored_layouts, logical_shapes):
    """What a launch of ``kernel`` costs its caller: ``kernel`` itself, or -- when it stores
    an output transposed -- a callable that launches it and brings every output to its
    logical layout.

    Only a split partial stores outputs transposed, and its outputs are only ever read
    through that copy (``Kernel.__call__`` hands them to the combine with ``to_logical``).
    The copy belongs to the tiling that chose the layout, so it is timed with the kernel:
    a split variant's latency is the sum of its stage timings, and would otherwise leave
    out part of every call. A kernel without transposed outputs is returned as it is.
    """
    layouts = list(stored_layouts or ())
    if not any(is_transposed(layout) for layout in layouts):
        return kernel
    shapes = [tuple(int(s) for s in shape) for shape in logical_shapes]
    if len(shapes) != len(layouts):
        raise ValueError(f"{len(layouts)} stored layouts for {len(shapes)} outputs")

    def launch_and_hand_off(*inputs):
        outputs = kernel(*inputs)
        outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
        return [to_logical(out, layout, shape)
                for out, layout, shape in zip(outputs, layouts, shapes)]

    return launch_and_hand_off


def reconcile_output_layout(reference, actual_shape, layout=None):
    """Return ``reference`` reshaped/permuted to the kernel's physical output shape.

    The executor emits an output in its *logical* axis order, but a scheduled kernel
    may store it differently: it can keep or drop size-1 axes (e.g. a split-partial
    rowsum stored ``(split_k, M, 1)``) and, for split-partial outputs, move the split
    axis to the front (``(split_k, B*H, S_Q, ...)`` vs logical ``(B*H, split_k, ...)``).

    ``layout`` is the stored layout ``apply_tiling`` recorded for this output
    (``Graph.stored_layouts``). With it the reindexing is applied as recorded and must
    produce exactly ``actual_shape``; every kernel the pipeline builds comes with one.

    Without it (a hand-written kernel has no schedule to record anything) the reindexing
    is recovered by matching non-unit axis extents, which is only well defined when those
    extents are distinct. If two non-unit axes share an extent AND a non-identity
    permutation is required to line up with ``actual_shape`` (e.g. a split-partial
    output where ``split_k == B*H``), the layout is ambiguous and we raise rather than
    silently risk a wrong permutation masking a real mismatch. Equal shapes are returned
    as they are, so a kernel that stores a permutation between equal extents is compared
    transposed: such a kernel needs the record.
    """
    import torch

    ref_shape = tuple(int(s) for s in reference.shape)
    actual_shape = tuple(int(s) for s in actual_shape)
    if layout is not None:
        stored = to_stored(reference, layout)
        if tuple(stored.shape) != actual_shape:
            raise ValueError(
                f"the recorded stored layout (axes={tuple(layout[0])}, pad={int(layout[1])}) "
                f"lays the reference of shape {ref_shape} out as {tuple(stored.shape)}, but "
                f"the kernel returned {actual_shape}"
            )
        return stored
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


def build_inputs_and_reference(graph, dtype, *, seed=None, device="cuda",
                               distribution="normal"):
    """Return ``(inputs_by_name, reference, out_idx)`` for a cleave graph.

    ``inputs_by_name`` maps each placeholder name to its tensor. Scheduling can
    reorder the kernel's parameter list relative to the flat graph's topo order, so
    inputs are keyed by name and ordered against the compiled kernel's actual
    parameter names at call time (see :func:`order_inputs_for_kernel`). ``reference``
    is a list with one tensor per output node and ``out_idx`` is the matching
    ``tl.compile`` index.
    """
    placeholders = placeholder_order(graph)
    tensors = make_inputs(graph, dtype, seed=seed, device=device, distribution=distribution)
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
