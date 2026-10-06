"""Shared cleave source-graph builders for correctness verification.

Reference and handwritten kernels are hand-written TileLang, but their numeric
reference is derived by executing an equivalent cleave source graph (see
``cleave.util.graph_executor``). The builders here construct those source graphs
so the graph executor produces inputs (in the kernel's parameter order) and the
matching reference outputs. Each builder's placeholder order must equal the
hand-written kernel's parameter order.
"""

from __future__ import annotations

from tilelang.carver.roller import Edge

from cleave.graph import Graph, OutputNode
from cleave.ops import (
    Add,
    Div,
    Exp,
    Matmul,
    Mul,
    PermuteView,
    Reduction,
    SplitView,
    Sub,
    wire_input,
)
from cleave.util import find_topo_sort, gather_all_nodes


DTYPE = "float16"
ACCUM_DTYPE = "float32"
OUT_DTYPE = "float16"


def _connect(src, dst, src_id, dst_id):
    edge = Edge(src, dst, src_id, dst_id)
    src._out_edges.append(edge)
    dst.set_inputs(edge.dst_id, edge)


def _finalize(*roots):
    nodes = gather_all_nodes(*roots)
    return Graph(nodes=find_topo_sort([n for n in nodes if n.is_output()]))


def matmul_graph(M, N, K, *, out_dtype=OUT_DTYPE):
    """A @ B -> (M, N). Kernel params: A (M,K), B (K,N)."""
    mma = Matmul(
        (M, N, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_0", warp_policy="rows"
    )
    OutputNode(mma, dtype=out_dtype)
    return _finalize(mma)


def batched_matmul_graph(B, M, N, K, *, out_dtype=OUT_DTYPE):
    """A @ B -> (B, M, N). Kernel params: A (B,M,K), B (B,K,N)."""
    mma = Matmul(
        (B, M, N, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_0", warp_policy="rows"
    )
    OutputNode(mma, dtype=out_dtype)
    return _finalize(mma)


def matmul_matmul_graph(M, N, K):
    """(A @ B) @ C -> (M, K). Kernel params: A (M,K), B (K,N), C (N,K)."""
    mma_0 = Matmul(
        (M, N, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_0", warp_policy="rows"
    )
    mma_1 = Matmul(
        (M, K, N), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_1", warp_policy="rows"
    )
    _connect(mma_0, mma_1, 0, 0)
    OutputNode(mma_1, dtype=OUT_DTYPE)
    return _finalize(mma_0, mma_1)


def matmul_matmul_bhsd_graph(B, H, S_Q, S_KV, D):
    """einsum(qk) then (qk)@v with no scaling. Kernel params: A,B,C (B*H,S,D)."""
    BH = B * H
    mma_0 = Matmul(
        (BH, S_Q, S_KV, D),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    mma_1 = Matmul(
        (BH, S_Q, D, S_KV),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
    )
    _connect(mma_0, mma_1, 0, 0)
    OutputNode(mma_1, dtype=OUT_DTYPE)
    return _finalize(mma_0, mma_1)


def matmul_matmul_bshd_graph(B, S_Q, S_KV, H, D):
    """(A @ B^T) @ C in interior-head (B,S,H,D) layout, no scaling.

    Kernel params (physical layout): A ``PlaceHolder_A`` (B,S_Q,H,D),
    B ``PlaceHolder_B`` (B,S_KV,H,D), C ``PlaceHolder_C`` (B,S_KV,H,D). Output is
    physical (B,S_Q,H,D).

    Interior-head (B,S,H,D) is not expressible directly in the cleave matmul
    vocabulary (matmul batch dims must lead). So each physical input is first
    permuted to the matmul-friendly (B,H,S,D) via a ``PermuteView`` (executor-only),
    the (B,H) two-leading-batch matmul chain from ``matmul_matmul_bhsd_graph`` is
    applied, and a trailing ``PermuteView`` restores the physical (B,S_Q,H,D) output
    layout so it matches the hand-written kernel's ``Output`` buffer.
    """
    perm_q = PermuteView((B, S_Q, H, D), (0, 2, 1, 3), name="perm_q", dtype=DTYPE)
    perm_k = PermuteView((B, S_KV, H, D), (0, 2, 1, 3), name="perm_k", dtype=DTYPE)
    perm_v = PermuteView((B, S_KV, H, D), (0, 2, 1, 3), name="perm_v", dtype=DTYPE)
    perm_q.inputs[0].src_node.name = "PlaceHolder_A"
    perm_k.inputs[0].src_node.name = "PlaceHolder_B"
    perm_v.inputs[0].src_node.name = "PlaceHolder_C"

    mma_0 = Matmul(
        (B, H, S_Q, S_KV, D),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    mma_1 = Matmul(
        (B, H, S_Q, D, S_KV),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
    )
    wire_input(perm_q, mma_0, 0)
    wire_input(perm_k, mma_0, 1)
    _connect(mma_0, mma_1, 0, 0)
    wire_input(perm_v, mma_1, 1)

    perm_out = PermuteView(
        (B, H, S_Q, D), (0, 2, 1, 3), name="perm_out", dtype=OUT_DTYPE
    )
    _connect(mma_1, perm_out, 0, 0)
    OutputNode(perm_out, dtype=OUT_DTYPE)
    return _finalize(perm_out)


def matmul_rowsum_graph(M, N, K):
    """(A @ B).sum(dim=1) -> (M,). Kernel params: A (M,K), B (K,N)."""
    mma = Matmul(
        (M, N, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_0", warp_policy="rows"
    )
    rowsum = Reduction(
        (M, N), axis=-1, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    _connect(mma, rowsum, 0, 0)
    OutputNode(rowsum, dtype=OUT_DTYPE)
    return _finalize(mma, rowsum)


def rmsnorm_matmul_graph(M, N, K):
    """rmsnorm(A) @ B via post-multiply. Kernel params: A (M,K), B (K,N).

    Matches the end2end graph exactly: square = A*A (``mul_0``), rowsum
    (``rowsum_0``), matmul A@B (``mma_0``), ``Div(const=K)`` (``div_0``) -> RSqrt
    (``rsqrt_0``) -> broadcast Mul of the matmul output (``mul_1``, out fp16).
    """
    from cleave.ops import RSqrt

    square = Mul((M, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mul_0")
    rowsum = Reduction(
        (M, K), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    mma = Matmul(
        (M, N, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_0", warp_policy="rows"
    )
    div = Div((M,), const=float(K), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="div_0")
    inv_rms = RSqrt((M,), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rsqrt_0")
    out = Mul(
        (M, N), broadcast=0, dtype=ACCUM_DTYPE, out_dtype=OUT_DTYPE, name="mul_1"
    )

    # A (square's operand 0) is shared: it also feeds the matmul, and is wired to
    # the square's operand 1 so square = A*A.
    a_node = square.inputs[0].src_node
    _connect(a_node, square, 0, 1)  # A -> square input 1 (A*A)
    _connect(square, rowsum, 0, 0)
    _connect(a_node, mma, 0, 0)  # A -> matmul input 0
    _connect(rowsum, div, 0, 0)
    _connect(div, inv_rms, 0, 0)
    _connect(mma, out, 0, 0)
    _connect(inv_rms, out, 0, 1)
    OutputNode(out, dtype=OUT_DTYPE)
    return _finalize(square, rowsum, mma, div, inv_rms, out)


def mha_bhsd_graph(B, H, S_Q, S_KV, D):
    """Flash-attention (safe-exp) MHA. Kernel params: Q,K,V (B*H,S,D)."""
    BH = B * H
    scale = 1.0 / (D**0.5)
    mma_0 = Matmul(
        (BH, S_Q, S_KV, D),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    mul_0 = Mul(
        (BH, S_Q, S_KV), const=scale, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="mul_0"
    )
    sub_0 = Sub(
        (BH, S_Q, S_KV),
        const=(50.0 * scale),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="sub_0",
    )
    exp_0 = Exp((BH, S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="exp_0")
    mma_1 = Matmul(
        (BH, S_Q, D, S_KV),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
    )
    rowsum_0 = Reduction(
        (BH, S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    div_0 = Div(
        (BH, S_Q, D), broadcast=0, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="div_0"
    )
    _connect(mma_0, mul_0, 0, 0)
    _connect(mul_0, sub_0, 0, 0)
    _connect(sub_0, exp_0, 0, 0)
    _connect(exp_0, mma_1, 0, 0)
    _connect(exp_0, rowsum_0, 0, 0)
    _connect(mma_1, div_0, 0, 0)
    _connect(rowsum_0, div_0, 0, 1)
    OutputNode(div_0, dtype=OUT_DTYPE)
    return _finalize(mma_0, mul_0, sub_0, exp_0, mma_1, rowsum_0, div_0)


def mha_bshd_graph(B, S_Q, S_KV, H, D):
    """Flash-attention (safe-exp) MHA in interior-head (B,S,H,D) layout.

    Kernel params (physical layout): Q ``PlaceHolder_A`` (B,S_Q,H,D),
    K ``PlaceHolder_B`` (B,S_KV,H,D), V ``PlaceHolder_C`` (B,S_KV,H,D). Output is
    physical (B,S_Q,H,D).

    Interior-head (B,S,H,D) is not expressible directly in the cleave matmul
    vocabulary (matmul batch dims must lead). So each physical input is first
    permuted to the matmul-friendly (B,H,S,D) via a ``PermuteView`` (executor-only),
    the (B,H) two-leading-batch safe-exp attention chain from ``mha_bhsd_graph`` is
    applied, and a trailing ``PermuteView`` restores the physical (B,S_Q,H,D) output
    layout so it matches the hand-written kernel's ``Output`` buffer. The safe-exp
    softmax (scale ``1/sqrt(D)``, bias ``50/sqrt(D)``) is numerically equal to the
    kernel's online reduce_max softmax by softmax shift-invariance.
    """
    scale = 1.0 / (D**0.5)
    perm_q = PermuteView((B, S_Q, H, D), (0, 2, 1, 3), name="perm_q", dtype=DTYPE)
    perm_k = PermuteView((B, S_KV, H, D), (0, 2, 1, 3), name="perm_k", dtype=DTYPE)
    perm_v = PermuteView((B, S_KV, H, D), (0, 2, 1, 3), name="perm_v", dtype=DTYPE)
    perm_q.inputs[0].src_node.name = "PlaceHolder_A"
    perm_k.inputs[0].src_node.name = "PlaceHolder_B"
    perm_v.inputs[0].src_node.name = "PlaceHolder_C"

    mma_0 = Matmul(
        (B, H, S_Q, S_KV, D),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    mul_0 = Mul(
        (B, H, S_Q, S_KV),
        const=scale,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mul_0",
    )
    sub_0 = Sub(
        (B, H, S_Q, S_KV),
        const=(50.0 * scale),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="sub_0",
    )
    exp_0 = Exp(
        (B, H, S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="exp_0"
    )
    mma_1 = Matmul(
        (B, H, S_Q, D, S_KV),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
    )
    rowsum_0 = Reduction(
        (B, H, S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    div_0 = Div(
        (B, H, S_Q, D),
        broadcast=0,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )
    wire_input(perm_q, mma_0, 0)
    wire_input(perm_k, mma_0, 1)
    _connect(mma_0, mul_0, 0, 0)
    _connect(mul_0, sub_0, 0, 0)
    _connect(sub_0, exp_0, 0, 0)
    _connect(exp_0, mma_1, 0, 0)
    wire_input(perm_v, mma_1, 1)
    _connect(exp_0, rowsum_0, 0, 0)
    _connect(mma_1, div_0, 0, 0)
    _connect(rowsum_0, div_0, 0, 1)

    perm_out = PermuteView(
        (B, H, S_Q, D), (0, 2, 1, 3), name="perm_out", dtype=OUT_DTYPE
    )
    _connect(div_0, perm_out, 0, 0)
    OutputNode(perm_out, dtype=OUT_DTYPE)
    return _finalize(perm_out)


def matmul_splitk_partial_graph(SPLIT_K, M, N, K_PARTIAL):
    """Split-K partial matmul. Kernel params: A (M,K), B (K,N) with K=SPLIT_K*K_PARTIAL.

    Output is the per-split product stacked on a leading split axis: (SPLIT_K, M, N).
    Mirrors ``tests/schedule/end2end/test_matmul_splitk_partial.py``.
    """
    K = SPLIT_K * K_PARTIAL
    split_a = SplitView((M, K), axis=1, factor=SPLIT_K, dtype=DTYPE, name="split_a")
    split_b = SplitView((K, N), axis=0, factor=SPLIT_K, dtype=DTYPE, name="split_b")
    mma = Matmul(
        (SPLIT_K, M, N, K_PARTIAL),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        warp_policy="rows",
    )
    wire_input(split_a, mma, 0)
    wire_input(split_b, mma, 1)
    OutputNode(mma, dtype=OUT_DTYPE)
    return _finalize(mma)


def matmul_splitk_combine_graph(SPLIT_K, M, N):
    """Split-K combine: sum the per-split partials. Kernel param: A (SPLIT_K, M, N).

    Mirrors ``tests/schedule/end2end/test_matmul_splitk_combine.py``.
    """
    reduction = Reduction(
        (SPLIT_K, M, N), axis=0, dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="reduction_0"
    )
    OutputNode(reduction, dtype=OUT_DTYPE)
    return _finalize(reduction)


def batched_matmul_splitk_partial_graph(B, SPLIT_K, M, N, K_PARTIAL):
    """Batched split-K partial matmul. Kernel params: A (B,M,K), B (B,K,N).

    Output: (B, SPLIT_K, M, N). Mirrors
    ``tests/schedule/end2end/test_batched_matmul_splitk_partial.py``.
    """
    K = SPLIT_K * K_PARTIAL
    split_a = SplitView((B, M, K), axis=2, factor=SPLIT_K, dtype=DTYPE, name="split_a")
    split_b = SplitView((B, K, N), axis=1, factor=SPLIT_K, dtype=DTYPE, name="split_b")
    mma = Matmul(
        (B, SPLIT_K, M, N, K_PARTIAL),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        warp_policy="rows",
    )
    wire_input(split_a, mma, 0)
    wire_input(split_b, mma, 1)
    OutputNode(mma, dtype=OUT_DTYPE)
    return _finalize(mma)


def batched_matmul_splitk_combine_graph(B, SPLIT_K, M, N):
    """Batched split-K combine. Kernel param: A (B, SPLIT_K, M, N) -> (B, M, N).

    Mirrors ``tests/schedule/end2end/test_batched_matmul_splitk_combine.py``.
    """
    reduction = Reduction(
        (B, SPLIT_K, M, N),
        axis=-3,
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="reduction_0",
    )
    OutputNode(reduction, dtype=OUT_DTYPE)
    return _finalize(reduction)


def swiglu_graph(M, N, K):
    """(A@B) * silu(A@C). Kernel buffers: A (M,K), B (K,N), C (K,N).

    The kernel computes ``silu(A@C) * (A@B)``: weight ``B`` feeds the bare product
    (``mma_up``) and weight ``C`` feeds silu (``mma_gate``). Inputs bind to the
    kernel by buffer name, so the placeholder ROLES must match: the up-matmul's
    weight must be named ``PlaceHolder_B`` and the gate-matmul's weight
    ``PlaceHolder_C``. Placeholder names are assigned explicitly below because the
    shared ``A`` input makes the automatic topo-order naming ambiguous.
    """
    from cleave.ops import Silu

    mma_up = Matmul(
        (M, N, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_up", warp_policy="rows"
    )
    mma_gate = Matmul(
        (M, N, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_gate", warp_policy="rows"
    )
    silu = Silu((M, N), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="silu_0")
    mul = Mul((M, N), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="mul_0")

    # Name placeholders by role to match the kernel's A/B/C buffers exactly.
    a_node = mma_up.inputs[0].src_node
    a_node.name = "PlaceHolder_A"
    mma_up.inputs[1].src_node.name = "PlaceHolder_B"  # up weight (bare product)
    wire_input(a_node, mma_gate, 0, src_id=0)  # share A into the gate matmul
    mma_gate.inputs[1].src_node.name = "PlaceHolder_C"  # gate weight (feeds silu)

    _connect(mma_gate, silu, 0, 0)
    _connect(mma_up, mul, 0, 0)
    _connect(silu, mul, 0, 1)
    OutputNode(mul, dtype=OUT_DTYPE)
    return _finalize(mma_up, mma_gate, silu, mul)


def lora_graph(M, N, K, R):
    """LoRA update ``O = X@W + (X@A)@B`` (rank R, no scaling). Kernel params (in
    order): X (M,K), W (K,N), A (K,R), B (R,N); Output (M,N).

    Combines a shared-X double projection (like ``swiglu_graph``) with a chained
    matmul (like ``matmul_matmul_graph``): X feeds both the base matmul ``X@W``
    (``mma_base``) and the down projection ``X@A`` (``mma_down``, rank R); the
    down result feeds the up projection ``D@B`` (``mma_up``) and the two N-wide
    products are summed (``add_0``). Placeholders are named by role in kernel
    parameter order -- ``PlaceHolder_A/B/C/D`` = X, W, A, B -- because the shared
    X input makes the automatic topo-order naming ambiguous; a hand-written
    kernel binds them positionally in that same X, W, A, B order.
    """
    mma_base = Matmul(
        (M, N, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_base", warp_policy="rows"
    )
    mma_down = Matmul(
        (M, R, K), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_down", warp_policy="rows"
    )
    mma_up = Matmul(
        (M, N, R), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name="mma_up", warp_policy="rows"
    )
    add_0 = Add((M, N), dtype=ACCUM_DTYPE, out_dtype=OUT_DTYPE, name="add_0")

    # Name placeholders by role to match the kernel's X/W/A/B params in order.
    x_node = mma_base.inputs[0].src_node
    x_node.name = "PlaceHolder_A"  # X
    mma_base.inputs[1].src_node.name = "PlaceHolder_B"  # W (base weight, K x N)
    wire_input(x_node, mma_down, 0, src_id=0)  # share X into the down projection
    mma_down.inputs[1].src_node.name = "PlaceHolder_C"  # A (down weight, K x R)
    _connect(mma_down, mma_up, 0, 0)  # D = X@A -> up projection operand 0
    mma_up.inputs[1].src_node.name = "PlaceHolder_D"  # B (up weight, R x N)
    _connect(mma_base, add_0, 0, 0)  # X@W -> add operand 0
    _connect(mma_up, add_0, 0, 1)  # (X@A)@B -> add operand 1
    OutputNode(add_0, dtype=OUT_DTYPE)
    return _finalize(mma_base, mma_down, mma_up, add_0)


def gqa_bhsd_graph(B, H_Q, H_KV, S_Q, S_KV, D):
    """Grouped-query flash-attention (safe-exp). Kernel params: Q (B*H_Q,S_Q,D),
    K (B*H_KV,S_KV,D), V (B*H_KV,S_KV,D).

    Q carries the full query-head batch (``B*H_Q``) while K/V carry the KV-head
    batch (``B*H_KV``); the ``input_batch_dims_A/B`` on both matmuls encode the
    grouped broadcast. Mirrors ``tests/schedule/end2end/test_gqa_bhsd.py``.
    """
    scale = 1 / (D**0.5)
    mma_0 = Matmul(
        (B * H_Q, S_Q, S_KV, D),
        input_batch_dims_A=(B * H_Q,),
        input_batch_dims_B=(B * H_KV,),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    mul_0 = Mul(
        (B * H_Q, S_Q, S_KV),
        const=scale,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mul_0",
    )
    sub_0 = Sub(
        (B * H_Q, S_Q, S_KV),
        const=(50.0 * scale),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="sub_0",
    )
    exp_0 = Exp(
        (B * H_Q, S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="exp_0"
    )
    mma_1 = Matmul(
        (B * H_Q, S_Q, D, S_KV),
        input_batch_dims_A=(B * H_Q,),
        input_batch_dims_B=(B * H_KV,),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
    )
    rowsum_0 = Reduction(
        (B * H_Q, S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    div_0 = Div(
        (B * H_Q, S_Q, D),
        broadcast=0,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )
    _connect(mma_0, mul_0, 0, 0)
    _connect(mul_0, sub_0, 0, 0)
    _connect(sub_0, exp_0, 0, 0)
    _connect(exp_0, mma_1, 0, 0)
    _connect(exp_0, rowsum_0, 0, 0)
    _connect(mma_1, div_0, 0, 0)
    _connect(rowsum_0, div_0, 0, 1)
    OutputNode(div_0, dtype=OUT_DTYPE)
    return _finalize(mma_0, mul_0, sub_0, exp_0, mma_1, rowsum_0, div_0)


def qknorm_graph(B, H_Q, H_KV, S_Q, S_KV, HD, eps=1e-6):
    """Grouped-query flash-attention with RMSNorm applied to Q and K (QKNorm).

    Kernel params: Q ``PlaceHolder_A`` (B*H_Q,S_Q,HD), K ``PlaceHolder_B``
    (B*H_KV,S_KV,HD), V ``PlaceHolder_C`` (B*H_KV,S_KV,HD).

    Each of Q and K is RMS-normalized over its feature axis
    (``x * rsqrt(mean(x*x) + eps)``) and the normalized tensors feed the grouped
    safe-exp attention block from ``gqa_bhsd_graph``. This mirrors the structure
    the superoptimizer emits for ``demo/qknorm.py`` (which the schedule can tile
    and codegen), which differs from the demo's literal source in two algebraically
    equivalent ways that the search discovers:
      * the reciprocal-sqrt is a single ``RSqrt`` (the integration fuses the demo's
        ``div(1, sqrt(...))`` into ``RSqrt``);
      * the softmax normalization divides AFTER the value matmul -- the flash form
        ``O = (E @ V) / rowsum(E)`` -- rather than the demo's pre-divide
        ``(E / rowsum(E)) @ V`` (equal, and the form ``gqa_bhsd_graph`` uses).
    The softmax scale/bias use ``Div(A, sqrt(HD))`` then ``Add(A, -50/sqrt(HD))``
    (vs gqa_bhsd's ``Mul``/``Sub``); numerically identical by shift-invariance.

    Q feeds both its square (``q_sq = Q*Q``) and the normalize ``Mul``, and K
    likewise, so placeholders are named by role to keep the kernel param order
    Q,K,V unambiguous (the shared inputs make topo-order naming ambiguous, as in
    ``rmsnorm_matmul_graph``/``swiglu_graph``). The two operands of each square
    ``Mul`` are wired to the SAME placeholder node so the executor treats it as a
    genuine square (positivity stops at the self-square, keeping Q/K general).
    """
    from cleave.ops import RSqrt

    def _rmsnorm(x_node, batch, seq, name):
        # x_node: the placeholder feeding this RMSNorm branch.
        sq = Mul((batch, seq, HD), dtype=DTYPE, out_dtype=ACCUM_DTYPE, name=f"{name}_sq")
        # Wire x into BOTH operands of the square (shared node -> genuine x*x).
        _connect(x_node, sq, 0, 0)
        _connect(x_node, sq, 0, 1)
        rowsum = Reduction(
            (batch, seq, HD), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name=f"{name}_rowsum"
        )
        # mean and eps_add operate on the rank-2 (batch, seq) reduction output whose
        # leading B*H dim is a grid/batch axis (elided by tiling). Mark it batch
        # (n_batch_dims=1) so codegen emits a 1-D loop matching the tiled buffer;
        # the default dims[:-2] would give 0 batch dims -> a 2-D loop over a 1-D
        # tile. Mirrors what the integration derives for the demo's search graph.
        mean = Div(
            (batch, seq), const=float(HD), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE,
            name=f"{name}_mean", n_batch_dims=1,
        )
        eps_add = Add(
            (batch, seq), const=eps, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE,
            name=f"{name}_eps", n_batch_dims=1,
        )
        inv = RSqrt(
            (batch, seq), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name=f"{name}_rsqrt",
            n_batch_dims=1,
        )
        norm = Mul(
            (batch, seq, HD), broadcast=0, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE,
            name=f"{name}_norm",
        )
        _connect(sq, rowsum, 0, 0)
        _connect(rowsum, mean, 0, 0)
        _connect(mean, eps_add, 0, 0)
        _connect(eps_add, inv, 0, 0)
        _connect(x_node, norm, 0, 0)  # x (shared) is operand 0 of the scale
        _connect(inv, norm, 0, 1)
        return sq, rowsum, mean, eps_add, inv, norm

    scale = HD**0.5
    mma_0 = Matmul(
        (B * H_Q, S_Q, S_KV, HD),
        input_batch_dims_A=(B * H_Q,),
        input_batch_dims_B=(B * H_KV,),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    # Q_norm is mma_0's operand 0, K_norm operand 1. Grab their placeholder inputs
    # and build each RMSNorm branch feeding them, naming placeholders by role.
    q_node = mma_0.inputs[0].src_node
    q_node.name = "PlaceHolder_A"
    q_nodes = _rmsnorm(q_node, B * H_Q, S_Q, "q")
    wire_input(q_nodes[-1], mma_0, 0)  # q_norm -> mma_0 operand 0
    k_node = mma_0.inputs[1].src_node
    k_node.name = "PlaceHolder_B"
    k_nodes = _rmsnorm(k_node, B * H_KV, S_KV, "k")
    wire_input(k_nodes[-1], mma_0, 1)  # k_norm -> mma_0 operand 1

    div_scale = Div(
        (B * H_Q, S_Q, S_KV), const=scale, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE,
        name="div_scale",
    )
    add_bias = Add(
        (B * H_Q, S_Q, S_KV), const=(-50.0 / scale), dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE, name="add_bias",
    )
    exp_0 = Exp(
        (B * H_Q, S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="exp_0"
    )
    # Flash form: P@V unnormalized, softmax rowsum a sibling, divide after (like
    # gqa_bhsd_graph). exp feeds BOTH mma_1 and rowsum_0; div_0 broadcasts rowsum.
    mma_1 = Matmul(
        (B * H_Q, S_Q, HD, S_KV),
        input_batch_dims_A=(B * H_Q,),
        input_batch_dims_B=(B * H_KV,),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
    )
    v_node = mma_1.inputs[1].src_node
    v_node.name = "PlaceHolder_C"
    rowsum_0 = Reduction(
        (B * H_Q, S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    div_0 = Div(
        (B * H_Q, S_Q, HD), broadcast=0, dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE,
        name="div_0",
    )

    _connect(mma_0, div_scale, 0, 0)
    _connect(div_scale, add_bias, 0, 0)
    _connect(add_bias, exp_0, 0, 0)
    _connect(exp_0, mma_1, 0, 0)
    _connect(exp_0, rowsum_0, 0, 0)
    _connect(mma_1, div_0, 0, 0)
    _connect(rowsum_0, div_0, 0, 1)
    OutputNode(div_0, dtype=OUT_DTYPE)
    return _finalize(mma_0, mma_1, rowsum_0, div_0)


def mqa_bhsd_graph(B, H_Q, H_KV, S_Q, S_KV, D):
    """Multi-query flash-attention (safe-exp). Kernel params: Q (B*H_Q,S_Q,D),
    K (B*H_KV,S_KV,D), V (B*H_KV,S_KV,D).

    Structurally identical to ``gqa_bhsd_graph``; MQA is the ``H_KV == 1`` case.
    Mirrors ``tests/schedule/end2end/test_mqa_bhsd.py``.
    """
    scale = 1 / (D**0.5)
    mma_0 = Matmul(
        (B * H_Q, S_Q, S_KV, D),
        input_batch_dims_A=(B * H_Q,),
        input_batch_dims_B=(B * H_KV,),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    mul_0 = Mul(
        (B * H_Q, S_Q, S_KV),
        const=scale,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mul_0",
    )
    sub_0 = Sub(
        (B * H_Q, S_Q, S_KV),
        const=(50.0 * scale),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="sub_0",
    )
    exp_0 = Exp(
        (B * H_Q, S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="exp_0"
    )
    mma_1 = Matmul(
        (B * H_Q, S_Q, D, S_KV),
        input_batch_dims_A=(B * H_Q,),
        input_batch_dims_B=(B * H_KV,),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
    )
    rowsum_0 = Reduction(
        (B * H_Q, S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    div_0 = Div(
        (B * H_Q, S_Q, D),
        broadcast=0,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )
    _connect(mma_0, mul_0, 0, 0)
    _connect(mul_0, sub_0, 0, 0)
    _connect(sub_0, exp_0, 0, 0)
    _connect(exp_0, mma_1, 0, 0)
    _connect(exp_0, rowsum_0, 0, 0)
    _connect(mma_1, div_0, 0, 0)
    _connect(rowsum_0, div_0, 0, 1)
    OutputNode(div_0, dtype=OUT_DTYPE)
    return _finalize(mma_0, mul_0, sub_0, exp_0, mma_1, rowsum_0, div_0)


def matmul_softmax_graph(B, H, S_Q, S_KV, D):
    """Scaled softmax over a QK matmul, no value matmul. Kernel params: Q,K (B*H,S,D).

    ``softmax(scale * (Q @ K^T))`` returning the full (B*H, S_Q, S_KV) attention
    weights (the row-normalised exp). Mirrors ``_build_matmul_softmax_graph`` in
    ``tests/graph/apply_tiling/test_matmul_softmax.py``.
    """
    scale = 1 / (D**0.5)
    mma_0 = Matmul(
        (B * H, S_Q, S_KV, D),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    mul_0 = Mul(
        (B * H, S_Q, S_KV),
        const=scale,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mul_0",
    )
    sub_0 = Sub(
        (B * H, S_Q, S_KV),
        const=(50.0 * scale),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="sub_0",
    )
    exp_0 = Exp(
        (B * H, S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="exp_0"
    )
    rowsum_0 = Reduction(
        (B * H, S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    div_0 = Div(
        (B * H, S_Q, S_KV),
        broadcast=0,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )
    wire_input(mma_0, mul_0, dst_id=0)
    wire_input(mul_0, sub_0, dst_id=0)
    wire_input(sub_0, exp_0, dst_id=0)
    wire_input(exp_0, rowsum_0, dst_id=0)
    wire_input(exp_0, div_0, dst_id=0)
    wire_input(rowsum_0, div_0, dst_id=1)
    OutputNode(div_0, dtype=OUT_DTYPE)
    return _finalize(mma_0, mul_0, sub_0, exp_0, rowsum_0, div_0)


def flash_attention_sd_graph(
    S_Q, S_KV, D, *, dtype=DTYPE, accum_dtype=ACCUM_DTYPE, out_dtype=OUT_DTYPE
):
    """Single-head flash-attention (safe-exp). Kernel params: Q,K,V (S,D).

    Same op chain as ``mha_bhsd_graph`` but with no batch dimension (rank-2/3
    tensors). Mirrors ``tests/schedule/end2end/test_flash_attention_sd.py``.

    The dtypes are arguments because this is also the dense body a paged attention
    operator is built from, and those operators are bfloat16.
    """
    scale = 1 / (D**0.5)
    node_0 = Matmul(
        (S_Q, S_KV, D),
        dtype=dtype,
        out_dtype=accum_dtype,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    node_1 = Mul(
        (S_Q, S_KV),
        const=scale,
        dtype=accum_dtype,
        out_dtype=accum_dtype,
        name="mul_0",
    )
    node_2 = Sub(
        (S_Q, S_KV),
        const=(50.0 * scale),
        dtype=accum_dtype,
        out_dtype=accum_dtype,
        name="sub_0",
    )
    node_3 = Exp((S_Q, S_KV), dtype=accum_dtype, out_dtype=accum_dtype, name="exp_0")
    node_4 = Matmul(
        (S_Q, D, S_KV),
        dtype=dtype,
        out_dtype=accum_dtype,
        name="mma_1",
        warp_policy="rows",
    )
    node_5 = Reduction(
        (S_Q, S_KV), dtype=accum_dtype, out_dtype=accum_dtype, name="rowsum_0"
    )
    node_6 = Div(
        (S_Q, D),
        broadcast=0,
        dtype=accum_dtype,
        out_dtype=accum_dtype,
        name="div_0",
    )
    _connect(node_0, node_1, 0, 0)
    _connect(node_1, node_2, 0, 0)
    _connect(node_2, node_3, 0, 0)
    _connect(node_3, node_4, 0, 0)
    _connect(node_3, node_5, 0, 0)
    _connect(node_4, node_6, 0, 0)
    _connect(node_5, node_6, 0, 1)
    OutputNode(node_6, dtype=out_dtype)
    return _finalize(node_0, node_1, node_2, node_3, node_4, node_5, node_6)


def mla_attention_sd_graph(
    S_Q, S_KV, HD, PE, *, dtype=DTYPE, accum_dtype=ACCUM_DTYPE, out_dtype=OUT_DTYPE
):
    """Single-head MLA attention (safe-exp) with RoPE positional scores, no batch dim.

    Kernel params (by placeholder role): Q ``PlaceHolder_A`` (S_Q,HD), KV
    ``PlaceHolder_B`` (S_KV,HD), Q_pe ``PlaceHolder_C`` (S_Q,PE), K_pe
    ``PlaceHolder_D`` (S_KV,PE). The score is ``(Q @ KV^T) + (Q_pe @ K_pe^T)``, then
    softmax, then a value matmul that reuses KV -- MLA has no separate V, which is why
    the placeholders are named by role.

    The rank-2 counterpart of ``mla_decode_bhsd_graph``: this is the dense body the paged
    MLA operators are built from, and those have no batch axis to tile (their batch is a
    ragged request list resolved at runtime) and are bfloat16.
    """
    scale = 1 / (HD**0.5)
    node_0 = Matmul(
        (S_Q, S_KV, HD),
        dtype=dtype,
        out_dtype=accum_dtype,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    q = node_0.inputs[0].src_node
    q.name = "PlaceHolder_A"
    kv = node_0.inputs[1].src_node
    kv.name = "PlaceHolder_B"

    node_pe = Matmul(
        (S_Q, S_KV, PE),
        dtype=dtype,
        out_dtype=accum_dtype,
        name="mma_pe",
        transpose_B=True,
        warp_policy="rows",
    )
    node_pe.inputs[0].src_node.name = "PlaceHolder_C"
    node_pe.inputs[1].src_node.name = "PlaceHolder_D"

    node_add = Add(
        (S_Q, S_KV), dtype=accum_dtype, out_dtype=accum_dtype, name="add_0"
    )
    node_1 = Mul(
        (S_Q, S_KV),
        const=scale,
        dtype=accum_dtype,
        out_dtype=accum_dtype,
        name="mul_0",
    )
    node_2 = Sub(
        (S_Q, S_KV),
        const=(50.0 * scale),
        dtype=accum_dtype,
        out_dtype=accum_dtype,
        name="sub_0",
    )
    node_3 = Exp((S_Q, S_KV), dtype=accum_dtype, out_dtype=accum_dtype, name="exp_0")
    node_4 = Matmul(
        (S_Q, HD, S_KV),
        dtype=dtype,
        out_dtype=accum_dtype,
        name="mma_1",
        warp_policy="rows",
    )
    node_5 = Reduction(
        (S_Q, S_KV), dtype=accum_dtype, out_dtype=accum_dtype, name="rowsum_0"
    )
    node_6 = Div(
        (S_Q, HD),
        broadcast=0,
        dtype=accum_dtype,
        out_dtype=accum_dtype,
        name="div_0",
    )
    _connect(node_0, node_add, 0, 0)
    _connect(node_pe, node_add, 0, 1)
    _connect(node_add, node_1, 0, 0)
    _connect(node_1, node_2, 0, 0)
    _connect(node_2, node_3, 0, 0)
    _connect(node_3, node_4, 0, 0)
    _connect(kv, node_4, 0, 1)
    _connect(node_3, node_5, 0, 0)
    _connect(node_4, node_6, 0, 0)
    _connect(node_5, node_6, 0, 1)
    OutputNode(node_6, dtype=out_dtype)
    return _finalize(
        node_0, node_pe, node_add, node_1, node_2, node_3, node_4, node_5, node_6
    )


def mla_prefill_bhsd_graph(B, H, S_Q, S_KV, HD_IN, HD_OUT):
    """MLA-prefill flash-attention (safe-exp). Kernel params: Q (B*H,S_Q,HD_IN),
    KV (B*H,S_KV,HD_IN) for the score matmul and V (B*H,S_KV,HD_OUT) for the
    value matmul.

    Same op chain as ``mha_bhsd_graph`` but with distinct input/output head dims
    (``HD_IN`` for the QK score, ``HD_OUT`` for the value projection). Mirrors
    ``tests/schedule/end2end/test_mla_prefill_bhsd.py``.
    """
    scale = 1 / (HD_IN**0.5)
    node_0 = Matmul(
        (B * H, S_Q, S_KV, HD_IN),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    node_1 = Mul(
        (B * H, S_Q, S_KV),
        const=scale,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mul_0",
    )
    node_2 = Sub(
        (B * H, S_Q, S_KV),
        const=(50.0 * scale),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="sub_0",
    )
    node_3 = Exp(
        (B * H, S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="exp_0"
    )
    node_4 = Matmul(
        (B * H, S_Q, HD_OUT, S_KV),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
    )
    node_5 = Reduction(
        (B * H, S_Q, S_KV), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="rowsum_0"
    )
    node_6 = Div(
        (B * H, S_Q, HD_OUT),
        broadcast=0,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )
    _connect(node_0, node_1, 0, 0)
    _connect(node_1, node_2, 0, 0)
    _connect(node_2, node_3, 0, 0)
    _connect(node_3, node_4, 0, 0)
    _connect(node_3, node_5, 0, 0)
    _connect(node_4, node_6, 0, 0)
    _connect(node_5, node_6, 0, 1)
    OutputNode(node_6, dtype=OUT_DTYPE)
    return _finalize(node_0, node_1, node_2, node_3, node_4, node_5, node_6)


def mla_decode_bhsd_graph(B, H, S_KV, HD, PE):
    """MLA-decode flash-attention (safe-exp) with RoPE positional scores.

    Kernel params (by placeholder role): Q ``PlaceHolder_A`` (B,H,HD), KV
    ``PlaceHolder_B`` (B,S_KV,HD), Q_pe ``PlaceHolder_C`` (B,H,PE), K_pe
    ``PlaceHolder_D`` (B,S_KV,PE). The score is ``(Q @ KV^T) + (Q_pe @ K_pe^T)``
    then softmax and value matmul reusing KV. Placeholder names are set by role
    because ``KV`` is shared between the score matmul and the value matmul.
    Mirrors ``tests/schedule/end2end/test_mla_decode_bhsd.py``.
    """
    scale = 1 / (HD**0.5)
    node_0 = Matmul(
        (B, H, S_KV, HD),
        input_batch_dims_A=(B,),
        input_batch_dims_B=(B,),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    q = node_0.inputs[0].src_node
    q.name = "PlaceHolder_A"
    kv = node_0.inputs[1].src_node
    kv.name = "PlaceHolder_B"

    node_pe = Matmul(
        (B, H, S_KV, PE),
        input_batch_dims_A=(B,),
        input_batch_dims_B=(B,),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_pe",
        transpose_B=True,
        warp_policy="rows",
    )
    q_pe = node_pe.inputs[0].src_node
    q_pe.name = "PlaceHolder_C"
    k_pe = node_pe.inputs[1].src_node
    k_pe.name = "PlaceHolder_D"

    node_add = Add(
        (B, H, S_KV),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="add_0",
    )
    node_1 = Mul(
        (B, H, S_KV),
        const=scale,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mul_0",
    )
    node_2 = Sub(
        (B, H, S_KV),
        const=(50.0 * scale),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="sub_0",
    )
    node_3 = Exp(
        (B, H, S_KV),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="exp_0",
    )
    node_4 = Matmul(
        (B, H, HD, S_KV),
        input_batch_dims_A=(B,),
        input_batch_dims_B=(B,),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
    )
    node_5 = Reduction(
        (B, H, S_KV),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="rowsum_0",
    )
    node_6 = Div(
        (B, H, HD),
        broadcast=0,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )
    wire_input(node_0, node_add, 0)
    wire_input(node_pe, node_add, 1)
    wire_input(node_add, node_1, 0)
    wire_input(node_1, node_2, 0)
    wire_input(node_2, node_3, 0)
    wire_input(node_3, node_4, 0)
    wire_input(kv, node_4, 1)
    wire_input(node_3, node_5, 0)
    wire_input(node_4, node_6, 0)
    wire_input(node_5, node_6, 1)
    OutputNode(node_6, dtype=OUT_DTYPE)
    return _finalize(
        node_0, node_pe, node_add, node_1, node_2, node_3, node_4, node_5, node_6
    )


def gqa_split_partial_graph(B, H_Q, H_KV, SPLIT_K, S_Q, S_KV_PARTIAL, D):
    """Grouped-query split-KV flash-attention partial (general MHA/MQA/GQA).

    Kernel params (by placeholder role, in kernel parameter order):
      Q ``PlaceHolder_A`` (B*H_Q, S_Q, D)
      K ``PlaceHolder_B`` (B*H_KV, S_KV, D)
      V ``PlaceHolder_C`` (B*H_KV, S_KV, D)   with ``S_KV = SPLIT_K * S_KV_PARTIAL``.

    ``H_Q`` query heads share ``H_KV`` key/value heads (group size
    ``H_Q // H_KV``); ``H_Q == H_KV`` is plain MHA and ``H_KV == 1`` is MQA. K/V are
    split on the sequence axis into ``SPLIT_K`` chunks; each chunk runs a local
    (unnormalised) attention producing a per-split exp-rowsum and a per-split value
    matmul. TWO outputs: ``OutputNode_0`` (rowsum, (SPLIT_K,B*H_Q,S_Q,1)) and
    ``OutputNode_1`` (value partial, (SPLIT_K,B*H_Q,S_Q,D)); order and names are
    preserved. Mirrors ``handwritten_schedules/{gqa,mqa,mha}_split_partial.py``.

    The grouping is expressed purely through the matmul batch dims: each matmul
    carries batch dims ``(B*H_Q, ...)`` on the A (query) operand and
    ``(B*H_KV, SPLIT_K)`` on the B (key/value) operand, so K/V heads broadcast
    across each group of ``H_Q // H_KV`` query heads (grouped ``repeat_interleave``,
    matching the closures' ``k.repeat_interleave(group_size, dim=0)``). For
    ``H_Q == H_KV`` this reproduces the plain-MHA builder structurally.
    """
    S_KV = SPLIT_K * S_KV_PARTIAL

    split_k = SplitView(
        (B * H_KV, S_KV, D),
        axis=1,
        factor=SPLIT_K,
        dtype=DTYPE,
        name="split_k",
    )
    split_v = SplitView(
        (B * H_KV, S_KV, D),
        axis=1,
        factor=SPLIT_K,
        dtype=DTYPE,
        name="split_v",
    )
    split_k.inputs[0].src_node.name = "PlaceHolder_B"
    split_v.inputs[0].src_node.name = "PlaceHolder_C"

    mma_0 = Matmul(
        (B * H_Q, SPLIT_K, S_Q, S_KV_PARTIAL, D),
        input_batch_dims_A=(B * H_Q, 1),
        input_batch_dims_B=(B * H_KV, SPLIT_K),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    q = mma_0.inputs[0].src_node
    q.name = "PlaceHolder_A"
    q.set_shape((B * H_Q, S_Q, D), overwrite=True)
    q.is_batch_dim = [True, False, False]
    wire_input(split_k, mma_0, 1)

    div_0 = Div(
        (B * H_Q, SPLIT_K, S_Q, S_KV_PARTIAL),
        const=float(D**0.5),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )
    add_0 = Add(
        (B * H_Q, SPLIT_K, S_Q, S_KV_PARTIAL),
        const=-(50.0 / (D**0.5)),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="add_0",
    )
    exp_0 = Exp(
        (B * H_Q, SPLIT_K, S_Q, S_KV_PARTIAL),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="exp_0",
    )
    rowsum_0 = Reduction(
        (B * H_Q, SPLIT_K, S_Q, S_KV_PARTIAL),
        axis=-1,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="rowsum_0",
    )
    mma_1 = Matmul(
        (B * H_Q, SPLIT_K, S_Q, D, S_KV_PARTIAL),
        input_batch_dims_A=(B * H_Q, SPLIT_K),
        input_batch_dims_B=(B * H_KV, SPLIT_K),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_1",
        warp_policy="rows",
    )

    wire_input(mma_0, div_0, 0)
    wire_input(div_0, add_0, 0)
    wire_input(add_0, exp_0, 0)
    wire_input(exp_0, rowsum_0, 0)
    wire_input(exp_0, mma_1, 0)
    wire_input(split_v, mma_1, 1)

    out_rowsum = OutputNode(rowsum_0, dtype=OUT_DTYPE)
    out_rowsum.name = "OutputNode_0"
    out_partial = OutputNode(mma_1, dtype=OUT_DTYPE)
    out_partial.name = "OutputNode_1"
    return Graph(nodes=find_topo_sort([out_rowsum, out_partial]))


def mha_split_partial_graph(B, H, SPLIT_K, S_Q, S_KV_PARTIAL, D):
    """Split-KV flash-attention partial (plain MHA). Kernel params (by role):
    Q ``PlaceHolder_A`` (B*H,S_Q,D), K ``PlaceHolder_B`` (B*H,S_KV,D),
    V ``PlaceHolder_C`` (B*H,S_KV,D) with ``S_KV = SPLIT_K * S_KV_PARTIAL``.

    Thin wrapper over :func:`gqa_split_partial_graph` with ``H_Q == H_KV == H`` (no
    head grouping). TWO outputs: ``OutputNode_0`` (rowsum, (SPLIT_K,B*H,S_Q,1)) and
    ``OutputNode_1`` (value partial, (SPLIT_K,B*H,S_Q,D)); order and names preserved.
    Mirrors ``tests/schedule/end2end/test_mha_split_partial.py``.
    """
    return gqa_split_partial_graph(B, H, H, SPLIT_K, S_Q, S_KV_PARTIAL, D)


def mha_split_combine_graph(H_Q, SPLIT_K, S_Q, D):
    """Split-KV flash-attention combine. Kernel params: rowsum-partials
    ``PlaceHolder_A`` (H_Q,SPLIT_K,S_Q) and value-partials ``PlaceHolder_B``
    (H_Q,SPLIT_K,S_Q,D).

    Sums the per-split rowsum and value partials over the split axis, then divides
    the combined value by the combined rowsum. Mirrors
    ``tests/schedule/end2end/test_mha_split_combine.py``.
    """
    rowsum_combine_0 = Reduction(
        (H_Q, SPLIT_K, S_Q),
        axis=1,
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="rowsum_combine_0",
    )
    value_combine_0 = Reduction(
        (H_Q, SPLIT_K, S_Q, D),
        axis=1,
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="value_combine_0",
    )
    div_0 = Div(
        (H_Q, S_Q, D),
        broadcast=0,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )
    _connect(value_combine_0, div_0, 0, 0)
    _connect(rowsum_combine_0, div_0, 0, 1)
    OutputNode(div_0, dtype=OUT_DTYPE)
    return _finalize(rowsum_combine_0, value_combine_0, div_0)


def rmsnorm_matmul_split_partial_graph(SPLIT_K, M, N, K_PARTIAL):
    """RMSNorm+matmul split-K partial. Kernel params: A ``PlaceHolder_A`` (M,K),
    B ``PlaceHolder_B`` (K,N) with ``K = SPLIT_K * K_PARTIAL``.

    A/B are split on the K axis into ``SPLIT_K`` chunks; each chunk contributes a
    per-split sum-of-squares (for the RMS norm) and a per-split matmul partial.
    TWO outputs: ``OutputNode_0`` (rowsum, (SPLIT_K,M,1)) and ``OutputNode_1``
    (matmul partial, (SPLIT_K,M,N)); order and names are preserved. Mirrors
    ``tests/schedule/end2end/test_rmsnorm_matmul_split_partial.py``.
    """
    K = SPLIT_K * K_PARTIAL

    split_a = SplitView(
        (M, K),
        axis=1,
        factor=SPLIT_K,
        dtype=DTYPE,
        name="split_a",
    )
    split_b = SplitView(
        (K, N),
        axis=0,
        factor=SPLIT_K,
        dtype=DTYPE,
        name="split_b",
    )
    split_a.inputs[0].src_node.name = "PlaceHolder_A"
    split_b.inputs[0].src_node.name = "PlaceHolder_B"

    mul_0 = Mul(
        (SPLIT_K, M, K_PARTIAL),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mul_0",
    )
    rowsum_0 = Reduction(
        (SPLIT_K, M, K_PARTIAL),
        axis=-1,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="rowsum_0",
    )
    mma_0 = Matmul(
        (SPLIT_K, M, N, K_PARTIAL),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_0",
        warp_policy="rows",
    )

    wire_input(split_a, mul_0, 0)
    wire_input(split_a, mul_0, 1)
    wire_input(mul_0, rowsum_0, 0)
    wire_input(split_a, mma_0, 0)
    wire_input(split_b, mma_0, 1)

    out_rowsum = OutputNode(rowsum_0, dtype=OUT_DTYPE)
    out_rowsum.name = "OutputNode_0"
    out_mma = OutputNode(mma_0, dtype=OUT_DTYPE)
    out_mma.name = "OutputNode_1"
    return Graph(nodes=find_topo_sort([out_rowsum, out_mma]))


def rmsnorm_matmul_split_combine_graph(SPLIT_K, M, N, K):
    """RMSNorm+matmul split-K combine. Kernel params: rowsum-partials
    ``PlaceHolder_A`` (SPLIT_K,M) and matmul-partials ``PlaceHolder_B``
    (SPLIT_K,M,N).

    Combines the per-split sum-of-squares and matmul partials over the split axis,
    forms ``inv_rms = 1 / sqrt(sum_sq / K)`` and post-multiplies the combined
    matmul by it. Mirrors ``tests/schedule/end2end/test_rmsnorm_matmul_split_combine.py``.
    """
    from cleave.ops import Sqrt

    rowsum_combine_0 = Reduction(
        (SPLIT_K, M),
        axis=0,
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="rowsum_combine_0",
    )
    mma_combine_0 = Reduction(
        (SPLIT_K, M, N),
        axis=0,
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="mma_combine_0",
    )
    div_0 = Div(
        (M,),
        const=float(K),
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_0",
    )
    sqrt_0 = Sqrt((M,), dtype=ACCUM_DTYPE, out_dtype=ACCUM_DTYPE, name="sqrt_0")
    div_1 = Div(
        (M,),
        const=1.0,
        const_on_left=True,
        dtype=ACCUM_DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="div_1",
    )
    mul_0 = Mul(
        (M, N),
        broadcast=0,
        dtype=ACCUM_DTYPE,
        out_dtype=OUT_DTYPE,
        name="mul_0",
    )
    _connect(rowsum_combine_0, div_0, 0, 0)
    _connect(div_0, sqrt_0, 0, 0)
    _connect(sqrt_0, div_1, 0, 0)
    _connect(mma_combine_0, mul_0, 0, 0)
    _connect(div_1, mul_0, 0, 1)
    OutputNode(mul_0, dtype=OUT_DTYPE)
    return _finalize(rowsum_combine_0, mma_combine_0, div_0, sqrt_0, div_1, mul_0)


def rowsum_graph(M, N):
    """Row-sum reduction over the last axis: A.sum(dim=-1) -> (M,). Kernel param:
    A (M, N). Mirrors ``tests/schedule/end2end/test_rowsum.py``.
    """
    node_0 = Reduction(
        (M, N),
        dtype=DTYPE,
        out_dtype=ACCUM_DTYPE,
        name="rowsum_0",
    )
    OutputNode(node_0, dtype=OUT_DTYPE)
    return _finalize(node_0)


def rowsum_div_graph(S_Q, S_KV):
    """Row-normalise: A / A.sum(dim=-1, keepdim) -> (S_Q, S_KV). Kernel param:
    A (S_Q, S_KV).

    The single input ``A`` feeds both the row-sum (denominator) and the numerator
    of the broadcast divide. Reduction and Div run entirely in ``DTYPE`` (no
    float32 accumulation), matching ``tests/schedule/end2end/test_rowsum_div.py``.
    """
    node_0 = Reduction((S_Q, S_KV), dtype=DTYPE, out_dtype=DTYPE, name="rowsum_0")
    node_1 = Div(
        (S_Q, S_KV),
        broadcast=0,
        dtype=DTYPE,
        out_dtype=DTYPE,
        name="div_0",
    )
    x_node = node_0.inputs[0].src_node
    _connect(x_node, node_1, 0, 0)
    _connect(node_0, node_1, 0, 1)
    OutputNode(node_1, dtype=OUT_DTYPE)
    return _finalize(x_node, node_0, node_1)
