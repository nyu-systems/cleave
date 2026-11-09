"""The dense attention bodies the access layer pages, built at a pinned tiling.

Pinning (rather than searching) the tiling is what reproduces an operator's tuned kernel
byte for byte; the searched path lives in ``demo/dynamic``.
"""

from __future__ import annotations

from tilelang.carver.roller import Edge

from ..ir.graph import Graph, OutputNode
from ..ir.ops import Add, Div, Exp, Matmul, Mul, ReduceSum, Sub
from ..ir.graph_utils import find_topo_sort, gather_all_nodes

DTYPE = "bfloat16"
ACCUM_DTYPE = "float32"

# The KV extent the dense stages tile against. It is a proxy for a length that is only
# known per request at runtime, and the paging pass replaces it with a runtime trip
# count, so its only lasting effect is which KV tile the policy considers legal.
KV_PROXY = 4096


def _connect(src, dst, src_id, dst_id):
    edge = Edge(src, dst, src_id, dst_id)
    src._out_edges.append(edge)
    dst.set_inputs(edge.dst_id, edge)


def _finalize(*roots):
    nodes = gather_all_nodes(*roots)
    return Graph(nodes=find_topo_sort([n for n in nodes if n.is_output()]))


def _softmax_tail(S_Q, S_KV, D, scale, accum_dtype, dtype):
    mul = Mul((S_Q, S_KV), const=scale, dtype=accum_dtype, out_dtype=accum_dtype, name="mul_0")
    sub = Sub(
        (S_Q, S_KV), const=(50.0 * scale), dtype=accum_dtype, out_dtype=accum_dtype, name="sub_0"
    )
    exp = Exp((S_Q, S_KV), dtype=accum_dtype, out_dtype=accum_dtype, name="exp_0")
    value = Matmul(
        (S_Q, D, S_KV), dtype=dtype, out_dtype=accum_dtype, name="mma_1", warp_policy="rows"
    )
    rowsum = ReduceSum((S_Q, S_KV), dtype=accum_dtype, out_dtype=accum_dtype, name="rowsum_0")
    div = Div((S_Q, D), broadcast=0, dtype=accum_dtype, out_dtype=accum_dtype, name="div_0")
    return mul, sub, exp, value, rowsum, div


def flash_attention_sd_graph(
    S_Q, S_KV, D, *, dtype="float16", accum_dtype=ACCUM_DTYPE, out_dtype="float16"
):
    """Single-head flash-attention (safe-exp), no batch dim. Kernel params: Q,K,V (S,D)."""
    scale = 1 / (D**0.5)
    score = Matmul(
        (S_Q, S_KV, D),
        dtype=dtype,
        out_dtype=accum_dtype,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    mul, sub, exp, value, rowsum, div = _softmax_tail(S_Q, S_KV, D, scale, accum_dtype, dtype)
    _connect(score, mul, 0, 0)
    _connect(mul, sub, 0, 0)
    _connect(sub, exp, 0, 0)
    _connect(exp, value, 0, 0)
    _connect(exp, rowsum, 0, 0)
    _connect(value, div, 0, 0)
    _connect(rowsum, div, 0, 1)
    OutputNode(div, dtype=out_dtype)
    return _finalize(score, mul, sub, exp, value, rowsum, div)


def mla_attention_sd_graph(
    S_Q, S_KV, HD, PE, *, dtype="float16", accum_dtype=ACCUM_DTYPE, out_dtype="float16"
):
    """Single-head MLA attention (safe-exp) with RoPE positional scores, no batch dim.

    Placeholders by role: Q ``PlaceHolder_A`` (S_Q,HD), KV ``PlaceHolder_B`` (S_KV,HD),
    Q_pe ``PlaceHolder_C`` (S_Q,PE), K_pe ``PlaceHolder_D`` (S_KV,PE); the value matmul
    reuses KV.
    """
    scale = 1 / (HD**0.5)
    score = Matmul(
        (S_Q, S_KV, HD),
        dtype=dtype,
        out_dtype=accum_dtype,
        name="mma_0",
        transpose_B=True,
        warp_policy="rows",
    )
    score.inputs[0].src_node.name = "PlaceHolder_A"
    kv = score.inputs[1].src_node
    kv.name = "PlaceHolder_B"
    pe = Matmul(
        (S_Q, S_KV, PE),
        dtype=dtype,
        out_dtype=accum_dtype,
        name="mma_pe",
        transpose_B=True,
        warp_policy="rows",
    )
    pe.inputs[0].src_node.name = "PlaceHolder_C"
    pe.inputs[1].src_node.name = "PlaceHolder_D"
    add = Add((S_Q, S_KV), dtype=accum_dtype, out_dtype=accum_dtype, name="add_0")
    mul, sub, exp, value, rowsum, div = _softmax_tail(S_Q, S_KV, HD, scale, accum_dtype, dtype)
    _connect(score, add, 0, 0)
    _connect(pe, add, 0, 1)
    _connect(add, mul, 0, 0)
    _connect(mul, sub, 0, 0)
    _connect(sub, exp, 0, 0)
    _connect(exp, value, 0, 0)
    _connect(kv, value, 0, 1)
    _connect(exp, rowsum, 0, 0)
    _connect(value, div, 0, 0)
    _connect(rowsum, div, 0, 1)
    OutputNode(div, dtype=out_dtype)
    return _finalize(score, pe, add, mul, sub, exp, value, rowsum, div)


def build_tiled_candidate(main_graph, output_tile, rstep_by_name):
    """The TileDict for an explicit (output_tile, rstep-by-node-name) choice."""
    from ..core.target import current_target
    from ..schedule.policy.default import DefaultPolicy, TCPolicy

    output_nodes, has_matmul = [], False
    for node in main_graph.nodes:
        if isinstance(node, Matmul):
            has_matmul = True
        elif isinstance(node, OutputNode):
            output_nodes.append(node)
    policy_cls = TCPolicy if has_matmul else DefaultPolicy
    policy = policy_cls(output_nodes, current_target().arch, flat_graph=main_graph)
    by_name = {
        n.name: n for n in main_graph.nodes if not n.is_placeholder() and not n.is_output()
    }
    cand = policy.compute_tile_dict(
        output_tile, {by_name[name]: rstep for name, rstep in rstep_by_name.items()}
    )
    assert cand.valid, "Constructed TileDict is invalid for the logged candidate."
    return cand


def tiled_dense_attention(dense, query_rows, head_dim, kv_tile, *, steps=None, n_threads=None):
    """``dense`` scheduled at a pinned tiling, ready for the paging pass.

    Reduction steps are keyed by node name; every other op gets an empty one, so a body
    with extra nodes still reaches the paging pass, which is where it should be refused.
    ``n_threads`` pins the block size the real pipeline would search.
    """
    from ..schedule import apply_tiling

    steps = steps or {
        # The score matmul reduces over the whole head dim; KV is stepped one tile per
        # iteration, which becomes the loop's trip count.
        "mma_0": {"v2": head_dim},
        "mma_1": {"v2": kv_tile},
        "rowsum_0": {"v1": kv_tile},
    }
    rstep_by_name = {
        node.name: steps.get(node.name, {})
        for node in dense.nodes
        if not node.is_placeholder() and not node.is_output()
    }
    cand = build_tiled_candidate(dense, [query_rows, head_dim], rstep_by_name)
    if n_threads is not None:
        cand.n_threads = n_threads
    return apply_tiling(cand)


def build_paged_graph(
    spec, *, query_rows=16, head_dim=128, kv_tile=64, kv_proxy=KV_PROXY, n_threads=None
):
    """The scheduled, paged graph for a ``PagingSpec`` at a pinned dense tiling."""
    from .paging import apply_paging

    dense = flash_attention_sd_graph(query_rows, kv_proxy, head_dim, dtype=DTYPE, out_dtype=DTYPE)
    return apply_paging(
        tiled_dense_attention(dense, query_rows, head_dim, kv_tile, n_threads=n_threads), spec
    )


def build_mla_paged_graph(
    spec, *, query_rows=16, ckv=512, kpe=64, kv_tile=64, kv_proxy=KV_PROXY, n_threads=None
):
    """The scheduled, paged graph for an ``MLASpec`` at a pinned dense tiling."""
    from .paging import apply_paging

    dense = mla_attention_sd_graph(query_rows, kv_proxy, ckv, kpe, dtype=DTYPE, out_dtype=DTYPE)
    tiled = tiled_dense_attention(
        dense,
        query_rows,
        ckv,
        kv_tile,
        n_threads=n_threads,
        # Both score matmuls reduce over their whole width; only KV is stepped.
        steps={
            "mma_0": {"v2": ckv},
            "mma_pe": {"v2": kpe},
            "mma_1": {"v2": kv_tile},
            "rowsum_0": {"v1": kv_tile},
        },
    )
    return apply_paging(tiled, spec)


def paged_kernel_source(spec, **kwargs) -> str:
    from ..codegen import TilelangCodeGen

    return TilelangCodeGen().gen(build_paged_graph(spec, **kwargs))


def mla_paged_kernel_source(spec, **kwargs) -> str:
    from ..codegen import TilelangCodeGen

    return TilelangCodeGen().gen(build_mla_paged_graph(spec, **kwargs))
