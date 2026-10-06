"""Wrap a scheduled dense attention graph in a paged/ragged access layer.

The superoptimizer and the tiling policy only ever see dense attention: Q @ K^T, a
softmax, P @ V, over whole tensors whose extents are concrete. That is deliberate --
neither stage has any notion of a per-request KV length, and the search is extent-
agnostic by construction (its "symbolic" graph is a shrunken-integer stand-in for
fingerprint verification, not a runtime-dynamic extent). So the dynamic part of a paged
or ragged operator cannot be searched; it is applied afterwards, here.

What the dense stages contribute is the tiling decision -- how many query rows a block
owns, how wide a KV tile is, how many threads run it -- and those are compile-time
integers that stay valid whatever the request lengths turn out to be. What this pass
contributes is everything that makes the kernel dynamic:

  * the KV tile stops being a slice of a dense tensor and becomes a `PagedGather` off a
    page table, or an affine slice of a ragged one;
  * the query tile is staged by a `RowGather` that zero-fills the rows a short request
    does not have;
  * `Exp` becomes `MaskedExp`, so columns past the request's window contribute nothing,
    and the softmax scale arrives as a runtime scalar instead of a folded constant;
  * the loop over KV tiles gets a runtime trip count (`loop_trip`) read out of an
    indptr, and the grid gets symbolic extents (`dyn_dims`), so one compiled kernel
    serves every request shape.

The concrete extent the scheduler tiled against ("the proxy") appears nowhere in the
result: it reached only `graph.extent` and the inner tier's `extent`, and this pass
overwrites both. The tile sizes it led the policy to choose are what survive, which is
why the proxy is load-bearing for performance and irrelevant to correctness.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from dataclasses import fields as spec_fields

from tilelang.carver.roller import Edge

from . import passes
from .graph import (
    AccumEdge,
    Graph,
    ScalarScan,
    GraphNode,
    Iterator,
    OutputNode,
    PlaceHolderNode,
)
from .ops import (
    Accumulate,
    Add,
    CausalMaskedExp,
    ConstFill,
    Copy,
    Div,
    Exp,
    IndexGather,
    IndptrTileScan,
    Log2,
    MaskedDiv,
    MaskedExp,
    MaskedLog2,
    Matmul,
    Mul,
    PagedGather,
    RangeGather,
    Reduction,
    RowGather,
    Sub,
)
from .ops.gather import BUFFER_BOUND

# exp2 is a single instruction where exp is a call, so the scale rides in the log2
# domain: exp(x * s) == exp2(x * s * LOG2E).
LOG2E = 1.4426950408889634
# Floors the softmax denominator. A request whose KV window is empty runs zero KV tiles,
# leaving a zero numerator over a zero denominator; the floor turns that 0/0 into the 0
# the operator defines rather than the NaN a benchmark's sanity check would reject.
DENOM_FLOOR = 1e-30



# The access layers this module can build, and the operator families they serve.
PAGED_DECODE = "paged_decode"
# Same operator as PAGED_DECODE, different access layer: at page_size > 1 a KV tile lies
# inside a single page, so the tile is a contiguous slab addressed by one page-table entry
# rather than BN gathered rows. That is not a knob on the decode layer -- it changes what
# the page table means (pages, not tokens), adds a sixth input to recover the token length
# of the last page, and swaps the gathers for BufferRegions -- so it is its own kind.
PAGED_DECODE_SLAB = "paged_decode_slab"
PAGED_PREFILL_CAUSAL = "paged_prefill_causal"
RAGGED_PREFILL_CAUSAL = "ragged_prefill_causal"
KINDS = (
    PAGED_DECODE,
    PAGED_DECODE_SLAB,
    PAGED_PREFILL_CAUSAL,
    RAGGED_PREFILL_CAUSAL,
)
# The layers that mask against the diagonal, and so need the per-row causal bound and the
# row predicate on their stores.
CAUSAL_KINDS = (PAGED_PREFILL_CAUSAL, RAGGED_PREFILL_CAUSAL)
# The layers that read a page table. A ragged operator's keys are already contiguous per
# request, so it has none -- and therefore no gather, only an affine slice.
PAGED_KINDS = (PAGED_DECODE, PAGED_DECODE_SLAB, PAGED_PREFILL_CAUSAL)
# The layers whose page table is indexed in pages rather than tokens.
SLAB_KINDS = (PAGED_DECODE_SLAB,)
# The layers split-KV is built for: one block per (request, kv head) with the whole KV
# window to walk, which is where an extra grid axis over KV slices adds parallelism. Set
# after the MLA kinds are declared below.
SPLIT_KINDS = ()

# Which `PagingSpec` fields each layer actually reads, for the fields that only some of
# them do. Derived from the builders (including the tensor-map helpers they call) and used
# by `validate` to refuse a field the chosen layer would silently drop -- the alternative
# being a caller who renames an input, sees no error, and gets a kernel that ignored the
# rename. Everything not listed here is read by every layer.
_OPTIONAL_FIELDS = (
    "batch_dim",
    "tokens_dim",
    "kv_tokens_dim",
    "pages_dim",
    "index_dim",
    "index_input",
    "last_page_len_input",
    "query_indptr_input",
    "grid_order",
)
_FIELDS_READ = {
    # A decode block is one (request, kv head) pair, so it has both grid axes to order.
    PAGED_DECODE: {"batch_dim", "pages_dim", "index_dim", "index_input", "grid_order"},
    PAGED_DECODE_SLAB: {
        "batch_dim",
        "pages_dim",
        "index_dim",
        "index_input",
        "grid_order",
        "last_page_len_input",
    },
    # A prefill block is one work-list slot, whose leading extent is a token count; the
    # request it belongs to is found by scanning the query indptr on the device.
    PAGED_PREFILL_CAUSAL: {
        "tokens_dim",
        "pages_dim",
        "index_dim",
        "index_input",
        "query_indptr_input",
    },
    # Ragged keys are packed like the queries: a second token axis in place of the page
    # pool, and so no page table and no page-table extents.
    RAGGED_PREFILL_CAUSAL: {"tokens_dim", "kv_tokens_dim", "query_indptr_input"},
}


def _assert_fields_are_read(spec, optional, read) -> None:
    """Refuse a field the chosen layer does not read, unless it is still at its default.

    Shared by both specs on purpose. The two ``validate`` methods drifted once already --
    ``MLASpec`` grew per-kind namespace checks that ``PagingSpec`` did not, and the gap let
    a prefill spec rename its query indptr onto its kv indptr and emit a kernel that had
    silently lost a parameter -- so anything both of them owe the caller lives here.
    """
    unread = []
    for spec_field in spec_fields(spec):
        if spec_field.name not in optional or spec_field.name in read:
            continue
        value, default = getattr(spec, spec_field.name), spec_field.default
        # A `default_factory` field has no `default` -- it is MISSING, which nothing equals,
        # so listing one as optional would reject every value including the intended one.
        assert default is not MISSING, (
            f"{spec_field.name} is built by a default_factory, so 'still at its default' "
            "is not a comparison this can make; it must not be listed as optional"
        )
        if isinstance(default, tuple):
            value = tuple(value)
        if value != default:
            unread.append(f"{spec_field.name}={value!r}")
    assert not unread, (
        f"the {spec.kind!r} layer does not read {', '.join(unread)}, so setting them "
        "would have no effect on the emitted kernel"
    )


@dataclass(frozen=True)
class PagingSpec:
    """The dynamic-access layer to wrap a dense attention body in.

    Everything here is a property of the *operator*, not of the schedule: the schedule's
    contribution is read off the graph this is applied to.
    """

    num_qo_heads: int
    num_kv_heads: int
    # Which access layer to build. These are not variations of one layer -- a paged decode
    # gathers rows off a page table while a prefill also scans a work list, masks causally
    # and fills the rows whose causal window is empty -- so each has its own builder and
    # its own required inputs, and `validate` refuses a field the chosen layer cannot
    # honour rather than quietly ignoring it.
    kind: str = "paged_decode"
    page_size: int = 1
    # Names for the inputs the layer introduces. They become the emitted kernel's
    # parameter names, so they are part of the operator's declared signature.
    index_input: str = "KV_INDICES"
    indptr_input: str = "KV_INDPTR"
    # A prefill request has its own variable query length, so it carries a second CSR
    # indptr and the work list is scanned from it on the device.
    query_indptr_input: str = "QO_INDPTR"
    # At page_size > 1 the indptr counts pages, which rounds a request's KV length up to a
    # whole page; only this input says how many tokens of the last page are live.
    last_page_len_input: str = "KV_LAST_PAGE_LEN"
    query_input: str = "Q"
    cache_inputs: tuple = ("K_CACHE", "V_CACHE")
    scale_input: str = "sm_scale"
    causal: bool = False
    emit_lse: bool = True
    # Symbolic extent names. Each must appear bare in some input tensor's shape so the
    # runtime can infer it; the defaults are the ones the shipped schedules use.
    batch_dim: str = "batch"
    # Prefill packs every request's queries into one ragged axis, so its leading extent is
    # a token count rather than a request count.
    tokens_dim: str = "total_q"
    # A ragged operator's keys are packed the same way its queries are, so its cache is a
    # token axis rather than a page pool.
    kv_tokens_dim: str = "total_kv"
    pages_dim: str = "num_pages"
    indptr_dim: str = "len_indptr"
    index_dim: str = "n_idx"
    # All 9 shipped kernels clamp every read themselves, so tilelang's bounds
    # predication is redundant -- and actively harmful, because it re-materialises the
    # page-table entry once per bounds test when that entry sits inside the address of
    # the cache read: three global loads per row instead of one.
    pass_configs: dict = field(
        default_factory=lambda: {"tl.disable_safe_memory_legalize": True}
    )
    # Whether to rasterise the grid. Measured off for every paged operator (the blocks
    # do not share enough KV to repay the per-CTA index remap) and on for ragged.
    # None defers to the codegen default (an L2 swizzle of 16). Measured off for every
    # paged operator -- the blocks do not share enough KV to repay the per-CTA index remap
    # -- and on for ragged, where they do.
    rasterization: int | None = 1
    # `cols` for every tuned schedule; the integration layer hands the search `rows`,
    # which is the opposite, so this is a wrapper decision rather than an inherited one.
    warp_policy: str = "cols"
    # Which grid axis varies fastest. Both orders ship: kv8 decode is ("batch", "head")
    # and kv4 is ("head", "batch"), a measured difference of a few percent that depends
    # on how much page table neighbouring blocks share, so it stays a declared choice.
    grid_order: tuple = ("batch", "head")

    @property
    def group(self) -> int:
        """Query heads per kv head -- the rows of one block's query tile that are real."""
        return self.num_qo_heads // self.num_kv_heads

    def validate(self) -> None:
        for name, value in (
            ("num_qo_heads", self.num_qo_heads),
            ("num_kv_heads", self.num_kv_heads),
            ("page_size", self.page_size),
        ) + ((("rasterization", self.rasterization),) if self.rasterization is not None else ()):
            assert isinstance(value, int) and not isinstance(value, bool), (
                f"{name} must be an int, got {value!r}"
            )
        assert self.num_qo_heads > 0 and self.num_kv_heads > 0, (
            f"head counts must be positive, got {self.num_qo_heads}/{self.num_kv_heads}"
        )
        assert self.num_qo_heads % self.num_kv_heads == 0, (
            f"{self.num_qo_heads} query heads do not divide into {self.num_kv_heads} "
            "kv heads, so a block's query tile would not be a whole head group"
        )
        # A block's query tile is one kv head's group of query heads, which needs at
        # least two rows to tile and at least two kv heads for the head axis to survive
        # codegen's unit-extent squeeze. Both edges emit a kernel that does not compile.
        assert self.num_kv_heads > 1, (
            "a single kv head leaves the cache with no head axis, which the gather's "
            "head scalar then indexes past the end of; use the MLA layer instead"
        )
        # A group of 1 squeezes the decode layers' (group, head_dim) output tile to a rank
        # the epilogue cannot write. The ragged layer gives one CTA a single query head and
        # writes (rows, 1, head_dim), so plain MHA is expressible there -- the group reaches
        # only its `kv_head = h // group` -- and this gate would reject it for nothing.
        if self.kind != RAGGED_PREFILL_CAUSAL:
            assert self.group > 1, (
                f"{self.num_qo_heads} query heads over {self.num_kv_heads} kv heads leaves "
                "a group of 1, whose output tile squeezes to a rank the epilogue cannot "
                "write"
            )
        # page_size is not a free parameter of a layer: it selects one. The row-gather
        # layers read a token-unit indptr and address the cache as one token per entry, so
        # a page size above 1 there would attend to one token per page. The slab layer
        # reads a page-unit indptr and needs a page to hold more than one token for the
        # address split to exist at all.
        if self.kind in SLAB_KINDS:
            assert self.page_size > 1, (
                f"page_size={self.page_size} makes the slab layer's address split "
                f"degenerate; use kind={PAGED_DECODE!r} for one token per page"
            )
        else:
            assert self.page_size == 1, (
                f"page_size={self.page_size} needs the slab access layer "
                f"(kind={PAGED_DECODE_SLAB!r}) and its {self.last_page_len_input} input; "
                f"only page_size=1 is expressible by the {self.kind!r} layer"
            )
        assert len(self.cache_inputs) == 2, (
            f"expected a (K, V) cache pair, got {self.cache_inputs}"
        )
        assert self.kind in KINDS, f"unknown access layer {self.kind!r}; expected {KINDS}"
        # A field the chosen layer does not read is refused rather than quietly ignored --
        # the promise made where `kind` is declared.
        read = _FIELDS_READ[self.kind]
        _assert_fields_are_read(self, _OPTIONAL_FIELDS, read)
        # Every declared name shares one namespace: a tensor and the scale collide in the
        # kernel signature, and an extent colliding with the scale makes a float32 the grid
        # bound. Only the names this kind emits take part, so a decode spec is not rejected
        # over a page-length input it never writes.
        names = [
            self.indptr_input,
            self.query_input,
            self.scale_input,
            *self.cache_inputs,
        ] + [
            getattr(self, name)
            for name in ("index_input", "last_page_len_input", "query_indptr_input")
            if name in read
        ]
        assert len(set(names)) == len(names), f"input names collide: {names}"
        dims = [self.indptr_dim, self.scale_input] + [
            getattr(self, name)
            for name in ("batch_dim", "tokens_dim", "kv_tokens_dim", "pages_dim", "index_dim")
            if name in read
        ]
        assert len(set(dims)) == len(dims), f"symbolic extent names collide: {dims}"
        assert self.warp_policy in ("rows", "cols", "even"), (
            f"unsupported warp policy {self.warp_policy!r}"
        )
        assert tuple(self.grid_order) in (("batch", "head"), ("head", "batch")), (
            f"grid_order must permute ('batch', 'head'), got {self.grid_order}"
        )
        # `causal` is not independent of the layer: masking against the diagonal needs the
        # per-row limit and the row predicate that only the prefill layer builds.
        assert self.causal == (self.kind in CAUSAL_KINDS), (
            f"causal={self.causal} does not match the {self.kind!r} layer; only the causal "
            "prefill layers mask against the diagonal"
        )


MLA_PAGED_DECODE = "mla_paged_decode"
MLA_PAGED_PREFILL_CAUSAL = "mla_paged_prefill_causal"
MLA_KINDS = (MLA_PAGED_DECODE, MLA_PAGED_PREFILL_CAUSAL)
SPLIT_KINDS = (PAGED_DECODE, PAGED_DECODE_SLAB, MLA_PAGED_DECODE)

# `_OPTIONAL_FIELDS` / `_FIELDS_READ` for the latent layers. A prefill block is one work
# slot over a ragged token axis and finds its request by scanning the query indptr; a decode
# block is one request, indexed directly.
_MLA_OPTIONAL_FIELDS = ("batch_dim", "tokens_dim", "query_indptr_input")
_MLA_FIELDS_READ = {
    MLA_PAGED_DECODE: {"batch_dim"},
    MLA_PAGED_PREFILL_CAUSAL: {"tokens_dim", "query_indptr_input"},
}


@dataclass(frozen=True)
class MLASpec:
    """The dynamic-access layer for a paged multi-head *latent* attention operator.

    A separate spec from ``PagingSpec``, not a mode of it, because almost nothing carries
    over. MLA absorbs the key/value projections into one compressed cache, so:

    * there is **one** kv head, and therefore no head group, no group divisibility and no
      head axis to put on the grid -- ``PagingSpec`` asserts the opposite of all three;
    * there are **two** caches of different widths (``ckv`` and the RoPE ``kpe``) and two
      query tensors to match, feeding two score matmuls whose results are summed;
    * the compressed cache **is** the value tensor, so it is read twice per tile.

    Fields that do carry over keep their names and meanings, and the tiling still comes off
    the dense graph rather than from here.
    """

    num_qo_heads: int
    kind: str = "mla_paged_decode"
    page_size: int = 1
    # KV tiles served by one shared page-table window. The gathers need one entry per
    # gathered row; read from global that is a dependent load in front of every cache read,
    # so it is staged in shared once per `index_window_tiles` tiles instead. Worth it here
    # and nowhere else because the cache tile is BN x 512, wide enough to evict the page
    # table from L1 between tiles.
    index_window_tiles: int = 8
    # Read one tile ahead out of that same window. Only pays together with a 256-thread
    # block (the BN x 512 tile spills as a fragment at 128), so the two are one decision.
    prefetch: bool = True
    query_inputs: tuple = ("QN", "QP")
    cache_inputs: tuple = ("CKV", "KPE")
    index_input: str = "KVI"
    indptr_input: str = "KV_INDPTR"
    query_indptr_input: str = "QO_INDPTR"
    scale_input: str = "sm_scale"
    causal: bool = False
    emit_lse: bool = True
    batch_dim: str = "batch"
    tokens_dim: str = "total_q"
    pages_dim: str = "num_pages"
    indptr_dim: str = "len_indptr"
    index_dim: str = "n_idx"
    pass_configs: dict = field(
        default_factory=lambda: {"tl.disable_safe_memory_legalize": True}
    )
    # None defers to the codegen default (an L2 swizzle of 16), which is what the tuned
    # decode kernel uses; the prefill kernel measured better with it off.
    rasterization: int | None = None
    warp_policy: str = "cols"

    def validate(self) -> None:
        for name, value in (
            ("num_qo_heads", self.num_qo_heads),
            ("page_size", self.page_size),
            ("index_window_tiles", self.index_window_tiles),
        ) + (
            (("rasterization", self.rasterization),)
            if self.rasterization is not None
            else ()
        ):
            assert isinstance(value, int) and not isinstance(value, bool), (
                f"{name} must be an int, got {value!r}"
            )
        # MLA puts one token's whole block of query heads in a single tile, and
        # `apply_paging` requires the tile to be exactly that block -- so the head count
        # *is* the matmul's M extent and inherits its granularity. Measured: 2 heads fails
        # with "stride == 4 || stride % 8 == 0", 4 and 8 with "M must be divisible by 16",
        # while 16, 32 and 64 compile on both layers.
        assert self.num_qo_heads >= 16 and self.num_qo_heads % 16 == 0, (
            f"{self.num_qo_heads} query heads is not a positive multiple of 16; MLA's query "
            "tile is the head block itself, so the head count is the matmul's M extent and "
            "a tensor-core fragment cannot be built from any other value"
        )
        assert self.index_window_tiles >= 1, (
            f"index_window_tiles={self.index_window_tiles} leaves no window to stage"
        )
        # The prefetch reads the *next* tile's entries out of the current window, so it
        # needs a window wider than one tile. At exactly one the emitted guard is
        # `win_off + bn < bn` with win_off always 0 -- never true, so the prefetch is dead
        # code, and yet it still forces the 256-thread block below.
        assert self.index_window_tiles > 1 or not self.prefetch, (
            "index_window_tiles=1 leaves the prefetch no next tile to read inside the "
            "window, so it would never fire; set prefetch=False or widen the window"
        )
        # The compressed cache carries only (page, slot, dim) -- MLA has no head axis for
        # the gather to index -- and at page_size 1 there is no slot axis either, which is
        # the shape the gather macro is written against.
        assert self.page_size == 1, (
            f"page_size={self.page_size} would give the MLA cache a slot axis the gather "
            "does not index; only page_size=1 is expressible here"
        )
        assert len(self.query_inputs) == 2, (
            f"expected a (nope, pe) query pair, got {self.query_inputs}"
        )
        assert len(self.cache_inputs) == 2, (
            f"expected a (ckv, kpe) cache pair, got {self.cache_inputs}"
        )
        assert self.warp_policy in ("rows", "cols", "even"), (
            f"unsupported warp policy {self.warp_policy!r}"
        )
        # Before the namespace checks, which are per-kind and would otherwise treat an
        # unknown kind as a decode.
        assert self.kind in MLA_KINDS, (
            f"unknown MLA access layer {self.kind!r}; expected {MLA_KINDS}"
        )
        read = _MLA_FIELDS_READ[self.kind]
        _assert_fields_are_read(self, _MLA_OPTIONAL_FIELDS, read)
        names = [
            *self.query_inputs,
            *self.cache_inputs,
            self.index_input,
            self.indptr_input,
            self.scale_input,
        ] + ([self.query_indptr_input] if "query_indptr_input" in read else [])
        assert len(set(names)) == len(names), f"input names collide: {names}"
        dims = [
            self.pages_dim,
            self.indptr_dim,
            self.index_dim,
            self.scale_input,
            self.tokens_dim if "tokens_dim" in read else self.batch_dim,
        ]
        assert len(set(dims)) == len(dims), f"symbolic extent names collide: {dims}"
        assert self.causal == (self.kind == MLA_PAGED_PREFILL_CAUSAL), (
            f"causal={self.causal} does not match the {self.kind!r} layer; only the causal "
            "prefill layer masks against the diagonal"
        )
        # The prefetch needs the wide block that keeps the ckv tile out of spill; the
        # prefill layer's tiles are one query block of 16 rows, too little to fill 8 warps,
        # so it runs 128 threads and must not prefetch. That pairing is measured, and the
        # thread count is read off the dense graph, so it is checked in the builder.


@dataclass(frozen=True)
class DenseMLA:
    """The tiling the dense stages chose, read off a scheduled dense MLA graph."""

    query_rows: int
    kv_tile: int
    head_dim_ckv: int
    head_dim_kpe: int
    n_threads: int
    dtype: str
    accum_dtype: str
    out_dtype: str


def read_dense_mla(graph: Graph) -> DenseMLA:
    """Recognise a scheduled dense MLA graph and extract its tiling.

    Same contract as ``read_dense_attention`` -- the layer rebuilds the body and keeps only
    the tiling, so every edge it relies on is checked here or nothing downstream will
    notice. What differs is the body being recognised: two transpose_B score matmuls whose
    results are *added*, and a value matmul whose operand is the very placeholder the wider
    score matmul reduced against.
    """
    inner_node = _sole(
        [n for n in graph.nodes if isinstance(n, GraphNode)], "reduction subgraph"
    )
    inner = inner_node.graph
    assert len(inner.accum_nodes) == 2, (
        "expected two accumulators (the row sum and the value matmul), got "
        f"{[n.name for n in inner.accum_nodes]}"
    )
    rowsum = _sole(
        [n for n in inner.accum_nodes if isinstance(n, Reduction)], "row-sum accumulator"
    )
    value_mma = _sole(
        [n for n in inner.accum_nodes if isinstance(n, Matmul)], "value matmul"
    )
    score_mmas = [
        n
        for n in inner.nodes
        if isinstance(n, Matmul) and n is not value_mma and n.transpose_B
    ]
    assert len(score_mmas) == 2, (
        "expected two score matmuls (compressed kv and positional), got "
        f"{[n.name for n in score_mmas]}"
    )
    exp = _sole([n for n in inner.nodes if isinstance(n, Exp)], "softmax exp")
    div = _sole([n for n in graph.nodes if isinstance(n, Div)], "softmax divide")
    out = _sole([n for n in graph.nodes if isinstance(n, OutputNode)], "output")

    # --- the dataflow --------------------------------------------------------------
    #
    # (Q @ CKV^T) + (Q_pe @ KPE^T) -> constant scaling -> exp -> {row sum, @ CKV} -> /.
    add = _sole(
        [n for n in inner.nodes if isinstance(n, Add) and len(n.inputs) == 2],
        "score sum (the two matmuls added)",
    )
    added = {id(edge.src_node) for edge in add.inputs}
    assert added == {id(score_mmas[0]), id(score_mmas[1])}, (
        "the score sum does not add the two score matmuls; this body is not the MLA "
        "attention the access layer would rebuild"
    )
    scale_chain = _chain_to(exp, add)
    assert scale_chain is not None, (
        "the softmax exp does not read the summed scores through constant scaling alone"
    )
    assert rowsum.inputs[0].src_node is exp, (
        f"the row sum reduces {rowsum.inputs[0].src_node.name}, not the softmax exp"
    )
    assert value_mma.inputs[0].src_node is exp, (
        f"the value matmul consumes {value_mma.inputs[0].src_node.name}, not the exp"
    )
    assert not value_mma.transpose_B, "the value matmul must not transpose the cache"
    # As in `read_dense_attention`: the returned tiling carries one dtype pair, taken from
    # the ckv score matmul, so the three matmuls have to agree or the layer would rebuild a
    # mixed-precision body in a single precision.
    for other in (score_mmas[0], score_mmas[1], value_mma):
        assert (str(other.dtype), str(other.out_dtype)) == (
            str(score_mmas[0].dtype),
            str(score_mmas[0].out_dtype),
        ), (
            f"{other.name} runs {other.dtype}/{other.out_dtype} but {score_mmas[0].name} "
            f"runs {score_mmas[0].dtype}/{score_mmas[0].out_dtype}; the access layer builds "
            "one precision for all three matmuls"
        )
    assert getattr(div, "broadcast", None) == 0, (
        f"the divide must broadcast the denominator over rows, got {div.broadcast}"
    )
    # Ordered, not a set: a body that divides the row sum BY the value matmul carries the
    # same two accumulators, and the layer -- which always builds the numerator from the
    # value matmul -- would rebuild it the right way round and silently compute something
    # else. `apply_tiling` happens to reject the swapped body earlier today, so this is a
    # guard against that changing rather than a path any test can reach.
    accums = [getattr(edge, "accum_node", None) for edge in div.inputs]
    assert accums[:2] == [value_mma, rowsum], (
        "the divide must take the value matmul over the row sum, in that order; got "
        f"{[None if n is None else n.name for n in accums]}"
    )
    _assert_output_is_the_divide(graph, div, out)
    placeholders = [n for n in inner.nodes if isinstance(n, PlaceHolderNode)]
    assert len(placeholders) == 4, (
        "expected exactly the two query tensors and the two caches in the loop tier, got "
        f"{[n.name for n in placeholders]}"
    )
    # Both operands of each score matmul, but only the cache operand of the value matmul --
    # its other one is the exp, checked above.
    operands = [(mma, slot) for mma in score_mmas for slot in (0, 1)]
    operands.append((value_mma, 1))
    for mma, slot in operands:
        assert isinstance(mma.inputs[slot].src_node, PlaceHolderNode), (
            f"{mma.name}'s operand {slot} is "
            f"{mma.inputs[slot].src_node.name}, not an input"
        )
    # The compressed cache is the value tensor: that aliasing is the whole point of MLA,
    # and it decides which of the two score matmuls is the ckv one.
    cache = value_mma.inputs[1].src_node
    ckv_mma = _sole(
        [n for n in score_mmas if n.inputs[1].src_node is cache],
        "score matmul reading the value cache (MLA aliases ckv as V)",
    )
    pe_mma = score_mmas[0] if score_mmas[1] is ckv_mma else score_mmas[1]
    # No assert that the two score matmuls read different caches or different queries: the
    # `_sole` above already refuses a shared cache (two matmuls would read it, so it would
    # not be sole), and a shared query leaves three distinct tensors in the loop tier, which
    # the four-placeholder census refuses. Both were checked here as well until a mutation
    # audit proved the conditions are tautologies at this point.

    # --- the tiling ----------------------------------------------------------------
    score_tile = graph_tile(inner, ckv_mma)
    pe_tile = graph_tile(inner, pe_mma)
    value_tile = graph_tile(inner, value_mma)
    assert len(score_tile) == 2 and len(value_tile) == 2, (
        f"expected 2-D score/value tiles, got {score_tile}/{value_tile}"
    )
    assert score_tile == pe_tile, (
        f"the two score matmuls disagree on the tile: {score_tile} vs {pe_tile}"
    )
    assert score_tile[0] == value_tile[0], (
        f"score and value matmuls disagree on the query tile: {score_tile} vs {value_tile}"
    )
    head_dim_ckv = int(value_mma.dims[-2])
    head_dim_kpe = int(pe_mma.dims[-1])
    assert int(value_tile[1]) == head_dim_ckv, (
        f"the compressed head dim is split into {value_tile[1]}-wide blocks (of "
        f"{head_dim_ckv}); the access layer replaces the grid, so those blocks would be "
        "lost. Re-tile with the whole head dim in one block."
    )
    assert int(graph_tile(inner, ckv_mma.inputs[0].src_node)[1]) == head_dim_ckv, (
        "the ckv score matmul reduces over part of the compressed head dim; it must "
        "reduce over all of it"
    )
    assert int(graph_tile(inner, pe_mma.inputs[0].src_node)[1]) == head_dim_kpe, (
        "the positional score matmul reduces over part of the RoPE head dim; it must "
        "reduce over all of it"
    )
    assert graph.extent in (1, (1,)), (
        f"the dense schedule has a grid of {graph.extent}, which paging would discard; "
        "it must tile to a single block"
    )
    # The dense body's constant has to be the conventional scale for its own reduction
    # width. The paged kernel does not keep it -- MLA's true scale comes from the
    # pre-absorption qk width and a YaRN correction, neither derivable from the compressed
    # width, so it arrives as a runtime input -- but a body scaled by anything else is a
    # body computing something other than attention, and that is worth rejecting.
    folded = _folded_scale(scale_chain)
    expected = head_dim_ckv**-0.5
    assert abs(folded - expected) <= 1e-6 * max(1.0, abs(expected)), (
        f"the dense body scales the scores by {folded}, not 1/sqrt({head_dim_ckv}) = "
        f"{expected}; the runtime scale this layer substitutes would not match"
    )
    return DenseMLA(
        query_rows=int(score_tile[0]),
        kv_tile=int(score_tile[1]),
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        n_threads=int(getattr(graph, "n_threads", 128)),
        dtype=str(ckv_mma.dtype),
        accum_dtype=str(ckv_mma.out_dtype),
        out_dtype=str(out.get_dtype(0)),
    )


@dataclass
class _Built:
    """The nodes ``_build_paged_decode`` created, for the tensor-map step to key on."""

    ph_q: object
    ph_k: object
    ph_v: object
    ph_idx: object
    ph_indptr: object
    loop_node: object
    inner_q: object
    inner_idx: object
    gather_k: object
    gather_v: object
    score_mma: object
    exp: object
    rowsum: object
    value_mma: object
    div: object
    log2: object
    outputs: list
    # Split-KV partial only: the number of splits and the two stores that replace the
    # divide and the log-sum-exp. None/absent for the single-kernel layer.
    split: int = None
    store_d: object = None
    store_o: object = None


@dataclass(frozen=True)
class DenseAttention:
    """The tiling the dense stages chose, read off a scheduled attention graph."""

    query_rows: int  # M tile: query rows per block
    kv_tile: int  # N tile of the score matmul: KV rows per iteration
    head_dim: int
    n_threads: int
    dtype: str
    accum_dtype: str
    out_dtype: str


@dataclass(frozen=True)
class CombineTiling:
    """The tiling the dense stages chose for a split-KV *combine* body.

    A dense combine sums ``split`` partial numerators ``(split, M, N)`` and denominators
    ``(split, M)`` and divides. What the tiling decides is how many partials one loop
    iteration folds (``split_step``), how many output columns one block owns (``cols``)
    and the block size. The rows a block owns are not a knob here: in the dynamic kernel
    they are the head group (or, for MLA, the whole head block), fixed by the operator.
    """

    split: int
    split_step: int
    cols: int
    n_threads: int
    # Serial by default: a pipelined loop over the partial slices dies in tilelang's TMA
    # lowering on the symbolic-batch partial tensor (an int32 overflow building the
    # descriptor), and the loop is a handful of iterations over tiles already in L2.
    loop_kind: str = "serial"
    loop_stages: int = 2
    # Where a partial slice is staged before the reduction reads it: "s" copies it to
    # shared first (what `annotate_tinfo` gives a searched combine), "g" lets the
    # reduction read global straight into its fragment.
    input_scope: str = "s"

    def validate(self, head_dim: int) -> None:
        assert self.split > 1, f"a split factor of {self.split} is not a split"
        assert 1 <= self.split_step <= self.split and self.split % self.split_step == 0, (
            f"split_step={self.split_step} does not divide split={self.split}"
        )
        assert 1 <= self.cols <= head_dim and head_dim % self.cols == 0, (
            f"cols={self.cols} does not divide the head dim {head_dim}"
        )
        assert self.loop_kind in ("serial", "pipelined"), self.loop_kind
        assert self.input_scope in ("s", "g"), self.input_scope


def _sole(nodes, what):
    nodes = list(nodes)
    assert len(nodes) == 1, f"expected exactly one {what}, found {[n.name for n in nodes]}"
    return nodes[0]


def _is_constant_scaling(node) -> bool:
    """Whether ``node`` scales or shifts its input by a compile-time constant.

    A constant is foldable into the runtime softmax scale; a second *tensor* operand is a
    bias or a mask, which the access layer would drop from the kernel's signature
    entirely, so those must not be mistaken for scaling.
    """
    return (
        isinstance(node, (Mul, Div, Add, Sub))
        and len(node.inputs) == 1
        and getattr(node, "const", None) is not None
    )


def _chain_to(node, target):
    """The constant-scaling nodes between ``target`` and ``node``, walking input 0.

    None when the path is anything else -- an extra tensor operand, another matmul, a
    second nonlinearity.
    """
    chain, current = [], node
    for _ in range(8):
        source = current.inputs[0].src_node
        if source is target:
            return chain
        if not _is_constant_scaling(source):
            return None
        chain.append(source)
        current = source
    return None


def _folded_scale(chain) -> float:
    """The multiplier ``chain`` applies to the scores; shifts do not affect it.

    A shift is dropped by the access layer, which is sound: softmax is shift-invariant,
    and the paged operator's log-sum-exp is defined without one (none of the shipped
    kernels subtract a max).
    """
    scale = 1.0
    for node in chain:
        if isinstance(node, Mul):
            scale *= float(node.const)
        elif isinstance(node, Div):
            assert not getattr(node, "const_on_left", False), (
                "the scores are divided INTO a constant, which is not a scaling"
            )
            scale /= float(node.const)
    return scale


def _assert_output_is_the_divide(graph: Graph, div, out) -> None:
    """The dense body must *end* at the softmax divide.

    The ``accums`` check constrains what feeds the divide; this constrains what consumes
    it. Both access layers store the divide's tile straight to the output, so an op the
    dense body applies after the divide -- a scaling, an activation, anything -- is not
    slower in the paged kernel, it is absent from it. That is a wrong kernel produced from
    a graph the recognizer accepted, which is the one failure mode this whole module is
    written to avoid.
    """
    assert out.inputs[0].src_node is div, (
        f"the output is produced by {out.inputs[0].src_node.name}, not the softmax divide; "
        "the access layer stores the divide's tile, so whatever the dense body computes "
        "after it would be dropped from the paged kernel"
    )
    # Closing the class rather than the one instance: the outer tier is rebuilt as exactly
    # the reduction loop plus the divide, so any other op there is discarded too, even one
    # that does not sit between the divide and the output.
    extra = [
        n
        for n in graph.nodes
        if n is not div
        and not isinstance(n, (GraphNode, PlaceHolderNode, OutputNode))
    ]
    assert not extra, (
        f"the outer tier also holds {[n.name for n in extra]}; the access layer rebuilds "
        "only the reduction loop and the softmax divide, and would drop these"
    )


def read_dense_attention(graph: Graph) -> DenseAttention:
    """Recognise a scheduled dense attention graph and extract its tiling.

    The access layer rebuilds the body from the spec and keeps only the tiling, so this
    has to be a real recognizer rather than a type census: anything it accepts and does
    not check is something the emitted kernel silently disagrees with. A dense graph that
    computes a biased, differently scaled or differently wired attention must be rejected
    here, because nothing downstream will notice.
    """
    inner_node = _sole(
        [n for n in graph.nodes if isinstance(n, GraphNode)], "reduction subgraph"
    )
    inner = inner_node.graph
    assert len(inner.accum_nodes) == 2, (
        "expected two accumulators (the row sum and the value matmul), got "
        f"{[n.name for n in inner.accum_nodes]}"
    )
    rowsum = _sole(
        [n for n in inner.accum_nodes if isinstance(n, Reduction)], "row-sum accumulator"
    )
    value_mma = _sole(
        [n for n in inner.accum_nodes if isinstance(n, Matmul)], "value matmul"
    )
    score_mma = _sole(
        [
            n
            for n in inner.nodes
            if isinstance(n, Matmul) and n is not value_mma and n.transpose_B
        ],
        "score matmul (transpose_B)",
    )
    exp = _sole([n for n in inner.nodes if isinstance(n, Exp)], "softmax exp")
    div = _sole([n for n in graph.nodes if isinstance(n, Div)], "softmax divide")
    # The value dtype is the OutputNode's, not the divide's: the divide accumulates in
    # fp32 and the store is what casts.
    out = _sole([n for n in graph.nodes if isinstance(n, OutputNode)], "output")

    # --- the dataflow, not just the node census ------------------------------------
    #
    # Q @ K^T -> (constant scaling) -> exp -> {row sum, @ V} -> divide. Every edge is
    # checked, because the layer re-creates the body and would otherwise paper over a
    # graph that computes something else.
    scale_chain = _chain_to(exp, score_mma)
    assert scale_chain is not None, (
        "the softmax exp does not read the score matmul through constant scaling alone; "
        "this body is not the attention the access layer would rebuild"
    )
    assert rowsum.inputs[0].src_node is exp, (
        f"the row sum reduces {rowsum.inputs[0].src_node.name}, not the softmax exp"
    )
    assert value_mma.inputs[0].src_node is exp, (
        f"the value matmul consumes {value_mma.inputs[0].src_node.name}, not the exp"
    )
    assert not value_mma.transpose_B, "the value matmul must not transpose V"
    # The tiling this returns carries one dtype pair, read off the score matmul. A body
    # whose two matmuls disagree is a mixed-precision body, and the layer would silently
    # rebuild it in the score matmul's precision throughout.
    assert (str(value_mma.dtype), str(value_mma.out_dtype)) == (
        str(score_mma.dtype),
        str(score_mma.out_dtype),
    ), (
        f"the value matmul runs {value_mma.dtype}/{value_mma.out_dtype} but the score "
        f"matmul runs {score_mma.dtype}/{score_mma.out_dtype}; the access layer builds one "
        "precision for both and cannot express the difference"
    )
    assert getattr(div, "broadcast", None) == 0, (
        f"the divide must broadcast the denominator over rows, got {div.broadcast}"
    )
    # Ordered, not a set: a body that divides the row sum BY the value matmul carries the
    # same two accumulators, and the layer -- which always builds the numerator from the
    # value matmul -- would rebuild it the right way round and silently compute something
    # else. `apply_tiling` happens to reject the swapped body earlier today, so this is a
    # guard against that changing rather than a path any test can reach.
    accums = [getattr(edge, "accum_node", None) for edge in div.inputs]
    assert accums[:2] == [value_mma, rowsum], (
        "the divide must take the value matmul over the row sum, in that order; got "
        f"{[None if n is None else n.name for n in accums]}"
    )
    _assert_output_is_the_divide(graph, div, out)
    placeholders = [n for n in inner.nodes if isinstance(n, PlaceHolderNode)]
    assert len(placeholders) == 3, (
        f"expected exactly Q, K and V in the loop tier, got {[n.name for n in placeholders]}"
    )
    for operand, name in ((score_mma.inputs[0], "Q"), (score_mma.inputs[1], "K")):
        assert isinstance(operand.src_node, PlaceHolderNode), (
            f"the score matmul's {name} operand is {operand.src_node.name}, not an input"
        )
    assert isinstance(value_mma.inputs[1].src_node, PlaceHolderNode), (
        "the value matmul's V operand is not an input"
    )

    # --- the tiling, and the ways it can be unwrappable ----------------------------
    score_tile = graph_tile(inner, score_mma)
    value_tile = graph_tile(inner, value_mma)
    assert len(score_tile) == 2 and len(value_tile) == 2, (
        f"expected 2-D score/value tiles, got {score_tile}/{value_tile}"
    )
    assert score_tile[0] == value_tile[0], (
        f"score and value matmuls disagree on the query tile: {score_tile} vs {value_tile}"
    )
    # The head dim comes off the node, not off the tile: they differ exactly when the
    # policy split the head dim, and then the tile is a fraction of the operator's D. A
    # kernel built from the fraction is a different operator, so reject rather than
    # silently narrow the cache and the output.
    head_dim = int(value_mma.dims[-2])
    assert int(value_tile[1]) == head_dim, (
        f"the head dim is split into {value_tile[1]}-wide blocks (of {head_dim}); the "
        "access layer replaces the grid, so those blocks would be lost. Re-tile with the "
        "whole head dim in one block."
    )
    assert int(graph_tile(inner, score_mma.inputs[0].src_node)[1]) == head_dim, (
        "the score matmul reduces over part of the head dim; it must reduce over all of it"
    )
    # The grid this pass writes is (batch, kv head); a dense schedule that still has
    # blocks of its own would lose them.
    assert graph.extent in (1, (1,)), (
        f"the dense schedule has a grid of {graph.extent}, which paging would discard; "
        "it must tile to a single block"
    )
    # The scale the dense body folded in has to be the operator's, or the runtime
    # sm_scale this layer substitutes means something different.
    folded = _folded_scale(scale_chain)
    expected = head_dim**-0.5
    assert abs(folded - expected) <= 1e-6 * max(1.0, abs(expected)), (
        f"the dense body scales the scores by {folded}, not 1/sqrt({head_dim}) = "
        f"{expected}; the runtime scale this layer substitutes would not match"
    )
    return DenseAttention(
        query_rows=int(score_tile[0]),
        kv_tile=int(score_tile[1]),
        head_dim=head_dim,
        n_threads=int(getattr(graph, "n_threads", 128)),
        dtype=str(score_mma.dtype),
        accum_dtype=str(score_mma.out_dtype),
        out_dtype=str(out.get_dtype(0)),
    )


def _split_of(graph: Graph) -> int:
    """The split factor of a scheduled split-K partial: its grid is the split axis alone.

    The dense stages tile a partial body so that one block owns one split's whole
    (query rows, head dim) tile -- anything else would be a block the paging pass discards
    -- so the only grid axis left is the split.
    """
    extent = graph.extent if isinstance(graph.extent, tuple) else (graph.extent,)
    assert len(extent) == 1 and int(extent[0]) > 1, (
        f"a split-K partial's grid must be the split axis alone, got {graph.extent}; "
        "re-tile with the whole query tile and head dim in one block"
    )
    return int(extent[0])


def _assert_split_slices(graph: Graph, placeholders, split: int, what: str) -> None:
    """Each KV operand's tile must be exactly a 1/split slice of its dense extent."""
    for ph in placeholders:
        tmap = graph.tmap[ph]
        whole, tile = int(tmap.shape[0]), int(tmap.tile_size[0])
        assert getattr(ph, "_split_axis", None) == 0 and whole == split * tile, (
            f"{what} {ph.name} is tiled {tile} of {whole} rows on split axis "
            f"{getattr(ph, '_split_axis', None)}; a split-K partial slices its KV axis "
            f"into {split} equal contiguous ranges, one per block"
        )


def _partial_outputs(graph: Graph, rowsum, value_mma) -> str:
    """Check the partial stores its two accumulators raw, and return their output dtype.

    Both must be stored unchanged: a partial that normalises by its own denominator, or
    scales anything, cannot be summed by the combine. The access layer stores the
    accumulators in their accumulation precision regardless of the dense output dtype,
    so what this returns is only the operator's *final* output dtype.
    """
    outs = [n for n in graph.nodes if isinstance(n, OutputNode)]
    assert len(outs) == 2, (
        f"a split-K partial stores exactly its two accumulators, found {len(outs)} outputs"
    )
    stored = []
    for out in outs:
        edge = out.inputs[0]
        accum = getattr(edge, "accum_node", None)
        assert accum is not None and edge.src_node.__class__.__name__ == "GraphNode", (
            f"output {out.name} is produced by {edge.src_node.name}, not stored straight "
            "from the reduction loop; a partial that post-processes its accumulators "
            "cannot be summed by the combine"
        )
        stored.append(accum)
    assert {id(n) for n in stored} == {id(rowsum), id(value_mma)}, (
        "the two outputs must be the row sum and the value matmul; got "
        f"{[n.name for n in stored]}"
    )
    extra = [
        n for n in graph.nodes if not isinstance(n, (GraphNode, PlaceHolderNode, OutputNode))
    ]
    assert not extra, (
        f"the outer tier also holds {[n.name for n in extra]}; a split-K partial's outer "
        "tier is the reduction loop and its two stores, nothing else"
    )
    dtypes = {str(o.get_dtype(0)) for o in outs}
    assert len(dtypes) == 1, f"the two partial outputs disagree on dtype: {dtypes}"
    return dtypes.pop()


def read_dense_partial(graph: Graph) -> tuple:
    """Recognise a scheduled split-K *partial* of dense attention; ``(tiling, split)``.

    The body is ``read_dense_attention``'s minus the divide: Q @ K^T -> constant scaling
    -> exp -> {row sum, @ V}, with both accumulators stored raw and the KV operands each
    tiled to a contiguous 1/split slice, one split per block. The same strictness applies
    for the same reason -- the access layer rebuilds the body and keeps only the tiling.
    """
    inner_node = _sole(
        [n for n in graph.nodes if isinstance(n, GraphNode)], "reduction subgraph"
    )
    inner = inner_node.graph
    assert len(inner.accum_nodes) == 2, (
        "expected two accumulators (the row sum and the value matmul), got "
        f"{[n.name for n in inner.accum_nodes]}"
    )
    rowsum = _sole(
        [n for n in inner.accum_nodes if isinstance(n, Reduction)], "row-sum accumulator"
    )
    value_mma = _sole(
        [n for n in inner.accum_nodes if isinstance(n, Matmul)], "value matmul"
    )
    score_mma = _sole(
        [
            n
            for n in inner.nodes
            if isinstance(n, Matmul) and n is not value_mma and n.transpose_B
        ],
        "score matmul (transpose_B)",
    )
    exp = _sole([n for n in inner.nodes if isinstance(n, Exp)], "softmax exp")
    assert not [n for n in graph.nodes if isinstance(n, Div)], (
        "a split-K partial must not divide: the softmax denominator is only known once "
        "every split has contributed, so the divide belongs to the combine"
    )
    scale_chain = _chain_to(exp, score_mma)
    assert scale_chain is not None, (
        "the softmax exp does not read the score matmul through constant scaling alone; "
        "this body is not the attention the access layer would rebuild"
    )
    assert rowsum.inputs[0].src_node is exp, (
        f"the row sum reduces {rowsum.inputs[0].src_node.name}, not the softmax exp"
    )
    assert value_mma.inputs[0].src_node is exp, (
        f"the value matmul consumes {value_mma.inputs[0].src_node.name}, not the exp"
    )
    assert not value_mma.transpose_B, "the value matmul must not transpose V"
    assert (str(value_mma.dtype), str(value_mma.out_dtype)) == (
        str(score_mma.dtype),
        str(score_mma.out_dtype),
    ), (
        f"the value matmul runs {value_mma.dtype}/{value_mma.out_dtype} but the score "
        f"matmul runs {score_mma.dtype}/{score_mma.out_dtype}; the access layer builds one "
        "precision for both and cannot express the difference"
    )
    out_dtype = _partial_outputs(graph, rowsum, value_mma)
    placeholders = [n for n in inner.nodes if isinstance(n, PlaceHolderNode)]
    assert len(placeholders) == 3, (
        f"expected exactly Q, K and V in the loop tier, got {[n.name for n in placeholders]}"
    )
    for operand, name in ((score_mma.inputs[0], "Q"), (score_mma.inputs[1], "K")):
        assert isinstance(operand.src_node, PlaceHolderNode), (
            f"the score matmul's {name} operand is {operand.src_node.name}, not an input"
        )
    assert isinstance(value_mma.inputs[1].src_node, PlaceHolderNode), (
        "the value matmul's V operand is not an input"
    )

    score_tile = graph_tile(inner, score_mma)
    value_tile = graph_tile(inner, value_mma)
    assert len(score_tile) == 2 and len(value_tile) == 2, (
        f"expected 2-D score/value tiles, got {score_tile}/{value_tile}"
    )
    assert score_tile[0] == value_tile[0], (
        f"score and value matmuls disagree on the query tile: {score_tile} vs {value_tile}"
    )
    head_dim = int(value_mma.dims[-2])
    assert int(value_tile[1]) == head_dim, (
        f"the head dim is split into {value_tile[1]}-wide blocks (of {head_dim}); the "
        "access layer replaces the grid, so those blocks would be lost. Re-tile with the "
        "whole head dim in one block."
    )
    assert int(graph_tile(inner, score_mma.inputs[0].src_node)[-1]) == head_dim, (
        "the score matmul reduces over part of the head dim; it must reduce over all of it"
    )
    split = _split_of(graph)
    outer_k = inner.binding[score_mma.inputs[1].src_node.name]
    outer_v = inner.binding[value_mma.inputs[1].src_node.name]
    _assert_split_slices(graph, [outer_k, outer_v], split, "KV operand")
    folded = _folded_scale(scale_chain)
    expected = head_dim**-0.5
    assert abs(folded - expected) <= 1e-6 * max(1.0, abs(expected)), (
        f"the dense body scales the scores by {folded}, not 1/sqrt({head_dim}) = "
        f"{expected}; the runtime scale this layer substitutes would not match"
    )
    dense = DenseAttention(
        query_rows=int(score_tile[0]),
        kv_tile=int(score_tile[1]),
        head_dim=head_dim,
        n_threads=int(getattr(graph, "n_threads", 128)),
        dtype=str(score_mma.dtype),
        accum_dtype=str(score_mma.out_dtype),
        out_dtype=out_dtype,
    )
    return dense, split


def read_dense_mla_partial(graph: Graph) -> tuple:
    """Recognise a scheduled split-K partial of dense MLA attention; ``(tiling, split)``.

    ``read_dense_mla``'s body minus the divide, with the two accumulators stored raw and
    both caches sliced 1/split per block.
    """
    inner_node = _sole(
        [n for n in graph.nodes if isinstance(n, GraphNode)], "reduction subgraph"
    )
    inner = inner_node.graph
    assert len(inner.accum_nodes) == 2, (
        "expected two accumulators (the row sum and the value matmul), got "
        f"{[n.name for n in inner.accum_nodes]}"
    )
    rowsum = _sole(
        [n for n in inner.accum_nodes if isinstance(n, Reduction)], "row-sum accumulator"
    )
    value_mma = _sole(
        [n for n in inner.accum_nodes if isinstance(n, Matmul)], "value matmul"
    )
    score_mmas = [
        n
        for n in inner.nodes
        if isinstance(n, Matmul) and n is not value_mma and n.transpose_B
    ]
    assert len(score_mmas) == 2, (
        "expected two score matmuls (compressed kv and positional), got "
        f"{[n.name for n in score_mmas]}"
    )
    exp = _sole([n for n in inner.nodes if isinstance(n, Exp)], "softmax exp")
    assert not [n for n in graph.nodes if isinstance(n, Div)], (
        "a split-K partial must not divide: the softmax denominator is only known once "
        "every split has contributed, so the divide belongs to the combine"
    )
    add = _sole(
        [n for n in inner.nodes if isinstance(n, Add) and len(n.inputs) == 2],
        "score sum (the two matmuls added)",
    )
    added = {id(edge.src_node) for edge in add.inputs}
    assert added == {id(score_mmas[0]), id(score_mmas[1])}, (
        "the score sum does not add the two score matmuls; this body is not the MLA "
        "attention the access layer would rebuild"
    )
    scale_chain = _chain_to(exp, add)
    assert scale_chain is not None, (
        "the softmax exp does not read the summed scores through constant scaling alone"
    )
    assert rowsum.inputs[0].src_node is exp, (
        f"the row sum reduces {rowsum.inputs[0].src_node.name}, not the softmax exp"
    )
    assert value_mma.inputs[0].src_node is exp, (
        f"the value matmul consumes {value_mma.inputs[0].src_node.name}, not the exp"
    )
    assert not value_mma.transpose_B, "the value matmul must not transpose the cache"
    for other in (score_mmas[0], score_mmas[1], value_mma):
        assert (str(other.dtype), str(other.out_dtype)) == (
            str(score_mmas[0].dtype),
            str(score_mmas[0].out_dtype),
        ), (
            f"{other.name} runs {other.dtype}/{other.out_dtype} but {score_mmas[0].name} "
            f"runs {score_mmas[0].dtype}/{score_mmas[0].out_dtype}; the access layer builds "
            "one precision for all three matmuls"
        )
    out_dtype = _partial_outputs(graph, rowsum, value_mma)
    placeholders = [n for n in inner.nodes if isinstance(n, PlaceHolderNode)]
    assert len(placeholders) == 4, (
        "expected exactly the two query tensors and the two caches in the loop tier, got "
        f"{[n.name for n in placeholders]}"
    )
    operands = [(mma, slot) for mma in score_mmas for slot in (0, 1)]
    operands.append((value_mma, 1))
    for mma, slot in operands:
        assert isinstance(mma.inputs[slot].src_node, PlaceHolderNode), (
            f"{mma.name}'s operand {slot} is "
            f"{mma.inputs[slot].src_node.name}, not an input"
        )
    cache = value_mma.inputs[1].src_node
    ckv_mma = _sole(
        [n for n in score_mmas if n.inputs[1].src_node is cache],
        "score matmul reading the value cache (MLA aliases ckv as V)",
    )
    pe_mma = score_mmas[0] if score_mmas[1] is ckv_mma else score_mmas[1]

    score_tile = graph_tile(inner, ckv_mma)
    pe_tile = graph_tile(inner, pe_mma)
    value_tile = graph_tile(inner, value_mma)
    assert len(score_tile) == 2 and len(value_tile) == 2, (
        f"expected 2-D score/value tiles, got {score_tile}/{value_tile}"
    )
    assert score_tile == pe_tile, (
        f"the two score matmuls disagree on the tile: {score_tile} vs {pe_tile}"
    )
    assert score_tile[0] == value_tile[0], (
        f"score and value matmuls disagree on the query tile: {score_tile} vs {value_tile}"
    )
    head_dim_ckv = int(value_mma.dims[-2])
    head_dim_kpe = int(pe_mma.dims[-1])
    assert int(value_tile[1]) == head_dim_ckv, (
        f"the compressed head dim is split into {value_tile[1]}-wide blocks (of "
        f"{head_dim_ckv}); the access layer replaces the grid, so those blocks would be "
        "lost. Re-tile with the whole head dim in one block."
    )
    assert int(graph_tile(inner, ckv_mma.inputs[0].src_node)[-1]) == head_dim_ckv, (
        "the ckv score matmul reduces over part of the compressed head dim; it must "
        "reduce over all of it"
    )
    assert int(graph_tile(inner, pe_mma.inputs[0].src_node)[-1]) == head_dim_kpe, (
        "the positional score matmul reduces over part of the RoPE head dim; it must "
        "reduce over all of it"
    )
    split = _split_of(graph)
    outer_ckv = inner.binding[cache.name]
    outer_kpe = inner.binding[pe_mma.inputs[1].src_node.name]
    _assert_split_slices(graph, [outer_ckv, outer_kpe], split, "cache operand")
    folded = _folded_scale(scale_chain)
    expected = head_dim_ckv**-0.5
    assert abs(folded - expected) <= 1e-6 * max(1.0, abs(expected)), (
        f"the dense body scales the scores by {folded}, not 1/sqrt({head_dim_ckv}) = "
        f"{expected}; the runtime scale this layer substitutes would not match"
    )
    dense = DenseMLA(
        query_rows=int(score_tile[0]),
        kv_tile=int(score_tile[1]),
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        n_threads=int(getattr(graph, "n_threads", 128)),
        dtype=str(ckv_mma.dtype),
        accum_dtype=str(ckv_mma.out_dtype),
        out_dtype=out_dtype,
    )
    return dense, split


def read_dense_combine(graph: Graph) -> CombineTiling:
    """Recognise a scheduled split-K *combine* and extract its tiling.

    The body: two axis-0 reductions over ``(split, rows)`` and ``(split, rows, cols)``
    partials, then a divide broadcasting the first over the rows of the second, stored
    straight to the output. The tiling read off it is how many partials one loop
    iteration folds (the value sum's split step), how many output columns one block owns,
    and the block size -- the three things the dynamic combine keeps. The rows a block
    owns are not read: in the dynamic kernel they are the head group, fixed by the operator.

    The scheduler places each reduction independently: inside its own loop tier stepping
    through the splits, or -- when its step is the whole split -- in the outer tier with no
    loop. Any mix is legal; the value sum's placement is the one that matters for cost, so
    it is the one whose step and staging are kept.
    """
    div = _sole([n for n in graph.nodes if isinstance(n, Div)], "softmax divide")
    out = _sole([n for n in graph.nodes if isinstance(n, OutputNode)], "output")
    assert getattr(div, "broadcast", None) == 0, (
        f"the divide must broadcast the denominator over rows, got {div.broadcast}"
    )
    assert out.inputs[0].src_node is div, (
        f"the output is produced by {out.inputs[0].src_node.name}, not the softmax divide; "
        "the access layer stores the divide's tile, so whatever the dense body computes "
        "after it would be dropped from the paged kernel"
    )
    # Where each reduction lives: (node, tier graph, or None for the outer tier).
    placed = []
    for node in graph.nodes:
        if isinstance(node, Reduction):
            placed.append((node, None))
        elif isinstance(node, GraphNode):
            for inner_node in node.graph.nodes:
                if isinstance(inner_node, Reduction):
                    assert inner_node in node.graph.accum_nodes, (
                        f"{inner_node.name} sits in a loop tier without accumulating "
                        "across it"
                    )
                    placed.append((inner_node, node.graph))
        elif not isinstance(node, (PlaceHolderNode, OutputNode, Div)):
            raise AssertionError(
                f"the combine also holds {node.name}; the access layer rebuilds only the "
                "two split sums and the divide, and would drop it"
            )
    assert len(placed) == 2 and all(r.axis == 0 for r, _ in placed), (
        "expected two reductions over the leading (split) axis, got "
        f"{[(n.name, getattr(n, 'axis', None)) for n, _ in placed]}"
    )
    by_rank = {len(r.dims): (r, tier) for r, tier in placed}
    assert set(by_rank) == {2, 3}, (
        "expected one (split, rows) and one (split, rows, cols) reduction, got ranks "
        f"{sorted(len(r.dims) for r, _ in placed)}"
    )
    value_sum, value_tier = by_rank[3]
    rowsum_sum, _ = by_rank[2]
    summed = [getattr(e, "accum_node", None) or e.src_node for e in div.inputs]
    assert summed[:2] == [value_sum, rowsum_sum], (
        "the divide must take the summed value partials over the summed row sums, in that "
        f"order; got {[n.name for n in summed]}"
    )
    ph_value = value_sum.inputs[0].src_node
    ph_rowsum = rowsum_sum.inputs[0].src_node
    assert isinstance(ph_value, PlaceHolderNode) and isinstance(ph_rowsum, PlaceHolderNode), (
        "both reductions must read a partial tensor directly"
    )

    def outer(ph, tier):
        return ph if tier is None else tier.binding[ph.name]

    # The split count is the partials' leading extent; the step is the value sum's tile.
    split = int(graph.tmap[outer(ph_value, value_tier)].shape[0])
    rowsum_tier = by_rank[2][1]
    assert int(graph.tmap[outer(ph_rowsum, rowsum_tier)].shape[0]) == split, (
        "the two partials disagree on the split count"
    )
    if value_tier is None:
        step, scope = split, "g"
    else:
        # The policy may hand the split axis a tile wider than its extent; that is the
        # whole extent in one step, not a step the loop could take.
        step = min(int(graph_tile(value_tier, ph_value)[0]), split)
        scope = value_tier.tinfo_map.get(ph_value, (None, "s"))[-1]
    div_tile = graph_tile(graph, div)
    assert int(div_tile[-1]) == int(graph_tile(graph, out)[-1]), (
        "the divide and the output disagree on the column tile"
    )
    return CombineTiling(
        split=split,
        split_step=step,
        cols=int(div_tile[-1]),
        n_threads=int(getattr(graph, "n_threads", 128)),
        input_scope=scope if scope in ("s", "g") else "s",
    )


def apply_paging_split(graphs, spec) -> list:
    """Wrap a scheduled split-K pair in a paged decode access layer: ``[partial, combine]``.

    ``graphs`` is ``[partial, combine]`` as the dense stages tile a split-K decode body --
    the partial's KV operands sliced one split per block, the combine summing the partials
    and dividing. The result is what ``apply_paging`` is for a single graph: the same
    tiling decisions, wrapped in the runtime page table, per-request lengths and symbolic
    grid, emitted by ``TilelangCodeGen().gen`` as one prim_func of two kernels.
    """
    assert len(graphs) == 2, f"expected a [partial, combine] pair, got {len(graphs)} graphs"
    partial, combine = graphs
    spec.validate()
    if isinstance(spec, MLASpec):
        dense, split = read_dense_mla_partial(partial)
    else:
        dense, split = read_dense_partial(partial)
    comb = read_dense_combine(combine)
    assert comb.split == split, (
        f"the partial makes {split} splits but the combine sums {comb.split}"
    )
    return build_split_decode(spec, dense, split, comb)


def graph_tile(graph: Graph, node) -> tuple:
    iterator = graph.tmap[node]
    return tuple(iterator.tile_size)


def _connect(src, dst, src_id, dst_id) -> None:
    edge = Edge(src, dst, src_id, dst_id)
    src._out_edges.append(edge)
    dst.set_inputs(dst_id, edge)


def _int32():
    import tvm

    return tvm.DataType("int32")


def apply_paging(graph: Graph, spec: PagingSpec) -> Graph:
    """Return a paged attention graph built around ``graph``'s tiling.

    ``graph`` must be a scheduled dense attention graph -- the output of
    ``apply_tiling`` on a Q@K^T/softmax/P@V source graph. An ``MLASpec`` selects the latent
    variant instead, whose dense body sums a compressed and a positional score matmul and
    reuses the compressed cache as V.
    """
    spec.validate()
    if isinstance(spec, MLASpec):
        mla = read_dense_mla(graph)
        # Both MLA layers put one token's whole block of query heads in a single tile: the
        # row gather stages `num_qo_heads` rows and the epilogue writes that many, neither
        # of them predicated. A tile of any other height is not a slower kernel, it is an
        # out-of-bounds one -- measured, a 32-row tile over 16 heads makes 37 invalid
        # global reads per launch under compute-sanitizer, and 16 rows under 32 heads emits
        # an epilogue reading rows 16..31 of a 16-row fragment. One copy of the check here
        # rather than one per builder: the prefill layer had it and the decode layer did
        # not, which is exactly how that hole opened.
        assert mla.query_rows == spec.num_qo_heads, (
            f"the query tile is {mla.query_rows} rows but the operator has "
            f"{spec.num_qo_heads} heads; an MLA tile is one token's whole block of heads, "
            "which it can only be when they are equal"
        )
        if spec.kind == MLA_PAGED_PREFILL_CAUSAL:
            return _build_mla_paged_prefill_causal(spec, mla)
        return _build_mla_paged_decode(spec, mla)
    dense = read_dense_attention(graph)
    if spec.kind == RAGGED_PREFILL_CAUSAL:
        # No `query_rows >= group` gate here: a ragged tile's rows are tokens of one query
        # head, not (token, head-slot) pairs, so the group never bounds them -- it reaches
        # only `kv_head = h // group`.
        return _build_ragged_prefill_causal(spec, dense)
    # The epilogue writes `group` rows out of a `query_rows`-row accumulator, and the q
    # gather only stages `query_rows` of them. A group that does not fit reads past both.
    assert dense.query_rows >= spec.group, (
        f"a block's query tile holds {dense.query_rows} rows but the head group needs "
        f"{spec.group}; re-tile with at least {spec.group} query rows"
    )
    if spec.kind == PAGED_PREFILL_CAUSAL:
        # A prefill block's query tile fuses (token, head slot), so its rows have to
        # divide into whole head groups.
        assert dense.query_rows % spec.group == 0, (
            f"a prefill query tile of {dense.query_rows} rows does not divide into "
            f"head groups of {spec.group}; the M axis fuses token and head slot"
        )
        return _build_paged_prefill_causal(spec, dense)
    if spec.kind == PAGED_DECODE_SLAB:
        # One KV tile has to lie inside one page, or the slab would straddle two
        # page-table entries and a single scalar page id could not address it.
        assert spec.page_size % dense.kv_tile == 0, (
            f"a page of {spec.page_size} tokens does not hold a whole number of "
            f"{dense.kv_tile}-token KV tiles, so a slab would straddle two pages"
        )
        return _build_paged_decode_slab(spec, dense)
    return _build_paged_decode(spec, dense)


def build_split_decode(
    spec, dense, split: int, comb: CombineTiling
) -> list:
    """The split-KV pair ``[partial, combine]`` for a decode layer, from an explicit tiling.

    ``dense`` is the partial body's tiling (a ``DenseAttention``), ``split`` the number of
    KV splits and ``comb`` the combine kernel's tiling. Hand this list to
    ``TilelangCodeGen().gen`` and it emits one prim_func with both kernels, sharing the
    partial tensors. ``apply_paging_split`` is the entry point that reads all three off
    searched graphs; this one exists so the layer can be built and tested from numbers.
    """
    spec.validate()
    assert comb.split == split, (
        f"the combine tiling is for {comb.split} splits but the partial makes {split}"
    )
    if isinstance(spec, MLASpec):
        assert spec.kind == MLA_PAGED_DECODE, (
            f"split-KV is a decode transformation; got the {spec.kind!r} layer"
        )
        assert dense.query_rows == spec.num_qo_heads, (
            f"the query tile is {dense.query_rows} rows but the operator has "
            f"{spec.num_qo_heads} heads; an MLA tile is one token's whole block of heads"
        )
        partial = _build_mla_paged_decode(spec, dense, split=split)
        combine = _build_decode_combine(
            spec,
            heads=spec.num_qo_heads,
            group=spec.num_qo_heads,
            head_dim=dense.head_dim_ckv,
            out_dtype=dense.out_dtype,
            accum_dtype=dense.accum_dtype,
            comb=comb,
            grid_order=("batch", "head"),
            batch_dim=spec.batch_dim,
            rasterization=spec.rasterization,
        )
        return [partial, combine]
    assert spec.kind in (PAGED_DECODE, PAGED_DECODE_SLAB), (
        f"split-KV is a decode transformation; got the {spec.kind!r} layer"
    )
    assert dense.query_rows >= spec.group, (
        f"a block's query tile holds {dense.query_rows} rows but the head group needs "
        f"{spec.group}; re-tile with at least {spec.group} query rows"
    )
    if spec.kind == PAGED_DECODE_SLAB:
        assert spec.page_size % dense.kv_tile == 0, (
            f"a page of {spec.page_size} tokens does not hold a whole number of "
            f"{dense.kv_tile}-token KV tiles, so a slab would straddle two pages"
        )
        partial = _build_paged_decode_slab(spec, dense, split=split)
    else:
        partial = _build_paged_decode(spec, dense, split=split)
    combine = _build_decode_combine(
        spec,
        heads=spec.num_qo_heads,
        group=spec.group,
        head_dim=dense.head_dim,
        out_dtype=dense.out_dtype,
        accum_dtype=dense.accum_dtype,
        comb=comb,
        grid_order=spec.grid_order,
        batch_dim=spec.batch_dim,
        rasterization=spec.rasterization,
    )
    return [partial, combine]


def _build_paged_decode(
    spec: PagingSpec, dense: DenseAttention, split: int | None = None
) -> Graph:
    """The paged decode layer, or with ``split`` the *partial* half of its split-KV form.

    Split-KV is the same kernel with three differences, each a consequence of one block
    owning a contiguous range of a request's KV tiles rather than all of them:

    * the grid gains a trailing split axis, and per block the loop runs ``my_trip`` tiles
      starting at ``tile0`` -- ``ceil(n_tiles / split)`` tiles per split, the last split
      taking the remainder, so every tile is owned by exactly one block and a split past
      the end runs zero iterations;
    * the look-ahead read stops at the split's own end (``split_end``) instead of the
      request's, so no block prefetches a tile another block owns;
    * the epilogue stores the two accumulators as they are -- the value matmul over
      ``group`` rows and the exp row sums -- to fp32 partial tensors indexed by split, in
      place of the divide and the log-sum-exp, which ``_build_decode_combine`` applies
      after summing over the splits. That is sound because this softmax subtracts no
      running max: the partial sums add.
    """
    mq, bn, head_dim = dense.query_rows, dense.kv_tile, dense.head_dim
    group, n_kv = spec.group, spec.num_kv_heads
    batch, pages = spec.batch_dim, spec.pages_dim
    indptr_dim, index_dim = spec.indptr_dim, spec.index_dim
    if split is not None:
        assert isinstance(split, int) and split > 1, f"split={split!r} is not a split"

    # --- the loop tier: gather a KV tile, score it, accumulate ----------------------
    #
    # The gathers read the page table in place off a token-unit base rather than through
    # a tile resolved into shared memory: the thread that gathers row g0 is not the one
    # that would have resolved entry g0, so a resolved tile costs a barrier before it can
    # be read and another before the next iteration may overwrite it.
    gather_kwargs = dict(
        source_rows=pages,
        index_extent=index_dim,
        page_size=spec.page_size,
        paged_axes=True,
        source_heads=n_kv,
        head_scalar=True,
        clamp_index=BUFFER_BOUND,
        clamp_row=BUFFER_BOUND,
        scalar_args=("kv_pos0", "head"),
        dtype=dense.dtype,
        out_dtype=dense.dtype,
    )
    gather_k = PagedGather((bn, head_dim), name="paged_gather_k", **gather_kwargs)
    gather_v = PagedGather((bn, head_dim), name="paged_gather_v", **gather_kwargs)
    score_mma = Matmul(
        (mq, bn, head_dim),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_0",
        transpose_B=True,
        warp_policy=spec.warp_policy,
    )
    # One MaskedExp replaces the dense Mul(1/sqrt(D)) + Sub(max estimate) + Exp chain:
    # the scale becomes a runtime scalar in the log2 domain, and the subtraction goes
    # away entirely because none of the tuned kernels subtract a max.
    exp = MaskedExp(
        (mq, bn),
        scale=None,
        scalar_args=("tile_base", "valid_len", "scale2"),
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="exp_0",
        n_batch_dims=0,
    )
    rowsum = Reduction(
        (mq, bn),
        axis=-1,
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="rowsum_0",
    )
    value_mma = Matmul(
        (mq, head_dim, bn),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_1",
        warp_policy=spec.warp_policy,
    )
    value_mma.stage_fragment_operands_in_shared = (0,)

    _connect(gather_k, score_mma, 0, 1)
    _connect(score_mma, exp, 0, 0)
    _connect(exp, rowsum, 0, 0)
    _connect(exp, value_mma, 0, 0)
    _connect(gather_v, value_mma, 0, 1)

    ph_k = gather_k.inputs[0].src_node
    ph_v = gather_v.inputs[0].src_node
    ph_q = score_mma.inputs[0].src_node
    # One page table feeds both gathers, so K's index placeholder is rewired to V rather
    # than each gather declaring its own copy of the same kernel input.
    ph_idx = gather_k.inputs[1].src_node
    _connect(ph_idx, gather_v, 0, 1)

    ph_q.name = f"PlaceHolder_{spec.query_input}"
    ph_k.name = f"PlaceHolder_{spec.cache_inputs[0]}"
    ph_v.name = f"PlaceHolder_{spec.cache_inputs[1]}"
    ph_idx.name = f"PlaceHolder_{spec.index_input}"

    from .util import gather_all_nodes, lock_graph

    inner = Graph(nodes=gather_all_nodes(rowsum, value_mma))
    # A serial loop with a runtime trip count never reads the static extent; the real
    # bound is loop_trip, which is per request.
    inner.extent = 1
    inner.loop_kind = "serial"
    # Both KV tiles are read one iteration ahead, so the gathers' global latency is
    # covered by the tile's own matmuls and softmax instead of standing in front of them.
    # This is the loop's only source of load/compute overlap: the reads are gathers,
    # which tilelang's pipeliner cannot double-buffer.
    inner.prefetch_nodes = [gather_k, gather_v]
    inner.sargs = [
        ("q_row0", "int32"),
        ("q_head0", "int32"),
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("head", "int32"),
        ("scale2", "float32"),
    ]
    inner.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("scale2", "float32"),
        ("kv_pos0", "int32"),
        ("head", "int32"),
    ]
    inner.smap = {
        "q_row0": lambda i: "q_row0",
        "q_head0": lambda i: "q_head0",
        "kv_base": lambda i: "kv_base",
        "head": lambda i: "head",
        "valid_len": lambda i: "valid_len",
        "scale2": lambda i: "scale2",
    }
    if split is None:
        inner.loop_trip = f"T.ceildiv(valid_len, {bn})"
        inner.prefetch_guard = f"tile_base + {bn} < valid_len"
        inner.smap["tile_base"] = lambda i: f"{bn} * {i}"
        # The tile's first kv position, in tokens: where the gathers start reading the
        # page table.
        inner.smap["kv_pos0"] = lambda i: f"kv_base + {bn} * {i}"
    else:
        # This block's tiles are `tile0 .. tile0 + my_trip` of the request's, and the
        # look-ahead must not run past the split's own end.
        inner.loop_trip = "my_trip"
        inner.prefetch_guard = f"tile_base + {bn} < split_end"
        inner.sargs += [("tile0", "int32"), ("my_trip", "int32"), ("split_end", "int32")]
        inner.iter_sargs += [("split_end", "int32")]
        inner.smap["tile0"] = lambda i: "tile0"
        inner.smap["my_trip"] = lambda i: "my_trip"
        inner.smap["split_end"] = lambda i: "split_end"
        inner.smap["tile_base"] = lambda i: f"{bn} * (tile0 + {i})"
        inner.smap["kv_pos0"] = lambda i: f"kv_base + {bn} * (tile0 + {i})"
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    rowsum_id = inner.accum_nodes.index(rowsum)
    value_id = inner.accum_nodes.index(value_mma)
    inner_q = inner.binding[ph_q.name]
    inner_idx = inner.binding[ph_idx.name]

    # Q is staged once per block by a gather that zero-fills the rows past the head
    # group, so a padded query tile costs no correctness.
    q_tile = RowGather(
        (mq, head_dim),
        source_rows=batch,
        n_valid=group,
        src_dims=3,
        source_heads=spec.num_qo_heads,
        scalar_args=("q_row0", "q_head0"),
        dtype=dense.dtype,
        out_dtype=dense.dtype,
        name="q_tile",
    )
    q_tile.source_args = (ph_q.name,)
    inner_q.producer_op = q_tile

    # --- the epilogue: normalise, and emit the log-sum-exp -------------------------
    #
    # Or, for a split-KV partial, store the two accumulators as they are: the combine
    # kernel normalises once it has every split's contribution.
    div, log2, store_d, store_o, outputs = _decode_epilogue(
        spec, dense, loop_node, rowsum, rowsum_id, value_mma, value_id, split
    )

    # Read for its values, not gathered from, so it stays out of the *_iter call.
    ph_indptr = PlaceHolderNode(name=spec.indptr_input)
    ph_indptr.set_shape((1,))
    ph_indptr.set_dtype(_int32())
    ph_indptr.index_only = True
    ph_indptr.name = f"PlaceHolder_{spec.indptr_input}"

    # Inputs lead the node list in the operator's declared order, both so the emitted
    # signature matches it and because a symbolic output extent is resolved at call time
    # from an already-bound input.
    main = Graph(
        nodes=[ph_q, ph_k, ph_v, ph_indptr, ph_idx] + gather_all_nodes(loop_node)
    )
    main.dyn_dims = [batch, pages, indptr_dim, index_dim]
    main.scalar_params = [(spec.scale_input, "float32")]
    main.n_threads = dense.n_threads
    main.pass_configs = dict(spec.pass_configs)
    # Unconditional in every builder: `Graph.rasterization` already defaults to None,
    # meaning "defer to the codegen default", so a None spec value and no assignment at all
    # are the same thing. Two spellings only invited someone to "fix" the wrong three.
    main.rasterization = spec.rasterization
    # One CTA per (request, kv head): a decode request is a single query token, so its
    # whole head group fits one tile and there is no work list to scan. A split-KV
    # partial adds the split as the trailing (slowest) grid axis, so the blocks sharing a
    # request's page table are launched together.
    batch_first = tuple(spec.grid_order) == ("batch", "head")
    axes = (f"{batch}", f"{n_kv}") if batch_first else (f"{n_kv}", f"{batch}")
    if split is not None:
        axes += (f"{split}",)
    main.extent = axes
    main.sargs = list(inner.sargs)

    def _grid(*iters):
        """Name the grid indices, so a permuted order costs nothing downstream."""
        lead = iters[:2] if batch_first else tuple(reversed(iters[:2]))
        return (*lead, *iters[2:])

    def _scalar(build):
        return lambda *iters: build(*_grid(*iters))

    grid = _grid

    indptr = ph_indptr.name
    main.smap = {
        "q_row0": _scalar(lambda b, h, *_: f"{b}"),
        "q_head0": _scalar(lambda b, h, *_: f"{h} * {group}"),
        "kv_base": _scalar(lambda b, h, *_: f"{indptr}[{b}]"),
        "valid_len": _scalar(lambda b, h, *_: f"{indptr}[{b} + 1] - {indptr}[{b}]"),
        "head": _scalar(lambda b, h, *_: f"{h}"),
        "scale2": _scalar(lambda b, h, *_: f"{spec.scale_input} * {LOG2E}"),
    }
    if split is not None:
        main.smap.update(
            _split_range_scalars(
                split, bn, lambda b: f"{indptr}[{b} + 1] - {indptr}[{b}]", _scalar
            )
        )
    lock_graph(main)

    built = _Built(
        ph_q=ph_q,
        ph_k=ph_k,
        ph_v=ph_v,
        ph_idx=ph_idx,
        ph_indptr=ph_indptr,
        loop_node=loop_node,
        inner_q=inner_q,
        inner_idx=inner_idx,
        gather_k=gather_k,
        gather_v=gather_v,
        score_mma=score_mma,
        exp=exp,
        rowsum=rowsum,
        value_mma=value_mma,
        div=div,
        log2=log2,
        outputs=outputs,
        split=split,
        store_d=store_d,
        store_o=store_o,
    )
    _build_tensor_maps(main, inner, spec, dense, built, grid)
    return _run_passes(main, inner, inner_idx)


def _split_range_scalars(split: int, bn: int, valid_len, scalar) -> dict:
    """``smap`` entries giving a split-KV block its range of the request's KV tiles.

    ``valid_len(b)`` is the request's token count as an expression over the grid index
    ``b``; ``scalar`` is the builder's grid-order adapter. The request's ``n_tiles`` are
    dealt out ``ceil(n_tiles / split)`` at a time, so the splits before the last are full
    and the last takes what remains -- possibly nothing, in which case ``my_trip`` is 0
    and the block stores zero partials, which the combine's sum is indifferent to.
    ``split_end`` is the token just past this block's last tile, clamped to the request,
    for the look-ahead guard. The three expressions repeat the indptr reads rather than
    naming an intermediate: the emitted call passes each as a keyword argument, and TVM
    folds the common subexpressions.
    """
    n_tiles = lambda b: f"T.ceildiv({valid_len(b)}, {bn})"  # noqa: E731
    per_split = lambda b: f"T.ceildiv({n_tiles(b)}, {split})"  # noqa: E731
    return {
        "tile0": scalar(lambda b, h, s: f"{s} * {per_split(b)}"),
        "my_trip": scalar(
            lambda b, h, s: f"T.max(T.min({per_split(b)}, {n_tiles(b)} - {s} * {per_split(b)}), 0)"
        ),
        "split_end": scalar(
            lambda b, h, s: f"T.min({valid_len(b)}, {bn} * ({s} * {per_split(b)} + {per_split(b)}))"
        ),
    }


def _build_paged_decode_slab(
    spec: PagingSpec, dense: DenseAttention, split: int | None = None
) -> Graph:
    """A paged decode whose KV tile is a contiguous slab inside one page.

    The same operator as ``_build_paged_decode`` at a page size above one, but not the same
    layer. Three things change together, and each depends on the others:

    * **No gather.** ``page_size % kv_tile == 0``, so a tile of BN keys lies wholly inside
      one page and its address is ``(one page id, an offset inside the page)`` -- an affine
      slab, not BN independent rows. The KV operands are therefore the cache placeholders
      themselves, sliced by a ``tmap`` Iterator whose leading index reads the page table.
    * **BufferRegions plus TMA.** A slab is contiguous, so a plain view already copies it
      as vectors and a region buys no wider access; what a region buys is a *TMA
      descriptor*, one bulk copy per tile instead of a per-element loop. And TMA is only
      lowered inside a **pipelined** loop, because that is what supplies the mbarrier
      protocol -- so the loop is pipelined at 2 stages, which also makes the manual
      ``prefetch_nodes`` of the ps1 layer unnecessary. Without TMA the region form is 5.5x
      slower than the view, so the three are one decision, not three.
    * **A page-unit indptr.** ``KV_INDPTR`` counts pages here, so it cannot give a token
      length: page count times page size rounds up to a whole page. The token length comes
      from a sixth input, ``KV_LAST_PAGE_LEN``.

    ``tl.disable_safe_memory_legalize`` is *not* wanted here (pass it ``pass_configs={}``):
    it exists to stop the legalizer re-reading a page-table entry inside a bounds test, and
    with one entry per tile rather than one per row there is nothing to re-read.

    With ``split`` this builds the partial half of the split-KV form, exactly as
    ``_build_paged_decode`` does for the row-gather layer: a trailing split grid axis, a
    per-block tile range, and the two accumulators stored raw. There is no look-ahead to
    bound here -- the pipelined loop's own trip count is the block's tile count.
    """
    mq, bn, head_dim = dense.query_rows, dense.kv_tile, dense.head_dim
    group, n_kv = spec.group, spec.num_kv_heads
    batch, pages, page = spec.batch_dim, spec.pages_dim, spec.page_size
    indptr_dim, index_dim = spec.indptr_dim, spec.index_dim
    if split is not None:
        assert isinstance(split, int) and split > 1, f"split={split!r} is not a split"

    # --- the loop tier: score one slab, accumulate ---------------------------------
    score_mma = Matmul(
        (mq, bn, head_dim),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_0",
        transpose_B=True,
        warp_policy=spec.warp_policy,
    )
    exp = MaskedExp(
        (mq, bn),
        scale=None,
        scalar_args=("tile_base", "valid_len", "scale2"),
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="exp_0",
        n_batch_dims=0,
    )
    rowsum = Reduction(
        (mq, bn),
        axis=-1,
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="rowsum_0",
    )
    value_mma = Matmul(
        (mq, head_dim, bn),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_1",
        warp_policy=spec.warp_policy,
    )
    value_mma.stage_fragment_operands_in_shared = (0,)

    _connect(score_mma, exp, 0, 0)
    _connect(exp, rowsum, 0, 0)
    _connect(exp, value_mma, 0, 0)

    # The caches are the matmuls' own operands: there is no gather between them and the
    # tile, only the slab slice that `_slab_tensor_maps` writes.
    ph_q = score_mma.inputs[0].src_node
    ph_k = score_mma.inputs[1].src_node
    ph_v = value_mma.inputs[1].src_node
    # The page table feeds no node -- it is read for the addresses in those slab slices --
    # but it is a declared kernel input and has to be in scope inside the tier that reads
    # it, so it is listed among both tiers' nodes.
    ph_idx = PlaceHolderNode(name=spec.index_input)
    ph_idx.set_shape((1,))
    ph_idx.set_dtype(_int32())
    ph_idx.is_batch_dim = False

    ph_q.name = f"PlaceHolder_{spec.query_input}"
    ph_k.name = f"PlaceHolder_{spec.cache_inputs[0]}"
    ph_v.name = f"PlaceHolder_{spec.cache_inputs[1]}"
    ph_idx.name = f"PlaceHolder_{spec.index_input}"

    from .util import gather_all_nodes, lock_graph

    inner = Graph(nodes=[ph_idx] + gather_all_nodes(rowsum, value_mma))
    inner.extent = 1
    # Pipelined for TMA, not for latency: a copy is only lowered to a bulk transfer when
    # it sits inside a pipelined producer/consumer loop, because that is what supplies the
    # mbarrier arrive/wait protocol.
    inner.loop_kind = "pipelined"
    inner.loop_stages = 2
    inner.sargs = [
        ("q_row0", "int32"),
        ("q_head0", "int32"),
        ("kv_page_base", "int32"),
        ("valid_len", "int32"),
        ("head", "int32"),
        ("scale2", "float32"),
    ]
    inner.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("scale2", "float32"),
    ]
    inner.smap = {
        "q_row0": lambda i: "q_row0",
        "q_head0": lambda i: "q_head0",
        "kv_page_base": lambda i: "kv_page_base",
        "head": lambda i: "head",
        "valid_len": lambda i: "valid_len",
        "scale2": lambda i: "scale2",
    }
    if split is None:
        inner.loop_trip = f"T.ceildiv(valid_len, {bn})"
        inner.smap["tile_base"] = lambda i: f"{bn} * {i}"
    else:
        inner.loop_trip = "my_trip"
        inner.sargs += [("tile0", "int32"), ("my_trip", "int32")]
        inner.smap["tile0"] = lambda i: "tile0"
        inner.smap["my_trip"] = lambda i: "my_trip"
        inner.smap["tile_base"] = lambda i: f"{bn} * (tile0 + {i})"
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    rowsum_id = inner.accum_nodes.index(rowsum)
    value_id = inner.accum_nodes.index(value_mma)
    inner_q = inner.binding[ph_q.name]
    inner_idx = inner.binding[ph_idx.name]
    inner_k = inner.binding[ph_k.name]
    inner_v = inner.binding[ph_v.name]
    # A slab lies inside one page addressed by a scalar page id, which tilelang can turn
    # into a TMA descriptor. This is the only reason the tiles are regions.
    inner.region_args = [inner_k, inner_v]

    q_tile = RowGather(
        (mq, head_dim),
        source_rows=batch,
        n_valid=group,
        src_dims=3,
        source_heads=spec.num_qo_heads,
        scalar_args=("q_row0", "q_head0"),
        dtype=dense.dtype,
        out_dtype=dense.dtype,
        name="q_tile",
    )
    q_tile.source_args = (ph_q.name,)
    inner_q.producer_op = q_tile

    # --- the epilogue -------------------------------------------------------------
    div, log2, store_d, store_o, outputs = _decode_epilogue(
        spec, dense, loop_node, rowsum, rowsum_id, value_mma, value_id, split
    )

    # Read for their values, not gathered from, so they stay out of the *_iter call.
    ph_indptr = PlaceHolderNode(name=spec.indptr_input)
    ph_indptr.set_shape((1,))
    ph_indptr.set_dtype(_int32())
    ph_indptr.index_only = True
    ph_indptr.name = f"PlaceHolder_{spec.indptr_input}"
    ph_last = PlaceHolderNode(name=spec.last_page_len_input)
    ph_last.set_shape((1,))
    ph_last.set_dtype(_int32())
    ph_last.index_only = True
    ph_last.name = f"PlaceHolder_{spec.last_page_len_input}"

    main = Graph(
        nodes=[ph_q, ph_k, ph_v, ph_indptr, ph_idx, ph_last]
        + gather_all_nodes(loop_node)
    )
    main.dyn_dims = [batch, pages, indptr_dim, index_dim]
    main.scalar_params = [(spec.scale_input, "float32")]
    main.n_threads = dense.n_threads
    main.pass_configs = dict(spec.pass_configs)
    main.rasterization = spec.rasterization
    batch_first = tuple(spec.grid_order) == ("batch", "head")
    main.extent = (f"{batch}", f"{n_kv}") if batch_first else (f"{n_kv}", f"{batch}")
    if split is not None:
        main.extent += (f"{split}",)
    main.sargs = list(inner.sargs)

    def grid(*iters):
        lead = iters[:2] if batch_first else tuple(reversed(iters[:2]))
        return (*lead, *iters[2:])

    def _scalar(build):
        return lambda *iters: build(*grid(*iters))

    indptr, last = ph_indptr.name, ph_last.name

    def n_pages(b):
        return f"({indptr}[{b} + 1] - {indptr}[{b}])"

    # Every page but the last is full; the last holds kv_last_page_len tokens. The whole
    # sum is clamped to the pages the request owns, which is what keeps a corrupt or
    # differently-conventioned entry from attending to the *next* request's pages --
    # clamping only the last term to `n_pages * page_size` would leave it up to
    # `(n_pages - 1) * page_size` tokens of room to do so. The clamp doubles as the
    # empty-request case: at zero pages it is min(anything, 0) = 0.
    def valid_len(b):
        return (
            f"T.min(T.max({n_pages(b)} - 1, 0) * {page}"
            f" + T.max({last}[{b}], 0), {n_pages(b)} * {page})"
        )

    main.smap = {
        "q_row0": _scalar(lambda b, h, *_: f"{b}"),
        "q_head0": _scalar(lambda b, h, *_: f"{h} * {group}"),
        # In pages: the slab slice multiplies back up to tokens itself.
        "kv_page_base": _scalar(lambda b, h, *_: f"{indptr}[{b}]"),
        "valid_len": _scalar(lambda b, h, *_: valid_len(b)),
        "head": _scalar(lambda b, h, *_: f"{h}"),
        "scale2": _scalar(lambda b, h, *_: f"{spec.scale_input} * {LOG2E}"),
    }
    if split is not None:
        main.smap.update(_split_range_scalars(split, bn, valid_len, _scalar))
    lock_graph(main)

    _slab_tensor_maps(
        main,
        inner,
        spec,
        dense,
        grid=grid,
        nodes=dict(
            ph_q=ph_q,
            ph_k=ph_k,
            ph_v=ph_v,
            ph_idx=ph_idx,
            ph_indptr=ph_indptr,
            ph_last=ph_last,
            loop_node=loop_node,
            inner_q=inner_q,
            inner_idx=inner_idx,
            inner_k=inner_k,
            inner_v=inner_v,
            score_mma=score_mma,
            exp=exp,
            rowsum=rowsum,
            value_mma=value_mma,
            div=div,
            log2=log2,
            outputs=outputs,
            split=split,
            store_d=store_d,
            store_o=store_o,
        ),
    )
    return _run_passes(main, inner, inner_idx)


def _decode_epilogue(spec, dense, loop_node, rowsum, rowsum_id, value_mma, value_id, split):
    """The decode layers' epilogue: normalise and emit the lse, or store the raw partials.

    Returns ``(div, log2, store_d, store_o, outputs)``; the unused half is None. Shared by
    the two GQA decode layers, whose epilogues are identical -- the MLA layer folds its
    denominator first and builds its own.
    """
    group, head_dim = spec.group, dense.head_dim
    div = log2 = store_d = store_o = None
    if split is None:
        div = Div(
            (group, head_dim),
            broadcast=0,
            denom_floor=DENOM_FLOOR,
            dtype=dense.accum_dtype,
            out_dtype=dense.out_dtype,
            name="div_0",
            n_batch_dims=0,
        )
        value_edge = AccumEdge(loop_node, div, value_id, 0, accum_node=value_mma)
        loop_node._out_edges.append(value_edge)
        div.set_inputs(0, value_edge)
        rowsum_edge = AccumEdge(loop_node, div, rowsum_id, 1, accum_node=rowsum)
        loop_node._out_edges.append(rowsum_edge)
        div.set_inputs(1, rowsum_edge)

        outputs = [OutputNode(div, dtype=dense.out_dtype)]
        outputs[0].name = "OutputNode_0"
        if spec.emit_lse:
            log2 = Log2(
                (group,),
                dtype=dense.accum_dtype,
                out_dtype=dense.accum_dtype,
                name="log2_0",
                n_batch_dims=0,
            )
            lse_edge = AccumEdge(loop_node, log2, rowsum_id, 0, accum_node=rowsum)
            loop_node._out_edges.append(lse_edge)
            log2.set_inputs(0, lse_edge)
            outputs.append(OutputNode(log2, dtype=dense.accum_dtype))
            outputs[1].name = "OutputNode_1"
        return div, log2, store_d, store_o, outputs

    # Row sum first, then the value matmul: the order the dense split-K goldens use, and
    # the order the combine kernel declares its inputs in -- the fused emitter pairs the
    # two lists positionally.
    store_d = Copy(
        (group,),
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="partial_rowsum",
        n_batch_dims=0,
    )
    d_edge = AccumEdge(loop_node, store_d, rowsum_id, 0, accum_node=rowsum)
    loop_node._out_edges.append(d_edge)
    store_d.set_inputs(0, d_edge)
    store_o = Copy(
        (group, head_dim),
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="partial_value",
        n_batch_dims=0,
    )
    o_edge = AccumEdge(loop_node, store_o, value_id, 0, accum_node=value_mma)
    loop_node._out_edges.append(o_edge)
    store_o.set_inputs(0, o_edge)
    outputs = [
        OutputNode(store_d, dtype=dense.accum_dtype),
        OutputNode(store_o, dtype=dense.accum_dtype),
    ]
    outputs[0].name = "OutputNode_0"
    outputs[1].name = "OutputNode_1"
    return div, log2, store_d, store_o, outputs


def _partial_output_maps(main, outputs, store_d, store_o, split, heads, group, head_dim,
                         batch_dim, grid) -> None:
    """``tmap``/``tinfo_map`` entries for a split-KV partial's two stores.

    The partials are indexed by split first, so one split's contribution to every request
    is contiguous and the combine reads a request's splits at one stride. ``grid`` maps the
    launch indices to ``(request, head block, split)``.
    """

    def _d_offset(*iters):
        b, h, s = grid(*iters)
        return (f"{s}", f"{b}", f"{group} * {h}")

    def _o_offset(*iters):
        b, h, s = grid(*iters)
        return (f"{s}", f"{b}", f"{group} * {h}", "0")

    main.tmap[store_d] = Iterator(
        _d_offset, shape=(split, batch_dim, heads), tile_size=(1, 1, group)
    )
    main.tmap[outputs[0]] = main.tmap[store_d]
    main.tmap[store_o] = Iterator(
        _o_offset, shape=(split, batch_dim, heads, head_dim), tile_size=(1, 1, group, head_dim)
    )
    main.tmap[outputs[1]] = main.tmap[store_o]
    main.tinfo_map[store_d] = ("OutputNode_0", "g")
    main.tinfo_map[outputs[0]] = ("OutputNode_0", "g")
    main.tinfo_map[store_o] = ("OutputNode_1", "g")
    main.tinfo_map[outputs[1]] = ("OutputNode_1", "g")


def _slab_tensor_maps(main, inner, spec, dense, *, grid, nodes) -> None:
    """Fill both tiers' maps for the slab layer.

    The one interesting entry is the KV slice: a 4-D Iterator over the cache whose leading
    index is a page-table read and whose tile is ``(1, BN, 1, head_dim)`` -- one page, BN
    consecutive tokens of it, one kv head, the whole head dim.
    """
    mq, bn, head_dim = dense.query_rows, dense.kv_tile, dense.head_dim
    group, page = spec.group, spec.page_size
    ph_q, ph_k, ph_v = nodes["ph_q"], nodes["ph_k"], nodes["ph_v"]
    ph_idx, ph_indptr, ph_last = nodes["ph_idx"], nodes["ph_indptr"], nodes["ph_last"]
    loop_node, inner_q, inner_idx = nodes["loop_node"], nodes["inner_q"], nodes["inner_idx"]
    inner_k, inner_v = nodes["inner_k"], nodes["inner_v"]
    score_mma, exp = nodes["score_mma"], nodes["exp"]
    rowsum, value_mma = nodes["rowsum"], nodes["value_mma"]
    div, log2, outputs = nodes["div"], nodes["log2"], nodes["outputs"]
    split = nodes.get("split")

    q_shape = (spec.batch_dim, spec.num_qo_heads, head_dim)
    cache_shape = (spec.pages_dim, page, spec.num_kv_heads, head_dim)

    def whole(shape):
        return Iterator(
            lambda *iters: tuple("0" for _ in shape),
            shape=tuple(shape),
            tile_size=tuple(shape),
        )

    def inner_tile(shape):
        return Iterator(
            lambda i: tuple("0" for _ in shape),
            shape=tuple(shape),
            tile_size=tuple(shape),
        )

    main.tmap[ph_q] = whole(q_shape)
    main.tmap[ph_k] = whole(cache_shape)
    main.tmap[ph_v] = whole(cache_shape)
    main.tmap[ph_idx] = whole((spec.index_dim,))
    main.tmap[ph_indptr] = whole((spec.indptr_dim,))
    main.tmap[ph_last] = whole((spec.batch_dim,))
    main.tmap[loop_node] = [
        Iterator(lambda *i: ("0",), shape=(mq,), tile_size=(mq,)),
        Iterator(lambda *i: ("0", "0"), shape=(mq, head_dim), tile_size=(mq, head_dim)),
    ]

    def _out_offset(*iters):
        b, h = grid(*iters)[:2]
        return (f"{b}", f"{group} * {h}", "0")

    def _lse_offset(*iters):
        b, h = grid(*iters)[:2]
        return (f"{b}", f"{group} * {h}")

    if split is None:
        main.tmap[div] = Iterator(
            _out_offset,
            shape=(spec.batch_dim, spec.num_qo_heads, head_dim),
            tile_size=(1, group, head_dim),
        )
        main.tmap[outputs[0]] = main.tmap[div]
        if log2 is not None:
            main.tmap[log2] = Iterator(
                _lse_offset,
                shape=(spec.batch_dim, spec.num_qo_heads),
                tile_size=(1, group),
            )
            main.tmap[outputs[1]] = main.tmap[log2]

    inner.extent = 1
    inner.tmap[ph_q] = inner_tile(q_shape)
    inner.tmap[inner_q] = inner_tile((mq, head_dim))
    inner.tmap[inner_idx] = inner_tile((spec.index_dim,))

    # `kv_page_base` counts pages and `tile_base` counts tokens within the request, so the
    # page holding tile i is `kv_page_base + tile_base / page_size`, and the offset inside
    # it `tile_base % page_size`. Both index reads are clamped, so a corrupt page table
    # reads a wrong page rather than off the end of the pool. A split-KV block's tile i is
    # the request's tile `tile0 + i`.
    def tokens(i):
        return f"({bn} * {i})" if split is None else f"({bn} * (tile0 + {i}))"

    def page_of(i):
        return (
            f"T.min({inner_idx.name}[T.min(kv_page_base + {tokens(i)} // {page}, "
            f"{spec.index_dim} - 1)], {spec.pages_dim} - 1)"
        )

    def slab():
        return Iterator(
            lambda i: (page_of(i), f"{tokens(i)} % {page}", "head", "0"),
            shape=cache_shape,
            tile_size=(1, bn, 1, head_dim),
        )

    inner.tmap[inner_k] = slab()
    inner.tmap[inner_v] = slab()
    inner.tmap[score_mma] = inner_tile((mq, bn))
    inner.tmap[exp] = inner.tmap[score_mma]
    inner.tmap[rowsum] = inner_tile((mq,))
    inner.tmap[value_mma] = inner_tile((mq, head_dim))

    main.tinfo_map = {
        ph_q: (ph_q.name, "g"),
        ph_k: (ph_k.name, "g"),
        ph_v: (ph_v.name, "g"),
        ph_idx: (ph_idx.name, "g"),
        ph_indptr: (ph_indptr.name, "g"),
        ph_last: (ph_last.name, "g"),
        loop_node: ("t_GraphNode_0", "r"),
    }
    if split is None:
        main.tinfo_map[div] = ("OutputNode_0", "g")
        main.tinfo_map[outputs[0]] = ("OutputNode_0", "g")
        if log2 is not None:
            main.tinfo_map[log2] = ("OutputNode_1", "g")
            main.tinfo_map[outputs[1]] = ("OutputNode_1", "g")
    else:
        _partial_output_maps(
            main, outputs, nodes["store_d"], nodes["store_o"], split,
            spec.num_qo_heads, group, head_dim, spec.batch_dim, grid,
        )

    inner.tinfo_map = {
        ph_q: (ph_q.name, "g"),
        ph_k: (ph_k.name, "g"),
        ph_v: (ph_v.name, "g"),
        ph_idx: (ph_idx.name, "g"),
        inner_q: (inner_q.name, "s"),
        inner_idx: (inner_idx.name, "g"),
        # Shared, unlike the ps1 layer's global cache view: the slab is copied into shared
        # memory by the TMA the region unlocks, and it is the matmul operand itself.
        inner_k: (inner_k.name, "s"),
        inner_v: (inner_v.name, "s"),
        score_mma: ("t_mma_0", "r"),
        exp: ("t_exp_0", "r"),
        rowsum: ("t_rowsum_0", "r"),
        value_mma: ("t_mma_1", "r"),
    }


def _build_ragged_prefill_causal(spec: PagingSpec, dense: DenseAttention) -> Graph:
    """A causal prefill over ragged (already-contiguous) keys.

    There is no page table here, and so no gather: a request's keys are a contiguous run of
    the packed KV axis, which means the KV tile is an affine slice. That single fact drives
    everything that differs from the paged prefill layer:

      * the tile is handed down as a ``BufferRegion`` (``region_args``) rather than staged
        by a gather ukernel, which is what lets tilelang reach for TMA;
      * TMA's mbarrier protocol needs a pipelined producer/consumer body to supply
        arrivals, so the loop is pipelined rather than serial -- and a pipelined loop is
        only sound *because* the source is affine; pipelining a gather reorders an
        unbuffered read;
      * one CTA owns one query head rather than a whole head group, because there is no
        gathered tile to amortise over, so the grid is over query heads and the kv head is
        derived from it.
    """
    from .util import gather_all_nodes, lock_graph

    bm, bn, head_dim = dense.query_rows, dense.kv_tile, dense.head_dim
    group = spec.group
    tokens, kv_tokens = spec.tokens_dim, spec.kv_tokens_dim
    indptr_dim = spec.indptr_dim
    # Keeping P in a fragment (see the note on the value matmul below) makes the P tile
    # serve two masters: the mask/exp `T.Parallel` loop's layout and the second gemm's
    # operand-A layout. Measured at bn=64, those agree only for a 128-thread block whose
    # query tile is a multiple of 64 rows; everything else fails deep inside tilelang with
    # "Layout infer conflict between A_ptr and A_ptr" or "No valid warp partition for
    # T.gemm", neither of which names anything in this file. The tiling policy really does
    # propose the rejected points -- it expands every candidate into a 128- and a
    # 256-thread variant -- so this is the difference between a searchable constraint and
    # an unattributable crash.
    assert bm >= 64 and bm % 64 == 0, (
        f"the ragged layer needs a query tile of 64 rows or a multiple of it, got {bm}: it "
        "keeps P in a fragment for wgmma_rs, and any other height makes the mask loop's "
        "fragment layout and the value matmul's operand layout irreconcilable"
    )
    assert dense.n_threads == 128, (
        f"the ragged layer needs a 128-thread block, got {dense.n_threads}: with P in a "
        "fragment, a wider block cannot reconcile the mask loop's layout with the value "
        "matmul's operand layout at any query tile (measured at kv_tile 64)"
    )

    score_mma = Matmul(
        (bm, bn, head_dim),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_0",
        transpose_B=True,
        warp_policy=spec.warp_policy,
    )
    # No `row_group`: a row of this tile is a query token, not a (token, head slot) pair.
    exp = CausalMaskedExp(
        (bm, bn),
        scale=None,
        scalar_args=("tile_base", "valid_len", "diag", "scale2"),
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="exp_0",
        n_batch_dims=0,
    )
    rowsum = Reduction(
        (bm, bn),
        axis=-1,
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="rowsum_0",
    )
    value_mma = Matmul(
        (bm, head_dim, bn),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_1",
        warp_policy=spec.warp_policy,
    )
    # P stays in its fragment rather than being staged through shared: tilelang then feeds
    # it to the second matmul in place (wgmma_rs) instead of storing it with ptx_stmatrix
    # and reading it back. The paged layers keep the shared hop, whose P tile is the one
    # their layout pass needs.

    _connect(score_mma, exp, 0, 0)
    _connect(exp, rowsum, 0, 0)
    _connect(exp, value_mma, 0, 0)

    ph_q = score_mma.inputs[0].src_node
    ph_k = score_mma.inputs[1].src_node
    ph_v = value_mma.inputs[1].src_node
    ph_q.name = f"PlaceHolder_{spec.query_input}"
    ph_k.name = f"PlaceHolder_{spec.cache_inputs[0]}"
    ph_v.name = f"PlaceHolder_{spec.cache_inputs[1]}"

    inner = Graph(nodes=gather_all_nodes(rowsum, value_mma))
    inner.extent = 1
    inner.loop_kind = "pipelined"
    inner.loop_stages = 2
    inner.loop_trip = f"T.ceildiv(valid_len, {bn})"
    body_sargs = [
        ("q_row0", "int32"),
        ("q_tile_base", "int32"),
        ("q_len", "int32"),
        ("q_head", "int32"),
        ("kv_row0", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("kv_head", "int32"),
        ("scale2", "float32"),
    ]
    inner.sargs = list(body_sargs)
    inner.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("scale2", "float32"),
        ("kv_row0", "int32"),
        ("kv_head", "int32"),
    ]
    inner.smap = {
        "q_row0": lambda i: "q_row0",
        "q_tile_base": lambda i: "q_tile_base",
        "q_len": lambda i: "q_len",
        "q_head": lambda i: "q_head",
        "kv_row0": lambda i: "kv_row0",
        "valid_len": lambda i: "valid_len",
        "diag": lambda i: "diag",
        "kv_head": lambda i: "kv_head",
        "scale2": lambda i: "scale2",
        "tile_base": lambda i: f"{bn} * {i}",
    }
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    rowsum_id = inner.accum_nodes.index(rowsum)
    value_id = inner.accum_nodes.index(value_mma)
    inner_q = inner.binding[ph_q.name]
    inner_k = inner.binding[ph_k.name]
    inner_v = inner.binding[ph_v.name]

    div = MaskedDiv(
        (bm, head_dim),
        broadcast=0,
        scalar_args=("n_rows",),
        denom_floor=DENOM_FLOOR,
        dtype=dense.accum_dtype,
        out_dtype=dense.out_dtype,
        name="div_0",
        n_batch_dims=0,
    )
    value_edge = AccumEdge(loop_node, div, value_id, 0, accum_node=value_mma)
    loop_node._out_edges.append(value_edge)
    div.set_inputs(0, value_edge)
    rowsum_edge = AccumEdge(loop_node, div, rowsum_id, 1, accum_node=rowsum)
    loop_node._out_edges.append(rowsum_edge)
    div.set_inputs(1, rowsum_edge)

    outputs = [OutputNode(div, dtype=dense.out_dtype)]
    outputs[0].name = "OutputNode_0"
    log2 = None
    if spec.emit_lse:
        log2 = MaskedLog2(
            (bm,),
            scalar_args=("n_rows",),
            dtype=dense.accum_dtype,
            out_dtype=dense.accum_dtype,
            name="log2_0",
            n_batch_dims=0,
        )
        lse_edge = AccumEdge(loop_node, log2, rowsum_id, 0, accum_node=rowsum)
        loop_node._out_edges.append(lse_edge)
        log2.set_inputs(0, lse_edge)
        outputs.append(OutputNode(log2, dtype=dense.accum_dtype))
        outputs[1].name = "OutputNode_1"

    indptrs = []
    for name in (spec.query_indptr_input, spec.indptr_input):
        node = PlaceHolderNode(name=name)
        node.set_shape((1,))
        node.set_dtype(_int32())
        node.index_only = True
        node.name = f"PlaceHolder_{name}"
        indptrs.append(node)
    ph_qo, ph_kv = indptrs

    main = Graph(nodes=[ph_q, ph_k, ph_v, ph_qo, ph_kv] + gather_all_nodes(loop_node))
    main.dyn_dims = [tokens, kv_tokens, indptr_dim]
    main.scalar_params = [(spec.scale_input, "float32")]
    main.pass_configs = dict(spec.pass_configs)
    main.n_threads = dense.n_threads
    main.rasterization = spec.rasterization
    # One CTA per (query tile, query head); see the paged prefill layer for why the slack
    # term covers the work list.
    main.extent = (
        f"T.ceildiv({tokens}, {bm}) + {indptr_dim} - 2",
        f"{spec.num_qo_heads}",
    )
    main.sargs = [("n_rows", "int32")] + body_sargs

    q_beg = f"{ph_qo.name}[_wl[0]]"
    q_end = f"{ph_qo.name}[_wl[0] + 1]"
    kv_beg = f"{ph_kv.name}[_wl[0]]"
    kv_end = f"{ph_kv.name}[_wl[0] + 1]"
    q_len = f"({q_end} - {q_beg})"
    kv_len = f"({kv_end} - {kv_beg})"
    tile_base = f"(_wl[1] * {bm})"
    n_rows = f"T.min({bm}, {q_len} - {tile_base})"
    diag = f"({kv_len} - {q_len} + {tile_base})"
    main.smap = {
        "n_rows": lambda i, h: n_rows,
        "q_row0": lambda i, h: q_beg,
        "q_tile_base": lambda i, h: tile_base,
        "q_len": lambda i, h: q_len,
        "q_head": lambda i, h: f"{h}",
        "kv_row0": lambda i, h: kv_beg,
        "valid_len": lambda i, h: f"T.max(T.min({kv_len}, {diag} + {n_rows}), 0)",
        "diag": lambda i, h: diag,
        # The grid is over QUERY heads, so the kv head a block reads follows from it.
        "kv_head": lambda i, h: f"{h} // {group}",
        "scale2": lambda i, h: f"{spec.scale_input} * {LOG2E}",
    }
    main.prologue = [
        ScalarScan(
            op=IndptrTileScan(rows=bm),
            buffer="_wl",
            size=2,
            inputs=(ph_qo,),
            scalar_args=lambda i, h: (i, indptr_dim),
        )
    ]
    main.block_guard = lambda i, h: "_wl[0] >= 0"
    # A BufferRegion is what lets tilelang bounds-predicate and bulk-copy the slice; an
    # `_S()` view is just a buffer of the slice's extent.
    main.region_args = [ph_q]
    lock_graph(main)

    q_abs = f"{q_beg} + {tile_base}"
    q_shape = (tokens, spec.num_qo_heads, head_dim)
    kv_shape = (kv_tokens, spec.num_kv_heads, head_dim)

    def whole(shape):
        return Iterator(
            lambda i, h: tuple("0" for _ in shape),
            shape=tuple(shape),
            tile_size=tuple(shape),
        )

    def inner_whole(shape):
        return Iterator(
            lambda i: tuple("0" for _ in shape),
            shape=tuple(shape),
            tile_size=tuple(shape),
        )

    main.tmap[ph_q] = Iterator(
        lambda i, h: (q_abs, f"{h}", "0"), shape=q_shape, tile_size=(bm, 1, head_dim)
    )
    main.tmap[ph_k] = whole(kv_shape)
    main.tmap[ph_v] = whole(kv_shape)
    main.tmap[ph_qo] = whole((indptr_dim,))
    main.tmap[ph_kv] = whole((indptr_dim,))
    main.tmap[loop_node] = [
        Iterator(lambda i, h: ("0",), shape=(bm,), tile_size=(bm,)),
        Iterator(lambda i, h: ("0", "0"), shape=(bm, head_dim), tile_size=(bm, head_dim)),
    ]
    out_iter = Iterator(
        lambda i, h: (q_abs, f"{h}", "0"), shape=q_shape, tile_size=(bm, 1, head_dim)
    )
    lse_iter = Iterator(
        lambda i, h: (q_abs, f"{h}"),
        shape=(tokens, spec.num_qo_heads),
        tile_size=(bm, 1),
    )
    main.tmap[div] = out_iter
    main.tmap[outputs[0]] = out_iter
    if log2 is not None:
        main.tmap[log2] = lse_iter
        main.tmap[outputs[1]] = lse_iter

    def kv_tile():
        """One request's KV window, `bn` tokens at a time -- an affine slice, so TMA can
        take it and the loop can be pipelined."""
        return Iterator(
            lambda i: (f"kv_row0 + {bn} * {i}", "kv_head", "0"),
            shape=kv_shape,
            tile_size=(bn, 1, head_dim),
        )

    inner.extent = 1
    inner.tmap[inner_q] = inner_whole((bm, head_dim))
    inner.tmap[inner_k] = kv_tile()
    inner.tmap[inner_v] = kv_tile()
    inner.region_args = [inner_k, inner_v]
    inner.tmap[score_mma] = inner_whole((bm, bn))
    inner.tmap[exp] = inner.tmap[score_mma]
    inner.tmap[rowsum] = inner_whole((bm,))
    inner.tmap[value_mma] = inner_whole((bm, head_dim))

    main.tinfo_map = {
        ph_q: (ph_q.name, "g"),
        ph_k: (ph_k.name, "g"),
        ph_v: (ph_v.name, "g"),
        ph_qo: (ph_qo.name, "g"),
        ph_kv: (ph_kv.name, "g"),
        loop_node: ("t_GraphNode_0", "r"),
        div: ("OutputNode_0", "g"),
        outputs[0]: ("OutputNode_0", "g"),
    }
    if log2 is not None:
        main.tinfo_map[log2] = ("OutputNode_1", "g")
        main.tinfo_map[outputs[1]] = ("OutputNode_1", "g")
    inner.tinfo_map = {
        ph_q: (ph_q.name, "g"),
        ph_k: (ph_k.name, "g"),
        ph_v: (ph_v.name, "g"),
        inner_q: (inner_q.name, "s"),
        inner_k: (inner_k.name, "s"),
        inner_v: (inner_v.name, "s"),
        score_mma: ("t_mma_0", "r"),
        exp: ("t_exp_0", "r"),
        rowsum: ("t_rowsum_0", "r"),
        value_mma: ("t_mma_1", "r"),
    }
    return _run_passes(main)


def _build_paged_prefill_causal(spec: PagingSpec, dense: DenseAttention) -> Graph:
    """A causal prefill over a paged cache, keyed on a device-scanned work list.

    Three things separate this from the decode layer, all of them consequences of a
    request having many query tokens rather than one:

      * the grid is over query TILES, and which request a tile belongs to is not known
        until a device-side scan of ``qo_indptr`` says so, which is what the prologue and
        ``block_guard`` are for;
      * the mask is causal, so a row's KV window ends at its own diagonal, and the block's
        M axis fuses (token, head slot) so one gathered KV tile serves the whole head
        group;
      * a tile whose causal window is empty has no attention to do at all, and on the
        captured workloads that is most of them, so it takes an alternative producer that
        just writes the zeros and -inf the operator defines.
    """
    from .util import gather_all_nodes, lock_graph

    bm, bn, head_dim = dense.query_rows, dense.kv_tile, dense.head_dim
    group, n_kv = spec.group, spec.num_kv_heads
    # Query tokens per block: the M axis is (token, head slot), so a block owns this many
    # tokens of a single kv head's group.
    btok = bm // group
    tokens, pages = spec.tokens_dim, spec.pages_dim
    indptr_dim, index_dim = spec.indptr_dim, spec.index_dim

    # --- the loop tier ---------------------------------------------------------------
    gather_kwargs = dict(
        source_rows=pages,
        index_extent=index_dim,
        page_size=spec.page_size,
        paged_axes=True,
        source_heads=n_kv,
        head_scalar=True,
        clamp_index=BUFFER_BOUND,
        clamp_row=BUFFER_BOUND,
        scalar_args=("kv_pos0", "kv_head"),
        dtype=dense.dtype,
        out_dtype=dense.dtype,
    )
    gather_k = PagedGather((bn, head_dim), name="paged_gather_k", **gather_kwargs)
    gather_v = PagedGather((bn, head_dim), name="paged_gather_v", **gather_kwargs)
    score_mma = Matmul(
        (bm, bn, head_dim),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_0",
        transpose_B=True,
        warp_policy=spec.warp_policy,
    )
    # `diag` is the tile's own end-aligned causal bound; `row_group` is what makes a row
    # index mean a token rather than a head slot.
    exp = CausalMaskedExp(
        (bm, bn),
        scale=None,
        scalar_args=("tile_base", "valid_len", "diag", "scale2"),
        row_group=group,
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="exp_0",
        n_batch_dims=0,
    )
    rowsum = Reduction(
        (bm, bn),
        axis=-1,
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="rowsum_0",
    )
    value_mma = Matmul(
        (bm, head_dim, bn),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_1",
        warp_policy=spec.warp_policy,
    )
    value_mma.stage_fragment_operands_in_shared = (0,)

    _connect(gather_k, score_mma, 0, 1)
    _connect(score_mma, exp, 0, 0)
    _connect(exp, rowsum, 0, 0)
    _connect(exp, value_mma, 0, 0)
    _connect(gather_v, value_mma, 0, 1)

    ph_k = gather_k.inputs[0].src_node
    ph_v = gather_v.inputs[0].src_node
    ph_q = score_mma.inputs[0].src_node
    ph_idx = gather_k.inputs[1].src_node
    _connect(ph_idx, gather_v, 0, 1)

    ph_q.name = f"PlaceHolder_{spec.query_input}"
    ph_k.name = f"PlaceHolder_{spec.cache_inputs[0]}"
    ph_v.name = f"PlaceHolder_{spec.cache_inputs[1]}"
    ph_idx.name = f"PlaceHolder_{spec.index_input}"

    inner = Graph(nodes=gather_all_nodes(rowsum, value_mma))
    inner.extent = 1
    inner.loop_kind = "serial"
    inner.loop_trip = f"T.ceildiv(valid_len, {bn})"
    inner.prefetch_nodes = [gather_k, gather_v]
    inner.prefetch_guard = f"tile_base + {bn} < valid_len"
    body_sargs = [
        ("q_row0", "int32"),
        ("q_tile_base", "int32"),
        ("q_len", "int32"),
        ("q_head0", "int32"),
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("kv_head", "int32"),
        ("scale2", "float32"),
    ]
    inner.sargs = list(body_sargs)
    inner.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("scale2", "float32"),
        ("kv_pos0", "int32"),
        ("kv_head", "int32"),
    ]
    inner.smap = {
        "q_row0": lambda i: "q_row0",
        "q_tile_base": lambda i: "q_tile_base",
        "q_len": lambda i: "q_len",
        "q_head0": lambda i: "q_head0",
        "kv_base": lambda i: "kv_base",
        "valid_len": lambda i: "valid_len",
        "diag": lambda i: "diag",
        "kv_head": lambda i: "kv_head",
        "scale2": lambda i: "scale2",
        "tile_base": lambda i: f"{bn} * {i}",
        # The tile's first kv position, in tokens, which is where the gathers start
        # reading the page table.
        "kv_pos0": lambda i: f"kv_base + {bn} * {i}",
    }
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    rowsum_id = inner.accum_nodes.index(rowsum)
    value_id = inner.accum_nodes.index(value_mma)
    inner_q = inner.binding[ph_q.name]
    inner_idx = inner.binding[ph_idx.name]

    # Q comes from a ragged token axis, so the tile is a token RANGE rather than a row and
    # a head; `row_group` splits the fused M axis back into (token, head slot).
    q_tile = RangeGather(
        (bm, head_dim),
        source_rows=tokens,
        src_dims=3,
        source_heads=spec.num_qo_heads,
        row_group=group,
        clamp_row=BUFFER_BOUND,
        scalar_args=("q_row0", "q_tile_base", "q_len", "q_head0"),
        dtype=dense.dtype,
        out_dtype=dense.dtype,
        name="q_tile",
    )
    q_tile.source_args = (ph_q.name,)
    inner_q.producer_op = q_tile

    # --- the epilogue ----------------------------------------------------------------
    # `n_rows` is how many of the tile's tokens the request actually has; the last tile of
    # a request is partial far more often than not, so both stores are row-predicated.
    div = MaskedDiv(
        (bm, head_dim),
        broadcast=0,
        scalar_args=("n_rows",),
        denom_floor=DENOM_FLOOR,
        row_group=group,
        dtype=dense.accum_dtype,
        out_dtype=dense.out_dtype,
        name="div_0",
        n_batch_dims=0,
    )
    value_edge = AccumEdge(loop_node, div, value_id, 0, accum_node=value_mma)
    loop_node._out_edges.append(value_edge)
    div.set_inputs(0, value_edge)
    rowsum_edge = AccumEdge(loop_node, div, rowsum_id, 1, accum_node=rowsum)
    loop_node._out_edges.append(rowsum_edge)
    div.set_inputs(1, rowsum_edge)

    outputs = [OutputNode(div, dtype=dense.out_dtype)]
    outputs[0].name = "OutputNode_0"
    log2 = None
    if spec.emit_lse:
        log2 = MaskedLog2(
            (bm,),
            scalar_args=("n_rows",),
            row_group=group,
            dtype=dense.accum_dtype,
            out_dtype=dense.accum_dtype,
            name="log2_0",
            n_batch_dims=0,
        )
        lse_edge = AccumEdge(loop_node, log2, rowsum_id, 0, accum_node=rowsum)
        loop_node._out_edges.append(lse_edge)
        log2.set_inputs(0, lse_edge)
        outputs.append(OutputNode(log2, dtype=dense.accum_dtype))
        outputs[1].name = "OutputNode_1"

    # The alternative producers for a tile with no causal window. They write the same
    # tiles as the epilogue, at the same per-kv-head shape -- one CTA per kv head, so the
    # store keeps the parallelism the attention path has.
    fills = [
        ConstFill(
            (bm, head_dim),
            value=0.0,
            out_dtype=dense.out_dtype,
            row_group=group,
            scalar_args=("n_rows",),
            name="fill_o",
        )
    ]
    if spec.emit_lse:
        fills.append(
            ConstFill(
                (bm,),
                value=float("-inf"),
                out_dtype=dense.accum_dtype,
                row_group=group,
                scalar_args=("n_rows",),
                name="fill_lse",
            )
        )

    # Read for their values, not gathered from, so they stay out of the *_iter call.
    indptrs = []
    for name in (spec.query_indptr_input, spec.indptr_input):
        node = PlaceHolderNode(name=name)
        node.set_shape((1,))
        node.set_dtype(_int32())
        node.index_only = True
        node.name = f"PlaceHolder_{name}"
        indptrs.append(node)
    ph_qo, ph_kv = indptrs

    main = Graph(
        nodes=[ph_q, ph_k, ph_v, ph_qo, ph_kv, ph_idx] + gather_all_nodes(loop_node)
    )
    main.dyn_dims = [tokens, pages, indptr_dim, index_dim]
    main.scalar_params = [(spec.scale_input, "float32")]
    main.pass_configs = dict(spec.pass_configs)
    main.n_threads = dense.n_threads
    main.rasterization = spec.rasterization
    # One CTA per (query tile, kv head). sum(ceil(q_len/btok)) is at most
    # ceil(total_q/btok) + n_requests - 1 (ceil is subadditive up to one per term), so this
    # grid always covers the work list; the slack slots scan to request -1 and the body
    # guard skips them.
    main.extent = (
        f"T.ceildiv({tokens}, {btok}) + {indptr_dim} - 2",
        f"{n_kv}",
    )
    main.sargs = [("n_rows", "int32")] + body_sargs

    # Every per-block scalar is an expression over the scan result -- _wl[0] is the
    # request, _wl[1] the tile within it -- and the two indptr tensors.
    q_beg = f"{ph_qo.name}[_wl[0]]"
    q_end = f"{ph_qo.name}[_wl[0] + 1]"
    kv_beg = f"{ph_kv.name}[_wl[0]]"
    kv_end = f"{ph_kv.name}[_wl[0] + 1]"
    q_len = f"({q_end} - {q_beg})"
    kv_len = f"({kv_end} - {kv_beg})"
    tile_base = f"(_wl[1] * {btok})"
    n_rows = f"T.min({btok}, {q_len} - {tile_base})"
    # End-aligned causal bound for the tile's first row; causality then leaves `live` for
    # the whole tile. `live <= 0` means every row's window is empty.
    diag = f"({kv_len} - {q_len} + {tile_base})"
    live = f"T.min({kv_len}, {diag} + {n_rows})"
    main.smap = {
        "n_rows": lambda i, h: n_rows,
        "q_row0": lambda i, h: q_beg,
        "q_tile_base": lambda i, h: tile_base,
        "q_len": lambda i, h: q_len,
        "q_head0": lambda i, h: f"{h} * {group}",
        "kv_base": lambda i, h: kv_beg,
        "valid_len": lambda i, h: f"T.max({live}, 0)",
        "diag": lambda i, h: diag,
        "kv_head": lambda i, h: f"{h}",
        "scale2": lambda i, h: f"{spec.scale_input} * {LOG2E}",
    }
    main.prologue = [
        ScalarScan(
            op=IndptrTileScan(rows=btok),
            buffer="_wl",
            size=2,
            inputs=(ph_qo,),
            scalar_args=lambda i, h: (i, indptr_dim),
        )
    ]
    main.block_guard = lambda i, h: "_wl[0] >= 0"
    # With an empty causal window the KV loop runs zero trips, so the accumulators stay
    # zero and the epilogue's floored divide yields exactly the 0 and -inf the fills
    # write -- the two branches agree by construction. Taking the fill instead skips
    # clearing the accumulator, gathering Q, and the divide.
    main.work_guard = lambda i, h: f"{live} > 0"
    main.else_nodes = list(fills)
    lock_graph(main)

    _prefill_tensor_maps(
        main,
        inner,
        spec,
        dense,
        btok=btok,
        q_abs=f"{q_beg} + {tile_base}",
        nodes=dict(
            ph_q=ph_q,
            ph_k=ph_k,
            ph_v=ph_v,
            ph_idx=ph_idx,
            ph_qo=ph_qo,
            ph_kv=ph_kv,
            loop_node=loop_node,
            inner_q=inner_q,
            inner_idx=inner_idx,
            gather_k=gather_k,
            gather_v=gather_v,
            score_mma=score_mma,
            exp=exp,
            rowsum=rowsum,
            value_mma=value_mma,
            div=div,
            log2=log2,
            outputs=outputs,
            fills=fills,
        ),
    )
    return _run_passes(main, inner, inner_idx)


def _prefill_tensor_maps(main, inner, spec, dense, *, btok, q_abs, nodes) -> None:
    """Both tiers' ``tmap``/``tinfo_map`` for the causal prefill layer."""
    bm, bn, head_dim = dense.query_rows, dense.kv_tile, dense.head_dim
    group = spec.group
    n = nodes
    q_shape = (spec.tokens_dim, spec.num_qo_heads, head_dim)
    cache_shape = (spec.pages_dim, spec.page_size, spec.num_kv_heads, head_dim)

    def whole(shape):
        return Iterator(
            lambda i, h: tuple("0" for _ in shape),
            shape=tuple(shape),
            tile_size=tuple(shape),
        )

    def inner_tile(shape):
        return Iterator(
            lambda i: tuple("0" for _ in shape),
            shape=tuple(shape),
            tile_size=tuple(shape),
        )

    main.tmap[n["ph_q"]] = whole(q_shape)
    main.tmap[n["ph_k"]] = whole(cache_shape)
    main.tmap[n["ph_v"]] = whole(cache_shape)
    main.tmap[n["ph_idx"]] = whole((spec.index_dim,))
    main.tmap[n["ph_qo"]] = whole((spec.indptr_dim,))
    main.tmap[n["ph_kv"]] = whole((spec.indptr_dim,))
    main.tmap[n["loop_node"]] = [
        Iterator(lambda i, h: ("0",), shape=(bm,), tile_size=(bm,)),
        Iterator(lambda i, h: ("0", "0"), shape=(bm, head_dim), tile_size=(bm, head_dim)),
    ]

    out_shape = (spec.tokens_dim, spec.num_qo_heads, head_dim)
    lse_shape = (spec.tokens_dim, spec.num_qo_heads)
    out_iter = Iterator(
        lambda i, h: (q_abs, f"{h} * {group}", "0"),
        shape=out_shape,
        tile_size=(btok, group, head_dim),
    )
    lse_iter = Iterator(
        lambda i, h: (q_abs, f"{h} * {group}"), shape=lse_shape, tile_size=(btok, group)
    )
    main.tmap[n["div"]] = out_iter
    main.tmap[n["outputs"][0]] = out_iter
    main.tmap[n["fills"][0]] = out_iter
    if n["log2"] is not None:
        main.tmap[n["log2"]] = lse_iter
        main.tmap[n["outputs"][1]] = lse_iter
        main.tmap[n["fills"][1]] = lse_iter

    inner.extent = 1
    inner.tmap[n["ph_q"]] = inner_tile(q_shape)
    inner.tmap[n["inner_q"]] = inner_tile((bm, head_dim))
    inner.tmap[n["inner_idx"]] = inner_tile((spec.index_dim,))
    inner.tmap[inner.binding[n["ph_k"].name]] = inner_tile(cache_shape)
    inner.tmap[inner.binding[n["ph_v"].name]] = inner_tile(cache_shape)
    inner.tmap[n["gather_k"]] = inner_tile((bn, head_dim))
    inner.tmap[n["gather_v"]] = inner_tile((bn, head_dim))
    inner.tmap[n["score_mma"]] = inner_tile((bm, bn))
    inner.tmap[n["exp"]] = inner.tmap[n["score_mma"]]
    inner.tmap[n["rowsum"]] = inner_tile((bm,))
    inner.tmap[n["value_mma"]] = inner_tile((bm, head_dim))

    main.tinfo_map = {
        n["ph_q"]: (n["ph_q"].name, "g"),
        n["ph_k"]: (n["ph_k"].name, "g"),
        n["ph_v"]: (n["ph_v"].name, "g"),
        n["ph_idx"]: (n["ph_idx"].name, "g"),
        n["ph_qo"]: (n["ph_qo"].name, "g"),
        n["ph_kv"]: (n["ph_kv"].name, "g"),
        n["loop_node"]: ("t_GraphNode_0", "r"),
        n["div"]: ("OutputNode_0", "g"),
        n["fills"][0]: ("OutputNode_0", "g"),
        n["outputs"][0]: ("OutputNode_0", "g"),
    }
    if n["log2"] is not None:
        main.tinfo_map[n["log2"]] = ("OutputNode_1", "g")
        main.tinfo_map[n["fills"][1]] = ("OutputNode_1", "g")
        main.tinfo_map[n["outputs"][1]] = ("OutputNode_1", "g")

    inner.tinfo_map = {
        n["ph_q"]: (n["ph_q"].name, "g"),
        n["ph_k"]: (n["ph_k"].name, "g"),
        n["ph_v"]: (n["ph_v"].name, "g"),
        n["ph_idx"]: (n["ph_idx"].name, "g"),
        n["inner_q"]: (n["inner_q"].name, "s"),
        n["inner_idx"]: (n["inner_idx"].name, "g"),
        inner.binding[n["ph_k"].name]: (inner.binding[n["ph_k"].name].name, "g"),
        inner.binding[n["ph_v"].name]: (inner.binding[n["ph_v"].name].name, "g"),
        n["gather_k"]: ("t_paged_gather_k", "s"),
        n["gather_v"]: ("t_paged_gather_v", "s"),
        n["score_mma"]: ("t_mma_0", "r"),
        n["exp"]: ("t_exp_0", "r"),
        n["rowsum"]: ("t_rowsum_0", "r"),
        n["value_mma"]: ("t_mma_1", "r"),
    }


def _build_tensor_maps(main, inner, spec, dense, built: "_Built", grid) -> None:
    """Fill both tiers' ``tmap``/``tinfo_map``.

    The paged tiers are built rather than patched: a dense schedule's maps describe
    slices of dense tensors, and every buffer here either changed scope (an index tile
    becomes a linear shared buffer), changed shape (a cache is handed down whole and
    gathered from), or did not exist (the page table, the lse).
    """
    mq, bn, head_dim = dense.query_rows, dense.kv_tile, dense.head_dim
    group = spec.group
    ph_q, ph_k, ph_v = built.ph_q, built.ph_k, built.ph_v
    ph_idx, ph_indptr = built.ph_idx, built.ph_indptr
    loop_node, inner_q, inner_idx = built.loop_node, built.inner_q, built.inner_idx
    gather_k, gather_v = built.gather_k, built.gather_v
    score_mma, exp = built.score_mma, built.exp
    rowsum, value_mma = built.rowsum, built.value_mma
    div, log2, outputs = built.div, built.log2, built.outputs

    q_shape = (spec.batch_dim, spec.num_qo_heads, head_dim)
    cache_shape = (spec.pages_dim, spec.page_size, spec.num_kv_heads, head_dim)

    def whole(shape):
        return Iterator(
            lambda *iters: tuple("0" for _ in shape),
            shape=tuple(shape),
            tile_size=tuple(shape),
        )

    def inner_tile(shape):
        """A loop-tier buffer, handed down whole: the tier takes no slice of its own."""
        return Iterator(
            lambda i: tuple("0" for _ in shape),
            shape=tuple(shape),
            tile_size=tuple(shape),
        )

    main.tmap[ph_q] = whole(q_shape)
    main.tmap[ph_k] = whole(cache_shape)
    main.tmap[ph_v] = whole(cache_shape)
    main.tmap[ph_idx] = whole((spec.index_dim,))
    main.tmap[ph_indptr] = whole((spec.indptr_dim,))
    main.tmap[loop_node] = [
        Iterator(lambda *i: ("0",), shape=(mq,), tile_size=(mq,)),
        Iterator(lambda *i: ("0", "0"), shape=(mq, head_dim), tile_size=(mq, head_dim)),
    ]

    def _out_offset(*iters):
        b, h = grid(*iters)[:2]
        return (f"{b}", f"{group} * {h}", "0")

    def _lse_offset(*iters):
        b, h = grid(*iters)[:2]
        return (f"{b}", f"{group} * {h}")

    if built.split is None:
        main.tmap[div] = Iterator(
            _out_offset,
            shape=(spec.batch_dim, spec.num_qo_heads, head_dim),
            tile_size=(1, group, head_dim),
        )
        main.tmap[outputs[0]] = main.tmap[div]
        if log2 is not None:
            main.tmap[log2] = Iterator(
                _lse_offset,
                shape=(spec.batch_dim, spec.num_qo_heads),
                tile_size=(1, group),
            )
            main.tmap[outputs[1]] = main.tmap[log2]

    inner.extent = 1
    inner.tmap[ph_q] = inner_tile(q_shape)
    inner.tmap[inner_q] = inner_tile((mq, head_dim))
    inner.tmap[inner_idx] = inner_tile((spec.index_dim,))
    inner.tmap[inner.binding[ph_k.name]] = inner_tile(cache_shape)
    inner.tmap[inner.binding[ph_v.name]] = inner_tile(cache_shape)
    inner.tmap[gather_k] = inner_tile((bn, head_dim))
    inner.tmap[gather_v] = inner_tile((bn, head_dim))
    inner.tmap[score_mma] = inner_tile((mq, bn))
    inner.tmap[exp] = inner.tmap[score_mma]
    inner.tmap[rowsum] = inner_tile((mq,))
    inner.tmap[value_mma] = inner_tile((mq, head_dim))

    main.tinfo_map = {
        ph_q: (ph_q.name, "g"),
        ph_k: (ph_k.name, "g"),
        ph_v: (ph_v.name, "g"),
        ph_idx: (ph_idx.name, "g"),
        ph_indptr: (ph_indptr.name, "g"),
        loop_node: ("t_GraphNode_0", "r"),
    }
    if built.split is None:
        main.tinfo_map[div] = ("OutputNode_0", "g")
        main.tinfo_map[outputs[0]] = ("OutputNode_0", "g")
        if log2 is not None:
            main.tinfo_map[log2] = ("OutputNode_1", "g")
            main.tinfo_map[outputs[1]] = ("OutputNode_1", "g")
    else:
        _partial_output_maps(
            main, outputs, built.store_d, built.store_o, built.split,
            spec.num_qo_heads, group, head_dim, spec.batch_dim, grid,
        )

    inner.tinfo_map = {
        ph_q: (ph_q.name, "g"),
        ph_k: (ph_k.name, "g"),
        ph_v: (ph_v.name, "g"),
        ph_idx: (ph_idx.name, "g"),
        inner_q: (inner_q.name, "s"),
        inner_idx: (inner_idx.name, "g"),
        inner.binding[ph_k.name]: (inner.binding[ph_k.name].name, "g"),
        inner.binding[ph_v.name]: (inner.binding[ph_v.name].name, "g"),
        gather_k: ("t_paged_gather_k", "s"),
        gather_v: ("t_paged_gather_v", "s"),
        score_mma: ("t_mma_0", "r"),
        exp: ("t_exp_0", "r"),
        rowsum: ("t_rowsum_0", "r"),
        value_mma: ("t_mma_1", "r"),
    }


# Names of the two partial tensors a split-KV pair hands from its partial kernel to its
# combine kernel. They are the combine's placeholder names, which the fused emitter turns
# into the kernel parameters `Partial_<name>_split` / `Partial_<name>_combine`.
PARTIAL_ROWSUM = "ROWSUM"
PARTIAL_VALUE = "VALUE"


def _build_decode_combine(
    spec,
    heads: int,
    group: int,
    head_dim: int,
    out_dtype: str,
    accum_dtype: str,
    comb: CombineTiling,
    grid_order,
    batch_dim: str,
    rasterization,
) -> Graph:
    """The combine half of a split-KV decode: sum the partials over the splits, normalise.

    One block owns ``group`` rows (a kv head's query heads; the whole head block for MLA)
    by ``comb.cols`` columns of one request, reads the ``split`` partials of that tile in
    ``comb.split_step``-deep slices through a loop tier, and applies the single-kernel
    layer's own epilogue -- the floored divide and the log2 -- to the sums. The loop over
    the splits is pipelined because its reads are affine slices of the partial tensors,
    which tilelang can double-buffer.
    """
    comb.validate(head_dim)
    split, step, cols = comb.split, comb.split_step, comb.cols
    n_col_blocks = head_dim // cols
    n_row_blocks = heads // group

    if step > 1:
        rowsum_c = Reduction(
            (step, group),
            axis=0,
            dtype=accum_dtype,
            out_dtype=accum_dtype,
            name="rowsum_combine_0",
        )
        value_c = Reduction(
            (step, group, cols),
            axis=0,
            dtype=accum_dtype,
            out_dtype=accum_dtype,
            name="value_combine_0",
        )
    else:
        # One partial per iteration: the split axis of the slice is a unit extent, which
        # codegen squeezes away, so there is no axis left to reduce -- the sum over the
        # splits is the loop itself, and each iteration adds its slice elementwise.
        rowsum_c = Accumulate(
            (group,), dtype=accum_dtype, out_dtype=accum_dtype, name="rowsum_combine_0"
        )
        value_c = Accumulate(
            (group, cols), dtype=accum_dtype, out_dtype=accum_dtype, name="value_combine_0"
        )
    ph_d = rowsum_c.inputs[0].src_node
    ph_o = value_c.inputs[0].src_node
    ph_d.name = f"PlaceHolder_{PARTIAL_ROWSUM}"
    ph_o.name = f"PlaceHolder_{PARTIAL_VALUE}"

    from .util import gather_all_nodes, lock_graph

    inner = Graph(nodes=gather_all_nodes(rowsum_c, value_c))
    inner.extent = split // step
    inner.loop_kind = comb.loop_kind
    inner.loop_stages = comb.loop_stages if comb.loop_kind == "pipelined" else None
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    d_id = inner.accum_nodes.index(rowsum_c)
    o_id = inner.accum_nodes.index(value_c)
    inner_d = inner.binding[ph_d.name]
    inner_o = inner.binding[ph_o.name]

    div = Div(
        (group, cols),
        broadcast=0,
        denom_floor=DENOM_FLOOR,
        dtype=accum_dtype,
        out_dtype=out_dtype,
        name="div_0",
        n_batch_dims=0,
    )
    value_edge = AccumEdge(loop_node, div, o_id, 0, accum_node=value_c)
    loop_node._out_edges.append(value_edge)
    div.set_inputs(0, value_edge)
    rowsum_edge = AccumEdge(loop_node, div, d_id, 1, accum_node=rowsum_c)
    loop_node._out_edges.append(rowsum_edge)
    div.set_inputs(1, rowsum_edge)
    outputs = [OutputNode(div, dtype=out_dtype)]
    outputs[0].name = "OutputNode_0"
    log2 = None
    if spec.emit_lse:
        log2 = Log2(
            (group,),
            dtype=accum_dtype,
            out_dtype=accum_dtype,
            name="log2_0",
            n_batch_dims=0,
        )
        lse_edge = AccumEdge(loop_node, log2, d_id, 0, accum_node=rowsum_c)
        loop_node._out_edges.append(lse_edge)
        log2.set_inputs(0, lse_edge)
        outputs.append(OutputNode(log2, dtype=accum_dtype))
        outputs[1].name = "OutputNode_1"

    main = Graph(nodes=[ph_d, ph_o] + gather_all_nodes(loop_node))
    main.dyn_dims = [batch_dim]
    main.n_threads = comb.n_threads
    main.pass_configs = dict(spec.pass_configs)
    main.rasterization = rasterization
    # The same grid as the partial's, without its split axis, plus a column axis when a
    # block owns less than the head dim. A single-kv-head operator (MLA) has no head axis.
    batch_first = tuple(grid_order) == ("batch", "head")
    if n_row_blocks == 1:
        lead = (f"{batch_dim}",)
    elif batch_first:
        lead = (f"{batch_dim}", f"{n_row_blocks}")
    else:
        lead = (f"{n_row_blocks}", f"{batch_dim}")
    main.extent = lead + ((f"{n_col_blocks}",) if n_col_blocks > 1 else ())

    def grid(*iters):
        """(request, row block, column offset) whatever the launch order."""
        if n_row_blocks == 1:
            b, h, rest = iters[0], "0", iters[1:]
        else:
            b, h = (iters[0], iters[1]) if batch_first else (iters[1], iters[0])
            rest = iters[2:]
        c = f"{cols} * {rest[0]}" if rest else "0"
        return b, h, c

    def whole(shape):
        return Iterator(
            lambda *iters: tuple("0" for _ in shape),
            shape=tuple(shape),
            tile_size=tuple(shape),
        )

    def _d_offset(*iters):
        b, h, _ = grid(*iters)
        return ("0", f"{b}", f"{group} * {h}")

    def _o_offset(*iters):
        b, h, c = grid(*iters)
        return ("0", f"{b}", f"{group} * {h}", c)

    def _out_offset(*iters):
        b, h, c = grid(*iters)
        return (f"{b}", f"{group} * {h}", c)

    def _lse_offset(*iters):
        b, h, _ = grid(*iters)
        return (f"{b}", f"{group} * {h}")

    main.tmap[ph_d] = Iterator(
        _d_offset, shape=(split, batch_dim, heads), tile_size=(split, 1, group)
    )
    main.tmap[ph_o] = Iterator(
        _o_offset,
        shape=(split, batch_dim, heads, head_dim),
        tile_size=(split, 1, group, cols),
    )
    # One entry per accumulator, in the loop tier's own accumulator order.
    main.tmap[loop_node] = [
        whole((group,)) if node is rowsum_c else whole((group, cols))
        for node in inner.accum_nodes
    ]
    main.tmap[div] = Iterator(
        _out_offset, shape=(batch_dim, heads, head_dim), tile_size=(1, group, cols)
    )
    main.tmap[outputs[0]] = main.tmap[div]
    if log2 is not None:
        main.tmap[log2] = Iterator(
            _lse_offset, shape=(batch_dim, heads), tile_size=(1, group)
        )
        main.tmap[outputs[1]] = main.tmap[log2]

    # A tile spanning every split is the whole slice, which codegen hands down unsliced
    # and therefore expects at a literal zero offset.
    s_off = (lambda i: "0") if step == split else (lambda i: f"{step} * {i}")
    inner.tmap[inner_d] = Iterator(
        lambda i: (s_off(i), "0"), shape=(split, group), tile_size=(step, group)
    )
    inner.tmap[inner_o] = Iterator(
        lambda i: (s_off(i), "0", "0"),
        shape=(split, group, cols),
        tile_size=(step, group, cols),
    )
    inner.tmap[rowsum_c] = whole((group,))
    inner.tmap[value_c] = whole((group, cols))

    main.tinfo_map = {
        ph_d: (ph_d.name, "g"),
        ph_o: (ph_o.name, "g"),
        loop_node: ("t_GraphNode_0", "r"),
        div: ("OutputNode_0", "g"),
        outputs[0]: ("OutputNode_0", "g"),
    }
    if log2 is not None:
        main.tinfo_map[log2] = ("OutputNode_1", "g")
        main.tinfo_map[outputs[1]] = ("OutputNode_1", "g")
    inner.tinfo_map = {
        ph_d: (ph_d.name, "g"),
        ph_o: (ph_o.name, "g"),
        inner_d: (inner_d.name, comb.input_scope),
        inner_o: (inner_o.name, comb.input_scope),
        rowsum_c: ("t_rowsum_combine_0", "r"),
        value_c: ("t_value_combine_0", "r"),
    }
    return _run_passes(main, inner)


def _mla_loop_tier(spec: MLASpec, dense: DenseMLA, exp):
    """The nodes inside the KV loop, shared by both MLA layers.

    Only the softmax mask differs between them, so ``exp`` is passed in: the decode layer
    masks the tail of the request's KV window, the prefill layer also masks the diagonal.
    Everything else -- the shared page-table window, the two gathers, the two score
    matmuls and their sum, the (M, N) denominator accumulator, and the value matmul reading
    the *same* gathered ckv tile -- is identical.
    """
    mq, bn = dense.query_rows, dense.kv_tile
    ckv, kpe = dense.head_dim_ckv, dense.head_dim_kpe
    window = bn * spec.index_window_tiles

    idx_window = IndexGather(
        (window,),
        index_extent=spec.index_dim,
        row_stride=1,
        page_size=spec.page_size,
        clamp_index=BUFFER_BOUND,
        scalar_args=("kv_base", "win_base", "valid_len"),
        name="idx_window",
    )
    # MLA has a single kv head, so the cache carries only (page, slot, dim) and the gather
    # picks no head; page_size 1 leaves no slot axis for the macro either.
    gather_kwargs = dict(
        source_rows=spec.pages_dim,
        index_extent=window,
        page_size=spec.page_size,
        paged_axes=True,
        clamp_index=BUFFER_BOUND,
        clamp_row=BUFFER_BOUND,
        scalar_args=("win_off",),
        dtype=dense.dtype,
        out_dtype=dense.dtype,
    )
    gather_ckv = PagedGather((bn, ckv), name="paged_gather_ckv", **gather_kwargs)
    gather_kpe = PagedGather((bn, kpe), name="paged_gather_kpe", **gather_kwargs)
    ckv_mma = Matmul(
        (mq, bn, ckv),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_0",
        transpose_B=True,
        warp_policy=spec.warp_policy,
    )
    pe_mma = Matmul(
        (mq, bn, kpe),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_pe",
        transpose_B=True,
        warp_policy=spec.warp_policy,
    )
    add = Add(
        (mq, bn),
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="add_0",
        n_batch_dims=0,
    )
    # The denominator is accumulated as a whole (M, N) tile and folded to a row vector once,
    # after the loop, rather than folded every iteration. A row fold is cross-lane work and
    # MLA runs the most iterations of any of these operators, so paying it once per block
    # instead of once per tile is worth the registers the tile costs.
    psum = Accumulate(
        (mq, bn), dtype=dense.accum_dtype, out_dtype=dense.accum_dtype, name="psum_0"
    )
    value_mma = Matmul(
        (mq, ckv, bn),
        dtype=dense.dtype,
        out_dtype=dense.accum_dtype,
        name="mma_1",
        warp_policy=spec.warp_policy,
    )
    value_mma.stage_fragment_operands_in_shared = (0,)

    _connect(gather_ckv, ckv_mma, 0, 1)
    _connect(gather_kpe, pe_mma, 0, 1)
    _connect(ckv_mma, add, 0, 0)
    _connect(pe_mma, add, 0, 1)
    _connect(add, exp, 0, 0)
    _connect(exp, psum, 0, 0)
    _connect(exp, value_mma, 0, 0)
    # The compressed cache is also the value tensor: MLA has no separate V, so the value
    # matmul reads the tile the score matmul already gathered rather than a second one.
    _connect(gather_ckv, value_mma, 0, 1)

    ph_qn = ckv_mma.inputs[0].src_node
    ph_ckv = gather_ckv.inputs[0].src_node
    # One window feeds both gathers, so the kpe gather is rewired to it rather than
    # declaring a second copy of the same page table.
    _connect(idx_window, gather_ckv, 0, 1)
    _connect(idx_window, gather_kpe, 0, 1)
    ph_idx = idx_window.inputs[0].src_node
    ph_qp = pe_mma.inputs[0].src_node
    ph_kpe = gather_kpe.inputs[0].src_node

    ph_qn.name = f"PlaceHolder_{spec.query_inputs[0]}"
    ph_ckv.name = f"PlaceHolder_{spec.cache_inputs[0]}"
    ph_idx.name = f"PlaceHolder_{spec.index_input}"
    ph_qp.name = f"PlaceHolder_{spec.query_inputs[1]}"
    ph_kpe.name = f"PlaceHolder_{spec.cache_inputs[1]}"

    return dict(
        idx_window=idx_window,
        gather_ckv=gather_ckv,
        gather_kpe=gather_kpe,
        ckv_mma=ckv_mma,
        pe_mma=pe_mma,
        add=add,
        exp=exp,
        psum=psum,
        value_mma=value_mma,
        ph_qn=ph_qn,
        ph_qp=ph_qp,
        ph_ckv=ph_ckv,
        ph_kpe=ph_kpe,
        ph_idx=ph_idx,
    )


def _mla_window_scalars(spec: MLASpec, dense: DenseMLA, split: bool = False) -> dict:
    """The window bookkeeping the loop tier's iteration scalars need.

    With ``split`` the loop index counts the block's own tiles and the request's tile is
    ``tile0 + i``, so every position is taken relative to that.
    """
    bn, iwt = dense.kv_tile, spec.index_window_tiles
    tile = (lambda i: f"(tile0 + {i})") if split else (lambda i: f"{i}")
    return {
        "tile_base": lambda i: f"{bn} * {tile(i)}",
        # The window this tile falls in, and the tile's offset inside it, both in tokens.
        "win_base": lambda i: f"{tile(i)} // {iwt} * {bn * iwt}",
        "win_off": lambda i: f"{tile(i)} % {iwt} * {bn}",
    }


def _build_mla_paged_decode(
    spec: MLASpec, dense: DenseMLA, split: int | None = None
) -> Graph:
    """A paged MLA decode: one CTA per request, every query head in the tile.

    With ``split``, the partial half of the split-KV form (see ``_build_paged_decode``):
    the grid is ``(request, split)``, each block walks its own range of the request's KV
    tiles, and the epilogue stores the folded denominator and the value accumulator raw.
    The page-table window is refilled on the block's first tile as well as at every
    window boundary, since a split may start in the middle of a window; codegen re-seeds
    the look-ahead from the fresh window on the same iteration, so nothing else changes.
    """
    mq, bn = dense.query_rows, dense.kv_tile
    ckv, kpe = dense.head_dim_ckv, dense.head_dim_kpe
    batch, pages = spec.batch_dim, spec.pages_dim
    indptr_dim, index_dim = spec.indptr_dim, spec.index_dim
    window = bn * spec.index_window_tiles
    if split is not None:
        assert isinstance(split, int) and split > 1, f"split={split!r} is not a split"

    exp = MaskedExp(
        (mq, bn),
        scale=None,
        scalar_args=("tile_base", "valid_len", "scale2"),
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="exp_0",
        n_batch_dims=0,
    )
    made = _mla_loop_tier(spec, dense, exp)
    idx_window, gather_ckv, gather_kpe = (
        made["idx_window"],
        made["gather_ckv"],
        made["gather_kpe"],
    )
    psum, value_mma = made["psum"], made["value_mma"]
    ph_qn, ph_qp = made["ph_qn"], made["ph_qp"]
    ph_ckv, ph_kpe, ph_idx = made["ph_ckv"], made["ph_kpe"], made["ph_idx"]

    from .util import gather_all_nodes, lock_graph

    inner = Graph(nodes=gather_all_nodes(psum, value_mma))
    inner.extent = 1
    inner.loop_kind = "serial"
    inner.window_nodes = [idx_window]
    # A split block's first tile need not sit at a window boundary, so it refills too.
    inner.window_guard = (
        "win_off == 0" if split is None else "(win_off == 0) | (tile_base == split_base)"
    )
    kv_end = "valid_len" if split is None else "split_end"
    if spec.prefetch:
        # Both cache tiles are read one iteration ahead, out of the same shared window. The
        # look-ahead stops at the window's edge, so the first tile of every window is not
        # covered by it; codegen re-seeds the fragments on the iteration that refills the
        # window, which is exactly where that tile's indices become available.
        assert dense.n_threads >= 256, (
            f"the MLA prefetch needs at least 256 threads to keep the {bn}x{ckv} cache "
            f"tile out of spill, but the schedule runs {dense.n_threads}; either re-tile "
            "or set prefetch=False"
        )
        inner.prefetch_nodes = [gather_ckv, gather_kpe]
        inner.prefetch_guard = (
            f"(win_off + {bn} < {window}) & (tile_base + {bn} < {kv_end})"
        )
    inner.sargs = [
        ("q_row0", "int32"),
        ("q_head0", "int32"),
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("scale2", "float32"),
    ]
    inner.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("scale2", "float32"),
        ("kv_base", "int32"),
        ("win_base", "int32"),
        ("win_off", "int32"),
    ]
    inner.smap = {
        "q_row0": lambda i: "q_row0",
        "q_head0": lambda i: "q_head0",
        "kv_base": lambda i: "kv_base",
        "valid_len": lambda i: "valid_len",
        "scale2": lambda i: "scale2",
        **_mla_window_scalars(spec, dense, split=split is not None),
    }
    if split is None:
        inner.loop_trip = f"T.ceildiv(valid_len, {bn})"
    else:
        inner.loop_trip = "my_trip"
        inner.sargs += [("tile0", "int32"), ("my_trip", "int32"), ("split_end", "int32")]
        inner.iter_sargs += [("split_base", "int32"), ("split_end", "int32")]
        inner.smap["tile0"] = lambda i: "tile0"
        inner.smap["my_trip"] = lambda i: "my_trip"
        inner.smap["split_end"] = lambda i: "split_end"
        inner.smap["split_base"] = lambda i: f"{bn} * tile0"
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    psum_id = inner.accum_nodes.index(psum)
    value_id = inner.accum_nodes.index(value_mma)
    inner_qn = inner.binding[ph_qn.name]
    inner_qp = inner.binding[ph_qp.name]
    inner_idx = inner.binding[ph_idx.name]

    # A decode request is one query token, so the tile's rows walk that token's head axis
    # from head 0 and the whole head count is valid.
    for tile_name, source, width, binding in (
        ("qn_tile", ph_qn, ckv, inner_qn),
        ("qp_tile", ph_qp, kpe, inner_qp),
    ):
        tile = RowGather(
            (mq, width),
            source_rows=batch,
            n_valid=mq,
            src_dims=3,
            source_heads=spec.num_qo_heads,
            scalar_args=("q_row0", "q_head0"),
            dtype=dense.dtype,
            out_dtype=dense.dtype,
            name=tile_name,
        )
        tile.source_args = (source.name,)
        binding.producer_op = tile

    # --- the epilogue: fold the accumulated denominator, then normalise -------------
    rowsum = Reduction(
        (mq, bn),
        axis=-1,
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="rowsum_0",
    )
    psum_edge = AccumEdge(loop_node, rowsum, psum_id, 0, accum_node=psum)
    loop_node._out_edges.append(psum_edge)
    rowsum.set_inputs(0, psum_edge)

    div = log2 = store_d = store_o = None
    if split is None:
        div = Div(
            (mq, ckv),
            broadcast=0,
            denom_floor=DENOM_FLOOR,
            dtype=dense.accum_dtype,
            out_dtype=dense.out_dtype,
            name="div_0",
            n_batch_dims=0,
        )
        value_edge = AccumEdge(loop_node, div, value_id, 0, accum_node=value_mma)
        loop_node._out_edges.append(value_edge)
        div.set_inputs(0, value_edge)
        _connect(rowsum, div, 0, 1)

        outputs = [OutputNode(div, dtype=dense.out_dtype)]
        outputs[0].name = "OutputNode_0"
        if spec.emit_lse:
            log2 = Log2(
                (mq,),
                dtype=dense.accum_dtype,
                out_dtype=dense.accum_dtype,
                name="log2_0",
                n_batch_dims=0,
            )
            _connect(rowsum, log2, 0, 0)
            outputs.append(OutputNode(log2, dtype=dense.accum_dtype))
            outputs[1].name = "OutputNode_1"
    else:
        # The folded denominator and the raw value accumulator, for the combine to sum.
        store_d = Copy(
            (mq,),
            dtype=dense.accum_dtype,
            out_dtype=dense.accum_dtype,
            name="partial_rowsum",
            n_batch_dims=0,
        )
        _connect(rowsum, store_d, 0, 0)
        store_o = Copy(
            (mq, ckv),
            dtype=dense.accum_dtype,
            out_dtype=dense.accum_dtype,
            name="partial_value",
            n_batch_dims=0,
        )
        o_edge = AccumEdge(loop_node, store_o, value_id, 0, accum_node=value_mma)
        loop_node._out_edges.append(o_edge)
        store_o.set_inputs(0, o_edge)
        outputs = [
            OutputNode(store_d, dtype=dense.accum_dtype),
            OutputNode(store_o, dtype=dense.accum_dtype),
        ]
        outputs[0].name = "OutputNode_0"
        outputs[1].name = "OutputNode_1"

    # Read for its values, not gathered from, so it stays out of the *_iter call.
    ph_indptr = PlaceHolderNode(name=spec.indptr_input)
    ph_indptr.set_shape((1,))
    ph_indptr.set_dtype(_int32())
    ph_indptr.index_only = True
    ph_indptr.name = f"PlaceHolder_{spec.indptr_input}"

    main = Graph(
        nodes=[ph_qn, ph_qp, ph_ckv, ph_kpe, ph_indptr, ph_idx]
        + gather_all_nodes(loop_node)
    )
    main.dyn_dims = [batch, pages, indptr_dim, index_dim]
    main.scalar_params = [(spec.scale_input, "float32")]
    main.pass_configs = dict(spec.pass_configs)
    main.n_threads = dense.n_threads
    main.rasterization = spec.rasterization
    # One CTA per request: MLA has a single kv head, so there is no head axis to split the
    # grid on and all query heads are the tile's rows. A split-KV partial adds the split
    # as the trailing grid axis.
    main.extent = (batch,) if split is None else (batch, f"{split}")
    main.sargs = list(inner.sargs)
    indptr = ph_indptr.name
    main.smap = {
        "q_row0": lambda i, *_: f"{i}",
        "q_head0": lambda i, *_: "0",
        "kv_base": lambda i, *_: f"{indptr}[{i}]",
        "valid_len": lambda i, *_: f"{indptr}[{i} + 1] - {indptr}[{i}]",
        "scale2": lambda i, *_: f"{spec.scale_input} * {LOG2E}",
    }
    if split is not None:
        main.smap.update(
            _split_range_scalars(
                split,
                bn,
                lambda b: f"{indptr}[{b} + 1] - {indptr}[{b}]",
                # (request, head block, split) for a grid of (request, split).
                lambda build: (lambda i, s: build(i, "0", s)),
            )
        )
    lock_graph(main)

    _mla_tensor_maps(
        main,
        inner,
        spec,
        dense,
        made=made,
        loop_node=loop_node,
        rowsum=rowsum,
        div=div,
        log2=log2,
        outputs=outputs,
        indptrs=[ph_indptr],
        q_shape_lead=batch,
        q_tiles=None,
        out_rows=1,
        out_offset=lambda i, *_: (f"{i}", "0", "0"),
        lse_offset=lambda i, *_: (f"{i}", "0"),
        split=split,
        store_d=store_d,
        store_o=store_o,
    )
    return _run_passes(main, inner, inner_idx)


def _build_mla_paged_prefill_causal(spec: MLASpec, dense: DenseMLA) -> Graph:
    """A causal paged MLA prefill over a ragged query axis.

    MLA has a single kv head, so a CTA owns *all* query heads of the tokens in its tile:
    the M axis fuses (token, head) and one staged KV tile feeds every head. The tuned
    schedule has ``query_rows == num_qo_heads``, i.e. one token per tile, and that is
    required rather than incidental -- the query tiles are affine regions over one token's
    contiguous block of heads, and a tile spanning several tokens is not one region.
    """
    mq, bn = dense.query_rows, dense.kv_tile
    ckv = dense.head_dim_ckv
    tokens, pages = spec.tokens_dim, spec.pages_dim
    indptr_dim, index_dim = spec.indptr_dim, spec.index_dim
    # `apply_paging` has already checked that mq == heads, for both MLA layers.
    heads = spec.num_qo_heads
    btok = mq // heads

    exp = CausalMaskedExp(
        (mq, bn),
        scale=None,
        scalar_args=("tile_base", "valid_len", "diag", "scale2"),
        row_group=heads,
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="exp_0",
        n_batch_dims=0,
    )
    made = _mla_loop_tier(spec, dense, exp)
    idx_window, psum, value_mma = made["idx_window"], made["psum"], made["value_mma"]
    ph_qn, ph_qp = made["ph_qn"], made["ph_qp"]
    ph_ckv, ph_kpe, ph_idx = made["ph_ckv"], made["ph_kpe"], made["ph_idx"]

    from .util import gather_all_nodes, lock_graph

    inner = Graph(nodes=gather_all_nodes(psum, value_mma))
    inner.extent = 1
    inner.loop_kind = "serial"
    inner.loop_trip = f"T.ceildiv(valid_len, {bn})"
    inner.window_nodes = [idx_window]
    inner.window_guard = "win_off == 0"
    # No prefetch, unlike the decode layer: a prefetch needs the wider block that keeps the
    # BN x ckv tile out of spill, and one query block of `mq` rows is too little work to
    # keep that many warps busy.
    assert not spec.prefetch, (
        "the MLA prefill layer's tiles are one query block, too little work for the wide "
        "block a prefetch needs; set prefetch=False"
    )
    inner.sargs = [
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("scale2", "float32"),
    ]
    inner.iter_sargs = [
        ("tile_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("scale2", "float32"),
        ("kv_base", "int32"),
        ("win_base", "int32"),
        ("win_off", "int32"),
    ]
    inner.smap = {
        "kv_base": lambda i: "kv_base",
        "valid_len": lambda i: "valid_len",
        "diag": lambda i: "diag",
        "scale2": lambda i: "scale2",
        **_mla_window_scalars(spec, dense),
    }
    lock_graph(inner)

    loop_node = GraphNode(name="GraphNode_0", graph=inner)
    psum_id = inner.accum_nodes.index(psum)
    value_id = inner.accum_nodes.index(value_mma)
    # The two query tensors are copied in as regions by the outer tier rather than tiled
    # here, so nothing below reads their loop-tier bindings -- but they still have to exist,
    # or the region would have nothing to copy into.
    for name in (ph_qn.name, ph_qp.name):
        assert name in inner.binding, (
            f"{name} did not resolve in the loop tier, so its region copy has no destination"
        )
    inner_idx = inner.binding[ph_idx.name]

    rowsum = Reduction(
        (mq, bn),
        axis=-1,
        dtype=dense.accum_dtype,
        out_dtype=dense.accum_dtype,
        name="rowsum_0",
    )
    psum_edge = AccumEdge(loop_node, rowsum, psum_id, 0, accum_node=psum)
    loop_node._out_edges.append(psum_edge)
    rowsum.set_inputs(0, psum_edge)

    # The epilogue is row-predicated: the last tile of a request holds fewer than `mq` rows
    # and the rows past it belong to the next request, so they must not be written.
    div = MaskedDiv(
        (mq, ckv),
        broadcast=0,
        scalar_args=("n_rows",),
        denom_floor=DENOM_FLOOR,
        row_group=heads,
        fuse_dst=False,
        dtype=dense.accum_dtype,
        out_dtype=dense.out_dtype,
        name="div_0",
        n_batch_dims=0,
    )
    value_edge = AccumEdge(loop_node, div, value_id, 0, accum_node=value_mma)
    loop_node._out_edges.append(value_edge)
    div.set_inputs(0, value_edge)
    _connect(rowsum, div, 0, 1)

    outputs = [OutputNode(div, dtype=dense.out_dtype)]
    outputs[0].name = "OutputNode_0"
    log2 = None
    if spec.emit_lse:
        log2 = MaskedLog2(
            (mq,),
            scalar_args=("n_rows",),
            row_group=heads,
            fuse_dst=False,
            dtype=dense.accum_dtype,
            out_dtype=dense.accum_dtype,
            name="log2_0",
            n_batch_dims=0,
        )
        _connect(rowsum, log2, 0, 0)
        outputs.append(OutputNode(log2, dtype=dense.accum_dtype))
        outputs[1].name = "OutputNode_1"

    # Read for their values, not gathered from, so they stay out of the *_iter call.
    ph_qo = PlaceHolderNode(name=spec.query_indptr_input)
    ph_qo.set_shape((1,))
    ph_qo.set_dtype(_int32())
    ph_qo.index_only = True
    ph_qo.name = f"PlaceHolder_{spec.query_indptr_input}"
    ph_kv = PlaceHolderNode(name=spec.indptr_input)
    ph_kv.set_shape((1,))
    ph_kv.set_dtype(_int32())
    ph_kv.index_only = True
    ph_kv.name = f"PlaceHolder_{spec.indptr_input}"

    main = Graph(
        nodes=[ph_qn, ph_qp, ph_ckv, ph_kpe, ph_qo, ph_kv, ph_idx]
        + gather_all_nodes(loop_node)
    )
    main.dyn_dims = [tokens, pages, indptr_dim, index_dim]
    main.scalar_params = [(spec.scale_input, "float32")]
    main.n_threads = dense.n_threads
    main.rasterization = spec.rasterization
    # One CTA per query tile; every head lives in the tile. sum(ceil(q_len/btok)) is at most
    # ceil(total_q/btok) + n_requests - 1 (ceil is subadditive up to one per term), so this
    # grid always covers the work list; the slack slots scan to request -1 and the block
    # guard skips them.
    main.extent = (f"T.ceildiv({tokens}, {btok}) + {indptr_dim} - 2",)
    main.prologue = [
        ScalarScan(
            op=IndptrTileScan(rows=btok),
            buffer="_wl",
            size=2,
            inputs=(ph_qo,),
            scalar_args=lambda i: (i, indptr_dim),
        )
    ]
    main.block_guard = lambda i: "_wl[0] >= 0"
    main.pass_configs = dict(spec.pass_configs)
    # A BufferRegion lets tilelang reach for TMA, whose mbarrier protocol assumes a
    # pipelined producer/consumer body; these nested serial macros provide no arrivals for
    # it and the kernel spins on the barrier. Nothing here needs TMA: the region exists to
    # replace a per-element gather with an affine copy, and this tile can never overrun
    # (its leading extent is btok and the other two are the buffer's own).
    main.pass_configs["tl.disable_tma_lower"] = True
    main.pass_configs["tl.disable_warp_specialized"] = True
    main.sargs = [
        ("n_rows", "int32"),
        ("kv_base", "int32"),
        ("valid_len", "int32"),
        ("diag", "int32"),
        ("scale2", "float32"),
    ]

    # Every scalar the body needs, as an expression over the scan result (_wl[0] is the
    # request, _wl[1] the tile within it) and the two indptr tensors.
    q_beg = f"{ph_qo.name}[_wl[0]]"
    q_end = f"{ph_qo.name}[_wl[0] + 1]"
    kv_beg = f"{ph_kv.name}[_wl[0]]"
    kv_end = f"{ph_kv.name}[_wl[0] + 1]"
    q_len = f"({q_end} - {q_beg})"
    kv_len = f"({kv_end} - {kv_beg})"
    tile_base = f"(_wl[1] * {btok})"
    n_rows = f"T.min({btok}, {q_len} - {tile_base})"
    # End-aligned causal bound for the tile's first row, then the KV window causality
    # leaves live for the whole tile.
    diag = f"({kv_len} - {q_len} + {tile_base})"
    main.smap = {
        "n_rows": lambda i: n_rows,
        "kv_base": lambda i: kv_beg,
        "valid_len": lambda i: f"T.max(T.min({kv_len}, {diag} + {n_rows}), 0)",
        "diag": lambda i: diag,
        "scale2": lambda i: f"{spec.scale_input} * {LOG2E}",
    }
    lock_graph(main)

    q_abs = f"{q_beg} + {tile_base}"
    _mla_tensor_maps(
        main,
        inner,
        spec,
        dense,
        made=made,
        loop_node=loop_node,
        rowsum=rowsum,
        div=div,
        log2=log2,
        outputs=outputs,
        indptrs=[ph_qo, ph_kv],
        q_shape_lead=tokens,
        # One token's whole block of heads, which is contiguous in (total_q, head, dim): the
        # tile the M axis wants is an affine slice, so it goes down as a region instead of a
        # gather. The caches cannot follow -- page_size is 1, so consecutive kv positions
        # are unrelated pages and the read is irreducibly a gather.
        q_tiles=lambda shape: Iterator(
            lambda i: (q_abs, "0", "0"),
            shape=shape,
            tile_size=(btok, heads, shape[-1]),
        ),
        out_rows=btok,
        out_offset=lambda i: (q_abs, "0", "0"),
        lse_offset=lambda i: (q_abs, "0"),
    )
    main.region_args = [ph_qn, ph_qp]
    return _run_passes(main, inner, inner_idx)


def _mla_tensor_maps(
    main,
    inner,
    spec: MLASpec,
    dense: DenseMLA,
    *,
    made,
    loop_node,
    rowsum,
    div,
    log2,
    outputs,
    indptrs,
    q_shape_lead,
    q_tiles,
    out_rows,
    out_offset,
    lse_offset,
    split=None,
    store_d=None,
    store_o=None,
) -> None:
    """Fill both tiers' maps for either MLA layer.

    ``q_tiles`` is what separates them: None means the queries are handed down whole and a
    ``RowGather`` stages the tile (decode), and an Iterator factory means the tile is an
    affine region of the ragged query axis (prefill).
    """
    mq, bn = dense.query_rows, dense.kv_tile
    ckv, kpe = dense.head_dim_ckv, dense.head_dim_kpe
    heads, window = spec.num_qo_heads, bn * spec.index_window_tiles
    ph_qn, ph_qp = made["ph_qn"], made["ph_qp"]
    ph_ckv, ph_kpe, ph_idx = made["ph_ckv"], made["ph_kpe"], made["ph_idx"]
    idx_window = made["idx_window"]
    gather_ckv, gather_kpe = made["gather_ckv"], made["gather_kpe"]
    ckv_mma, pe_mma, add = made["ckv_mma"], made["pe_mma"], made["add"]
    exp, psum, value_mma = made["exp"], made["psum"], made["value_mma"]
    inner_qn = inner.binding[ph_qn.name]
    inner_qp = inner.binding[ph_qp.name]
    inner_idx = inner.binding[ph_idx.name]

    qn_shape = (q_shape_lead, heads, ckv)
    qp_shape = (q_shape_lead, heads, kpe)
    ckv_shape = (spec.pages_dim, spec.page_size, ckv)
    kpe_shape = (spec.pages_dim, spec.page_size, kpe)

    def whole(shape):
        return Iterator(
            lambda *i: tuple("0" for _ in shape),
            shape=tuple(shape),
            tile_size=tuple(shape),
        )

    if q_tiles is None:
        main.tmap[ph_qn] = whole(qn_shape)
        main.tmap[ph_qp] = whole(qp_shape)
    else:
        main.tmap[ph_qn] = q_tiles(qn_shape)
        main.tmap[ph_qp] = q_tiles(qp_shape)
    main.tmap[ph_ckv] = whole(ckv_shape)
    main.tmap[ph_kpe] = whole(kpe_shape)
    main.tmap[ph_idx] = whole((spec.index_dim,))
    for indptr in indptrs:
        main.tmap[indptr] = whole((spec.indptr_dim,))
    main.tmap[loop_node] = [
        Iterator(lambda *i: ("0",), shape=(mq,), tile_size=(mq,)),
        Iterator(lambda *i: ("0", "0"), shape=(mq, ckv), tile_size=(mq, ckv)),
    ]

    main.tmap[rowsum] = Iterator(lambda *i: ("0",), shape=(mq,), tile_size=(mq,))
    if split is None:
        main.tmap[div] = Iterator(
            out_offset, shape=(q_shape_lead, heads, ckv), tile_size=(out_rows, heads, ckv)
        )
        main.tmap[outputs[0]] = main.tmap[div]
        if log2 is not None:
            main.tmap[log2] = Iterator(
                lse_offset, shape=(q_shape_lead, heads), tile_size=(out_rows, heads)
            )
            main.tmap[outputs[1]] = main.tmap[log2]

    inner.extent = 1
    # A gathered query tile needs the whole tensor in scope inside the tier; a region tile
    # arrives already sliced, so the tier sees only the slice.
    if q_tiles is None:
        inner.tmap[ph_qn] = whole(qn_shape)
        inner.tmap[ph_qp] = whole(qp_shape)
    inner.tmap[inner_qn] = whole((mq, ckv))
    inner.tmap[inner_qp] = whole((mq, kpe))
    inner.tmap[inner_idx] = whole((spec.index_dim,))
    inner.tmap[inner.binding[ph_ckv.name]] = whole(ckv_shape)
    inner.tmap[inner.binding[ph_kpe.name]] = whole(kpe_shape)
    inner.tmap[idx_window] = Iterator(
        lambda i: ("0",), shape=(window,), tile_size=(window,)
    )
    inner.tmap[gather_ckv] = Iterator(
        lambda i: ("0", "0"), shape=(bn, ckv), tile_size=(bn, ckv)
    )
    inner.tmap[gather_kpe] = Iterator(
        lambda i: ("0", "0"), shape=(bn, kpe), tile_size=(bn, kpe)
    )
    inner.tmap[ckv_mma] = Iterator(
        lambda i: ("0", "0"), shape=(mq, bn), tile_size=(mq, bn)
    )
    inner.tmap[pe_mma] = inner.tmap[ckv_mma]
    inner.tmap[add] = inner.tmap[ckv_mma]
    inner.tmap[exp] = inner.tmap[ckv_mma]
    inner.tmap[psum] = inner.tmap[exp]
    inner.tmap[value_mma] = Iterator(
        lambda i: ("0", "0"), shape=(mq, ckv), tile_size=(mq, ckv)
    )

    main.tinfo_map = {
        ph_qn: (ph_qn.name, "g"),
        ph_qp: (ph_qp.name, "g"),
        ph_ckv: (ph_ckv.name, "g"),
        ph_kpe: (ph_kpe.name, "g"),
        ph_idx: (ph_idx.name, "g"),
        **{indptr: (indptr.name, "g") for indptr in indptrs},
        loop_node: ("t_GraphNode_0", "r"),
        rowsum: ("t_rowsum_0", "r"),
    }
    if split is None:
        main.tinfo_map[div] = ("OutputNode_0", "g")
        main.tinfo_map[outputs[0]] = ("OutputNode_0", "g")
        if log2 is not None:
            main.tinfo_map[log2] = ("OutputNode_1", "g")
            main.tinfo_map[outputs[1]] = ("OutputNode_1", "g")
    else:
        # A decode grid of (request, split): the whole head block is the row tile.
        _partial_output_maps(
            main, outputs, store_d, store_o, split, heads, heads, ckv, q_shape_lead,
            lambda i, s: (i, "0", s),
        )

    inner.tinfo_map = {
        ph_qn: (ph_qn.name, "g"),
        ph_qp: (ph_qp.name, "g"),
        ph_ckv: (ph_ckv.name, "g"),
        ph_kpe: (ph_kpe.name, "g"),
        ph_idx: (ph_idx.name, "g"),
        inner_qn: (inner_qn.name, "s"),
        inner_qp: (inner_qp.name, "s"),
        inner_idx: (inner_idx.name, "g"),
        inner.binding[ph_ckv.name]: (inner.binding[ph_ckv.name].name, "g"),
        inner.binding[ph_kpe.name]: (inner.binding[ph_kpe.name].name, "g"),
        idx_window: ("t_idx_window", "s"),
        gather_ckv: ("t_paged_gather_ckv", "s"),
        gather_kpe: ("t_paged_gather_kpe", "s"),
        ckv_mma: ("t_mma_0", "r"),
        pe_mma: ("t_mma_pe", "r"),
        add: ("t_add_0", "r"),
        exp: ("t_exp_0", "r"),
        psum: ("t_psum_0", "r"),
        value_mma: ("t_mma_1", "r"),
    }


def _run_passes(main: Graph, inner: Graph = None, index_tile=None) -> Graph:
    """Annotate the finished graph, then correct the one annotation that is wrong for it.

    ``annotate_tinfo`` is deliberately absent: this pass writes ``tinfo_map`` itself,
    because every buffer in a paged tier either changed scope, changed shape, or did not
    exist in the dense graph.

    ``annotate_tc_layout`` marks every placeholder it can reach from a matmul as a
    tensor-core layout, and it reaches the int32 page table through the gathers. Today
    that mislabel changes nothing in the emitted kernel, because the tile is given global
    scope and codegen only consults the layout when it allocates a shared or fragment
    buffer -- verified by the golden tests, which are byte-identical either way. It is
    corrected regardless: the label is simply wrong, a page table is read one entry per
    row rather than in tensor-core fragments, and a tile that ever moves to shared (an
    MLA-style page-table window does) would silently get a swizzled layout.
    """
    _assert_loop_kind_matches_its_source(main)
    main = passes.annotate_args(main, is_main=True)
    main = passes.annotate_tc_layout(main)
    main = passes.annotate_fragment_layout_conflicts(main)
    if index_tile is not None:
        inner.tlayout_map[index_tile] = "linear"
    return main


def _assert_loop_kind_matches_its_source(main: Graph) -> None:
    """A reduction loop that pipelines a gathered tile is a bug, not a missed optimisation.

    ``graph.py`` says why: pipelining reorders a read the gather has not buffered. Six
    builders pick ``loop_kind`` and their tile source independently, nothing downstream
    checks that they agree, and one of them -- the ragged layer -- was switched from serial
    to pipelined for exactly this reason, so the pairing is live knowledge.

    Two deliberate limits. Only the nested tiers are checked: ``main`` is a grid rather than
    a loop, and ``Graph.loop_kind`` defaults to ``"pipelined"`` which no builder overrides
    there, so asserting anything about it would be satisfied by a default rather than by a
    decision. And only the tier's own nodes count: the layers that stage Q through a
    binding's producer gather have it hoisted above the loop by codegen, so it is not part
    of the loop body this is about.
    """
    gathers = (PagedGather, RowGather, RangeGather, IndexGather)
    tiers = [n.graph for n in main.nodes if isinstance(n, GraphNode)]
    while tiers:
        tier = tiers.pop()
        gathered = [n.name for n in tier.nodes if isinstance(n, gathers)]
        if tier.loop_kind == "pipelined":
            assert not gathered, (
                f"a pipelined loop stages {gathered} with a gather; pipelining reorders "
                "reads the gather has not buffered, so this loop must be serial (a gather "
                "layer overlaps with prefetch_nodes instead)"
            )
        else:
            assert not tier.region_args, (
                f"a {tier.loop_kind!r} loop hands down region_args "
                f"{list(tier.region_args)}; the BufferRegion exists so tilelang can reach "
                "for TMA, whose mbarriers need a pipelined producer/consumer body to "
                "supply arrivals"
            )
        tiers += [n.graph for n in tier.nodes if isinstance(n, GraphNode)]
