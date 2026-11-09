"""The operator-level description of each access layer, and its self-consistency checks."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from dataclasses import fields as spec_fields

from .errors import InvalidSpec
from .kinds import (
    _FIELDS_READ,
    _MLA_FIELDS_READ,
    _MLA_OPTIONAL_FIELDS,
    _OPTIONAL_FIELDS,
    CAUSAL_KINDS,
    KINDS,
    MLA_KINDS,
    MLA_PAGED_PREFILL_CAUSAL,
    PAGED_DECODE,
    PAGED_DECODE_SLAB,
    RAGGED_PREFILL_CAUSAL,
    SLAB_KINDS,
)


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
    if unread:
        raise InvalidSpec(
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
    # do not share enough KV to repay the per-CTA index remap) and on for ragged. None
    # defers to the codegen default (an L2 swizzle of 16).
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
            if not (isinstance(value, int) and not isinstance(value, bool)):
                raise InvalidSpec(
                    f"{name} must be an int, got {value!r}"
                )
        if not (self.num_qo_heads > 0 and self.num_kv_heads > 0):
            raise InvalidSpec(
                f"head counts must be positive, got {self.num_qo_heads}/{self.num_kv_heads}"
            )
        if self.num_qo_heads % self.num_kv_heads != 0:
            raise InvalidSpec(
                f"{self.num_qo_heads} query heads do not divide into {self.num_kv_heads} "
                "kv heads, so a block's query tile would not be a whole head group"
            )
        # A block's query tile is one kv head's group of query heads, which needs at
        # least two rows to tile and at least two kv heads for the head axis to survive
        # codegen's unit-extent squeeze. Both edges emit a kernel that does not compile.
        if not self.num_kv_heads > 1:
            raise InvalidSpec(
                "a single kv head leaves the cache with no head axis, which the gather's "
                "head scalar then indexes past the end of; use the MLA layer instead"
            )
        # A group of 1 squeezes the decode layers' (group, head_dim) output tile to a rank
        # the epilogue cannot write. The ragged layer gives one CTA a single query head and
        # writes (rows, 1, head_dim), so plain MHA is expressible there -- the group reaches
        # only its `kv_head = h // group` -- and this gate would reject it for nothing.
        if self.kind != RAGGED_PREFILL_CAUSAL:
            if not self.group > 1:
                raise InvalidSpec(
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
            if not self.page_size > 1:
                raise InvalidSpec(
                    f"page_size={self.page_size} makes the slab layer's address split "
                    f"degenerate; use kind={PAGED_DECODE!r} for one token per page"
                )
        else:
            if self.page_size != 1:
                raise InvalidSpec(
                    f"page_size={self.page_size} needs the slab access layer "
                    f"(kind={PAGED_DECODE_SLAB!r}) and its {self.last_page_len_input} input; "
                    f"only page_size=1 is expressible by the {self.kind!r} layer"
                )
        if len(self.cache_inputs) != 2:
            raise InvalidSpec(
                f"expected a (K, V) cache pair, got {self.cache_inputs}"
            )
        if self.kind not in KINDS:
            raise InvalidSpec(f"unknown access layer {self.kind!r}; expected {KINDS}")
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
        if len(set(names)) != len(names):
            raise InvalidSpec(f"input names collide: {names}")
        dims = [self.indptr_dim, self.scale_input] + [
            getattr(self, name)
            for name in ("batch_dim", "tokens_dim", "kv_tokens_dim", "pages_dim", "index_dim")
            if name in read
        ]
        if len(set(dims)) != len(dims):
            raise InvalidSpec(f"symbolic extent names collide: {dims}")
        if self.warp_policy not in ("rows", "cols", "even"):
            raise InvalidSpec(
                f"unsupported warp policy {self.warp_policy!r}"
            )
        if tuple(self.grid_order) not in (("batch", "head"), ("head", "batch")):
            raise InvalidSpec(
                f"grid_order must permute ('batch', 'head'), got {self.grid_order}"
            )
        # `causal` is not independent of the layer: masking against the diagonal needs the
        # per-row limit and the row predicate that only the prefill layer builds.
        if self.causal != (self.kind in CAUSAL_KINDS):
            raise InvalidSpec(
                f"causal={self.causal} does not match the {self.kind!r} layer; only the causal "
                "prefill layers mask against the diagonal"
            )


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
            if not (isinstance(value, int) and not isinstance(value, bool)):
                raise InvalidSpec(
                    f"{name} must be an int, got {value!r}"
                )
        # MLA puts one token's whole block of query heads in a single tile, and
        # `apply_paging` requires the tile to be exactly that block -- so the head count
        # *is* the matmul's M extent and inherits its granularity. Measured: 2 heads fails
        # with "stride == 4 || stride % 8 == 0", 4 and 8 with "M must be divisible by 16",
        # while 16, 32 and 64 compile on both layers.
        if not (self.num_qo_heads >= 16 and self.num_qo_heads % 16 == 0):
            raise InvalidSpec(
                f"{self.num_qo_heads} query heads is not a positive multiple of 16; MLA's query "
                "tile is the head block itself, so the head count is the matmul's M extent and "
                "a tensor-core fragment cannot be built from any other value"
            )
        if not self.index_window_tiles >= 1:
            raise InvalidSpec(
                f"index_window_tiles={self.index_window_tiles} leaves no window to stage"
            )
        # The prefetch reads the *next* tile's entries out of the current window, so it
        # needs a window wider than one tile. At exactly one the emitted guard is
        # `win_off + bn < bn` with win_off always 0 -- never true, so the prefetch is dead
        # code, and yet it still forces the 256-thread block below.
        if not (self.index_window_tiles > 1 or not self.prefetch):
            raise InvalidSpec(
                "index_window_tiles=1 leaves the prefetch no next tile to read inside the "
                "window, so it would never fire; set prefetch=False or widen the window"
            )
        # The compressed cache carries only (page, slot, dim) -- MLA has no head axis for
        # the gather to index -- and at page_size 1 there is no slot axis either, which is
        # the shape the gather macro is written against.
        if self.page_size != 1:
            raise InvalidSpec(
                f"page_size={self.page_size} would give the MLA cache a slot axis the gather "
                "does not index; only page_size=1 is expressible here"
            )
        if len(self.query_inputs) != 2:
            raise InvalidSpec(
                f"expected a (nope, pe) query pair, got {self.query_inputs}"
            )
        if len(self.cache_inputs) != 2:
            raise InvalidSpec(
                f"expected a (ckv, kpe) cache pair, got {self.cache_inputs}"
            )
        if self.warp_policy not in ("rows", "cols", "even"):
            raise InvalidSpec(
                f"unsupported warp policy {self.warp_policy!r}"
            )
        # Before the namespace checks, which are per-kind and would otherwise treat an
        # unknown kind as a decode.
        if self.kind not in MLA_KINDS:
            raise InvalidSpec(
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
        if len(set(names)) != len(names):
            raise InvalidSpec(f"input names collide: {names}")
        dims = [
            self.pages_dim,
            self.indptr_dim,
            self.index_dim,
            self.scale_input,
            self.tokens_dim if "tokens_dim" in read else self.batch_dim,
        ]
        if len(set(dims)) != len(dims):
            raise InvalidSpec(f"symbolic extent names collide: {dims}")
        if self.causal != (self.kind == MLA_PAGED_PREFILL_CAUSAL):
            raise InvalidSpec(
                f"causal={self.causal} does not match the {self.kind!r} layer; only the causal "
                "prefill layer masks against the diagonal"
            )
        # The prefetch needs the wide block that keeps the ckv tile out of spill; the
        # prefill layer's tiles are one query block of 16 rows, too little to fill 8 warps,
        # so it runs 128 threads and must not prefetch. That pairing is measured, and the
        # thread count is read off the dense graph, so it is checked in the builder.


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
        if not self.split > 1:
            raise InvalidSpec(f"a split factor of {self.split} is not a split")
        if not (1 <= self.split_step <= self.split and self.split % self.split_step == 0):
            raise InvalidSpec(
                f"split_step={self.split_step} does not divide split={self.split}"
            )
        if not (1 <= self.cols <= head_dim and head_dim % self.cols == 0):
            raise InvalidSpec(
                f"cols={self.cols} does not divide the head dim {head_dim}"
            )
        if self.loop_kind not in ("serial", "pipelined"):
            raise InvalidSpec(self.loop_kind)
        if self.input_scope not in ("s", "g"):
            raise InvalidSpec(self.input_scope)
