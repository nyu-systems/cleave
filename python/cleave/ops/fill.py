import math

from ..graph import Node
from .base import MacroOp, const_token, mgroup_row, mgroup_slot


class ConstFill(MacroOp, Node):
    """Write a constant over a tile, row-predicated, with no tensor input.

    This is an *alternative producer* of a tile that the dataflow already produces: it
    goes in ``Graph.else_nodes``, guarded by ``Graph.work_guard``, for the case where the
    computation the body would run is known in advance to yield that constant. An
    attention tile whose causal window is empty is the motivating case -- its softmax
    numerator and denominator are both zero, so the epilogue's whole job is to arrive at
    the operator's defined 0 (output) and -inf (log-sum-exp). Storing those directly skips
    clearing the accumulators, gathering Q and running the divide, which on a workload
    where nearly every tile is blind is most of the kernel's time.

    It is a ``Node`` because it takes part in the graph's structure -- it has
    ``tinfo_map`` and ``tmap`` entries and is validated against the tile the body writes.
    It is a ``MacroOp`` because its body has no prim_func to derive from: there is no
    input to read.
    """

    def __init__(
        self,
        dims: tuple[int, ...],
        value: float,
        out_dtype: str = "float32",
        row_group: int = 1,
        fuse_dst: bool = True,
        scalar_args: tuple[str, ...] = ("n_rows",),
        name: str = "const_fill",
    ):
        assert len(dims) >= 1, dims
        assert row_group >= 1, row_group
        assert dims[0] % row_group == 0, (dims, row_group)
        assert not math.isnan(value), (
            "NaN has no TileLang literal (`nan` is not a name in the generated module), "
            "and no operator defines it as an output value."
        )
        assert len(scalar_args) == 1, (
            "the body reads exactly one scalar, the live row count; got "
            f"{scalar_args!r}"
        )
        Node.__init__(self, name=name)
        self.dims = tuple(int(d) for d in dims)
        self.value = value
        self.out_dtype = out_dtype
        self.row_group = int(row_group)
        self.fuse_dst = bool(fuse_dst)
        self.scalar_args = tuple(scalar_args)

    @property
    def min_size(self) -> int:
        """Elements the body writes, so the codegen can check the tile is big enough."""
        prod = 1
        for d in self.dims:
            prod *= d
        return prod

    def dst_rank(self) -> int:
        """Rank of the destination this body indexes.

        A fused row axis splits into two subscripts, unless the destination's leading
        extent was squeezed away because it is 1 -- which is what ``fuse_dst=False``
        means, matching the same flag on the epilogue ops this stands in for.
        """
        return len(self.dims) + (1 if self.row_group > 1 and self.fuse_dst else 0)

    def _dst_index(self) -> str:
        """Destination subscript, matching what the op this stands in for writes.

        With a row group the leading axis is a fused (token, head-slot) pair, so it
        splits back into two subscripts exactly as ``MaskedDiv``/``MaskedLog2`` split it;
        the two paths have to address the same elements or the output depends on which
        branch ran.
        """
        if self.row_group > 1:
            head = (
                [mgroup_row("i0", self.row_group), mgroup_slot("i0", self.row_group)]
                if self.fuse_dst
                else [mgroup_slot("i0", self.row_group)]
            )
        else:
            head = ["i0"]
        return ", ".join(head + [f"i{d}" for d in range(1, len(self.dims))])

    def gen_macro(self, **kwargs) -> str:
        loop_vars = ", ".join(f"i{d}" for d in range(len(self.dims)))
        extents = ", ".join(str(d) for d in self.dims)
        row = mgroup_row("i0", self.row_group) if self.row_group > 1 else "i0"
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}({self.scalar_args[0]}, C):
            for {loop_vars} in T.Parallel({extents}):
                if {row} < {self.scalar_args[0]}:
                    C[{self._dst_index()}] = {self._value_str()}"""
        return self._format_code(macro_str)

    def _value_str(self) -> str:
        # Always float32, never the destination's dtype: TVM refuses to produce a
        # bfloat16 infinity ("Cannot decide infinity for type bfloat16"), while a float32
        # one stored into a bf16 buffer is cast on the way in. That also keeps `out_dtype`
        # out of the body, and so out of the macro name.
        if math.isinf(self.value):
            sign = "-" if self.value < 0 else ""
            return f'{sign}T.infinity("float32")'
        return repr(float(self.value))

    def _value_tag(self) -> str:
        """Identifier-safe spelling of the value, for the macro name.

        Derived, not passed in: a caller-supplied label could disagree with the value,
        and two fills differing only in a value would then collide on one macro name.
        (Named `_value_tag`, not `_tag`: the carver ``Node`` base already owns `_tag`.)
        """
        if math.isinf(self.value):
            return "neginf" if self.value < 0 else "posinf"
        return const_token(float(self.value))

    def get_macro_name(self, **kwargs) -> str:
        # Every attribute the body reads, and nothing else: the value, the row group and
        # whether it is fused (together they fix the subscripts), the extents (they are
        # literals in the loop) and the scalar's name (it is the parameter name and the
        # predicate). `out_dtype` is deliberately absent -- see `_value_str`.
        shape = "x".join(str(d) for d in self.dims)
        grp = ""
        if self.row_group > 1:
            grp = f"_mgrp{self.row_group}" + ("" if self.fuse_dst else "_nofuse")
        return (
            f"ukernel_const_fill_{self._value_tag()}{grp}_{shape}_"
            f"{self.scalar_args[0]}_{kwargs['mem_scope']}"
        )
