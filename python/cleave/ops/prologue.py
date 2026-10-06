from .base import MacroOp


class PrologueOp(MacroOp):
    """A device-side scan emitted at kernel scope, ahead of the iterative body.

    A prologue op never joins ``graph.nodes`` and has no ``tmap``/``tinfo_map`` entry:
    it exists to derive the per-block scalars the body is then called with.
    """


class IndptrTileScan(PrologueOp):
    """Map a flat work-tile slot to (request, tile-within-request) over a CSR indptr.

    ``DST[0]`` is the owning request, or -1 when the slot is past the end of the work
    list; ``DST[1]`` is the tile ordinal inside that request. Together they replace a
    host-precomputed work list, so a graph can take the operator's declared
    ``qo_indptr`` and derive per-block scalars itself.

    The body is branch-free on purpose. The natural two-``if`` form, whose second test
    reads what the first just wrote, is not visible within one loop iteration and
    silently yields wrong slots; here the only predicate reads ``_cum``, written
    unconditionally, and at most one iteration can satisfy it.
    """

    min_size = 2

    def __init__(self, rows: int, name: str = "indptr_scan"):
        assert rows >= 1, f"rows per tile must be positive, got {rows}"
        self.rows = int(rows)
        self.name = name

    def gen_macro(self, **kwargs) -> str:
        macro_str = f"""\
        @T.macro
        def {self.get_macro_name(**kwargs)}(IDX, slot, n_entries, DST):
            _cum = T.alloc_local((1,), "int32")
            DST[0] = -1
            DST[1] = 0
            _cum[0] = 0
            for s0 in T.serial(n_entries - 1):
                _n = T.ceildiv(IDX[s0 + 1] - IDX[s0], {self.rows})
                _hit = T.if_then_else((_cum[0] <= slot) & (slot < _cum[0] + _n), 1, 0)
                DST[0] += _hit * (s0 + 1)
                DST[1] += _hit * (slot - _cum[0])
                _cum[0] += _n"""
        return self._format_code(macro_str)

    def get_macro_name(self, **kwargs) -> str:
        # `rows` is the only attribute the body reads; the destination size does not
        # change a line of it, so it is deliberately absent from the name.
        return f"ukernel_indptr_scan_tile{self.rows}_{kwargs['mem_scope']}"
