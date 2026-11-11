"""Stage 1: Program -> symbolic tile graph -> superoptimizer search -> GraphSet."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..core.errors import SuperoptError
from . import cache
from .config import SuperoptConfig

_IO_OPS = ("kn_input_op", "kn_output_op")


def program_dtype(program) -> str:
    return str(getattr(program, "dtype", None) or "float16")


def _op_name(op_type: str) -> str:
    name = op_type.removeprefix("kn_").removesuffix("_op")
    for op in ("sum", "max"):
        if name.startswith(f"reduce_{op}"):
            return op
    return name


@dataclass(eq=False)
class SearchedGraph:
    index: int
    structure: list[dict]
    symbol_map: dict | None
    is_split: bool
    program: Any = field(repr=False, default=None)
    split: tuple[str, ...] = ()
    invalid_split: bool = False  # has split ops but no partial/combine cut
    split_rejection: str | None = None  # why, when invalid_split

    @property
    def dtype(self) -> str:
        return program_dtype(self.program)

    @property
    def kind(self) -> str:
        if self.invalid_split:
            return "invalid-split"
        return f"split({','.join(self.split)})" if self.is_split else "solo"

    @property
    def ops(self) -> list[str]:
        return [_op_name(o["op_type"]) for o in self.structure if o["op_type"] not in _IO_OPS]

    def summary(self) -> str:
        return f"{self.kind} " + ">".join(self.ops)

    def __repr__(self):
        return f"SearchedGraph({self.index}, {self.summary()})"


class GraphSet(Sequence):
    def __init__(self, graphs, program=None, config: SuperoptConfig | None = None, *,
                 max_ops: int | None = None, checkpoint: str | None = None,
                 from_cache: bool = False, seconds: float = 0.0):
        self.graphs = list(graphs)
        self.program, self.config, self.max_ops = program, config, max_ops
        self.checkpoint, self.from_cache, self.seconds = checkpoint, from_cache, seconds

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return self.graphs[i]

    def filter(self, fn: Callable[[SearchedGraph], bool]) -> "GraphSet":
        return GraphSet([g for g in self.graphs if fn(g)], self.program, self.config,
                        max_ops=self.max_ops, checkpoint=self.checkpoint,
                        from_cache=self.from_cache, seconds=self.seconds)

    @property
    def dtype(self) -> str:
        return program_dtype(self.program)

    @classmethod
    def identity(cls, program) -> "GraphSet":
        """The un-optimized program as its only graph (schedule-only baseline)."""
        structure = program.to_graph_structure()
        return cls([SearchedGraph(0, structure, None, False, program)], program, None)

    def __str__(self):
        head = (f"GraphSet: {len(self)} graph(s) max_ops={self.max_ops} "
                f"{'cache hit' if self.from_cache else 'searched'} {self.seconds:.2f}s")
        return "\n".join([head] + [f"  {g.index}: {g.summary()}" for g in self])


def superoptimize(program, config: SuperoptConfig | None = None) -> GraphSet:
    import cleave.superoptimizer as csopt
    from ..lower.split import Rejected, _contains_split_ops, cut_split_graph

    config = config or SuperoptConfig()
    max_ops = config.resolve_max_ops(program)
    t0 = time.perf_counter()
    tile_graph = program.to_tile_graph(split=config.split)  # SpecError on a bad split dim
    try:
        symbolic, symbol_map = csopt.create_symbolic_tile_graph(tile_graph)
        search_cfg = symbolic.setup_search_config()
        search_cfg.max_num_kernel_graph_op = max_ops
        if config.threads:
            search_cfg.search_thread = int(config.threads)
        ckpt, from_cache = "", False
        if config.cache:
            key = cache.superopt_key(program, max_ops, config.split, config.concat_matmul)
            ckpt = str(cache.checkpoint_path(key))
            from_cache = _checkpoint_valid(symbolic, ckpt, config.concat_matmul, search_cfg)
        found = symbolic.search_tile_graph(ckpt, config.concat_matmul, False, search_cfg)
        structures = [g.get_graph_structure() for g in found]
    except Exception as e:  # noqa: BLE001
        raise SuperoptError(f"superoptimizer search failed: {type(e).__name__}: {e}") from e
    graphs = []
    for i, s in enumerate(structures):
        cut = cut_split_graph(s) if _contains_split_ops(s) else None
        rejected = cut.reason if isinstance(cut, Rejected) else None
        graphs.append(SearchedGraph(i, s, dict(symbol_map), cut is not None and rejected is None,
                                    program, config.split, invalid_split=rejected is not None,
                                    split_rejection=rejected))
    return GraphSet(graphs, program, config, max_ops=max_ops, checkpoint=ckpt or None,
                    from_cache=from_cache, seconds=time.perf_counter() - t0)


def _checkpoint_valid(symbolic, ckpt: str, concat: bool, search_cfg) -> bool:
    try:
        with open(ckpt + ".key") as f:
            stored = f.read().strip()
    except OSError:
        return False
    return os.path.exists(ckpt) and stored == symbolic.checkpoint_key(concat, search_cfg)
