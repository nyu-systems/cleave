import math

from ..core.errors import CodegenError
from ..ir.graph import AccumEdge, ComputeOpNode, GraphNode, Iterator, OutputNode, PlaceHolderNode
from ..ir.ops.reduction import ReduceMax
from ..schedule.policy.hint import INTER, INTRA


class GraphValidator:
    """Structural checks and the module-level declarations (pass configs, symbols)."""

    def _validate_graph(self, graph, *, is_main: bool, seen=None):
        """Reject a graph that does not have the shape the emitter assumes.

        Everything below is an invariant the emitter already relies on; without these
        checks a schedule that breaks one produces malformed TileLang or an error from
        deep inside TVM, attributed to the wrong place. Each escape-hatch field added for
        a performance reason (`prologue`, `region_args`, `scalar_params`, `work_guard` /
        `else_nodes`) is checked here too, so it stays a controlled extension of the
        structure rather than a way around it.
        """
        seen = seen if seen is not None else set()
        if id(graph) in seen:
            return
        seen.add(id(graph))
        where = "main graph" if is_main else "subgraph"

        assert graph.nodes, f"{where}: no nodes."

        self._validate_accumulation(graph, is_main=is_main, where=where)

        # --- args: only declared IO, inputs first, each fully described -----------------
        seen_output = False
        for arg in graph.args:
            assert isinstance(
                arg, (PlaceHolderNode, OutputNode)
            ), f"{where}: arg {arg.name!r} is a {type(arg).__name__}, not declared IO."
            if isinstance(arg, OutputNode):
                seen_output = True
            elif graph.scalar_params:
                # Only load-bearing when scalars are emitted at the input/output boundary.
                # `annotate_args` puts a main graph's outputs last; `args` filled any other
                # way (a hand-built graph launched with an explicit `out_idx`) may still
                # interleave and is not rejected for it.
                assert not seen_output, (
                    f"{where}: input {arg.name!r} comes after an output in `args`. The "
                    "emitted signature keeps this order, and scalar params are inserted "
                    "at the boundary, so inputs must all precede outputs."
                )
        tmap = graph.tmap
        if isinstance(tmap, dict):
            for arg in graph.args:
                # A subgraph argument may instead be bound from the parent (`binding`),
                # in which case the parent's tmap is what slices it and the subgraph has
                # no entry of its own.
                assert (
                    arg in tmap or not is_main
                ), f"{where}: arg {arg.name!r} has no tmap entry."
                assert (
                    arg in graph.tinfo_map
                ), f"{where}: arg {arg.name!r} has no tinfo_map entry."
                if arg in tmap:
                    it = tmap[arg]
                    assert len(it.tile_size) == len(it.shape), (
                        f"{where}: arg {arg.name!r} tile_size {it.tile_size} and shape "
                        f"{it.shape} disagree in rank."
                    )

        # --- accumulators and regions must be part of this graph -----------------------
        for node in graph.accum_nodes:
            assert (
                node in graph.nodes
            ), f"{where}: accum node {node.name!r} is not in `nodes`."
        for node in graph.region_args:
            # Membership in `args` is deliberately NOT the invariant: a subgraph's region
            # args are its own inner placeholders, which are not in `args` at the point
            # this runs. What the emitter actually depends on is that a region arg is an IO
            # node whose tile is known, since the region's scalar subscripts come from it.
            assert isinstance(node, (PlaceHolderNode, OutputNode)), (
                f"{where}: region arg {node.name!r} is a {type(node).__name__}. Only a "
                "graph argument can be passed as a region."
            )
            assert not isinstance(tmap, dict) or node in tmap, (
                f"{where}: region arg {node.name!r} has no tmap entry, so its region "
                "subscripts cannot be built."
            )

        # --- scalars: every name an op or a guard reads must be derivable ---------------
        for name, _dtype in graph.sargs:
            assert name in graph.smap, f"{where}: sarg {name!r} has no smap expression."
        if graph.iter_sargs is not None:
            sarg_names = {n for n, _ in graph.sargs}
            for name, _dtype in graph.iter_sargs:
                # Either a scalar this graph derives, or one the loop supplies (a tile
                # base, say), which is resolved through `smap` at the call site.
                assert name in sarg_names or name in graph.smap, (
                    f"{where}: iter_sarg {name!r} is neither one of `sargs` nor "
                    "resolvable through `smap`."
                )

        # --- fields that only mean anything for the outermost graph ---------------------
        for field_name in ("prologue", "scalar_params", "else_nodes"):
            if not is_main:
                assert not getattr(graph, field_name, None), (
                    f"{where}: `{field_name}` is set, but it is only emitted for the main "
                    "graph -- a subgraph setting it would be silently ignored."
                )
        if not is_main:
            assert (
                getattr(graph, "rasterization", None) is None
            ), f"{where}: `rasterization` is only emitted for the main graph."
            for field_name in ("block_guard", "work_guard"):
                assert (
                    getattr(graph, field_name, None) is None
                ), f"{where}: `{field_name}` is only emitted for the main graph."

        # --- prefetched and windowed nodes stay part of the dataflow --------------------
        for field_name in ("prefetch_nodes", "window_nodes", "invariant_nodes"):
            nodes = getattr(graph, field_name, ())
            assert not (nodes and is_main), (
                f"{where}: `{field_name}` only means something inside a loop, and the main "
                "graph's body runs once per grid slot."
            )
            for node in nodes:
                assert node in graph.nodes, (
                    f"{field_name} entry {node.name!r} is not in `nodes`. It is still an "
                    "ordinary node of the dataflow; only where its buffer lives changes."
                )
                assert isinstance(node, ComputeOpNode), (
                    f"{field_name} entry {node.name!r} is a {type(node).__name__}; only a "
                    "compute node has a read to move."
                )
                assert node not in graph.accum_nodes, (
                    f"{field_name} entry {node.name!r} accumulates across iterations, so "
                    "its buffer is carried, not refilled."
                )
        assert not (graph.window_nodes and graph.window_guard is None), (
            f"{where}: `window_nodes` without a `window_guard` would refill every "
            "iteration, which is what hoisting the buffer is meant to avoid."
        )
        for node in graph.prefetch_nodes:
            for sname in getattr(node, "scalar_args", ()):
                assert sname in graph.smap, (
                    f"prefetched node {node.name!r} reads scalar {sname!r}, which has no "
                    "`smap` entry, so its value one iteration ahead cannot be derived."
                )

        self._validate_rolling_state(graph, where, is_main)

        # --- the else branch has to be an alternative producer of the same tile ---------
        if graph.work_guard is not None or graph.else_nodes:
            assert graph.work_guard is not None, (
                "else_nodes need a work_guard to be the alternative to; without one they "
                "would run unconditionally alongside the body they are meant to replace."
            )
            out_tiles = {}
            if isinstance(tmap, dict):
                for arg in graph.args:
                    if isinstance(arg, OutputNode):
                        out_tiles.setdefault(graph.tinfo_map[arg][0], []).append(
                            tuple(tmap[arg].tile_size)
                        )
            for node in graph.else_nodes:
                assert not any(n is node for n in graph.nodes), (
                    f"else node {node.name!r} is also in `nodes`. It replaces part of the "
                    "dataflow, so it must not also be part of it."
                )
                assert not node.inputs, (
                    f"else node {node.name!r} has tensor inputs. The else branch runs "
                    "precisely when the body that would produce them is skipped, so it "
                    "can only depend on scalars."
                )
                assert (
                    node in graph.tinfo_map
                ), f"else node {node.name!r} has no tinfo_map entry."
                assert node in tmap, f"else node {node.name!r} has no tmap entry."
                buf = graph.tinfo_map[node][0]
                assert buf in out_tiles, (
                    f"else node {node.name!r} writes {buf!r}, which is not one of this "
                    f"graph's outputs ({sorted(out_tiles)}). The else branch may only "
                    "produce declared outputs."
                )
                tile = tuple(tmap[node].tile_size)
                shapes = out_tiles[buf]
                assert tile in shapes, (
                    f"else node {node.name!r} writes a {tile} tile of {buf!r}, but the "
                    f"body writes {shapes}. The two paths must cover the same tile, or "
                    "the outputs disagree depending on which branch ran."
                )
                for name in getattr(node, "scalar_args", ()):
                    assert name in graph.smap, (
                        f"else node {node.name!r} reads scalar {name!r}, which has no "
                        "smap expression."
                    )
            # Coverage, which is the one way this mechanism can silently produce a wrong
            # answer: an output the body writes but the else branch does not is simply
            # never written for the slots that take the else branch, and no other slot
            # owns that region.
            filled = {graph.tinfo_map[n][0] for n in graph.else_nodes}
            # An empty else branch is included deliberately: `work_guard` with no fills
            # just skips the body, which is what `block_guard` already means, and leaves
            # every output unwritten for the slots it skips.
            assert filled == set(out_tiles), (
                "the else branch writes "
                f"{sorted(filled)} but the body writes {sorted(out_tiles)}. Every output "
                "needs a producer on both paths, or it is left uninitialised whenever the "
                "guard is false."
            )

        for node in graph.nodes:
            if isinstance(node, GraphNode):
                self._validate_graph(node.graph, is_main=False, seen=seen)

    def _validate_rolling_state(self, graph, where, is_main):
        """Stabilized exps, lag snapshots, rescaled accumulators and carried maxima."""
        tmap = graph.tmap if isinstance(graph.tmap, dict) else {}
        in_graph = {id(n) for n in graph.nodes}
        accum = {id(n) for n in graph.accum_nodes}
        lags = {id(n) for n in graph.lag_nodes}

        def squeezed(node):
            it = tmap.get(node)
            return None if not isinstance(it, Iterator) else [t for t in it.tile_size if t != 1]

        for node in graph.nodes:
            assert getattr(node, "stabilize", None) is None, (
                f"{where}: {node.name!r} is marked for stabilization but was not rewritten; "
                "emitting it as a plain exp would silently drop the max."
            )
            # The searched twin of the rule above: a full-extent value whose update rule
            # (``node.streaming``) is still on it was not assembled. That is right where
            # its dim is whole in the tile it is computed from, and wrong in a loop that
            # carries it or steps its dim: without the lag copy and the repairs the sums
            # next to it would be relative to the max of one slice.
            if is_main or getattr(node, "streaming", None) is None:
                continue
            sliced = False
            operand = node.inputs[0].src_node
            it = None if isinstance(operand, GraphNode) else tmap.get(operand)
            if isinstance(it, Iterator) and hasattr(node, "position_in"):
                axis = node.position_in(it)
                sliced = axis is not None and it.tile_size[axis] != it.shape[axis]
            assert id(node) not in accum and not sliced, (
                f"{where}: {node.name!r} has an update rule that was not assembled; "
                "emitting it as a plain max would take the max of one slice."
            )

        # --- lag snapshots and rescaled accumulators: state carried across iterations ---
        assert not ((graph.lag_nodes or graph.accum_scale) and is_main), (
            f"{where}: `lag_nodes` / `accum_scale` describe state carried across loop "
            "iterations, and the main graph's body runs once per grid slot."
        )
        for node in graph.lag_nodes:
            assert id(node) in in_graph, f"lag node {node.name!r} is not in `nodes`."
            assert isinstance(node, ComputeOpNode) and len(node.inputs) == 1, (
                f"lag node {node.name!r} must be a compute node with exactly one input."
            )
            assert id(node) not in accum, (
                f"lag node {node.name!r} is a per-iteration snapshot, not carried state."
            )
            assert node not in graph.prefetch_nodes and node not in graph.window_nodes, (
                f"lag node {node.name!r} cannot also be prefetched or windowed."
            )
            src = node.inputs[0].src_node
            assert id(src) in accum, (
                f"lag node {node.name!r} snapshots {src.name!r}, which is not an "
                "accumulator of this graph: only carried state has a previous value."
            )
        for acc, factor in graph.accum_scale.items():
            assert id(acc) in accum, (
                f"rescaled node {acc.name!r} is not an accumulator of this graph."
            )
            assert id(factor) in in_graph and isinstance(factor, ComputeOpNode), (
                f"rescale factor {factor.name!r} of {acc.name!r} is not a compute node "
                "of this graph."
            )
            assert id(factor) not in accum and id(factor) not in lags, (
                f"rescale factor {factor.name!r} of {acc.name!r} must be a per-iteration "
                "value, not an accumulator or a lag node."
            )
            # Through ordinary edges only: a lag node reads the previous value.
            stack, seen = [factor], set()
            while stack:
                node = stack.pop()
                assert node is not acc, (
                    f"rescale factor {factor.name!r} is computed from {acc.name!r}, the "
                    "accumulator it has to be emitted in front of."
                )
                if id(node) in seen or id(node) in lags:
                    continue
                seen.add(id(node))
                stack.extend(e.src_node for e in node.inputs if id(e.src_node) in in_graph)
            assert acc.out_dtype == factor.out_dtype == "float32", (
                f"rescale of {acc.name!r} by {factor.name!r}: both must be float32, got "
                f"{acc.out_dtype} and {factor.out_dtype}."
            )
            assert getattr(acc, "accum_init", None) is None, (
                f"rescaled accumulator {acc.name!r} starts at {acc.accum_init}; the first "
                "iteration multiplies it by a factor of 0, which only leaves a zero as it is."
            )
            acc_tile, fac_tile = squeezed(acc), squeezed(factor)
            if acc_tile is not None and fac_tile is not None:
                assert fac_tile and acc_tile[: len(fac_tile)] == fac_tile, (
                    f"rescale factor {factor.name!r} tile {fac_tile} is not a leading "
                    f"slice of accumulator {acc.name!r} tile {acc_tile}."
                )
                # The accumulator's buffer is allocated unsqueezed, the macro indexes the
                # squeezed rank.
                assert len(acc_tile) == len(tmap[acc].tile_size), (
                    f"rescaled accumulator {acc.name!r} has a unit extent in its tile "
                    f"{tuple(tmap[acc].tile_size)}."
                )

        # --- carried values -----------------------------------------------------------
        for node in graph.accum_nodes:
            init = getattr(node, "accum_init", None)
            if init is not None and math.isfinite(init) and init != 0:
                assert node.out_dtype == "float32", (
                    f"accumulator {node.name!r} starts at {init}, a float32 value, but is "
                    f"stored as {node.out_dtype}."
                )
            readers = [
                e.dst_node.name
                for e in node.outputs
                if id(e.dst_node) in in_graph and id(e.dst_node) not in lags
            ]
            assert not readers or getattr(node, "running_max", False), (
                f"accumulator {node.name!r} is read by {readers} inside the loop that "
                "accumulates it. Only a running max is meaningful before its loop ends; "
                "a partial sum is not."
            )
        if not is_main:
            for node in graph.nodes:
                if not isinstance(node, ReduceMax) or id(node) in accum:
                    continue
                operand = node.inputs[0].src_node
                it = None if isinstance(operand, GraphNode) else tmap.get(operand)
                if not isinstance(it, Iterator):
                    continue
                # Found from the axis the node was built with, not from ``axis_arg``:
                # that is the ``dim`` of the emitted macro, an index into the stored
                # operand, which has its unit extents squeezed out and the tile here
                # has not.
                axis = node.position_in(it)
                if axis is None:
                    given, whole = 1, node.dims[node.flat_axis]
                else:
                    given, whole = it.tile_size[axis], it.shape[axis]
                assert given == whole, (
                    f"{where}: {node.name!r} takes a max over a slice: the loop gives it "
                    f"{given} of {whole} elements of the reduced axis "
                    "per iteration, and it is not an accumulator, so every iteration "
                    "starts over."
                )

    def _declare_pass_configs(self, graph):
        """Emit the tilelang pass configs this kernel must be compiled under.

        They travel in the generated module rather than in the caller, because a kernel
        that only works with (say) TMA lowering off is not correct otherwise, and every
        caller -- tests, the benchmark harness -- has to agree.
        """
        graphs = graph if isinstance(graph, (list, tuple)) else [graph]
        merged = {}
        for g in graphs:
            for key, value in getattr(g, "pass_configs", {}).items():
                assert merged.get(key, value) == value, (
                    f"Split graphs disagree on pass config {key!r}: "
                    f"{merged[key]} vs {value}"
                )
                merged[key] = value

        # Restore two tilelang 0.1.5 defaults that 0.1.13 changed (each cost ~24%):
        # fast math (0.1.5's exact numerics) and a higher ptxas register budget, without
        # which the masked-softmax exp2 stalls global loads (measured +30% long_scoreboard).
        merged.setdefault("tl.enable_fast_math", True)
        merged.setdefault("tl.ptxas_register_usage_level", 10)

        if not merged:
            return
        entries = "".join(f'    "{k}": {v!r},\n' for k, v in sorted(merged.items()))
        self.pass_code = f"PASS_CONFIGS = {{\n{entries}}}\n"

    def _validate_accumulation(self, graph, *, is_main: bool, where: str):
        """Refuse an accumulation context the emitter cannot honour.

        Raises, not asserts: emitting an inter output as an intra one would silently
        miscompile (it would accumulate the wrong tiles), and ``-O`` drops asserts.
        Supported: the main graph's inter OutputNodes are each fed by an accum node of
        one of its loops that the loop marks inter; static extents; no split kernel.
        """
        if not graph.accumulation:
            return
        unknown = sorted({c for c in graph.accumulation.values()} - {INTRA, INTER})
        if unknown:
            raise CodegenError(f"{where}: unknown accumulation contexts {unknown}.")
        inter = [n for n, c in graph.accumulation.items() if c == INTER]
        if not inter:
            return

        def refuse(why):
            raise CodegenError(
                f"{where}: inter accumulation (summed across CTAs) of "
                f"{[n.name for n in inter]} is not supported: {why}."
            )

        if getattr(graph, "dyn_dims", ()):
            refuse("symbolic extents (the fp32 buffers are allocated per static shape)")
        for node in inter:
            tmap = graph.tmap[node]
            if not all(isinstance(e, int) for e in (*tmap.shape, *tmap.tile_size)):
                refuse(f"{node.name} has a symbolic tile {tuple(tmap.tile_size)}")
            if is_main:
                edge = node.inputs[0] if isinstance(node, OutputNode) else None
                loop = edge.src_node if isinstance(edge, AccumEdge) else None
                if loop not in graph.nodes or not isinstance(loop, GraphNode):
                    refuse(f"{node.name} is not fed by an accumulator of a loop")
                if loop.graph.accumulation.get(edge.accum_node) != INTER:
                    refuse(
                        f"{node.name} reads {edge.accum_node.name}, which its loop does "
                        "not sum across CTAs"
                    )
            else:
                if node not in graph.accum_nodes:
                    refuse(f"{node.name} is not an accumulator of its loop")
                if any(e.dst_node in graph.nodes for e in node.outputs):
                    refuse(f"{node.name}'s per-iteration partial feeds another node")

    @staticmethod
    def _refuse_split_inter(graphs):
        from .kernel import inter_nodes

        if any(inter_nodes(g) for g in graphs):
            raise CodegenError(
                "inter accumulation (summed across CTAs) is not supported in split kernels."
            )

    def _declare_inter_outputs(self, graph):
        """Emit ``INTER_OUTPUTS = {param: dtype}`` for the outputs summed across CTAs.

        The kernel takes each as an fp32 buffer that must be zero when it starts and is
        cast to ``dtype`` afterwards; the runtime (``runtime.compile._compile``) reads this
        to do both around every launch. Nothing is emitted for any other kernel.
        """
        from .kernel import inter_nodes, inter_output_layout

        outputs = inter_nodes(graph)
        if not outputs:
            return
        entries = "".join(
            f'    "{n.name}": "{n.get_dtype(0)}",\n'
            for n in graph.args
            if n in outputs
        )
        self.pass_code += f"INTER_OUTPUTS = {{\n{entries}}}\n"
        # buffers stored in another layout than row-major (``kernel.inter_blocked_width``)
        layouts = "".join(
            f'    "{n.name}": "{layout}",\n'
            for n in graph.args
            if n in outputs and (layout := inter_output_layout(n)) is not None
        )
        if layouts:
            self.pass_code += f"INTER_LAYOUTS = {{\n{layouts}}}\n"

    def _declare_dyn_dims(self, graph):
        """Emit ``name = T.symbolic("name")`` for every symbolic extent in the graph.

        A symbolic extent must appear bare in at least one tensor shape so the runtime
        can infer it from the bound tensor; once inferred it may be used inside extent
        and grid expressions, and is in scope inside the emitted macros.
        """
        roots = list(graph) if isinstance(graph, (list, tuple)) else [graph]
        names, pending = [], list(roots)
        while pending:
            g = pending.pop(0)
            for name in getattr(g, "dyn_dims", ()):
                if name not in names:
                    names.append(name)
            pending.extend(n.graph for n in g.nodes if isinstance(n, GraphNode))
        for name in names:
            assert any(
                name in tuple(g.tmap[arg].shape)
                for g in roots
                for arg in g.args
                if isinstance(arg, PlaceHolderNode) and arg in g.tmap
            ), (
                f"symbolic extent {name!r} never appears bare in an input tensor shape, "
                "so the runtime has nothing to infer it from"
            )
            self.symbol_code += f'{name} = T.symbolic("{name}")\n'
