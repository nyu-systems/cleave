from ..ir.graph import ComputeOpNode, GraphNode, OutputNode, PlaceHolderNode


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

        # --- args: only declared IO, inputs first, each fully described -----------------
        seen_output = False
        for arg in graph.args:
            assert isinstance(
                arg, (PlaceHolderNode, OutputNode)
            ), f"{where}: arg {arg.name!r} is a {type(arg).__name__}, not declared IO."
            if isinstance(arg, OutputNode):
                seen_output = True
            elif graph.scalar_params:
                # Only load-bearing when scalars are emitted at the input/output boundary;
                # `annotate_args` appends in topological order, so a legitimate
                # multi-output graph can interleave and must not be rejected for it.
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
        for field_name in ("prefetch_nodes", "window_nodes"):
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
