from typing import Sequence

from ..ir.graph import Graph, Node, OutputNode, PlaceHolderNode
from .tensor_codegen import gen_tensor_type


def partial_name(node: Node) -> str:
    if node.name.startswith("PlaceHolder_"):
        return "Partial_" + node.name[len("PlaceHolder_") :]
    return f"Partial_{node.name}"


class SplitEmitter:
    """Split-k: a partial graph and a combine graph emitted as one two-kernel prim_func."""

    def _get_partial_name(self, node: Node) -> str:
        return partial_name(node)

    def _gen_split_kernel(self, graphs: Sequence[Graph]):
        assert (
            len(graphs) == 2
        ), "Split-k combined codegen currently supports exactly two graphs."
        split_graph, combine_graph = graphs
        for g, prefix in ((split_graph, "split"), (combine_graph, "combine")):
            # Each half gets its own macro table, so equal names in the two do not collide.
            half = type(self)(
                apply_rasterization=self.apply_rasterization, debug=self.debug
            )
            half.gen_graph(
                g,
                name="main_kernel",
                is_main=True,
                name_prefix=prefix,
                emit_wrapper=False,
            )
            self.emit(half.prim_code + half.graph_code)
        self._emit_combined_main_kernel(split_graph, combine_graph)

    @staticmethod
    def _check_split_halves(split_graph, combine_graph, split_outputs, combine_inputs):
        assert (
            len(split_outputs) == len(combine_inputs) and len(split_outputs) > 0
        ), "Split-k combined codegen expects matching split outputs and combine inputs."
        for g in (split_graph, combine_graph):
            # One prim_func body: two prologues would have to agree on buffer names.
            assert (
                not g.prologue and g.block_guard is None
            ), "Split-k combined codegen does not support prologue/block_guard."
            # Split-k renames the partial parameter, which an else node's tinfo name misses.
            assert (
                not g.else_nodes and g.work_guard is None
            ), "Split-k combined codegen does not support work_guard/else_nodes."
        for split_output, combine_input in zip(split_outputs, combine_inputs):
            split_shape = split_graph.tmap[split_output].shape
            combine_shape = combine_graph.tmap[combine_input].shape
            assert (
                split_shape == combine_shape
            ), f"Partial tensor shape mismatch: {split_shape} vs {combine_shape}"
            assert split_output.get_dtype(0) == combine_input.get_dtype(
                0
            ), "Partial tensor dtype mismatch."

    @staticmethod
    def _merged_scalar_params(graphs):
        """Scalar inputs shared by both halves, declared once after the tensor inputs."""
        scalar_params = []
        for g in graphs:
            for sname, sdtype in g.scalar_params:
                known = dict(scalar_params).get(sname)
                assert known is None or known == sdtype, (
                    f"scalar param {sname!r} is declared as {known} by one half and "
                    f"{sdtype} by the other."
                )
                if known is None:
                    scalar_params.append((sname, sdtype))
        return scalar_params

    def _emit_combined_main_kernel(self, split_graph: Graph, combine_graph: Graph):
        split_outputs = [a for a in split_graph.args if isinstance(a, OutputNode)]
        combine_inputs = [
            a for a in combine_graph.args if isinstance(a, PlaceHolderNode)
        ]
        self._check_split_halves(
            split_graph, combine_graph, split_outputs, combine_inputs
        )
        partials = [
            (out, inp, f"{partial_name(inp)}_split", f"{partial_name(inp)}_combine")
            for out, inp in zip(split_outputs, combine_inputs)
        ]

        params = [
            f"{a.name}: {gen_tensor_type(split_graph.tmap[a].shape, a.get_dtype(0))}"
            for a in split_graph.args
            if not isinstance(a, OutputNode)
        ]
        params += [
            f"{n}: T.{t}"
            for n, t in self._merged_scalar_params((split_graph, combine_graph))
        ]
        for _, inp, split_name, combine_name in partials:
            ty = gen_tensor_type(combine_graph.tmap[inp].shape, inp.get_dtype(0))
            params += [f"{split_name}: {ty}", f"{combine_name}: {ty}"]
        params += [
            f"{a.name}: {gen_tensor_type(combine_graph.tmap[a].shape, a.get_dtype(0))}"
            for a in combine_graph.args
            if isinstance(a, OutputNode)
        ]
        self.emit("@T.prim_func\n")
        self.emit(f"def main_kernel({''.join(p + ', ' for p in params)}):\n")

        for graph, prefix, arg_name_map in (
            (split_graph, "split", {out: name for out, _, name, _ in partials}),
            (combine_graph, "combine", {inp: name for _, inp, _, name in partials}),
        ):
            dep_code, body_code = self._build_graph_call(
                graph,
                name="main_kernel",
                is_main=True,
                name_prefix=prefix,
                arg_name_map=arg_name_map,
            )
            self.emit(dep_code)
            self.emit(body_code)
