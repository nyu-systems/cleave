from ..ir.graph import ComputeOpNode, Node
from ..ir.ops.base import MacroOp


def qualify_name(name: str, prefix: str = "") -> str:
    return f"{prefix}_{name}" if prefix else name


def replace_func_name(code: str, src_name: str, dst_name: str) -> str:
    if src_name == dst_name:
        return code
    return code.replace(f"def {src_name}(", f"def {dst_name}(", 1)


class MacroEmitter:
    """Emits each op's macro once per (name, body); colliding bodies get `_vN` names."""

    def _reset_macros(self):
        self.prim_code = ""
        self.generated_macros = {}
        self._macro_calls = {}

    def emit_op_node(
        self, node: Node, mem_scope: str, clear: bool, name_prefix: str = ""
    ) -> str:
        assert isinstance(
            node, (ComputeOpNode, MacroOp)
        ), f"Only ComputeOpNode or MacroOp can generate prim, but got {node=}"
        # The op's name and body are a pure function of this key, so each is built once.
        key = (id(node), mem_scope, clear, name_prefix)
        hit = self._macro_calls.get(key)
        if hit is not None:
            return hit[1]
        macro_name = node.get_macro_name(mem_scope=mem_scope, clear=clear)
        macro = node.gen_macro(mem_scope=mem_scope, clear=clear)
        name = self._register_macro(
            macro, macro_name, qualify_name(macro_name, name_prefix)
        )
        self._macro_calls[key] = (node, name)
        return name

    def _register_macro(self, macro: str, macro_name: str, base_name: str) -> str:
        qualified_name, variant = base_name, 1
        while True:
            body = replace_func_name(macro, macro_name, qualified_name)
            emitted = self.generated_macros.get(qualified_name)
            if emitted is None:
                self.generated_macros[qualified_name] = body
                self.prim_code += body
                return qualified_name
            if emitted == body:
                return qualified_name
            variant += 1
            qualified_name = f"{base_name}_v{variant}"
